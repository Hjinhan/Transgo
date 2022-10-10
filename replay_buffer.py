import numpy as np
import torch
import ray
import copy


class ReplayMemory:
    def __new__(cls, config):
        
        return ReplayMemory_Random.remote(config)

#---------------------------------------------------------------------------------------------------------------------------


#1\-------------------------------------------------------------------------------------------------------------------
@ray.remote
class ReplayMemory_Random():    # random

    def __init__(self,config):
      self.capacity = config.buffer_size
      self.full = False   # 用于跟踪实际容量
      self.index = 0
      self.last_save_index = 0
      self.load_index = 0

      self.blank_trans = (np.zeros((config.encode_state_channels,config.board_size,config.board_size)), np.zeros((config.board_size**2+1)), 
                                                             0.0, np.zeros((config.board_size**2)))  
      self.data = np.array([self.blank_trans] * self.capacity,dtype = object)  # 构建结构化数组
 
    def append(self, observation, act_prob, win_z, own_z):    # 往 replaybuffer 添加数据
      data = (observation, act_prob, win_z, own_z)
      self.data[self.index] = data  # 在底层数据结构中存储数据
      self.index = (self.index + 1) % self.capacity  #  index从0开始 ， 则输出 循环 【0，size-1】
      self.full = self.full or self.index == 0  # capacity 满时 ，full为 true

    def sample(self, batch_size):   # 从replaybuffer 取样本进行训练
        if self.full == True :
            buffer_len = self.capacity
        else:
            buffer_len = self.index

        if buffer_len < batch_size :
             data_idxs = np.random.choice(buffer_len , batch_size)   # 可重复取同一个数据
        else:
             data_idxs = np.random.choice(buffer_len , batch_size, replace = False)    # 不能重复取
        transitions = self.data[data_idxs]
        return  transitions
  
    def save(self):    # 保存到硬盘上的数据

       if self.capacity <= 1500000:   # 小于 150万直接全部保存
            save_buffer ={"buffer_capacity": self.capacity,
                            "index": self.index,
                            "full": self.full,
                            "data": self.data,
                            "save_len": self.capacity}
       else:
            if 0 < self.index - self.last_save_index < 300000 :   # 不保存
                  return False
            if self.index - self.last_save_index  < 0 and  self.index < 300000 :  # 不保存
                  return False

            save_len = 1000000
            if self.index  < save_len : # 若是数据不到100万，则按照现有数据的数量进行保存
                save_len = self.index 
            save_buffer ={"buffer_capacity": self.capacity,
                        "data": self.data[self.index - save_len : self.index ],  # 不包括 index
                        "save_len": save_len,
                        "index": self.index  }  
            self.last_save_index = self.index

       return save_buffer

    def load(self, replay_buffer_infos):  # 从硬盘读取的数据
         
        save_len = replay_buffer_infos["save_len"]
        if self.load_index + save_len < self.capacity :
            self.data[self.load_index : self.load_index + save_len ]  = replay_buffer_infos["data"]
            self.load_index+= save_len
        else :
            end_len = self.capacity - self.load_index
            self.data[self.load_index : ]  = replay_buffer_infos["data"][: end_len]
            self.load_index= 0
            self.full = True
        self.index = self.load_index
        
        return self.full
 
    def info(self):
         t = {"capacity":self.capacity, 
              "index":self.index,
              "full": self.full,
             }
         return t



#------------------------------------------------------------------------------------
# 暂时不用


@ray.remote
class ReplayMemory_PER:
    def __init__(self,config):

      self.capacity = config.buffer_size
      self.priority_exponent = config.priority_exponent
      self.transitions = SegmentTree(config)   # 和线段树

    def append(self, state, mcts_prob,  win_z, own_z):
      self.transitions.append((state, mcts_prob,  win_z, own_z), self.transitions.max)  # 如果是新的数据，以最大优先级 去存储

    def get_samples_from_segments(self, batch_size, p_total):
      global idxs, probs, tree_idxs
      segment_length = p_total / batch_size                     #　是 p_total / batch_size ，不是capacity，因此，即使容量没满依然可以训练　
      segment_starts = np.arange(batch_size) * segment_length
      valid = False
      while not valid:              #　若不满足，容易导致在这里出现死循环
        samples = np.random.uniform(0.0, segment_length, [batch_size]) + segment_starts   #  要抽出的样本 （在总优先级划分的区域）
        probs, idxs, tree_idxs = self.transitions.find(samples)  # 检索样本并取出
        if  np.all(probs != 0) and np.all(idxs<= self.capacity):
          valid = True  # 取出的 idxs 必须是优先级不为0（即不能取出没有数据的节点）
      transitions =self.transitions.get(idxs)
      # print('idxs:',idxs,'probs:',probs,'p_total:',p_total)
      return probs, idxs, tree_idxs, transitions

    def sample(self, batch_size):

      p_total = self.transitions.total()
      probs, idxs, tree_idxs, transitions = self.get_samples_from_segments(batch_size, p_total)
      return tree_idxs, transitions

    def update_priorities(self, idxs, priorities):
      priorities = np.power(priorities, self.priority_exponent)
      self.transitions.update(idxs, priorities)

    def save(self):
       save_buffer ={
                  "buffer_capacity": self.capacity,
                  "priority_exponent": self.priority_exponent,
                  "transitions_index": self.transitions.index,
                  "transitions_size": self.transitions.size,
                  "transitions_full": self.transitions.full,
                  "transitions_tree_start": self.transitions.tree_start,
                  "transitions_sum_tree": self.transitions.sum_tree,
                  "transitions_data": self.transitions.data,
                  "transitions_max": self.transitions.max,
                  }
       return save_buffer

    def load(self,replay_buffer_infos):
        self.capacity = replay_buffer_infos["buffer_capacity"]
        self.transitions.index = replay_buffer_infos["transitions_index"]
        self.transitions.size = replay_buffer_infos["transitions_size"]
        self.transitions.full = replay_buffer_infos["transitions_full"]
        self.transitions.tree_start = replay_buffer_infos["transitions_tree_start"]
        self.transitions.sum_tree = replay_buffer_infos["transitions_sum_tree"].copy()
        self.transitions.data = replay_buffer_infos["transitions_data"]
        self.transitions.max = replay_buffer_infos["transitions_max"]

    def info(self):
         t = {"capacity":self.capacity, 
              "transitions_index":self.transitions.index,
              "full": self.transitions.full,
              "transitions_max":self.transitions.max,
         }
         return t

   
class SegmentTree():
    def __init__(self, config):
      self.index = 0
      self.size = config.buffer_size
      self.full = False   # 用于跟踪实际容量
      self.tree_start = 2**(self.size-1).bit_length()-1  # 所有非叶子节点个数  .bit_length()  计算二进制的位数
      self.sum_tree = np.zeros((self.tree_start + self.size,), dtype=np.float32)

      self.blank_trans = (np.zeros((config.encode_state_channels,config.board_size,config.board_size)), np.zeros((config.board_size**2+1)),
                                 0.0, np.zeros((config.board_size**2)))   
      self.data = np.array([self.blank_trans] * self.size,dtype = object)  # 构建结构化数组
      self.max = 1     #初始化segment tree时新存入节点的默认权重值，初始化为 1    ，也用于判断叶子节点的最大值

    # 重新更新
    def _update_nodes(self, indices):
      children_indices = indices * 2 + np.expand_dims([1, 2], axis=1)           # [[1],[2]]
      self.sum_tree[indices] = np.sum(self.sum_tree[children_indices], axis=0)

    def _propagate(self, indices):
      parents = (indices - 1) // 2
      unique_parents = np.unique(parents)
      self._update_nodes(unique_parents)
      if parents[0] != 0:
        self._propagate(parents)

    def update(self, indices, values):
      self.sum_tree[indices] = values
      self._propagate(indices)
      current_max_value = np.max(values)
      self.max = max(current_max_value, self.max)

    # 添加
    def _propagate_index(self, index):
      parent = (index - 1) // 2
      left, right = 2 * parent + 1, 2 * parent + 2
      self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
      if parent != 0:
        self._propagate_index(parent)

    def _update_index(self, index, value):
      self.sum_tree[index] = value
      self._propagate_index(index)
      self.max = max(value, self.max)

    def append(self, data, value):
      self.data[self.index] = data  # 在底层数据结构中存储数据
      self._update_index(self.index + self.tree_start, value)  # 更新树
      self.index = (self.index + 1) % self.size  #  index从0开始 ， 则输出 循环 【0，size-1】
      self.full = self.full or self.index == 0  # capacity 满时 ，full为 true
      self.max = max(value, self.max)
  
     # 检索
    def _retrieve(self, indices, values):
      children_indices = (indices * 2 + np.expand_dims([1, 2], axis=1))  
                                                                                                     
      if children_indices[0, 0] >= self.sum_tree.shape[0]: #如果索引对应于叶节点，则返回它们
        return indices
      elif children_indices[0, 0] >= self.tree_start:  # 此时，在叶子节点
        children_indices = np.minimum(children_indices, self.sum_tree.shape[0] - 1)  # 防止 children_indices 溢出，取最小值 ，树索引 self.sum_tree.shape[0] - 1
      left_children_values = self.sum_tree[children_indices[0]]

      successor_choices = np.greater(values, left_children_values).astype(np.int32)  # 检查values 是否大于 left_children_values，若大于，则为1true），则下一查找的节点为右子节点
      successor_indices = children_indices[successor_choices, np.arange(indices.size)] # np.arange(indices.size) 应该可以替换为 ：
      successor_values = values - successor_choices * left_children_values # 如果下次迭代从右子节点开始查找对应的权重值，则需要将待查找的权重value减去左子节点的权重。 successor_choices 为0/1
      return self._retrieve(successor_indices, successor_values)

    def find(self, values):
      indices = self._retrieve(np.zeros(values.shape, dtype=np.int32), values)
      data_index = indices - self.tree_start
      return (self.sum_tree[indices], data_index, indices)

    def get(self, data_index):
      return self.data[data_index % self.size]

    def total(self):
      return self.sum_tree[0]



  


