import numpy as np
import ray

  
class ReplayMemory:   
    def __new__(cls, config):

        return ReplayMemory_Random.remote(config)


#1\-------------------------------------------------------------------------------------------------------------------
@ray.remote
class ReplayMemory_Random():    # random

    def __init__(self,config):
      self.capacity = config.buffer_size
      self.full = False   # 用于跟踪实际容量
      self.index = 0

      self.blank_trans = (np.zeros((config.encode_state_channels,config.board_size,config.board_size)), 
                                        np.zeros((config.board_size**2+1)), 0.0,np.zeros((config.board_size**2)))  
      self.data = np.array([self.blank_trans] * self.capacity,dtype = object)  # 构建结构化数组
 
    def append(self, observation, act_prob, win_z, own):    # 往 replaybuffer 添加数据
      data = (observation, act_prob, win_z, own)
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
       save_buffer ={"buffer_capacity": self.capacity,
                     "index": self.index,
                     "full": self.full,
                     "data": self.data}
       return save_buffer

    def load(self,replay_buffer_infos):  # 从硬盘读取的数据
        self.capacity = replay_buffer_infos["buffer_capacity"]
        self.index = replay_buffer_infos["index"]
        self.full = replay_buffer_infos["full"]
        self.data = replay_buffer_infos["data"]
 
    def info(self):
         t = {"capacity":self.capacity, 
              "index":self.index,
              "full": self.full,
             }
         return t

