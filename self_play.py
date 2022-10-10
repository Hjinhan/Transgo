
import numpy as np
import ray
import time
import torch
import math

from model import TransGoNetwork, MutativeNetwork
from GoEnv.environment import GoEnv
      
  
class Node_M:
    def __init__(self, prior):
        
        self.prior = prior
        self.state = None
        self.total_visit_count = 0
        self.value_sum = 0
        self.children = {}
        self.ons = 0                 # 只有在使用 WU_UCT时才使用到
        self.real_expanded = False   # 判断是否完成了真正的扩展（子节点的prior不为0）
           
    def expanded(self):     # 判断子节点是否扩展过
        return self.real_expanded    # 创建子节点并不算真正扩展（子节点内容全空），只有子节点的prior不为0才算真正扩展

    def value(self):       # 期望价值
        return self.value_sum /( self.total_visit_count +1)    # 在子节点扩展时，会将父节点的价值赋予给它，所以要多加1

    def expand(self, action_priors, value = 0.0):   # 扩展新节点 ， 一次扩展所有合法动作 （伪扩展）
 
        for action, p in action_priors.items():
            self.children[action] = Node_M(p)

            # 子节点的value以父节点的价值作为初始值
            # 从子节点的角度看，父节点价值需要转换成负的
            self.children[action].value_sum = -1 * value         
                                                               
    def visit_count(self, action):   # 取出每个动作的访问次数（不一定是子节点）
        if action in self.children:
            return self.children[action].total_visit_count
        return 0
    
    def dirichlet_prior(self):  # 添加狄利克雷噪声   0.03 ， 0.25 是可调参数 ， 一般固定 0.25 调0.03这个参数
 
        actions = list(self.children.keys())
        noise = np.random.dirichlet([0.03] * len(actions))
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - 0.25) + n * 0.25


class Node_V:
    def __init__(self, prior):
        
        self.prior = prior
        self.state = None
        self.total_visit_count = 0
        self.ons = 0
        self.value_sum = 0
        self.value_mean = 0.    #!
        self.value_var = 0.     #!
        self.children = {}
        self.real_expanded = False   # 判断是否完成了真正的扩展（子节点的prior不为0）
         
    def expanded(self):     # 判断子节点是否扩展过
        return self.real_expanded    # 创建子节点并不算真正扩展（子节点内容全空），只有子节点的prior不为0才算真正扩展

    def value(self):       # 期望价值
        return self.value_sum /( self.total_visit_count +1 )    # 在子节点扩展时，会将父节点的价值赋予给它，所以要多加1

    def expand(self, action_priors, value = 0.0):   # 扩展新节点 ， 一次扩展所有合法动作 （伪扩展）
 
        for action, p in action_priors.items():
            self.children[action] = Node_V(p)

            # 子节点的value以父节点的价值作为初始值
            # 从子节点的角度看，父节点价值需要转换成负的
            self.children[action].value_sum = -1 * value         
                                                               
    def visit_count(self, action):   # 取出每个动作的访问次数（不一定是子节点）
        if action in self.children:
            return self.children[action].total_visit_count
        return 0

    def value_mean_var(self,value):    #!
        t = self.value_mean
        # self.value_mean = ((self.total_visit_count -1)*self.value_mean+ value)/self.total_visit_count
        self.value_mean = self.value()
        self.value_var = self.value_var + (value-t)*(value-self.value_mean)
    
    def dirichlet_prior(self):  # 添加狄利克雷噪声   0.03 ， 0.25 是可调参数 ， 一般固定 0.25 调0.03这个参数
 
        actions = list(self.children.keys())
        noise = np.random.dirichlet([0.03] * len(actions))
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - 0.25) + n * 0.25


class WU_UCT(object):

        def __init__(self,config, env, model):

            self.config = config
            self.board_size = config.board_size 
            self.parallel_readouts =  config.parallel_readouts
            self.model = model
            self.num_simulations =  config.num_simulation
            self.c = config.c_puct
            self.wu_loss = config.wu_loss
            self.env = env

            # Must run this once at the start to expand the root node. 
            self.reset_root()

        def reset_root(self):
            self.root = Node_M(0)
            self.root.state , _ = self.env.reset()
             
            policy, value = self.computeValuePolicy([self.root.state])
            policy, value = policy[0], value[0][0]
 
            legal_actions = self.env.getLegalAction(self.root.state)    
            action_priors = { idx: p for idx, p in enumerate(policy) if idx in legal_actions} 
            self.root.expand(action_priors, value)
            self.root.real_expanded = True
  
        def run(self):
                """
                执行 parallel_readouts 次并行搜索 , 
                使用master-slave结构,只并行网络推理部分(因为调用网络占用时间超过整个搜索的80%) 
                """
                paths = []        # parallel_readouts次搜索路径列表
                leaf_states = [] # 叶子节点状态
                failsafe = 0    # 当 game_over , search_path 并不会记录进 paths ， 但是failsafe 会继续计数

                while len(paths) < self.parallel_readouts and failsafe < self.parallel_readouts * 2:
                                            #  循环次数不一定小于 self.parallel_readouts 

                    node = self.root        # 相当于引用 ， node的改变会修改self.root
                    current_tree_depth = 0 
                    search_path = [node]
                    failsafe += 1
                    while node.expanded():   #  选择
                        
                        current_tree_depth += 1
                        next_action, node = self.select_child(node)
                        search_path.append(node)
                     
                    leaf_state ,done = self.env.step(search_path[-2].state, next_action)    # 下一个状态
                    node.state = leaf_state
                       
                    if not done:
                       # 伪扩展 ， 策略跟价值由于是并行输出，会在后面再传递给子节点
                       legal_actions = self.env.getLegalAction(leaf_state)     
                       action_priors_init = { idx: 0.0 for idx  in legal_actions} 
                       node.expand(action_priors_init)     # 扩展， TODO：此时该节点的 real_expanded依然为 false
                   
                    else:        # 如果已经到达终局，不再往下搜索，并且直接使用实际的输赢作为评估 
                        value =  1 if self.env.getPlayer(node.state) ==  self.env.getWinner(node.state) else -1
                        self.backpropagate(search_path, value)
                        continue

                    self.incomplete_update(search_path)
                    paths.append(search_path)
                    leaf_states.append(leaf_state)
                
                if paths:
                    probs, values = self.computeValuePolicy(leaf_states)   # leaf_states 是一个 batch的数量
                    for path, leaf_state, prob, value in zip(paths, leaf_states, probs, values):

                        self.complete_update(prob, value[0], path, leaf_state)


        def get_action_probs(self, is_selfplay = True):  # 返回  下一步动作 ， π， 根节点观测状态
                
                if is_selfplay == True:    # 添加狄利克雷噪声   用的时候把注释去掉
                    self.root.dirichlet_prior()

                current_readouts = self.root.total_visit_count          # current_readouts是继承的 访问次数
                while self.root.total_visit_count < current_readouts + self.num_simulations:   
                    self.run()

                visit_counts = np.array([ self.root.visit_count(idx)
                                            for idx in range(self.board_size**2+1)])
                visit_counts = np.where(visit_counts == 1, 0, visit_counts)             # 若访问次数为1，直接置0
  
                visit_sums = np.sum(visit_counts)                                          # 归一化                                         
                action_probs = visit_counts / visit_sums          # 概率和 board_size**2+1 对应

                if is_selfplay ==False:
                    self.temperature = 0.12    # 不是自对弈直接选较大访问次数的动作，不能直接选最大，若直接选最大每局评估的结果都会一样
                else:
                    gameStep = self.env.getStep(self.root.state)
                    self.temperature = self.config.epsilon_by_frame(gameStep)   # 根据gameStep调节温度系数
             
                visit_tem = np.power(visit_counts, 1.0 / self.temperature)
                visit_probs = np.array(visit_tem )/np.sum(visit_tem)     # 相对于 action_probs 添加了温度系数 （ π也可以是 visit_probs，不过是经过温度变换的）  
                                                                         # visit_probs 也是和 board_size**2+1 对应
                candidate_actions = np.arange(self.board_size**2+1)
                action = np.random.choice(candidate_actions, p=visit_probs)   # 没有加入狄利克雷噪声的影响，访问次数为0的概率为0，不会被选到
               
                root_observation = self.env.encode(self.root.state)
                
                # act_visit = {}
                # for act in range(self.board_size**2):
                #     act_visit[act] = visit_counts[act]
                # print("visit_counts:\n",visit_counts)    
                # print("act_visit:\n",act_visit)
                # print("action1:",action," current_player: ", self.env.getPlayer(self.root.state))


                return  action, action_probs, root_observation    # 下一步动作 ， π， 观测状态

        def select_action(self, gamestate):  # 实际对弈的时候通过该函数 选下一步动作 （不包括自对弈）

             self.root = Node_M(0)
             self.root.state = gamestate

             # 初始化扩展root
             policy, value = self.computeValuePolicy([self.root.state])
             policy, value = policy[0], value[0][0]
             legal_actions = self.env.getLegalAction(self.root.state)    
             action_priors = { idx: p for idx, p in enumerate(policy) if idx in legal_actions} 
             self.root.expand(action_priors, value)
             self.root.real_expanded = True

             action, _, _=self.get_action_probs(is_selfplay=False)
             return action

             
        def select_child(self, node):  # 树搜索选择阶段使用
      
            max_ucb = max(self.ucb_score(node, child)for act, child in node.children.items())
            action = np.random.choice(        # 等于max_ucb的 action可能有好几个 ，在几个中随机选一个
                    [action                      
                        for action, child in node.children.items()
                        if (self.ucb_score(node, child) == max_ucb)
                    ])
            return action, node.children[action]

        def ucb_score(self, parent, child):  #ucb  select_child的子函数
                                                                                        # 这里要加上 ons
            prior_score = self.c* child.prior * np.sqrt(parent.total_visit_count + parent.ons) / (child.total_visit_count +child.ons + 1)  
            value_score =  -child.value()   # 要从本节点的角度看 子节点的价值   取负
            return prior_score + value_score
  
        def complete_update(self, policy, value, path, leaf_state):
            assert policy.shape == (self.board_size **2 + 1,)

            # If a node was picked multiple times (despite vlosses), we shouldn't 
            # expand it more than once.    # 如果一个节点被多次选择(尽管有虚拟损失)，也不应该扩展它超过一次
            leaf_node = path[-1]
            if leaf_node.expanded():
                return
            
            # legal moves.
            legal_actions = self.env.getLegalAction(leaf_state)
  
            scale = sum(policy[legal_actions])
            if scale > 0:
                for act in legal_actions :   # 重新将概率规范化， 去除不合法节点的概率值
                # Re-normalize probabilities.
                       prob = policy[act] / scale
                       leaf_node.children[act].prior = prob
            
                       # initialize child Q as current node's value, to prevent dynamics where
                       # if B is winning, then B will only ever explore 1 move, because the Q
                       # estimation will be so much larger than the 0 of the other moves.
            
                       # Conversely, if W is winning, then B will explore all 362 moves before
                       # continuing to explore the most favorable move. This is a waste of search.
                       leaf_node.children[act].value_sum = -1 * value
          
            leaf_node.real_expanded = True
  
            self.revert_incomplete_update(path)
            self.backpropagate(path, value)    # 回溯
            

        def backpropagate(self, search_path, value):   # 回溯

            for node in reversed(search_path):
                node.value_sum += value 
                node.total_visit_count += 1
                value = -value

  
        def incomplete_update(self, search_path):    # 添加虚拟损失至路径上的节点
           
            for node in reversed(search_path):
                node.ons +=self.wu_loss

        def revert_incomplete_update(self,search_path):
            for node in reversed(search_path):
                node.ons -=self.wu_loss   
  
        def policyValueFn(self, observations):   # 计算 policy 、 Value

            observations = torch.from_numpy(observations).to(next(self.model.parameters()).device)
            policy, value  = self.model.main_prediction(observations)

            policy =policy.detach().cpu().numpy()
            value = value.detach().cpu().numpy()
           
            return  policy, value

        def computeValuePolicy(self, leaf_states):# 计算 action_policy、Value  ,  为了方便后续的扩展
            
            observations = np.array([self.env.encode(leaf_state) for leaf_state in leaf_states],dtype="float32")
            policies, values = self.policyValueFn(observations)

            return policies, values
            

        def update_with_action(self, fall_action):   # 从之前的搜索树继承

                next_state, done =  self.env.step(self.root.state, fall_action)
                self.root = self.root.children[fall_action]
                if not self.root.expanded() :      # 根节点的子节点 可能会存在未扩展的情况
                 
                        self.root.state =next_state
                        policy, value = self.computeValuePolicy([self.root.state])
                        policy, value = policy[0], value[0][0]

                        legal_actions = self.env.getLegalAction(self.root.state)    
                        action_priors = { idx: p for idx, p in enumerate(policy) if idx in legal_actions} 
                        self.root.expand(action_priors, value)
                        self.root.real_expanded = True                        
 
                return done

        def __str__(self):
            return "WU_UCT"


class MCTS(object):

        def __init__(self,config, env, model):

            self.config = config
            self.board_size = config.board_size
            self.virtual_loss = config.virtual_loss  
            self.parallel_readouts =  config.parallel_readouts
            self.model = model
            self.num_simulations =  config.num_simulation
            self.c = config.c_puct
            self.env = env

            # Must run this once at the start to expand the root node. 
            self.reset_root()

        def reset_root(self):
            self.root = Node_M(0)
            self.root.state , _ = self.env.reset()
             
            policy, value = self.computeValuePolicy([self.root.state])
            policy, value = policy[0], value[0][0]

            legal_actions = self.env.getLegalAction(self.root.state)    
            action_priors = { idx: p for idx, p in enumerate(policy) if idx in legal_actions} 
            self.root.expand(action_priors, value)
            self.root.real_expanded = True
  
        def run(self):
                """
                执行 parallel_readouts 次并行搜索 , 
                使用master-slave结构,只并行网络推理部分(因为调用网络占用时间超过整个搜索的80%) 
                """
                paths = []        # parallel_readouts次搜索路径列表
                leaf_states = [] # 叶子节点状态
                failsafe = 0    # 当 game_over , search_path 并不会记录进 paths ， 但是failsafe 会继续计数

                while len(paths) < self.parallel_readouts and failsafe < self.parallel_readouts * 2:
                                            #  循环次数不一定小于 self.parallel_readouts 

                    node = self.root        # 相当于引用 ， node的改变会修改self.root
                    current_tree_depth = 0 
                    search_path = [node]
                    failsafe += 1
                    while node.expanded():   #  选择
                        
                        current_tree_depth += 1
                        next_action, node = self.select_child(node)
                        search_path.append(node)
                     
                    leaf_state ,done = self.env.step(search_path[-2].state, next_action)    # 下一个状态
                    node.state = leaf_state
                       
                    if not done:
                       # 伪扩展 ， 策略跟价值由于是并行输出，会在后面再传递给子节点
                       legal_actions = self.env.getLegalAction(leaf_state)     
                       action_priors_init = { idx: 0.0 for idx  in legal_actions} 
                       node.expand(action_priors_init)     # 扩展， TODO：此时该节点的 real_expanded依然为 false
                   
                    else:        # 如果已经到达终局，不再往下搜索，并且直接使用实际的输赢作为评估 
                        value =  1 if self.env.getPlayer(node.state) ==  self.env.getWinner(node.state) else -1
                        self.backpropagate(search_path, value)
                        continue

                    self.add_virtual_loss(search_path)
                    paths.append(search_path)
                    leaf_states.append(leaf_state)
                
                if paths:
                    probs, values = self.computeValuePolicy(leaf_states)   # leaf_states 是一个 batch的数量
                    for path, leaf_state, prob, value in zip(paths, leaf_states, probs, values):
                        self.revert_virtual_loss(path)
                        self.incorporate_results(prob, value[0], path, leaf_state)


        def get_action_probs(self, is_selfplay = True):  # 返回  下一步动作 ， π， 根节点观测状态
                
                if is_selfplay == True:    # 添加狄利克雷噪声   用的时候把注释去掉
                    self.root.dirichlet_prior()

                current_readouts = self.root.total_visit_count          # current_readouts是继承的 访问次数
                while self.root.total_visit_count < current_readouts + self.num_simulations:   
                    self.run()

                visit_counts = np.array([ self.root.visit_count(idx)
                                            for idx in range(self.board_size**2+1)])
                visit_counts = np.where(visit_counts == 1, 0, visit_counts)             # 若访问次数为1，直接置0

                visit_sums = np.sum(visit_counts)                                          # 归一化                                         
                action_probs = visit_counts / visit_sums          # 概率和 board_size**2+1 对应

                if is_selfplay ==False:
                    self.temperature = 0.12    # 不是自对弈直接选较大访问次数的动作，不能直接选最大，若直接选最大每局评估的结果都会一样
                else:
                    gameStep = self.env.getStep(self.root.state)
                    self.temperature = self.config.epsilon_by_frame(gameStep)   # 根据gameStep调节温度系数
             
                visit_tem = np.power(visit_counts, 1.0 / self.temperature)
                visit_probs = np.array(visit_tem )/np.sum(visit_tem)     # 相对于 action_probs 添加了温度系数 （ π也可以是 visit_probs，不过是经过温度变换的）  
                                                                         # visit_probs 也是和 board_size**2+1 对应
                candidate_actions = np.arange(self.board_size**2+1)
                action = np.random.choice(candidate_actions, p=visit_probs)   # 没有加入狄利克雷噪声的影响，访问次数为0的概率为0，不会被选到
               
                root_observation = self.env.encode(self.root.state)
  
                return  action, action_probs, root_observation    # 下一步动作 ， π， 观测状态

        def select_action(self, gamestate):  # 实际对弈的时候通过该函数 选下一步动作 （不包括自对弈）

             self.root = Node_M(0)
             self.root.state = gamestate

             # 初始化扩展root
             policy, value = self.computeValuePolicy([self.root.state])
             policy, value = policy[0], value[0][0]
             legal_actions = self.env.getLegalAction(self.root.state)    
             action_priors = { idx: p for idx, p in enumerate(policy) if idx in legal_actions} 
             self.root.expand(action_priors, value)
             self.root.real_expanded = True

             action, _, _=self.get_action_probs(is_selfplay=False)
             return action

             
        def select_child(self, node):  # 树搜索选择阶段使用
      
            max_ucb = max(self.ucb_score(node, child)for act, child in node.children.items())
            action = np.random.choice(        # 等于max_ucb的 action可能有好几个 ，在几个中随机选一个
                    [action                      
                        for action, child in node.children.items()
                        if (self.ucb_score(node, child) == max_ucb)
                    ])
            return action, node.children[action]

        def ucb_score(self, parent, child):  #ucb  select_child的子函数

            prior_score = self.c* child.prior * np.sqrt(parent.total_visit_count) / (child.total_visit_count + 1)  
            value_score =  -child.value()   # 要从本节点的角度看 子节点的价值   取负
            return prior_score + value_score
  
        def incorporate_results(self, policy, value, path, leaf_state):
            assert policy.shape == (self.board_size **2 + 1,)

            # If a node was picked multiple times (despite vlosses), we shouldn't 
            # expand it more than once.    # 如果一个节点被多次选择(尽管有虚拟损失)，也不应该扩展它超过一次
            leaf_node = path[-1]
            if leaf_node.expanded():
                return
            
            # legal moves.
            legal_actions = self.env.getLegalAction(leaf_state)
  
            scale = sum(policy[legal_actions])
            if scale > 0:
                for act in legal_actions :   # 重新将概率规范化， 去除不合法节点的概率值
                # Re-normalize probabilities.
                       prob = policy[act] / scale
                       leaf_node.children[act].prior = prob
            
                       # initialize child Q as current node's value, to prevent dynamics where
                       # if B is winning, then B will only ever explore 1 move, because the Q
                       # estimation will be so much larger than the 0 of the other moves.
            
                       # Conversely, if W is winning, then B will explore all 362 moves before
                       # continuing to explore the most favorable move. This is a waste of search.
                       leaf_node.children[act].value_sum = -1 * value
          
            leaf_node.real_expanded = True
  
            self.backpropagate(path, value)    # 回溯


        def backpropagate(self, search_path, value):   # 回溯
            for node in reversed(search_path):
                node.value_sum += value 
                node.total_visit_count += 1
                value = -value


        def add_virtual_loss(self, search_path):    # 添加虚拟损失至路径上的节点
            """
            Propagate a virtual loss up to the root node.
            Args:
                up_to: The node to propagate until. (Keep track of this! You'll
                    need it to reverse the virtual loss later.)
            """  
            # This is a "win" for the current node; hence a loss for its parent node
            # who will be deciding whether to investigate this node again.
         
            for node in reversed(search_path):
                node.value_sum += self.virtual_loss    # 虚拟损失为正 ，在搜索的选择阶段，需要从父节点的角度看时就为负
               

        def revert_virtual_loss(self,search_path):   # 从虚拟损失恢复 ， 至根节点
           
            revert = -1 * self.virtual_loss 
            for node in reversed(search_path):
                 node.value_sum += revert
        

        def policyValueFn(self, observations):   # 计算 policy 、 Value

            observations = torch.from_numpy(observations).to(next(self.model.parameters()).device)
            policy, value  = self.model.main_prediction(observations)

            policy =policy.detach().cpu().numpy()
            value = value.detach().cpu().numpy()
           
            return  policy, value

        def computeValuePolicy(self, leaf_states):# 计算 action_policy、Value  ,  为了方便后续的扩展
            
            observations = np.array([self.env.encode(leaf_state) for leaf_state in leaf_states],dtype="float32")
            
            policies, values = self.policyValueFn(observations)

            return policies, values
            

        def update_with_action(self, fall_action):   # 从之前的搜索树继承

                next_state, done =  self.env.step(self.root.state, fall_action)
                self.root = self.root.children[fall_action]
                if not self.root.expanded() :      # 根节点的子节点 可能会存在未扩展的情况
                 
                        self.root.state =next_state
                        policy, value = self.computeValuePolicy([self.root.state])
                        policy, value = policy[0], value[0][0]

                        legal_actions = self.env.getLegalAction(self.root.state)    
                        action_priors = { idx: p for idx, p in enumerate(policy) if idx in legal_actions} 
                        self.root.expand(action_priors, value)
                        self.root.real_expanded = True                        
 
                return done

        def __str__(self):
            return "MCTS"
   

class WP_MCTS(object):
  
        def __init__(self,config, env, model, sub_model = None):

            self.config = config
            self.board_size = config.board_size
            self.komi = config.komi
            self.tanh_norm =  config.tanh_norm
            self.parallel_readouts =  config.parallel_readouts
            self.model = model
            self.sub_model = sub_model
            self.num_simulations =  config.num_simulation
            self.c1 = config.c_puct1
            self.c2 = config.c_puct2
            self.env = env
            self.wu_loss = config.wu_loss
          
            # Must run this once at the start to expand the root node. 
            self.reset_root()
    
        def reset_root(self):
            self.root = Node_V(0)
            self.root.state , _ = self.env.reset()
              
            policy, value = self.computeValuePolicy([self.root.state])
            policy, value = policy[0], value[0][0]
   
            legal_actions = self.env.getLegalAction(self.root.state)    
            action_priors = { idx: p for idx, p in enumerate(policy) if idx in legal_actions} 
            self.root.expand(action_priors, value)
            self.root.real_expanded = True
    
        def run(self,now_train_step):
                """
                执行 parallel_readouts 次并行搜索 , 
                使用master-slave结构,只并行网络推理部分(因为调用网络占用时间超过整个搜索的80%) 
                """
                paths = []        # parallel_readouts次搜索路径列表
                leaf_states = [] # 叶子节点状态
                failsafe = 0    # 当 game_over , search_path 并不会记录进 paths ， 但是failsafe 会继续计数

                while len(paths) < self.parallel_readouts and failsafe < self.parallel_readouts * 2:
                                            #  循环次数不一定小于 self.parallel_readouts 

                    node = self.root        # 相当于引用 ， node的改变会修改self.root
                    current_tree_depth = 0 
                    search_path = [node]
                    failsafe += 1
                    while node.expanded():   #  选择
                         
                        current_tree_depth += 1
                        next_action, node = self.select_child(node)
                        search_path.append(node)
                      
                    leaf_state ,done = self.env.step(search_path[-2].state, next_action)    # 下一个状态
                    node.state = leaf_state
                       
                    if not done:
                       # 伪扩展 ， 策略跟价值由于是并行输出，会在后面再传递给子节点
                       legal_actions = self.env.getLegalAction(leaf_state)     
                       action_priors_init = { idx: 0.0 for idx  in legal_actions} 
                       node.expand(action_priors_init)     # 扩展， TODO：此时该节点的 real_expanded依然为 false
                   
                    else:        # 如果已经到达终局，不再往下搜索，并且直接使用实际的输赢作为评估 
                        value =  1 if self.env.getPlayer(node.state) ==  self.env.getWinner(node.state) else -1
            
                        self.backpropagate(search_path, value)
                        continue

                    self.incomplete_update(search_path)
                    paths.append(search_path)
                    leaf_states.append(leaf_state)
                
                if paths:
                    len_paths = len(paths)
                    probs, values = self.computeValuePolicy(leaf_states, now_train_step,len_paths)   # leaf_states 是一个 batch的数量
                    for path, leaf_state, prob, value in zip(paths, leaf_states, probs, values):
                    
                        self.revert_incomplete_update(path)
                        self.complete_update(prob, value[0], path, leaf_state)
  
    
        def get_action_probs(self, is_selfplay = True, now_train_step = 0):  # 返回  下一步动作 ， π， 根节点观测状态
                
                if is_selfplay == True:    # 添加狄利克雷噪声   用的时候把注释去掉
                    self.root.dirichlet_prior()

                current_readouts = self.root.total_visit_count          # current_readouts是继承的 访问次数
                while self.root.total_visit_count < current_readouts + self.num_simulations:   
                    self.run(now_train_step)

                visit_counts = np.array([ self.root.visit_count(idx)
                                            for idx in range(self.board_size**2+1)])
                visit_counts = np.where(visit_counts == 1, 0, visit_counts)             # 若访问次数为1，直接置0

                visit_sums = np.sum(visit_counts)                                          # 归一化                                         
                action_probs = visit_counts / visit_sums          # 概率和 board_size**2+1 对应
   
                if is_selfplay ==False:
                    self.temperature = 0.12    # 不是自对弈直接选较大访问次数的动作，不能直接选最大，若直接选最大每局评估的结果都会一样
                else:
                    gameStep = self.env.getStep(self.root.state)
                    self.temperature = self.config.epsilon_by_frame(gameStep)   # 根据gameStep调节温度系数
             
                visit_tem = np.power(visit_counts, 1.0 / self.temperature)
                visit_probs = np.array(visit_tem )/np.sum(visit_tem)     # 相对于 action_probs 添加了温度系数 （ π也可以是 visit_probs，不过是经过温度变换的）  
                                                                         # visit_probs 也是和 board_size**2+1 对应
                candidate_actions = np.arange(self.board_size**2+1)
                action = np.random.choice(candidate_actions, p=visit_probs)   # 没有加入狄利克雷噪声的影响，访问次数为0的概率为0，不会被选到
               
                root_observation = self.env.encode(self.root.state)
  
                return  action, action_probs, root_observation    # 下一步动作 ， π， 观测状态

        def select_action(self, gamestate):  # 实际对弈的时候通过该函数 选下一步动作 （不包括自对弈）

             self.root = Node_V(0)
             self.root.state = gamestate

             # 初始化扩展root
             policy, value = self.computeValuePolicy([self.root.state])
             policy, value = policy[0], value[0][0]
             legal_actions = self.env.getLegalAction(self.root.state)    
             action_priors = { idx: p for idx, p in enumerate(policy) if idx in legal_actions} 
             self.root.expand(action_priors, value)
             self.root.real_expanded = True

             action, _, _=self.get_action_probs(is_selfplay=False)
             return action
 
               
        def select_child(self, node):  # 树搜索选择阶段使用
      
            max_ucb = max(self.ucb_score(node, child)for act, child in node.children.items())
            action = np.random.choice(        # 等于max_ucb的 action可能有好几个 ，在几个中随机选一个
                    [action                      
                        for action, child in node.children.items()
                        if (self.ucb_score(node, child) == max_ucb)
                    ])
            return action, node.children[action]
  
        def ucb_score(self, parent, child):  #ucb  select_child的子函数

            prior_score = self.c1* child.prior * np.sqrt(parent.total_visit_count+
                                           parent.ons) / (child.total_visit_count +child.ons+ 1)  
      
            value_var = np.clip(child.value_var, 0, 3)
            var_score = self.c2* np.sqrt(1+value_var)

            value_score =  -child.value()   # 要从本节点的角度看 子节点的价值   取负
            return prior_score + var_score + value_score
  
        def complete_update(self, policy, value, path, leaf_state):
            assert policy.shape == (self.board_size **2 + 1,)

            # If a node was picked multiple times (despite vlosses), we shouldn't 
            # expand it more than once.    # 如果一个节点被多次选择(尽管有虚拟损失)，也不应该扩展它超过一次
            leaf_node = path[-1]
            if leaf_node.expanded():
                return
            
            # legal moves.
            legal_actions = self.env.getLegalAction(leaf_state)
  
            scale = sum(policy[legal_actions])
            if scale > 0:
                for act in legal_actions :   # 重新将概率规范化， 去除不合法节点的概率值
                # Re-normalize probabilities.
                       prob = policy[act] / scale
                       leaf_node.children[act].prior = prob
            
                       # initialize child Q as current node's value, to prevent dynamics where
                       # if B is winning, then B will only ever explore 1 move, because the Q
                       # estimation will be so much larger than the 0 of the other moves.
            
                       # Conversely, if W is winning, then B will explore all 362 moves before
                       # continuing to explore the most favorable move. This is a waste of search.
                       leaf_node.children[act].value_sum = -1 * value
          
            leaf_node.real_expanded = True
            self.backpropagate(path, value)    # 回溯
  

        def backpropagate(self, search_path, value):   # 回溯
            for node in reversed(search_path):
                node.value_sum += value 
                node.total_visit_count += 1

                node.value_mean_var(value)
                value = -value
  

        def incomplete_update(self, search_path):    # 添加虚拟损失至路径上的节点
           
            for node in reversed(search_path):
                node.ons +=self.wu_loss

        def revert_incomplete_update(self,search_path):
            for node in reversed(search_path):
                node.ons -=self.wu_loss 
               

        def policyValueFn(self, observations, main=True): 

            if main:
                observations = torch.from_numpy(observations).to(next(self.model.parameters()).device)
                policy, value, own  = self.model.main_prediction(observations)
 
                policy =policy.detach().cpu().numpy()
                value = value.detach().cpu().numpy()
                own = own.detach().cpu().numpy()
                return policy, value, own

            else:  
                 observations = torch.from_numpy(observations).to(next(self.model.parameters()).device)
                 policy, value , own  = self.sub_model.main_prediction(observations)
                 policy =policy.detach().cpu().numpy()
                 own = own.detach().cpu().numpy()
                 return policy, own
 
  
        def computeValuePolicy(self, leaf_states, now_train_steps = 0,len_paths = 1):# 计算 action_policy、Value  ,  为了方便后续的扩展
              
            observations = np.array([self.env.encode(leaf_state) for leaf_state in leaf_states],dtype="float32")
            policies, values, own = self.policyValueFn(observations)
  
            # -----------------------------------------策略迁移、价值迁移处理过程----------------------
            komi = [self.komi if self.env.getPlayer(leaf_state) == 2 else -self.komi for leaf_state in leaf_states ]
            if self.sub_model and now_train_steps <=3500 : 
                gamma = self.config.value_ratio_by_frame(now_train_steps)   #
     
                sub_observations = np.array([self.env.subEncode(observation) for observation in observations])
                sub_observations = sub_observations.reshape(-1,sub_observations.shape[-3],sub_observations.shape[-2],sub_observations.shape[-1])

                sub_policy, sub_own = self.policy_value_fn(sub_observations,False)
                global_own =np.array([self.sub_to_global_encode(sub_own[i*4:i*4+4]) for i in range(len_paths)])

                own_error = np.sum(global_own,1)
                trans_values = own_error + komi
                trans_values = np.tanh(self.tanh_norm * trans_values)
                values = (1 - gamma) * values + gamma * trans_values              # 混合价值   
 
                sub_policy = np.array([p[0:-1] for p in sub_policy])
                global_policies =np.array([self.sub_to_global_encode(sub_policy[i*4:i*4+4]) for i in range(len_paths)])

                act_prob_pass = 1e-5
                global_policies = np.array([np.append(global_policy, act_prob_pass) for global_policy in global_policies])
                    
                self.temperature_p = 0.45
                global_policies = np.power(global_policies, 1.0 /self.temperature_p )
                trans_prior = global_policies /np.sum(global_policies, keepdims=True) #keepdims=True 保持维度相同才能触发广播机制 
                policies = (1 - gamma) * policies + gamma *trans_prior    # 混合策略
            #-----------------------------------------------------------------------------------------------------
  
            return policies , values

              
        def sub_to_global_encode(self,sub_features):   # 将局部own、 局部policy 做成 全局own、 全局policy

            local_board_size = self.config.local_board_size 
            sub_feature_0 = sub_features[0].reshape(local_board_size,local_board_size)
            sub_feature_1 = sub_features[1].reshape(local_board_size, local_board_size)
            sub_feature_2 = sub_features[2].reshape(local_board_size, local_board_size)
            sub_feature_3 = sub_features[3].reshape(local_board_size, local_board_size)
   
            global_feature = np.zeros((self.board_size,self.board_size))
             # 左上
            global_feature[0:(self.board_size+1)//2, 0:(self.board_size+1)//2] += sub_feature_0[0:(self.board_size+1)//2, 0:(self.board_size+1)//2] 
             # 右上
            global_feature[0:(self.board_size+1)//2, (self.board_size-1)//2: ] += sub_feature_1[0:(self.board_size+1)//2,local_board_size - (self.board_size+1)//2 :]     
             # 左下 
            global_feature[(self.board_size-1)//2: , 0:(self.board_size+1)//2] += sub_feature_2[local_board_size - (self.board_size+1)//2 :, 0:(self.board_size+1)//2] 
             # 右下
            global_feature[(self.board_size-1)//2: , (self.board_size-1)//2: ] += \
                                                      sub_feature_3[local_board_size - (self.board_size+1)//2 :, local_board_size - (self.board_size+1)//2 :]

            global_feature[(self.board_size-1)//2,:] =  global_feature[(self.board_size-1)//2,:]/2
            global_feature[ :, (self.board_size-1)//2] =  global_feature[ : , (self.board_size-1)//2]/2
  
            return global_feature


        def update_with_action(self, fall_action):   # 从之前的搜索树继承

                next_state, done =  self.env.step(self.root.state, fall_action)
                self.root = self.root.children[fall_action]
                if not self.root.expanded() :      # 根节点的子节点 可能会存在未扩展的情况
                 
                        self.root.state =next_state
                        policy, value = self.computeValuePolicy([self.root.state])
                        policy, value = policy[0], value[0][0]

                        legal_actions = self.env.getLegalAction(self.root.state)    
                        action_priors = { idx: p for idx, p in enumerate(policy) if idx in legal_actions} 
                        self.root.expand(action_priors, value)
                        self.root.real_expanded = True                        
 
                return done

        def __str__(self):
            return "WP_MCTS"



    

@ray.remote
class SelfPlay():

    def __init__(self, config):
          
          self.config = config
          self.env = GoEnv(config)
          self.device = config.device
         
          print(self.device)
          self.model = TransGoNetwork(self.config).to(self.device)
          self.model.eval()

          self.sub_model = None
          if config.init_sub_model:
              self.sub_model = MutativeNetwork(self.config).to(self.device)
              net_params = torch.load(config.init_sub_model)
              self.sub_model.load_state_dict(net_params)
              self.sub_model.eval()
        

    def continuous_self_play(self, shared_storage_worker, mem):  # 自对弈

        with torch.no_grad():

            train_agent = WP_MCTS(self.config,self.env,self.model, self.sub_model)  # 创建自对弈代理
            while True:  # 循环进行自对弈
          
                start = time.time()
                train_agent.reset_root()
              
                observations, mcts_probs, current_players = [], [], []
                train_agent.model.set_weights(ray.get(shared_storage_worker.get_info.remote("weights")))

                while True:  # 未到终局不会跳出这个循环

                    act_action, action_probs, root_observation =  train_agent.get_action_probs()
                    # print("root:",root_observation)
                    # print("root:",len(action_probs))
                    observations.append(root_observation)
                    mcts_probs.append(action_probs)
        
                    root_current_player = self.env.getPlayer(train_agent.root.state)
                    current_players.append(root_current_player)
                    
                    done = train_agent.update_with_action(act_action)   # 执行下一步 ，继承之前的搜索树
                
                    shared_storage_worker.set_info.remote("now_play_steps")
                    if  done == True:  # 到达终局   收集数据

                        win_zs = np.zeros(len(current_players))
                        winner = self.env.getWinner(train_agent.root.state)
                        win_zs[np.array(current_players) == winner] =  1
                        win_zs[np.array(current_players) !=winner] = -1
                    
                        BLACK  = 1
                        score, territory = self.env.getScoreAndTerritory(train_agent.root.state)
                        own_zs = np.zeros((len(current_players), self.config.board_size ** 2),)
                        own_zs[np.array(current_players) == BLACK] = territory
                        own_zs[np.array(current_players) != BLACK] = -1*territory
  
  
                        for observation, mcts_prob, win_z, own_z in zip(observations, mcts_probs, win_zs, own_zs):    #  数据增强 x8   （数据增强也可以放到取batchsize训练样本的下一步，即取出数据再进行数据变换）
                                for i in [1, 2, 3, 4]:         #  i=1，2，3，4代表逆时针旋转的角度，90，180，270，360
                          # 旋转------------------------------------------
                                    prob_p = mcts_prob[:-1]     # 棋盘点概率分布
                                    prob_pass = mcts_prob[-1]   # pass 停着点 概率

                                    equi_prob = np.rot90(prob_p.reshape(self.config.board_size, self.config.board_size), i)
                                    equi_prob1 = equi_prob.flatten()
                                    equi_prob1 = np.append(equi_prob1, prob_pass)

                                    equi_observation = np.array([np.rot90(s, i) for s in observation])   #一个状态里的平面一个个单独出来旋转                      
                                    equi_own = np.rot90(own_z.reshape(self.config.board_size, self.config.board_size), i)
                                    
                                    mem.append.remote(equi_observation, equi_prob1, win_z, equi_own.flatten())   # 收集数据
                          # 翻转------------------------------------------
                                    equi_prob2 = np.fliplr(equi_prob)
                                    equi_prob2 = equi_prob2.flatten()
                                    equi_prob2 = np.append(equi_prob2, prob_pass)

                                    equi_observation = np.array([np.fliplr(s) for s in equi_observation])    # 左右翻转
                                    equi_own = np.fliplr(equi_own)

                                    mem.append.remote(equi_observation, equi_prob2, win_z, equi_own.flatten())   # 收集数据

                        shared_storage_worker.set_info.remote("now_play_games")
                        break   # 到达终局跳出循环 ， 执行下一局

                while (          # 调整训练/自对弈比率
                    ray.get(shared_storage_worker.get_info.remote("now_train_steps"))
                    / max(
                        1, ray.get(shared_storage_worker.get_info.remote("now_play_steps"))
                    )
                    < ray.get(shared_storage_worker.get_info.remote("train_play_ratio"))) \
                    and ray.get(shared_storage_worker.get_info.remote("adjust_train_play_ratio")) \
                    and ray.get(shared_storage_worker.get_info.remote("now_play_games")) \
                        < ray.get(shared_storage_worker.get_info.remote("game_total_num")):  
                   
                    time.sleep(0.5)

                end = time.time()
                print("run time:%.4fs" % (end - start))
    
    
    @torch.no_grad()
    def policy_evaluate(self, n_games=10, shared_storage_worker=None):    # 评估 ，新模型每次和评估模型对弈10局  ，   
                                                 #（为了提高效率 ，可以用更少的搜索次数，但是在这个程序里设定评估搜索次数跟自对弈的一样），效率变高，可以试一elo进行评分
       
        train_agent =  WP_MCTS(self.config,self.env,self.model)    # train_agent 使用最新模型
        train_agent.model.set_weights(ray.get(shared_storage_worker.get_info.remote("weights")))

        model_evaluate = TransGoNetwork(self.config).to(self.device)
        model_evaluate.eval()
        model_evaluate.set_weights(ray.get(shared_storage_worker.get_info.remote("evaluate_weights")))
        evaluate_agent = WP_MCTS(self.config,self.env,model_evaluate)    # evaluate_agent 使用评估模型（旧模型）
        
        BLACK = 1
        WHITE = 2
        color = BLACK     # color 对应的是训练模型的执子颜色， 黑/白轮着
      
        win_num = 0
        lose_num = 0
        evaluate_score = ray.get(shared_storage_worker.get_info.remote("evaluate_score"))

        info2 = None
        for i in range(n_games):
            state, done =  self.env.reset()
            if color == BLACK:
                bots = {BLACK: train_agent, WHITE: evaluate_agent}
            else:
                bots = {BLACK: evaluate_agent, WHITE: train_agent}
            while not done:
    
                bot_action = bots[self.env.getPlayer(state)].select_action(state)
                state, done = self.env.step(state, bot_action)
  
            winner = self.env.getWinner(state)
            print("simulate round: {}".format(i+1),",  winer is :",winner, ',  model player is :',color)   # 1为黑 ， 2为白  
            info2 = "simulate round: {},  winer is : {},  model player is : {}\n".format(i+1,winner,color)

            if winner == color:
                    win_num += 1
            else:
                    lose_num += 1
            color = BLACK + WHITE - color

        win_ratio = win_num / n_games

        print("evaluate_score:{}, win: {},  lose: {}".format(evaluate_score,
                win_num,  lose_num))
        info3 = "evaluate_score:{}, win: {}, lose: {}\n".format(evaluate_score,
                win_num, lose_num)

        if win_ratio == 1:   # 
            shared_storage_worker.set_info.remote("evaluate_score",evaluate_score+100)     # 只要10局全胜 ，得分 + 100
            shared_storage_worker.set_info.remote("evaluate_weights",shared_storage_worker.get_info.remote("weights"))  # 更新"evaluate_weights"为 "weights"的权重
                                            # 这里"weights"其实已经不是评估时的weights，但是不要紧，只要知道模型在一直变强就行
      
        return win_ratio, info2, info3
