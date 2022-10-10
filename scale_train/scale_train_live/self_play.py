
import numpy as np
import ray
import time
import torch
import math
import random

from model import AlphaZeroNetwork
from GoEnv.environment import GoEnv


class Node:
    def __init__(self, prior):
        
        self.action_priors = {}   # 字典 {动作：概率} 所有子节点对应的动作跟先验概率
        self.prior = prior
        self.state = None
        self.total_visit_count = 0
        self.value_sum = 0
        self.children = {}
      
    def expanded(self):     # 判断子节点是否扩展过
        return len(self.children) > 0

    def value(self):       # 期望价值
        if self.total_visit_count == 0:
            return 0
        return self.value_sum / self.total_visit_count

    def expand(self, action_priors):   # 扩展新节点 ， 一次扩展所有合法动作

        self.action_priors = action_priors  
        for action, p in self.action_priors.items():
            self.children[action] = Node(p)

    def visit_count(self, action):   # 取出每个动作的访问次数（不一定是子节点）
        if action in self.children:
            return self.children[action].total_visit_count
        return 0
    
    def dirichlet_prior(self):  # 添加狄利克雷噪声   0.03 ， 0.25 是可调参数 ， 一般固定 0.25 调0.03这个参数
 
        actions = list(self.children.keys())
        noise = np.random.dirichlet([0.03] * len(actions))
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - 0.25) + n * 0.25

   
class MCTS(object):

        def __init__(self,config, env, model):

            self.config = config
            self.board_size = config.board_size
            self.local_board_size = config.local_board_size

            self.model = model
            self.num_simulations =  config.num_simulation
            self.c = config.c_puct
            self.komi = config.komi
            self.tanh_norm = config.tanh_norm
            self.env = env

            self.root = Node(0)
            self.root.state , _ = self.env.reset()
            action_priors,  _ = self.computeValuePolicy(self.root.state)
            self.root.expand(action_priors)

        def run(self, is_selfplay):

                if is_selfplay == True:    # 添加狄利克雷噪声   用的时候把注释去掉
                    self.root.dirichlet_prior()

                for _ in range(self.num_simulations):  
                    node = self.root        # 相当于引用 ， node的改变会修改self.root
                    search_path = [node]
                    current_tree_depth = 0     

                    while node.expanded():   #   选择
                        
                        current_tree_depth += 1
                        next_action, node = self.select_child(node)
                        search_path.append(node)
                   
                    next_action = self.env.local_act_to_act(next_action)      #! mutative
                    new_state ,done = self.env.step(search_path[-2].state, next_action)    # 下一个状态
                    node.state = new_state
                    
                    action_priors, value = self.computeValuePolicy(new_state)   # 评估

                    if not done:
                       node.expand(action_priors)     # 扩展
                    else:        # 如果已经到达终局，不再往下搜索 ， 并且直接使用实际的输赢作为评估 ，这里跟之前的版本有点改动
                        value =  1 if self.env.getPlayer(node.state) ==  self.env.getWinner(node.state) else -1

                    self.backpropagate(search_path, value)    #  回溯   当前节点的value在回溯时才更新  

        def get_action_probs(self, is_selfplay = True):  # 返回  下一步动作，π，根节点观测状态

                self.run(is_selfplay)
                visit_counts = np.array([ self.root.visit_count(idx)
                                            for idx in range(self.local_board_size**2+1)])   #! mutative
                
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
                candidate_actions = np.arange(self.local_board_size**2+1)   #! mutative
                local_action = np.random.choice(candidate_actions, p=visit_probs)   # 没有加入狄利克雷噪声的影响，访问次数为0的概率为0，不会被选到

                action = self.env.local_act_to_act(local_action)    #! local  mutative
                # print("local_action:", local_action)
                # print("action:", action)
                root_local_observation = self.env.localEncode(self.root.state) #! mutative
  
                return  action, action_probs, root_local_observation    # 下一步动作 ， π， 观测状态

        def select_action(self, gamestate):  # 实际对弈的时候通过该函数 选下一步动作 （不包括自对弈）

             self.root = Node(0)
             self.root.state = gamestate

             action_priors, value = self.computeValuePolicy(self.root.state)
             self.root.expand(action_priors)

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
            if child.total_visit_count > 0:
                value_score =  -child.value()   # 要从本节点的角度看 子节点的价值   取负
            else:
                value_score = 0
            return prior_score + value_score

        def backpropagate(self, search_path, value):   # 回溯
            for node in reversed(search_path):
                node.value_sum += value 
                node.total_visit_count += 1
                value = -value

        def policyValueFn(self, local_observation):   # 计算 policy 、 Value   #! mutative

            model_input = np.expand_dims(local_observation,0)
            model_input = torch.from_numpy(model_input).to(next(self.model.parameters()).device)
            policy, value, own  = self.model.main_prediction(model_input)

            policy =policy.detach().cpu().numpy()
            value = value.detach().cpu().numpy()
            own =own.detach().cpu().numpy()
           
            return  policy[0], value[0][0], own[0]

        def computeValuePolicy(self, gamestate):# 计算 action_policy、Value  ,  为了方便后续的扩展
            
            local_observation =  self.env.localEncode(gamestate)            #!  mutative
            policy, value1, own = self.policyValueFn(local_observation)

            komi = self.komi if self.env.getPlayer(gamestate) == 2 else -1 * self.komi
            own_error = np.sum(own)
            value2 = own_error + komi
            value2 = math.tanh(self.tanh_norm * value2)
            value = 0.9 * value1 +0.1 * value2

            local_legal_actions = self.env.getLegalAction(gamestate)       #! mutative
            action_priors = { idx: p for idx, p in enumerate(policy) if idx in local_legal_actions}    # action_priors 是合法动作对应的概率

            return action_priors, value
            

        def update_with_action(self, fall_action):   # 从之前的搜索树继承

                next_state, done =  self.env.step(self.root.state, fall_action)

                local_fall_action = self.env.act_to_local_act(fall_action)    #!  mutative
                # print("local_action2 :", local_fall_action)
                self.root = self.root.children[local_fall_action]
                if not self.root.expanded() :      # 根节点的子节点 可能会存在未扩展的情况
                        self.root.state =next_state
                        action_priors, _ = self.computeValuePolicy(next_state)
                        self.root.expand(action_priors)
                return done
         
        def reset_root(self, random_play_init = False):
            self.root = Node(0)
            state, _ =  self.env.reset()
            num_init_action = random.randint(self.local_board_size-3,self.local_board_size+4)
            if random_play_init == True:
                for _ in range(num_init_action) :
                  random_bound_action = self.env.get_random_init_action(state)   
                  state, _ =  self.env.step(state, random_bound_action)
            self.root.state = state
            action_priors, _ = self.computeValuePolicy(self.root.state)
            self.root.expand(action_priors)

        def __str__(self):
            return "MCTS"

@ray.remote
class SelfPlay():

    def __init__(self, config):
          
          self.config = config
          self.env = GoEnv()
          self.device = config.device
         
          print(self.device)
          self.model = AlphaZeroNetwork(self.config).to(self.device)
          self.model.eval()

    def continuous_self_play(self, shared_storage_worker, mem):  # 自对弈

        with torch.no_grad():

            train_agent = MCTS(self.config,self.env,self.model)  # 创建自对弈代理
            while True:  # 循环进行自对弈
          
                start = time.time()
                train_agent.reset_root(self.config.random_play_init)    # 是否开局随机局部边界外围落子
              
                local_observations, mcts_probs, current_players = [], [], []    #! local_observations
                train_agent.model.set_weights(ray.get(shared_storage_worker.get_info.remote("weights")))

                while True:  # 未到终局不会跳出这个循环

                    act_action, action_probs, root_local_observation =  train_agent.get_action_probs()

                    local_observations.append(root_local_observation)
                    mcts_probs.append(action_probs)

                    root_current_player = self.env.getPlayer(train_agent.root.state)
                    current_players.append(root_current_player)
                    # print("children",train_agent.root.children.keys())

                    done = train_agent.update_with_action(act_action)   # 执行下一步 ，继承之前的搜索树

                    # self.env.board_grid(train_agent.root.state)

                    shared_storage_worker.set_info.remote("now_play_steps")
                    if  done == True:  # 到达终局   收集数据

                        win_z = np.zeros(len(current_players))
                        winner = self.env.getWinner(train_agent.root.state)
                        win_z[np.array(current_players) == winner] =  1
                        win_z[np.array(current_players) !=winner] = -1

                        BLACK  = 1
                        score, local_territory = self.env.getScoreAndTerritory(train_agent.root.state)
                        own_z = np.zeros((len(current_players), self.config.local_board_size ** 2),)
                        own_z[np.array(current_players) == BLACK] = local_territory
                        own_z[np.array(current_players) != BLACK] = -1*local_territory
                   
                        # print("own_z.shape :",own_z.shape)
                        for observation, mcts_prob, per_win_z, per_own_z in zip(local_observations, mcts_probs, win_z,own_z):   
                                mem.append.remote(observation, mcts_prob, per_win_z, per_own_z)   # 收集数据

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
                        < ray.get(shared_storage_worker.get_info.remote("game_batch_num")):  
                   
                    time.sleep(0.5)
                end = time.time()
                print("run time:%.4fs" % (end - start))
    
    @torch.no_grad()
    def policy_evaluate(self, n_games=10, shared_storage_worker=None):    # 评估 ，新模型每次和评估模型对弈10局  ，   
                                                 #（为了提高效率 ，可以用更少的搜索次数，但是在这个程序里设定评估搜索次数跟自对弈的一样），效率变高，可以试一elo进行评分
     
        train_agent =  MCTS(self.config,self.env,self.model)    # train_agent 使用最新模型
        train_agent.model.set_weights(ray.get(shared_storage_worker.get_info.remote("weights")))

        model_evaluate = AlphaZeroNetwork(self.config).to(self.device)
        model_evaluate.eval()
        model_evaluate.set_weights(ray.get(shared_storage_worker.get_info.remote("evaluate_weights")))
        evaluate_agent = MCTS(self.config,self.env,model_evaluate)    # evaluate_agent 使用评估模型（旧模型）
        
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
