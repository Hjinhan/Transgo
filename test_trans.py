import random
import numpy as np
import os
import pickle
import time

from GoEnv.environment import GoEnv
from self_play import MCTS
from configure import Config
from model import TransGoNetwork
from model import MutativeNetwork


class RandomBot():

    def __init__(self, go_env):
        self.go_env = go_env

    def select_action(self, state):
        candidates = []
        candidates = self.go_env.getLegalAction(state)
        action = random.choice(candidates)
        return action


def evaluate1( model1, sub_model1, model2, n_games=10, config= None):   # 用了变尺度训练的模型和其他模型对弈
      
    go_env = GoEnv(config)
    trans_agent =  MCTS(config, go_env, model1, sub_model1)
    base_agent = MCTS(config, go_env, model2)

    BLACK = 1
    WHITE = 2
    color = BLACK     # color 对应的是训练模型的执子颜色， 黑/白轮着
    win_num = 0
    lose_num = 0

    for i in range(n_games):
            state, done =  go_env.reset()
            if color == BLACK:
                bots = {BLACK: trans_agent, WHITE: base_agent}
            else:
                bots = {BLACK: base_agent, WHITE: trans_agent}
            while not done:
                bot_action = bots[go_env.getPlayer(state)].select_action(state)
                state, done = go_env.step(state, bot_action)

            winner = go_env.getWinner(state)
            print("simulate round: {}".format(i+1),",  winer is :",winner, ',  model player is :',color)   # 1为黑 ， 2为白  

            if winner == color:
                    win_num += 1
            else:
                    lose_num += 1
            color = BLACK + WHITE - color

    print("win: {}, lose: {}".format(win_num, n_games- win_num))


def evaluate2(model1,sub_model1,n_games=20, config= None):  # 变尺度模型和随机走子对弈
      
    go_env = GoEnv(config)

    trans_agent =  MCTS(config, go_env, model1, sub_model1)
    base_agent = RandomBot(go_env)

    BLACK = 1
    WHITE = 2
    color = BLACK     # color 对应的是训练模型的执子颜色， 黑/白轮着
    win_num = 0
    lose_num = 0

    for i in range(n_games):
            state, done =  go_env.reset()
            if color == BLACK:
                bots = {BLACK: trans_agent, WHITE: base_agent}
            else:
                bots = {BLACK: base_agent, WHITE: trans_agent}
            while not done:
                bot_action = bots[go_env.getPlayer(state)].select_action(state)
                state, done = go_env.step(state, bot_action)

                # print("bot_action:", bot_action)
                # go_env.encode_grid(state)

            winner = go_env.getWinner(state)
            print("simulate round: {}".format(i+1),",  winer is :",winner, ',  model player is :',color)   # 1为黑 ， 2为白  

            if winner == color:
                    win_num += 1
            else:
                    lose_num += 1
            color = BLACK + WHITE - color

    print("win: {}, lose: {}".format(win_num, n_games- win_num))


if __name__ == '__main__':
    
    config = Config()
    go_env = GoEnv(config)
    num_games = 10
    
    model = TransGoNetwork(config).to(config.device)

    sub_model = MutativeNetwork(config).to(config.device)
    path =  "./results/sub_policy_600.model"
  
    if os.path.exists(path):
            print("agent_trans_path is exists")
            with open(path, "rb") as f:
                    model_weights = pickle.load(f)
                    sub_model.set_weights( model_weights["weights"])
         
    start = time.time()
    evaluate2(model, sub_model, n_games=40,config= config)
    end = time.time()
    print("run time:%.4fs" % (end - start))