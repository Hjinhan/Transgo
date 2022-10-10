import copy
import os
import ray
import torch

@ray.remote
class SharedStorage:

    def __init__(self, checkpoint,config):
        self.config = config
        self.current_checkpoint = copy.deepcopy(checkpoint)

    def get_info(self, keys):
        if isinstance(keys, str):
            return self.current_checkpoint[keys]
        elif isinstance(keys, list):
            return {key: self.current_checkpoint[key] for key in keys}
        else:
            raise TypeError

    def set_info(self, keys, values=None):
        if keys == "now_play_steps" or keys == "now_play_games" :
            self.current_checkpoint[keys]+=1
        elif isinstance(keys, str) and values is not None:
            self.current_checkpoint[keys] = values
        elif isinstance(keys, dict):
            self.current_checkpoint.update(keys)
        else:
            print(keys)
            raise TypeError

        # 修改学习率     / 按照对弈局数修改 
        if keys == "now_play_games" and self.current_checkpoint["adjust_lr"] == True:
            lr_ = self.config.ad_lr(self.current_checkpoint["now_play_games"], self.current_checkpoint["learn_rate"])
            self.current_checkpoint["learn_rate"] = lr_

         # 修改训练/对弈比率     / 按照到目前为止进行的所有对弈的回合数修改 
        if keys == "now_play_steps" and self.current_checkpoint["adjust_train_play_ratio"] == True \
                                      and self.current_checkpoint["now_play_games"]  > 0 :
            now_play_steps = self.current_checkpoint["now_play_steps"]
            current_train_play_ratio = self.current_checkpoint["train_play_ratio"]
            train_play_ratio_ = self.config.ad_train_play_ratio( now_play_steps, current_train_play_ratio )
            self.current_checkpoint["train_play_ratio"] = train_play_ratio_

      