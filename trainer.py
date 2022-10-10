import copy
import time

import ray
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
  
from model import TransGoNetwork


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

@ray.remote
class Trainer:

    def __init__(self, initial_checkpoint, config, model_file = None):

        self.config = config

        self.now_train_step= initial_checkpoint["now_train_steps"]
        self.checkpoint_interval = config.checkpoint_interval
        self.l2_const = config.l2_const
        self.batch_size = config.batch_size
        self.device = config.device

        self.model = TransGoNetwork(config).to(self.device)
        self.model.train()

        self.optimizer = optim.Adam( self.model.parameters(),betas=(0.5, 0.999),  
                                    weight_decay=self.l2_const)

        if model_file:
            self.model.set_weights(initial_checkpoint["weights"])

    def train_step(self, mem, shared_storage):    # 训练

        while ray.get(shared_storage.get_info.remote("now_play_games")) < 1 :
              time.sleep(0.1)
           
        while True:
           
            mini_batch = ray.get(mem.sample.remote(self.batch_size))   # tree_idxs 用random采样返回的是None 
         
           #------网络输入 、 策略标签、价值标签，归属标签
            state_batch, act_probs_batch, win_batch, own_z_batch = map(np.stack, zip(*mini_batch))     # map(function, iterable)
            # print("act_probs_batch",act_probs_batch)
            state_batch = torch.FloatTensor(state_batch).to(self.device)
            act_probs_batch = torch.FloatTensor(act_probs_batch).to(self.device)
            win_batch = torch.FloatTensor(win_batch).to(self.device)
            own_z_batch = torch.FloatTensor(own_z_batch).to(self.device)
         
            self.optimizer.zero_grad()   # 清除历史梯度
            set_learning_rate(self.optimizer, ray.get(shared_storage.get_info.remote("learn_rate")))     # 设置学习率

            act_probs, value, own = self.model.main_prediction(state_batch)  # 正向传播
             
            value_loss = F.mse_loss(value.view(-1), win_batch)   # 均方损失
            own_loss = F.mse_loss(own, own_z_batch)

            # 动作策略损失
            act_policy_loss = -torch.mean(torch.sum(act_probs_batch * torch.log(act_probs), 1)) 
                                  #实际只要观察策略损失下降情况，就能够看出来训练的进展 ， 价值均方误差一般都比较小
  
            # 最大熵损失
            entropy_loss = torch.mean(torch.sum(act_probs * torch.log(act_probs), 1))    # 注意：无负号

            loss = value_loss + 0.75*own_loss + 1.15*act_policy_loss  + 0.02*entropy_loss
            loss.backward()            # 反向传播，求导
            self.optimizer.step()      # 更新参数
            
            self.now_train_step += 1
            if self.now_train_step % self.checkpoint_interval == 0:  # 每间隔 config.checkpoint_interval 就将 权重保存起来
                shared_storage.set_info.remote(
                        "weights", copy.deepcopy(self.model.get_weights()),
                       )


            shared_storage.set_info.remote(
                {
                    "now_train_steps": self.now_train_step,
                    "total_loss": loss.item(),
                    "value_loss": value_loss.item(),
                    "act_policy_loss": act_policy_loss.item(),
                    "entropy_loss": entropy_loss.item(),
                    "own_loss": own_loss.item(),
                }
            )
     
 
            while (
                        ray.get(shared_storage.get_info.remote("now_train_steps"))
                        / max(
                    1, ray.get(shared_storage.get_info.remote("now_play_steps"))
                )
                        > ray.get(shared_storage.get_info.remote("train_play_ratio"))
                        and ray.get(shared_storage.get_info.remote("now_play_games"))
                        < ray.get(shared_storage.get_info.remote("game_total_num"))
                    and ray.get(shared_storage.get_info.remote("adjust_train_play_ratio"))
                    == True
                ):
                    time.sleep(0.5)

