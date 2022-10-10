"""
多线程版本
"""

import numpy as np
import torch
import time
import pickle

import os
import ray
import copy

from model import AlphaZeroNetwork
from self_play import SelfPlay
from configure import Config
from trainer import Trainer
from replay_buffer import ReplayMemory
import shared_storage

from tensorboardX import SummaryWriter
writer = SummaryWriter('runs1/loss1')
 
ray.init()

class TrainPipeline(object):
    def __init__(self,config):

        self.config = config
        self.now_train_steps = 0 # 训练迭代次数
        self.now_play_steps = 0  # 对弈回合数
        self.now_play_games = 0  # 对弈局数
        self.best_win_ratio = 0.0 # 评估模型时的最好胜率（胜率为1时evaluate_score得分会+100，并且best_win_ratio会重置为0）
        
        self.game_batch_num = config.game_batch_num
        self.init_evaluate_score = config.init_evaluate_score
        self.evaluate_num = config.evaluate_num
        self.results_path = config.results_path
        self.record_train = config.record_train

        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

        self.checkpoint = {
            "weights": None,
            "evaluate_weights":None,
            "learn_rate":self.config.learn_rate,
            "adjust_lr": self.config.adjust_lr,
            "adjust_train_play_ratio": self.config.adjust_train_play_ratio,
            "train_play_ratio":self.config.train_play_ratio,

            "now_train_steps": self.now_train_steps,
            "now_play_steps": self.now_play_steps,
            "now_play_games": self.now_play_games,     
            "game_batch_num": self.game_batch_num,
            "evaluate_score": self.init_evaluate_score,

            "total_loss": 0.,
            "value_loss": 0.,
            "policy_loss": 0.,
            "own_loss": 0.,
        }

    def train(self):

        self.num_gpus = torch.cuda.device_count()  
        print(self.num_gpus) 

        cpu_actor = CPUActor.remote()
        cpu_weights = cpu_actor.get_initial_weights.remote(self.config)
        self.checkpoint["weights"] = copy.deepcopy(ray.get(cpu_weights))
        self.checkpoint["evaluate_weights"] = copy.deepcopy(ray.get(cpu_weights))

        if self.config.init_model:   # 加载模型
            if os.path.exists(self.config.init_model):
                with open(self.config.init_model, "rb") as f:
                    model_weights = pickle.load(f)
                    self.checkpoint["weights"] = model_weights["weights"]
                    self.checkpoint["evaluate_weights"] = model_weights["evaluate_weights"]

        #! 训练及自对弈的gpu需在下面根据需要调整
        
        #开多卡时：
        # 1、每个线程使用的gpu比如是0.5（张） ， 就必须是同一张gpu上的0.5
        # 2、training_worker使用的gpu资源有时候比如是0.8，则必须首先单独安排一张gpu给它，如果这张卡先被其他线程占用，剩下的不到0.8，也会报错
        # 3、单个线程使用的gpu大于1时，那它能使用的gpu资源也必须是整数，1、2、3..   不能是1.3（张），1.5

        # inint workers

        self.shared_storage_worker = shared_storage.SharedStorage.remote(self.checkpoint,self.config)   # 存储需要记录和共享的数据
        self.mem = ReplayMemory(self.config)  # 启动 ReplayMemory ， 也是单独开一个线程

        if self.config.init_buffer:  # 加载数据  要在 self.mem 启动之后
            self.load_buffer(replay_buffer_path= self.config.init_buffer)

        totalGpuThreads =  self.config.play_workers_num+2    # 使用到gpu的线程数
        num_gpus_per_worker = self.num_gpus/totalGpuThreads  # 每个线程使用的gpu数量，可以小于1， 平均即可，单卡的时候会自动分配
    
        self.self_play_workers = [                      # 自对弈并行num_workers局
                    SelfPlay.options( num_gpus=num_gpus_per_worker,).     #   
                    remote(self.config)
                   for _ in range(self.config.play_workers_num)
                ]
     
        self.training_worker = Trainer.options(num_gpus=num_gpus_per_worker, ). \
            remote(self.checkpoint,self.config, self.config.init_model)             # 

        self.evaluate_worker = SelfPlay.options(num_gpus=num_gpus_per_worker,).remote(self.config)

        # Launch workers
        [
            self_play_worker.continuous_self_play.remote(
                self.shared_storage_worker, self.mem
            )
            for self_play_worker in self.self_play_workers
        ]

        self.training_worker.train_step.remote(self.mem, self.shared_storage_worker)

        self.logging_loop()    # 一个循环运行的函数，如果没有，上面的线程不会一直运行下去

    def logging_loop(self):   # 记录训练情况

        counter = 0
        a = 0

        keys = [
            "now_train_steps",
            "now_play_steps",
            "now_play_games",
            "train_play_ratio",
            "learn_rate",

            "total_loss",
            "value_loss",
            "policy_loss",
            "own_loss",        
        ]
        info = ray.get(self.shared_storage_worker.get_info.remote(keys))  # info是字典 ；取出keys中的数据
        try:
            while info["now_play_games"] < self.game_batch_num:

                if (counter) % 15 == 0:
                    actual_train_play_ratio = 0
                    if info["now_play_steps"]>1:
                        actual_train_play_ratio = info["now_train_steps"] / info["now_play_steps"]
                    
                    print("counter:", counter)
                    print("learn_rate:",info["learn_rate"],", total_loss:",info["total_loss"], ", value_loss:",info["value_loss"],",policy_loss:", 
                        info["policy_loss"],  ", own_loss:",info["own_loss"], ", now_train_steps:",info["now_train_steps"],", play_steps_num:",info["now_play_steps"],", actual_train_play_ratio:",
                        actual_train_play_ratio,", set_train_play_ratio:",info["train_play_ratio"],", play_games_num:", info["now_play_games"], )

                    with open(os.path.join(self.results_path, self.record_train), "a") as f:
                      f.write(("counter:{}\n").format(counter))
                      f.write(( "learn_rate:{:.6f},""loss:{}," "value_loss:{}," "policy_loss:{}," "own_loss:{},"
                        "now_play_games:{}," "actual_train_play_ratio:{}," "set_train_play_ratio:{},\n")
                        .format(info["learn_rate"], info["total_loss"], info["value_loss"], info["policy_loss"],info["own_loss"],
                         info["now_play_games"],actual_train_play_ratio, info["train_play_ratio"],))
                
                # 将损失 用tensorboard画出来
                info = ray.get(self.shared_storage_worker.get_info.remote(keys))
                writer.add_scalars('Train_loss', {'total_loss': info["total_loss"]}, counter)
                writer.add_scalars('Train_loss', {'value_loss': info["value_loss"]}, counter)
                writer.add_scalars('Train_loss', {'policy_loss': info["policy_loss"]}, counter)
                writer.add_scalars('Train_loss', {'own_loss': info["own_loss"]}, counter)
              
              # ------ 评估的线程也可以放到while 循环之前，像自对弈一样持续运行，（但是有可能会在存储数据到硬盘的时候报错，可能线程错误当时遇到的没细查） 
                if (counter+1) % self.evaluate_num == 0:

                    win_ratio, _, info3 =ray.get(self.evaluate_worker.policy_evaluate.remote(shared_storage_worker =self.shared_storage_worker))  #  启动评估的线程
                    with open(os.path.join(self.results_path, self.record_train), "a") as f:
                        f.write(info3)

                    pickle.dump(
                        ray.get(self.shared_storage_worker.get_info.remote(["weights","evaluate_weights"])),
                        open(os.path.join(self.results_path, "current_policy.model"), "wb")
                    )

                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                       
                        evaluate_score = ray.get(self.shared_storage_worker.get_info.remote("evaluate_score"))
                        pickle.dump(
                            ray.get(self.shared_storage_worker.get_info.remote(["weights","evaluate_weights"])),
                            open(os.path.join(self.results_path, "best_policy_{}.model".format(evaluate_score)), "wb")
                        )

                        if self.best_win_ratio == 1.0:
                            self.best_win_ratio = 0.0

                t = a % 3     # 按时间保存3份数据，新保存的数据不会覆盖上一份数据
                if (counter+1) % 15000 == 0:  # 数据存储
                    self.save_buffer(t)
                    a+=1 

                counter += 1
                time.sleep(1)

        except KeyboardInterrupt:
            pass
        self.terminate_workers()

    def terminate_workers(self):  # 程序终止

        print("\nShutting down workers...")
        self.self_play_workers = None
        self.training_worker = None
        self.mem = None
        self.shared_storage_worker = None

    def save_buffer(self,t):   # 保存数据
        start = time.time()
        print("\nPersisting replay buffer games to disk...")
        pickle.dump(
            ray.get(self.mem.save.remote()),
            open(os.path.join(self.results_path, "replay_buffer{}.pkl".format(t)), "wb")
        )
        print("Data saving completed.\n")
        end = time.time()
        print("run time:%.4fs" % (end - start))

    def load_buffer(self,replay_buffer_path=None):  # 加载数据
        if replay_buffer_path:
            if os.path.exists(replay_buffer_path):
                with open(replay_buffer_path, "rb") as f:
                    replay_buffer_infos = pickle.load(f)
                    print("load success")
                self.mem.load.remote(replay_buffer_infos)

        info = ray.get(self.mem.info.remote())
        if self.config.PER == True:
             print(info["capacity"],info["transitions_index"],info["full"],info["transitions_max"],)
        else:
             print(info["capacity"],info["index"],info["full"])

@ray.remote(num_cpus=0, num_gpus=0)
class CPUActor:
    # Trick to force DataParallel to stay on CPU to get weights on CPU even if there is a GPU
    def __init__(self):
        pass

    def get_initial_weights(self, config):
        model = AlphaZeroNetwork(config)
        weigths = model.get_weights()
        return weigths

if __name__ == '__main__':
    start = time.time()
    config = Config()
    training_pipeline = TrainPipeline(config)
    training_pipeline.train()

    end = time.time()
    print("run time:%.4fs" % (end - start))
