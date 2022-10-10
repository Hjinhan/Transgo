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

from model import TransGoNetwork
from self_play import SelfPlay
from configure import Config
from trainer import Trainer
from replay_buffer import ReplayMemory
import shared_storage

from tensorboardX import SummaryWriter
writer = SummaryWriter('runs1/loss_1')
  
ray.init()

class TrainPipeline(object):
    def __init__(self,config):

        self.config = config
        self.now_train_steps = config.load_train_steps or 0 # 训练迭代次数
        self.now_play_steps = config.load_play_steps or 0  # 对弈回合数
        self.now_play_games = config.load_play_games or 0  # 对弈局数
        self.best_win_ratio = 0.0 # 评估模型时的最好胜率（胜率为1时evaluate_score得分会+100，并且best_win_ratio会重置为0）
        
        self.game_total_num = config.game_total_num
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
            "game_total_num": self.game_total_num,
            "evaluate_score": self.init_evaluate_score,

            "total_loss": 0.,
            "value_loss": 0.,
            "act_policy_loss": 0.,
            "entropy_loss": 0.,
            "own_loss": 0.,

        }

    def train(self):

        self.num_gpus = torch.cuda.device_count()  
        print(self.num_gpus) 

        if self.config.init_model:
            if os.path.exists( self.config.init_model):
                with open( self.config.init_model, "rb") as f:
                    model_weights = pickle.load(f)
                    self.checkpoint["weights"] = model_weights["weights"]
                    self.checkpoint["evaluate_weights"] = model_weights["evaluate_weights"]
                    print(" load model success...\n")
        else:
            cpu_actor = CPUActor.remote()
            cpu_weights = cpu_actor.get_initial_weights.remote(self.config)
            self.checkpoint["weights"] = copy.deepcopy(ray.get(cpu_weights))
            self.checkpoint["evaluate_weights"] = copy.deepcopy(ray.get(cpu_weights))
  
        #! 训练及自对弈的gpu需在下面根据需要调整
        # inint workers
        self.shared_storage_worker = shared_storage.SharedStorage.remote(self.checkpoint,self.config)
        self.mem = ReplayMemory(self.config)

        if  self.config.init_buffer:
            self.load_buffer(replay_buffer_paths= self.config.init_buffer)

        self.self_play_workers = [                      # 自对弈并行num_workers局
                    SelfPlay.options( num_gpus=1/(self.config.play_workers_num+2),).     # 
                    remote(self.config)
                   for _ in range(self.config.play_workers_num)
                ]
     
        self.training_worker = Trainer.options(num_gpus=1/(self.config.play_workers_num+2), ). \
            remote(self.checkpoint,self.config,  self.config.init_model)

        self.evaluate_worker = SelfPlay.options(num_gpus= 1/(self.config.play_workers_num+2),).remote(self.config)

        # Launch workers
        [
            self_play_worker.continuous_self_play.remote(
                self.shared_storage_worker, self.mem
            )
            for self_play_worker in self.self_play_workers
        ]

        self.training_worker.train_step.remote(self.mem, self.shared_storage_worker)
        self.logging_loop()

    def logging_loop(self):

        counter = 0
        clock = 0
        check = 0

        keys = [
            "now_train_steps",
            "now_play_steps",
            "now_play_games",
            "train_play_ratio",
            "learn_rate",

            "total_loss",
            "value_loss",
            "act_policy_loss",
            "entropy_loss",
            "own_loss",

        ]
        info = ray.get(self.shared_storage_worker.get_info.remote(keys))
        try:
            while info["now_play_games"] < self.game_total_num:

                if (counter) % 15 == 0:
                    actual_train_play_ratio = 0
                    if info["now_play_steps"]>1:
                        actual_train_play_ratio = info["now_train_steps"] / info["now_play_steps"]
                    
                    print("counter:", counter)
                    print("learn_rate:",info["learn_rate"],", total_loss:",info["total_loss"], ", value_loss:",info["value_loss"],",act_policy_loss:", 
                        info["act_policy_loss"],",entropy_loss:", info["entropy_loss"], ", own_loss:",info["own_loss"],
                        ", now_train_steps:",info["now_train_steps"],", play_steps_num:",info["now_play_steps"],
                        ", actual_train_play_ratio:",actual_train_play_ratio,", set_train_play_ratio:",info["train_play_ratio"],
                        ", play_games_num:", info["now_play_games"], )

                    with open(os.path.join(self.results_path, self.record_train), "a") as f:
                      f.write(("counter:{}\n").format(counter))
                      f.write(( "learn_rate:{:.6f},""loss:{}," "value_loss:{}," "act_policy_loss:{}," "entropy_loss:{}," 
                        "own_loss:{}," "now_play_games:{}," "actual_train_play_ratio:{}," "set_train_play_ratio:{},\n")
                        .format(info["learn_rate"], info["total_loss"], info["value_loss"], info["act_policy_loss"], info["entropy_loss"], 
                        info["own_loss"], info["now_play_games"],actual_train_play_ratio, info["train_play_ratio"],))

                info = ray.get(self.shared_storage_worker.get_info.remote(keys))
                writer.add_scalars('Train_loss', {'total_loss': info["total_loss"]}, counter)
                writer.add_scalars('Train_loss', {'value_loss': info["value_loss"]}, counter)
                writer.add_scalars('Train_loss', {'act_policy_loss': info["act_policy_loss"]}, counter)
                writer.add_scalars('Train_loss', {'entropy_loss': info["entropy_loss"]}, counter)
                writer.add_scalars('Train_loss', {'own_loss': info["own_loss"]}, counter)

              
                if (counter+1) % self.evaluate_num == 0:

                    win_ratio, _, info3 =ray.get(self.evaluate_worker.policy_evaluate.remote(shared_storage_worker =self.shared_storage_worker))
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

                # 分段存储  
                info_men = copy.deepcopy(ray.get(self.mem.info.remote()))    
                t = clock % self.config.store_batch     # 按时间保存5份数据，新保存的数据不会覆盖上一份数据
                if self.config.buffer_size > 1500000 and info_men["index"] > 10000 \
                                                     and info_men["index"] % (1000000-10000) < 10000 \
                                                     and check != info_men["index"] // (1000000-10000)  \
                                                     and self.config.is_save_buffer: # 数据存储 
                    
                    print("now is save buffer ..... \n")
                    with open(os.path.join(self.results_path, self.record_train), "a") as f:
                       f.write("\n Persisting replay buffer games to disk... \n")

                    self.save_buffer(t)
                    clock += 1
                    check = info_men["index"] // (1000000-10000)
                
                # 一次全部存储
                elif self.config.buffer_size <= 1500000 and (counter+1) % 15000 == 0:
                        self.save_buffer(t)
                        clock += 1

                counter += 1
                time.sleep(1)

        except KeyboardInterrupt:
            pass
        self.terminate_workers()

    def terminate_workers(self):

        print("\nShutting down workers...")
        self.self_play_workers = None
        self.training_worker = None
        self.mem = None
        self.shared_storage_worker = None

    def save_buffer(self,t):   # 保存数据
        start = time.time()
        print("\nPersisting replay buffer games to disk...")
        save_buffer = ray.get(self.mem.save.remote())
        if save_buffer :
            pickle.dump(save_buffer,
                open(os.path.join(self.results_path, "replay_buffer{}.pkl".format(t)), "wb")
            )
            print("Data saving completed.\n")
        end = time.time()
        print("run time:%.4fs" % (end - start))

    def load_buffer(self,replay_buffer_paths=None):  # 加载数据
    
        for i , path in enumerate(replay_buffer_paths) :
            if path:
                if os.path.exists(path):
                    with open(path, "rb") as f:

                        replay_buffer_infos = pickle.load(f)
                        print("load success {}".format(i))
                        is_full = self.mem.load.remote(replay_buffer_infos)
                        if ray.get(is_full):
                            break
        info = ray.get(self.mem.info.remote())
        print(info["capacity"], info["index"], info["full"])
            
@ray.remote(num_cpus=0, num_gpus=0)
class CPUActor:
    # Trick to force DataParallel to stay on CPU to get weights on CPU even if there is a GPU
    def __init__(self):
        pass

    def get_initial_weights(self, config):
        model = TransGoNetwork(config)
        weigths = model.get_weights()
        return weigths

if __name__ == '__main__':
    start = time.time()
    config = Config()
    training_pipeline = TrainPipeline(config)
  
    training_pipeline.train()

    end = time.time()
    print("run time:%.4fs" % (end - start))
