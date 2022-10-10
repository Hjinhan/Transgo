import torch
import math

  
class Config:
    def __init__(self):     #  不同gpu跟cpu性能， 要调节的参数  1、batchsize ；2、play_workers_num ； 3、train_play_ratio
          
    #----------环境相应参数------------
        self.board_size = 9
        self.encode_state_channels = 10  # 己方1,2,3及以上气连通块, 对方1,2,3及以上气连通块,上一个历史落子点,非法落子,己方真眼,己方活棋块
                                         # encode_state_channels 有三种可选； 9个特征；10个特征；后面加了一个13特征
        self.komi = 6.5    
        self.tanh_norm = 0.45
        self.black = 1
        self.white = 2
   
    #---------局部自对弈参数---------------
        self.local_board_size = 7
        # self.num_init_action = 8    # 开局前随机落8个子
        self.random_play_init = True
  
    #----------储存区参数---------------
        self.buffer_size = 1500000
        self.priority_exponent =0.45   # PER参数，这个参数要根据经验调合适
        self.PER = False      # 是否启用PER ， 默认不用PER    

    #----------自对弈参数----------------
        self.game_batch_num = 1500000   # 总对弈局数
        self.play_workers_num = 8
        self.c_puct = 3             # 3 是比较中性的值；  9x9 用1.5训练效率不如用3
                                    #根据经验 ，实际越容易到达终局，puct需要越大。  19x19 需要调小一点； 但是像五子棋会很快就直接终局 需要调大一点
        self.num_simulation= 120

    #----------网络模型及训练参数---------
        self.input_dim = self.encode_state_channels
        self.num_features = 128
        self.l2_const = 1e-4
        self.checkpoint_interval = 3  # 每训练3次更新一次自对弈模型
        self.adjust_lr = True
        self.learn_rate = 6.5e-5   #! 初始学习率
        self.batch_size = 1024

    #----------评估------------------
        self.init_evaluate_score = 100   #! 初始评估分
        self.evaluate_num = 1500   # 间隔 1500 个计数 评估一次  // 
        
    #----------其他----------------------
        self.train_play_ratio = 2500/100000  # 初始  #! 初始 训练/自对弈比率
        self.adjust_train_play_ratio = True    # f
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.results_path = "./results1"   
        self.record_train = "train_record.txt"

     #---------加载模型----------

        self.init_model = False
        self.init_buffer = False
        # self.init_model = "./results1/best_policy_600.model"
        # self.init_buffer = "./results1/replay_buffer0.pkl"

    #----------控制温度系数----------------------
    def epsilon_by_frame(self, game_step):  #   温度系数从1 衰减到 0.65
        epsilon_start = 1.0
        epsilon_final = 0.65
        epsilon_decay = 10
        return epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * game_step / epsilon_decay)

 
    #----------根据对弈局数动态控制学习率----------------------
    def ad_lr(self,now_play_games,current_lr):
         if (now_play_games+1)% 1500 ==0 and now_play_games< 3100 and current_lr> 0.5*0.5*6.5e-5 :
                return current_lr*0.5
         return current_lr

   
    #----------动态控制训练/自对弈比率----------------------  自对弈太快，训练迭代次数太少，会使模型欠拟合；相反，就容易过拟合（无法兼顾宁愿过拟合）
    def ad_train_play_ratio(self, now_play_steps, current_train_play_ratio ):   # 调 5跟4.6这个参数 ，主要调4.6

        if (now_play_steps+1)% 45==0 and current_train_play_ratio< 5.1/10 :   
            train_play_ratio_ = (current_train_play_ratio*100000 +1)/100000
            return train_play_ratio_
        return current_train_play_ratio