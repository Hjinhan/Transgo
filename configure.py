import torch
import math

    
class Config:
    def __init__(self):     #  不同gpu跟cpu性能， 要调节的参数  1、batchsize ；2、play_workers_num ； 3、train_play_ratio
                
    #----------环境相应参数------------
        self.board_size = 9
        self.local_board_size = 7
        self.encode_state_channels = 10  # 己方1,2,3及以上气连通块, 对方1,2,3及以上气连通块,上一个历史落子点,非法落子,己方真眼,己方活棋块
                                         # encode_state_channels 有三种可选； 9个特征；10个特征；13特征
        self.komi = 7.5                  
        self.black = 1
        self.white = 2
        self.max_step = 120    # 最大步数    
    
    #----------储存区参数---------------
        self.buffer_size = 1500000   
        self.is_save_buffer = True   # 
        self.store_batch = 5         # 按照时间 保存5份数据
      
    #----------自对弈参数----------------
        self.game_total_num = 1e8   # 总对弈局数
        self.play_workers_num = 6
        self.c_puct1 = 3      
        self.c_puct2 = 0.05           
                                     
        self.num_simulation= 210
        self.tanh_norm = 0.55   # 低尺度模型计算value2  时的 tanh_norm
  
        self.wu_loss = 2      # 
        self.parallel_readouts = 4  # 树并行数量

    #----------网络模型及训练参数---------
        self.input_dim = self.encode_state_channels
        self.num_features = 128
        self.l2_const = 1e-4
        self.checkpoint_interval = 3  # 每训练3次更新一次自对弈模型
        self.adjust_lr = True
        self.learn_rate = 6.5e-5   # 初始学习率
        self.batch_size = 2048

    #----------评估------------------
        self.init_evaluate_score = 100   # 初始评估分
        self.evaluate_num = 1500   # 间隔 1500 个计数 评估一次  // 
 
    #---------加载模型----------   # 除了以下参数， 还要重新调整的参数有 ： learn_rate ， init_evaluate_score ，train_play_ratio
  
        self.init_model = False
        self.init_buffer = False
        self.init_sub_model = False

        # 加载的时候，需要和之前训练的情况对应，保持 训练/自对弈比率
        self.load_train_steps = None # 训练迭代次数
        self.load_play_steps = None  # 对弈回合数
        self.load_play_games = None  # 对弈局数

        # self.init_model = "./results1/best_policy_600.model"
        # self.init_buffer = ["./results1/replay_buffer0.pkl", "./results1/replay_buffer1.pkl","./results1/replay_buffer1.pkl"]  # 列表 分段加载


    #----------其他----------------------
        self.train_play_ratio = 7500/100000  # 初始 训练/自对弈比率
        self.adjust_train_play_ratio = True    # f
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.results_path = "./results1"   
        self.record_train = "train_record.txt"
  
        # self.init_model = "./results1/best_policy_600.model"
        # self.init_buffer = ["./results1/replay_buffer0.pkl", "./results1/replay_buffer1.pkl","./results1/replay_buffer1.pkl"]  # 列表 分段加载

    #----------控制温度系数----------------------
    def epsilon_by_frame(self, game_step):  #   温度系数从1 衰减到 0.65
        epsilon_start = 1.0
        epsilon_final = 0.65
        epsilon_decay = 10
        return epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * game_step / epsilon_decay)

    #----------控制低尺度迁移----------------------
    def value_ratio_by_frame(self,now_train_steps):
        value_ratio_start = 1.0
        value_ratio_final = 0.0
        value_ratio_decay = 1200
        return value_ratio_final + (value_ratio_start - value_ratio_final) * math.exp(-1. * now_train_steps / value_ratio_decay)
  

    #----------根据对弈局数动态控制学习率----------------------
    def ad_lr(self,now_play_games,current_lr):
         if (now_play_games+1)% 1500 ==0 and now_play_games< 3100 and current_lr> 0.5*0.5*6.5e-5 :
                return current_lr*0.5
         return current_lr


    #----------动态控制训练/自对弈比率----------------------  自对弈太快，训练迭代次数太少，会使模型欠拟合；相反，就容易过拟合（无法兼顾宁愿过拟合）
    def ad_train_play_ratio(self, now_play_steps, current_train_play_ratio ):   

        if (now_play_steps+1)% 6==0 and current_train_play_ratio< 2.6/10 :   
            # batchsize为 1024 时，用4.6左右；      batchsize为 2048，2.3左右，学习率可以稍微调高一点
            train_play_ratio_ = (current_train_play_ratio*100000 +1)/100000
            return train_play_ratio_
        return current_train_play_ratio

