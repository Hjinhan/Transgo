# environment.py

from ctypes import *
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer

# 构造一个跟c语言一模一样的结构体来传输GoState
BOARD_SIZE = 9             # 棋盘尺寸 (要跟go_comm.h里的GoComm::BOARD_SIZE保持一致)
MAX_COORD = BOARD_SIZE * BOARD_SIZE     # 最大坐标数
MAX_BLOCK = 64             # 最大连通块数 (跟go_comm.h里的GoComm::MAX_BLOCK保持一致)
MAX_HISTORY_DIM = 1         # 最大历史棋盘数 (跟go_env.h里的GoState::MAX_HISTORY_DIM保持一致)

BLACK = 1
WHITE = 2

class c_Info(Structure):    # 棋盘格点信息, 见board.h的Info
    _fields_ = [('color', c_uint8), ('id', c_int16), ('next', c_int16), ('last_placed', c_uint16)]

class c_Block(Structure):   # 连通块, 见board.h的Block
    _fields_ = [('color', c_uint8), ('start', c_int16), ('num_stones', c_int16), ('liberties', c_int16)]

class c_Board(Structure):   # 棋盘, 见board.h的Board
    _fields_ = [('infos', c_Info * MAX_COORD), ('blocks', c_Block * MAX_BLOCK), ('next_player', c_int16),
                ('step_count', c_uint16), ('last_move1', c_int16), ('last_move2', c_int16), ('removed_block_ids', c_int*4),
                ('num_block_removed', c_int16), ('ko_location', c_int16), ('ko_color', c_uint8), ('ko_age', c_int16)]

class c_GoState(Structure): # 围棋状态, 见go_env.h的GoState
    _fields_ = [('_boards', c_Board * MAX_HISTORY_DIM), ('_terminate', c_bool)]


class GoEnv:
    def __init__(self):
        self.history_dim = 1    # 每个状态的历史棋盘数N(含当前棋盘)
        self.encoded_dim = 10   # 编码的特征平面数M   需要跟configure里面的对应
        self.max_step = 74    # 最大步数    
        self.komi = 6.5         # 贴目
        self.board_size = 9
        self.local_board_size = 7

        self.lib = ctypes.cdll.LoadLibrary("./GoEnv/go_env.so")       # 加载动态库
        self.c_init = self.lib.Init
        self.c_init.argtypes = [c_int, c_int ,c_int, c_float]   # 初始化动态库
        self.c_init(self.history_dim, self.encoded_dim, self.max_step, self.komi)

        self.c_reset = self.lib.Reset    # 重置状态
        self.c_reset.argtypes = [POINTER(c_GoState)]

        self.c_step = self.lib.Step      # 下一步(创建新状态), 返回棋局是否结束
        self.c_step.argtypes = [POINTER(c_GoState), POINTER(c_GoState), c_int]
        self.c_step.restype = c_bool

        self.c_checkAction = self.lib.checkAction    # 检查动作合法性
        self.c_checkAction.argtypes = [POINTER(c_GoState), c_int]
        self.c_checkAction.restype = c_bool

        self.c_encode = self.lib.Encode      # 编码特征平面    # 己方1,2,3及以上气连通块, 对方1,2,3及以上气连通块, 上一个历史落子点, 非法落子, 己方真眼, 己方活棋块
        self.c_encode.argtypes = [POINTER(c_GoState), ndpointer(c_float)]

        self.c_getScore = self.lib.getScore  # 获取盘面差
        self.c_getScore.argtypes = [POINTER(c_GoState)]
        self.c_getScore.restype = c_float

        self.c_getTerritory = self.lib.getTerritory  # 获取盘面差及归属, 返回值是盘面差
        self.c_getTerritory.argtypes = [POINTER(c_GoState), ndpointer(c_float)]
        self.c_getTerritory.restype = c_float

        self.c_getLegalAction = self.lib.getLegalAction  # 获取合法动作集, 返回值是动作数
        self.c_getLegalAction.argtypes = [POINTER(c_GoState), ndpointer(c_int)]
        self.c_getLegalAction.restype = c_int

        self.c_getLegalNoEye = self.lib.getLegalNoEye    # 获取合法动作集(不含己方真眼), 返回值是动作数    # 真眼的定义见board.cc中的isTrueEye()函数
        self.c_getLegalNoEye.argtypes = [POINTER(c_GoState), ndpointer(c_int)]
        self.c_getLegalNoEye.restype = c_int

        self.c_show = self.lib.Show      # 打印棋盘及其他信息
        self.c_show.argtypes = [POINTER(c_GoState)]

        self.c_getPlayer = self.lib.getPlayer  # 获取下一个玩家
        self.c_getPlayer.argtypes = [POINTER(c_GoState)]
        self.c_getPlayer.restype = c_int

        self.c_getStep = self.lib.getStep   # 获取步数
        self.c_getStep.argtypes = [POINTER(c_GoState)]
        self.c_getStep.restype = c_int
    
    def reset(self):            # 重置状态(创建新状态), 返回值同step
        new_state = c_GoState()
        self.c_reset(new_state)
        done = False
        return new_state , done

    def step(self, state, action):  # 下一步(创建新状态), 返回新状态
        new_state = c_GoState()
        done = self.c_step(state, new_state, action)
        # print("done:", done)
        # print("terminate :",state._terminate)
        return new_state , done
    
    def localEncode(self, state):        # 编码特征平面, (M==10时)己方1,2,3及以上气连通块, 对方1,2,3及以上气连通块, 上一个历史落子点, 非法落子, 己方真眼, 己方活棋块
        encode_state = np.zeros([self.history_dim * self.encoded_dim, BOARD_SIZE, BOARD_SIZE], dtype="float32")
        self.c_encode(state, encode_state)
        
        local_encode_state = encode_state[:,0:self.local_board_size,0:self.local_board_size]

        return local_encode_state
    
    def encode(self, state):        
        encode_state = np.zeros([self.history_dim * self.encoded_dim, BOARD_SIZE, BOARD_SIZE], dtype="float32")
        self.c_encode(state, encode_state)
        return encode_state


    def getScore(self, state):      # 返回盘面差
        return self.c_getScore(state)
    
    def getWinner(self, state):
         return BLACK if self.getScore(state) > 0 else WHITE

   
    def getLegalAction(self, state):    # 返回合法动作, 数组长度等于动作数
        legal_action = np.zeros([BOARD_SIZE * BOARD_SIZE + 1], dtype='int32')
        num_action = self.c_getLegalAction(state, legal_action)
        legal_overall_acts = legal_action[:num_action]
        
        local_legal_acts = self.getLocalAction(legal_overall_acts)
        return local_legal_acts

    def getLocalAction(self,overall_legal_acts):
     
        leal_local_acts = []

        for act in overall_legal_acts:

            if act != self.board_size**2:
                row,col = self.action_to_location(act)
                if  0 <= row < self.local_board_size and  0 <= col < self.local_board_size :
                        act = self.location_to_local_action([row,col])
                        leal_local_acts.append(act)
              
        if len(leal_local_acts) == 0:
                leal_local_acts.append(self.local_board_size**2)
     
        return leal_local_acts                      # 动作集 对应 整个 local_board_size ：  0 ~ local_board_size**2

    def get_random_init_action(self,state):               # 边界外一格 随机填子
            legal_action = np.zeros([BOARD_SIZE * BOARD_SIZE + 1], dtype='int32')
            num_action = self.c_getLegalAction(state, legal_action)
            legal_overall_acts = legal_action[:num_action]
            
            random_init_action = []
            for act in legal_overall_acts :           
                row,col = self.action_to_location(act)
                if (row == self.local_board_size  and 0 <= col <= self.local_board_size) or \
                              (col == self.local_board_size  and 0 <= row <= self.local_board_size):
                        random_init_action.append( act )  
            random_act = np.random.choice(random_init_action)
            return random_act

    def getPlayer(self, state):     # 获取玩家(1:黑, 2:白)
        return self.c_getPlayer(state)

    def action_to_location(self,action):
        row = action // self.board_size
        col = action % self.board_size
        return [row,col]


    def location_to_local_action(self,location):
        row = location[0] 
        col = location[1]
        return self.local_board_size * row +col


    def act_to_local_act(self,act):

        if act < self.board_size**2 :
            row,col = self.action_to_location(act)
            local_act =  self.local_board_size * row +col
            if local_act < self.local_board_size **2:
                return local_act
            else:
                raise NotImplementedError("illegal local action 1.")
    
        elif act == self.board_size**2 :
            return self.local_board_size**2
        else:
            raise NotImplementedError("illegal action 1.")


    def local_act_to_act(self, local_act):

        if local_act < self.local_board_size**2:
            row = local_act // self.local_board_size
            col = local_act % self.local_board_size
            act =  self.board_size * row + col
            if act < self.board_size **2:
                return act
            else:
                raise NotImplementedError("illegal action 2.")

        elif local_act == self.local_board_size**2 : 
            return self.board_size**2
        else:
            raise NotImplementedError("illegal local action 2.")

    def board_grid(self,state):
         encode = self.encode(state)
         grid = encode[:6]
         grid = np.sum(grid,0)
         print("grid2:\n",grid)
         return grid

    def getStep(self, state):       # 获取步数
        return self.c_getStep(state)

    def getScoreAndTerritory(self, state):  # 返回盘面差, 和19x19的盘面归属(黑1.0, 中立0.5, 白0.0)
        territory = np.zeros([BOARD_SIZE, BOARD_SIZE], dtype="float32")
        score = self.c_getTerritory(state, territory)

        local_territory = territory[0:self.local_board_size, 0:self.local_board_size]  #!
        return score, local_territory.reshape(-1) 
        
#---------------------------------------------------------------------------------------------------------------
 #下面的函数暂时没用到

    def checkAction(self, state, action):   # 检查动作合法性(非必须, step本身也会检查), 返回True/False
        return self.c_checkAction(state, action)
  
    def getLegalNoEye(self, state):     # 返回候选动作(不含己方真眼的合法动作), 数组长度等于动作数    # 真眼的定义见board.cc中的isTrueEye()函数
        candidate_action = np.zeros([BOARD_SIZE * BOARD_SIZE + 1], dtype='int32')
        num_action = self.c_getLegalNoEye(state, candidate_action)
        return candidate_action[:num_action]
    
    def show(self, state):     # 打印棋盘及其他信息
        self.c_show(state)
    
    def justStarted(self, state):   # 棋局是否刚开始
        return self.c_getStep(state) == 1


