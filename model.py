import math

import torch
import torch.nn.functional as F
  
import torch
import torch.nn as nn
import torch.nn.functional as F

 
class TransGoNetwork(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.main_network = MainNetwork(config.board_size, config.board_size, config.input_dim, config.num_features)

    def main_prediction(self,state):  # self_play
                                                                                 
         act_policy,  value, own  = self.main_network(state)  #  
         return act_policy, value, own 
 
  
    def get_weights(self):
        return dict_to_cpu(self.state_dict())
    
    def set_weights(self, weights):
        self.load_state_dict(weights)

def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict

   
class MainNetwork(nn.Module):
      
    def __init__(self, board_width, board_height, input_dim, num_features):
        super(MainNetwork, self).__init__()                      

        self.board_width = board_width
        self.board_height = board_height

        self.conv1 = CNNBlock(input_dim, num_features)
        self.res_conv2 = ResidualBlock(num_features, num_features)
        self.res_conv3 = Self_Attention(num_features)
        self.res_conv4 = ResidualBlock(num_features, num_features)
        self.res_conv5 = ResidualBlock(num_features, num_features)
        self.res_conv6 = ResidualBlock(num_features, num_features)
        self.res_conv7 = Self_Attention(num_features)
        self.res_conv8 = ResidualBlock(num_features, num_features)
        self.res_conv9 = ResidualBlock(num_features, num_features)
        self.res_conv10 = ResidualBlock(num_features, num_features)
        self.res_conv11 = ResidualBlock(num_features, num_features)
        self.res_conv12 = Self_Attention(num_features)
        self.res_conv13 = ResidualBlock(num_features, num_features)
        self.bn_res_end = nn.BatchNorm2d(num_features)

        #---------------value_head----------------------------
        self.conv_val_own = CNNBlock(num_features, 2)
        self.fc_val_own = nn.Linear(2 * board_width * board_height, 64)

        self.fc_val = nn.Linear(64, 1)           # 预测价值
        self.fc_own = nn.Linear(64, board_width * board_height)   # 预测归属
  
        # ------------- policy_head -------------------------
        self.attention_act = Self_Attention(num_features)   
        self.conv_act = CNNBlock(num_features, 4)

        self.fc_act = nn.Linear(4*board_width*board_height,          # 预测策略动作概率
                                 board_width*board_height+1)
        

    def forward(self, state_input): 

        x = self.conv1(state_input)
        x = self.res_conv2(x)
        x = self.res_conv3(x)
        x = self.res_conv4(x)
        x = self.res_conv5(x)
        x = self.res_conv6(x)
        x = self.res_conv7(x)
        x = self.res_conv8(x)
        x = self.res_conv9(x)
        x = self.res_conv10(x)
        x = self.res_conv11(x)
        x = self.res_conv12(x)
        x = self.res_conv13(x)
        x = F.relu(self.bn_res_end(x))
  
        #---------------value_head----------------------------
        x_val_own = self.conv_val_own(x)
        x_val_own = x_val_own.view(-1, 2 * self.board_width * self.board_height)
        x_val_own = F.relu(self.fc_val_own(x_val_own))

        x_val = torch.tanh(self.fc_val(x_val_own))   #价值
        x_own = torch.tanh(self.fc_own(x_val_own))   #归属
      
        # ------------- policy_head -------------------------
  
        x_act_com = self.attention_act(x)   # attention
        x_act_com = self.conv_act(x_act_com)
        x_act_com = x_act_com.view(-1, 4*self.board_width*self.board_height)

        x_act = self.fc_act(x_act_com)       # 己方策略
        x_act =torch.softmax(x_act,-1)  
  

        return x_act,  x_val, x_own

#-----------------------------------------------------------------------------------------

# 小尺度棋盘模块
class MutativeNetwork(torch.nn.Module):  # 其他模块主要会调用这个函数
    def __init__(self, config):
        super().__init__()
        self.network = torch.nn.DataParallel(MiniNetwork(config.local_board_size, config.local_board_size, config.input_dim,config.num_features))
    
    def main_prediction(self,state):

         act_policy, value, own = self.network(state)
         return act_policy, value, own

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights)


def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict


class MiniNetwork(nn.Module):   

    def __init__(self, board_width, board_height, input_dim, num_features):
        super(Network, self).__init__()

        self.board_width = board_width
        self.board_height = board_height

        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=num_features, kernel_size=3, stride=1 , padding=1)
        self.batchnormlize_1 = nn.BatchNorm2d(num_features)
    
        self.res_conv2 = ResidualBlock(num_features, num_features)
        self.res_conv3 = ResidualBlock(num_features, num_features)
        self.res_conv4 = ResidualBlock(num_features, num_features)
        self.res_conv5 = ResidualBlock(num_features, num_features)
        self.batchnormlize_2 = nn.BatchNorm2d(num_features)

        self.res_act = nn.Conv2d(in_channels=num_features, out_channels=4, kernel_size=3, stride=1 , padding=1)
        self.batchnormlize_3 = nn.BatchNorm2d(4)
        self.act_fc1 = nn.Linear(4*board_width*board_height,   
                                 board_width*board_height+1)

        self.res_val_own = ResidualBlock(num_features, 4)
        self.batchnormlize_4 = nn.BatchNorm2d(4)
        self.val_own_fc1 = nn.Linear(4 * board_width * board_height, 64)

        self.val_fc1 = nn.Linear(64, 1)
        self.own_fc1 = nn.Linear(64, board_width * board_height)


    def forward(self, state_input):

        x = self.conv1(state_input)
        x = F.relu(self.batchnormlize_1(x))

        x = self.res_conv2(x)
        x = self.res_conv3(x)
        x = self.res_conv4(x)
        x = self.res_conv5(x)
        x = self.batchnormlize_2(x)
 
        x_act = self.res_act(x)
        x_act = F.relu(self.batchnormlize_3(x_act))
        x_act = x_act.view(-1, 4*self.board_width*self.board_height)
        x_act = self.act_fc1(x_act)
        x_act = torch.softmax(x_act,-1)

        x_val_own = self.res_val_own(x)
        x_val_own = F.relu(self.batchnormlize_4(x_val_own))
        x_val_own = x_val_own.view(-1, 4 * self.board_width * self.board_height)
        x_val_own = F.relu(self.val_own_fc1(x_val_own))

        x_val = torch.tanh(self.val_fc1(x_val_own))
        x_own = torch.tanh(self.own_fc1(x_val_own))

        return x_act, x_val, x_own

#----------------------------------------------------------------------------------------------
# other   utils

class ResidualBlock(nn.Module):          # 残差模块  resnetV2
    def __init__(self, input_dim, output_dim, resample=None):
        super(ResidualBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample
        self.batchnormlize_1 = nn.BatchNorm2d(input_dim)

        if resample == 'down':
            self.conv_0 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv_shortcut = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=1, stride=1)
            self.conv_1 = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1)
            self.conv_2 = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1)
            self.conv_3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.batchnormlize_2 = nn.BatchNorm2d(input_dim)

        elif resample == 'up':
            self.conv_0 = nn.Upsample(scale_factor=2)
            self.conv_shortcut = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=1, stride=1)
            self.conv_1 = nn.Upsample(scale_factor=2)
            self.conv_2 = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1)
            self.conv_3 = nn.Conv2d(in_channels=output_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1)
            self.batchnormlize_2 = nn.BatchNorm2d(output_dim)

        elif resample == None:
            self.conv_shortcut = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=1, stride=1)
            self.conv_1 = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1)
            self.conv_2 = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1)
            self.batchnormlize_2 = nn.BatchNorm2d(input_dim)

    def forward(self, inputs):
        if self.output_dim == self.input_dim and self.resample == None:
            shortcut = inputs
            x = inputs
            x = self.batchnormlize_1(x)
            x = F.relu(x)
            x = self.conv_1(x)
            x = self.batchnormlize_2(x)
            x = F.relu(x)
            x = self.conv_2(x)
            return shortcut + x

        elif self.resample is None:
            y = inputs
            shortcut = self.conv_shortcut(y)
            x = inputs
            x = self.batchnormlize_1(x)
            x = F.relu(x)
            x = self.conv_1(x)
            x = self.batchnormlize_2(x)
            x = F.relu(x)
            x = self.conv_2(x)
            return shortcut + x

        elif self.resample == 'down':
            y = self.conv_0(inputs)
            shortcut = self.conv_shortcut(y)
            x = inputs
            x = self.batchnormlize_1(x)
            x = F.relu(x)
            x = self.conv_1(x)
            x = self.batchnormlize_2(x)
            x = F.relu(x)
            x = self.conv_2(x)
            x = self.conv_3(x)
            return shortcut + x

        else:
            y = self.conv_0(inputs)
            shortcut = self.conv_shortcut(y)
            x = inputs
            x = self.batchnormlize_1(x)
            x = F.relu(x)
            x = self.conv_1(x)
            x = self.conv_2(x)
            x = self.batchnormlize_2(x)
            x = F.relu(x)
            x = self.conv_3(x)
            return shortcut + x

class Self_Attention(nn.Module):

    def __init__(self, in_dim):
        super(Self_Attention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim //4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim //4 , kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.bn = nn.BatchNorm2d(self.chanel_in)  # 后加

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X (W*H) X C
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # B X (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention)
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        out = F.relu(self.bn(out)) # 后加

        return out

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
                                 nn.Conv2d(in_channels, out_channels, kernel_size=3, stride = 1, padding=1),
                                 nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True),)
    def forward(self, x):
        return self.conv(x)


class Self_Attention_Fusion(nn.Module):

    def __init__(self, in_dim):
        super(Self_Attention_Fusion, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim //4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim //4 , kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_query,x_key,x_value):
        m_batchsize, C, width, height = x_value.size()
        proj_query = self.query_conv(x_query).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X (W*H) X C
        proj_key = self.key_conv(x_key).view(m_batchsize, -1, width * height)  # B X C x (W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # B X (N) X (N)
        proj_value = self.value_conv(x_value).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention)
        out = out.view(m_batchsize, C, width, height)

        return out

class GAPBlock(nn.Module):  # 替换全连接层
    def __init__(self, in_channels, out_units):
        super(GAPBlock, self).__init__()
        self.conv = nn.Sequential(
                                 nn.Conv2d(in_channels, out_units, kernel_size=3, stride = 1, padding=1),
                                 nn.AdaptiveAvgPool2d(1),)
    def forward(self, x):
        x = self.conv(x).flatten(1)
        return x

class NoisyLinear(nn.Module):  # 增加探索
  def __init__(self, in_features, out_features, std_init=0.5):
    super(NoisyLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.std_init = std_init

    self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
    self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))    # register_buffer 是不进行参数更新 的变量

    self.bias_mu = nn.Parameter(torch.empty(out_features))
    self.bias_sigma = nn.Parameter(torch.empty(out_features))
    self.register_buffer('bias_epsilon', torch.empty(out_features))
    self.reset_parameters()
    self.reset_noise()

  def reset_parameters(self):   # 模型参数的初始化
    mu_range = 1 / math.sqrt(self.in_features)
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))    # 表示用 self.std_init / math.sqrt(self.in_features) 填充 self.weight_sigma
    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

  def _scale_noise(self, size):
    x = torch.randn(size, device=self.weight_mu.device)
    return x.sign().mul_(x.abs().sqrt_())

  def reset_noise(self):
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)           # ger：epsilon_in 行 ， epsilon_out列
    self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))    
    self.bias_epsilon.copy_(epsilon_out)                        #  torch.ger是对tensor进行扩维，torch.ger(a,b)实际意思是b中的每一个元素乘以a中的元素，进行扩维

  def forward(self, input):
    if self.training:
      return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
    else:
      return F.linear(input, self.weight_mu, self.bias_mu)


# ------------------------------------------------------------
# alphazero

class AlphaZeroNetwork(torch.nn.Module):  # 其他模块主要会调用这个函数
    def __init__(self, config):
        super().__init__()
        self.network = torch.nn.DataParallel(Network(config.board_size, config.board_size, config.input_dim,config.num_features))

    def main_prediction(self,state):

         act_policy, value = self.network(state)
         return act_policy, value

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights)



class Network(nn.Module):   # alphazero 网络    ，如果gpu显存太小，可以注释掉一些层

    def __init__(self, board_width, board_height, input_dim, num_features):
        super(Network, self).__init__()

        self.board_width = board_width
        self.board_height = board_height

        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=num_features, kernel_size=3, stride=1 , padding=1)
        self.batchnormlize_1 = nn.BatchNorm2d(num_features)
    
        self.res_conv2 = ResidualBlock(num_features, num_features)
        self.res_conv3 = Self_Attention(num_features)
        self.res_conv4 = ResidualBlock(num_features, num_features)
        self.res_conv5 = ResidualBlock(num_features, num_features)
        self.res_conv6 = ResidualBlock(num_features, num_features)
        self.res_conv7 = Self_Attention(num_features)
        self.res_conv8 = ResidualBlock(num_features, num_features)
        self.res_conv9 = ResidualBlock(num_features, num_features)
        self.res_conv10 = Self_Attention(num_features)
        self.res_conv11 = ResidualBlock(num_features, num_features)
        self.res_conv12 = ResidualBlock(num_features, num_features)
        self.batchnormlize_2 = nn.BatchNorm2d(num_features)
  
        self.res_act = nn.Conv2d(in_channels=num_features, out_channels=4, kernel_size=3, stride=1 , padding=1)
        self.batchnormlize_3 = nn.BatchNorm2d(4)
        self.act_fc1 = nn.Linear(4*board_width*board_height,   
                                 board_width*board_height+1)
     
        self.res_val =  nn.Conv2d(in_channels=num_features, out_channels=2, kernel_size=3, stride=1 , padding=1)
        self.batchnormlize_4 = nn.BatchNorm2d(2)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)


    def forward(self, state_input):

        x = self.conv1(state_input)
        x = F.relu(self.batchnormlize_1(x))

        x = self.res_conv2(x)
        x = self.res_conv3(x)
        x = self.res_conv4(x)
        x = self.res_conv5(x)
        x = self.res_conv6(x)
        x = self.res_conv7(x)
        x = self.res_conv8(x)
        x = self.res_conv9(x)
        x = self.res_conv10(x)
        x = self.res_conv11(x)
        x = self.res_conv12(x)
        x = self.batchnormlize_2(x)
     
        x_act = self.res_act(x)
        x_act = F.relu(self.batchnormlize_3(x_act))
        x_act = x_act.view(-1, 4*self.board_width*self.board_height)
        x_act = self.act_fc1(x_act)
        x_act = torch.softmax(x_act,-1)

        x_val = self.res_val(x)
        x_val = F.relu(self.batchnormlize_4(x_val))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))

        return x_act, x_val


























    
    


