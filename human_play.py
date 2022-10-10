
import time
import pickle
import os
  
import tkinter as tk
from PIL import ImageTk

from GoEnv.environment import GoEnv
from configure import Config
from model import TransGoNetwork
from self_play import MCTS
  
class Human_vs_human:   # 这个从 Human_vs_bot 改过来的，重复部分比较多，注释去看 Human_vs_bot就行
    def __init__(self, config):   

        self.config = config
        self.game = GoEnv(config)
        self.state,self.done = self.game.reset()

        self.start = 0
        self.resign = 0
        self.save_canvas ={}
        self.label = {}
        
        self.board_9_file = r'go_gui\board_9.png'
        self.black_9_file = r'go_gui\b_9.png'
        self.white_9_file = r'go_gui\w_9.png'
        self.board_9_data = (65+5, 65+80, 55, 572)
  

    def graphic(self):
        window = tk.Tk()
        window.title('9*9围棋对弈')

        window.geometry('732x700')  # 设定窗口的大小(宽*高)
        self.canvas = tk.Canvas(window, height=650, width=572+160,background="light blue")

        board = ImageTk.PhotoImage(file= self.board_9_file)
        self.canvas.create_image(366, 5, anchor='n', image=board)
        self.black_stone = ImageTk.PhotoImage(file=self.black_9_file)
        self.white_stone = ImageTk.PhotoImage(file=self.white_9_file)
        self.row_start = self.board_9_data[0]
        self.col_start = self.board_9_data[1]
        self.board_interval = self.board_9_data[2]

        self.button = tk.Button(window, text="start game!", command=self.run)

        self.button_pass = tk.Button(window,text = " pass " , command = self.pass_)
        self.button_pass.place(x=200,y=572+15)

        self.button_resign = tk.Button(window,text = "resign" , command = self.resign_)
        self.button_resign.place(x=485,y=572+15)

        self.canvas.bind('<Button-1>', self.click)
        self.canvas.pack()
        self.button.pack()
        window.mainloop()

    def run(self):
        self.start = 1
 
        if self.done:
            win = self.game.getWinner(self.state)
            if win == 1:
                winner = "black"
            else:
                winner = "white"
            self.canvas.create_text(365, 630,
                                    text="Game over. Winner is {}".format(winner),fill= "red",font = "4")
            self.canvas.unbind('<Button-1>')
            return winner
        else:
            self.canvas.after(100, self.run)    # 每隔100ms执行一次run函数

        if self.resign != 0:
            if self.resign == 1:
                winner = "white"
            else:
                winner = "black"
            self.canvas.create_text(365, 630,
                                    text="Game over. Winner is {}".format(winner),fill= "red",font = "4")
            self.canvas.unbind('<Button-1>')


    def click(self, event):

        next_player = self.game.getPlayer(self.state)
        if  self.start == 1:
            i = (event.x - self.col_start ) // self.board_interval
            j = (event.y - self.row_start)  //  self.board_interval
            ri = (event.x - self.col_start ) % self.board_interval
            rj = (event.y - self.row_start)  %  self.board_interval
            i = i  if ri < 27.5 else i+1
            j = j  if rj < 27.5 else j+1

            action = self.game.location_to_action([j,i])

            if self.game.checkAction(self.state, action):
                x = i * self.board_interval + self.col_start
                y = j * self.board_interval + self.row_start
                if next_player == 1 :  # 黑
                   self.save_canvas[action]= self.canvas.create_image(i * self.board_interval + self.col_start, j * self.board_interval +
                                        self.row_start, anchor='center', image=self.black_stone)
                   if self.label !={}:
                        self.canvas.delete(self.label[1]) 
                   self.label[1] = self.canvas.create_polygon(x-15,y-15,x+15,y-15,x,y+15,fill="red",outline="")
               
                elif next_player == 2 :  # 白
                   self.save_canvas[action]= self.canvas.create_image(i * self.board_interval + self.col_start, j * self.board_interval +
                                        self.row_start, anchor='center', image=self.white_stone)
                   if self.label !={}:
                        self.canvas.delete(self.label[1]) 
                   self.label[1] = self.canvas.create_polygon(x-15,y-15,x+15,y-15,x,y+15,fill="red",outline="")

                self.state, self.done = self.game.step(self.state,action)
                g = self.game.board_grid(self.state).reshape(-1)
              
                for index, value  in enumerate(g) :
                    if value == 0 and (index in self.save_canvas) :
                        self.canvas.delete(self.save_canvas[index]) 

    def pass_(self):
        if self.start == 1:
             action = self.game.board_size**2
             self.state , _= self.game.step(self.state, action)
             g = self.game.board_grid(self.state).reshape(-1)

    def resign_(self):
        self.resign = self.game.getPlayer(self.state)
        self.start = 0
 
class Human_vs_bot:
    def __init__(self,config):
 
        self.game = GoEnv(config)
        self.state,self.done = self.game.reset()

        self.start = 0         # 开始标志，为1则开始
        self.save_canvas ={}   # 方便将棋子删除
        self.label = {}        # 红色三角形标志（当前落子点位置）,方便添加删除
        self.resign = 0        # 当有一方resign时，赋值给self.resign , 确认是哪一方认输
        
        # 素材图
        self.board_9_file = r'go_gui\board_9.png'
        self.black_9_file = r'go_gui\b_9.png'
        self.white_9_file = r'go_gui\w_9.png'
        self.board_9_data = (65+5, 65+80, 55, 572)   # 棋盘左上距离画布左上的高，宽，棋盘格子间距，棋盘尺寸 

        init_model = None
        init_model = "./save_weight/best_policy_1500.model"      # 换模型，改这里就行
    
        if init_model:
            if os.path.exists(init_model):
                print("load model success")
                with open(init_model, "rb") as f:
                    model_weights = pickle.load(f)

        self.device = config.device
        self.model = TransGoNetwork(config).to(self.device)          
        self.model.set_weights(model_weights["weights"])

        self.agent =  MCTS(config, self.game, self.model)

    def play_with_bot(self,sel_color):   # 确定 human 和 agent 执手颜色
        self.sel_color = sel_color
       
        if self.sel_color =="black":
            self.human_value = 1
            self.agent_value = 2
        elif self.sel_color =="white":
            self.human_value = 2
            self.agent_value = 1
        
        self.bots = {self.human_value: "agent_human", self.agent_value: self.agent}
 
        self.graphic()
            
    def graphic(self):                # tk 主程序
        window = tk.Tk()
        window.title('9*9围棋对弈')

        window.geometry('732x700')  # 设定窗口的大小(宽*高)
        self.canvas = tk.Canvas(window, height=650, width=572+160,background="light blue")   #创建画布

        board = ImageTk.PhotoImage(file= self.board_9_file)
        self.canvas.create_image(366, 5, anchor='n', image=board)        # 在画布添加棋盘
        self.black_stone = ImageTk.PhotoImage(file=self.black_9_file)    # 读取黑子图案
        self.white_stone = ImageTk.PhotoImage(file=self.white_9_file)    # 读取白子图案
       
        self.row_start = self.board_9_data[0]
        self.col_start = self.board_9_data[1]
        self.board_interval = self.board_9_data[2]

        # 创建按钮
        self.button = tk.Button(window, text="start game!", command=self.run)     # 链接 run()函数

        self.button_pass = tk.Button(window,text = " pass " , command = self.pass_) # 链接 pass_()函数
        self.button_pass.place(x=200,y=572+15)   # 位置

        self.button_resign = tk.Button(window,text = "resign" , command = self.resign_) # 链接 resign_()函数
        self.button_resign.place(x=485,y=572+15)

        self.canvas.bind('<Button-1>', self.click)   # 关联鼠标点击
       
        self.canvas.pack()
        self.button.pack()
        # self.button_pass.pack()
        # self.button_resign.pack()

        window.mainloop()    # 循环刷新

    def run(self):
        self.start = 1
        next_player = self.game.getPlayer(self.state)

        if next_player == self.agent_value and not self.done:   # 轮到 agent落子 ， 执行if语句内程序

            action = self.bots[self.agent_value].select_action(self.state)   # agent选择动作
            # print("action:", action)
            self.state, self.done = self.game.step(self.state,action)
            [row,col] = self.game.action_to_location(action)
                  
            if action != self.game.board_size*self.game.board_size : 
                i = row     # 高
                j = col     # 宽
                x = j * self.board_interval + self.col_start
                y = i * self.board_interval + self.row_start
                # x,y 为落子点坐标

                # 在 对应坐标 创建棋子
                if self.sel_color == "black" :
                    self.save_canvas[action]= self.canvas.create_image(j * self.board_interval + self.col_start, i * self.board_interval +
                                        self.row_start, anchor='center', image=self.white_stone)
                    if self.label !={}:
                        self.canvas.delete(self.label[1]) # 删除倒三角形标志（之前的）
                    self.label[1] = self.canvas.create_polygon(x-15,y-15,x+15,y-15,x,y+15,fill="red",outline="") # 新增倒三角形标志（现在的）
                else:
                    self.save_canvas[action]= self.canvas.create_image(j * self.board_interval + self.col_start, i * self.board_interval +
                                        self.row_start, anchor='center', image=self.black_stone)
                    if self.label !={}:
                        self.canvas.delete(self.label[1]) 
                    self.label[1] = self.canvas.create_polygon(x-15,y-15,x+15,y-15,x,y+15,fill="red",outline="")
            
            # 吃子，删除棋子
            g = self.game.board_grid(self.state).reshape(-1)
            for index, value in enumerate(g) :
                if value == 0 and (index in self.save_canvas) :
                    self.canvas.delete(self.save_canvas[index]) 

        if self.done:
            win = self.game.getWinner(self.state)
            if win == 1:
                winner = "black"
            else:
                winner = "white"
            self.canvas.create_text(365, 630,
                                    text="Game over. Winner is {}".format(winner),fill= "red",font = "4")
            self.canvas.unbind('<Button-1>')
            return winner
        else:
            self.canvas.after(100, self.run)   #间隔100ms ，重新执行run（）函数

        if self.resign != 0:
            if self.resign == 1:
                winner = "white"
            else:
                winner = "black"
            self.canvas.create_text(365, 630,
                                    text="Game over. Winner is {}".format(winner),fill= "red",font = "4")
            self.canvas.unbind('<Button-1>')


    def click(self, event):

        next_player = self.game.getPlayer(self.state)
        if next_player == self.human_value and self.start == 1:

            # 鼠标点击，判断在哪个盘格交叉点的周围；将点击的坐标转换成交叉点坐标。
            i = (event.x - self.col_start ) // self.board_interval
            j = (event.y - self.row_start)  //  self.board_interval
            ri = (event.x - self.col_start ) % self.board_interval
            rj = (event.y - self.row_start)  %  self.board_interval
            i = i  if ri < 27.5 else i+1
            j = j  if rj < 27.5 else j+1

            action = self.game.location_to_action([j,i])
            
            if self.game.checkAction(self.state, action):   # 判断是否合法
                x = i * self.board_interval + self.col_start
                y = j * self.board_interval + self.row_start

                # 在 对应坐标 创建棋子
                if self.sel_color == "black" :
                   self.save_canvas[action]= self.canvas.create_image(i * self.board_interval + self.col_start, j * self.board_interval +
                                        self.row_start, anchor='center', image=self.black_stone)
                   if self.label !={}:
                        self.canvas.delete(self.label[1])   # 删除倒三角形标志（之前的）
                   self.label[1] = self.canvas.create_polygon(x-15,y-15,x+15,y-15,x,y+15,fill="red",outline="")
                else:
                   self.save_canvas[action]= self.canvas.create_image(i * self.board_interval + self.col_start, j * self.board_interval +
                                        self.row_start, anchor='center', image=self.white_stone)
                   if self.label !={}:
                        self.canvas.delete(self.label[1]) 
                   self.label[1] = self.canvas.create_polygon(x-15,y-15,x+15,y-15,x,y+15,fill="red",outline="")

                self.state, self.done = self.game.step(self.state, action)
                
                #有吃子，删除棋子
                g = self.game.board_grid(self.state).reshape(-1)
                for index, value in enumerate(g) :
                    if value == 0 and (index in self.save_canvas) :
                        self.canvas.delete(self.save_canvas[index]) 

    def pass_(self):
        next_player = self.game.getPlayer(self.state)
        if next_player == self.human_value and self.start == 1:
             action = self.game.board_size**2
             self.state , _= self.game.step(self.state, action)

    def resign_(self):
        self.resign = self.game.getPlayer(self.state)
        self.start = 0
   
      
if __name__ == "__main__":
    config = Config()

    # play = Human_vs_human(config)
    # play.graphic()

    play = Human_vs_bot(config)
    # play.play_with_bot("white")   # human执手颜色
    play.play_with_bot("black")   # human执手颜色
    

