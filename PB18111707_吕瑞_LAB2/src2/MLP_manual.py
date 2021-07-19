import torch
import torch.nn as nn
import numpy as np
import math
from matplotlib import pyplot as plt
import copy

class MLP:
    def __init__(self, lr=0.05, epochs=1000, y_history=None, width=3, layers=4,W_dic=None, b_dic=None,manual_grad_w=None,auto_grad_w=None,manual_grad_b=None):
        self.lr = lr # 定义学习率
        self.epochs = epochs
        self.width = width # 定义最后输出层的 label 数量
        self.y_history = y_history
        self.W_dic = copy.deepcopy(W_dic)
        self.b_dic = copy.deepcopy(b_dic)
        self.layers = layers
        self.manual_grad_w = manual_grad_w
        self.manual_grad_b = manual_grad_b
        self.auto_grad_w = auto_grad_w

    def softmax(self,x_matrix):
        """
        x_matrix: (100,3)
        s3 : (100,3)
        """
        exp = torch.exp(x_matrix)
        all_sum = (exp.sum(axis=1)).reshape((-1,1))
        res = exp/all_sum
        # print(res)
        return res

    def cross_entropy(self,train_label,softmax_value):
        """
        train_label:(100,1)
        softmax_value: (100,3)
        return 测试数据的交叉熵(100,1)
        """
        m = len(train_label)
        loss = torch.zeros((m,1))
        for i in range(0,m):
            loss[i] = -torch.log(softmax_value[i][train_label[i]])
        return loss

    def d_cross_entropy(self,train_label,softmax_value):
        """
        train_label:(100,1)
        softmax_value: (100,3)
        return 交叉熵的导数 (100,3)
        """
        m = len(train_label)
        d_loss = copy.deepcopy(softmax_value)
        for i in range(0,m):
            d_loss[i][train_label[i]] -=1
        return d_loss


    def sigmoid(self,x):
        """
        x: (m,n)
        res: (m,n)
        """
        return 1/(1+torch.exp(-x))


    def manual_fit(self,train_data,train_label):
        """
        train_data: (100,5)
        train_label: (100,1)
        """
        dim = len(train_label)
        manual_loss = []
        # 迭代开始
        for ep in range(0,self.epochs):
            # 前向传播 - 保存 y_history
            for i in range(1,self.layers):
                w = self.W_dic[i]
                b = self.b_dic[i]
                y=self.y_history[i-1].T
                self.y_history[i] = self.sigmoid((torch.matmul(w, y)+b).T)
            # 反向传播
            delta_i = None
            delta_i_1 = None
            # delta={}
            for i in range(self.layers-1,0,-1):
                # 第 i 层
                delta_i_1=copy.deepcopy(delta_i)
                d_y = (self.y_history[i]*(1-self.y_history[i])).T
                if i < self.layers-1:
                    # 隐层
                    delta_i = torch.matmul(self.W_dic[i+1].T,delta_i_1)*d_y
                else:
                    # 输出层
                    softmax_value = self.softmax(self.y_history[self.layers-1])
                    loss = self.cross_entropy(train_label,softmax_value).mean()
                    manual_loss.append(loss.item())
                    delta_i = self.d_cross_entropy(train_label,softmax_value).T*d_y

                # delta[i] = delta_i

                grad_w = (torch.matmul(delta_i, self.y_history[i-1]))/dim  # 100个梯度的平均值
                self.manual_grad_w[i] = copy.deepcopy(grad_w)
                # self.W_dic[i] -= self.lr*grad_w
                grad_b = (delta_i.mean(axis=1)).reshape((-1, 1))
                # self.b_dic[i] -= self.lr* gard_b
                # print(delta_i)
                # print(grad_b)
                self.manual_grad_b[i] = copy.deepcopy(grad_b)

            for i in range(self.layers-1, 0, -1):
                # 梯度下降，更新 W,b
                self.W_dic[i] -= self.lr * self.manual_grad_w[i]
                self.b_dic[i] -= self.lr * self.manual_grad_b[i]

        return manual_loss

    def auto_fit(self,train_data,train_label):
        """
        train_data: (100,5)
        train_label: (100,1)
        """
        dim = len(train_label)
        W_a_dic = copy.deepcopy(self.W_dic)
        b_a_dic = copy.deepcopy(self.b_dic)
        # 自动求导
        for i in range(self.layers-1, 0, -1):
            W_a_dic[i].requires_grad = True
            b_a_dic[i].requires_grad = True

        # 迭代开始
        for ep in range(0,self.epochs):
            y_a_history = copy.deepcopy(self.y_history[0])
            # 前向传播 - 保存 y_history
            for i in range(1,self.layers):

                y_a_history = self.sigmoid(
                    (torch.matmul(W_a_dic[i], y_a_history.T)+b_a_dic[i]).T)

            # 构建损失函数，自动计算反向传播梯度
            softmax_value = self.softmax(y_a_history)
            loss = self.cross_entropy(train_label, softmax_value).mean()
            loss.backward()

            for i in range(self.layers-1, 0, -1):
                self.auto_grad_w[i] = copy.deepcopy(W_a_dic[i].grad)

                W_a_dic[i].data -= self.lr*W_a_dic[i].grad
                b_a_dic[i].data -= self.lr*b_a_dic[i].grad
                W_a_dic[i].grad.data.zero_()
                b_a_dic[i].grad.data.zero_()

def main():
    # 初始化参数
    lr = 0.05
    epochs = 1000
    dim = 100
    width = 3
    layers = 4

    shape=[5,4,4,3]
    W_dic = {}
    b_dic = {}
    y_history = {}
    manual_grad_w = {}
    auto_grad_w = {}
    manual_grad_b = {}
    for i in range(1,layers):
        W_dic[i]=torch.rand(shape[i],shape[i-1])
        manual_grad_w[i] = torch.zeros(shape[i], shape[i-1])
        auto_grad_w[i] = torch.zeros(shape[i], shape[i-1])
        b_dic[i]=torch.rand(shape[i],1)
        manual_grad_b[i] = torch.zeros(shape[i],1)
        y_history[i] = torch.zeros(dim,shape[i])

    train_data = torch.rand(dim,shape[0])
    y_history[0] = copy.deepcopy(train_data)
    trian_label = torch.randint(low=0,high=3,size=(dim,1))

    # 手算 MLP
    MLP_model = MLP(lr, epochs, y_history, width, layers, W_dic,
                    b_dic, manual_grad_w, auto_grad_w, manual_grad_b)
    manual_loss = MLP_model.manual_fit(train_data,trian_label)

    # 自动求导 MLP
    MLP_auto = MLP(lr, epochs, y_history, width, layers,
                   W_dic, b_dic, manual_grad_w, auto_grad_w, manual_grad_b)
    MLP_auto.auto_fit(train_data,trian_label)

    # 画图
    # plt.figure(figsize=(20, 8), dpi=80)
    # x=range(0,epochs)
    # plt.plot(x,manual_loss)

    # 输出梯度 对比检验
    my_mlp = open('my_mlp.txt','w')
    # my_mlp.write('loss\n')
    # my_mlp.write(str(manual_loss))
    my_mlp.write('\nmanual_grad_w\n')
    for i in range(1,layers):
        my_mlp.write(str(MLP_model.manual_grad_w[i]))
    # my_mlp.write('\nmanual_grad_b\n')
    # for i in range(1, layers):
    #     my_mlp.write(str(MLP_model.manual_grad_b[i]))
    my_mlp.close()

    auto_mlp = open('auto_mlp.txt','w')
    auto_mlp.write('\nauto_grard_w\n')
    for i in range(1,layers):
        auto_mlp.write(str(MLP_auto.auto_grad_w[i]))
    auto_mlp.close()
main()

