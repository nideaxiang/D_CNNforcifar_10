"""
"让我来写点酷的"
@author: 段盛祥
@file: torch_test.py
@time: 2025/10/02
@desc:
"""
'基本操作'
import torch
import numpy as np
import torch.nn as nn

import  torch.nn.functional as F

'定义一个5*8的全连接层'
w=torch.randn(5,8,requires_grad=True)
b=torch.randn(8,requires_grad=True)
x=torch.randn(1,5)
Y=torch.randn(1,8)
'计算一个全连接层'
y=F.softmax(x @ w + b)
print(y)

'计算损失'
loss=F.cross_entropy(y,Y)

'梯度下降'
loss.backward() #找梯度
# print(w.grad) #w的梯度
# print(b.grad)
#向梯度相反的方向下降 学习率为lr
lr=0.001
w = w - lr*w.grad #下降一次

'搭建模型'
#建类
class D_CNN(nn.Module):
    def __init__(self):
        '初始化函数方法 模型的结构'
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=16,stride=1,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(in_channels=16,out_channels=32,stride=1,kernel_size=3,padding=1)
        self.bn1=nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.MaxPool=nn.MaxPool2d(kernel_size=2)
        self.flatten=nn.Flatten()
        self.fcl1=nn.Linear(32*8*8,100)
        self.fcl2=nn.Linear(100,2)
    def forward(self,x):
        x=F.relu(self.bn1(self.conv1(x)))
        x=self.MaxPool(x)
        x=F.relu(self.bn2(self.conv2(x)))
        x=self.MaxPool(x)
        x=x.view(-1,12*12*32)
        x=F.relu(self.fcl1(x))
        x=F.relu(self.fcl2(x))

        return x


