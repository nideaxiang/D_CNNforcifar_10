"""
"让我来写点酷的"
@author: 段盛祥
@file: model.py
@time: 2025/10/02
@desc:
"""
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
from torch import optim as optim
import  torch.nn.functional as F

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
        self.fcl2=nn.Linear(100,10)
    def forward(self,x):
        x=F.relu(self.bn1(self.conv1(x)))
        x=self.MaxPool(x)
        x=F.relu(self.bn2(self.conv2(x)))
        x=self.MaxPool(x)
        x=x.view(-1,8*8*32)
        x=F.relu(self.fcl1(x))
        y=F.relu(self.fcl2(x))

        return y


