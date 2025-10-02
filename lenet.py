"""
"让我来写点酷的"
@author: 段盛祥
@file: lenet.py
@time: 2025/10/01
@desc:
"""

import torch
from torch import nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(
nn.Conv2d(in_channels=3,out_channels=6, kernel_size=5, stride=1,padding=0)
,nn.ReLU()
,nn.MaxPool2d(kernel_size=2, stride=2)
,nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1,padding=0)
,nn.ReLU()
,nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.classifier = nn.Sequential(
nn.Linear(in_features=16 * 5 * 5, out_features=120)
, nn.ReLU()
,nn.Linear(in_features=120,out_features=84)
            , nn.ReLU()
,nn.Linear(in_features=84,out_features=10))
    def forward(self,x):
        x=self.features(x)
        x=torch.flatten(x,1)#1代表从第1维开始展平
        x=self.classifier(x)

"搭积木 逐个写"
# class Net(nn.Module):
#    def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels=3,out_channels=6, kernel_size=5,stride=1,padding=0)
#         self.relu1=nn.ReLU()
#         self.pool1 = nn. MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1,padding=0)
#         self.relu2=nn.ReLU()
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.flatten=nn.Flatten() #展平
#         self.fc1 = nn.Linear(in_features=16 * 5 * 5,out_features=120)
#         #展平以后是400
#         self.relu3 = nn.ReLU()
#         self.fc2 = nn. Linear(in_features=120,out_features=84)
#         self.relu4=nn.ReLU()
#         self.fc3 = nn.Linear(in_features=84, out_features=10)#十分类
#
#
#     def forward(self,x):
#         x=self.conv1(x)
#         x=self.relu1(x)
#         x=self.pool1(x)
#         x=self.conv2(x)
#         x=self.relu2(x)
#         x=self.pool2(x)
#         x=self.flatten(x)
#         print(x.shape)
#         x=self.fc1(x)
#         x=self.relu3(x)
#         x=self.fc2(x)
#         x = self.relu4(x)
#         x = self.fc3(x)
#         return x


x=torch.ones(3,3,32,32)
net=Net()
print(net)
y=net(x)