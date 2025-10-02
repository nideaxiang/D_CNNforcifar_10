"""
"让我来写点酷的"
@author: 段盛祥
@file: test_lost.py
@time: 2025/10/02
@desc:
"""
import torch
# # loss = torch.nn.L1Loss(reduction="sum")
# L1loss = torch.nn.L1Loss()
# logits = torch.tensor((10,20,30),dtype=torch.float)
# labels = torch.tensor((15,20,30),dtype=torch.float)
# loss = L1loss(logits,labels)
# print(loss)
# #loss.backward()

import torch
from torch import nn
output=torch.tensor([[1,2,3.0],[1,2,3.0]])
target=torch.tensor([1,0])
#target=torch.tensor([0,1,0],[1,0,0]) 由于直接运算，维度无法对上，所以需要展开为one-hot编码 1代表第二类010 0代表第一类100
nll=nn.NLLLoss()
res=nll(output,target)#res(-2+-1)/2
print(res)