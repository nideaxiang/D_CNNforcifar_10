"""
"让我来写点酷的"
@author: 段盛祥
@file: utils.py
@time: 2025/10/01
@desc:
"""
import torch
import torch.utils.data
import torchvision
from torchvision import transforms


# ========= 训练集部分（你已有的） =========
def get_train_dataset(data_root):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = torchvision.datasets.CIFAR10(root=data_root,
                                                 train=True,
                                                 download=False,
                                                 transform=transform)
    return train_dataset

def get_train_loader(data_root, batch_size=32, num_workers=2):
    train_dataset = get_train_dataset(data_root)          # 先拿数据集
    return torch.utils.data.DataLoader(dataset=train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=num_workers)
# ========= 测试集部分（新增） =========
def get_test_dataset(data_root):
    """
    返回 CIFAR-10 测试集
    预处理必须与训练集完全一致
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dataset = torchvision.datasets.CIFAR10(root=data_root,
                                                train=False,   # 关键区别
                                                download=False,
                                                transform=transform)
    return test_dataset


def get_test_loader(data_root, batch_size=32, num_workers=2):
    """
    返回测试集的 DataLoader
    测试时不需要 shuffle，drop_last 也无需设置
    """
    test_dataset = get_test_dataset(data_root)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers)
    return test_loader




