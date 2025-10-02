"""
"让我来写点酷的"
@author: 段盛祥
@file: main.py
@time: 2025/10/02
@desc:
"""
import torch
from model import D_CNN
from torchvision import models
import torch.nn.functional as F
from torch import optim
import utils
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import numpy as np

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader = utils.get_train_loader('dataset')
    test_loader = utils.get_test_loader('dataset')

    # 1. 创建模型
    # 载入预训练 ResNet18
    model = models.resnet18(pretrained=True)

        # 替换最后全连接层，输出 CIFAR-10 10 类
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    model = model.to(device)

    # 2. 定义优化器
    opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    from torch.optim.lr_scheduler import StepLR

    criterion =nn.CrossEntropyLoss()
    scheduler = StepLR(opt, step_size=30, gamma=0.1)
    # 3. 训练循环
    epochs = 10
    best_acc = 0.0  # 用于保存最佳模型
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], "
                      f"Loss: {running_loss / 100:.4f}")
                running_loss = 0.0

            # ===== 验证集评估 Accuracy & multi-class AUC =====
        model.eval()
        y_true, y_score = [], []
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                y_true.extend(labels.cpu().numpy())
                y_score.extend(probs.cpu().numpy())

        acc = 100 * correct / total
        y_true_onehot = np.eye(10)[y_true]
        auc = roc_auc_score(y_true_onehot, np.array(y_score), multi_class='ovr')

        print(f"Epoch [{epoch + 1}/{epochs}] finished => Test Accuracy: {acc:.2f}%, Test AUC: {auc:.4f}")

        # ===== 保存最佳模型 =====
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_resnet18_cifar10.pth")
            print("Best model saved.")

        scheduler.step()