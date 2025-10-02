"""
"让我来写点酷的"
@author: 段盛祥
@file: main.py
@time: 2025/10/02
@desc:
"""
import torch
from model import D_CNN
import torch.nn.functional as F
from torch import optim
import utils

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader = utils.get_train_loader('./dataset')
    test_loader = utils.get_test_loader('./dataset')

    # 1. 创建模型
    model = D_CNN().to(device)

    # 2. 定义优化器
    opt = optim.Adam(model.parameters(), lr=0.001)

    # 3. 训练循环
    epochs = 8
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            y = model(images)
            loss = F.cross_entropy(y, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()

            # 每100个batch打印一次平均loss
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], "
                      f"Loss: {running_loss/100:.4f}")
                running_loss = 0.0

        # ===== 每个epoch结束后在测试集上评估精度 =====
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] finished, Test Accuracy: {acc:.2f}%\n")
