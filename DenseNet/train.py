# -*- coding: UTF-8 -*-
import torchvision.datasets
from torchvision import transforms
from tqdm import tqdm

from module_densenet import *
from torch import nn
from torch.utils.data import DataLoader

normMean = [0.49139968, 0.48215827, 0.44653124]
normStd = [0.24703233, 0.24348505, 0.26158768]
# 计算数据集的平均值和方差所得

normTransform = transforms.Normalize(normMean, normStd)
trainTransform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normTransform
    ])
testTransform = transforms.Compose([
        transforms.ToTensor(),
        normTransform
])

train_data = torchvision.datasets.CIFAR10(root="./data/data", train=True, transform=trainTransform,
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="./data/data", train=False, transform=testTransform,
                                         download=True)

train_data_size = len(train_data)  # length 长度
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mymodule = densenet169()
mymodule = mymodule.to(device)

loss_fn = nn.CrossEntropyLoss()  # 分类问题可以用交叉熵
loss_fn = loss_fn.to(device)

learning_rate = 0.001
optimizer = torch.optim.SGD(mymodule.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

total_train_step = 0  # 记录训练次数
total_test_step = 0  # 记录测试次数
epoch = 10  # 训练轮数


for i in range(epoch):
    print("----------第{}轮训练开始-----------".format(i + 1))  # i从0-9
    with tqdm(total=len(train_dataloader), desc="第{}轮训练".format(i + 1)) as pbar:
        mymodule.train()
        for data in train_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = mymodule(imgs)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()  # 首先要梯度清零
            loss.backward()  # 反向传播得到每一个参数节点的梯度
            optimizer.step()  # 对参数进行优化
            total_train_step += 1
            pbar.update(1)

with tqdm(total=len(test_dataloader), desc="第{}轮测试".format(i + 1)) as pbar:
    mymodule.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():  # 无梯度，不进行调优
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = mymodule(imgs)
            loss = loss_fn(outputs, targets)  # 该loss为部分数据在网络模型上的损失，为tensor数据类型
            # 求整体测试数据集上的误差或正确率
            total_test_loss = total_test_loss + loss.item()  # loss为tensor数据类型，而total_test_loss为普通数字
            # 求整体测试数据集上的误差或正确率
            accuracy = (outputs.argmax(1) == targets).sum().item()
            total_accuracy = total_accuracy + accuracy
            pbar.update(1)
            print("整体测试集上的Loss：{}".format(total_test_loss))
            print("整体测试集上的正确率：{}".format(total_accuracy / test_data_size))
    total_test_step += 1

torch.save(mymodule, "module_latest.pth")  # 每一轮保存一个结果
print("模型已保存")
