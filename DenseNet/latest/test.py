# -*- coding: UTF-8 -*-
import torch
import torchvision.transforms
from PIL import Image
import numpy as np
from torch import nn
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_folder = "./imgs"  # 图片文件夹路径
model_path = "module_latest.pth"  # 模型文件路径

# 读取图片（PIL Image），再用ToTensor进行转换
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

# 加载网络模型
model = torch.load(model_path)
model = model.to(device)
model.eval()  # 将模型设置为评估模式

# 遍历文件夹中的所有图片文件
for file_name in os.listdir(image_folder):
    if file_name.endswith(".png") or file_name.endswith(".jpg"):
        image_path = os.path.join(image_folder, file_name)
        # 读取图片（PIL Image），再用ToTensor进行转换
        image = Image.open(image_path)
        image = transform(image)
        image = torch.reshape(image, (1, 3, 32, 32))
        image = image.to(device)

        # 使用模型进行预测
        with torch.no_grad():
            output = model(image)
        answer = output.argmax(dim=1).item()

        # 输出预测结果
        labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        print(labels[answer])
