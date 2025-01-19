import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        
        # 定义第1个卷积层: 输入1个通道的图像，输出64个通道，卷积核大小为11x11，步幅为4，填充为2
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2)
        # 定义第2个卷积层: 输入64个通道，输出192个通道，卷积核大小为5x5，步幅为1，填充为2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2)
        # 定义第3个卷积层: 输入192个通道，输出384个通道，卷积核大小为3x3，步幅为1，填充为1
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1)
        # 定义第4个卷积层: 输入384个通道，输出256个通道，卷积核大小为3x3，步幅为1，填充为1
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        # 定义第5个卷积层: 输入256个通道，输出256个通道，卷积核大小为3x3，步幅为1，填充为1
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        
        # 定义全连接层
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)  # 输入大小为256x6x6，输出大小为4096
        self.fc2 = nn.Linear(4096, 4096)  # 输入大小为4096，输出大小为4096
        self.fc3 = nn.Linear(4096, num_classes)  # 最后一层，全连接到类别数
        
        # 定义Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        # 卷积层1 + 激活 + 池化
        x = F.relu(self.conv1(x))  # ReLU激活
        x = F.max_pool2d(x, kernel_size=3, stride=2)  # 最大池化，池化核3x3，步幅2
        
        # 卷积层2 + 激活 + 池化
        x = F.relu(self.conv2(x))  # ReLU激活
        x = F.max_pool2d(x, kernel_size=3, stride=2)  # 最大池化，池化核3x3，步幅2
        
        # 卷积层3 + 激活
        x = F.relu(self.conv3(x))  # ReLU激活
        
        # 卷积层4 + 激活
        x = F.relu(self.conv4(x))  # ReLU激活
        
        # 卷积层5 + 激活 + 池化
        x = F.relu(self.conv5(x))  # ReLU激活
        x = F.max_pool2d(x, kernel_size=3, stride=2)  # 最大池化，池化核3x3，步幅2
        
        # 展平
        x = torch.flatten(x, start_dim=1)  # 展平操作，跳过第0维（批次维度）
        
        # 全连接层1 + Dropout
        x = F.relu(self.fc1(x))  # ReLU激活
        x = self.dropout(x)  # Dropout
        
        # 全连接层2 + Dropout
        x = F.relu(self.fc2(x))  # ReLU激活
        x = self.dropout(x)  # Dropout
        
        # 输出层
        x = self.fc3(x)  # 输出最终类别
        
        return x

# 初始化网络
model = AlexNet(num_classes=1000)
print(model)
