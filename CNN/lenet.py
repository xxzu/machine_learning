import torch
import torch.nn as nn
import torch.nn.functional as F

# 图长 = 图- 核长 + 2 填充   /  步长   +1
# 定义LeNet-5网络结构
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):  # num_classes默认为10，用于MNIST分类
        super(LeNet5, self).__init__()
        
        # 定义卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        # 输入为1通道(灰度图)，输出为6通道，卷积核大小为5×5，步幅为1，padding为2保持尺寸不变 图大小 28
        
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # 输入为6通道，输出为16通道，卷积核大小为5×5，默认步幅为1，padding为0  
        
        # 定义全连接层
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        # 输入特征为16×5×5（由卷积层和池化层计算得到），输出为120     
        
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        # 输入为120，输出为84
        
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)
        # 输入为84，输出为分类数（默认为10）

    def forward(self, x):
        # 输入数据流过网络的前向传播过程
        
        x = F.avg_pool2d(F.relu(self.conv1(x)), kernel_size=2, stride=2)
        # 第1层卷积 + ReLU激活函数 + 平均池化
        # 输入大小：28×28，经过卷积（padding=2）后大小不变，池化后大小变为14×14
        
        x = F.avg_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        # 第2层卷积 + ReLU激活函数 + 平均池化
        # 输入大小：14×14，经过卷积后变为10×10，池化后变为5×5
        
        x = torch.flatten(x, start_dim=1)
        # 展平操作，将多维特征图变为一维向量，方便输入全连接层
        # 输入形状：[batch_size, 16, 5, 5]，展平后为：[batch_size, 16*5*5]
        
        x = F.relu(self.fc1(x))
        # 第1个全连接层 + ReLU激活函数
        
        x = F.relu(self.fc2(x))
        # 第2个全连接层 + ReLU激活函数
        
        x = self.fc3(x)
        # 第3个全连接层（输出分类结果）
        
        return x

# 测试模型结构
if __name__ == "__main__":
    model = LeNet5(num_classes=10)
    print(model)
    
    # 创建一个假输入（MNIST图像大小为28×28）
    input_tensor = torch.rand(1, 1, 28, 28)  # batch_size=1，单通道，28×28大小
    output = model(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
