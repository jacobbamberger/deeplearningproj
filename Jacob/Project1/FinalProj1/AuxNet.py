import torch
from torch import nn
from BaseNet import BaseNet
from torch.nn import functional as F
    
############################################# Models ###########################################################
    
class AuxNet(BaseNet):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 30, kernel_size=3) #makes 14x14 -> 12x12 
        self.conv2 = nn.Conv2d(30, 60, kernel_size=3) #makes 6x6 -> 4x4 
        self.bn1 = nn.BatchNorm2d(30)
        self.bn2 = nn.BatchNorm2d(60)
        self.fc1 = nn.Linear(240, 20)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), kernel_size=2)  #should be 12x12->6x6
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(self.conv2(x), kernel_size=2)   #4x4->2x2
        x = self.bn2(x)
        x = F.relu(x)
        y = F.relu(self.fc1(x.view(-1, 240)))
        x = self.fc2(y)
        return x, y[:, :10], y[:, 10:]
    
