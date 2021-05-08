import torch
from torch import nn
from BaseNet import BaseNet
from torch.nn import functional as F

class ConvNet(BaseNet):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 60, kernel_size=3) #makes 14x14 -> 12x12 
        self.conv2 = nn.Conv2d(60, 120, kernel_size=3) #makes 6x6 -> 4x4 
        self.fc1 = nn.Linear(480, 250)
        self.fc2 = nn.Linear(250, 2)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), kernel_size=2)  #should be 12x12->6x6
        x = F.relu(x)
        x = F.max_pool2d(self.conv2(x), kernel_size=2)   #4x4->2x2
        x = F.relu(x)
        x = F.relu(self.fc1(x.view(-1, 480)))
        x = self.fc2(x)
        return x, None
    
