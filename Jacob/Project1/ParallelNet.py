import torch
from torch import nn
from torch.nn import functional as F
from BaseNet import BaseNetwork

class ParallelNet(BaseNetwork):

    def __init__(self, nb_hidden_subnet=200, nb_hidden_full=200):
        super().__init__()

        self.subnet1 = SubNet(nb_hidden_subnet) # Does the preprocessing of the images to guess the classes
        self.subnet2 = SubNet(nb_hidden_subnet)
        # Two simple additional layers to determine which number is bigger.
        self.fc1 = nn.Linear(20, nb_hidden_full)
        self.fc2 = nn.Linear(nb_hidden_full, 2)

    def forward(self, x):
        # Split the channels
        image1 = x[:, 0:1]
        image2 = x[:, 1:2]

        # Preprocess/ predict class probabilities
        classes_1 = self.subnet1(image1)
        classes_2 = self.subnet2(image2)

        # Combine predictions and compute final result
        x = torch.stack((classes_1, classes_2), dim=1)
        #print(x.size())

        x = F.relu(self.fc1(x.view(-1,20)))
        #print(x.size())
        x = self.fc2(x)

        # Return output and classes for the auxiliary loss
        return x, classes_1, classes_2




class SubNet(nn.Module):
    'Takes an image as input and outputs a prediction for the 10 classes'

    def __init__(self, nb_hidden, nb_out_1=4, nb_out_2=8):
        super().__init__()
        image_size = 14
        self.nb_out_2 = nb_out_2
        self.conv1 = nn.Conv2d(1, nb_out_1, kernel_size=4, padding=image_size % 4) # padding = (4-1 // 2) )
        self.conv2 = nn.Conv2d(nb_out_1, nb_out_2, kernel_size=4, padding=image_size % 4)
        self.fc1 = nn.Linear(nb_out_2*4*4, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)
        self.bn1 = nn.BatchNorm2d(nb_out_1)
        self.bn2 = nn.BatchNorm2d(nb_out_2)

    def forward(self, x):

        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = self.bn1(x)
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))

        #x = self.bn2(x)
        x = F.relu(self.fc1(x.view(-1, self.nb_out_2*4*4)))

        x = F.relu(self.fc2(x))
        return x
