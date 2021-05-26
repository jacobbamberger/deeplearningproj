import torch
from torch import nn
from torch.nn import functional as F


class ConvNet(nn.Module):
    '''A simple convolution network that processes both images at the same time and directlys predicts the
    output of interest.'''
    def __init__(self, nb_hidden = 250):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 30, kernel_size=3) #makes 14x14 -> 12x12
        self.conv2 = nn.Conv2d(30, 60, kernel_size=3) #makes 6x6 -> 4x4
        self.fc1 = nn.Linear(240, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 2)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), kernel_size=2)  #should be 12x12->6x6
        x = F.relu(x)
        x = F.max_pool2d(self.conv2(x), kernel_size=2)   #4x4->2x2
        x = F.relu(x)
        x = F.relu(self.fc1(x.view(-1, 240)))
        x = self.fc2(x)
        return x, None


class AuxNet(nn.Module):
    """ A simple convolution network that processes both images at the same time, but can make use of an auxiliary loss. """
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
        y = self.fc1(x.view(-1, 240))
        x = self.fc2(F.relu(y))
        return x, y[:, :10], y[:, 10:]


class FullNet(nn.Module):
    """A two part network, that has different subnets for predicting the classes from the two images,
     and then uses linear layers to predict the output."""
    def __init__(self, nb_hidden_subnet, nb_hidden_full, subnet_type = 'siamese', use_softmax= True):
        super().__init__()
        self.subnet_type = subnet_type
        self.use_softmax = use_softmax

        # A subnet to preprocess the images/ predict class probabilities
        if self.subnet_type == 'siamese':
            self.subnet = SiameseSubNet(nb_hidden_subnet)
        elif self.subnet_type == 'parallel':
            self.subnet1 = SiameseSubNet(nb_hidden_subnet)
            self.subnet2 = SiameseSubNet(nb_hidden_subnet)
        elif self.subnet_type =='naive':
            self.naivesubnet = NaiveSubNet(nb_hidden_subnet)
        else:
            raise ValueError("Not a valid type for the subnetwork")

        # Two simple additional layers to determine which number is bigger.
        self.fc1 = nn.Linear(20, nb_hidden_full)
        self.fc2 = nn.Linear(nb_hidden_full, 2)

    def forward(self, x):
        # Split the channels
        image1 = x[:, 0:1]
        image2 = x[:, 1:2]

        # Preprocess/ predict class probabilities
        if self.subnet_type == 'siamese':
            classes_1 = self.subnet(image1)
            classes_2 = self.subnet(image2)
        elif self.subnet_type == 'parallel':
            classes_1 = self.subnet1(image1)
            classes_2 = self.subnet2(image2)
        elif self.subnet_type =='naive':
            classes_1, classes_2 = self.naivesubnet(x)
        else:
            print('Invalid subnetwork type')
            return None

        # Combine predictions and compute final result
        if self.use_softmax:
            x = torch.stack((F.softmax(classes_1), F.softmax(classes_2)), dim=1)
        else:
            x = torch.stack((F.relu(classes_1), F.relu(classes_2)), dim=1)
        x = F.relu(self.fc1(x.view(-1,20)))
        x = self.fc2(x)

        # Return output and classes for the auxiliary loss
        return x, classes_1, classes_2


class SiameseSubNet(nn.Module):
    """Takes an image as input and outputs a prediction for the 10 classes. This subnet only processes one image"""

    def __init__(self, nb_hidden, nb_out_1=4, nb_out_2=8):

        super().__init__()
        image_size = 14
        self.nb_out_2 = nb_out_2
        self.conv1 = nn.Conv2d(1, nb_out_1, kernel_size=4, padding=image_size % 4) # padding = (4-1 // 2) )
        self.conv2 = nn.Conv2d(nb_out_1, nb_out_2, kernel_size=4, padding=image_size % 4)
        self.fc1 = nn.Linear(nb_out_2*4*4, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)
        self.bn1 = nn.BatchNorm2d(nb_out_1)

    def forward(self, x):

        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = self.bn1(x)
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = F.relu(self.fc1(x.view(-1, self.nb_out_2*4*4)))
        x = self.fc2(x)

        return x


class NaiveSubNet(nn.Module):
    """Takes two image as input and outputs a prediction for the 20 classes. This subnet processes both images at the
     same time."""

    def __init__(self, nb_hidden, nb_out_1=8, nb_out_2=16):
        super().__init__()
        image_size = 14
        self.nb_out_2 = nb_out_2
        self.conv1 = nn.Conv2d(2, nb_out_1, kernel_size=4, padding=image_size % 4) # padding = (4-1 // 2) )
        self.conv2 = nn.Conv2d(nb_out_1, nb_out_2, kernel_size=4, padding=image_size % 4)
        self.fc1 = nn.Linear(nb_out_2*4*4, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 20)
        self.bn1 = nn.BatchNorm2d(nb_out_1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = self.bn1(x)
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = F.relu(self.fc1(x.view(-1, self.nb_out_2*4*4)))
        x = self.fc2(x)

        return x[:, :10], x[:, 10:]