import torch
from torch import nn
from torch.nn import functional as F

class FullNet(nn.Module):
    'Takes the two images as input and outputs the target and a prediction for the 10 classes for the auxiliary loss'
    def __init__(self, nb_hidden_subnet, nb_hidden_full, is_siamese=True):
        super().__init__()

        self.subnet = SiameseSubNet(nb_hidden_subnet) # Does the preprocessing of the images to guess the classes
        # Two simple additional layers to determine which number is bigger.
        self.fc1 = nn.Linear(20, nb_hidden_full)
        self.fc2 = nn.Linear(nb_hidden_full, 2)

    def forward(self, x):
        # Split the channels
        image1 = x[:, 0:1]
        image2 = x[:, 1:2]

        # Preprocess/ predict class probabilities
        classes_1 = self.subnet(image1)
        classes_2 = self.subnet(image2)

        # Combine predictions and compute final result
        x = torch.stack((F.softmax(classes_1), F.softmax(classes_2)), dim=1)
        #print(x.size())

        x = F.relu(self.fc1(x.view(-1,20)))
        #print(x.size())
        x = self.fc2(x)

        # Return output and classes for the auxiliary loss
        return x, classes_1, classes_2


class SiameseSubNet(nn.Module):
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

        x = self.fc2(x)
        return x


class NaiveSubNet(nn.Module):
    'Takes an image as input and outputs a prediction for the 10 classes'

    def __init__(self, nb_hidden, nb_out_1=4, nb_out_2=8):
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

        #x = self.bn2(x)
        x = F.relu(self.fc1(x.view(-1, self.nb_out_2*4*4)))

        x = F.relu(self.fc2(x))
        return x
