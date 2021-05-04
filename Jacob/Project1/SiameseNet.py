import torch
from torch import nn
from torch.nn import functional as F
import math

######################################################################

#Weight sharing attempt. Siamese network

######################################################################

    
class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32,
                               kernel_size = 4,
                               padding = (4 - 1) // 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 128,
                               kernel_size = 4)
                               #padding = (4 - 1) // 2)
        self.bn2 = nn.BatchNorm2d(128)
        # self.fc1 = nn.Linear( 32 , 10)

        #these are 4x4x8 + 4x4x8x32 +10x32 = 4 544 params (does tachnorm have param...?)
        
        #self.conv3 = nn.Conv2d(64, 32, kernel_size=2)
        #self.bn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear( 128 , 128)
        self.fc2 = nn.Linear(128, 10)
        
        self.softmax = nn.Softmax(dim=1) 
        #self.finalLin = nn.Linear(20 , 2)




        
    def forward(self, x):
        #print(x.shape)
        x = torch.stack((F.max_pool2d(self.conv1(x[:, 0:1]), kernel_size=2), 
                        F.max_pool2d(self.conv1(x[:, 1:2]), kernel_size=2)), dim=1)
        #print(x.shape)
        x = torch.stack((self.bn1(x[:, 0]), self.bn1(x[:, 1])), dim=1)
        #print(x.shape)
        x = F.relu(x)
        #print(x.shape)
        #x = self.conv2(x)
        x = torch.stack((F.max_pool2d(self.conv2(x[:, 0]), kernel_size=2), 
                        F.max_pool2d(self.conv2(x[:, 1]), kernel_size=2)), dim=1)
        x = torch.stack((self.bn2(x[:, 0]), self.bn2(x[:, 1])), dim=1)
        #print(x.shape)
        #x = F.relu(x)
        #x = F.max_pool2d(self.conv3(x), kernel_size=2)
        #print(x.shape)
        #x = self.bn3(x)
        #x = F.relu(x)
        x = torch.stack((F.relu(self.fc1(x[:, 0].view(-1, 128))), F.relu(self.fc1(x[:, 1].view(-1, 128)))), dim=1)
        x = torch.stack((self.softmax(self.fc2(x[:, 0].view(-1, 128))), self.softmax(self.fc2(x[:, 1].view(-1, 128)))), dim=1)
        
        #x = self.sig1(x)
        #print(x.shape)
        #print(x.view(-1, 20).shape)
        #x = self.finalLin(x.view(-1, 20)) #batch size!
        
        return x

    def train_model(self, train_input, train_target, mini_batch_size, nb_epochs = 100, criterion=nn.MSELoss()):
        eta = 1e-1

        for e in range(nb_epochs):
            acc_loss = 0

            for b in range(0, train_input.size(0), mini_batch_size):
                output = self(train_input.narrow(0, b, mini_batch_size))
                loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
                acc_loss = acc_loss + loss.item()

                self.zero_grad()
                loss.backward()

                with torch.no_grad():
                    for p in self.parameters():
                        p -= eta * p.grad
            if e%10 == 0:
                print(e, acc_loss)


    def compute_nb_errors(self, input, target, mini_batch_size):
        nb_errors = 0

        for b in range(0, input.size(0), mini_batch_size):
            output = self(input.narrow(0, b, mini_batch_size))
            output1, output2 = output[:, 0], output[:, 1]
            _, predicted_classes1 =  output1.max(1)
            _, predicted_classes2 =  output2.max(1)
            #print(predicted_classes)
            for k in range(mini_batch_size):
                if target[b + k, 0, predicted_classes1[k]] <= 0 or target[b + k, 1, predicted_classes2[k]] <= 0:
                    nb_errors = nb_errors + 1

        return nb_errors









###########################################