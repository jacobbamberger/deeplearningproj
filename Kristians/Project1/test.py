#Implement auxilary loss
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import dlc_practical_prologue as prologue
import time
    
############################################# Models ###########################################################
    
class AuxNet(nn.Module):
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
        return x,y
    
class ConvNet(nn.Module):
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
        return x
    
################################# Training part (and testing...) #####################################################3    
    
def train_model(model, train_input, train_target,train_classes, test_input, test_target,test_classes, mini_batch_size, epochs, weight=2):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr = 1e-1)
        nb_epochs = epochs
        eta=0.1

        for e in range(nb_epochs):
            nb_train_errors=0
            for b in range(0, train_input.size(0), mini_batch_size): #do mini-batches
                if model.__class__.__name__== 'AuxNet':
                    output1, output2 = model(train_input[0+b:b+mini_batch_size])
                    loss_final = criterion(output1, train_target.narrow(0, b, mini_batch_size))  #larger/smaller loss
                    loss_class1 = criterion(output2[:,:10], train_classes[:,0].narrow(0, b, mini_batch_size)) #digit loss
                    loss_class2 = criterion(output2[:,10:], train_classes[:,1].narrow(0, b, mini_batch_size)) #digit loss
                    loss=loss_final+(loss_class1+loss_class2)*weight
                if model.__class__.__name__== 'ConvNet':
                    output1 = model(train_input[0+b:b+mini_batch_size])
                    loss = criterion(output1, train_target.narrow(0, b, mini_batch_size))
                pred = output1.max(1)[1]  #select the max value out from 2 clases 
                real = train_target[0+b:mini_batch_size+b]
                nb_train_errors+=mini_batch_size-torch.eq(pred,real).long().sum().item() #check if correct guess, then convert to integers, then sum-up and subtract form min_batch size to get nb of errors
                model.zero_grad()
                loss.backward()
                optimizer.step()
            if e==24:   #after the last training epoch stop the time and do a run with the test set
                toc = time.perf_counter()
                nb_test_errors=0
                for b in range(0, test_input.size(0), mini_batch_size):   #running test
                    if model.__class__.__name__== 'AuxNet':
                        output1, output2 = model(test_input[0+b:b+mini_batch_size])
                    if model.__class__.__name__== 'ConvNet':
                        output1 = model(test_input[0+b:b+mini_batch_size])
                    pred = output1.max(1)[1]
                    real = test_target[0+b:mini_batch_size+b]
                    nb_test_errors+=mini_batch_size-torch.eq(pred,real).long().sum().item() #check if correct guess, then convert to integers, then sum-up and subtract form min_batch size to get errors
                train_error[run]=nb_train_errors        
                test_error[run]=nb_test_errors
                time_spent[run]=toc-tic
                
###############################################Run the test##############################################################                

for m in [ConvNet,AuxNet]:                            
    nb_runs =25 #how many runs we average over
    train_error=torch.empty(nb_runs,1) #make empty tensors to store errors and elapsed time from every run we average over
    test_error=torch.empty(nb_runs,1)
    time_spent=torch.empty(nb_runs,1)

    model=m() #choose model
    mini_batch_size=50  #mini batch size - smaller sizes take longer time, but will converge faster
    epochs=25  #limited to 25 epochs to have some idea of comparison with the task

    for run in range (nb_runs):
        for i,layer in enumerate(model.children()):  #Need to reset parameters otherwise "m" keeps the weights from previous training runs and results in better test
            layer.reset_parameters()

        tic = time.perf_counter()  #time the run
        train_input,train_target,train_classes,test_input, test_target,test_classes = prologue.generate_pair_sets(1000) #generate pairs (they are random each time)


        mu, std = train_input.mean(), train_input.std()  #normalize input data
        train_input.sub_(mu).div_(std)
        test_input.sub_(mu).div_(std)

        train_model(model,train_input,train_target,train_classes,test_input,test_target,test_classes,mini_batch_size, epochs) #train for 25 epochs and test it out on last one

    print('Model: {}, Train error: {:.2f}\u00B1{:.2f}%, Test error: {:.1f}\u00B1{:.1f}%, Average time elapsed: {:.1f}\u00B1{:.1f}s'.format(
                model.__class__.__name__,  
                train_error.mean().item()/10, train_error.std().item()/10,
                test_error.mean().item()/10, test_error.std().item()/10,
                time_spent.mean().item(), time_spent.std().item(),
            ))