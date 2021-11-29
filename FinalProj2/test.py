import torch
from torch import empty as t_empty
import framework
import math as math
torch.set_grad_enabled(False)

######################################################################
################ Data generator, Train and test functions ############
######################################################################

def generate_disc_set(nb):
    input = torch.empty(nb, 2).uniform_(-1, 1)
    target = input.pow(2).sum(1).sub(2 / math.pi).sign().add(1).div(2).float()
    return input, target


def train(model, train_input, train_target, nb_epochs, batch_size, learning_rate, loss=None):
    nb_samples = train_input.size(0)
    if loss is None:
        loss = framework.MSE()
    if train_input.size(0)%batch_size !=0:
        print("Bactch size should divid length of training data.")
    for e in range(nb_epochs):
        loss_acc=0
        for b in range(0, nb_samples, batch_size):
            prediction = model.forward(train_input.narrow(0, b, batch_size))
            loss_acc += loss.forward(prediction, train_target.narrow(0, b, batch_size))
            model.backward(loss.backward(prediction, train_target.narrow(0, b, batch_size)))
            model.SGD_step(learning_rate)
        if e%5 == 0:
            print('epoch nb: ', e, 'loss: ', loss_acc)

    print("Final train error: ", compute_nb_errors(model, train_input, train_target))

def compute_nb_errors(model, data_input, data_target, batch_size=1):
    tot_right = 0

    for b in range(0, data_input.size(0), batch_size):
        for i in range(batch_size):
            output = model.forward(data_input[b+i])
            if output <0.5 and  data_target[b+i]==0: #this is specific to the toy dataset.
                tot_right+=1
            elif output >=0.5 and  data_target[b+i]==1:
                tot_right+=1

    return 1-tot_right/data_input.size(0)

######################################################################
######################### Testing Script #############################
######################################################################

# Generate data
train_input, train_target = generate_disc_set(1000)
test_input, test_target = generate_disc_set(1000)

# Center the data:
mean, std = train_input.mean(), train_input.std()

train_input.sub_(mean).div_(std)
test_input.sub_(mean).div_(std)

# Model initialization:

# Network sugested in the miniproj handout:
model = framework.Sequential((framework.Linear(2, 25),
                              framework.ReLu(),
                              framework.Linear(25, 25),
                              framework.ReLu(),
                              framework.Linear(25, 25),
                              framework.ReLu(),
                              framework.Linear(25, 1),
                              framework.Sigmoid()))
                             
loss = framework.MSE()



print("training on 250 epochs, batch size 50, and learning rate 0.01.")
train(model, train_input, train_target, nb_epochs=250, batch_size=50, learning_rate=0.01, loss=loss)

print("Test error: ", compute_nb_errors(model, test_input, test_target))


