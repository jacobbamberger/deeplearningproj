#!/usr/bin/env python

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

import dlc_practical_prologue as prologue
import Networks


nb = 1000
image_size = 14
nb_output = 4
nb_channels_in = 2
nb_out_1 = 4
nb_out_2 = 4
train_input, train_target, train_classes, test_input, test_target, test_classes =\
    prologue.generate_pair_sets(nb)

mu, std = train_input.mean(), train_input.std()  # normalize input data
train_input.sub_(mu).div_(std)
test_input.sub_(mu).div_(std)


print(train_classes.size())



######################################################################

def train_model(model, train_input, train_target, mini_batch_size, nb_epochs = 25):
    main_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-1)
    aux_criterion_1 = nn.CrossEntropyLoss()
    aux_criterion_2 = nn.CrossEntropyLoss()
    eta = 1e-1

    for e in range(nb_epochs):
        acc_loss = 0

        for b in range(0, train_input.size(0), mini_batch_size):

            batch_input = train_input.narrow(0, b, mini_batch_size)
            #print(batch_input.size())
            output, classes_1, classes_2 = model(batch_input)
            #print(output.size())
            #print( train_target.narrow(0, b, mini_batch_size).size())
            main_loss =  main_criterion(output, train_target.narrow(0, b, mini_batch_size))
            aux_loss_1 =  aux_criterion_1(classes_1, train_classes.narrow(0,b,mini_batch_size)[:,0])
            aux_loss_2 = aux_criterion_2(classes_2, train_classes.narrow(0, b, mini_batch_size)[:,1])
            total_loss = main_loss + 0.2 *aux_loss_1 + 0.2* aux_loss_2

            acc_loss = acc_loss + total_loss.item()
            model.zero_grad()
            total_loss.backward()
            optimizer.step()

        print(e, acc_loss)



def train_model_with_auxiliary_loss(model, train_input, train_target, train_classes, mini_batch_size, nb_epochs = 100):
    criterion = nn.CrossEntropyLoss()
    eta = 1e-1

    for e in range(nb_epochs):
        acc_loss = 0

        for b in range(0, train_input.size(0), mini_batch_size):

            batch_input = train_input.narrow(0, b, mini_batch_size)
            #print(batch_input.size())
            output, classes_1, classes_2 = model(batch_input)
            #print(output.size())
            #print( train_target.narrow(0, b, mini_batch_size).size())
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))

            acc_loss = acc_loss + loss.item()
            model.zero_grad()
            loss.backward()

            with torch.no_grad():
                for p in model.parameters():
                    p -= eta * p.grad

        print(e, acc_loss)


###########################################


def compute_nb_errors(model, input, target, mini_batch_size):
    nb_errors = 0

    for b in range(0, input.size(0), mini_batch_size):
        output, classes_1, classes_2 = model(input.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.max(1)
        #print(predicted_classes)

        for k in range(mini_batch_size):
            if target[b + k] != predicted_classes[k]:
                nb_errors = nb_errors + 1

    return nb_errors

######################################################################

mini_batch_size = 50

######################################################################
# Question 2

for k in range(10):
    model = Networks.FullNet(40,40)
    train_model(model, train_input, train_target, mini_batch_size, nb_epochs=50)
    nb_test_errors = compute_nb_errors(model, test_input, test_target, mini_batch_size)
    print('test error Net {:0.2f}% {:d}/{:d}'.format((100 * nb_test_errors) / test_input.size(0),
                                                      nb_test_errors, test_input.size(0)))


