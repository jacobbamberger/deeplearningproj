import torch
from torch import nn
import numpy as np
from torch import optim


class BaseNet(nn.Module):
    # The idea is to have a base Network class that 
    # we all extend from, to have uniform training, and testing
    def __init__(self):
        super().__init__()


    def train_model(self, train_input, train_target, train_classes = None, 
                    learning_rate = 1e-1, mini_batch_size=50, nb_epochs=25,
                    aux_loss_weight = 0.2, main_criterion = nn.CrossEntropyLoss(), 
                    aux_criterion_1 = None, aux_criterion_2 = None, optimizer=None):
        #the following is to reuse some parameters:
        self.mini_batch_size = mini_batch_size
        self.nb_epochs = nb_epochs

        #The following if/else is for using auxiliary loss or not.
        if train_classes is None:
            use_aux_loss = False
        else:
            use_aux_loss = True
            if aux_criterion_1 is None:
                aux_criterion_1 = nn.CrossEntropyLoss()
                aux_criterion_2 = nn.CrossEntropyLoss()

        if optimizer is None:
            optimizer = optim.SGD(self.parameters(), lr=learning_rate)

        print("Trainin model {}...".format(
            self.__class__.__name__))

        print("Printing acc loss every 5 epochs...")
        for e in range(self.nb_epochs):
            acc_loss = 0

            for b in range(0, train_input.size(0), self.mini_batch_size):

                batch_input = train_input.narrow(0, b, self.mini_batch_size)

                # In the line below, self should input a tensor of shape 
                # batch_size*2*14*14 and output either just a size 2 vector, 
                # or a size 2 vector and two size 10 vector.
                output, *classes = self(batch_input) 

                main_loss = main_criterion(output, train_target.narrow(0, b, self.mini_batch_size))
                if use_aux_loss:
                    classes_1, classes_2 = classes
                    aux_loss_1 = aux_criterion_1(classes_1, train_classes.narrow(0, b, self.mini_batch_size)[:, 0])
                    aux_loss_2 = aux_criterion_2(classes_2, train_classes.narrow(0, b, self.mini_batch_size)[:, 1])
                    total_loss = main_loss + aux_loss_weight * (aux_loss_1 + aux_loss_2)
                else:
                    total_loss = main_loss

                acc_loss = acc_loss + total_loss.item()
                self.zero_grad()
                total_loss.backward()
                optimizer.step()
            if e%5 == 0:
                print("Accumulated loss at epoch: ", e, " is: ", acc_loss)


    def compute_nb_errors(self, test_input, test_target, test_classes=None,):
        # computes number of errors on test set, not caring about image classes

        nb_errors=0
        for b in range(0, test_input.size(0), self.mini_batch_size):
            batch_input = test_input.narrow(0, b, self.mini_batch_size)

            output, *classes = self(batch_input) 
            _, predicted_classes = output.max(1)  

            for k in range(self.mini_batch_size):
                if test_target[b + k] != predicted_classes[k]:
                    nb_errors += 1
        #print("In this set of size: ", len(test_input), " there were:", nb_errors, "errors")
        #print("This is an accuracy of {:.02f}%".format(nb_errors / test_input.size(0) * 100))
        return nb_errors

    def train_and_test(self, train_input, train_target,
                    test_input, test_target, 
                    test_classes=None, train_classes = None,
                    learning_rate = 1e-1, mini_batch_size=50, nb_epochs=25,
                    aux_loss_weight = 0.2, main_criterion = nn.CrossEntropyLoss(), 
                    aux_criterion_1 = None, aux_criterion_2 = None, optimizer=None,
                    k = 10):
        #This trains, computes training accuracy, then computes testing accuracy,
        # and finally does k-fold cross accuracy on the training set 
        if len(test_input)%k != 0:
            print("Your k should divide the length of the test set!")
            return None

        #First train:
        self.train_model(train_input, train_target, train_classes, 
                        learning_rate, mini_batch_size, nb_epochs,
                        aux_loss_weight, main_criterion, 
                        aux_criterion_1, aux_criterion_2, optimizer)
        #train accuracy
        print("Training accuracy of {:.02f}%".format( 
                 (1 - (self.compute_nb_errors(train_input, train_target) / train_input.size(0))) * 100))

        print("Now testing...")
        print("Testing accuracy of {:.02f}%".format( 
                (1 - (self.compute_nb_errors(test_input, test_target) / test_input.size(0))) * 100))

        #k-fold (for standard deviation of testing)
        k_acc_mean, k_acc_std = self.k_fold_cross(k, test_input, test_target)
        print("{:d}-fold accuracy mean on test_set is {:.02f}% with {:f} standard deviation".format(
                k, 
                k_acc_mean * 100,
                k_acc_std))

    def k_fold_cross(self, k, test_input, test_target):
        split_size = int(test_input.size(0)/k)
        inputs = torch.split(test_input, split_size , dim=0)
        targets = torch.split(test_target, split_size, dim=0)
        nb_errors = []
        accuracies = []
        mean_acc = 0
        for i in range(k):
            nb_errors += [self.compute_nb_errors(inputs[i], targets[i])]
            accuracies += [self.compute_nb_errors(inputs[i], targets[i]) / split_size]

        return 1 - np.mean(accuracies), np.std(accuracies)

