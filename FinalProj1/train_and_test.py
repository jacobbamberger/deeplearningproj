

from torch import optim
from torch import nn
import torch
import time


def evaluate_model(model, train_input, train_target,test_input, test_target, mini_batch_size=50, nb_runs=25,
                   train_classes=None, learning_rate=1e-1, nb_epochs=25, use_aux_loss=True, aux_loss_weight=0.2):
    train_error = torch.empty(nb_runs, 1)
    test_error = torch.empty(nb_runs, 1)
    elapsed_time = torch.empty(nb_runs, 1)
    nb = train_input.size()[0]
    for run in range(nb_runs):
        model.apply(weight_reset)  #This makes sure that we actually start from the same weight initialization
        # for each run.

        tic = time.time()
        train_model(model, train_input, train_target, train_classes=train_classes, learning_rate=learning_rate,
                         mini_batch_size=mini_batch_size, nb_epochs=nb_epochs,use_aux_loss=use_aux_loss,
                         aux_loss_weight=aux_loss_weight)
        elapsed_time[run] = time.time() - tic

        nb_train_errors = compute_nb_errors(model, train_input, train_target, mini_batch_size)
        nb_test_errors = compute_nb_errors(model, test_input, test_target, mini_batch_size)
        train_error[run] = nb_train_errors / nb
        test_error[run] = nb_test_errors / nb

    return train_error.mean(), train_error.std(), test_error.mean(), test_error.std(), elapsed_time.mean()



def train_model(model, train_input, train_target, train_classes=None, learning_rate=1e-1, mini_batch_size=50,
                nb_epochs=25, use_aux_loss=True, aux_loss_weight=0.2):

    if  use_aux_loss and train_classes is None:
        print('No training classes provided. Turning auxilliary loss off.')
        use_aux_loss = False


    main_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    if use_aux_loss:
        aux_criterion_1 = nn.CrossEntropyLoss()
        aux_criterion_2 = nn.CrossEntropyLoss()

    for e in range(nb_epochs):
        acc_loss = 0

        for b in range(0, train_input.size(0), mini_batch_size):

            batch_input = train_input.narrow(0, b, mini_batch_size)
            output, *classes = model(batch_input)

            if classes[0] is None and use_aux_loss:
                print('Model does not predict classes. Turning auxilliary loss off.')
                use_aux_loss = False

            main_loss = main_criterion(output, train_target.narrow(0, b, mini_batch_size))
            if use_aux_loss:
                classes_1, classes_2 = classes
                aux_loss_1 = aux_criterion_1(classes_1, train_classes.narrow(0, b, mini_batch_size)[:, 0])
                aux_loss_2 = aux_criterion_2(classes_2, train_classes.narrow(0, b, mini_batch_size)[:, 1])
                total_loss = main_loss + aux_loss_weight * (aux_loss_1 + aux_loss_2)
            else:
                total_loss = main_loss

            acc_loss = acc_loss + total_loss.item()
            model.zero_grad()
            total_loss.backward()
            optimizer.step()



def compute_nb_errors(model, input, target, mini_batch_size):
    nb_errors = 0

    for b in range(0, input.size(0), mini_batch_size):
        output, *classes = model(input.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.max(1)

        for k in range(mini_batch_size):
            if target[b + k] != predicted_classes[k]:
                nb_errors = nb_errors + 1

    return nb_errors

# Function to reset the weights, so that we do not have to reinitialize the model every time.
# We only use the three listed module/layer types that actually have parameters. (Checked on the current pytorch implementation.)
def weight_reset(m):

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()