

from torch import optim
from torch import nn


def train_model(model, train_input, train_target, train_classes = None, learning_rate = 1e-1, mini_batch_size=50, nb_epochs=25,
                aux_loss_weight = 0.2):

    if train_classes is None:
        use_aux_loss = False
    else:
        use_aux_loss = True

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

        #print(e, acc_loss)

def compute_nb_errors(model, input, target, mini_batch_size):
    nb_errors = 0

    for b in range(0, input.size(0), mini_batch_size):
        output, *classes = model(input.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.max(1)

        for k in range(mini_batch_size):
            if target[b + k] != predicted_classes[k]:
                nb_errors = nb_errors + 1

    return nb_errors