
import dlc_practical_prologue as prologue
import Networks
import training
import torch
import time

# Load and preprocess data
nb = 1000 # Number of test and train samples is 1000 as stated in the exercise
mini_batch_size = 50
nb_runs = 25

train_input, train_target, train_classes, test_input, test_target, test_classes =\
    prologue.generate_pair_sets(nb)


# normalize input data
mu, std = train_input.mean(), train_input.std()
train_input.sub_(mu).div_(std)
test_input.sub_(mu).div_(std)

nb_hidden_siamese = 200
nb_hidden_full = 200


train_error = torch.empty(nb_runs, 1)
test_error = torch.empty(nb_runs, 1)
elapsed_time = torch.empty(nb_runs, 1)




for run in range(nb_runs):
    model = Networks.FullNet(nb_hidden_siamese, nb_hidden_full)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    tic = time.time()
    training.train_model(model, train_input, train_target, train_classes = train_classes, learning_rate = 1e-1, mini_batch_size=50, nb_epochs=25,
                aux_loss_weight = 0.25)
    elapsed_time[run] = time.time()- tic

    nb_train_errors = training.compute_nb_errors(model, train_input, train_target, mini_batch_size)
    nb_test_errors = training.compute_nb_errors(model, test_input, test_target, mini_batch_size)
    train_error[run] = nb_train_errors/nb
    test_error[run] = nb_test_errors/nb
    print(train_error[run], test_error[run])

train_error_mean, train_error_std, = train_error.mean(), train_error.std()
test_error_mean, test_error_std = test_error.mean(), test_error.std()
train_time = elapsed_time.mean()


print( train_error_mean, train_error_std)
print(test_error_mean, test_error_std)
print(train_time)