import dlc_practical_prologue as prologue
import Networks
import train_and_test
import torch
import plotting

# Load and preprocess data
nb = 1000 # Number of test and train samples is 1000 as stated in the exercise
mini_batch_size = 50
nb_runs = 5 # Only 5 runs, so this script is reasonably fast

train_input, train_target, train_classes, test_input, test_target, test_classes =\
    prologue.generate_pair_sets(nb)


# normalize input data
mu, std = train_input.mean(), train_input.std()
train_input.sub_(mu).div_(std)
test_input.sub_(mu).div_(std)

# Fix these for all FullNet architectures
nb_hidden_siamese = 200
nb_hidden_full = 200


labels = ['LinearNet','ConvNet', 'ConvNet, size AuxNet', 'AuxNet', 'ParallelNet', 'SiameseNet', 'SiameseNet, no softmax', 'NaiveNet']

models = [ Networks.LinearNet(),
          Networks.ConvNet(),
          Networks.ConvNet(20),
          Networks.AuxNet(),
          Networks.FullNet(nb_hidden_siamese,nb_hidden_full, subnet_type ='parallel'),
          Networks.FullNet(nb_hidden_siamese,nb_hidden_full, subnet_type ='siamese'),
          Networks.FullNet(nb_hidden_siamese, nb_hidden_full, use_softmax=False),
          Networks.FullNet(nb_hidden_siamese, nb_hidden_full, subnet_type = 'naive')
           ]
n_models = len(models)

aux_loss_weights = [0, 0, 0, 2, 0.7, 0.7, 0.7, 0.7]
using_aux_loss = [False, False, False, True, True,True, True, True]
train_error_means = torch.empty(n_models, )
train_error_stds = torch.empty(n_models, )
test_error_means = torch.empty(n_models, )
test_error_stds = torch.empty(n_models, )
avg_train_time = torch.empty(n_models, )

for m_index, model in enumerate(models):
    print('Evaluating model ', m_index+1,',', labels[m_index], ':')
    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters', pytorch_total_params)
    train_error_means[m_index], train_error_stds[m_index],test_error_means[m_index], test_error_stds[m_index], \
    avg_train_time[m_index] = train_and_test.evaluate_model(model, train_input, train_target, test_input, test_target,
                                                            mini_batch_size=mini_batch_size, nb_runs=nb_runs,
                                                            train_classes=train_classes, use_aux_loss=using_aux_loss[m_index],
                                                            aux_loss_weight=aux_loss_weights[m_index])

    print('Train error', "{:.5}".format(train_error_means[m_index].item()), ' +/-', "{:.3}".format(train_error_stds[m_index].item()))
    print('Test error', "{:.5}".format(test_error_means[m_index].item()), ' +/-', "{:.3}".format(test_error_stds[m_index].item()))
    print('Average model train time: ', "{:.3}".format(avg_train_time[m_index].item()), 's')
    print()

