import torch
import math


######################################################################

def generate_disc_set(nb):
    input = torch.empty(nb, 2).uniform_(-1, 1)
    target = input.pow(2).sum(1).sub(2 / math.pi).sign().add(1).div(2).long()
    return input, target

train_input, train_target = generate_disc_set(1000)
test_input, test_target = generate_disc_set(1000)