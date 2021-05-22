import torch
import math

torch.set_grad_enabled(False)

class Module(object):
    def forward(self , *input):
        raise  NotImplementedError

    def backward(self , *gradwrtoutput):
        raise  NotImplementedError

    def param(self):
        return []



class Linear(Module): # right now this assumes that mini_batch_size = 1 
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        # Use pytorch style weight initialization
        dist = 1. / math.sqrt(self.dim_out)
        self.weights = torch.empty(dim_in, dim_out).uniform_(-dist, dist)

        self.bias = torch.empty(dim_in, dim_out).uniform_(-dist, dist)

    def forward(self, *input):
        output = input.mm(self.weights) + self.bias
        return output

    def backward(self , *gradwrtoutput):
        raise  NotImplementedError

# The following is taken from Problem set 3
# def backward_pass(w1, b1, #current weights and biases of the network
#                   t, # target vector
#                   x, s1, x1, # output of layer
#                   dl_dw1, dl_db1): # tensors used to stor the sumulated sums of the gradient
#     x0 = x
#     dl_dx2 = dloss(x2, t)
#     dl_ds2 = dsigma(s2) * dl_dx2
#     dl_dx1 = w2.t().mv(dl_ds2)
#     dl_ds1 = dsigma(s1) * dl_dx1

#     dl_dw2.add_(dl_ds2.view(-1, 1).mm(x1.view(1, -1)))
#     dl_db2.add_(dl_ds2)
#     dl_dw1.add_(dl_ds1.view(-1, 1).mm(x0.view(1, -1)))
#     dl_db1.add_(dl_ds1)

    def param(self):
        return [self.weights, self.bias]


class Sequential(Module):
    def __init__(self, tuple_of_layers):

        self.layers = tuple_of_layers


    def forward(self, *input):
        x = input
        for layer in self.layers:
            x = layer(x)

        return x

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return [self.layers]


class Tanh(Module):

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

class ReLu(Module):

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []