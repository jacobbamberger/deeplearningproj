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



class Linear(Module):
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