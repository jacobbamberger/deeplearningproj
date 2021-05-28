# import torch
import math
from torch import empty as t_empty


# The chain rule states that (f g)' = g' * f'(g())

class Module(object):
    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


class Linear(Module):
    def __init__(self, dim_in, dim_out, weight_init="uniform"):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        ## Weight initialization:
        
        ## Uniform initialization:
        if weight_init == "uniform":
            dist = 1. / math.sqrt(self.dim_out)
            self.weights = t_empty(dim_out, dim_in).uniform_(-dist, dist)
            self.bias = t_empty(dim_out).uniform_(-dist, dist)
        else:
        ## Normal distribution initialization:
            self.weights = torch.nn.init.normal_(torch.empty(dim_out, dim_in), mean=0.0, std=1.0)
            self.bias = torch.nn.init.normal_(torch.empty(dim_out), mean=0.0, std=1.0)

        # this is where we store this layer's gradient:
        self.weights_grad_accum = t_empty(self.dim_out, self.dim_in).fill_(0)
        self.bias_grad_accum = t_empty(self.dim_out).fill_(0)

    def forward(self, input):
        if len(input.shape) <2:
          self.current_input = input.unsqueeze(0)
        else:
            self.current_input = input
        output = self.current_input.mm(self.weights.t()) + self.bias
        return output

    def backward(self, gradwrtoutput):
        x = self.current_input  # We store this because it is used in backwards

        dl_dy = gradwrtoutput
        dl_dx = self.weights.t().mm(dl_dy)

        dl_dw = dl_dy.mm(x)
        dl_db = dl_dy
        self.weights_grad_accum = self.weights_grad_accum.add(dl_dw)
        self.bias_grad_accum = self.bias_grad_accum.add(dl_db.sum(dim=1))

        return dl_dx

    def param(self):
        return [(self.weights, self.weights_grad_accum), (self.bias, self.bias_grad_accum)]

    def SGD_step(self, learning_rate):
        self.weights = self.weights.sub(learning_rate * self.weights_grad_accum)
        self.bias = self.bias.sub(learning_rate * self.bias_grad_accum)

        # reinitialization:
        self.weights_grad_accum = t_empty(self.dim_out, self.dim_in).fill_(0)
        self.bias_grad_accum = t_empty(self.dim_out).fill_(0)


class Sequential(Module):
    def __init__(self,
                 tuple_of_layers):  # should we have an accum_grad for sequential too, or should de SGD optimize each one etc...
        self.layers = tuple_of_layers

    def forward(self, input):
        x = input
        for layer in self.layers:
            # print(x)
            x = layer.forward(x)
        return x

    def backward(self, gradwrtoutput):
        current_grad = gradwrtoutput
        for i in range(len(self.layers)):
            # print(current_grad)
            current_grad = self.layers[len(self.layers) - 1 - i].backward(current_grad)
        # for layer in self.layers:
        #    print(current_grad)
        #    current_grad = layer.backward(current_grad)
        return current_grad

    def param(self):
        params = []
        for layer in self.layers:
            params += layer.param()
        return params

    def SGD_step(self, learning_rate):
        for layer in self.layers:
            layer.SGD_step(learning_rate)


class ReLu(Module):

    def forward(self, input):
        self.current_output = input.max(t_empty(input.shape).fill_(0))
        return self.current_output

    def backward(self, gradwrtoutput):
        return (self.current_output > 0).t() * gradwrtoutput

    def param(self):
        return []

    def SGD_step(self, learning_rate):
        return None


class Tanh(Module):  # Tanh doesn't have any params so we don't need to update anything in gradient
    def forward(self, input):  # Do i need to initialize anything?? s is the input and x is output of activation
        self.current_input = input
        return input.tanh()

    def backward(self, gradwrtoutput):
        dsigma = (1 - self.current_input.tanh().pow(2))
        return dsigma.t()*(gradwrtoutput)

    def param(self):
        return []

    def SGD_step(self, learning_rate):
        return None


class Sigmoid(Module):
    def forward(self, input):
        argument = -input
        self.current_output = 1 / (1 + argument.exp())
        return self.current_output

    def backward(self, gradwrtoutput):
        dsigma = self.current_output * (1 - self.current_output)
        return dsigma.t() * gradwrtoutput

    def param(self):
        return []

    def SGD_step(self, learning_rate):
        return None


class MSE(Module):

    def forward(self, prediction, target):  # computes the error, still assumes batch_size=1

        if len(target.shape) < 2:
            target_to_use  = target.unsqueeze(1)
        else:
            target_to_use = target

        return (prediction - target_to_use).pow(2).sum()

    def backward(self, prediction,
                 target):  # computes the gradient of the loss function with respect to the predictions

        if len(target.shape) < 2:
            target_to_use = target.unsqueeze(1)
        else:
            target_to_use = target

        return 2 * (prediction - target_to_use).t()
