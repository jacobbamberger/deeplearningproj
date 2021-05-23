import torch
import math

torch.set_grad_enabled(False)

#The chain rule states that (f g)' = g' * f'(g())

class Module(object):
    def forward(self , *input):
        raise  NotImplementedError

    def backward(self , *gradwrtoutput):
        raise  NotImplementedError

    def param(self):
        return []



class Linear(Module):
    def __init__(self, dim_in, dim_out): #right now batch size is assumed to be 1.
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        # Use pytorch style weight initialization
        dist = 1. / math.sqrt(self.dim_out)
        self.weights = torch.empty(dim_out, dim_in).uniform_(-dist, dist)
        self.bias = torch.empty(dim_out).uniform_(-dist, dist)

        #this is where we store this layer's gradient:
        self.weights_grad_accum = torch.zeros(dim_out, dim_in)
        self.bias_grad_accum = torch.zeros(dim_out)

    def forward(self, input): # *input):  #input has to be of size..? Why was there the *? is it for bigger batch size??
        self.current_input = input
        output = self.weights.mv(input) + self.bias
        self.current_output = output # minibatchsize = 1 is necessary for this?
        return output

    def backward(self , gradwrtoutput): # *gradwrtouput is assumed to be just one vector.
        y = self.current_output # or input??
        x = self.current_input
        
        dl_dy = gradwrtoutput
        dl_dx = self.weights.t().mv(dl_dy)

        dl_dw = dl_dy.view(-1, 1).mm(x.view(1, -1)) 
        dl_db = dl_dy
        self.weights_grad_accum = self.weights_grad_accum.add(dl_dw)
        self.bias_grad_accum = self.bias_grad_accum.add(dl_db)

        #raise  NotImplementedError #what should be returned? dl_dx, or dl_dw, or both?
        return dl_dx


    def param(self):
        return [self.weights, self.bias]


class Sequential(Module):
    def __init__(self, tuple_of_layers): #should we have an accum_grad for sequential too, or should de SGD optimize each one etc...
        self.layers = tuple_of_layers


    def forward(self, input):
        x = input
        for layer in self.layers:
            print(x)
            x = layer.forward(x)
        return x

    def backward(self, gradwrtoutput):
        current_grad = gradwrtoutput
        for layer in self.layers:
            print(current_grad)
            current_grad = layer.backward(current_grad)
        return current_grad


    def param(self):
        return [layer.param for layer in self.layers]


class Tanh(Module):

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

class ReLu(Module):

    def forward(self, input):
        return torch.max(torch.stack((input, torch.zeros(input.shape))), dim=0)[0]

    def backward(self, gradwrtoutput):
        return torch.max(torch.stack((gradwrtoutput, torch.zeros(gradwrtoutput.shape))), dim=0)[0]

    def param(self):
        return []

class MSE(Module):

    def forward(prediction, target): #computes the error
        raise NotImplementedError

    def backward(prediction, target):#computes the gradient of the loss function with respect to the predictions
        raise NotImplementedError
