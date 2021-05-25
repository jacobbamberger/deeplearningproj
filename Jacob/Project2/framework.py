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
        self.weights_grad_accum = torch.zeros(self.dim_out, self.dim_in)
        self.bias_grad_accum = torch.zeros(self.dim_out)

    def forward(self, input): # *input):  #input has to be of size..? Why was there the *? is it for bigger batch size??
        self.current_input = input
        output = self.weights.mv(input) + self.bias
        self.current_output = output # minibatchsize = 1 is necessary for this?
        return output

    def backward(self, gradwrtoutput): # *gradwrtouput is assumed to be just one vector.
        #y = self.current_output # or input??
        x = self.current_input
        
        dl_dy = gradwrtoutput
        # print(self.weights.shape)
        # print(dl_dy.shape)
        dl_dx = self.weights.t().mv(dl_dy)

        dl_dw = dl_dy.view(-1, 1).mm(x.view(1, -1)) 
        dl_db = dl_dy
        self.weights_grad_accum = self.weights_grad_accum.add(dl_dw)
        self.bias_grad_accum = self.bias_grad_accum.add(dl_db)

        #raise  NotImplementedError #what should be returned? dl_dx, or dl_dw, or both?
        return dl_dx


    def param(self):
        return [self.weights, self.bias]

    def SGD_step(self, learning_rate):
        self.weights = self.weights.sub(learning_rate * self.weights_grad_accum)
        self.bias = self.bias.sub(learning_rate * self.bias_grad_accum)
        
        # reinitialization:
        self.weights_grad_accum = torch.zeros(self.dim_out, self.dim_in)
        self.bias_grad_accum = torch.zeros(self.dim_out)


class Sequential(Module):
    def __init__(self, tuple_of_layers): #should we have an accum_grad for sequential too, or should de SGD optimize each one etc...
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
            current_grad = self.layers[len(self.layers)-1-i].backward(current_grad)
        #for layer in self.layers:
        #    print(current_grad)
        #    current_grad = layer.backward(current_grad)
        return current_grad

    def param(self):
        return [layer.param for layer in self.layers]

    def SGD_step(self, learning_rate):
        for layer in self.layers:
            layer.SGD_step(learning_rate)


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

    def SGD_step(self, learning_rate):
        return 0

class Tanh(Module): #Tanh doesn't have any params so we don't need to update anything in gradient 
    def forward(self,s):    #Do i need to initialize anything?? s is the input and x is output of activation 
        self.s = s   #need to save s to get dsigma and dl_ds
        x = s.tanh()
        return x

    def backward(self,dl_dx):
        self.dsigma = (1-self.s.tanh().pow(2))
        return self.dsigma*dl_dx  #returns dl_ds, which depends on dsigma(deriv of sigma) and dl_dx (last layer it's just dloss)

    def param(self):
        return [self.s, self.dsigma]

    def SGD_step(self, learning_rate):
        return 0

class MSE(Module):

    def forward(self, prediction, target): #computes the error, still assumes batch_size=1
        return (prediction - target)**2

    def backward(self, prediction, target):#computes the gradient of the loss function with respect to the predictions
        return 2*(prediction-target)

