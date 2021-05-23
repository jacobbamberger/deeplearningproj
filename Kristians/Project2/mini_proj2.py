import torch
import math

torch.set_grad_enabled(False)


class myLinear():
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        # Use pytorch style weight initialization
        dist = 1. / math.sqrt(self.dim_out)
        self.weights = torch.empty(dim_in, dim_out).uniform_(-dist, dist)   #dim_out dim_in

        self.bias = torch.empty(1, dim_out).uniform_(-dist, dist)  #Shouldn't it be just 1D vector?  dim_out

        
        self.dl_db = torch.empty(1, dim_out) #gradient for updating biases   INITIALIZE TO ZERO
        self.dl_dw = torch.empty(dim_in, dim_out) #gradient for updating weights
        
    def forward(self, x0):  #Input should be x0 and output is s
        self.x0 = x0   #can I initialize self.x here? We need to save it for backward pass
        s = self.weights.mv(self.x0) + self.bias  #should be mv
        return s

    def backward(self , dl_ds): #For backward need x0 - input, dl_ds which is the dsigma*dl_dx (for last layer dl_dx=dloss)
        self.dl_db.add_(dl_ds)  # just add the gradient that is computed through previous dl_dx and activation function (or w.r.t to loss function)
        self.dl_dw.add_(dl_ds.view(-1, 1).mm(self.x0.view(1, -1)))  #making a matrix and transposing the dl_ds2 so that we can do mm
        output = self.weights.t().mv(dl_ds) #computed dl_dx which should be the input for calculation of dl_ds in activation layers backward pass
        return output

    def param(self):
        return [self.weights, self.bias, self.x, self.dl_db, self.dl_dw]
      
    
class myTanh(): #Tanh doesn't have any params so we don't need to update anything in gradient 
    def forward(self,s):    #Do i need to initialize anything?? s is the input and x is output of activation 
        self.s = s   #need to save s to get dsigma and dl_ds
        x = s.tanh()
        return x

    def backward(self,dl_dx):
        self.dsigma = (1-self.s.tanh().pow(2))
        return self.dsigma*dl_dx  #returns dl_ds, which depends on dsigma(deriv of sigma) and dl_dx (last layer it's just dloss)

    def param(self):
        return [self.s, self.dsigma]

class mySequential():     #Here probably need something like tuple of layers and then forward is easy-peasy, backwards would be a pain in the ass, but essentially just needs previous dl_dwhatever
    def __init__(self, tuple_of_layers):

        self.layers = tuple_of_layers


    def forward(self, *input):   #should work
        x = input
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def backward(self, dloss):  #Here need to input the dloss from loss function and calculate next dl_whatever w.r.t to input dl_whatever-1
        for layer in self.layers:   #GO BACKWARDS START FROM LAST LAYER
            dloss = layer.backward(dloss)  #essentially calculates loss w.r.t to input loss, should work for both activation and linear layers 
        return dloss

    def param(self):
        return [self.layers]
    
class myLossMSE(): # The MSE is just the square and sum of the target-real, right? Or am I stupid? Also, should we divide by the number of elements 1/N? In the end doesn't really make a difference I think.
    def forward(self,predicted,target):
        self.p = predicted
        self.t = target
        return (self.p - self.t).pow(2).sum()
    def backward(self):
        return 2 * (self.p - self.t)
    
class ReLu():
    def forward(self, input):
        return torch.max(torch.stack((input, torch.zeros(input.shape))), dim=0)[0]

    def backward(self, gradwrtoutput):
        return torch.max(torch.stack((gradwrtoutput, torch.zeros(gradwrtoutput.shape))), dim=0)[0]

