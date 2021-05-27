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

    def train(self, train_input, train_target, nb_epochs, batch_size, learning_rate, loss=None):
        nb_samples = train_input.size(0)
        if loss is None:
            loss = MSE()
        if train_input.size(0)%batch_size !=0:
            print("Bactch size should divid length of training data.")
        for e in range(nb_epochs):
            loss_acc=0
            for b in range(0, nb_samples, batch_size):
                for i in range(batch_size):
                    prediction = self.forward(train_input[b+i])
                    loss_acc += loss.forward(prediction, train_target[b+i])
                    self.backward(loss.backward(prediction, train_target[b+i]))
                self.SGD_step(learning_rate)
            if e%5 == 0:
                print('epoch nb: ', e, 'loss: ', loss_acc)

    def compute_nb_errors(self, data_input, data_target, batch_size=1):
        tot_right = 0

        for b in range(0, data_input.size(0), batch_size):
            for i in range(batch_size):
                output = self.forward(data_input[b+i])
                if output <0.0 and  data_target[b+i]==0: #this is specific to the toy dataset.
                    tot_right+=1
                elif output >=0.0 and  data_target[b+i]==1:
                    tot_right+=1

        print("Accuracy on the data set is: ", tot_right/data_input.size(0))
        return tot_right/data_input.size(0)



class Linear(Module):
    def __init__(self, dim_in, dim_out): #right now batch size is assumed to be 1.
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        # Use pytorch style weight initialization
        #dist = 1. / math.sqrt(self.dim_out)
        #self.weights = torch.empty(dim_out, dim_in).uniform_(-dist, dist)
        #self.bias = torch.empty(dim_out).uniform_(-dist, dist)
        self.weights = torch.nn.init.normal_(torch.empty(dim_out, dim_in), mean=0.0, std=1.0)
        self.bias = torch.nn.init.normal_(torch.empty(dim_out), mean=0.0, std=1.0)

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

