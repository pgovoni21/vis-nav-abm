import torch
import torch.nn as nn
import numpy as np

class RNN(nn.Module):
    """
    RNN model
    Euler-discretized dynamical system

    Types:
        Static:
            Traditional NN parameters (weights/biases) static in simulation time + updated in evolutionary time
            Time constant static in sim + evolutionary time
        Plastic:
            Hebbian trace plastic in simulation time + zeroed at start (non-evolved)
            Other parameters (weights/biases / plasticity coefficients / learning constants) static in sim time + evolved
            Time constant static in sim + evolutionary time

    Parameters:
        architecture: 3-int tuple : input / hidden / output size (# neurons)
        dt: discretization time step *in ms* <-- check
            If None, equals time constant tau

    forward() eqns from + layout inspired by: https://colab.research.google.com/github/gyyang/nn-brain/blob/master/RNN_tutorial.ipynb
    """
    def __init__(self, arch, RNN_type, rule='hebb', activ='relu', dt=100, init='xavier', copy_network=None, var=0.05):
        super().__init__()

        # construct architecture of NN
        input_size, hidden_size, output_size = arch
        self.hidden_size = hidden_size
        
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

        self.RNN_type = RNN_type
        self.rule = rule
        if RNN_type == 'static-Yang' or RNN_type == 'static-Yang-noise':
            self.h2h = nn.Linear(hidden_size, output_size)
        elif RNN_type == 'static-Miconi':
            self.w = nn.Parameter( .01 * torch.rand(hidden_size, hidden_size) )
        elif RNN_type == 'plastic-Miconi':
            self.w = nn.Parameter( .01 * torch.rand(hidden_size, hidden_size) )
            self.plas = nn.Parameter( .01 * torch.rand(hidden_size, hidden_size) )
            self.learn = nn.Parameter( .01 * torch.ones(1) )
        else: raise ValueError(f'Invalid RNN type: {RNN_type}')

        # set activation function
        if activ == 'relu':
            self.activ = torch.relu
        elif activ == 'tanh':
            self.activ = torch.tanh
        else:
            raise ValueError(f'Invalid activation function: {activ}')

        # set time constant
        self.tau = 100
        self.alpha = dt / self.tau # default --> alpha = 1

        # set parameter noise values
        sigma_in = 0.1
        sigma_rec = 0.5
        self.std_noise_in = np.sqrt(2 / self.alpha) * sigma_in
        self.std_noise_rec = np.sqrt(2 / self.alpha) * sigma_rec

        # set params
        if init: # gen = 0 --> initialize params from specified scheme
            self.initialize_params(init, activ)

        elif copy_network: # gen > 0 --> mutate child NN from parents
            self.mutate(copy_network, var)


    def initialize_params(self, init, activ):

        if init == 'normal':
            for p in self.parameters():
                nn.init.normal_(p, mean=0, std=1)
            return # does not run for-loop below
        
        elif init == 'identity':
            for p in self.parameters():
                if p.dim() > 1: 
                    nn.init.eye_(p)
        
        elif init == 'xavier':
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p, gain = nn.init.calculate_gain(activ))

        elif init == 'sparse':
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.sparse_(p, sparsity = 0.1, std=0.01)

        # above initialization schemes cannot be performed on tensors of 1 dimension -> pull from normal
        for p in self.parameters():
            if p.dim() == 1:
                nn.init.normal_(p, mean=0, std=1)


    def mutate(self, copy_network, var):
        # iterate over copied (parent) layers + self (child) layers
        for parent_param, self_param in zip(copy_network.parameters(), self.parameters()):
            
            # shift by perturbation chosen from normal/gaussian dist of mean = 0 / var = 1
            noise = var * torch.randn(parent_param.shape) # rescale noise to desired degree
            self_param.data = parent_param + noise


    def forward(self, input, hidden, hebb=None):
        """Propagate input through RNN
        
        Inputs:
            input: 1D array of shape (input_size)
            hidden: tensor of shape (hidden_size)
        
        Outputs:
            hidden: tensor of shape (hidden_size)
            actions: 1D array of shape (output_shape)
        """
        # run calculations without storing parameters since we're not computing gradient
        with torch.no_grad():
            
            # initialize hidden state activity (t = 0 for sim run)
            if hidden is None:
                hidden, hebb = self.initialize_activities()

            # converts np.array to torch.tensor + adds 1 dimension --> size [4] becomes [1, 4]
            input = torch.from_numpy(input).float().unsqueeze(0)

            # carry out recurrent calculations according to model formulation
            # pull current hidden (+ hebbian) activity post calculation (saved in Agent) for next time step
            if self.RNN_type == 'static-Yang':
                x = self.activ(self.i2h(input) + self.h2h(hidden))
                x = hidden * (1 - self.alpha) + x * self.alpha 
                hidden = x 

            elif self.RNN_type == 'static-Yang-noise':
                input = input + torch.randn(input.shape) * self.std_noise_in
                x = self.activ(self.i2h(input) + self.h2h(hidden))
                x = hidden * (1 - self.alpha) + x * self.alpha
                x = x + torch.randn(hidden.shape) * self.std_noise_in 
                hidden = x 

            elif self.RNN_type == 'static-Miconi':
                x = self.activ(self.i2h(input) + hidden.mm(self.w))
                hidden = x
            
            elif self.RNN_type == 'plastic-Miconi':
                x = self.activ(self.i2h(input) + hidden.mm(self.w + torch.mul(self.plas, hebb)))

                if self.rule == 'hebb':
                    hebb = (1 - self.learn) * hebb + self.learn * torch.bmm(hidden.unsqueeze(2), x.unsqueeze(1))[0]
                elif self.rule == 'oja':
                    hebb = hebb + self.learn * torch.mul((hidden[0].unsqueeze(1) - torch.mul(hebb , x[0].unsqueeze(0))) , x[0].unsqueeze(0)) 
                else:
                    raise ValueError(f'Invalid learning rule: {self.RNN_type}')
                
                hidden = x

            # reduce to action dimension + pass through Tanh function (bounding to -1:1)
            x = self.h2o(x)
            x = torch.tanh(x)

            # convert to np.array (requires one strip for NN_output_size>1 // two strips for NN_output_size=1)
            # output = x.detach().numpy()[0]
            output = x.detach().numpy()[0][0]

        return output, hidden, hebb


    def initialize_activities(self):

        if self.RNN_type == 'static-Yang' or self.RNN_type == 'static-Yang-noise':
            hidden = torch.zeros(self.hidden_size)
            hebb = None

        elif self.RNN_type == 'static-Miconi':
            hidden = torch.zeros(self.hidden_size, self.hidden_size)
            hebb = None

        elif self.RNN_type == 'plastic-Miconi':
            hidden = torch.zeros(self.hidden_size, self.hidden_size)
            hebb = torch.zeros(self.hidden_size, self.hidden_size)

        return hidden, hebb
