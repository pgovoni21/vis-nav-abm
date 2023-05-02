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
    def __init__(self, arch, params=None, activ='relu', dt=100):
        super().__init__()

        # construct architecture of NN
        input_size, hidden_size, output_size = arch
        self.hidden_size = hidden_size
        
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.layers = [self.i2h, self.h2h, self.h2o]

        # disable autograd computation since we're not computing gradient
        for layer in self.layers:
            layer.requires_grad_(False)

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

        # # set parameter noise values ## uncomment for noise
        # sigma_in = 0.1
        # sigma_rec = 0.5
        # self.std_noise_in = np.sqrt(2 / self.alpha) * sigma_in
        # self.std_noise_rec = np.sqrt(2 / self.alpha) * sigma_rec


    def forward(self, input, hidden):
        """Propagate input through RNN
        
        Inputs:
            input: 1D array of shape (input_size)
            hidden: tensor of shape (hidden_size)
        
        Outputs:
            hidden: tensor of shape (hidden_size)
            actions: 1D array of shape (output_shape)
        """            
        # initialize hidden state activity (t = 0 for sim run, saved in Agent instance)
        if hidden is None:
            hidden = torch.zeros(self.hidden_size)

        # converts np.array to torch.tensor + adds 1 dimension --> size [4] becomes [1, 4]
        input = torch.from_numpy(input).float().unsqueeze(0)

        # carry out recurrent calculations according to model formulation (with or without noisy activations)
        # input += torch.randn(input.shape) * self.std_noise_in ## uncomment for noise
        x = self.activ(self.i2h(input) + self.h2h(hidden))
        x = hidden * (1 - self.alpha) + x * self.alpha 
        # x += torch.randn(hidden.shape) * self.std_noise_rec ## uncomment for noise

        # pull current hidden activity for next time step
        hidden = x 

        # reduce to action dimension + pass through Tanh function (bounding to -1:1)
        x = self.h2o(x)
        x = torch.tanh(x)

        # convert to np.array (requires one strip for NN_output_size>1 // two strips for NN_output_size=1)
        # output = x.detach().numpy()[0]
        output = x.detach().numpy()[0][0]

        return output, hidden