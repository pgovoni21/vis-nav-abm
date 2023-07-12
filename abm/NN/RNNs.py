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

        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm_h = nn.InstanceNorm1d(hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.layers = [self.i2h, self.h2h, self.norm_h, self.h2o]
        # self.i2h = nn.Linear(input_size, hidden_size)
        # self.h2h = nn.Linear(hidden_size, hidden_size)
        # self.h2o = nn.Linear(hidden_size, output_size)
        # self.layers = [self.i2h, self.h2h, self.h2o]

        # disable autograd computation since we're not computing gradient
        for layer in self.layers:
            layer.requires_grad_(False)      

        # set activation function
        if activ == 'relu':
            self.activ = torch.relu
        elif activ == 'tanh':
            self.activ = torch.tanh
        elif activ == 'silu':
            self.activ = torch.nn.SiLU()
        else:
            raise ValueError(f'Invalid activation function: {activ}')

        # # set time constant ## uncomment for time constant
        # tau = 100
        # self.alpha = dt / tau # default --> alpha = 1

        # # construct param vector if not provided by EA
        # if params is None:
        #     # num_weights = hidden_size * (input_size + hidden_size + output_size)
        #     # num_biases = 2*hidden_size + output_size
        #     # params = np.random.randn(num_weights + num_biases) * np.sqrt(2/hidden_size)
        #     num_params = sum(p.numel() for p in self.parameters())
        #     params = np.random.randn(num_params) * np.sqrt(2/hidden_size)

        # assign w+b parameters
        if params is not None:
            self.assign_params(params)

    def assign_params(self, param_vector):

        # set params by chunking according to model architecture
        param_num = 0
        for l in self.layers:
            # for param in [l.weight, l.bias]:
            #     chunk = param_vector[param_num : param_num + param.numel()]
            #     param.data = chunk.reshape(param.shape)
            #     param_num += param.numel()
            if l.weight is not None:
                chunk = param_vector[param_num : param_num + l.weight.numel()]

                reshaped_chunk = chunk.reshape(l.weight.shape)
                torched_chunk = torch.from_numpy(reshaped_chunk).float()
                l.weight.data = torched_chunk

                param_num += l.weight.numel()
            if l.bias is not None:
                chunk = param_vector[param_num : param_num + l.bias.numel()]

                reshaped_chunk = chunk.reshape(l.bias.shape)
                torched_chunk = torch.from_numpy(reshaped_chunk).float()
                l.bias.data = torched_chunk

                param_num += l.bias.numel()


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
            hidden = torch.zeros(self.hidden_size).unsqueeze(0)

        # converts np.array to torch.tensor + adds 1 dimension --> size [4] becomes [1, 4]
        input = torch.from_numpy(input).float().unsqueeze(0)

        # transform input + normalized hidden state (with or without noisy input)
        # input += torch.randn(input.shape) * self.std_noise_in ## uncomment for noise
        i = self.i2h(input)
        h = self.norm_h(self.h2h(hidden))
        # h = self.h2h(hidden)

        # combine + apply nonlinearity (with or without time constant + noisy hidden activity)
        x = self.activ(i + h)
        # x = hidden * (1 - self.alpha) + x * self.alpha ## uncomment for time constant
        # x += torch.randn(hidden.shape) * self.std_noise_rec ## uncomment for noise

        # pull current hidden activity for next time step
        hidden = x 

        # pass output through final layer
        x = self.h2o(x)

        # convert to np.array (requires one strip for NN_output_size>1 // two strips for NN_output_size=1)
        # output = x.detach().numpy()[0]
        output = x.detach().numpy()[0][0]

        return output, hidden
