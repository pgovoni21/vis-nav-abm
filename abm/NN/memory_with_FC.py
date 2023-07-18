import torch
import torch.nn as nn


class FNN(nn.Module):
    def __init__(self, arch, activ='relu', param_vector=None):
        super().__init__()
        input_size, hidden_size, output_size = arch
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.layers = [self.i2h, self.h2o]
        
        if activ == 'relu': self.activ = torch.relu
        elif activ == 'tanh': self.activ = torch.tanh
        elif activ == 'silu': self.activ = torch.nn.SiLU()
        elif activ == 'gelu': self.active = torch.nn.GELU()
        else: raise ValueError(f'Invalid activation function: {activ}')

        if param_vector is not None:
            self.assign_params(param_vector)

    def assign_params(self, param_vector):
        param_num = 0
        for l in self.layers:
            for param in [l.weight, l.bias]:
                if param is not None:
                    chunk = param_vector[param_num : param_num + param.numel()]
                    reshaped_chunk = chunk.reshape(param.shape)	
                    torched_chunk = torch.from_numpy(reshaped_chunk).float()	
                    param.data = torched_chunk
                    param_num += param.numel()

    def forward(self, state, hidden_null):
        x = torch.from_numpy(state).float().unsqueeze(0)
        x = self.activ(self.i2h(x))
        x = self.h2o(x)
        action = x.detach().numpy()[0][0]
        return action, hidden_null


#--------------------------------------------------------------------------------------------------------#

class CTRNN(nn.Module):
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
    def __init__(self, arch, activ='relu', param_vector=None):
        super().__init__()

        # construct architecture of NN
        input_size, hidden_size, output_size = arch
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm_h = nn.InstanceNorm1d(hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.layers = [self.i2h, self.h2h, self.norm_h, self.h2o]
        # self.layers = [self.i2h, self.h2h, self.h2o]

        # set activation function
        if activ == 'relu': self.activ = torch.relu
        elif activ == 'tanh': self.activ = torch.tanh
        elif activ == 'silu': self.activ = torch.nn.SiLU()
        elif activ == 'gelu': self.active = torch.nn.GELU()
        else: raise ValueError(f'Invalid activation function: {activ}')

        # # set time constant
        # tau = 100
        # self.alpha = dt / tau # default --> alpha = 1

        # assign parameters from optimizer
        if param_vector is not None:
            self.assign_params(param_vector)

    def assign_params(self, param_vector):

        # chunk mutated params + send to model data
        param_num = 0
        for l in self.layers:
            for param in [l.weight, l.bias]:
                if param is not None:
                    chunk = param_vector[param_num : param_num + param.numel()]
                    reshaped_chunk = chunk.reshape(param.shape)	
                    torched_chunk = torch.from_numpy(reshaped_chunk).float()	
                    param.data = torched_chunk
                    param_num += param.numel()

    def forward(self, state, hidden):
        """Propagate input through RNN
        
        Inputs:
            input: 1D array of shape (input_size)
            hidden: tensor of shape (hidden_size)
        
        Outputs:
            hidden: tensor of shape (hidden_size)
            actions: 1D array of shape (output_shape)
        """
        # initialize hidden state to zeros (t = 0 for sim run, saved in Agent instance)
        if hidden is None:
            hidden = torch.zeros(self.hidden_size).unsqueeze(0)

        input = torch.from_numpy(state).float().unsqueeze(0) # --> convert np to torch

        # transform input + normalized hidden state (with or without noisy input)
        # input += torch.randn(input.shape) * self.std_noise_in    ## uncomment for noise
        i = self.i2h(input)
        h = self.norm_h(self.h2h(hidden))
        # h = self.h2h(hidden)

        # combine + apply nonlinearity (with or without time constant + noisy hidden activity)
        x = self.activ(i + h)
        # x = hidden * (1 - self.alpha) + x * self.alpha ## uncomment for time constant
        # x += torch.randn(hidden.shape) * self.std_noise_rec ## uncomment for noise

        hidden = x # --> pull current hidden activity + return this as second variable

        # pass output through final layer
        action = self.h2o(x)

        # convert torch to np (requires one strip for NN_output_size>1 // two strips for NN_output_size=1)	
        # action = action.detach().numpy()[0]
        action = action.detach().numpy()[0][0]

        return action, hidden
    
#--------------------------------------------------------------------------------------------------------#

class GRU(nn.Module):
    def __init__(self, arch, activ='', param_vector=None):
        super().__init__()
        input_size, hidden_size, output_size = arch
        self.hidden_size = hidden_size
        
        self.gru = nn.GRU(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.layers = [self.gru, self.fc]

        if param_vector is not None:
            self.assign_params(param_vector)

    def assign_params(self, param_vector):
        param_num = 0
        for l in self.layers:
            if isinstance(l, nn.Linear):
                for param in [l.weight, l.bias]:
                    if param is not None:
                        chunk = param_vector[param_num : param_num + param.numel()]
                        reshaped_chunk = chunk.reshape(param.shape)	
                        torched_chunk = torch.from_numpy(reshaped_chunk).float()	
                        param.data = torched_chunk
                        param_num += param.numel()
            elif isinstance(l, nn.GRU):
                for param in [l.weight_ih_l0, l.weight_hh_l0, l.bias_ih_l0, l.bias_hh_l0]:
                    if param is not None:
                        chunk = param_vector[param_num : param_num + param.numel()]
                        reshaped_chunk = chunk.reshape(param.shape)	
                        torched_chunk = torch.from_numpy(reshaped_chunk).float()	
                        param.data = torched_chunk
                        param_num += param.numel()
    
    def forward(self, state, hidden):
        if hidden is None:
            hidden = torch.zeros(self.hidden_size).unsqueeze(0)
        input = torch.from_numpy(state).float().unsqueeze(0) # converts np.array to torch.tensor
        x, hidden = self.gru(input, hidden)
        action = self.fc(x)
        # action = action.detach().numpy()[0] # --> NN_output.size > 1
        action = action.detach().numpy()[0][0] # --> NN_output.size = 1
        return action, hidden
    

#--------------------------------------------------------------------------------------------------------#

class LSTM(nn.Module):
    def __init__(self, arch, activ='', param_vector=None):
        super().__init__()
        input_size, hidden_size, output_size = arch
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.layers = [self.lstm, self.fc]

        if param_vector is not None:
            self.assign_params(param_vector)

    def assign_params(self, param_vector):
        param_num = 0
        for l in self.layers:
            if isinstance(l, nn.Linear):
                for param in [l.weight, l.bias]:
                    if param is not None:
                        chunk = param_vector[param_num : param_num + param.numel()]
                        reshaped_chunk = chunk.reshape(param.shape)	
                        torched_chunk = torch.from_numpy(reshaped_chunk).float()	
                        param.data = torched_chunk
                        param_num += param.numel()
            elif isinstance(l, nn.LSTM):
                for param in [l.weight_ih_l0, l.weight_hh_l0, l.bias_ih_l0, l.bias_hh_l0]:
                    if param is not None:
                        chunk = param_vector[param_num : param_num + param.numel()]
                        reshaped_chunk = chunk.reshape(param.shape)	
                        torched_chunk = torch.from_numpy(reshaped_chunk).float()	
                        param.data = torched_chunk
                        param_num += param.numel()

    def forward(self, state, hidden, cell):
        if hidden is None:
            hidden = torch.zeros(self.hidden_size).unsqueeze(0)
            cell = torch.zeros(self.hidden_size).unsqueeze(0)
        input = torch.from_numpy(state).float().unsqueeze(0)
        x, (hidden, cell) = self.lstm(input, (hidden, cell))
        action = self.fc(x)
        # action = action.detach().numpy()[0] # --> NN_output.size > 1
        action = action.detach().numpy()[0][0] # --> NN_output.size = 1
        return action, hidden, cell
    