import torch
import torch.nn as nn

class CTRNN(nn.Module):
    """
    RNN model
    Euler-discretized dynamical system

    Parameters:
        architecture: 3-int tuple : input / hidden / output size (# neurons)
        dt: discretization time step *in ms* <-- check
            If None, equals time constant tau

    forward() eqns from + layout inspired by: https://colab.research.google.com/github/gyyang/nn-brain/blob/master/RNN_tutorial.ipynb
    """
    def __init__(self, architecture, dt=100, init=None, copy_network=None, var=0.05):
        super().__init__()
        input_size, hidden_size, output_size = architecture
        self.hidden_size = hidden_size
        
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.ordered_layers = [self.i2h, self.h2h, self.h2o]

        self.tau = 100
        self.alpha = dt / self.tau # default --> alpha = 1

        # initialize NN parameters
        if init: # gen = 0  &  weight initialization scheme specified
            type, spec = init
            if type == 'sparse':
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.sparse_(m.weight, sparsity = spec)

        elif copy_network: # gen > 0 --> mutate child NN from parents
            self.mutate(copy_network, var)

    def mutate(self, copy_network, var):
        # copy parent layers
        parent_layers = copy_network.ordered_layers
        self_layers = self.ordered_layers
        # parent_layers = [m for m in copy_network.modules() if isinstance(m, nn.Linear)]
        # self_layers = [m for m in self.modules() if isinstance(m, nn.Linear)]

        # iterate over self layers
        for parent_layer, self_layer in zip(parent_layers, self_layers):
            
            # shift weights by perturbation chosen from normal/gaussian dist of mean = 0 / var = 1
            w = parent_layer.weight
            noise = var * torch.randn(w.shape) # rescale noise to desired degree
            self_layer.weight.data = w + noise

            # shift biases as well
            b = parent_layer.bias
            noise = var * torch.randn(b.shape)
            self_layer.bias.data = b + noise

    def forward(self, input, hidden):
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
                hidden = torch.zeros(self.hidden_size)

            # converts np.array to torch.tensor + adds 1 dimension --> size [4] becomes [1, 4]
            input = torch.from_numpy(input).float().unsqueeze(0)

            # carry out recurrent calculation given input + existing hidden state using ReLu
            x = torch.relu(self.i2h(input) + self.h2h(hidden))
            x = hidden * (1 - self.alpha) + x * self.alpha

            hidden = x # --> pull current hidden activity for next time step

            # reduce to action dimension + pass through Tanh function (bounding to -1:1)
            x = self.h2o(x)
            x = torch.tanh(x)

            # convert to np.array (requires one strip for NN_output_size>1 // two strips for NN_output_size=1)
            # output = x.detach().numpy()[0]
            output = x.detach().numpy()[0][0]

        return output, hidden

    # def pull_parameters(self):
    #     # extract weights/biases for each layer + output as list of numpy arrays
    #     parameter_list = []
    #     for layer in self.ordered_layers:
    #         layer_weight_np = layer.weight.data.numpy()
    #         layer_bias_np = layer.bias.data.numpy()

    #         layer_weight_nestedlist = layer_weight_np.tolist()
    #         layer_bias_nestedlist = layer_bias_np.tolist()

    #         parameter_list.append(layer_weight_nestedlist)
    #         parameter_list.append(layer_bias_nestedlist)

    #     return parameter_list

    # def push_parameters(self, parameter_list):
    #     # input list of numpy arrays as weights/biases for each layer
    #     for i, param in enumerate(parameter_list):
    #         self.ordered_layers[i].weight.data = torch.from_numpy(param)