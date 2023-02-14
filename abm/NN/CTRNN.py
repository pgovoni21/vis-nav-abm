import torch
import torch.nn as nn

# import wandb
# wandb.init(project="testing_ground")

class CTRNN(nn.Module):
    """Continuous-time RNN

    Parameters:
        architecture: 3-int tuple : input / hidden / output size (# neurons)
        dt: discretization time step *in ms* <-- check
            If None, equals time constant tau
    """
    def __init__(self, architecture, dt=100, init=None, copy_network=None, var=0.05):
        super().__init__()
        input_size, hidden_size, output_size = architecture
        self.hidden_size = hidden_size
        
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        # self.ordered_layers = [self.i2h, self.h2h, self.h2o]

        self.tau = 100
        self.alpha = dt / self.tau # default --> alpha = 1

        # init NN parameters
        self.init_weightbias(init, copy_network, var)


    def init_weightbias(self, init, copy_network, var):
        """
        Initial weight distribution @ t = 0
        """
        if copy_network is None: # gen = 0
        
            if isinstance(init, tuple): # specified scheme
                type, spec = init
                if type == 'sparse':
                    for m in self.modules():
                        if isinstance(m, nn.Linear):
                            nn.init.sparse_(m.weight, sparsity = spec)
            
            else: pass # default scheme
        
        else: # gen > 0 --> mutate from parent NN
          
            # copy parent layers
            parent_layers = [m for m in copy_network.modules() if isinstance(m, nn.Linear)]
            self_layers = [m for m in self.modules() if isinstance(m, nn.Linear)]

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
