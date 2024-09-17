import torch
import torch.nn as nn
import numpy as np

from abm.NN.memory import FNN, FNN2, FNN_noise, CTRNN, GRU, GRU_parallel
from abm.NN.vision import ConvNeXt as CNN
from abm.NN.vision import LayerNorm, GRN
# from abm.helpers import timer

class WorldModel(nn.Module):
    # @timer
    def __init__(self,
                 arch=((4,8),[1],[4],1,2,1),
                 activ='relu',
                 RNN_type='fnn',
                #  CL_type='linear',
                 param_vector=None,
                 ):
        super().__init__()

        # unpack NN parameters from architecture tuple
        (   CNN_input_size, 
            CNN_depths, 
            CNN_dims, 
            RNN_other_input_size, 
            RNN_hidden_size, 
            LCL_output_size, 
        ) = arch

        # init vision module
        num_class_elements, vis_field_res = CNN_input_size
        CNN_out_size = CNN_dims[-1] 
        self.cnn = CNN(
            in_dims=num_class_elements,
            depths=CNN_depths, 
            dims=CNN_dims,
            activ=activ,
            )

        # init memory module
        self.RNN_hidden_size = RNN_hidden_size
        RNN_in_size = CNN_out_size + RNN_other_input_size
        RNN_arch = (RNN_in_size, RNN_hidden_size)
        if RNN_type == 'fnn': self.rnn = FNN(arch=RNN_arch,activ=activ)
        elif RNN_type == 'fnn2': self.rnn = FNN2(arch=RNN_arch,activ=activ)
        elif RNN_type == 'fnn_noise': self.rnn = FNN_noise(arch=RNN_arch,activ=activ)
        elif RNN_type == 'ctrnn': self.rnn = CTRNN(arch=RNN_arch,activ=activ)
        elif RNN_type == 'gru': self.rnn = GRU(arch=RNN_arch,activ=activ)
        elif RNN_type == 'gru_para': self.rnn = GRU_parallel(arch=RNN_arch,activ=activ)
        else: raise ValueError(f'Invalid RNN type: {RNN_type}')

        # init linear controller layer (LCL)
        LCL_in_size = RNN_hidden_size
        self.lcl = nn.Linear(LCL_in_size, LCL_output_size)
        # self.CL_type = CL_type
        # if CL_type == 'linear':
        #     LCL_in_size = RNN_hidden_size
        #     self.lcl = nn.Linear(LCL_in_size, LCL_output_size)
        # elif CL_type == 'skip':
        #     LCL_in_size = RNN_hidden_size + CNN_out_size 
        #     self.lcl = nn.Linear(LCL_in_size, LCL_output_size)
        # elif CL_type == 'variational':
        #     LCL_in_size = RNN_hidden_size
        #     self.fc_mu = nn.Linear(LCL_in_size, LCL_output_size)
        #     self.fc_sigma = nn.Linear(LCL_in_size, LCL_output_size)

        # init discrete action space
        self.model_out_type = 'cont'
        # self.model_out_type = 'disc'
        # a_size = LCL_output_size
        # self.actions = np.linspace(-1 + 2/a_size, 1, a_size)

        # initialize w+b according to passed vector (via optimizer) or init distribution
        if param_vector is not None:
            self.assign_params(param_vector)
        else:
            self._init_weights

        # disable autograd computation since we're not computing gradients
        for param in self.parameters():
            param.requires_grad = False

        # limit parallel computation to avoid CPU interference with multiproc sims
        torch.set_num_threads(1)


    def assign_params(self, param_vector):
        # chunk mutated params + send to model data
        param_num = 0
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.LayerNorm, LayerNorm, GRN, nn.Linear, nn.GRU)):
                for p in m.parameters():
                    chunk = param_vector[param_num : param_num + p.numel()]
                    reshaped_chunk = chunk.reshape(p.shape)	
                    torched_chunk = torch.from_numpy(reshaped_chunk).float()	
                    p.data = torched_chunk
                    param_num += p.numel()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.LayerNorm, LayerNorm, GRN, nn.Linear, nn.InstanceNorm1d, nn.GRU)):
                for p in m.parameters():
                    nn.init.trunc_normal_(m.weight, std=.02)
                    nn.init.constant_(m.bias, 0)

    # @timer
    def forward(self, vis_input, other_input, hidden):

        # initialize hidden state to zeros (t = 0 for sim run, saved in Agent instance)
        if hidden is None:
            hidden = torch.zeros(self.RNN_hidden_size).unsqueeze(0)

        # convert np to torch
        vis_input = torch.from_numpy(vis_input).float().unsqueeze(0)
        other_input = torch.from_numpy(other_input).float().unsqueeze(0)

        # pass through visual module
        vis_features = self.cnn(vis_input)

        # concatenate + pass through memory module
        RNN_in = torch.cat((vis_features, other_input), dim=1)
        RNN_out, hidden = self.rnn(RNN_in, hidden)

        # pass through final layer
        LCL_out = self.lcl(RNN_out)
        # if self.CL_type == 'linear':
        #     LCL_out = self.lcl(RNN_out)
        # elif self.CL_type == 'skip':
        #     LCL_in = RNN_in + RNN_out
        #     LCL_out = self.lcl(RNN_out) + LCL_in
        # elif self.CL_type == 'variational':
        #     self.mu = self.fc_mu(RNN_out)
        #     self.log_sigma = self.fc_sigma(RNN_out)
        #     eps = torch.randn(self.mu.shape[0], self.mu.shape[1])
        #     LCL_out = self.mu + torch.exp(self.log_sigma / 2) * eps

        if self.model_out_type == 'cont':
            # scale to action space [-1,1] + convert to numpy
            action = torch.tanh(LCL_out)
            # action = action.detach().numpy()[0] # --> NN_output.size > 1
            action = action.detach().numpy()[0][0] # --> NN_output.size = 1

        elif self.model_out_type == 'disc':
            decision = torch.argmax(LCL_out[0])
            action = self.actions[decision]

        return action, hidden


# ----------------------------------------------------------------------------------------------

if __name__ == '__main__':

    from abm.NN.vision import LayerNorm,GRN

    CNN_input_size = (4,32) # number elements, visual resolution
    CNN_depths = [1]
    CNN_dims = [4]
    RNN_input_other_size = 1
    RNN_hidden_size = 2
    LCL_output_size = 1
    RNN_type = 'fnn'

    arch = (
        CNN_input_size, 
        CNN_depths, 
        CNN_dims, 
        RNN_input_other_size, 
        RNN_hidden_size, 
        LCL_output_size,
        )

    model = WorldModel(arch=arch, RNN_type=RNN_type,)

    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.InstanceNorm1d, nn.GRU, nn.LSTM, nn.Conv1d, nn.LayerNorm, LayerNorm, GRN)):
        
            print(f'Layer: {m}')
            params = sum(p.numel() for p in m.parameters())
            print(f'#Params: {params}')
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total #Params: {total_params}')