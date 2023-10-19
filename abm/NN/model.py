import torch
import torch.nn as nn

from abm.NN.memory import FNN, CTRNN, GRU
from abm.NN.vision import ConvNeXt as CNN
from abm.NN.vision import LayerNorm, GRN
# from abm.helpers import timer

class WorldModel(nn.Module):
    # @timer
    def __init__(self,
                 arch=((4,8),[1],[4],2,2,1,0),
                 activ='relu',
                 RNN_type='fnn',
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
            sensory_noise_std
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
        elif RNN_type == 'ctrnn': self.rnn = CTRNN(arch=RNN_arch,activ=activ)
        elif RNN_type == 'gru': self.rnn = GRU(arch=RNN_arch,activ=activ)
        else: raise ValueError(f'Invalid RNN type: {RNN_type}')

        # init linear controller layer (LCL)
        LCL_in_size = RNN_hidden_size
        # LCL_in_size = RNN_hidden_size + CNN_out_size ## concatenate with vis if using residual
        self.lcl = nn.Linear(LCL_in_size, LCL_output_size)

        # init noise
        self.sensory_noise_std = sensory_noise_std

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

        # add sensory noise
        if self.sensory_noise_std > 0:
            vis_features += torch.randn(vis_features.shape) * self.sensory_noise_std
            other_input += torch.randn(other_input.shape) * self.sensory_noise_std

        # concatenate + pass through memory module
        RNN_in = torch.cat((vis_features, other_input), dim=1)
        RNN_out, hidden = self.rnn(RNN_in, hidden)
        # LCL_in = RNN_in + RNN_out # skip connection

        # pass through final layer
        LCL_out = self.lcl(RNN_out)
        # action = LCL_in + LCL_out # skip connection
        action = torch.tanh(LCL_out) # scale to [-1:1]

        # action = action.detach().numpy()[0] # --> NN_output.size > 1
        action = action.detach().numpy()[0][0] # --> NN_output.size = 1

        return action, hidden


# ----------------------------------------------------------------------------------------------

if __name__ == '__main__':

    from abm.NN.vision import LayerNorm,GRN

    CNN_input_size = (4,8) # number elements, visual resolution
    CNN_depths = [1,1]
    CNN_dims = [2,4]
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