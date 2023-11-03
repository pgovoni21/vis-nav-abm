import torch
import torch.nn as nn


class FNN(nn.Module):
    def __init__(self, arch, activ='relu'):
        super().__init__()
        input_size, hidden_size = arch
        
        self.i2h = nn.Linear(input_size, hidden_size)
        
        if activ == 'relu': self.activ = torch.relu
        elif activ == 'tanh': self.activ = torch.tanh
        elif activ == 'silu': self.activ = torch.nn.SiLU()
        elif activ == 'gelu': self.active = torch.nn.GELU()
        else: raise ValueError(f'Invalid activation function: {activ}')

    def forward(self, state, hidden_null):
        x = self.i2h(state)
        x = self.activ(x)
        return x, hidden_null


#--------------------------------------------------------------------------------------------------------#

class CTRNN(nn.Module):
    def __init__(self, arch, activ='relu'):
        super().__init__()
        input_size, hidden_size = arch
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm_h = nn.InstanceNorm1d(hidden_size)
        self.layers = [self.i2h, self.h2h, self.norm_h]

        if activ == 'relu': self.activ = torch.relu
        elif activ == 'tanh': self.activ = torch.tanh
        elif activ == 'silu': self.activ = torch.nn.SiLU()
        elif activ == 'gelu': self.active = torch.nn.GELU()
        else: raise ValueError(f'Invalid activation function: {activ}')
        
        # # set time constant
        # tau = 100
        # self.alpha = dt / tau # default --> alpha = 1

    def forward(self, state, hidden):
        if hidden is None:
            hidden = torch.zeros(self.hidden_size).unsqueeze(0)

        i = self.i2h(state)
        h = self.norm_h(self.h2h(hidden))
        x = self.activ(i + h)
        # x = hidden * (1 - self.alpha) + x * self.alpha ## uncomment for time constant

        hidden = x # --> pull current hidden activity + return this as second variable

        return x, hidden
    
#--------------------------------------------------------------------------------------------------------#

class GRU(nn.Module):
    def __init__(self, arch, activ=''):
        super().__init__()
        input_size, hidden_size = arch
        self.hidden_size = hidden_size
        
        self.gru = nn.GRU(input_size, hidden_size)
    
    def forward(self, state, hidden):
        if hidden is None:
            hidden = torch.zeros(self.hidden_size).unsqueeze(0)
        x, hidden = self.gru(state, hidden)
        return x, hidden
    

#--------------------------------------------------------------------------------------------------------#

class LSTM(nn.Module):
    def __init__(self, arch, activ=''):
        super().__init__()
        input_size, hidden_size = arch
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, state, hidden, cell):
        if hidden is None:
            hidden = torch.zeros(self.hidden_size).unsqueeze(0)
            cell = torch.zeros(self.hidden_size).unsqueeze(0)
        x, (hidden, cell) = self.lstm(input, (hidden, cell))
        return x, hidden, cell
    

# ----------------------------------------------------------------------------------------------

if __name__ == '__main__':

    model = FNN(
        arch=(15,8),
    )

    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.InstanceNorm1d, nn.GRU, nn.LSTM)):
        
            print(m)
            params = sum(p.numel() for p in m.parameters())
            print(params)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total #Params: {total_params}')