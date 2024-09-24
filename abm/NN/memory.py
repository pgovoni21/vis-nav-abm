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
        elif activ == 'gelu': self.activ = torch.nn.GELU()
        else: raise ValueError(f'Invalid activation function: {activ}')

    def forward(self, state, hidden_null):
        x = self.i2h(state)
        x = self.activ(x)
        return x, hidden_null

class FNN2(nn.Module):
    def __init__(self, arch, activ='relu'):
        super().__init__()
        input_size, hidden_size = arch
        
        self.h1 = nn.Linear(input_size, hidden_size)
        self.h2 = nn.Linear(hidden_size, hidden_size)
        
        if activ == 'relu': self.activ = torch.relu
        elif activ == 'tanh': self.activ = torch.tanh
        elif activ == 'silu': self.activ = torch.nn.SiLU()
        elif activ == 'gelu': self.activ = torch.nn.GELU()
        else: raise ValueError(f'Invalid activation function: {activ}')

    def forward(self, state, hidden_null):
        x = self.h1(state)
        x = self.activ(x)
        x = self.h2(x)
        x = self.activ(x)
        return x, hidden_null

class FNN_noise(nn.Module):
    def __init__(self, arch, activ='relu'):
        super().__init__()
        input_size, hidden_size = arch
        
        self.i2h = nn.Linear(input_size, hidden_size)
        self.noise = Noise(hidden_size)
        
        if activ == 'relu': self.activ = torch.relu
        elif activ == 'tanh': self.activ = torch.tanh
        elif activ == 'silu': self.activ = torch.nn.SiLU()
        elif activ == 'gelu': self.activ = torch.nn.GELU()
        else: raise ValueError(f'Invalid activation function: {activ}')

    def forward(self, state, hidden_null):
        x = self.i2h(state)
        x = self.activ(x)
        x = self.noise(x)
        return x, hidden_null

class Noise(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x):
        noise = torch.randn_like(x) * self.weight
        return x + noise


class FNN_cognoise(nn.Module):
    def __init__(self, arch, activ='relu'):
        super().__init__()
        input_size, hidden_size = arch
        
        self.i2h = nn.Linear(input_size, hidden_size)
        self.noise = 0.01
        
        if activ == 'relu': self.activ = torch.relu
        elif activ == 'tanh': self.activ = torch.tanh
        elif activ == 'silu': self.activ = torch.nn.SiLU()
        elif activ == 'gelu': self.activ = torch.nn.GELU()
        else: raise ValueError(f'Invalid activation function: {activ}')

    def forward(self, state, hidden_null):
        x = state + self.noise*torch.randn_like(state)
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


class GRU_parallel(nn.Module):
    def __init__(self, arch, activ=''):
        super().__init__()
        input_size, hidden_size = arch
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size)
        self.h1 = nn.Linear(input_size, hidden_size)
        self.o1 = nn.Linear(hidden_size*2, hidden_size)

        if activ == 'relu': self.activ = torch.relu
        elif activ == 'tanh': self.activ = torch.tanh
        elif activ == 'silu': self.activ = torch.nn.SiLU()
        elif activ == 'gelu': self.activ = torch.nn.GELU()
        else: raise ValueError(f'Invalid activation function: {activ}')

    def forward(self, state, hidden):
        if hidden is None:
            hidden = torch.zeros(self.hidden_size).unsqueeze(0)

        x = self.activ(self.h1(state))
        y, hidden = self.gru(state, hidden)
        z = torch.cat((x, y), dim=1)
        z = self.activ(self.o1(z))
        return z, hidden
    

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
        x, (hidden, cell) = self.lstm(state, (hidden, cell))
        return x, hidden, cell


# ----------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # model = FNN(
    #     arch=(4,2),
    # )
    model = FNN_noise(
        arch=(4,2),
    )

    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.InstanceNorm1d, nn.GRU, nn.LSTM, Noise)):
        
            print(m)
            params = sum(p.numel() for p in m.parameters())
            print(params)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total #Params: {total_params}')

    model.forward(torch.rand(1,4), None)