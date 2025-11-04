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

class FNN_random_as_choice(nn.Module):
    def __init__(self, arch, activ='relu', weight=0.25):
        super().__init__()
        input_size, hidden_size = arch

        self.h1 = nn.Linear(input_size, hidden_size)
        self.h2 = nn.Linear(hidden_size, 2)
        self.hmove = nn.Linear(input_size, hidden_size)
        self.weight = weight

        if activ == 'relu': self.activ = torch.relu
        elif activ == 'tanh': self.activ = torch.tanh
        elif activ == 'silu': self.activ = torch.nn.SiLU()
        elif activ == 'gelu': self.activ = torch.nn.GELU()
        else: raise ValueError(f'Invalid activation function: {activ}')

    def forward(self, state, random=None):
        x1 = self.activ(self.h1(state))
        x2 = self.activ(self.h2(x1))
        # print(state,x1,x2)
        x = x2.argmax(dim=1)
        if x:
            x = self.hmove(state)
            x = self.activ(x)
        else:
            random = torch.randn(1) * self.weight
        return x, random


class FNN_nowall_3choice(nn.Module):
    def __init__(self, arch, activ='relu', hidden_mult=4, misc_weight=0.25):
        super().__init__()
        input_size, hidden_size = arch

        self.h1 = nn.Linear(input_size, hidden_size*hidden_mult)
        self.h2 = nn.Linear(hidden_size*hidden_mult, 3)
        self.act_explore = nn.Linear(input_size, hidden_size)
        self.act_exploit = nn.Linear(input_size, hidden_size)
        self.rand_choice_stdev = misc_weight

        if activ == 'relu': self.activ = torch.relu
        elif activ == 'tanh': self.activ = torch.tanh
        elif activ == 'silu': self.activ = torch.nn.SiLU()
        elif activ == 'gelu': self.activ = torch.nn.GELU()
        else: raise ValueError(f'Invalid activation function: {activ}')

    def forward(self, state, random=None):
        x = self.activ(self.h1(state))
        x = self.activ(self.h2(x))
        choice = x.argmax(dim=1)
        if choice == 0: # explore
            out = self.activ(self.act_explore(state))
        elif choice == 1: # exploit
            out = self.activ(self.act_exploit(state))
        elif choice == 2: # random
            random = torch.randn(1) * self.rand_choice_stdev
        return out, random


class FNN_gaussian(nn.Module):
    def __init__(self, arch, activ='relu'):
        super().__init__()
        input_size, hidden_size = arch

        self.h = nn.Linear(input_size, hidden_size)
        self.h_mu = nn.Linear(hidden_size, hidden_size)
        self.h_sig = nn.Linear(hidden_size, hidden_size)

        if activ == 'relu': self.activ = torch.relu
        elif activ == 'tanh': self.activ = torch.tanh
        elif activ == 'silu': self.activ = torch.nn.SiLU()
        elif activ == 'gelu': self.activ = torch.nn.GELU()
        else: raise ValueError(f'Invalid activation function: {activ}')

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        # self.kl = 0

    def forward(self, state, hidden_null):
        x = self.activ(self.h(state))
        mu = self.h_mu(x)
        sig = self.h_sig(x)

        out = mu + sig*self.N.sample(mu.shape)
        # self.kl = (sig**2 + mu**2 - torch.log(sig) - 1/2).sum()

        return out, hidden_null


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