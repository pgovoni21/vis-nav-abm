import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, 
                 arch, 
                 activ='relu',
                 sharpness=1,
                 param_vector=None,
                 mode='train',):
        super().__init__()
        input_size, hidden_size, act_size = arch
        self.hidden_size = hidden_size
        self.act_size = act_size
        self.mode = mode

        # print(f'Input Size: {input_size}, Hidden Size: {hidden_size}, Action Size: {act_size}')
        # print(f'Model Mode: {mode}')

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=False)
        # self.norm_h = nn.InstanceNorm1d(hidden_size, affine=True)
        self.h2o_pred = nn.Linear(hidden_size, input_size-1)
        self.h2o_act = nn.Linear(hidden_size, act_size)

        if activ == 'relu': self.activ = torch.relu
        elif activ == 'tanh': self.activ = torch.tanh
        elif activ == 'silu': self.activ = torch.nn.SiLU()
        elif activ == 'gelu': self.active = torch.nn.GELU()
        else: raise ValueError(f'Invalid activation function: {activ}')

        dt = 20
        # tau = 20 # normal neural dynamics
        tau = 100 # slower dynamics via NDMA receptors
        self.alpha = dt / tau
        noise_in = 0.01
        noise_rec = 0.05
        self.noise_in = np.sqrt(2/self.alpha)*noise_in
        self.noise_rec = np.sqrt(2/self.alpha)*noise_rec

        self.actions = np.linspace(-1 + 2/act_size, 1, act_size)
        self.probs = np.array([np.exp(-sharpness*x**2) for x in self.actions])
        self.probs /= self.probs.sum()


        # initialize w+b according to passed vector (via optimizer) or init distribution
        if param_vector is not None:
            self.assign_params(param_vector)
        # else:
        #     self._init_weights()

        # disable autograd computation since we're not computing gradients
        for param in self.parameters():
            param.requires_grad = False

        # limit parallel computation to avoid CPU interference with multiproc sims
        torch.set_num_threads(1)


    def assign_params(self, param_vector):
        # chunk mutated params + send to model data
        param_num = 0
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.LayerNorm, nn.InstanceNorm1d, nn.Linear, nn.GRU)):
                for p in m.parameters():
                    chunk = param_vector[param_num : param_num + p.numel()]
                    reshaped_chunk = chunk.reshape(p.shape)	
                    torched_chunk = torch.from_numpy(reshaped_chunk).float()	
                    p.data = torched_chunk
                    param_num += p.numel()

    # def _init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, (nn.Conv1d, nn.LayerNorm, nn.InstanceNorm1d, nn.Linear, nn.GRU)):
    #             for p in m.parameters():
    #                 nn.init.trunc_normal_(m.weight, std=.02)
    #                 nn.init.constant_(m.bias, 0)

    def assign_params_h2o_act(self, param_vector):
        param_num = 0
        for p in self.h2o_act.parameters():
            chunk = param_vector[param_num : param_num + p.numel()]
            reshaped_chunk = chunk.reshape(p.shape)	
            torched_chunk = torch.from_numpy(reshaped_chunk).float()	
            p.data = torched_chunk
            param_num += p.numel()


    def forward(self, state, hidden):
        if hidden is None:
            hidden = torch.zeros(self.hidden_size).unsqueeze(0)

        # convert np to torch
        state = torch.from_numpy(state).float().unsqueeze(0)

        i = self.i2h(state + self.noise_in*torch.randn_like(state))
        x = self.h2h(hidden)
        # x = self.norm_h(x)
        # x = self.activ(i + x)
        x = self.activ(i + x + self.noise_rec*torch.randn_like(x))
        x = hidden * (1 - self.alpha) + x * self.alpha ## uncomment for time constant
        hidden = x # --> pull current hidden activity + return this as second variable

        if self.mode == 'train_pred':
            o = self.h2o_pred(x)

            act_idx = np.random.choice(self.act_size, 1, p=self.probs).item()
            action = self.actions[act_idx]

        elif self.mode == 'train_act' or self.mode == 'test':
            o = self.h2o_act(x)

            act_idx = torch.softmax(o, dim=1).argmax().item()
            action = self.actions[act_idx]

        return o, hidden, action



    # def forward(self, obs, act, hidden):
    #     if hidden is None:
    #         hidden = torch.zeros(self.hidden_size).unsqueeze(0)

    #     # convert np to torch
    #     obs = torch.from_numpy(obs).float().unsqueeze(0)
    #     act = torch.from_numpy(act).float().unsqueeze(0)

    #     i = self.obs2h(obs) + self.act2h(act)
    #     x = self.h2h(hidden)
    #     # x = self.norm_h(x)
    #     x = self.activ(i + x)
    #     # x = hidden * (1 - self.alpha) + x * self.alpha ## uncomment for time constant
    #     hidden = x # --> pull current hidden activity + return this as second variable

    #     if self.mode == 'train':
    #         o = self.h2o_pred(x)

    #         act_idx = np.random.choice(self.act_size, 1, p=self.probs).item()
    #         action = self.actions[act_idx]

    #     elif self.mode == 'test':
    #         o = self.h2o_act(x)

    #         act_idx = torch.softmax(o, dim=1).argmax().item()
    #         action = self.actions[act_idx]

    #     return o, hidden, action
