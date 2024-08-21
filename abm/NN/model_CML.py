import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def matVec(mat, vec):
    # print(mat.shape, vec[:,None].shape)
    return np.squeeze( mat @ vec[:,None] )

class Model(torch.nn.Module):
    def __init__(self, o_size, a_size, s_dim, sharpness):
        super(Model, self).__init__()
        # # onehot
        # self.Q = torch.nn.Parameter(1*torch.randn(s_dim, o_size, device=device))
        # self.V = torch.nn.Parameter(0.1*torch.randn(s_dim, a_size, device=device))
        # self.W = torch.nn.Parameter(0.1*torch.randn(a_size, s_dim, device=device))
        # vector
        self.Q = torch.nn.Parameter(torch.normal(0, 0.1, (s_dim, o_size), device=device))
        self.V = torch.nn.Parameter(torch.normal(0, 1, (s_dim, a_size), device=device))
        self.o_size = o_size
        self.a_size = a_size

        # actions = np.linspace(-np.pi/2 + np.pi/32, np.pi/2, a_size)
        self.actions = np.linspace(-1 + 2/a_size, 1, a_size)

        self.probs = np.array([np.exp(-sharpness*x**2) for x in self.actions])
        # print(self.probs)
        self.probs /= self.probs.sum()

        # # onehot
        # # # views[idx] --> NNNNNEEE + goal
        # # self.goal = 61 + 131
        # # views[idx] --> WNNNNNEE + goal
        # # self.goal = 108 + 131
        # # views[idx] --> WWWWWWWW
        # self.goal = 131

        # vector, dist (only vis obs)
        ## x,y,theta = 400,400,np.pi (west)
        self.goal = np.array([0,0,0,0,0,0, 0.21748388, 0.25667073, 0.17557771, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0.17728608, 0.241245, 0.26853046, 0.26853046, 0.241245, 0,0])

        # # limit parallel computation to avoid CPU interference with multiproc sims
        # torch.set_num_threads(1)

    def gaussian_action(self):
        act_idx = np.random.choice(self.a_size, 1, p=self.probs).item()
        return act_idx, self.actions[act_idx]


    # def planned_action(self, obs, a_record, affordance):
    def planned_action(self, obs):

        affordance_vector = torch.ones(self.a_size)
        # affordance_vector = torch.tensor([np.exp(-10000*x**2) for x in self.actions])

        # affordance_vector[affordance] = 1
        # affordance_vector_fix = affordance_vector.clone()
        # not_recommended_actions = a_record
        # affordance_vector_fix[not_recommended_actions] *= 0.

        # # onehot
        # delta = self.Q[:,self.goal] - self.Q[:,obs]
        # # utility = (self.W@delta) * affordance_vector_fix
        # utility = (self.W@delta) * affordance_vector # onehot
        # # print(utility)

        # vector
        goal = torch.from_numpy(self.goal.astype(np.float32))
        obs = torch.from_numpy(obs.astype(np.float32))

        delta = matVec(self.Q, goal) - matVec(self.Q, obs)
        # print(self.V.T.shape, delta.shape, delta[:,None].shape)
        # print(torch.mean(delta))
        utility = matVec(self.V.T, delta)
        # # gating --> availability of action in env, maxmin for limb angles for ant (very useful), not as useful for grid nav
        # action = self.f_a(gating * utility)

        if torch.max(utility)!= 0:
            act_idx = torch.argmax(utility).item() # never used + forcing use decr perf
        else:
            utility = (self.V.T@delta) * affordance_vector
            act_idx = torch.argmax(utility).item()

        return act_idx, self.actions[act_idx]

