from abm.NN.model_CML import Model
from abm import start_sim_CML

from pathlib import Path
import shutil, os, warnings, time
import numpy as np
import pickle
import random
from tqdm import tqdm
import dotenv as de

import matplotlib.pyplot as plt

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def matVec(mat, vec):
    return np.squeeze( mat @ vec[:,None] )



def train_patchintensive(EA_save_name, num_trajs=1000, num_steps=100, a_size=8, s_size=2500, sharpness=1, n_q=0.0025, n_v=0.0005, n_w=0.0005, norm=True):

    overall_time = time.time()

    root_dir = Path(__file__).parent.parent.parent
    data_dir = Path(root_dir, 'abm/data/simulation_data')
    EA_save_dir = Path(data_dir, EA_save_name)

    # Create save directory + copy .env file over
    if os.path.isdir(EA_save_dir):
        warnings.warn("Temporary directory for env files is not empty and will be overwritten")
        shutil.rmtree(EA_save_dir)
    Path(EA_save_dir).mkdir()
    shutil.copy(
        Path(root_dir, '.env'), 
        Path(EA_save_dir, '.env')
        )

    envconf = de.dotenv_values(fr'{EA_save_dir}/.env')
    vfr = int(envconf["VISUAL_FIELD_RESOLUTION"])

    # # onehot
    # with open(fr'{data_dir}/views_vfr{vfr}.bin', 'rb') as f:
    #     views = pickle.load(f)
    # o_size = len(views)

    # vector, dist
    o_size = vfr*2
    views = None

    model = Model(o_size, a_size, s_size, sharpness)
    # dataloader = np.zeros([num_trajs+int(num_trajs/10), num_steps-1, 3])
    loss_record = []

    x_patch, y_patch = 400,400
    radius_patch = 50

    # print(model.Q.shape, model.V.shape)
    # print(o_size, a_size, s_size)

    with torch.no_grad():
        for i in tqdm(range(num_trajs), desc="Epochs"):
            set_seed(i)

            if i % 20 == 0: # init near patch/wall
                x = np.random.uniform(x_patch - radius_patch, x_patch + radius_patch)
                y = np.random.uniform(y_patch - radius_patch, y_patch + radius_patch)
                # x = np.random.uniform(5, 20)
                # y = np.random.uniform(1, 1000)
                (o_pre, action, o_next),_,_ = start_sim_CML.start(load_dir=EA_save_dir, NN=model, mode='train', T=num_steps, views=views, x=x, y=y)

            else:
                (o_pre, action, o_next),_,_ = start_sim_CML.start(load_dir=EA_save_dir, NN=model, mode='train', T=num_steps, views=views)
    
            # print(o_pre.shape, action.shape, o_next.shape)
            # dataloader[i,:,0] = o_pre
            # dataloader[i,:,1] = action
            # dataloader[i,:,2] = o_next

            # # onehot, graph way
            # state_diff = model.Q[:,o_next] - model.Q[:,o_pre]
            # pred_error = state_diff - model.V[:,action]
            # identity = torch.eye(model.a_size)
            # desired = identity[action].mT

            # # onehot, ant way
            # state = model.Q[:,o_pre]
            # next_state_pred = state + model.V[:,action]
            # next_state = model.Q[:,o_next]
            # pred_error = next_state - next_state_pred

            loss = 0
            d_wq = torch.zeros_like(model.Q)
            d_wv = torch.zeros_like(model.V)
            for t in range(num_steps-1):
                # print(o_pre.shape, o_next.shape, action.shape)
                # print(o_pre[t,:].shape, o_next[t,:].shape, action[t].shape)

                # # vector + W
                # state_diff = matVec(model.Q, o_pre) - matVec(model.Q, o_next)
                # pred_error = state_diff - matVec(model.V, action)
                # identity = torch.eye(model.a_size)
                # desired = identity[action].mT

                # vector, no W
                state_diff = matVec(model.Q, o_pre[t,:]) - matVec(model.Q, o_next[t,:])
                pred_error = state_diff - model.V[:,int(action[t])]

                d_wq += pred_error[:, None] * torch.from_numpy(o_next[None,t,:])
                d_wv += pred_error[:, None] * torch.from_numpy(action[None,t])
                loss += np.linalg.norm(pred_error)

            # # onehot
            # model.Q[:,o_next] -= n_q * pred_error
            # model.V[:,action] += n_v * pred_error
            # model.W += n_w * desired@state_diff.mT
            # if norm:
            #     model.V.data = model.V / torch  .norm(model.V, dim=0)
            # loss = torch.nn.MSELoss()(pred_error, torch.zeros_like(pred_error))
            # loss_record.append(loss.cpu().item())

            # vector
            model.Q -= n_q * d_wq/num_steps
            model.V += n_v * d_wv/num_steps
            loss_record.append(loss/num_steps)

    
    # with open(Path(EA_save_dir, 'dataloader.bin'), 'wb') as f:
    #     pickle.dump(dataloader, f)
    with open(Path(EA_save_dir, 'model.bin'), 'wb') as f:
        pickle.dump(model, f)

    print('Training time:', time.time() - overall_time)
    return loss_record



def test(EA_save_name, num_trajs=25, num_steps=500):

    root_dir = Path(__file__).parent.parent.parent
    data_dir = Path(root_dir, 'abm/data/simulation_data')
    EA_save_dir = Path(data_dir, EA_save_name)

    # env_path = fr'{data_dir}/{EA_save_name}/.env'
    # envconf = de.dotenv_values(env_path)
    # vfr = int(envconf["VISUAL_FIELD_RESOLUTION"])
    # with open(fr'{data_dir}/views_vfr{vfr}.bin', 'rb') as f:
    #     views = pickle.load(f)
    views = None

    with open(Path(EA_save_dir, 'model.bin'), 'rb') as f:
        model = pickle.load(f)

    time = np.zeros(num_trajs)
    dist = np.zeros(num_trajs)

    with torch.no_grad():
        for i in tqdm(range(num_trajs), desc="Epochs"):
            set_seed(i*1000)
            _,timesteps,start_dist = start_sim_CML.start(load_dir=EA_save_dir, NN=model, mode='test', T=num_steps, views=views)
            time[i] = timesteps
            dist[i] = start_dist

    avg_time = np.mean(time)
    avg_dist = np.mean(dist)

    print(f'avg time: {avg_time}, avg dist: {avg_dist}')


def loss_plot(loss_record):

    plt.rcParams['font.size'] = 15
    plt.figure(dpi=100)
    plt.plot(loss_record)
    plt.yscale('log')
    plt.title('Training loss')
    plt.xlabel('Number of weight updates')
    plt.show()






if __name__ == '__main__':

    for num_trajs in [5000]:
    # for num_trajs in [10000]:
        # for num_steps in [100]:
        for num_steps in [250]:
            for a_size in [8]:
            # for a_size in [16]:
                # for s_size in [1000]:
                # for s_size in [2000]:
                for s_size in [4000]:
                    # for sharpness in [0]:
                    for sharpness in [10]:
                        # for n_q in [.1]:
                        # for n_q in [.05]:
                        # for n_q in [.01]:
                        # for n_q in [.0025]:
                        for n_q in [.001]:

                            n_v=n_q/5
                            n_w=n_q/5

                            name = f'CML_traj{num_trajs}_step{num_steps}_a{a_size}_s{s_size}_sh{sharpness}_nq{n_q}_goalpatchW_vfr8_patchintensive'
                            print(name)
                            loss_record = train_patchintensive(
                                EA_save_name=name, 
                                num_trajs=num_trajs, 
                                num_steps=num_steps, 
                                a_size=a_size,
                                s_size=s_size, 
                                sharpness=sharpness, 
                                n_q=n_q, n_v=n_v, n_w=n_w
                                )
                            print(loss_record)
                            # loss_plot(loss_record)
                            test(EA_save_name=name)

