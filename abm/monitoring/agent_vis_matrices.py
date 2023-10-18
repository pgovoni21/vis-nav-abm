from abm.start_sim import reconstruct_NN
from abm.sprites.agent import Agent

import dotenv as de
from pathlib import Path
import numpy as np
import torch
import multiprocessing as mp
import pickle
import matplotlib.pyplot as plt


# -------------------------- neuron activity -------------------------- #

def agent_Nact_from_xyo(envconf, arch, NN, boundary_endpts, x, y, orient):

    agent = Agent(
            id=0,
            position=(x,y),
            orientation=orient,
            max_vel=int(envconf["MAXIMUM_VELOCITY"]),
            FOV=float(envconf['AGENT_FOV']),
            vision_range=int(envconf["VISION_RANGE"]),
            visual_exclusion=bool(int(envconf["VISUAL_EXCLUSION"])),
            consumption=1,
            arch=arch,
            model=NN,
            RNN_type=str(envconf["RNN_TYPE"]),
            NN_activ=str(envconf["NN_ACTIVATION_FUNCTION"]),
            boundary_endpts=boundary_endpts,
            radius=int(envconf["RADIUS_AGENT"]),
            color=(0,0,0),
        )

    # gather visual input
    agent.visual_sensing([])
    vis_field_onehot = agent.encode_one_hot(agent.vis_field)

    # pass through CNN
    vis_input = torch.from_numpy(vis_field_onehot).float().unsqueeze(0)
    vis_features = agent.model.cnn(vis_input)

    # from null initial activity (+ zero contact field)
    other_input = torch.zeros( int(envconf["RNN_OTHER_INPUT_SIZE"]) ).unsqueeze(0)
    hidden = torch.zeros(int(envconf["RNN_HIDDEN_SIZE"])).unsqueeze(0)

    RNN_in = torch.cat((vis_features, other_input), dim=1)
    RNN_out, hidden = agent.model.rnn(RNN_in, hidden)

    action = agent.model.lcl(RNN_out)
    action = torch.tanh(action) # scale to [-1:1]

    Nact = np.concatenate((vis_features, RNN_out, action), axis=None)
    
    return Nact



def build_Nact_matrix_parallel(name, space_step, orient_step):

    exp_name, gen_ext = name

    # pull pv + envconf from save folders
    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    NN_pv_path = fr'{data_dir}/{exp_name}/{gen_ext}_NN0_pickle.bin'
    with open(NN_pv_path,'rb') as f:
        pv = pickle.load(f)
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)

    # reconstruct model
    NN, arch = reconstruct_NN(envconf, pv)

    # construct boundary endpts
    width=int(envconf["ENV_WIDTH"])
    height=int(envconf["ENV_HEIGHT"])
    x_min, x_max = 0, width
    y_min, y_max = 0, height
    boundary_endpts = [
            np.array([ x_min, y_min ]),
            np.array([ x_max, y_min ]),
            np.array([ x_min, y_max ]),
            np.array([ x_max, y_max ])
            ]

    # every grid position/direction
    coll_boundary_thickness = int(envconf["RADIUS_AGENT"])
    x_range = np.arange(x_min + coll_boundary_thickness, x_max - coll_boundary_thickness + 1, space_step)
    y_range = np.arange(y_min + coll_boundary_thickness, y_max - coll_boundary_thickness + 1, space_step)
    orient_range = np.arange(0, 2*np.pi, orient_step) 

    # construct matrix of each neuron activity for each grid pos/dir
    vis_field_res = list(map(int,envconf["CNN_DIMS"].split(',')))[-1]
    rnn_hidden_size = int(envconf["RNN_HIDDEN_SIZE"])
    lcl_output_size = int(envconf["LCL_OUTPUT_SIZE"])
    Nact_size = vis_field_res + rnn_hidden_size + lcl_output_size

    Nact_matrix = np.zeros((len(x_range),
                            len(y_range),
                            len(orient_range),
                            Nact_size))
    print(f'Nact matrix shape (x, y, orient, vis_field_res): {Nact_matrix.shape}')

    # pack inputs for multiprocessing map
    mp_inputs = []
    for x in x_range:
        for y in y_range:
            for orient in orient_range:
                mp_inputs.append( (envconf, arch, NN, boundary_endpts, x, y, orient) )
    
    # run agent NNs in parallel
    with mp.Pool() as pool:
        results = pool.starmap_async( agent_Nact_from_xyo, mp_inputs)
        pool.close()
        pool.join()

    # unpack results into matrix + save
    results_list = results.get()
    n = 0
    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            for k, orient in enumerate(orient_range):
                Nact_matrix[i,j,k,:] = results_list[n]
                n += 1

    Path(fr'{data_dir}/Nact_matrices').mkdir(parents=True, exist_ok=True)
    with open(fr'{data_dir}/Nact_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}.bin', 'wb') as f:
        pickle.dump(Nact_matrix, f)
    
    return Nact_matrix

def plot_Nactmat_imshow(name, space_step, orient_step):

    exp_name, gen_ext = name

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    with open(fr'{data_dir}/Nact_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}.bin', 'rb') as f:
        mat = pickle.load(f)

    num_x, num_y, num_orient, num_neurons = mat.shape
    orient_range = np.linspace(0, 2*np.pi, num_orient+1)[:-1]

    fig, axs = plt.subplots(
        nrows=num_orient, 
        ncols=num_neurons, 
        figsize=(25, 25), 
        subplot_kw={'xticks': [], 'yticks': []}
        )

    # plot spatial activity for every neuron (columns) at every orientation (rows)
    # normalize min/max activity for each neuron with respect to itself (at every position x orientation)
    ax = np.array(axs)
    for o in range(num_orient):
        for n in range(num_neurons):
            ax[o,n].imshow(mat[:,:,o,n],
                           vmin = mat[:,:,:,n].flatten().min(), 
                           vmax = mat[:,:,:,n].flatten().max(),
                           )
            # ax[o,n].set_title(f'orient: {int(orient_range[o]*180/np.pi)} | neuron: {n}', fontsize=10)

    plt.savefig(fr'{data_dir}/Nact_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}.png')
    plt.close()


# -------------------------- early trajectory -------------------------- #

def agent_traj_from_xyo(envconf, arch, NN, boundary_endpts, x, y, orient, timesteps):

    agent = Agent(
            id=0,
            position=(x,y),
            orientation=orient,
            max_vel=int(envconf["MAXIMUM_VELOCITY"]),
            FOV=float(envconf['AGENT_FOV']),
            vision_range=int(envconf["VISION_RANGE"]),
            visual_exclusion=bool(int(envconf["VISUAL_EXCLUSION"])),
            consumption=1,
            arch=arch,
            model=NN,
            RNN_type=str(envconf["RNN_TYPE"]),
            NN_activ=str(envconf["NN_ACTIVATION_FUNCTION"]),
            boundary_endpts=boundary_endpts,
            radius=int(envconf["RADIUS_AGENT"]),
            color=(0,0,0),
        )

    traj = np.zeros((timesteps,2))
    for t in range(timesteps):

        agent.visual_sensing([])
        vis_field_onehot = agent.encode_one_hot(agent.vis_field)
        agent.action, agent.hidden = agent.model.forward(vis_field_onehot, np.array([0]), agent.hidden)
        agent.move(agent.action)

        traj[t,:] = agent.position
    
    return traj


def build_agent_trajs_parallel(name, space_step, orient_step, timesteps):

    exp_name, gen_ext = name

    # pull pv + envconf from save folders
    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    NN_pv_path = fr'{data_dir}/{exp_name}/{gen_ext}_NN0_pickle.bin'
    with open(NN_pv_path,'rb') as f:
        pv = pickle.load(f)
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)

    # reconstruct model
    NN, arch = reconstruct_NN(envconf, pv)

    # construct boundary endpts
    width=int(envconf["ENV_WIDTH"])
    height=int(envconf["ENV_HEIGHT"])
    x_min, x_max = 0, width
    y_min, y_max = 0, height
    boundary_endpts = [
            np.array([ x_min, y_min ]),
            np.array([ x_max, y_min ]),
            np.array([ x_min, y_max ]),
            np.array([ x_max, y_max ])
            ]

    # every grid position/direction
    coll_boundary_thickness = int(envconf["RADIUS_AGENT"])
    x_range = np.arange(x_min + coll_boundary_thickness, x_max - coll_boundary_thickness + 1, space_step)
    y_range = np.arange(y_min + coll_boundary_thickness, y_max - coll_boundary_thickness + 1, space_step)
    orient_range = np.arange(0, 2*np.pi, orient_step)
    
    # construct matrix of each traj for each grid pos/dir
    num_inits = len(x_range) * len(y_range) * len(orient_range)
    traj_matrix = np.zeros( (num_inits, timesteps, 4) ) # (pos_x, pos_y, _, _) --> to match self.data_agent format
    print(f'traj matrix shape (# initializations, timesteps, ): {traj_matrix.shape}')
    
    # pack inputs for multiprocessing map
    mp_inputs = []
    for x in x_range:
        for y in y_range:
            for orient in orient_range:
                mp_inputs.append( (envconf, arch, NN, boundary_endpts, x, y, orient, timesteps) )
    
    # run agent NNs in parallel
    with mp.Pool() as pool:
        results = pool.starmap_async( agent_traj_from_xyo, mp_inputs)
        pool.close()
        pool.join()

    # unpack results into matrix (y coords transformed for plotting)
    results_list = results.get()
    for n,pos_array in enumerate(results_list):
        traj_matrix[n,:,:2] = pos_array
    
    # transform data to plotting coords
    # traj_matrix[:,:,0] = traj_matrix[:,:,0] # pos_x = pos_x
    traj_matrix[:,:,1] = y_max - traj_matrix[:,:,1]

    # build resource coord matrix
    ag_data = np.zeros((1,1,3)) # 1 patch
    ag_data = np.zeros((2,1,3)) # 2 patches
    res_radius = int(envconf["RADIUS_RESOURCE"])
    x,y = width*.4, height*.4

    ag_data[0,0,:] = np.array((x, y_max - y, res_radius))
    # ag_data[1,0,:] = np.array((x, y, res_radius))

    # pack + save
    plot_data = traj_matrix, ag_data
    Path(fr'{data_dir}/traj_matrices').mkdir(parents=True, exist_ok=True)
    with open(fr'{data_dir}/traj_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}.bin', 'wb') as f:
        pickle.dump(plot_data, f)


def plot_agent_trajs(name, space_step, orient_step, timesteps):

    from abm.monitoring.plot_funcs import plot_map_iterative_traj

    exp_name, gen_ext = name

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)

    width=int(envconf["ENV_WIDTH"])
    height=int(envconf["ENV_HEIGHT"])

    with open(fr'{data_dir}/traj_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}.bin', 'rb') as f:
        traj_plot_data = pickle.load(f)

    save_name = fr'{data_dir}/traj_matrices/{exp_name}_{gen_ext}_traj_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_ap2'
    plot_map_iterative_traj(traj_plot_data, x_max=width, y_max=height, save_name=save_name)


if __name__ == '__main__':

    space_step = 25
    orient_step = np.pi/8
    timesteps = 500

    names = []


    # exp_name = 'singlecorner_exp_CNN1124_FNN2_p50e15_vis8_rep0'
    # gen_ext = 'gen961' # 388
    # names.append((exp_name,gen_ext))
    # exp_name = 'singlecorner_exp_CNN1124_FNN2_p50e15_vis8_rep1'
    # gen_ext = 'gen661' # 284
    # names.append((exp_name,gen_ext))
    # exp_name = 'singlecorner_exp_CNN1124_FNN2_p50e15_vis8_rep2'
    # gen_ext = 'gen666' # 286
    # names.append((exp_name,gen_ext))
    # exp_name = 'singlecorner_exp_CNN1124_FNN2_p50e15_vis8_rep3'
    # gen_ext = 'gen908' # 298
    # names.append((exp_name,gen_ext))
    exp_name = 'singlecorner_exp_CNN1124_FNN2_p50e15_vis8_rep4'
    gen_ext = 'gen837' # 292
    names.append((exp_name,gen_ext))

    exp_name = 'singlecorner_exp_CNN1124_FNN2_p50e15_vis8_rep1'
    gen_ext = 'gen944' # 291
    names.append((exp_name,gen_ext))
    exp_name = 'singlecorner_exp_CNN1124_FNN2_p50e15_vis8_rep2'
    gen_ext = 'gen944' # 297
    names.append((exp_name,gen_ext))
    exp_name = 'singlecorner_exp_CNN1124_FNN2_p50e15_vis8_rep4'
    gen_ext = 'gen971' # 302
    names.append((exp_name,gen_ext))


    # exp_name = 'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_rep0'
    # gen_ext = 'gen956' # 291
    # names.append((exp_name,gen_ext))
    # exp_name = 'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_rep1'
    # gen_ext = 'gen857' # 279
    # names.append((exp_name,gen_ext))
    # exp_name = 'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_rep2'
    # gen_ext = 'gen804' # 284
    # names.append((exp_name,gen_ext))


    for name in names:
        print(f'build/plot matrix for: {name}')
        build_Nact_matrix_parallel(name, space_step, orient_step)
        plot_Nactmat_imshow(name, space_step, orient_step)
        build_agent_trajs_parallel(name, space_step, orient_step, timesteps) # doesn't have wall collisions - is this ok?
        plot_agent_trajs(name, space_step, orient_step, timesteps)