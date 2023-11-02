from abm.start_sim import reconstruct_NN
from abm.sprites.agent import Agent

import dotenv as de
from pathlib import Path
import numpy as np
import torch
import multiprocessing as mp
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation


# -------------------------- neuron activity -------------------------- #

def agent_Nact_from_xyo(envconf, NN, boundary_endpts, x, y, orient):

    agent = Agent(
            id=0,
            position=(x,y),
            orientation=orient,
            max_vel=int(envconf["MAXIMUM_VELOCITY"]),
            FOV=float(envconf['AGENT_FOV']),
            vis_field_res=int(envconf["VISUAL_FIELD_RESOLUTION"]),
            vision_range=int(envconf["VISION_RANGE"]),
            num_class_elements=4,
            consumption=1,
            model=NN,
            boundary_endpts=boundary_endpts,
            window_pad=30,
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


def build_Nact_matrix_parallel(exp_name, gen_ext, space_step, orient_step):

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
    print(f'Nact matrix shape (x, y, orient, # neurons): {Nact_matrix.shape}')

    # pack inputs for multiprocessing map
    mp_inputs = []
    for x in x_range:
        for y in y_range:
            for orient in orient_range:
                mp_inputs.append( (envconf, NN, boundary_endpts, x, y, orient) )
    
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


def plot_Nact_imshow(exp_name, gen_ext, space_step, orient_step):

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
            ax[o,n].imshow(np.transpose(mat[:,:,o,n]), # since imshow plots (y,x)
                           vmin = mat[:,:,:,n].flatten().min(), 
                           vmax = mat[:,:,:,n].flatten().max(),
                           )
            # ax[o,n].set_title(f'orient: {int(orient_range[o]*180/np.pi)} | neuron: {n}', fontsize=10)

    plt.savefig(fr'{data_dir}/Nact_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}.png')
    plt.close()


def anim_Nact(exp_name, gen_ext, space_step, orient_step):

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    with open(fr'{data_dir}/Nact_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}.bin', 'rb') as f:
        mat = pickle.load(f)
    
    num_x, num_y, num_orient, num_neurons = mat.shape
    
    fig = plt.figure()
    frames = []

    n = -1 # output neuron only

    min = mat[:,:,:,n].flatten().min()
    max = mat[:,:,:,n].flatten().max()

    for o in range(num_orient):

        frames.append([plt.imshow(np.transpose(mat[:,:,o,0]), vmin = min, vmax = max, animated=True)])

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
    
    ani.save(fr'{data_dir}/Nact_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}.mp4')
    plt.close()



# -------------------------- early trajectory -------------------------- #

def agent_traj_from_xyo(envconf, NN, boundary_endpts, x, y, orient, timesteps):

    agent = Agent(
            id=0,
            position=(x,y),
            orientation=orient,
            max_vel=int(envconf["MAXIMUM_VELOCITY"]),
            FOV=float(envconf['AGENT_FOV']),
            vis_field_res=int(envconf["VISUAL_FIELD_RESOLUTION"]),
            vision_range=int(envconf["VISION_RANGE"]),
            num_class_elements=4,
            consumption=1,
            model=NN,
            boundary_endpts=boundary_endpts,
            window_pad=30,
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


def build_agent_trajs_parallel(exp_name, gen_ext, space_step, orient_step, timesteps):

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
                mp_inputs.append( (envconf, NN, boundary_endpts, x, y, orient, timesteps) )
    
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
    # ag_data = np.zeros((2,1,3)) # 2 patches
    res_radius = int(envconf["RADIUS_RESOURCE"])
    x,y = width*.4, height*.4

    ag_data[0,0,:] = np.array((x, y_max - y, res_radius))
    # ag_data[1,0,:] = np.array((x, y_max - y, res_radius))

    # pack + save
    plot_data = traj_matrix, ag_data
    Path(fr'{data_dir}/traj_matrices').mkdir(parents=True, exist_ok=True)
    with open(fr'{data_dir}/traj_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}.bin', 'wb') as f:
        pickle.dump(plot_data, f)


def plot_agent_trajs(exp_name, gen_ext, space_step, orient_step, timesteps):

    from abm.monitoring.plot_funcs import plot_map_iterative_traj

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)

    width=int(envconf["ENV_WIDTH"])
    height=int(envconf["ENV_HEIGHT"])

    with open(fr'{data_dir}/traj_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}.bin', 'rb') as f:
        traj_plot_data = pickle.load(f)

    save_name = fr'{data_dir}/traj_matrices/{exp_name}_{gen_ext}_traj_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_ap2'
    plot_map_iterative_traj(traj_plot_data, x_max=width, y_max=height, save_name=save_name)


# -------------------------- stationary vis -------------------------- #

def agent_visfield_from_xyo(envconf, NN, boundary_endpts, x, y, orient):

    agent = Agent(
            id=0,
            position=(x,y),
            orientation=orient,
            max_vel=int(envconf["MAXIMUM_VELOCITY"]),
            FOV=float(envconf['AGENT_FOV']),
            vis_field_res=int(envconf["VISUAL_FIELD_RESOLUTION"]),
            vision_range=int(envconf["VISION_RANGE"]),
            num_class_elements=4,
            consumption=1,
            model=NN,
            boundary_endpts=boundary_endpts,
            window_pad=30,
            radius=int(envconf["RADIUS_AGENT"]),
            color=(0,0,0),
        )

    agent.visual_sensing([])

    field_int = 0

    for i in agent.vis_field:
        if i == 'wall_north': pass
        elif i == 'wall_east': field_int += 1
        elif i == 'wall_south': field_int += 2
        elif i == 'wall_west': field_int += 3
        else: print('error')

    return field_int


def build_agent_visfield_parallel(exp_name, gen_ext, space_step, orient_step):

    # pull pv + envconf from save folders
    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    NN_pv_path = fr'{data_dir}/{exp_name}/{gen_ext}_NN0_pickle.bin'
    with open(NN_pv_path,'rb') as f:
        pv = pickle.load(f)
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)
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

    visfield_dim = 1

    visfield_matrix = np.zeros((len(x_range),
                            len(y_range),
                            len(orient_range),
                            visfield_dim))
    print(f'visfield matrix shape (x, y, orient, vis_field dims): {visfield_matrix.shape}')

    # pack inputs for multiprocessing map
    mp_inputs = []
    for x in x_range:
        for y in y_range:
            for orient in orient_range:
                mp_inputs.append( (envconf, NN, boundary_endpts, x, y, orient) )
    
    # run agent NNs in parallel
    with mp.Pool() as pool:
        results = pool.starmap_async( agent_visfield_from_xyo, mp_inputs)
        pool.close()
        pool.join()

    # unpack results into matrix + save
    results_list = results.get()
    n = 0
    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            for k, orient in enumerate(orient_range):
                visfield_matrix[i,j,k,:] = results_list[n]
                n += 1

    Path(fr'{data_dir}/visfield_matrices').mkdir(parents=True, exist_ok=True)
    with open(fr'{data_dir}/visfield_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}.bin', 'wb') as f:
        pickle.dump(visfield_matrix, f)


def plot_visfield_imshow(exp_name, gen_ext, space_step, orient_step):

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    with open(fr'{data_dir}/visfield_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}.bin', 'rb') as f:
        mat = pickle.load(f)

    num_x, num_y, num_orient, num_visfield_dims = mat.shape
    orient_range = np.linspace(0, 2*np.pi, num_orient+1)[:-1]

    fig, axs = plt.subplots(
        nrows=num_orient, 
        ncols=num_visfield_dims, 
        figsize=(25, 25), 
        subplot_kw={'xticks': [], 'yticks': []}
        )

    # plot spatial activity for every neuron (columns) at every orientation (rows)
    
    ax = np.array(axs)
    for n in range(num_visfield_dims):

        min = mat[:,:,:,n].flatten().min()
        max = mat[:,:,:,n].flatten().max()
        
        for o in range(num_orient):

            # transpose space since imshow plots (y,x)
            # normalize min/max activity for each neuron with respect to itself (at every position x orientation)
        
            ax[o].imshow(np.transpose(mat[:,:,o,n]), vmin = min, vmax = max)

    plt.savefig(fr'{data_dir}/visfield_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}.png')
    plt.close()


def anim_visfield(exp_name, gen_ext, space_step, orient_step):

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    with open(fr'{data_dir}/visfield_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}.bin', 'rb') as f:
        mat = pickle.load(f)
    
    num_x, num_y, num_orient, num_visfield_dims = mat.shape
    
    fig = plt.figure()
    frames = []

    for n in range(num_visfield_dims):

        min = mat[:,:,:,n].flatten().min()
        max = mat[:,:,:,n].flatten().max()

        for o in range(num_orient):

            frames.append([plt.imshow(np.transpose(mat[:,:,o,0]), vmin = min, vmax = max, animated=True)])

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
    
    ani.save(fr'{data_dir}/visfield_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}.mp4')
    plt.close()



# -------------------------- traj + Nact + vis -------------------------- #

def agent_trajall_from_xyo(envconf, NN, boundary_endpts, x, y, orient, timesteps, vis_field_res, Nact_size):

    agent = Agent(
            id=0,
            position=(x,y),
            orientation=orient,
            max_vel=2,
            FOV=0.4,
            vis_field_res=vis_field_res,
            vision_range=2000,
            num_class_elements=4,
            consumption=1,
            model=NN,
            boundary_endpts=boundary_endpts,
            window_pad=30,
            radius=10,
            color=(0,0,0),
        )
    
    # from null initial activity (+ constant zero food presence sense)
    hidden = torch.zeros(int(envconf["RNN_HIDDEN_SIZE"])).unsqueeze(0)
    other_input = torch.zeros( int(envconf["RNN_OTHER_INPUT_SIZE"]) ).unsqueeze(0)

    # initialize data storing matrix + step through trajectory
    data_vector_size = 3 + vis_field_res + Nact_size # first 3 are for (x,y,o)
    traj_data = np.zeros((timesteps, data_vector_size)) 
    for t in range(timesteps):

        # store current positional info
        x,y = agent.position
        traj_data[t,:3] = np.array((x,y,agent.orientation))

        # gather visual input
        agent.visual_sensing([])
        vis_field_onehot = agent.encode_one_hot(agent.vis_field)

        # translate to vector + store
        vis_field = np.zeros(vis_field_res)
        for n,i in enumerate(agent.vis_field):
            if i == 'wall_north': vis_field[n] = 0
            elif i == 'wall_east': vis_field[n] = 1
            elif i == 'wall_south': vis_field[n] = 2
            else: # i == 'wall_west': 
                vis_field[n] = 3
        traj_data[t,3:3+vis_field_res] = vis_field

        # pass through model.forward
        vis_input = torch.from_numpy(vis_field_onehot).float().unsqueeze(0)
        vis_features = agent.model.cnn(vis_input)
        RNN_in = torch.cat((vis_features, other_input), dim=1)
        RNN_out, hidden = agent.model.rnn(RNN_in, hidden)
        action = agent.model.lcl(RNN_out)
        action = torch.tanh(action)
        action = action

        # align neural activities + store
        Nact = np.concatenate((vis_features, RNN_out, action), axis=None)
        traj_data[t,-Nact_size:] = Nact

        # move agent
        agent.move(action.detach().numpy()[0][0])

    return traj_data

def build_agent_trajalls_parallel(exp_name, gen_ext, space_step, orient_step, timesteps):

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

    # calculate data vector size
    vis_field_res = int(envconf["VISUAL_FIELD_RESOLUTION"])
    encoded_vis_size = list(map(int,envconf["CNN_DIMS"].split(',')))[-1]
    rnn_hidden_size = int(envconf["RNN_HIDDEN_SIZE"])
    lcl_output_size = int(envconf["LCL_OUTPUT_SIZE"])
    Nact_size = encoded_vis_size + rnn_hidden_size + lcl_output_size
    data_vector_size = 3 + vis_field_res + Nact_size # first 3 are for (x,y,o)
    
    # construct matrix of each traj for each grid pos/dir
    num_inits = len(x_range) * len(y_range) * len(orient_range)
    data_matrix = np.zeros( (num_inits, timesteps, data_vector_size) ) 
    print(f'traj matrix shape (initializations, timesteps, stored values): {data_matrix.shape}')
    
    # pack inputs for multiprocessing map
    mp_inputs = []
    for x in x_range:
        for y in y_range:
            for orient in orient_range:
                mp_inputs.append( (envconf, NN, boundary_endpts, x, y, orient, timesteps, vis_field_res, Nact_size) )
    
    # run agent NNs in parallel
    with mp.Pool() as pool:
        results = pool.starmap_async( agent_trajall_from_xyo, mp_inputs )
        pool.close()
        pool.join()

    # unpack results into matrix
    results_list = results.get()
    for t, traj_data in enumerate(results_list):
        data_matrix[t,:,:] = traj_data
    
    # transform data to plotting coords
    # traj_matrix[:,:,0] = traj_matrix[:,:,0] # pos_x = pos_x
    data_matrix[:,:,1] = y_max - data_matrix[:,:,1]

    # build resource coord matrix
    ag_data = np.zeros((1,1,3)) # 1 patch
    # ag_data = np.zeros((2,1,3)) # 2 patches
    res_radius = int(envconf["RADIUS_RESOURCE"])
    x,y = width*.4, height*.4

    ag_data[0,0,:] = np.array((x, y_max - y, res_radius))
    # ag_data[1,0,:] = np.array((x, y_max - y, res_radius))

    # pack + save
    plot_data = data_matrix, ag_data
    Path(fr'{data_dir}/trajall_matrices').mkdir(parents=True, exist_ok=True)
    with open(fr'{data_dir}/trajall_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}.bin', 'wb') as f:
        pickle.dump(plot_data, f)
    

def plot_agent_trajalls(exp_name, gen_ext, space_step, orient_step, timesteps):

    from abm.monitoring.plot_funcs import plot_map_iterative_trajall

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)

    width=int(envconf["ENV_WIDTH"])
    height=int(envconf["ENV_HEIGHT"])

    with open(fr'{data_dir}/trajall_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}.bin', 'rb') as f:
        traj_plot_data = pickle.load(f)

    save_name = fr'{data_dir}/trajall_matrices/{exp_name}_{gen_ext}_traj_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_vischange'
    plot_map_iterative_trajall(traj_plot_data, x_max=width, y_max=height, save_name=save_name, var_pos=3, change=True)
    save_name = fr'{data_dir}/trajall_matrices/{exp_name}_{gen_ext}_traj_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_visflat'
    plot_map_iterative_trajall(traj_plot_data, x_max=width, y_max=height, save_name=save_name, var_pos=3, change=False)

    # loop over encoded vis (CNN output)
    vis_field_res = int(envconf["VISUAL_FIELD_RESOLUTION"])
    encoded_vis_size = list(map(int,envconf["CNN_DIMS"].split(',')))[-1]
    for i in range(encoded_vis_size):
        var_pos = 3 + vis_field_res + i
        save_name = fr'{data_dir}/trajall_matrices/{exp_name}_{gen_ext}_traj_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_NactCNN{i}'
        plot_map_iterative_trajall(traj_plot_data, x_max=width, y_max=height, save_name=save_name, var_pos=var_pos)

    save_name = fr'{data_dir}/trajall_matrices/{exp_name}_{gen_ext}_traj_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_action'
    plot_map_iterative_trajall(traj_plot_data, x_max=width, y_max=height, save_name=save_name, var_pos=-1)
    save_name = fr'{data_dir}/trajall_matrices/{exp_name}_{gen_ext}_traj_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_action_inv'
    plot_map_iterative_trajall(traj_plot_data, x_max=width, y_max=height, save_name=save_name, var_pos=-1, inv=True)


# -------------------------- misc -------------------------- #

def find_top_val_gen(exp_name):

    # parse val results text file
    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    with open(fr'{data_dir}/{exp_name}/val_results.txt') as f:
    # with open(fr'{data_dir}/{exp_name}/val_results_cen.txt') as f:
        lines = f.readlines()

        val_data = np.zeros((len(lines)-1, 3))
        for i, line in enumerate(lines[1:]):
            data = [item.strip() for item in line.split(' ')]
            val_data[i,0] = data[1] # generation
            val_data[i,1] = data[4] # train fitness
            val_data[i,2] = data[7] # val fitness

        # sort according to val fitness
        top_ind = np.argsort(val_data[:,2])[0] 
        top_gen = int(val_data[top_ind,0])
        top_valfit = int(val_data[top_ind,2])
        # print(f'gen {top_gen}: fit {top_valfit}')
    return f'gen{top_gen}', top_valfit


if __name__ == '__main__':

    # space_step = 5
    # orient_step = np.pi/256

    space_step = 25
    orient_step = np.pi/8

    # space_step = 500
    # orient_step = np.pi/2

    timesteps = 500

    names = [
    #     'singlecorner_exp_CNN1124_FNN2_p100e20_vis8_rep1',
    #     'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss075_rep1'
        'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_rep3',
        # 'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_rep4',
        # 'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_rep1',
        ]

    for name in names:
        gen, valfit = find_top_val_gen(name)
        print(f'build/plot matrix for: {name} @ {gen} w {valfit} fitness')
        
        # build_Nact_matrix_parallel(name, gen, space_step, orient_step)
        # plot_Nact_imshow(name, gen, space_step, orient_step)
        # anim_Nact(name, space_step, orient_step)

        # build_agent_trajs_parallel(name, gen, space_step, orient_step, timesteps)
        # plot_agent_trajs(name, gen, space_step, orient_step, timesteps)

        # build_agent_visfield_parallel(name, space_step, orient_step)
        # plot_visfield_imshow(name, space_step, orient_step)
        # anim_visfield(name, space_step, orient_step)

        build_agent_trajalls_parallel(name, gen, space_step, orient_step, timesteps)
        plot_agent_trajalls(name, gen, space_step, orient_step, timesteps)