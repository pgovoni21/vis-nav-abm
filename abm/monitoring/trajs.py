from abm.start_sim import reconstruct_NN
from abm.sprites.agent import Agent
# from abm.sprites.agent_LM import Agent
from abm.sprites.landmark import Landmark

import dotenv as de
from pathlib import Path
import numpy as np
import scipy
import torch
import multiprocessing as mp
import _pickle as pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


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

    # Nact = np.concatenate((vis_features, RNN_out, action), axis=None)
    Nact = action
    
    return Nact


def build_Nact_matrix_parallel(exp_name, gen_ext, space_step, orient_step):

    # pull pv + envconf from save folders
    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    # NN_pv_path = fr'{data_dir}/{exp_name}/{gen_ext}_NN0_pickle.bin'
    NN_pv_path = fr'{data_dir}/{exp_name}/{gen_ext}_NNcen_pickle.bin'
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
    X, Y = np.meshgrid(np.linspace(0, 40, num_x), np.linspace(0, 40, num_y))
    print(X.shape)
    for n in range(num_neurons):

        min_act = mat[:,:,:,n].flatten().min()
        max_act = mat[:,:,:,n].flatten().max()
        act_bound = max(abs(min_act),abs(max_act))
        print(n, min_act, max_act)

        for o in range(num_orient):
            ax[o,n].imshow(np.transpose(mat[:,:,o,n]), # since imshow plots (y,x)
                        #    vmin = min_act, 
                        #    vmax = max_act,
                           vmin = -act_bound, 
                           vmax = act_bound,
                           )
            # ax[o,n].set_title(f'orient: {int(orient_range[o]*180/np.pi)} | neuron: {n}', fontsize=10)

            # convert actions into relative orientation changes
            actions = np.transpose(mat[:,:,o,-1])
            turns = actions * np.pi / 2
            change = turns + orient_range[o]

            U = np.cos(change)
            V = np.sin(change)
            
            # ax[int(n/2)].streamplot( X, Y, U, V )
            # ax[o,-1].quiver( X, Y, U, V, pivot='mid' )
            ax[o,-1].quiver( X[::2,::2], Y[::2,::2], U[::2,::2], V[::2,::2], pivot='mid' )

    plt.savefig(fr'{data_dir}/Nact_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}.png')
    plt.close()


def plot_Nact_action_phase(exp_name, gen_ext, space_step, orient_step):

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    with open(fr'{data_dir}/Nact_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}.bin', 'rb') as f:
        mat = pickle.load(f)

    num_x, num_y, num_orient, num_neurons = mat.shape
    orient_range = np.linspace(0, 2*np.pi, num_orient+1)[:-1]

    fig, axs = plt.subplots(
        nrows=num_orient, 
        ncols=1, 
        figsize=(1.8, 32), 
        # figsize=(4, 32),
        subplot_kw={'xticks': [], 'yticks': []}
        )

    # set up space
    X, Y = np.meshgrid(np.linspace(0, 100, num_x), np.linspace(0, 100, num_y))

    # plot for every orientation
    ax = np.array(axs)
    for o in range(num_orient):

        # convert actions into relative orientation changes
        actions = np.flip(np.transpose(mat[:,:,o,-1]),axis=0)
        turns = actions * np.pi / 2
        change = turns + orient_range[o]

        U = np.cos(change)
        V = np.sin(change)
        
        # ax[int(n/2)].streamplot( X, Y, U, V )
        ax[o].quiver( X[::2,::2], Y[::2,::2], U[::2,::2], V[::2,::2], pivot='mid' )

    plt.savefig(fr'{data_dir}/Nact_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_action_phase.png')
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


# -------------------------- action -------------------------- #

def agent_action_from_xyo(envconf, NN, boundary_endpts, x, y, orient):

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
            vis_transform='',
            # vis_transform=str(envconf["VIS_TRANSFORM"]),
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
    
    return action


def build_action_matrix(exp_name, gen_ext, space_step, orient_step):

    # pull pv + envconf from save folders
    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    NN_pv_path = fr'{data_dir}/{exp_name}/{gen_ext}_NNcen_pickle.bin'
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
    x_range = np.linspace(x_min + coll_boundary_thickness, 
                        x_max - coll_boundary_thickness + 1, 
                        int((width - coll_boundary_thickness*2) / space_step))
    y_range = np.linspace(y_min + coll_boundary_thickness, 
                        y_max - coll_boundary_thickness + 1, 
                        int((height - coll_boundary_thickness*2) / space_step))
    orient_range = np.arange(0, 2*np.pi, orient_step) 

    # construct matrix for each grid pos/dir
    act_matrix = np.zeros((len(x_range),
                            len(y_range),
                            len(orient_range),
                            ))
    print(f'act matrix shape (x, y, orient): {act_matrix.shape}')


    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            for k, orient in enumerate(orient_range):
                act_matrix[i,j,k] = agent_action_from_xyo(envconf, NN, boundary_endpts, x, y, orient)


    # # pack inputs for multiprocessing map
    # mp_inputs = []
    # for x in x_range:
    #     for y in y_range:
    #         for orient in orient_range:
    #             mp_inputs.append( (envconf, NN, boundary_endpts, x, y, orient) )
    
    # # run agent NNs in parallel
    # with mp.Pool() as pool:
    #     results = pool.starmap_async( agent_action_from_xyo, mp_inputs)
    #     pool.close()
    #     pool.join()

    # # unpack results into matrix + save
    # results_list = results.get()
    # n = 0
    # for i, x in enumerate(x_range):
    #     for j, y in enumerate(y_range):
    #         for k, orient in enumerate(orient_range):
    #             act_matrix[i,j,k] = results_list[n]
    #             n += 1

    with open(fr'{data_dir}/act_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}.bin', 'wb') as f:
        pickle.dump(act_matrix, f)
    
    return act_matrix


def plot_action_volume(exp_name, gen_ext, space_step, orient_step, transform='high'):

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    with open(fr'{data_dir}/act_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}.bin', 'rb') as f:
        mat = pickle.load(f)

    # NN_pv_path = fr'{data_dir}/{exp_name}/{gen_ext}_NN0_pickle.bin'
    NN_pv_path = fr'{data_dir}/{exp_name}/{gen_ext}_NNcen_pickle.bin'
    with open(NN_pv_path,'rb') as f:
        pv = pickle.load(f)
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)

    # gather grid params
    x_min, x_max = 0, int(envconf["ENV_WIDTH"])
    y_min, y_max = 0, int(envconf["ENV_HEIGHT"])
    coll_boundary_thickness = int(envconf["RADIUS_AGENT"])
    x_range = np.linspace(x_min + coll_boundary_thickness, 
                        x_max - coll_boundary_thickness + 1, 
                        int((x_max - coll_boundary_thickness*2) / space_step))
    y_range = np.linspace(y_min + coll_boundary_thickness, 
                        y_max - coll_boundary_thickness + 1, 
                        int((y_max - coll_boundary_thickness*2) / space_step))
    orient_range = np.arange(0, 2*np.pi, orient_step) 

    # construct vectors for each grid pos/dir
    xs, ys, os = [], [], []
    for x in x_range:
        for y in y_range:
            for o in orient_range:
                xs.append(x)
                ys.append(y)
                os.append(o)

    # set up plot
    fig = plt.figure(
        figsize=(25, 25), 
        )
    ax = fig.add_subplot(projection='3d')

    # transform action data
    actions = abs(mat)
    if transform == 'low':
        actions = ( actions.max() - actions ) / actions.max()
    elif transform == 'high':
        actions = actions / actions.max()
    actions = actions.flatten()

    # plot abs(action) for every position (x,y) at every orientation (z)
    ax.scatter(xs, ys, os,
        cmap = 'Blues',
        c = actions,
        alpha = .1*actions,
        s = 100*actions,
        )

    plt.savefig(fr'{data_dir}/act_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_{transform}.png')
    plt.close()


def transform_action_mesh(exp_name, gen_ext, space_step, orient_step):
    
    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    with open(fr'{data_dir}/act_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}.bin', 'rb') as f:
        action_mat = pickle.load(f)

    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)
    max_vel = int(envconf["MAXIMUM_VELOCITY"])
    orient_range = np.arange(0, 2*np.pi, orient_step) 

    # construct meshgrid for change in x, y, o
    dO = action_mat * np.pi/2
    velocity = max_vel * (1 - abs(action_mat))
    dX = velocity * np.cos(orient_range + dO)
    dY = velocity * -np.sin(orient_range + dO)

    print(f'meshgrid shape (dX, dY, dO): {dX.shape}, {dY.shape}, {dO.shape}')
    meshgrid = (dX, dY, dO)

    with open(fr'{data_dir}/act_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_vel.bin', 'wb') as f:
        pickle.dump(meshgrid, f)


def plot_action_volume(exp_name, gen_ext, space_step, orient_step, transform='high'):

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    with open(fr'{data_dir}/act_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}.bin', 'rb') as f:
        mat = pickle.load(f)

    # NN_pv_path = fr'{data_dir}/{exp_name}/{gen_ext}_NN0_pickle.bin'
    NN_pv_path = fr'{data_dir}/{exp_name}/{gen_ext}_NNcen_pickle.bin'
    with open(NN_pv_path,'rb') as f:
        pv = pickle.load(f)
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)

    # gather grid params
    x_min, x_max = 0, int(envconf["ENV_WIDTH"])
    y_min, y_max = 0, int(envconf["ENV_HEIGHT"])
    coll_boundary_thickness = int(envconf["RADIUS_AGENT"])
    x_range = np.linspace(x_min + coll_boundary_thickness, 
                        x_max - coll_boundary_thickness + 1, 
                        int((x_max - coll_boundary_thickness*2) / space_step))
    y_range = np.linspace(y_min + coll_boundary_thickness, 
                        y_max - coll_boundary_thickness + 1, 
                        int((y_max - coll_boundary_thickness*2) / space_step))
    orient_range = np.arange(0, 2*np.pi, orient_step) 

    # construct vectors for each grid pos/dir
    xs, ys, os = [], [], []
    for x in x_range:
        for y in y_range:
            for o in orient_range:
                xs.append(x)
                ys.append(y)
                os.append(o)

    # set up plot
    fig = plt.figure(
        figsize=(25, 25), 
        )
    ax = fig.add_subplot(projection='3d')

    # transform action data
    actions = abs(mat)
    if transform == 'low':
        actions = ( actions.max() - actions ) / actions.max()
    elif transform == 'high':
        actions = actions / actions.max()
    actions = actions.flatten()

    # plot abs(action) for every position (x,y) at every orientation (z)
    ax.scatter(xs, ys, os,
        cmap = 'Blues',
        c = actions,
        alpha = .1*actions,
        s = 100*actions,
        )

    plt.savefig(fr'{data_dir}/act_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_{transform}.png')
    plt.close()


# -------------------------- early trajectory -------------------------- #


def agent_traj_from_xyo(envconf, NN, boundary_endpts, x, y, orient, timesteps, extra=''):

    width, height = tuple(eval(envconf["ENV_SIZE"]))
    window_pad = int(envconf["WINDOW_PAD"])
    other_input = int(envconf["RNN_OTHER_INPUT_SIZE"])
    agent_radius = int(envconf["RADIUS_AGENT"])
    vis_transform = str(envconf["VIS_TRANSFORM"])
    angl_noise_std = float(envconf["PERCEP_ANGLE_NOISE_STD"])
    dist_noise_std = float(envconf["PERCEP_DIST_NOISE_STD"])
    act_noise_std = float(envconf["ACTION_NOISE_STD"])
    LM_dist_noise_std = float(envconf["LM_DIST_NOISE_STD"])
    LM_angle_noise_std = float(envconf["LM_ANGLE_NOISE_STD"])
    LM_radius_noise_std = float(envconf["LM_RADIUS_NOISE_STD"])
    if extra.startswith('n0'):
        angl_noise_std = 0.
        dist_noise_std = 0.
        act_noise_std = 0.
        LM_dist_noise_std = 0.
        LM_angle_noise_std = 0.
        LM_radius_noise_std = 0.
    # if extra.startswith('nhalf'):
    #     angl_noise_std /= 2
    #     LM_noise_std /= 2
    #     dist_noise_std /= 2
    #     act_noise_std /= 2

    max_dist = np.hypot(width, height)
    min_dist = agent_radius*2

    landmarks = []
    if envconf["SIM_TYPE"] == "LM":
        ids = ('TL', 'TR', 'BL', 'BR')
        for id, pos in zip(ids, boundary_endpts):
            landmark = Landmark(
                id=id,
                color=(0,0,0),
                radius=int(envconf["RADIUS_LANDMARK"]),
                position=pos,
                window_pad=int(envconf["WINDOW_PAD"]),
            )
            landmarks.append(landmark)

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
                window_pad=window_pad,
                radius=agent_radius,
                color=(0,0,0),
                vis_transform=vis_transform,
                percep_angle_noise_std=angl_noise_std,
                LM_dist_noise_std=LM_dist_noise_std,
                LM_angle_noise_std=LM_angle_noise_std,
                LM_radius_noise_std=LM_radius_noise_std,
            )
    else:
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
                window_pad=window_pad,
                radius=agent_radius,
                color=(0,0,0),
                vis_transform=vis_transform,
                percep_angle_noise_std=angl_noise_std,
            )


    traj = np.zeros((timesteps,4))
    for t in range(timesteps):

        if not landmarks: agent.visual_sensing([])
        else: agent.visual_sensing(landmarks,[])

        vis_input = agent.encode_one_hot(agent.vis_field)

        if vis_transform != '':
            dist_input = np.array(agent.dist_field)
            if vis_transform == 'close':
                dist_input = (1 - dist_input + max_dist) / max_dist
            elif vis_transform == 'far':
                dist_input = 1/np.power(dist_input/(min_dist), 1)
            elif vis_transform == 'minmax':
                dist_input = (dist_input - min_dist)/(max_dist - min_dist)
            elif vis_transform == 'maxWF':
                dist_input = 1.465 - np.log(dist_input) / 5 # bounds [min, max] within [0, 1]
            elif vis_transform == 'p9WF':
                dist_input = 1.29 - np.log(dist_input) / 6.1 # bounds [min, max] within [0.1, 0.9]
            elif vis_transform == 'p8WF':
                dist_input = 1.09 - np.log(dist_input) / 8.2 # bounds [min, max] within [0.2, 0.8]
            elif vis_transform == 'WF':
                dist_input = 1.24 - np.log(dist_input) / 7 # bounds [min, max] within [0.2, 0.9]
            elif vis_transform == 'mlWF':
                dist_input = 1 - np.log(dist_input) / 9.65 # bounds [min, max] within [0.25, 0.75]
            elif vis_transform == 'mWF':
                dist_input = .9 - np.log(dist_input) / 12 # bounds [min, max] within [0.3, 0.7]
            elif vis_transform == 'msWF':
                dist_input = .8 - np.log(dist_input) / 16 # bounds [min, max] within [0.35, 0.65]
            elif vis_transform == 'sWF':
                dist_input = .7 - np.log(dist_input) / 24 # bounds [min, max] within [0.4, 0.6]
            elif vis_transform == 'ssWF':
                dist_input = .6 - np.log(dist_input) / 48 # bounds [min, max] within [0.45, 0.55]

            noise = np.random.randn(dist_input.shape[0]) * dist_noise_std
            if extra.endswith('pos'): noise[noise > 0] = 0
            elif extra.endswith('neg'): noise[noise < 0] = 0
            dist_input += noise
            # dist_input /= 1.5
            # dist_input += .05
            dist_input = np.clip(dist_input, 0,1)
            vis_input *= dist_input

        if other_input == 2:
            agent.action, agent.hidden = agent.model.forward(vis_input, np.array([0, agent.acceleration / 2]), agent.hidden)
        else:
            agent.action, agent.hidden = agent.model.forward(vis_input, np.array([0]), agent.hidden)

        if not landmarks:
            agent.collided_points = []
            if agent.rect.center[0] < window_pad + 2*agent_radius:
                agent.mode = 'collide'
                collided_pt = np.array(agent.rect.center) - window_pad
                collided_pt[0] -= agent.radius
                agent.collided_points.append(collided_pt)
            if agent.rect.center[0] > window_pad - 2*agent_radius + height:
                agent.mode = 'collide'
                collided_pt = np.array(agent.rect.center) - window_pad
                collided_pt[0] += agent.radius
                agent.collided_points.append(collided_pt)
            if agent.rect.center[1] < window_pad + 2*agent_radius:
                agent.mode = 'collide'
                collided_pt = np.array(agent.rect.center) - window_pad
                collided_pt[1] -= agent.radius
                agent.collided_points.append(collided_pt)
            if agent.rect.center[1] > window_pad - 2*agent_radius + height:
                agent.mode = 'collide'
                collided_pt = np.array(agent.rect.center) - window_pad
                collided_pt[1] += agent.radius
                agent.collided_points.append(collided_pt)

        noise = np.random.randn()*act_noise_std
        if extra.endswith('pos') and noise > 0: noise = 0
        elif extra.endswith('neg') and noise < 0: noise = 0
        action = agent.action + noise
        agent.move(action)

        traj[t,:2] = agent.pt_eye
        traj[t,2] = agent.orientation
        traj[t,3] = agent.action * np.pi / 2
    
    return traj


def build_agent_trajs_parallel(exp_name, gen_ext, space_step, orient_step, timesteps, rank='cen', eye=True, extra='', landmarks=False):
    print(f'building {exp_name}, {gen_ext}, {space_step}, {int(np.pi/orient_step)}, {timesteps}')

    # pull pv + envconf from save folders
    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    if rank == 'top':   NN_pv_path = fr'{data_dir}/{exp_name}/{gen_ext}_NN0_pickle.bin'
    elif rank == 'cen': NN_pv_path = fr'{data_dir}/{exp_name}/{gen_ext}_NNcen_pickle.bin'
    with open(NN_pv_path,'rb') as f:
        pv = pickle.load(f)
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)

    # reconstruct model
    NN, arch = reconstruct_NN(envconf, pv)

    # construct boundary endpts
    width, height = tuple(eval(envconf["ENV_SIZE"]))
    x_min, x_max = 0, width
    y_min, y_max = 0, height
    boundary_endpts = [
            np.array([ x_min, y_min ]),
            np.array([ x_max, y_min ]),
            np.array([ x_min, y_max ]),
            np.array([ x_max, y_max ])
            ]

    # every grid position/direction
    if not landmarks:
        coll_boundary_thickness = int(envconf["RADIUS_AGENT"])*2
        x_range = np.linspace(x_min + coll_boundary_thickness, 
                            x_max - coll_boundary_thickness, 
                            int((width - coll_boundary_thickness*2) / space_step))
        y_range = np.linspace(y_min + coll_boundary_thickness, 
                            y_max - coll_boundary_thickness, 
                            int((height - coll_boundary_thickness*2) / space_step))
    else:
        x_range = np.linspace(x_min, x_max, int(width / space_step))
        y_range = np.linspace(y_min, y_max, int(height / space_step))
    orient_range = np.arange(0, 2*np.pi, orient_step)
    print(f'testing ranges (max, min): x[{x_range[0], x_range[-1]}], y[{y_range[0], y_range[-1]}], o[{orient_range[0], orient_range[-1]}]')
    
    # construct matrix of each traj for each grid pos/dir
    num_inits = len(x_range) * len(y_range) * len(orient_range)
    traj_matrix = np.zeros( (num_inits, timesteps, 4) ) # (pos_x, pos_y, _, _) --> to match self.data_agent format
    print(f'traj matrix shape (# initializations, timesteps, ): {traj_matrix.shape}')
    
    # pack inputs for multiprocessing map
    mp_inputs = []
    for x in x_range:
        for y in y_range:
            for orient in orient_range:
                mp_inputs.append( (envconf, NN, boundary_endpts, x, y, orient, timesteps, extra) )
    
    # run agent NNs in parallel
    with mp.Pool() as pool:
        results = pool.starmap_async( agent_traj_from_xyo, mp_inputs)
        pool.close()
        pool.join()

    # unpack results into matrix (y coords transformed for plotting)
    results_list = results.get()
    for n, output in enumerate(results_list):
        traj_matrix[n,:,:] =  output
    traj_matrix[:,:,1] = y_max - traj_matrix[:,:,1]

    if extra == '':
        save_name = fr'{data_dir}/traj_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e{int(eye)}.bin'
    else:
        save_name = fr'{data_dir}/traj_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e{int(eye)}_{extra}.bin'
    with open(save_name, 'wb') as f:
        pickle.dump(traj_matrix, f)


def plot_agent_trajs(exp_name, gen_ext, space_step, orient_step, timesteps, rank='cen', ellipses=False, eye=True, ex_lines=False, extra='', landmarks=False):
    print(f'plotting map - {exp_name}, {gen_ext}, ell{ellipses}, ex_lines{ex_lines}, extra{extra}, lm{landmarks}')

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)

    if extra == '' or extra == '3d' or extra == 'clip' or extra == 'turn':
        save_name = fr'{data_dir}/traj_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e{int(eye)}'
    elif extra.startswith('3d'):
        save_name = fr'{data_dir}/traj_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e{int(eye)}_{extra[3:]}'
    else:
        save_name = fr'{data_dir}/traj_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e{int(eye)}_{extra}'
    with open(save_name+'.bin', 'rb') as f:
        ag_data = pickle.load(f)

    if extra == 'clip':
        ag_data = ag_data[:,25:250,:]
        save_name += '_clip'

    # build resource coord matrix
    res_data = np.zeros((1,1,3)) # 1 patch
    # res_data = np.zeros((2,1,3)) # 2 patches

    res_radius = int(envconf["RADIUS_RESOURCE"])
    width, height = tuple(eval(envconf["ENV_SIZE"]))
    x,y = tuple(eval(envconf["RESOURCE_POS"]))
    y_max = height

    res_data[0,0,:] = np.array((x, y_max - y, res_radius))
    traj_plot_data = (ag_data, res_data)
    
    # # before
    # with open(save_name+'.bin', 'rb') as f:
    #     traj_plot_data = pickle.load(f)

    if extra.startswith('3d'):
        from abm.monitoring.plot_funcs import plot_map_iterative_traj_3d
        # plot_map_iterative_traj_3d(traj_plot_data, x_max=width, y_max=height, save_name=save_name, plt_type='scatter', var='cturn')
        plot_map_iterative_traj_3d(traj_plot_data, x_max=width, y_max=height, save_name=save_name, plt_type='scatter', var='str_manif')
        # plot_map_iterative_traj_3d(traj_plot_data, x_max=width, y_max=height, save_name=save_name, plt_type='lines', var='cturn')
        # plot_map_iterative_traj_3d(traj_plot_data, x_max=width, y_max=height, save_name=save_name, plt_type='lines', var='ctime')
        # plot_map_iterative_traj_3d(traj_plot_data, x_max=width, y_max=height, save_name=save_name, plt_type='lines', var='ctime_flat')
        # plot_map_iterative_traj_3d(traj_plot_data, x_max=width, y_max=height, save_name=save_name, plt_type='lines', var='ctime_arrows')
        # plot_map_iterative_traj_3d(traj_plot_data, x_max=width, y_max=height, save_name=save_name, plt_type='lines', var='ctime_arrows_only')
        # plot_map_iterative_traj_3d(traj_plot_data, x_max=width, y_max=height, save_name=save_name, plt_type='lines', var='ctime_arrows_only', sv_typ='anim')
    else:
        from abm.monitoring.plot_funcs import plot_map_iterative_traj
        if not landmarks:
            plot_map_iterative_traj(traj_plot_data, x_max=width, y_max=height, save_name=save_name, ellipses=ellipses, ex_lines=ex_lines, extra=extra)
        else:
            lms = (int(envconf["RADIUS_LANDMARK"]),
                   [
                    np.array([ 0, 0 ]),
                    np.array([ width, 0 ]),
                    np.array([ 0, height ]),
                    np.array([ width, height ])
                    ]
            )
            plot_map_iterative_traj(traj_plot_data, x_max=width, y_max=height, save_name=save_name, ellipses=ellipses, ex_lines=ex_lines, extra=extra, landmarks=lms)


def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True, colored=False):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    x = (x+np.pi) % (2*np.pi) - np.pi
    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    if colored:
        my_cmap = plt.get_cmap('plasma')
        rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))

        patches = ax.bar(bins[:-1], radius, align='edge', width=widths, 
                        edgecolor=my_cmap(rescale(n)), fill=False, linewidth=1)
    else:
        patches = ax.bar(bins[:-1], radius, align='edge', width=widths, 
                        edgecolor='k', fill=False, linewidth=1)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    labels = ['$0$', r'-$\pi/4$',  r'-$\pi/2$', r'-$3\pi/4$', r'$\pi$', r'$3\pi4$', r'$\pi/2$', r'$\pi/4$', ]
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(labels)

    return n, bins, patches



def plot_agent_orient_corr(exp_name, gen_ext, space_step, orient_step, timesteps, rank='cen', eye=True, dpi=None):
    print(f'plotting corr - {exp_name} @ {dpi} dpi')

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    # env_path = fr'{data_dir}/{exp_name}/.env'
    # envconf = de.dotenv_values(env_path)

    save_name = fr'{data_dir}/traj_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e{int(eye)}'
    with open(save_name+'.bin', 'rb') as f:
        ag_data = pickle.load(f)
    
    num_runs,t_len,_ = ag_data.shape
    t = np.linspace(0,t_len,t_len)
    # print(f'ag_data shape: {num_runs, len(t)}')

    # # using precalc'd corr to init
    # corr = ag_data[:,:,3]
    # corr = np.insert(corr, 0, 1, axis=1)
    # # print(f'corr shape: {corr.shape}')

    # corr to init
    delay = 25
    t = t[:-delay]
    t_len -= delay
    orient = ag_data[:,delay:,2]
    orient_0 = orient[:,0]
    orient_0 = np.tile(orient_0,(t_len,1)).transpose()
    corr_init_angle_diff = orient - orient_0
    corr_init_angle_diff_scaled = (corr_init_angle_diff + np.pi) % (2*np.pi) - np.pi
    corr_init = np.cos(corr_init_angle_diff)
    # corr_init = np.insert(corr_init, 0, 1, axis=1)
    # print(f'corr shape: {corr.shape}')

    # patch position
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)
    pt_target = np.array(eval(envconf["RESOURCE_POS"]))
    pt_target[1] = 1000 - pt_target[1]

    # distance to patch
    x = ag_data[:,delay:,0]
    y = ag_data[:,delay:,1]
    pt_self = np.array([x,y])
    disp = pt_self.transpose() - pt_target
    dist = np.linalg.norm(disp, axis=2)

    # # corr to patch
    # angle_to_target = np.arctan2(disp[:,:,1], disp[:,:,0]) + np.pi # shift by pi for [0-2pi]
    # # print('max/min angle_to_target: ', np.max(angle_to_target), np.min(angle_to_target))
    # corr_patch_angle_diff = angle_to_target - orient.transpose()
    # corr_patch = np.cos(corr_patch_angle_diff)
    # # print(f'corr shape: {angle_to_target.shape}')

    ### temporal correlations ###

    fig = plt.figure(figsize=(3,3))
    ax0 = plt.subplot()
    # fig = plt.figure(figsize=(15,5))
    # ax0 = plt.subplot(131)
    # ax1 = plt.subplot(132)
    # ax2 = plt.subplot(133, projection='polar')

    corr_init_avg = np.mean(corr_init, 0)
    ax0.plot(t, corr_init_avg, 'k')
    ax0.axhline(color='gray', ls='--')

    peaks,_ = scipy.signal.find_peaks(corr_init_avg[:300-delay], prominence=.05)
    # if len(peaks) >= 1:
    #     ax0.plot(t[peaks],corr_init_avg[peaks],'o',color='dodgerblue')
    peaks_neg,_ = scipy.signal.find_peaks(-corr_init_avg[:300-delay], prominence=.05)
    # if len(peaks_neg) >= 1:
    #     ax0.plot(t[peaks_neg],corr_init_avg[peaks_neg],'o',color='dodgerblue')
    corr_peaks = len(peaks) + len(peaks_neg)

    ax0.set_xlim(-20,520)
    ax0.set_ylim(-1.05,1.05)
    ax0.set_ylim(-0.05,1.05)
    ax0.set_xlabel('Timesteps')
    ax0.set_ylabel('Orientation Correlation')

    plt.tight_layout()
    plt.savefig(fr'{save_name}_corr_orient_{dpi}.png', dpi=dpi)
    plt.close()


    ### spatial correlation trajectories ###

    fig = plt.figure(figsize=(3,3))
    ax1 = plt.subplot()

    # corr_init_times = []
    for r in range(num_runs)[::50]:
        # ax[0,0].plot(t, corr_init[r,:], c='k', alpha=5/255)
        # ax[0,3].plot(dist[:,r], corr_init[r,:], c='k', alpha=5/255)
        ax1.plot(dist[:,r], -corr_init_angle_diff_scaled[r,:], c='k', alpha=5/255) # negative to match orientation of polar plot

        # x = corr_init_angle_diff[r,:]
        # # corr = np.array([1. if ts==0 else np.corrcoef(x[ts:],x[:-ts])[0][1] for ts,ts_float in enumerate(t)])
        # corr = np.correlate((x-x.mean()),(x-x.mean()),'full')[len(x)-1:]/np.var(x)/len(x)
        # ax[1].plot(t, corr, c='k', alpha=5/255)
        # peaks,_ = scipy.signal.find_peaks(corr, prominence=.1)
        # if len(peaks) >= 1:
        #     corr_init_times.append(peaks[0])
        #     ax[1].plot(t[peaks[0]],corr[peaks[0]],'o',color='dodgerblue',alpha=.1)
    # corr_median = np.median(np.array(corr_init_times))
    # ax[1].axvline(t[int(corr_median)], color='gray', ls='--')
    # p1 = f'median peak of indiv init autocorrelations: {corr_median}'
    # print(p1)

    # plot trajectories for X initializations

    # ax1.plot(np.arange(11),np.arange(11))
    ins = ax1.inset_axes([0.7,0.05,0.25,0.25])
    ins.set_yticks([])
    ins.set_xticks([])

    inits = [
        [700, 200, np.pi], #BR-W
        [800, 900, 3*np.pi/2], #TR-S
        [100, 200, np.pi/2], #BL-N
        [100, 900, 3*np.pi/2], #TL-S
    ]
    colors = [
        'cornflowerblue',
        'tomato',
        'forestgreen',
        'gold',
    ]
    for pt,color in zip(inits,colors):
        # search across xy plane
        distance, index_xy = scipy.spatial.KDTree(ag_data[:,0,:2]).query(pt[:2])
        # search locally for best ori
        index_ori = (np.abs(ag_data[index_xy:index_xy+16,0,2] - pt[2])).argmin()
        # combine
        index = index_xy + index_ori

        # ax[0,0].plot(t, corr_init[index,:], c=color, alpha=.5)
        # ax[0,2].plot(dist[:,index], corr_init[index,:], c=color, alpha=.5)
        # ax[0,3].plot(dist[:,index], corr_init_angle_diff[index,:], c=color, alpha=.5)
        # ax[1,0].plot(t, corr_patch[:,index], c=color, alpha=.5)
        # ax[1,2].plot(dist[:,index], corr_patch[:,index], c=color, alpha=.5)
        # ax[1,3].plot(dist[:,index], corr_patch_angle_diff[:,index], c=color, alpha=.5)
        ins.plot(dist[:,index], corr_init_angle_diff_scaled[index,:], c=color, alpha=.5, linewidth=1)

    # labels = [r'-$2\pi$', r'-$3\pi/2$', r'-$\pi$', r'-$\pi/2$', '$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']
    labels = [r'-$\pi$', r'-$\pi/2$', '$0$', r'$\pi/2$', r'$\pi$']

    ax1.set_xlim(-20,820)
    ax1.set_yticks(np.arange(-np.pi, np.pi+0.01, np.pi/2))
    ax1.set_yticklabels(labels)
    ax1.set_xlabel('Distance to Patch')
    ax1.set_ylabel('Relative Orientation')

    plt.tight_layout()
    plt.savefig(fr'{save_name}_corr_trajs_{dpi}.png', dpi=dpi)
    plt.close()


    ### spatial correlation - polar hist ###

    fig = plt.figure(figsize=(3,3))
    ax2 = plt.subplot(projection='polar')

    dist = dist.flatten()
    m = np.ma.masked_less(dist, 100)

    corr_init_angle_diff = corr_init_angle_diff_scaled.transpose().flatten()
    corr_init_angle_diff_masked = (1-m.mask)*corr_init_angle_diff
    corr_init_angle_diff_comp = corr_init_angle_diff_masked[corr_init_angle_diff_masked != 0]

    # n1,bins,patches = ax[4].hist(corr_init_angle_diff_comp, bins=100, density=True)
    # ax[4].axvline(np.mean(corr_init_angle_diff_comp), color='gray', ls='--')

    # ax[4].set_yscale('log')
    # ax[4].set_ylim(10**-3,10**2)
    # ax[4].set_ylabel('Frequency')
    # ax[4].set_xlabel('Orientation Persistence')
    # ax[4].set_xticks(np.arange(-np.pi, np.pi+0.01, np.pi/2))
    # ax[4].set_xticklabels(labels)

    # Visualise by area of bins
    # circular_hist(ax, corr_init_angle_diff_comp, bins=100, offset=np.pi/2)
    # circular_hist(ax[0], corr_init_angle_diff_comp, bins=100, offset=np.pi/2)
    n, bins, patches = circular_hist(ax2, corr_init_angle_diff_comp, bins=100, offset=np.pi/2, colored=True)
    # # Visualise by radius of bins
    # circular_hist(ax[1], corr_init_angle_diff_comp, bins=100, offset=np.pi/2, density=False)

    x = corr_init_angle_diff_comp
    x = (x + np.pi) % (2*np.pi) - np.pi
    histo_avg = np.mean(corr_init_angle_diff_comp)/np.pi*2
    ax2.axvline(histo_avg, color='gray', ls='--')

    peaks,_ = scipy.signal.find_peaks(n, prominence=25000)
    histo_peaks = len(peaks)

    print(f'num peaks: {corr_peaks} // histo avg: {-round(histo_avg,2)} // histo peaks: {histo_peaks}')

    plt.tight_layout()
    plt.savefig(fr'{save_name}_corr_polar_{dpi}.png', dpi=dpi)
    plt.close()

    return corr_peaks, -histo_avg, histo_peaks


def plot_agent_valnoise_dists(run_name, noise_types, val='cen', dpi=None):
    print(f'plotting valnoise - {run_name}')
    num_noise_types = len(noise_types)

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'

    if val == 'top': filename = 'val_matrix'
    elif val == 'cen': filename = 'val_matrix_cen'

    # init plot details
    fig, ax1 = plt.subplots(figsize=(3,3)) 
    cmap = plt.get_cmap('plasma')
    cmap_range = num_noise_types
    xtick_locs = []
    labels = []
    medians = []

    for n_num, (label, noise_type) in enumerate(noise_types):

        if noise_type == 'no_noise':
            with open(fr'{data_dir}/{run_name}/{filename}.bin','rb') as f:
                data = pickle.load(f)
        else:
            with open(fr'{data_dir}/{run_name}/{filename}_{noise_type}_noise.bin','rb') as f:
                data = pickle.load(f)

        xtick_loc = n_num/num_noise_types
        xtick_locs.append(xtick_loc)
        labels.append(label)

        l0 = ax1.violinplot(data.flatten(), 
                    positions=[xtick_loc],
                    widths=1/num_noise_types, 
                    showmedians=True,
                    showextrema=False,
                    )
        for p in l0['bodies']:
            p.set_facecolor(cmap(n_num/cmap_range))
            p.set_edgecolor(cmap(n_num/cmap_range))
        l0['cmedians'].set_edgecolor(cmap(n_num/cmap_range))

        # color = l0["bodies"][0].get_facecolor().flatten()
        # violin_labs.append((mpatches.Patch(color=color), group_name))

        dist_median = np.median(data)
        medians.append((noise_type, dist_median))
        print(f'noise type: {noise_type} // median: {dist_median}')

    # plt.grid(axis = 'x')
    # plt.xticks(np.arange(0, n_num+1, 1))
    plt.xticks(xtick_locs)
    # ax1.xaxis.set_ticklabels([])
    # ax1.set_xticks([])
    ax1.set_xticklabels(labels)
    ax1.set_xlabel('Visual Angle Noise')
    ax1.set_ylabel('Time to Find Patch')
    ax1.set_ylim(-20,1020)

    plt.tight_layout()
    plt.savefig(fr'{data_dir}/{run_name}_valnoise_{dpi}.png', dpi=dpi)
    plt.close()

    return medians


def plot_agent_turn_histo(exp_name, gen_ext, space_step, orient_step, timesteps, rank='cen', eye=True):
    print(f'plotting {exp_name}, {gen_ext}, {space_step}, {int(np.pi/orient_step)}, {timesteps}')

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'

    save_name = fr'{data_dir}/traj_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e{int(eye)}'
    with open(save_name+'.bin', 'rb') as f:
        ag_data = pickle.load(f)

    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)
    pt_target = np.array(eval(envconf["RESOURCE_POS"]))
    pt_target[1] = 1000 - pt_target[1]
    # print('patch pos: ', pt_target)

    num_runs,t_len,_ = ag_data.shape
    # print(f'ag_data shape: {num_runs, len(t)}')

    delay = 25
    x = ag_data[:,delay:,0]
    y = ag_data[:,delay:,1]
    orient = ag_data[:,delay:,2]
    turn = ag_data[:,delay:,3]
    # print('avg final pos: ', np.mean(x[:,-1]), np.mean(y[:,-1]))
    # print('max/min orient: ', np.max(orient), np.min(orient))

    pt_self = np.array([x,y])
    disp = pt_self.transpose() - pt_target
    dist = np.linalg.norm(disp, axis=2)

    # use dist to filter for only those >100 units away
    m = np.ma.masked_less(dist.flatten(), 100)
    turn_masked = (1-m.mask)*turn.flatten()

    # histogram by turn
    # fig,ax = plt.subplots(figsize=(5,5))
    fig,ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].hist(turn.flatten(), bins=9)
    ax[1].hist(turn_masked, bins=9)

    ax[0].set_yscale('log')
    ax[1].set_yscale('log')

    ax[1].set_ylabel('Frequency')
    ax[1].set_xlabel('Turning Angle')

    labels = [r'-$\pi/2$', r'-$3\pi/8$', r'-$\pi/4$', r'-$\pi/8$', '$0$', r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$']
    ax[1].set_xticks(np.arange(-np.pi/2, np.pi/2+0.01, np.pi/8))
    ax[1].set_xticklabels(labels)

    ax[0].set_ylabel('Frequency')
    ax[0].set_xlabel('Turning Angle')
    ax[0].set_xticks(np.arange(-2*np.pi, 2*np.pi+0.01, np.pi/2))
    ax[0].set_xticklabels(labels)
    plt.subplots_adjust(wspace=.25)

    plt.savefig(fr'{save_name}_turn_histo.png')
    plt.close()


# -------------------------- stationary vis -------------------------- #


def agent_traj_from_xyo_PRW(envconf, NN, boundary_endpts, x, y, orient, timesteps, rot_diff):

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
            window_pad=int(envconf["WINDOW_PAD"]),
            radius=int(envconf["RADIUS_AGENT"]),
            color=(0,0,0),
            vis_transform='',
            percep_angle_noise_std=0,
        )

    traj = np.zeros((timesteps,4))
    for t in range(timesteps):

        agent.gather_self_percep_info()

        action = (2*rot_diff)**.5 * np.random.uniform(-1,1)
        agent.move(action)

        traj[t,:2] = agent.pt_eye
        traj[t,2] = agent.orientation
        traj[t,3] = np.cos(agent.orientation - orient)
    
    return traj


def build_agent_trajs_parallel_PRW(exp_name, space_step, orient_step, timesteps, rot_diff):
    print(f'building PRW w {rot_diff} rot_diff, @ {space_step}, {int(np.pi/orient_step)}, {timesteps}, {rot_diff}')

    # pull pv + envconf from save folders
    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)

    # construct dummy model
    from abm.NN.model import WorldModel as Model
    NN = Model()

    # construct boundary endpts
    width, height = tuple(eval(envconf["ENV_SIZE"]))
    x_min, x_max = 0, width
    y_min, y_max = 0, height
    boundary_endpts = [
            np.array([ x_min, y_min ]),
            np.array([ x_max, y_min ]),
            np.array([ x_min, y_max ]),
            np.array([ x_max, y_max ])
            ]

    # every grid position/direction
    x_range = np.linspace(x_min, x_max, int(width / space_step))
    y_range = np.linspace(y_min, y_max, int(height / space_step))
    orient_range = np.arange(0, 2*np.pi, orient_step)
    # print(f'testing ranges (max, min): x[{x_range[0], x_range[-1]}], y[{y_range[0], y_range[-1]}], o[{orient_range[0], orient_range[-1]}]')
    
    # construct matrix of each traj for each grid pos/dir
    num_inits = len(x_range) * len(y_range) * len(orient_range)
    traj_matrix = np.zeros( (num_inits, timesteps, 4) ) # (pos_x, pos_y, _, _) --> to match self.data_agent format
    print(f'traj matrix shape (# initializations, timesteps, ): {traj_matrix.shape}')
    
    # pack inputs for multiprocessing map
    mp_inputs = []
    for x in x_range:
        for y in y_range:
            for orient in orient_range:
                mp_inputs.append( (envconf, NN, boundary_endpts, x, y, orient, timesteps, rot_diff) )
    
    # run agent NNs in parallel
    with mp.Pool() as pool:
        results = pool.starmap_async(agent_traj_from_xyo_PRW, mp_inputs)
        pool.close()
        pool.join()

    # unpack results into matrix (y coords transformed for plotting)
    results_list = results.get()
    for n, output in enumerate(results_list):
        traj_matrix[n,:,:] =  output
    traj_matrix[:,:,1] = y_max - traj_matrix[:,:,1]

    save_name = fr'{data_dir}/traj_matrices/PRW_rd{str(rot_diff).replace(".","p")}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}'
    with open(save_name+'.bin', 'wb') as f:
        pickle.dump(traj_matrix, f)


def plot_agent_trajs_PRW(exp_name, space_step, orient_step, timesteps, rot_diff):
    print(f'plotting PRW w {rot_diff} rot_diff, @ {space_step}, {int(np.pi/orient_step)}, {timesteps}')

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)

    save_name = fr'{data_dir}/traj_matrices/PRW_rd{str(rot_diff).replace(".","p")}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}'
    with open(save_name+'.bin', 'rb') as f:
        ag_data = pickle.load(f)

    # build resource coord matrix
    res_data = np.zeros((1,1,3)) # 1 patch
    # res_data = np.zeros((2,1,3)) # 2 patches

    res_radius = int(envconf["RADIUS_RESOURCE"])
    width, height = tuple(eval(envconf["ENV_SIZE"]))
    x,y = tuple(eval(envconf["RESOURCE_POS"]))
    y_max = height

    res_data[0,0,:] = np.array((x, y_max - y, res_radius))
    traj_plot_data = (ag_data, res_data)

    from abm.monitoring.plot_funcs import plot_map_iterative_traj
    plot_map_iterative_traj(traj_plot_data, x_max=width, y_max=height, save_name=save_name)


def plot_agent_corr_PRW(space_step, orient_step, timesteps, rot_diff):
    print(f'plotting PRW w {rot_diff} rot_diff, @ {space_step}, {int(np.pi/orient_step)}, {timesteps}')

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'

    save_name = fr'{data_dir}/traj_matrices/PRW_rd{str(rot_diff).replace(".","p")}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}'
    with open(save_name+'.bin', 'rb') as f:
        ag_data = pickle.load(f)
    
    num_runs,t_len,_ = ag_data.shape
    t = np.linspace(0,t_len,t_len+1)
    # print(f'ag_data shape: {num_runs, len(t)}')

    # # using precalc'd corr
    # corr = ag_data[:,:,3]
    # corr = np.insert(corr, 0, 1, axis=1)
    # # print(f'corr shape: {corr.shape}')

    # # auto corr calc
    # orient = ag_data[:,:,2]
    # orient_0 = orient[:,0]
    # orient_0 = np.tile(orient_0,(t_len,1)).transpose()
    # corr = np.cos(orient - orient_0)
    # corr = np.insert(corr, 0, 1, axis=1)
    # # print(f'corr shape: {corr.shape}')

    # auto corr calc - delayed orient_0
    t = t[25:]
    t_len -= 25
    orient = ag_data[:,25:,2]
    orient_0 = orient[:,0]
    orient_0 = np.tile(orient_0,(t_len,1)).transpose()
    corr = np.cos(orient - orient_0)
    corr = np.insert(corr, 0, 1, axis=1)
    # print(f'corr shape: {corr.shape}')

    # construct plot
    fig,ax = plt.subplots(1,2, figsize=(10,5))

    for r in range(num_runs)[::50]:
        ax[0].plot(t, corr[r,:], c='k', alpha=1/255)
    ax[0].set_xlim(-10,510)
    ax[0].set_ylim(-1.05,1.05)

    corr_avg = np.mean(corr[:,:], 0)
    ax[1].plot(t, corr_avg)
    ax[1].set_xlim(-15,515)
    ax[1].set_ylim(-1.05,1.05)

    plt.savefig(fr'{save_name}_corr_auto_delayed.png')
    plt.show()



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
    orient_range = np.arange(0, 2*np.pi, orient_step)
    # for n, o in enumerate(orient_range):
    #     print(f'number {n}, orient {round(o,2)}, angle {int(o*180/np.pi)}')
    
    fig = plt.figure()
    frames = []

    for n in range(num_visfield_dims):

        min = mat[:,:,:,n].flatten().min()
        max = mat[:,:,:,n].flatten().max()

        # for n in range(num_orient):
            # frames.append([plt.imshow(np.transpose(mat[:,:,n,0]), vmin = min, vmax = max, animated=True)])
        for n,o in enumerate(orient_range[192:321]): # 135-225 deg
            frames.append([plt.imshow(np.transpose(mat[:,:,n+192,0]), vmin = min, vmax = max, animated=True)])

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
    
    # ani.save(fr'{data_dir}/visfield_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}.mp4')
    ani.save(fr'{data_dir}/visfield_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_135t225deg.mp4')
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
            vis_transform='',
            # vis_transform=str(envconf["VIS_TRANSFORM"]),
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
            if i == 'wall_north': vis_field[n] = 1
            elif i == 'wall_east': vis_field[n] = 2
            elif i == 'wall_south': vis_field[n] = 3
            else: # i == 'wall_west': 
                vis_field[n] = 4
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
    # NN_pv_path = fr'{data_dir}/{exp_name}/{gen_ext}_NN0_pickle.bin'
    NN_pv_path = fr'{data_dir}/{exp_name}/{gen_ext}_NNcen_pickle.bin'
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
    plot_map_iterative_trajall(traj_plot_data, x_max=width, y_max=height, save_name=save_name, var_pos=0, change=True)
    # save_name = fr'{data_dir}/trajall_matrices/{exp_name}_{gen_ext}_traj_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_visflat'
    # plot_map_iterative_trajall(traj_plot_data, x_max=width, y_max=height, save_name=save_name, var_pos=0, change=False)
    # save_name = fr'{data_dir}/trajall_matrices/{exp_name}_{gen_ext}_traj_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_viswallN'
    # plot_map_iterative_trajall(traj_plot_data, x_max=width, y_max=height, save_name=save_name, var_pos=0, change=True, wall=1)
    # save_name = fr'{data_dir}/trajall_matrices/{exp_name}_{gen_ext}_traj_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_viswallE'
    # plot_map_iterative_trajall(traj_plot_data, x_max=width, y_max=height, save_name=save_name, var_pos=0, change=True, wall=2)
    # save_name = fr'{data_dir}/trajall_matrices/{exp_name}_{gen_ext}_traj_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_viswallS'
    # plot_map_iterative_trajall(traj_plot_data, x_max=width, y_max=height, save_name=save_name, var_pos=0, change=True, wall=3)
    # save_name = fr'{data_dir}/trajall_matrices/{exp_name}_{gen_ext}_traj_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_viswallW'
    # plot_map_iterative_trajall(traj_plot_data, x_max=width, y_max=height, save_name=save_name, var_pos=0, change=True, wall=4)

    # loop over encoded vis (CNN output)
    vis_field_res = int(envconf["VISUAL_FIELD_RESOLUTION"])
    encoded_vis_size = list(map(int,envconf["CNN_DIMS"].split(',')))[-1]
    for i in range(encoded_vis_size):
        var_pos = 3 + vis_field_res + i
        save_name = fr'{data_dir}/trajall_matrices/{exp_name}_{gen_ext}_traj_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_NactCNN{i}'
        plot_map_iterative_trajall(traj_plot_data, x_max=width, y_max=height, save_name=save_name, var_pos=var_pos)

    save_name = fr'{data_dir}/trajall_matrices/{exp_name}_{gen_ext}_traj_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_action'
    plot_map_iterative_trajall(traj_plot_data, x_max=width, y_max=height, save_name=save_name, var_pos=-1)
    # save_name = fr'{data_dir}/trajall_matrices/{exp_name}_{gen_ext}_traj_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_action_inv'
    # plot_map_iterative_trajall(traj_plot_data, x_max=width, y_max=height, save_name=save_name, var_pos=-1, inv=True)



def plot_agent_trajalls_pca(exp_name, gen_ext, space_step, orient_step, timesteps, analyses, datasets, comps):

    from abm.monitoring.plot_funcs import plot_map_iterative_trajall

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)

    width=int(envconf["ENV_WIDTH"])
    height=int(envconf["ENV_HEIGHT"])

    # pca or ica
    for analysis in analyses:

        # input, CNN, or output
        for name in datasets:
            with open(fr'{data_dir}/trajall_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{analysis}_{name}_projdata.bin', 'rb') as f:
                traj_plot_data = pickle.load(f)

            # plot first X components
            for comp in comps:
                var_pos = 3 + comp
                save_name = fr'{data_dir}/trajall_matrices/{exp_name}_{gen_ext}_traj_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{analysis}_{name}_c{comp}'
                plot_map_iterative_trajall(traj_plot_data, x_max=width, y_max=height, save_name=save_name, var_pos=var_pos, change=True)


# -------------------------- script -------------------------- #

def run_gamut(names):

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    with open(fr'{data_dir}/traj_matrices/gamut.bin', 'rb') as f:
        data = pickle.load(f)

    rank = 'cen'
    space_step = 25
    orient_step = np.pi/8
    timesteps = 500
    eye = True
    extra = ''
    dpi = 50
    noise_types = [
        (0, 'no_noise'), 
        (0.05, 'angle_n05'), 
        (0.10, 'angle_n10'),
        ]

    for name in names:

        gen, valfit = find_top_val_gen(name, rank)
        print(f'{name} @ {gen} w {valfit} fitness')

        # filter out poor performers
        if valfit >= 500:
            print('skip')
            print('')
            continue
        if name in data:
            print('already there')
            print('')
            continue

        build_agent_trajs_parallel(name, gen, space_step, orient_step, timesteps)
        plot_agent_trajs(name, gen, space_step, orient_step, timesteps, ex_lines=True)
        corr_peaks, histo_avg, histo_peaks = plot_agent_orient_corr(name, gen, space_step, orient_step, timesteps, dpi=dpi)
        # angle_medians = plot_agent_valnoise_dists(name, noise_types, dpi=dpi)

        # save in dict + update pickle
        data[name] = (corr_peaks, histo_avg, histo_peaks)
        # data[name] = (corr_peaks, histo_avg, histo_peaks, angle_medians)
        print(f'data dict len: {len(data)}')
        print('')

        with open(fr'{data_dir}/traj_matrices/gamut.bin', 'wb') as f:
            pickle.dump(data, f)

        # delete .bin file using same code as for saving in build_agent_trajs_parallel()
        save_name = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e{int(eye)}.bin'
        os.remove(save_name)


# -------------------------- misc -------------------------- #

def find_top_val_gen(exp_name, rank='cen'):

    # parse val results text file
    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'

    if rank == 'top': 
        with open(fr'{data_dir}/{exp_name}/val_results.txt') as f:
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

    elif rank == 'cen': 
        with open(fr'{data_dir}/{exp_name}/val_results_cen.txt') as f:
            lines = f.readlines()
        # with open(fr'{data_dir}/{exp_name}/val_matrix_cen.bin') as f:
        #     val_data = pickle.load(f)

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

    ## anim / action vol
    # space_step = 5
    # orient_step = np.pi/64

    ## traj / Nact
    space_step = 25
    orient_step = np.pi/8

    ## action
    # space_step = 100
    # orient_step = np.pi/8

    ## traj - turn
    # space_step = 100
    # orient_step = np.pi/4

    ## quick test
    # space_step = 500
    # orient_step = np.pi/2

    timesteps = 500
    noise_types = [
        (0, 'no_noise'), 
        (0.05, 'angle_n05'), 
        (0.10, 'angle_n10'),
        ]
    # noise_types = [
    #     ('no noise', 'no_noise'), 
    #     ('angle: .05', 'angle_n05'),
    #     ('angle: .10', 'angle_n10'),
    #     ('dist: .025', 'dist_n025'),
    #     ('dist: .050', 'dist_n05'),
    #     ]

    names = []

    # names.append('sc_CNN12_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep18')
    # for i in [1,2,4,5,6,7,11,12,13,14,15]:
    # for i in [3,4,9]:
        # names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{str(i)}')
    # for i in [0,1,2,5,6,9,13,14,16,17,18,19]:
    # for i in [3]:
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep{str(i)}')

    # for i in [0,1,2,3,5,7,8,9,10,13,14,16,18]:
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep{str(i)}')

    # for i in [2]: 
    # for i in [3,5]: 
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_n0_rep{str(i)}')
    # for i in [0,2,4,8,12,17]:
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep{str(i)}')
    # for i in [0,3,8,12,13]:
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep{str(i)}')

    # for i in [8,9,11,16]:
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_n1_rep{str(i)}')
    # for i in [0,2,5]:
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_n2_rep{str(i)}')
    # for i in [1,3,7,15]:
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_n4_rep{str(i)}')

    # for i in [0,6,12,13,14,15]:
        # names.append(f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_rep{str(i)}')
    # for i in [1,2,8,10,11,12]:
    #     names.append(f'sc_lm_CNN14_FNN2_p50e20_vis32_lm100_rep{str(i)}')

    # data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    # with open(fr'{data_dir}/traj_matrices/gamut.bin', 'rb') as f:
    #     data = pickle.load(f)

    for name in names:
        gen, valfit = find_top_val_gen(name, 'cen')
        print(f'{name} @ {gen} w {valfit} fitness')
        
        # # filter out poor performers
        # if valfit >= 500:
        #     print('skip')
        #     print('')
        #     continue
        # if name in data:
        #     print('already there')
        #     print('')
        #     continue

        # build_agent_trajs_parallel(name, gen, space_step, orient_step, timesteps)
        # plot_agent_trajs(name, gen, space_step, orient_step, timesteps, ex_lines=True)
        # plot_agent_trajs(name, gen, space_step, orient_step, timesteps, extra='3d')
        # corr_peaks, histo_avg, histo_peaks = plot_agent_orient_corr(name, gen, space_step, orient_step, timesteps, dpi=100)
        # angle_medians = plot_agent_valnoise_dists(name, noise_types, dpi=100)

        # # save in dict + update pickle
        # data[name] = (corr_peaks, histo_avg, histo_peaks, angle_medians)
        # print(f'data dict len: {len(data)}')
        # with open(fr'{data_dir}/traj_matrices/gamut.bin', 'wb') as f:
        #     pickle.dump(data, f)


    # name = 'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep4'
    # gen = 'gen941'
    # plot_agent_trajs(name, gen, space_step, orient_step, timesteps)
    # for e in ['FOV39','FOV41','TLx100','TLy100']:
    #     plot_agent_trajs(name, gen, space_step, orient_step, timesteps, extra=e)
    # for e in ['','move75','move125']:
    #     plot_agent_trajs(name, gen, space_step, orient_step, timesteps, ellipses=True, extra=e)


    # # build PRW data

    # dummy_exp = 'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep0'
    # for rot_diff in [0.01]:
    #     plot_agent_corr_PRW(space_step, orient_step, timesteps, rot_diff)

    # for rot_diff in [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]:
    #     # build_agent_trajs_parallel_PRW(dummy_exp, space_step, orient_step, timesteps, rot_diff)
    #     # plot_agent_trajs_PRW(dummy_exp, space_step, orient_step, timesteps, rot_diff)
    #     plot_agent_corr_PRW(space_step, orient_step, timesteps, rot_diff)

    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)

    # for name in [f'sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
        names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
        names.append(name)

    seeds = [10000,20000,30000,40000]

    for s in seeds:
        for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed{str(int(s/1000))}k_rep{x}' for x in range(20)]:
            names.append(name)
    for s in seeds:
        for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed{str(int(s/1000))}k_rep{x}' for x in range(20)]:
            names.append(name)
    for s in seeds:
        for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed{str(int(s/1000))}k_rep{x}' for x in range(20)]:
            names.append(name)
    for s in seeds:
        for name in [f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed{str(int(s/1000))}k_rep{x}' for x in range(20)]:
            names.append(name)
    for s in seeds:
        for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_seed{str(int(s/1000))}k_rep{x}' for x in range(20)]:
            names.append(name)
    for s in seeds:
        for name in [f'sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed{str(int(s/1000))}k_rep{x}' for x in range(20)]:
            names.append(name)

    run_gamut(names)
