from abm.start_sim import reconstruct_NN
from abm.sprites.agent import Agent
# from abm.sprites.agent_LM import Agent
from abm.sprites.landmark import Landmark

import os
import dotenv as de
from pathlib import Path
import numpy as np
import torch
import multiprocessing as mp
import _pickle as pickle
import matplotlib.pyplot as plt
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

    plt.savefig(fr'{data_dir}/Nact_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}.png', dpi=400)
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

    plt.savefig(fr'{data_dir}/Nact_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_action_phase.png', dpi=1000)
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
    if float(envconf["RADIUS_LANDMARK"]) > 0:
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
    print(f'building {exp_name}, {gen_ext}, {space_step}, {int(np.pi/orient_step)}, {timesteps}, {rank}, {int(eye)}, {extra}')

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


def plot_agent_trajs(exp_name, gen_ext, space_step, orient_step, timesteps, rank='cen', ellipses=False, eye=True, extra='', landmarks=False):
    print(f'plotting {exp_name}, {gen_ext}, {space_step}, {int(np.pi/orient_step)}, {timesteps}, {rank}, {int(eye)}, {extra}, {landmarks}')

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)

    if extra == '' or extra == '3d' or extra == 'clip':
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
        plot_map_iterative_traj_3d(traj_plot_data, x_max=width, y_max=height, save_name=save_name, plt_type='lines', var='ctime_arrows')
        plot_map_iterative_traj_3d(traj_plot_data, x_max=width, y_max=height, save_name=save_name, plt_type='lines', var='ctime_arrows_only')
        # plot_map_iterative_traj_3d(traj_plot_data, x_max=width, y_max=height, save_name=save_name, plt_type='lines', var='ctime_arrows_only', sv_typ='anim')
    else:
        from abm.monitoring.plot_funcs import plot_map_iterative_traj
        if not landmarks:
            plot_map_iterative_traj(traj_plot_data, x_max=width, y_max=height, save_name=save_name, ellipses=ellipses, extra=extra)
        else:
            lms = (int(envconf["RADIUS_LANDMARK"]),
                   [
                    np.array([ 0, 0 ]),
                    np.array([ width, 0 ]),
                    np.array([ 0, height ]),
                    np.array([ width, height ])
                    ]
            )
            plot_map_iterative_traj(traj_plot_data, x_max=width, y_max=height, save_name=save_name, ellipses=ellipses, extra=extra, landmarks=lms)


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



# -------------------------- misc -------------------------- #

def find_top_val_gen(exp_name, rank='top'):

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

    names = []


    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_angl_n05_rep{x}' for x in [3,5,8,11]]:
    #     names.append(name)
    # names.append('sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_angl_n10_rep9')
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_act_n10_rep{x}' for x in [0,5,13,15]]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_angl_n15_rep{x}' for x in [9,14,15,16]]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_angl_n20_rep{x}' for x in [7,12,14,18]]:
    #     names.append(name)

    # for name in [f'sc_lm_CNN14_FNN2_p50e20_vis8_lm100_rep{x}' for x in [6,9,13,14]]: 
    #     names.append(name)
    # for name in [f'sc_lm_CNN14_FNN2_p50e20_vis10_lm100_rep{x}' for x in [3,7,9,15]]:
    #     names.append(name)
    # for name in [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_rep{x}' for x in [0,6,13,14]]: 
    #     names.append(name)
    # for name in [f'sc_lm_CNN14_FNN2_p50e20_vis16_lm100_rep{x}' for x in [2,5,8,19]]: 
    #     names.append(name)

    # for name in [f'sc_lm_CNN14_FNN2_p50e20_vis8_lm100_angl_n10_rep{x}' for x in [15,2]]:
    #     names.append(name)
    # for name in [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_angl_n10_rep{x}' for x in [3,7,10,14]]:
    #     names.append(name)
    # for name in [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_angl_n05_rep{x}' for x in [0,2,5,13]]:
    #     names.append(name)

    # for name in [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmdist_n100_rep{x}' for x in [5,9,13,16]]: 
    #     names.append(name)
    # for name in [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmdist_n050_rep{x}' for x in [0,10,13,16]]:
    #     names.append(name)
    # for name in [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmdistpost_n50_rep{x}' for x in [6,9,13,19]]:
    #     names.append(name)
    # for name in [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmdistpost_n100_rep{x}' for x in [1,2,9,17]]: 
    #     names.append(name)
    
    # for name in [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmangle_n05_rep{x}' for x in [7,12,14,17]]:
    #     names.append(name)
    # for name in [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmangle_n10_rep{x}' for x in [1,5,7,16]]: 
        # names.append(name)

    # for name in [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmradius_n50_rep{x}' for x in [0,3,15,17]]:
    #     names.append(name)
    # for name in [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmradius_n100_rep{x}' for x in [8,12,14,15]]:
    #     names.append(name)

    # for name in [f'sc_lm_CNN14_FNN2_p50e20_vis8_lm300_rep{x}' for x in [6,7,12,15]]: 
    #     names.append(name)

    # trajalls

    # names.append('sc_CNN12_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep18')
    # names.append('sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep1')
    # names.append('sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep9')

    # names.append('sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep4')
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in [1,3,10,11,14]]: # OG best
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in [0,2,4,5,6,7,9,12,13,15]]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep{x}' for x in [5,1,0,16,2,17,14,6,19,4,18,3,9,13]]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed20k_rep{x}' for x in [7,1,16,0,10,15,13,6,19,12,2,9,11,5,14,3,17,18]]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed30k_rep{x}' for x in [15,7,17,16,6,14,1,10,4,0,3,9,2,5]]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed40k_rep{x}' for x in [15,1,5,17,16,19,12,10,14,3]]:
    #     names.append(name)

    # for i in [1,3,4,10]:
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{str(i)}')
    
    # for i in [3,5]:
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_rep{str(i)}')
    # for i in [3,7]:
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep{str(i)}')
    # for i in [7]:
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_n4_rep{str(i)}')


    # for i in [8]:
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{str(i)}')
    for i in [0,8,12]:
        names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep{str(i)}')


    for name in names:
        gen, valfit = find_top_val_gen(name, 'cen')
        print(f'build/plot matrix for: {name} @ {gen} w {valfit} fitness')
        
        build_agent_trajs_parallel(name, gen, space_step, orient_step, timesteps)
        plot_agent_trajs(name, gen, space_step, orient_step, timesteps, ellipses=False)
        # plot_agent_trajs(name, gen, space_step, orient_step, timesteps, ellipses=True)
        # plot_agent_trajs(name, gen, space_step, orient_step, timesteps, ellipses=False, extra='3d')
        # plot_agent_trajs(name, gen, space_step, orient_step, timesteps, extra='clip')

        # build_agent_trajs_parallel(name, gen, space_step, orient_step, timesteps, extra='n0')
        # plot_agent_trajs(name, gen, space_step, orient_step, timesteps, ellipses=False, extra='n0')
        # plot_agent_trajs(name, gen, space_step, orient_step, timesteps, ellipses=False, extra='3d_n0')

        # build_agent_trajs_parallel(name, gen, space_step, orient_step, timesteps, extra='n_pos')
        # plot_agent_trajs(name, gen, space_step, orient_step, timesteps, ellipses=False, extra='n_pos')
        # build_agent_trajs_parallel(name, gen, space_step, orient_step, timesteps, extra='n_neg')
        # plot_agent_trajs(name, gen, space_step, orient_step, timesteps, ellipses=False, extra='n_neg')
    
        # build_agent_trajs_parallel(name, gen, space_step, orient_step, timesteps, landmarks=True)
        # plot_agent_trajs(name, gen, space_step, orient_step, timesteps, ellipses=False, landmarks=True)
        # build_agent_trajs_parallel(name, gen, space_step, orient_step, timesteps, extra='n0', landmarks=True)
        # plot_agent_trajs(name, gen, space_step, orient_step, timesteps, ellipses=False, extra='n0', landmarks=True)
        # build_agent_trajs_parallel(name, gen, space_step, orient_step, timesteps, extra='nhalf', landmarks=True)
        # plot_agent_trajs(name, gen, space_step, orient_step, timesteps, ellipses=False, extra='nhalf', landmarks=True)

        # build_agent_trajalls_parallel(name, gen, space_step, orient_step, timesteps)
        # plot_agent_trajalls(name, gen, space_step, orient_step, timesteps)

        # analyses = ['pca', 'ica']
        # datasets = ['CNN']
        # comps = [2,3]
        # plot_agent_trajalls_pca(name, gen, space_step, orient_step, timesteps, analyses, datasets, comps)

        # build_Nact_matrix_parallel(name, gen, space_step, orient_step)
        # plot_Nact_imshow(name, gen, space_step, orient_step)
        # anim_Nact(name, space_step, orient_step)

        # plot_Nact_action_phase(name, gen, space_step, orient_step)

        # build_action_matrix(name, gen, space_step, orient_step)
        # plot_action_volume(name, gen, space_step, orient_step, 'low')
        # plot_action_volume(name, gen, space_step, orient_step, 'high')
        # transform_action_mesh(name, gen, space_step, orient_step)

        # build_agent_visfield_parallel(name, space_step, orient_step)
        # plot_visfield_imshow(name, space_step, orient_step)
        # anim_visfield(name, gen, space_step, orient_step)
