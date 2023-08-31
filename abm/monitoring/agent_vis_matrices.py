from abm.start_sim import reconstruct_NN
from abm.agent.agent import Agent

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
            position=(0,0),
            orientation=0,
            max_vel=5,
            collision_slowdown=0.1,
            FOV=float(envconf['AGENT_FOV']),
            vision_range=int(envconf["VISION_RANGE"]),
            visual_exclusion=bool(int(envconf["VISUAL_EXCLUSION"])),
            consumption=1,
            arch=arch,
            model=NN,
            NN_activ='relu',
            RNN_type='fnn',
            boundary_info=boundary_endpts,
            radius=10,
            color=(0,0,0),
        )

    agent.position = np.array((x,y))
    agent.orientation = orient
    agent.visual_sensing()
    vis_field_onehot = agent.encode_one_hot(agent.vis_field)

    vis_input = torch.from_numpy(vis_field_onehot).float().unsqueeze(0)
    vis_features = agent.model.cnn(vis_input)

    # from null initial activity (+ zero contact field)
    other_input = torch.zeros( int(envconf["CONTACT_FIELD_RESOLUTION"]) + int(envconf["RNN_INPUT_OTHER_SIZE"]) ).unsqueeze(0)
    hidden = torch.zeros(int(envconf["RNN_HIDDEN_SIZE"])).unsqueeze(0)

    RNN_in = torch.cat((vis_features, other_input), dim=1)
    RNN_out, hidden = agent.model.rnn(RNN_in, hidden)

    action = agent.model.lcl(RNN_out)
    action = torch.tanh(action) # scale to [-1:1]

    Nact = np.concatenate((vis_features, RNN_out, action), axis=None)
    
    return Nact



def build_Nact_matrix_parallel(name):

    exp_name, gen_ext, NN_ext = name

    # pull pv + envconf from save folders
    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    NN_pv_path = fr'{data_dir}/{exp_name}/{gen_ext}/{NN_ext}/NN_pickle.bin'
    with open(NN_pv_path,'rb') as f:
        pv = pickle.load(f)
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)

    # reconstruct model
    NN, arch = reconstruct_NN(envconf, pv)

    # construct boundary endpts
    width=int(envconf["ENV_WIDTH"])
    height=int(envconf["ENV_HEIGHT"])
    window_pad=int(envconf["WINDOW_PAD"])
    x_min, x_max = window_pad, window_pad + width
    y_min, y_max = window_pad, window_pad + height
    boundary_endpts = (x_min, x_max, y_min, y_max)

    # every grid position/direction
    x_range = np.arange(x_min, x_max+1, 1)
    y_range = np.arange(y_min, y_max+1, 1) 
    orient_range = np.arange(0, 2*np.pi, np.pi/8) 

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
    
    with open(fr'{data_dir}/Nact_matrices/{exp_name}_{gen_ext}_{NN_ext}_matrix.bin', 'wb') as f:
        pickle.dump(Nact_matrix, f)
    
    return Nact_matrix


def plot_Nactmat_imshow(name):

    exp_name, gen_ext, NN_ext = name

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    with open(fr'{data_dir}/Nact_matrices/{exp_name}_{gen_ext}_{NN_ext}_matrix.bin', 'rb') as f:
        mat = pickle.load(f)

    num_x, num_y, num_orient, num_neurons = mat.shape
    orient_range = np.linspace(0, 2*np.pi, num_orient+1)[:-1]

    fig, axs = plt.subplots(
        nrows=num_orient, 
        ncols=num_neurons, 
        figsize=(25, 25), 
        subplot_kw={'xticks': [], 'yticks': []}
        )

    ax = np.array(axs)
    for o in range(num_orient):
        for n in range(num_neurons):
            ax[o,n].imshow(mat[:,:,o,n],
                           vmin = mat[:,:,:,n].flatten().min(), 
                           vmax = mat[:,:,:,n].flatten().max(),
                           )
            # ax[o,n].set_title(f'orient: {int(orient_range[o]*180/np.pi)} | neuron: {n}', fontsize=10)

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    exp_name, gen_ext, NN_ext = name
    plt.savefig(fr'{data_dir}/Nact_matrices/{exp_name}_{gen_ext}_{NN_ext}_imshow_notitle.png')
    plt.close()

# -------------------- #

def plot_isomap(mat):

    import plotly.express as px
    from sklearn.manifold import Isomap

    num_x, num_y, num_orient, num_neurons = mat.shape

    # flatten neural activity matrix for each pos/dir * number of neurons
    mat_data = mat.reshape(num_x*num_y*num_orient, num_neurons)
    print(f'data: {mat_data.shape}')

    # build orientation matrix for labeling matrix above
    mat_label = np.zeros((num_x*num_y, num_orient))

    for i in range(num_x*num_y):
        for k in range(num_orient):
            mat_label[i,k] = k
    
    mat_label = mat_label.flatten()
    print(f'labels: {mat_label.shape}')

    ### Step 1 - Configure the Isomap function, note we use default hyperparameter values in this exa
    embed3 = Isomap(
        n_neighbors=5, # default=5, algorithm finds local structures based on the nearest neighbors
        n_components=3, # number of dimensions
        eigen_solver='auto', # {‘auto’, ‘arpack’, ‘dense’}, default=’auto’
        tol=0, # default=0, Convergence tolerance passed to arpack or lobpcg. not used if eigen_solve
        max_iter=None, # default=None, Maximum number of iterations for the arpack solver. not used i
        path_method='auto', # {‘auto’, ‘FW’, ‘D’}, default=’auto’, Method to use in finding shortest
        neighbors_algorithm='auto', # neighbors_algorithm{‘auto’, ‘brute’, ‘kd_tree’, ‘ball_tree’}, d
        n_jobs=-1, # n_jobsint or None, default=None, The number of parallel jobs to run. -1 means us
        metric='minkowski', # string, or callable, default=”minkowski”
        p=2, # default=2, Parameter for the Minkowski metric. When p = 1, this is equivalent to using
        metric_params=None # default=None, Additional keyword arguments for the metric function.
    )

    ### Step 2 - Fit the data and transform it, so we have 3 dimensions instead of 64
    X_trans3 = embed3.fit_transform(mat_data)

    ### Step 3 - Print shape to test
    print('The new shape of X: ',X_trans3.shape)

    # Create a 3D scatter plot
    fig = px.scatter_3d(None,
                        x=X_trans3[:,0], y=X_trans3[:,1], z=X_trans3[:,2],
                        color=mat_label.astype(str),
                        height=900, width=900
                        )

    # Update chart looks
    fig.update_layout(#title_text="Scatter 3D Plot",
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="top", y=0, xanchor="center", x=0.5),
                    scene_camera=dict(up=dict(x=0, y=0, z=1),
                                        center=dict(x=0, y=0, z=-0.2),
                                        eye=dict(x=-1.5, y=1.5, z=0.5),
                                        ),
                    margin=dict(l=0, r=0, b=0, t=0),
                    scene = dict(xaxis=dict(backgroundcolor='white',
                                            color='black',
                                            gridcolor='#f0f0f0',
                                            title_font=dict(size=10),
                                            tickfont=dict(size=10),
                                            ),
                                yaxis=dict(backgroundcolor='white',
                                            color='black',
                                            gridcolor='#f0f0f0',
                                            title_font=dict(size=10),
                                            tickfont=dict(size=10),
                                            ),
                                zaxis=dict(backgroundcolor='lightgrey',
                                            color='black',
                                            gridcolor='#f0f0f0',
                                            title_font=dict(size=10),
                                            tickfont=dict(size=10),
                                            )))

    # Update marker size
    fig.update_traces(marker=dict(size=2))
    fig.show()


# -------------------------- early trajectory -------------------------- #

def agent_traj_from_xyo(envconf, arch, NN, boundary_endpts, x, y, orient, timesteps):

    agent = Agent(
            id=0,
            position=(0,0),
            orientation=0,
            max_vel=5,
            collision_slowdown=0.1,
            FOV=float(envconf['AGENT_FOV']),
            vision_range=int(envconf["VISION_RANGE"]),
            visual_exclusion=bool(int(envconf["VISUAL_EXCLUSION"])),
            consumption=1,
            arch=arch,
            model=NN,
            NN_activ='relu',
            RNN_type='fnn',
            boundary_info=boundary_endpts,
            radius=10,
            color=(0,0,0),
        )

    agent.position = np.array((x,y))
    agent.orientation = orient

    traj = np.zeros((timesteps,2))
    for t in range(timesteps):

        agent.visual_sensing()
        agent.wall_contact_sensing() 

        vis_input, other_input = agent.assemble_NN_inputs()
        agent.action, agent.hidden = agent.model.forward(vis_input, other_input, agent.hidden)
        agent.move(agent.action)

        traj[t,:] = agent.pt_center
    
    return traj


def build_agent_trajs_parallel(name, space_step, orient_step, timesteps):

    exp_name, gen_ext, NN_ext = name

    # pull pv + envconf from save folders
    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    NN_pv_path = fr'{data_dir}/{exp_name}/{gen_ext}/{NN_ext}/NN_pickle.bin'
    with open(NN_pv_path,'rb') as f:
        pv = pickle.load(f)
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)

    # reconstruct model
    NN, arch = reconstruct_NN(envconf, pv)

    # construct boundary endpts
    width=int(envconf["ENV_WIDTH"])
    height=int(envconf["ENV_HEIGHT"])
    window_pad=int(envconf["WINDOW_PAD"])
    x_min, x_max = window_pad, window_pad + width
    y_min, y_max = window_pad, window_pad + height
    boundary_endpts = (x_min, x_max, y_min, y_max)

    # every grid position/direction
    x_range = np.arange(x_min, x_max+1, space_step)
    y_range = np.arange(y_min, y_max+1, space_step)
    orient_range = np.arange(0, 2*np.pi, orient_step)
    
    # construct matrix of each traj for each grid pos/dir
    num_inits = len(x_range) * len(y_range) * len(orient_range)
    traj_matrix = np.zeros( (num_inits, timesteps, 4) ) # (pos_x, pos_y, _, _) --> to match self.data_agent format
    print(f'traj matrix shape (# initializations, x, y): {traj_matrix.shape}')
    
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

    # unpack results into matrix 
    results_list = results.get()
    for n,pos_array in enumerate(results_list):
        traj_matrix[n,:,:2] = pos_array
    

    # transform data to plotting coords
    traj_matrix[:,:,0] = traj_matrix[:,:,0] - window_pad
    traj_matrix[:,:,1] = y_max - traj_matrix[:,:,1]

    # build resource coords for environment
    ag_data = np.zeros((2,1,3))
    res_radius = int(envconf["RADIUS_RESOURCE"])

    ag_data[0,0,:] = np.array((x_min + 120, 
                               y_min + 30,
                               res_radius))
    ag_data[1,0,:] = np.array((x_max - res_radius/2 - window_pad - 30, 
                               y_max - res_radius/2 - window_pad - 120,
                               res_radius))

    ag_data[:,:,0] = ag_data[:,:,0] + res_radius - window_pad
    ag_data[:,:,1] = y_max - (ag_data[:,:,1] + res_radius)


    # pack + save
    plot_data = traj_matrix, ag_data

    with open(fr'{data_dir}/traj_matrices/{exp_name}_{gen_ext}_{NN_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}.bin', 'wb') as f:
        pickle.dump(plot_data, f)

    return plot_data


def plot_agent_trajs(name, space_step, orient_step, timesteps):

    from abm.monitoring.plot_funcs import plot_map_iterative_traj

    exp_name, gen_ext, NN_ext = name

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)

    width=int(envconf["ENV_WIDTH"])
    height=int(envconf["ENV_HEIGHT"])

    with open(fr'{data_dir}/traj_matrices/{exp_name}_{gen_ext}_{NN_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}.bin', 'rb') as f:
        traj_plot_data = pickle.load(f)

    save_name = fr'{data_dir}/traj_matrices/{exp_name}_{gen_ext}_{NN_ext}_traj_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_ap2'
    plot_map_iterative_traj(traj_plot_data, x_max=width, y_max=height, save_name=save_name)


if __name__ == '__main__':

    space_step = 25
    orient_step = np.pi/8
    timesteps = 500

    names = []

    ### --- simple model --- ###

    # exp_name = 'doublepoint_CNN1128_FNN2_p25e5g1000_sig0p1_vis8_dirfit_simp_rep5'
    # gen_ext = 'gen969'
    # NN_ext = 'NN0_af8'
    # names.append( (exp_name, gen_ext, NN_ext) )
    # exp_name = 'doublepoint_CNN1128_FNN2_p25e5g1000_sig0p1_vis8_dirfit_simp_rep0'
    # gen_ext = 'gen922'
    # NN_ext = 'NN0_af6'
    # names.append( (exp_name, gen_ext, NN_ext) )
    # exp_name = 'doublepoint_CNN1128_FNN2_p25e5g1000_sig0p1_vis8_dirfit_simp_rep1'
    # gen_ext = 'gen998'
    # NN_ext = 'NN0_af6'
    # names.append( (exp_name, gen_ext, NN_ext) )

    # exp_name = 'doublepoint_CNN1128_CTRNN2_p25e5g1000_sig0p1_vis8_dirfit_simp_rep0'
    # gen_ext = 'gen913'
    # NN_ext = 'NN0_af6'
    # names.append( (exp_name, gen_ext, NN_ext) )
    # exp_name = 'doublepoint_CNN1128_CTRNN2_p25e5g1000_sig0p1_vis8_dirfit_simp_rep2'
    # gen_ext = 'gen990'
    # NN_ext = 'NN0_af5'
    # gen_ext = 'gen816'
    # NN_ext = 'NN0_af5'
    # names.append( (exp_name, gen_ext, NN_ext) )
    # exp_name = 'doublepoint_CNN1128_CTRNN2_p25e5g1000_sig0p1_vis8_dirfit_simp_rep3'
    # gen_ext = 'gen930'
    # NN_ext = 'NN0_af8'
    # names.append( (exp_name, gen_ext, NN_ext) )

    # exp_name = 'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_simp_rep3'
    # gen_ext = 'gen876'
    # NN_ext = 'NN0_af8'
    # names.append( (exp_name, gen_ext, NN_ext) )
    # exp_name = 'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_simp_rep4'
    # gen_ext = 'gen890'
    # NN_ext = 'NN0_af8'
    # names.append( (exp_name, gen_ext, NN_ext) )


    ### --- complex model --- ###


    # exp_name = 'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep2'
    # gen_ext = 'gen549'
    # NN_ext = 'NN0_af8'
    # names.append( (exp_name, gen_ext, NN_ext) )

    # exp_name = 'doublepoint_CNN1128_CTRNN2_p25e5g1000_sig0p1_vis8_dirfit_other0_rep4'
    # gen_ext = 'gen969'
    # NN_ext = 'NN0_af7'
    # names.append( (exp_name, gen_ext, NN_ext) )

    # exp_name = 'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_other0_rep0'
    # gen_ext = 'gen997'
    # NN_ext = 'NN0_af7'
    # names.append( (exp_name, gen_ext, NN_ext) )

    # exp_name = 'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_other0_rep2'
    # gen_ext = 'gen598'
    # NN_ext = 'NN0_af8'
    # names.append( (exp_name, gen_ext, NN_ext) )
    # exp_name = 'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_other0_rep2'
    # gen_ext = 'gen935'
    # NN_ext = 'NN0_af7'
    # names.append( (exp_name, gen_ext, NN_ext) )
    # exp_name = 'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_other0_rep3'
    # gen_ext = 'gen964'
    # NN_ext = 'NN0_af7'
    # names.append( (exp_name, gen_ext, NN_ext) )
    

    exp_name = 'doublepoint_CNN1122_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep0'
    gen_ext = 'gen969'
    NN_ext = 'NN0_af7'
    names.append( (exp_name, gen_ext, NN_ext) )



    for name in names:
        print(f'build/plot matrix for: {name}')
        Nact_matrix = build_Nact_matrix_parallel(name)
        plot_Nactmat_imshow(name)
        build_agent_trajs_parallel(name, space_step, orient_step, timesteps)
        plot_agent_trajs(name, space_step, orient_step, timesteps)