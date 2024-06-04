from abm.start_sim import reconstruct_NN
from abm.sprites.agent import Agent
# from abm.sprites.agent_LM import Agent
from abm.sprites.landmark import Landmark
from abm.monitoring.plot_funcs import plot_map_iterative_traj, plot_map_iterative_traj_3d, plot_map_iterative_trajall

import dotenv as de
from pathlib import Path
import numpy as np
import scipy
import torch
import multiprocessing as mp
import _pickle as pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


# -------------------------- action -------------------------- #

def agent_action_from_xyo(envconf, NN, boundary_endpts, x, y, orient):

    width, height = tuple(eval(envconf["ENV_SIZE"]))
    agent_radius = int(envconf["RADIUS_AGENT"])
    vis_transform = str(envconf["VIS_TRANSFORM"])
    angl_noise_std = float(envconf["PERCEP_ANGLE_NOISE_STD"])
    dist_noise_std = float(envconf["PERCEP_DIST_NOISE_STD"])
    other_input = int(envconf["RNN_OTHER_INPUT_SIZE"])

    max_dist = np.hypot(width, height)
    min_dist = agent_radius*2

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
            vis_transform=vis_transform,
            percep_angle_noise_std=angl_noise_std,
        )

    # gather visual input
    agent.visual_sensing([])
    vis_input = agent.encode_one_hot(agent.vis_field)

    if vis_transform != '':
        dist_input = np.array(agent.dist_field)
        if vis_transform == 'minmax':
            dist_input = (dist_input - min_dist) / (max_dist - min_dist)
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
        dist_input += noise
        # dist_input /= 1.5
        # dist_input += .05
        dist_input = np.clip(dist_input, 0,1)
        vis_input *= dist_input

    if other_input == 2:
        agent.action, agent.hidden = agent.model.forward(vis_input, np.array([0, agent.acceleration / 2]), agent.hidden)
    else:
        agent.action, agent.hidden = agent.model.forward(vis_input, np.array([0]), agent.hidden)
    
    return agent.action


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

    with open(fr'{data_dir}/action_maps/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_action.bin', 'wb') as f:
        pickle.dump(act_matrix, f)
    
    return np.min(act_matrix), np.max(act_matrix)

    
def plot_action_vecfield(exp_name, gen_ext, space_step, orient_step, plot_type='', colored='count', ex_lines=False, dpi=50):
    print(f'plotting action vector field - {exp_name} @ {dpi} dpi')

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)

    width, height = tuple(eval(envconf["ENV_SIZE"]))
    x_min, x_max = 0, width
    y_min, y_max = 0, height
    coll_boundary_thickness = int(envconf["RADIUS_AGENT"])
    x_range = np.linspace(x_min + coll_boundary_thickness, 
                        x_max - coll_boundary_thickness + 1, 
                        int((width - coll_boundary_thickness*2) / space_step))
    y_range = np.linspace(y_min + coll_boundary_thickness, 
                        y_max - coll_boundary_thickness + 1, 
                        int((height - coll_boundary_thickness*2) / space_step))
    orient_range = np.arange(0, 2*np.pi, orient_step)
    # print(f'orient_range: {np.round(orient_range,2)}')
    # print('')

    X,Y = np.meshgrid(x_range, y_range)
    U,V = np.meshgrid(x_range, y_range)
    M = np.zeros_like(U)

    if plot_type == '_peaks_fwd' or plot_type == '_peaks_turn' or plot_type == '_fwdentropy':
        Ua = np.zeros_like(U)
        Ub = np.zeros_like(U)
        Uc = np.zeros_like(U)
        Ud = np.zeros_like(U)
        Ue = np.zeros_like(U)
        Uf = np.zeros_like(U)
        Ug = np.zeros_like(U)
        Uh = np.zeros_like(U)
        Ui = np.zeros_like(U)

        Va = np.zeros_like(V)
        Vb = np.zeros_like(V)
        Vc = np.zeros_like(V)
        Vd = np.zeros_like(V)
        Ve = np.zeros_like(V)
        Vf = np.zeros_like(V)
        Vg = np.zeros_like(V)
        Vh = np.zeros_like(V)
        Vi = np.zeros_like(V)

        U_array = [U, Ua, Ub, Uc, Ud, Ue, Uf, Ug, Uh, Ui]
        V_array = [V, Va, Vb, Vc, Vd, Ve, Vf, Vg, Vh, Vi]

    # elif plot_type == '_fwd_dirent' or plot_type == '_turn_dirent':
    def calc_entropy(h):
        h_norm = h / np.sum(h)
        e = -np.sum( h_norm*np.log(h_norm) )
        return e

    h = np.ones(len(orient_range))/10000
    e_max = calc_entropy(h) # random
    h[0] = 1
    e_min = calc_entropy(h) # uniform
    # print(f'e_max: {e_max}, e_min: {e_min}')


    fig, axes = plt.subplots() 
    axes.set_xlim(0, x_max)
    axes.set_ylim(0, y_max)
    h,w = 8,8
    l,r,t,b = fig.subplotpars.left, fig.subplotpars.right, fig.subplotpars.top, fig.subplotpars.bottom
    fig.set_size_inches( float(w)/(r-l) , float(h)/(t-b) )

    save_name = fr'{data_dir}/action_maps/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_action'
    with open(save_name+'.bin', 'rb') as f:
        act_matrix = pickle.load(f)
    # print(act_matrix.shape)
    # print(f'action range: {abs(act_matrix.max() - act_matrix.min())}')
    act_matrix = np.abs(act_matrix)
    act_matrix = (act_matrix - act_matrix.min()) / (act_matrix.max() - act_matrix.min())

    inits = [
        [700, 200], #BR
        [700, 600], #TR
        [100, 200], #BL
        [200, 700], #TL
        [200, 500], #midL
        [900, 500], #midR
        [500, 500], #mid
        [350, 350], #patch
    ]
    # for x,y in inits:
    #     x_idx = (np.abs(x_range - x)).argmin()
    #     y_idx = (np.abs(y_range - y)).argmin()
    #     print(f'x: {x}, y: {y}')
    #     # print(f'x_idx in range: {x_range[x_idx]}, y_idx in range: {y_range[y_idx]}')
    #     print(f'act_matrix[x_idx,y_idx,ori_idx]: {act_matrix[x_idx,y_idx,:]}')

    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):

            actions = act_matrix[i,j,:]
            xs = np.cos(orient_range)
            ys = np.sin(orient_range)

            if plot_type == '_avg':
                if np.sum((1-actions)) == 0: # if all actions are 1, np.average cannot compute
                    U[j,i] = 0
                    V[j,i] = 0
                else:
                    U[j,i] = np.average(xs, weights=(1-actions))
                    V[j,i] = np.average(ys, weights=(1-actions))

            elif plot_type == '_peaks_fwd' or plot_type == '_peaks_turn':

                if plot_type == '_peaks_fwd':
                    actions = 1-actions
                elif plot_type == '_peaks_turn':
                    actions = actions
                aa = np.concatenate((actions,actions))
                xsxs = np.concatenate((xs,xs))
                ysys = np.concatenate((ys,ys))

                peaks,_ = scipy.signal.find_peaks(aa, height=.75, prominence=.3)
                # peaks,_ = scipy.signal.find_peaks(fafa, height=.5, prominence=.2)
                peaks_shift = peaks + len(actions)
                pp = np.concatenate((peaks,peaks_shift))
                repeats = [item for item in set(pp) if list(pp).count(item) > 1]
                peaks = np.setdiff1d(peaks, repeats)

                M[j,i] = len(peaks)

                if len(peaks) == 0:
                    for z in range(7):
                        U_array[z][j,i] = 0
                        V_array[z][j,i] = 0
                else:
                    for z,p in enumerate(peaks):
                        U_array[z][j,i] = xsxs[p]
                        V_array[z][j,i] = ysys[p]
                        if z == 6: break

            elif plot_type == '_dirent_turn':
                e = calc_entropy(actions+.00000000001)
                M[j,i] = (e_max - e) / (e_max - e_min) # directedness
                # M[j,i] = (e - e_min) / (e_max - e_min) # entropy

            elif plot_type == '_dirent_fwd':
                e = calc_entropy((1.00000000001-actions))
                M[j,i] = (e_max - e) / (e_max - e_min) # directedness
                # M[j,i] = (e - e_min) / (e_max - e_min) # entropy
            
            elif plot_type == '_fwdentropy':
                e = calc_entropy((1.00000000001-actions))
                # M[j,i] = (e_max - e) / (e_max - e_min) # directedness
                M[j,i] = (e - e_min) / (e_max - e_min) # entropy

                # actions = actions
                actions = 1-actions
                aa = np.concatenate((actions,actions))
                xsxs = np.concatenate((xs,xs))
                ysys = np.concatenate((ys,ys))

                peaks,_ = scipy.signal.find_peaks(aa, height=.75, prominence=.3)
                # peaks,_ = scipy.signal.find_peaks(fafa, height=.5, prominence=.2)
                peaks_shift = peaks + len(actions)
                pp = np.concatenate((peaks,peaks_shift))
                repeats = [item for item in set(pp) if list(pp).count(item) > 1]
                peaks = np.setdiff1d(peaks, repeats)

                if len(peaks) == 0:
                    for z in range(7):
                        U_array[z][j,i] = 0
                        V_array[z][j,i] = 0
                else:
                    for z,p in enumerate(peaks):
                        U_array[z][j,i] = xsxs[p]
                        V_array[z][j,i] = ysys[p]
                        if z == 6: break
            
            elif plot_type == '_avgact':
                M[j,i] = np.mean(actions)
            
            else:
                print('invalid plot type')


    if colored == 'ori':
        M = np.arctan2(V, U)
        M = (M + 2*np.pi)%(2*np.pi)
        norm = mpl.colors.Normalize(vmin=0, vmax=2*np.pi)
        # norm = mpl.colors.Normalize(vmin=-np.pi, vmax=np.pi)

        # axes.contourf(X, Y, M, cmap=cmap, norm=norm) # contoured background (issue with 0-2pi discontinuity)
        im = axes.imshow(M, cmap='hsv', norm=norm, extent=(x_min, x_max, y_min, y_max), origin='lower', alpha=.6) # pixelated background
        axes.quiver(X,Y, U,V) # black arrows
        # Q = axes.quiver(X,Y, U,V, M, angles='xy', cmap=cmap, norm=norm) # colored arrows
        # Q = axes.streamplot(X,Y, U,-V, color='k', broken_streamlines=False) # streams
        # plt.colorbar(im, label='Orientation')

    elif colored == 'len':
        M = np.hypot(U, V)
        # norm = mpl.colors.Normalize(vmin=0, vmax=0.25)
        # # norm = mpl.colors.Normalize(vmin=0, vmax=0.5)
        # # norm = mpl.colors.Normalize(vmin=0, vmax=np.max(M))

        # im = axes.imshow(M, cmap='plasma', norm=norm, extent=(x_min, x_max, y_min, y_max), origin='lower', alpha=.6)
        # plt.colorbar(im, label='Mean Action Vector Length (Assuredness)')
        # Q = axes.quiver(X,Y, U,V, pivot='mid')
    
        levs = np.linspace(0, .5, 11)
        im = axes.contourf(X,Y,M, levs, cmap='plasma', alpha=.6, extend='max')
        # im = axes.contourf(X,Y,M, cmap='plasma', alpha=.6)
        plt.colorbar(im, label='Mean Action Vector Length')
        axes.set_title(f'Avg: {np.mean(M):.2f}, Median: {np.median(M):.2f}, Min: {np.min(M):.2f}, Max: {np.max(M):.2f}')
        axes.quiver(X,Y, U,V)

    elif plot_type == '_peaks_fwd' or plot_type == '_peaks_turn':
        # print(f'max peaks: {np.max(M)}')
        # norm = mpl.colors.Normalize(vmin=0, vmax=7)
        # im = axes.imshow(M, cmap='plasma', norm=norm, extent=(x_min, x_max, y_min, y_max), origin='lower', alpha=.6)
        levs = np.linspace(0, 7, 8)
        im = axes.contourf(X,Y,M, levs, cmap='plasma', alpha=.6, extend='max')
        plt.colorbar(im, label='# Peaks in Action Vector')
        axes.set_title(f'Avg: {np.mean(M):.2f}, Median: {np.median(M):.2f}, Min: {np.min(M):.2f}, Max: {np.max(M):.2f}')

        for z in range(10):
            r = np.power(np.add(np.power(U_array[z],2), np.power(V_array[z],2)),0.5)
            # axes.quiver(X,Y, U_array[z]/r, V_array[z]/r, pivot='mid')
            axes.quiver(X,Y, U_array[z]/r, V_array[z]/r)

    elif plot_type == '_dirent_fwd' or plot_type == '_dirent_turn':
        # norm = mpl.colors.Normalize(vmin=0, vmax=.3)
        # im = axes.imshow(M, cmap='plasma', norm=norm, extent=(x_min, x_max, y_min, y_max), origin='lower', alpha=.6)

        levs = np.linspace(0, .3, 11)
        im = axes.contourf(X, Y, M, levs, cmap='plasma', alpha=.6, extend='max')
        # im = axes.contourf(X, Y, M, cmap='plasma', alpha=.6)
        if plot_type == '_dirent_fwd':
            plt.colorbar(im, label='Entropic Directedness - Fwd')
        elif plot_type == '_dirent_turn':
            plt.colorbar(im, label='Entropic Directedness - Turn')
        axes.set_title(f'Avg: {np.mean(M):.2f}, Median: {np.median(M):.2f}, Min: {np.min(M):.2f}, Max: {np.max(M):.2f}')

    elif plot_type == '_fwdentropy':
        levs = np.linspace(.8, 1, 11)
        im = axes.contourf(X,Y,M, levs, cmap='plasma', alpha=.6, extend='min')
        # im = axes.contourf(X,Y,M, cmap='plasma', alpha=.6)
        plt.colorbar(im, label='Entropy')
        axes.set_title(f'Avg: {np.mean(M):.2f}, Median: {np.median(M):.2f}, Min: {np.min(M):.2f}, Max: {np.max(M):.2f}')

        for z in range(10):
            r = np.power(np.add(np.power(U_array[z],2), np.power(V_array[z],2)),0.5)
            # axes.quiver(X,Y, U_array[z]/r, V_array[z]/r, pivot='mid')
            axes.quiver(X,Y, U_array[z]/r, V_array[z]/r)

    elif plot_type == '_avgact':
        levs = np.linspace(0, 1, 11)
        im = axes.contourf(X,Y,M, levs, cmap='plasma', alpha=.6)
        # im = axes.contourf(X,Y,M, cmap='plasma', alpha=.6)
        plt.colorbar(im, label='Mean Action')
        axes.set_title(f'Avg: {np.mean(M):.2f}, Median: {np.median(M):.2f}, Min: {np.min(M):.2f}, Max: {np.max(M):.2f}')

    else:
        Q = axes.quiver(X,Y, U,V)


    if ex_lines:
        save_name_traj = fr'{data_dir}/traj_matrices/{exp_name}_{gen_ext}_c25_o8_t500_cen_e1'
        with open(save_name_traj+'.bin', 'rb') as f:
            ag_data = pickle.load(f)

        from scipy import spatial

        traj_inits = [
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

        for pt,color in zip(traj_inits,colors):

            distance, index_xy = spatial.KDTree(ag_data[:,0,:2]).query(pt[:2])
            array = ag_data[index_xy:index_xy+16,0,2]
            value = pt[2]
            index_ori = (np.abs(array - value)).argmin()

            index = index_xy + index_ori
            pos_x = ag_data[index,:,0]
            pos_y = ag_data[index,:,1]

            axes.plot(pos_x, height-pos_y, color)
            axes.plot(pos_x, height-pos_y, 'k:')
            axes.plot(pos_x[0], height-pos_y[0], marker='o', c=color, markeredgecolor='k', ms=10)    


    # for x,y in inits:
    #     x_idx = (np.abs(x_range - x)).argmin()
    #     y_idx = (np.abs(y_range - y)).argmin()
    #     # axes.scatter(x, y, c='k', s=50)

    #     xs = np.cos(orient_range)
    #     ys = np.sin(orient_range)
    #     acts = abs(act_matrix[x_idx,y_idx,:])

    #     x_avg = np.average(xs, weights=(1-acts))
    #     y_avg = np.average(ys, weights=(1-acts))

    #     avg_len = np.hypot(x_avg, y_avg)
    #     avg_ori = np.arctan2(y_avg, x_avg)
    #     num_peaks = len(scipy.signal.find_peaks((1-acts), height=.75, prominence=.3)[0])
    #     e = calc_entropy(acts+.00000000001)
    #     dir = (e_max - e) / (e_max - e_min)
    #     e = calc_entropy((1.00000001-acts))
    #     dir_neg = (e_max - e) / (e_max - e_min)

    #     print(f'x: {x}, y: {y}, avg_ori: {avg_ori:.2f}, avg_len: {avg_len:.2f}, num_peaks: {num_peaks}, turn_dirent: {dir:.2f}, fwd_dirent: {dir_neg:.2f}')
    #     print(f'actions: {np.round((1-acts),2)}')
    # print('')


    axes.set_ylim(axes.get_ylim()[1], axes.get_ylim()[0]) # flip y-axis

    radius = int(envconf["RADIUS_RESOURCE"])
    x,y = tuple(eval(envconf["RESOURCE_POS"]))
    axes.add_patch( plt.Circle((x, y), radius, edgecolor='k', fill=False, zorder=1) )

    if colored == 'count':
        plt.savefig(fr'{save_name}{plot_type}.png', dpi=dpi)
    else:
        plt.savefig(fr'{save_name}{plot_type}_C{colored}.png', dpi=dpi)
    plt.close()

    return np.mean(M), np.median(M), np.min(M), np.max(M)


# def plot_action_volume(exp_name, gen_ext, space_step, orient_step, transform='high'):

#     data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
#     with open(fr'{data_dir}/act_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}.bin', 'rb') as f:
#         mat = pickle.load(f)

#     # NN_pv_path = fr'{data_dir}/{exp_name}/{gen_ext}_NN0_pickle.bin'
#     NN_pv_path = fr'{data_dir}/{exp_name}/{gen_ext}_NNcen_pickle.bin'
#     with open(NN_pv_path,'rb') as f:
#         pv = pickle.load(f)
#     env_path = fr'{data_dir}/{exp_name}/.env'
#     envconf = de.dotenv_values(env_path)

#     # gather grid params
#     x_min, x_max = 0, int(envconf["ENV_WIDTH"])
#     y_min, y_max = 0, int(envconf["ENV_HEIGHT"])
#     coll_boundary_thickness = int(envconf["RADIUS_AGENT"])
#     x_range = np.linspace(x_min + coll_boundary_thickness, 
#                         x_max - coll_boundary_thickness + 1, 
#                         int((x_max - coll_boundary_thickness*2) / space_step))
#     y_range = np.linspace(y_min + coll_boundary_thickness, 
#                         y_max - coll_boundary_thickness + 1, 
#                         int((y_max - coll_boundary_thickness*2) / space_step))
#     orient_range = np.arange(0, 2*np.pi, orient_step) 

#     # construct vectors for each grid pos/dir
#     xs, ys, os = [], [], []
#     for x in x_range:
#         for y in y_range:
#             for o in orient_range:
#                 xs.append(x)
#                 ys.append(y)
#                 os.append(o)

#     # set up plot
#     fig = plt.figure(
#         figsize=(25, 25), 
#         )
#     ax = fig.add_subplot(projection='3d')

#     # transform action data
#     actions = abs(mat)
#     if transform == 'low':
#         actions = ( actions.max() - actions ) / actions.max()
#     elif transform == 'high':
#         actions = actions / actions.max()
#     actions = actions.flatten()

#     # plot abs(action) for every position (x,y) at every orientation (z)
#     ax.scatter(xs, ys, os,
#         cmap = 'Blues',
#         c = actions,
#         alpha = .1*actions,
#         s = 100*actions,
#         )

#     plt.savefig(fr'{data_dir}/act_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_{transform}.png')
#     plt.close()


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
        # plot_map_iterative_traj_3d(traj_plot_data, x_max=width, y_max=height, save_name=save_name, plt_type='scatter', var='cturn')
        plot_map_iterative_traj_3d(traj_plot_data, x_max=width, y_max=height, save_name=save_name, plt_type='scatter', var='str_manif')
        # plot_map_iterative_traj_3d(traj_plot_data, x_max=width, y_max=height, save_name=save_name, plt_type='lines', var='cturn')
        # plot_map_iterative_traj_3d(traj_plot_data, x_max=width, y_max=height, save_name=save_name, plt_type='lines', var='ctime')
        # plot_map_iterative_traj_3d(traj_plot_data, x_max=width, y_max=height, save_name=save_name, plt_type='lines', var='ctime_flat')
        # plot_map_iterative_traj_3d(traj_plot_data, x_max=width, y_max=height, save_name=save_name, plt_type='lines', var='ctime_arrows')
        # plot_map_iterative_traj_3d(traj_plot_data, x_max=width, y_max=height, save_name=save_name, plt_type='lines', var='ctime_arrows_only')
        # plot_map_iterative_traj_3d(traj_plot_data, x_max=width, y_max=height, save_name=save_name, plt_type='lines', var='ctime_arrows_only', sv_typ='anim')
    else:
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


# -------------------------- correlations -------------------------- #

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
    if not os.path.exists(save_name+'.bin'):
        print(f'no data found for {save_name}')
        return 0,0,0
    with open(save_name+'.bin', 'rb') as f:
        ag_data = pickle.load(f)
    save_name = fr'{data_dir}/corrs/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e{int(eye)}'

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

    # wrap before peak finding
    nn = np.concatenate((n,n))
    peaks,_ = scipy.signal.find_peaks(nn, prominence=25000)
    peaks_shift = peaks + len(n)
    pp = np.concatenate((peaks,peaks_shift))
    repeats = [item for item in set(pp) if list(pp).count(item) > 1]
    peaks = np.setdiff1d(peaks, repeats)
    histo_peaks = len(peaks)

    plt.tight_layout()
    plt.savefig(fr'{save_name}_corr_polar_{dpi}.png', dpi=dpi)
    plt.close()

    ### directedness ###

    def calc_entropy(h):
        h_norm = h / np.sum(h)
        e = -np.sum( h_norm*np.log(h_norm) )
        return e

    h = np.ones(len(n))/10000
    e_max = calc_entropy(h) # random
    h[0] = 1
    e_min = calc_entropy(h) # uniform

    e = calc_entropy(n)
    dirent = (e_max - e) / (e_max - e_min)

    print(f'num peaks: {corr_peaks} // histo avg: {-round(histo_avg,2)} // histo peaks: {histo_peaks} // dirent: {round(dirent,2)}')

    return corr_peaks, -histo_avg, histo_peaks, dirent


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


def plot_binned_orient_turn_gradients(exp_name, gen_ext, space_step, orient_step, timesteps, rank='cen', eye=True, dpi=None):
    print(f'plotting corr - {exp_name} @ {dpi} dpi')

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)

    # build resource coord matrix
    res_radius = int(envconf["RADIUS_RESOURCE"])
    width, height = tuple(eval(envconf["ENV_SIZE"]))
    x,y = tuple(eval(envconf["RESOURCE_POS"]))
    res_data = np.zeros((1,1,3)) # 1 patch
    res_data[0,0,:] = np.array((x, height - y, res_radius))

    # pull in ag data
    save_name = fr'{data_dir}/traj_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e{int(eye)}'
    with open(save_name+'.bin', 'rb') as f:
        ag_data = pickle.load(f)
    print(ag_data.shape)

    # flatten ag arrays
    x = ag_data[:,:,0].flatten()
    y = ag_data[:,:,1].flatten()
    orient = ag_data[:,:,2].flatten()
    turn = ag_data[:,:,3].flatten()

    # slice data into binned orients + mask outsiders
    num_cuts = 8
    angle_slices = np.linspace(0, 2*np.pi, num_cuts+1)

    for i in range(num_cuts):

        print(f'slice: {round(angle_slices[i],2), round(angle_slices[i+1],2)}')
        m = np.ma.masked_outside(orient, angle_slices[i], angle_slices[i+1])

        x_masked = (1-m.mask)*x
        x_masked = x_masked[x_masked != 0]

        y_masked = (1-m.mask)*y
        y_masked = y_masked[y_masked != 0]

        # orient_masked = (1-m.mask)*orient
        # orient_masked = orient_masked[orient_masked != 0]
        # print(np.min(orient_masked), np.max(orient_masked))

        turn_masked = (1-m.mask)*turn
        turn_masked = turn_masked[turn_masked != 0]
        # print(np.min(turn_masked), np.max(turn_masked))

        ag_data = np.array([x_masked, y_masked, turn_masked]).transpose()
        ag_data = np.expand_dims(ag_data, axis=0)
        print(ag_data.shape)

        traj_plot_data = (ag_data, res_data)
        
        plot_map_iterative_traj(traj_plot_data, x_max=width, y_max=height, save_name=save_name+f'_turn_slice{i}', ellipses=False, ex_lines=False, extra='turn')


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



# -------------------------- vis matching -------------------------- #

def agent_pano_from_xyo(envconf, NN, boundary_endpts, x, y, orient, vis_field_res):

    agent = Agent(
            id=0,
            position=(x,y),
            orientation=orient,
            max_vel=int(envconf["MAXIMUM_VELOCITY"]),
            FOV=float(envconf['AGENT_FOV']),
            # vis_field_res=int(envconf["VISUAL_FIELD_RESOLUTION"]),
            vis_field_res=vis_field_res,
            # vision_range=int(envconf["VISION_RANGE"]),
            vision_range=1,
            num_class_elements=4,
            consumption=1,
            model=NN,
            boundary_endpts=boundary_endpts,
            window_pad=30,
            radius=int(envconf["RADIUS_AGENT"]),
            color=(0,0,0),
            vis_transform='',
            percep_angle_noise_std=0,
        )

    agent.visual_sensing([])

    return agent.vis_field


def build_IDM(exp_name, gen_ext, space_step, orient_step, template_orient=0, vis_field_res=32):

    # pull pv + envconf from save folders
    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    NN_pv_path = fr'{data_dir}/{exp_name}/{gen_ext}_NNcen_pickle.bin'
    with open(NN_pv_path,'rb') as f:
        pv = pickle.load(f)
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)
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
    coll_boundary_thickness = int(envconf["RADIUS_AGENT"])
    x_range = np.linspace(x_min + coll_boundary_thickness, 
                        x_max - coll_boundary_thickness + 1, 
                        int((width - coll_boundary_thickness*2) / space_step))
    y_range = np.linspace(y_min + coll_boundary_thickness, 
                        y_max - coll_boundary_thickness + 1, 
                        int((height - coll_boundary_thickness*2) / space_step))
    orient_range = np.arange(0, 2*np.pi, orient_step)

    # template at patch
    x,y = tuple(eval(envconf["RESOURCE_POS"]))
    patch_pano = agent_pano_from_xyo(envconf, NN, boundary_endpts, x, y, template_orient, vis_field_res)
    # print(patch_pano)

    # rotational matrix at patch
    IDM = np.zeros((len(orient_range),
                            ))
    print(f'img diff matrix (rot) shape (orient): {IDM.shape}')

    for k, orient in enumerate(orient_range):
        rot_pano = agent_pano_from_xyo(envconf, NN, boundary_endpts, x, y_max-y, orient, vis_field_res)
        IDM[k] = sum(1 for i, j in zip(patch_pano, rot_pano) if i != j)

    with open(fr'{data_dir}/IDM/IDM_rot_o{int(np.pi/orient_step)}_tempori{round(template_orient,2)}_vsres{vis_field_res}.bin', 'wb') as f:
        pickle.dump(IDM, f)


    # translational matrix across grid
    IDM = np.zeros((len(x_range),
                            len(y_range),
                            ))
    print(f'img diff matrix (trans) shape (x, y): {IDM.shape}')

    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            trans_pano = agent_pano_from_xyo(envconf, NN, boundary_endpts, x, y, template_orient, vis_field_res)
            diffcount = sum(1 for i, j in zip(patch_pano, trans_pano) if i != j)
            IDM[i,j] = diffcount

            # if diffcount < 1:
            #     print(x,y,diffcount)

    with open(fr'{data_dir}/IDM/IDM_trans_c{space_step}_tempori{round(template_orient,2)}_vsres{vis_field_res}.bin', 'wb') as f:
        pickle.dump(IDM, f)


    # trans across grid + closest rot view
    IDM = np.zeros((len(x_range),
                            len(y_range),
                            2
                            ))
    print(f'img diff matrix (trans) shape (x, y, [count/orient]): {IDM.shape}')

    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            counts = []
            for k, orient in enumerate(orient_range):
                trans_pano = agent_pano_from_xyo(envconf, NN, boundary_endpts, x, y, orient, vis_field_res)
                diffcount = sum(1 for i, j in zip(patch_pano, trans_pano) if i != j)
                counts.append(diffcount)
            counts = np.array(counts)
            IDM[i,j,0] = np.min(counts)
            IDM[i,j,1] = orient_range[np.argmin(counts)]

    with open(fr'{data_dir}/IDM/IDM_transrot_c{space_step}_o{int(np.pi/orient_step)}_tempori{round(template_orient,2)}_vsres{vis_field_res}.bin', 'wb') as f:
        pickle.dump(IDM, f)



def plot_IDM(exp_name, gen_ext, space_step, orient_step, template_orient=0, vis_field_res=32, plot_type='', dpi=50):

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)

    width, height = tuple(eval(envconf["ENV_SIZE"]))
    x_min, x_max = 0, width
    y_min, y_max = 0, height
    coll_boundary_thickness = int(envconf["RADIUS_AGENT"])
    x_range = np.linspace(x_min + coll_boundary_thickness, 
                        x_max - coll_boundary_thickness + 1, 
                        int((width - coll_boundary_thickness*2) / space_step))
    y_range = np.linspace(y_min + coll_boundary_thickness, 
                        y_max - coll_boundary_thickness + 1, 
                        int((height - coll_boundary_thickness*2) / space_step))
    orient_range = np.arange(0, 2*np.pi, orient_step)

    patch_radius = int(envconf["RADIUS_RESOURCE"])
    patch_x, patch_y = tuple(eval(envconf["RESOURCE_POS"]))

    if plot_type == '_rot':
        print(f'plotting IDM (rot) @ {dpi} dpi')

        fig, axes = plt.subplots()
        axes.set_xlim(0, 2*np.pi)
        # axes.set_ylim(0, 2*np.pi)
        # h,w = 8,8
        # l,r,t,b = fig.subplotpars.left, fig.subplotpars.right, fig.subplotpars.top, fig.subplotpars.bottom
        # fig.set_size_inches( float(w)/(r-l) , float(h)/(t-b) )

        save_name = fr'{data_dir}/IDM/IDM_rot_o{int(np.pi/orient_step)}_tempori{round(template_orient,2)}_vsres{vis_field_res}'
        with open(save_name+'.bin', 'rb') as f:
            IDM = pickle.load(f)

        axes.plot(orient_range, IDM, 'k', alpha=.7)
        axes.plot(orient_range, IDM, 'ko', alpha=.7)
    
        axes.set_title(f'Rotation w. template ori: {round(template_orient,2)}')
        axes.set_ylabel(f'Pixel Diff Count')
        axes.set_xlabel('Orientation')
        plt.savefig(fr'{save_name}.png', dpi=dpi)
        plt.close()
    
    else:

        fig, axes = plt.subplots() 
        axes.set_xlim(0, x_max)
        axes.set_ylim(0, y_max)
        h,w = 8,8
        l,r,t,b = fig.subplotpars.left, fig.subplotpars.right, fig.subplotpars.top, fig.subplotpars.bottom
        fig.set_size_inches( float(w)/(r-l) , float(h)/(t-b) )

        save_name = fr'{data_dir}/IDM/IDM_trans_c{space_step}_tempori{round(template_orient,2)}_vsres{vis_field_res}'
        with open(save_name+'.bin', 'rb') as f:
            IDM = pickle.load(f)

        if plot_type == '_trans':
            print(f'plotting IDM (trans) @ {dpi} dpi')

            save_name = fr'{data_dir}/IDM/IDM_trans_c{space_step}_tempori{round(template_orient,2)}_vsres{vis_field_res}'
            with open(save_name+'.bin', 'rb') as f:
                IDM = pickle.load(f)

            norm = mpl.colors.Normalize(vmin=0, vmax=np.max(IDM))
            im = axes.imshow(IDM.transpose(), cmap='plasma', norm=norm, extent=(x_min, x_max, y_min, y_max), alpha=.6)
            plt.colorbar(im, label='Pixel Diff Count')
            axes.set_title(f'Translation w. template ori: {round(template_orient,2)}')

        elif plot_type == '_transrot_count':
            print(f'plotting IDM (transrot - count) @ {dpi} dpi')

            save_name = fr'{data_dir}/IDM/IDM_transrot_c{space_step}_o{int(np.pi/orient_step)}_tempori{round(template_orient,2)}_vsres{vis_field_res}'
            with open(save_name+'.bin', 'rb') as f:
                IDM = pickle.load(f)

            norm = mpl.colors.Normalize(vmin=0, vmax=np.max(IDM[:,:,0]))
            im = axes.imshow(IDM[:,:,0].transpose(), cmap='plasma', norm=norm, extent=(x_min, x_max, y_min, y_max), alpha=.6)
            plt.colorbar(im, label='Pixel Diff Count')
            axes.set_title(f'Translation + Rotation to Best Match w. template ori: {round(template_orient,2)}')
        
        elif plot_type == '_transrot_ori':
            print(f'plotting IDM (transrot - ori) @ {dpi} dpi')

            save_name = fr'{data_dir}/IDM/IDM_transrot_c{space_step}_o{int(np.pi/orient_step)}_tempori{round(template_orient,2)}_vsres{vis_field_res}'
            with open(save_name+'.bin', 'rb') as f:
                IDM = pickle.load(f)
            
            norm = mpl.colors.Normalize(vmin=0, vmax=2*np.pi)
            im = axes.imshow(IDM[:,:,1].transpose(), cmap='hsv', norm=norm, extent=(x_min, x_max, y_min, y_max), alpha=.6)
            plt.colorbar(im, label='Ori of Closest Fit')
            axes.set_title(f'Translation + Rotation to Best Match w. template ori: {round(template_orient,2)} + Scatter at Perfect Match')

            count = IDM[:,:,0]
            nx,ny = count.shape
            x,y = zip(*[ (x_range[i],height-y_range[j]) for i in range(nx) for j in range(ny) if count[i,j] == 0 ])
            axes.scatter(x,y, edgecolors='none', facecolors='k', s=.5)

        else:
            print('invalid plot type')
            return

        axes.add_patch( plt.Circle((patch_x, height-patch_y), patch_radius, edgecolor='k', fill=False, zorder=1) )
        plt.savefig(fr'{save_name}{plot_type}.png', dpi=dpi)
        plt.close()


# -------------------------- script -------------------------- #

def run_gamut(names):

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    with open(fr'{data_dir}/traj_matrices/gamut.bin', 'rb') as f:
        data = pickle.load(f)

    rank = 'cen'
    space_step = 25
    timesteps = 500
    eye = True
    dpi = 50
    # noise_types = [
    #     (0, 'no_noise'), 
    #     (0.05, 'angle_n05'), 
    #     (0.10, 'angle_n10'),
    #     ]

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

        # traj data
        orient_step = np.pi/8
        save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e{int(eye)}.bin'
        traj_exists = False
        if os.path.exists(save_name_traj):
            print('traj already built')
            traj_exists = True
        else:
            build_agent_trajs_parallel(name, gen, space_step, orient_step, timesteps)

        save_name_trajmap = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e{int(eye)}_ex_lines'
        if os.path.exists(save_name_trajmap+'_100.png'):
            print('traj already plotted at dpi100')
        elif os.path.exists(save_name_trajmap+'_50.png'):
            print('traj already plotted at dpi50')
        else:
            plot_agent_trajs(name, gen, space_step, orient_step, timesteps, ex_lines=True)

        corr_peaks, histo_avg, histo_peaks, dirent = plot_agent_orient_corr(name, gen, space_step, orient_step, timesteps, dpi=dpi)
        # angle_medians = plot_agent_valnoise_dists(name, noise_types, dpi=dpi)

        # action data
        orient_step = np.pi/32
        save_name_act = fr'{data_dir}/action_maps/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_action.bin'
        action_exists = False
        if os.path.exists(save_name_act):
            print('action already built')
            action_exists = True
            with open(save_name_act, 'rb') as f:
                act_matrix = pickle.load(f)
            min_action, max_action = act_matrix.min(), act_matrix.max()
        else:
            min_action, max_action = build_action_matrix(name, gen, space_step, orient_step)

        avglen_mean, avglen_med, avglen_min, avglen_max = plot_action_vecfield(name, gen, space_step, orient_step, plot_type='_avg', colored='len', dpi=dpi)
        mean, med, min, max = plot_action_vecfield(name, gen, space_step, orient_step, plot_type='_avg', colored='ori', dpi=dpi)
        pkf_mean, pkf_med, pkf_min, pkf_max = plot_action_vecfield(name, gen, space_step, orient_step, plot_type='_peaks_fwd', colored='count', ex_lines=True, dpi=dpi)
        pkt_mean, pkt_med, pkt_min, pkt_max = plot_action_vecfield(name, gen, space_step, orient_step, plot_type='_peaks_turn', colored='count', ex_lines=True, dpi=dpi)
        def_mean, def_med, def_min, def_max = plot_action_vecfield(name, gen, space_step, orient_step, plot_type='_dirent_fwd', dpi=dpi)
        det_mean, det_med, det_min, det_max = plot_action_vecfield(name, gen, space_step, orient_step, plot_type='_dirent_turn', dpi=dpi)

        # save in dict + update pickle
        data[name] = (corr_peaks, histo_avg, histo_peaks, dirent, min_action, max_action, avglen_mean, avglen_med, avglen_min, avglen_max, pkf_mean, pkf_med, pkf_min, pkf_max, pkt_mean, pkt_med, pkt_min, pkt_max, def_mean, def_med, def_min, def_max, det_mean, det_med, det_min, det_max)
        print(f'data dict len: {len(data)}')
        print('')

        with open(fr'{data_dir}/traj_matrices/gamut.bin', 'wb') as f:
            pickle.dump(data, f)

        # delete .bin files if not already saved
        if not traj_exists:
            save_name = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e{int(eye)}.bin'
            os.remove(save_name_traj)
        if not action_exists:
            save_name = fr'{data_dir}/action_maps/{name}_{gen}_c{space_step}_o32_action.bin'
            os.remove(save_name_act)


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

    # ## traj
    # space_step = 25
    # orient_step = np.pi/8

    ## action : vec-avg
    space_step = 25
    orient_step = np.pi/32

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
    # for i in [3,15]:
    # # for i in [1,3,4,9,14,15]:
    # # # for i in [2,5,6,7,11,12,13]:
    # # # for i in [1,2,3,4,5,6,7,9,11,12,13,14,15]:
    # # # for i in [0,10,17,18]:
    # for i in [0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,17,18]:
        # names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{str(i)}')
    # # for i in [3,4,9,13]:
    # # # # for i in [0,1,2,5,6,14,16,17,18,19]:
    # for i in [0,1,2,3,4,5,6,9,13,14,16,17,18,19]:
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep{str(i)}')

    # # # # for i in [0,1,2,3,5,7,8,9,10,13,14,16,18]:
    # # for i in [0,2,3,9,13,16,18]:
    # for i in [0,1,2,3,5,6,7,8,9,10,11,13,14,16,18,19]:
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep{str(i)}')

    # for i in [2,3,5]: 
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

    # for i in [1,3,7,10]:
    #     names.append(f'sc_CNN14_FNN8_p50e20_vis8_PGPE_ss20_mom8_gru_rep{i}')
    # for i in [1,10,11,15]:
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_silu_rep{i}')

    # data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    # with open(fr'{data_dir}/traj_matrices/gamut.bin', 'rb') as f:
    #     data = pickle.load(f)

    # for name in names:
    #     gen, valfit = find_top_val_gen(name, 'cen')
    #     print(f'{name} @ {gen} w {valfit} fitness')

    #     # build_agent_trajs_parallel(name, gen, space_step, orient_step, timesteps)
    #     # plot_agent_trajs(name, gen, space_step, orient_step, timesteps, ex_lines=True)
    #     # # plot_agent_trajs(name, gen, space_step, orient_step, timesteps, extra='3d')
    #     # corr_peaks, histo_avg, histo_peaks = plot_agent_orient_corr(name, gen, space_step, orient_step, timesteps, dpi=100)
    #     # # angle_medians = plot_agent_valnoise_dists(name, noise_types, dpi=100)

    #     # plot_binned_orient_turn_gradients(name, gen, space_step, orient_step, timesteps, dpi=100)
    #     # build_action_matrix(name, gen, space_step, orient_step)
    #     # plot_action_map(name, gen, space_step, orient_step, plot_type='', dpi=500)
    #     # plot_action_map(name, gen, space_step, orient_step, plot_type='_inv', dpi=1000)
    #     # plot_action_map(name, gen, space_step, orient_step, plot_type='_grad', dpi=100)
    #     # plot_action_map(name, gen, space_step, orient_step, plot_type='_actvec', dpi=100)
    #     # plot_action_map(name, gen, space_step, orient_step, plot_type='_fwd', dpi=100)
    #     # plot_action_map(name, gen, space_step, orient_step, plot_type='_turn', dpi=100)
    #     # plot_action_map(name, gen, space_step, orient_step, plot_type='_avgori', dpi=100)
    #     # plot_action_map(name, gen, space_step, orient_step, plot_type='_avglen', dpi=100)
    #     # plot_action_vecfield(name, gen, space_step, orient_step, plot_type='_avg', dpi=100)
    #     # plot_action_vecfield(name, gen, space_step, orient_step, plot_type='_avg', colored='ori', dpi=100)
    #     # plot_action_vecfield(name, gen, space_step, orient_step, plot_type='_avg', colored='len', dpi=100)
    #     plot_action_vecfield(name, gen, space_step, orient_step, plot_type='_peaks_fwd', ex_lines=True, dpi=100)
    #     plot_action_vecfield(name, gen, space_step, orient_step, plot_type='_peaks_turn', ex_lines=True, dpi=100)
    #     # plot_action_vecfield(name, gen, space_step, orient_step, plot_type='_dirent_turn', dpi=100)
    #     # plot_action_vecfield(name, gen, space_step, orient_step, plot_type='_dirent_fwd', dpi=100)
    #     # plot_action_vecfield(name, gen, space_step, orient_step, plot_type='_fwdentropy', dpi=100)
    #     # plot_action_vecfield(name, gen, space_step, orient_step, plot_type='_avgact', dpi=100)

    #     # # save in dict + update pickle
    #     # data[name] = (corr_peaks, histo_avg, histo_peaks, angle_medians)
    #     # print(f'data dict len: {len(data)}')
    #     # with open(fr'{data_dir}/traj_matrices/gamut.bin', 'wb') as f:
    #     #     pickle.dump(data, f)

    # name = names[0]
    # gen, valfit = find_top_val_gen(name, 'cen')

    # space_step = 5
    # vfr = 32
    # orient_step = np.pi/32
    # orient_range = np.arange(0, 2*np.pi, np.pi/8)
    # # orient_range = np.arange(0, 2*np.pi, np.pi/2)
    # for to in orient_range:
    #     # build_IDM(name, gen, space_step, orient_step, template_orient=to, vis_field_res=vfr)
    #     # plot_IDM(name, gen, space_step, orient_step, template_orient=to, vis_field_res=vfr, plot_type='_rot', dpi=100)
    #     # plot_IDM(name, gen, space_step, orient_step, template_orient=to, vis_field_res=vfr, plot_type='_trans', dpi=200)
    #     # plot_IDM(name, gen, space_step, orient_step, template_orient=to, vis_field_res=vfr, plot_type='_transrot_count', dpi=200)
    #     plot_IDM(name, gen, space_step, orient_step, template_orient=to, vis_field_res=vfr, plot_type='_transrot_ori', dpi=200)

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




    # data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    # # data = {}
    # # with open(fr'{data_dir}/traj_matrices/gamut.bin', 'wb') as f:
    # #     pickle.dump(data, f)
    # with open(fr'{data_dir}/traj_matrices/gamut.bin', 'rb') as f:
    #     data = pickle.load(f)

    # print(data.keys())
    # for name in list(data.keys()):
    #     if 'CNN12' in name:
    #         print(name)
    #         data.pop(name, None)
    #     if 'p9WF' in name:
    #         print(name)
    #         data.pop(name, None)
    # for name in list(data.keys()):
    #     if 'p8WF' in name:
    #         print(name)
    #         data.pop(name, None)
    # print(data.keys())

    # with open(fr'{data_dir}/traj_matrices/gamut.bin', 'wb') as f:
    #     pickle.dump(data, f)


    names.append('sc_CNN12_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep18')

    for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
        names.append(name)

    for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep{x}' for x in range(20)]:
        names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep{x}' for x in range(20)]:
        names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep{x}' for x in range(20)]:
        names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep{x}' for x in range(20)]:
        names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep{x}' for x in range(20)]:
        names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep{x}' for x in range(20)]:
        names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep{x}' for x in range(20)]:
        names.append(name)

    for name in [f'sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
        names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
        names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
        names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
        names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
        names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
        names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
        names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
        names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
        names.append(name)



    for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep{x}' for x in range(20)]:
        names.append(name)

    for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep{x}' for x in range(20)]:
        names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep{x}' for x in range(20)]:
        names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep{x}' for x in range(20)]:
        names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep{x}' for x in range(20)]:
        names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep{x}' for x in range(20)]:
        names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep{x}' for x in range(20)]:
        names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep{x}' for x in range(20)]:
        names.append(name)

    for name in [f'sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_seed10k_rep{x}' for x in range(20)]:
        names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_seed10k_rep{x}' for x in range(20)]:
        names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed10k_rep{x}' for x in range(20)]:
        names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_seed10k_rep{x}' for x in range(20)]:
        names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_seed10k_rep{x}' for x in range(20)]:
        names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_seed10k_rep{x}' for x in range(20)]:
        names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_seed10k_rep{x}' for x in range(20)]:
        names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed10k_rep{x}' for x in range(20)]:
        names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_seed10k_rep{x}' for x in range(20)]:
        names.append(name)



    run_gamut(names)










    # seeds = [10000,20000]

    # for s in seeds:
    #     for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed{str(int(s/1000))}k_rep{x}' for x in range(20)]:
    #         names.append(name)
    #     for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed{str(int(s/1000))}k_rep{x}' for x in range(20)]:
    #         names.append(name)
    #     for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed{str(int(s/1000))}k_rep{x}' for x in range(20)]:
    #         names.append(name)
    #     for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed{str(int(s/1000))}k_rep{x}' for x in range(20)]:
    #         names.append(name)
    #     for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed{str(int(s/1000))}k_rep{x}' for x in range(20)]:
    #         names.append(name)
    #     for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed{str(int(s/1000))}k_rep{x}' for x in range(20)]:
    #         names.append(name)
    #     for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_seed{str(int(s/1000))}k_rep{x}' for x in range(20)]:
    #         names.append(name)
    #     for name in [f'sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_seed{str(int(s/1000))}k_rep{x}' for x in range(20)]:
    #         names.append(name)
    #     for name in [f'sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_seed{str(int(s/1000))}k_rep{x}' for x in range(20)]:
    #         names.append(name)
    #     for name in [f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed{str(int(s/1000))}k_rep{x}' for x in range(20)]:
    #         names.append(name)
    #     for name in [f'sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_seed{str(int(s/1000))}k_rep{x}' for x in range(20)]:
    #         names.append(name)
    #     for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_seed{str(int(s/1000))}k_rep{x}' for x in range(20)]:
    #         names.append(name)
    #     for name in [f'sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_seed{str(int(s/1000))}k_rep{x}' for x in range(20)]:
    #         names.append(name)
    #     for name in [f'sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed{str(int(s/1000))}k_rep{x}' for x in range(20)]:
    #         names.append(name)
    #     for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_seed{str(int(s/1000))}k_rep{x}' for x in range(20)]:
    #         names.append(name)

