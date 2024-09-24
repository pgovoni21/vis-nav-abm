from abm.start_sim import reconstruct_NN
from abm.sprites.agent import Agent
# from abm.sprites.agent_LM import Agent
from abm.sprites.landmark import Landmark
from abm.monitoring.plot_funcs import plot_map_iterative_traj, plot_map_iterative_traj_3d, beeswarm

import dotenv as de
from pathlib import Path
import numpy as np
import scipy
import multiprocessing as mp
import _pickle as pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
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
    agent.visual_sensing([],[])
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
    h,w = 5,5
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
        plt.colorbar(im, label='Mean Action Vector Orientation', 
                     fraction=0.046, pad=0.04,
                    ticks=np.arange(0, 2*np.pi+0.01, np.pi/2),
                    format=mpl.ticker.FixedFormatter(['$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']),)
        # Q = axes.quiver(X,Y, U,V, M, angles='xy', cmap=cmap, norm=norm) # colored arrows
        # Q = axes.streamplot(X,Y, U,-V, color='k', broken_streamlines=False) # streams
        # axes.set_title(f'Avg: {np.mean(M):.2f}, Median: {np.median(M):.2f}, Min: {np.min(M):.2f}, Max: {np.max(M):.2f}')
        axes.quiver(X,Y, U,V)

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
        plt.colorbar(im, label='Mean Action Vector Magnitude',
                     fraction=0.056, pad=0.04,)
        # axes.set_title(f'Avg: {np.mean(M):.2f}, Median: {np.median(M):.2f}, Min: {np.min(M):.2f}, Max: {np.max(M):.2f}')
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
    # plt.savefig(fr'{save_name}{plot_type}_C{colored}_colorbar.png', dpi=dpi*4)
    plt.show()
    # plt.close()

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
    sim_type = str(envconf["SIM_TYPE"])
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
    if sim_type == "LM":
        ids = ('TL', 'TR', 'BL', 'BR')
        for id, pos in zip(ids, boundary_endpts):
            landmarks.append(Landmark(
                id=id,
                color=(0,0,0),
                radius=int(envconf["RADIUS_LANDMARK"]),
                position=pos,
                window_pad=int(envconf["WINDOW_PAD"]),
            ))
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
    elif sim_type == "walls, 2x pinball":
        radius = 20
        for id, lm_x in enumerate(range(200,400,20)):
            landmarks.append(Landmark(
                id=id,
                color=(0,0,0),
                radius=radius,
                position=np.array([lm_x, 250 + lm_x]),
                window_pad=int(envconf["WINDOW_PAD"])
            ))
        id_last = id
        for id, lm_y in enumerate(range(300,500,20)):
            landmarks.append(Landmark(
                id=id+id_last+1,
                color=(0,0,0),
                radius=radius,
                position=np.array([600, lm_y]),
                window_pad=int(envconf["WINDOW_PAD"])
            ))
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

        if sim_type == 'walls': agent.visual_sensing([],[])
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

        agent.collided_points = []
        if sim_type.startswith('walls'):
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
        if sim_type == 'LM' or sim_type == 'walls, 2x pinball':
            for lm in landmarks:
                if np.linalg.norm(agent.rect.center - (window_pad + lm.position)) < (lm.radius + agent_radius):
                    agent.mode = 'collide'
                    mid_pos = (agent.position + lm.position) / 2
                    agent.collided_points.append(np.array(mid_pos) - window_pad)
                    if t > 0:
                        print(f'collided: {t,x,y, agent.rect.center, lm.position}')

        if t == 0 and len(agent.collided_points) > 0:
            # print(f'collided at zero, {x,y}')
            return traj

        noise = np.random.randn()*act_noise_std
        if extra.endswith('pos') and noise > 0: noise = 0
        elif extra.endswith('neg') and noise < 0: noise = 0
        action = agent.action + noise
        agent.move(action)

        traj[t,:2] = agent.pt_eye
        traj[t,2] = agent.orientation
        traj[t,3] = agent.action * np.pi / 2
    
    return traj


def build_agent_trajs_parallel(exp_name, gen_ext, space_step, orient_step, timesteps, rank='cen', eye=True, extra=''):
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
    boundary_scale = int(envconf["BOUNDARY_SCALE"])
    x_min, x_max = 0, width
    y_min, y_max = 0, height
    boundary_endpts = [
        np.array([ x_min-boundary_scale, y_min-boundary_scale ]),
        np.array([ x_max+boundary_scale, y_min-boundary_scale ]),
        np.array([ x_min-boundary_scale, y_max+boundary_scale ]),
        np.array([ x_max+boundary_scale, y_max+boundary_scale ])
    ]

    # every grid position/direction
    coll_boundary_thickness = int(envconf["RADIUS_AGENT"])*2
    x_range = np.linspace(x_min + coll_boundary_thickness, 
                        x_max - coll_boundary_thickness, 
                        int((width - coll_boundary_thickness*2) / space_step))
    y_range = np.linspace(y_min + coll_boundary_thickness, 
                        y_max - coll_boundary_thickness, 
                        int((height - coll_boundary_thickness*2) / space_step))
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

    empties = []
    for n, output in enumerate(results_list):
        traj_matrix[n,:,:] =  output
        if not output[0,:].any():
            empties.append(n) 
    for n in empties:
        traj_matrix = np.delete(traj_matrix, n, axis=0)

    traj_matrix[:,:,1] = y_max - traj_matrix[:,:,1]

    if extra == '':
        save_name = fr'{data_dir}/traj_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e{int(eye)}.bin'
    else:
        save_name = fr'{data_dir}/traj_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e{int(eye)}_{extra}.bin'
    with open(save_name, 'wb') as f:
        pickle.dump(traj_matrix, f)


def plot_agent_trajs(exp_name, gen_ext, space_step, orient_step, timesteps, rank='cen', ellipses=False, eye=True, ex_lines=False, act_arr=False, extra='', dpi=50):
    print(f'plotting map - {exp_name}, {gen_ext}, ell{ellipses}, ex_lines{ex_lines}, extra{extra}')

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

    if extra.startswith('3d'):
        # plot_map_iterative_traj_3d(traj_plot_data, x_max=width, y_max=height, save_name=save_name, plt_type='scatter', var='cturn')
        plot_map_iterative_traj_3d(traj_plot_data, x_max=width, y_max=height, save_name=save_name, plt_type='scatter', var='str_manif')
        # plot_map_iterative_traj_3d(traj_plot_data, x_max=width, y_max=height, save_name=save_name, plt_type='lines', var='cturn')
        # plot_map_iterative_traj_3d(traj_plot_data, x_max=width, y_max=height, save_name=save_name, plt_type='lines', var='ctime')
        # plot_map_iterative_traj_3d(traj_plot_data, x_max=width, y_max=height, save_name=save_name, plt_type='lines', var='ctime_flat')
        # plot_map_iterative_traj_3d(traj_plot_data, x_max=width, y_max=height, save_name=save_name, plt_type='lines', var='ctime_arrows')
        # plot_map_iterative_traj_3d(traj_plot_data, x_max=width, y_max=height, save_name=save_name, plt_type='lines', var='ctime_arrows_only')
    else:
        sim_type = str(envconf["SIM_TYPE"])
        if sim_type == 'walls':
            if not act_arr:
                plot_map_iterative_traj(traj_plot_data, x_max=width, y_max=height, save_name=save_name, ellipses=ellipses, ex_lines=ex_lines, extra=extra, dpi=dpi)
            else:
                save_name = fr'{data_dir}/action_maps/{exp_name}_{gen_ext}_c{space_step}_o32_action' # hardset at 32
                with open(save_name+'.bin', 'rb') as f:
                    act_matrix = pickle.load(f)
                act_matrix = np.abs(act_matrix)
                act_matrix = (act_matrix - act_matrix.min()) / (act_matrix.max() - act_matrix.min())
                act_matrix = (1-act_matrix)
                plot_map_iterative_traj(traj_plot_data, x_max=width, y_max=height, save_name=save_name, ellipses=ellipses, ex_lines=ex_lines, act_mat=act_matrix, envconf=envconf, extra=extra, dpi=dpi)

        elif sim_type == 'LM':
            lms = (int(envconf["RADIUS_LANDMARK"]),
                   [
                    np.array([ 0, 0 ]),
                    np.array([ width, 0 ]),
                    np.array([ 0, height ]),
                    np.array([ width, height ])
                    ]
            )
            plot_map_iterative_traj(traj_plot_data, x_max=width, y_max=height, save_name=save_name, ellipses=ellipses, ex_lines=ex_lines, extra=extra, landmarks=lms, dpi=dpi)

        elif sim_type == 'walls, 2x pinball':
            radius_obj = 20
            lm_pos = []
            for x in range(200,400,20):
                lm_pos.append(np.array([x, height - (250 + x)]))
            for y in range(300,500,20):
                lm_pos.append(np.array([600, height - y]))
            lms = (radius_obj, lm_pos)
            plot_map_iterative_traj(traj_plot_data, x_max=width, y_max=height, save_name=save_name, ellipses=ellipses, ex_lines=ex_lines, extra=extra, landmarks=lms, dpi=dpi)


def calc_entropy(h):
    h_norm = h / np.sum(h)
    e = -np.sum( h_norm*np.log(h_norm) )
    return e

def calc_dirent(i, j, bin_ori, orient_range):
    if len(bin_ori) > 1:
        print(i,j)
        # est range
        h = np.ones(len(bin_ori))/10000
        e_max = calc_entropy(h) # random
        h[0] = 1
        e_min = calc_entropy(h) # uniform
        # hist + calc
        h = np.histogram(bin_ori, bins=orient_range)[0]
        e = calc_entropy(h + 1/10000)
        d = (e_max - e) / (e_max - e_min)
    else:
        print(i,j,'no data')
        d = 0
    return d


def plot_traj_vecfield(exp_name, gen_ext, space_step, orient_step, timesteps, plot_type='', ex_lines=False, dpi=50):
    print(f'plotting traj vector field - {exp_name}{plot_type} @ {dpi} dpi')

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)

    width, height = tuple(eval(envconf["ENV_SIZE"]))
    x_min, x_max = 0, width
    y_min, y_max = 0, height
    # coll_boundary_thickness = int(envconf["RADIUS_AGENT"])
    # x_range = np.linspace(x_min + coll_boundary_thickness, 
    #                     x_max - coll_boundary_thickness + 1, 
    #                     int((width - coll_boundary_thickness*2) / space_step))
    # y_range = np.linspace(y_min + coll_boundary_thickness, 
    #                     y_max - coll_boundary_thickness + 1, 
    #                     int((height - coll_boundary_thickness*2) / space_step))
    orient_range = np.arange(0, 2*np.pi, orient_step)
    # print(f'orient_range: {np.round(orient_range,2)}')
    # print('')


    fig, axes = plt.subplots() 
    axes.set_xlim(0, x_max)
    axes.set_ylim(0, y_max)
    h,w = 8,8
    l,r,t,b = fig.subplotpars.left, fig.subplotpars.right, fig.subplotpars.top, fig.subplotpars.bottom
    fig.set_size_inches( float(w)/(r-l) , float(h)/(t-b) )

    save_name = fr'{data_dir}/traj_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_cen_e1'
    with open(save_name+'.bin', 'rb') as f:
        ag_data = pickle.load(f)
    # print(ag_data.shape)
    
    # if os.path.exists(fr'{save_name}_hist{plot_type}.png'):
    #     print('plot exists')
    #     return

    delay = 25
    x = ag_data[:,delay:,0].flatten()
    y = ag_data[:,delay:,1].flatten()
    ori = ag_data[:,delay:,2].flatten()
    action = ag_data[:,delay:,3].flatten()
    action = np.abs(action)

    num_bins = 101
    x_bins = np.linspace(0, x_max, num_bins)
    y_bins = np.linspace(0, y_max, num_bins)

    if plot_type == '_count':
        H,_,_ = np.histogram2d(x, y, bins=[x_bins, y_bins])
 
        X,Y = np.meshgrid(x_bins, y_bins)
        axes.pcolormesh(X, Y, H.T, cmap='plasma')

    elif plot_type == '_avgact':
        H_count,_,_ = np.histogram2d(x, y, bins=[x_bins, y_bins])
        H,_,_ = np.histogram2d(x, y, bins=[x_bins, y_bins], weights=action)
        H = np.divide(H, H_count, out=np.zeros_like(H), where=H_count!=0)
 
        X,Y = np.meshgrid(x_bins, y_bins)
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        axes.pcolormesh(X, Y, H.T, cmap='plasma', norm=norm)

    elif plot_type == '_avgori':
        H_count,_,_ = np.histogram2d(x, y, bins=[x_bins, y_bins])
        H_x,_,_ = np.histogram2d(x, y, bins=[x_bins, y_bins], weights=np.cos(ori))
        H_x = np.divide(H_x, H_count, out=np.zeros_like(H_x), where=H_count!=0)
        H_y,_,_ = np.histogram2d(x, y, bins=[x_bins, y_bins], weights=np.sin(ori))
        H_y = np.divide(H_y, H_count, out=np.zeros_like(H_y), where=H_count!=0)
        H = np.arctan2(H_y, H_x)
 
        X,Y = np.meshgrid(x_bins, y_bins)
        norm = mpl.colors.Normalize(vmin=-np.pi, vmax=np.pi)
        axes.pcolormesh(X, Y, H.T, cmap='hsv', norm=norm)

    elif plot_type == '_avglen':
        H_count,_,_ = np.histogram2d(x, y, bins=[x_bins, y_bins])
        H_x,_,_ = np.histogram2d(x, y, bins=[x_bins, y_bins], weights=np.cos(ori))
        H_x = np.divide(H_x, H_count, out=np.zeros_like(H_x), where=H_count!=0)
        H_y,_,_ = np.histogram2d(x, y, bins=[x_bins, y_bins], weights=np.sin(ori))
        H_y = np.divide(H_y, H_count, out=np.zeros_like(H_y), where=H_count!=0)
        H = np.hypot(H_x, H_y)

        X,Y = np.meshgrid(x_bins, y_bins)
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        axes.pcolormesh(X, Y, H.T, cmap='plasma', norm=norm)

    elif plot_type == '_dirent':
        # simplify data
        x = ag_data[::10,delay:,0].flatten()
        y = ag_data[::10,delay:,1].flatten()
        ori = ag_data[::10,delay:,2].flatten()
        num_bins = 51
        x_bins = np.linspace(0, x_max, num_bins)
        y_bins = np.linspace(0, y_max, num_bins)

        # drop into bins + organize
        hitx = np.digitize(x, x_bins)
        hity = np.digitize(y, y_bins)
        hitbins = list(zip(hitx, hity))
        ori_and_bins = list(zip(ori, hitbins))

        h = np.ones(len(orient_range))/10000
        e_max = calc_entropy(h) # random
        h[0] = 1
        e_min = calc_entropy(h) # uniform

        H = np.zeros((num_bins-1, num_bins-1))
        for i in range(num_bins-1):
            for j in range(num_bins-1):
                # print(i,j)
                bin_ori = [ori for (ori,bin) in ori_and_bins if bin == (i,j)]
                if bin_ori:
                    h = np.histogram(bin_ori, bins=orient_range)[0]
                    e = calc_entropy(h + 1/10000)
                    d = (e_max - e) / (e_max - e_min)
                    H[i,j] = d
                # else:
                #     print(i,j,'no data')

        X,Y = np.meshgrid(x_bins, y_bins)
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        axes.pcolormesh(X, Y, H.T, cmap='plasma', norm=norm)

    radius = int(envconf["RADIUS_RESOURCE"])
    x,y = tuple(eval(envconf["RESOURCE_POS"]))
    axes.add_patch( plt.Circle((x, height-y), radius, edgecolor='k', fill=False, zorder=1) )

    plt.savefig(fr'{save_name}_hist{plot_type}.png', dpi=dpi)
    plt.close()

    # mask edges (100 from each edge) + patch vicinity (100 from center)
    mask = np.zeros([num_bins-1, num_bins-1])
    mask[0:10,:] = 1
    mask[:,0:10] = 1
    mask[89:,:] = 1
    mask[:,89:] = 1
    mask[29:50,49:70] = 1
    H_mask = np.ma.array(H, mask=mask)

    print(f'{np.mean(H_mask):.2f}, {np.min(H_mask):.2f}, {np.max(H_mask):.2f}')
    return np.mean(H_mask), np.min(H_mask), np.max(H_mask)


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
    save_name = fr'{data_dir}/traj_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e{int(eye)}'
    if not os.path.exists(save_name+'.bin'):
        print(f'no data found for {save_name}')
        return 0,0,0,0
    with open(save_name+'.bin', 'rb') as f:
        ag_data = pickle.load(f)
    save_name = fr'{data_dir}/corrs/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e{int(eye)}'

    num_runs,t_len,_ = ag_data.shape
    t = np.linspace(0,t_len,t_len)
    # print(f'ag_data shape: {num_runs, len(t)}')

    # corr to init
    delay = 25
    t = t[:-delay]
    t_len -= delay
    orient = ag_data[:,delay:,2]
    orient_0 = orient[:,0]
    orient_0 = np.tile(orient_0,(t_len,1)).transpose()
    corr_init_angle_diff = orient - orient_0
    corr_init_angle_diff_scaled = (corr_init_angle_diff + np.pi) % (2*np.pi) - np.pi # [-pi/2, pi/2]
    corr_init = np.cos(corr_init_angle_diff)
    # print(f'corr_init shape: {corr_init.shape}')

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

    # corr to patch
    angle_to_target = np.arctan2(disp[:,:,1], disp[:,:,0]) + np.pi # shift by pi for [0-2pi]
    # print('max/min angle_to_target: ', np.max(angle_to_target), np.min(angle_to_target))
    corr_patch_angle_diff = angle_to_target.transpose() - orient
    corr_patch_angle_diff_scaled = (corr_patch_angle_diff + np.pi) % (2*np.pi) - np.pi # [-pi/2, pi/2]
    corr_patch = np.cos(corr_patch_angle_diff)
    # print(f'corr_patch shape: {corr_patch.shape}')

    # action
    action = np.abs(ag_data[:,delay:,3])/(np.pi/2) # [0,1]


    ### temporal correlations ###

    fig = plt.figure(figsize=(3,3))
    ax0 = plt.subplot()

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

    decorr_idx = np.argmax(corr_init_avg < 0.5)
    decorr_time = t[decorr_idx]
    decorr_val = corr_init_avg[decorr_idx]

    ax0.vlines(decorr_time, 0, decorr_val, color='red', ls='--')
    # ax0.set_title(f'Decorr Time: {decorr_time:.2f}')

    ax0.set_xlim(-20,520)
    ax0.set_ylim(-1.05,1.05)
    ax0.set_ylim(-0.05,1.05)
    ax0.set_xlabel('Timesteps')
    ax0.set_ylabel('Orientation Correlation')

    plt.tight_layout()
    plt.savefig(fr'{save_name}_corr_orient_{dpi}.png', dpi=dpi)
    # plt.show()
    plt.close()


    ### spatial correlation trajectories - init heading ###

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
        ins.plot(dist[:,index], -corr_init_angle_diff_scaled[index,:], c=color, alpha=.5, linewidth=1)

    # labels = [r'-$2\pi$', r'-$3\pi/2$', r'-$\pi$', r'-$\pi/2$', '$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']
    labels = [r'-$\pi$', r'-$\pi/2$', '$0$', r'$\pi/2$', r'$\pi$']

    ax1.set_xlim(-20,820)
    ax1.set_yticks(np.arange(-np.pi, np.pi+0.01, np.pi/2))
    ax1.set_yticklabels(labels)
    ax1.set_xlabel('Distance to Patch')
    ax1.set_ylabel('Relative Orientation')

    plt.tight_layout()
    plt.savefig(fr'{save_name}_corr_trajs_init_{dpi}.png', dpi=dpi)
    # plt.show()
    plt.close()


    ### spatial correlation trajectories - patch ###

    fig = plt.figure(figsize=(3,3))
    ax1 = plt.subplot()

    for r in range(num_runs)[::50]:
        # ax1.plot(dist[:,r], corr_patch[r,:], c='k', alpha=5/255)
        ax1.plot(dist[:,r], -corr_patch_angle_diff_scaled[r,:], c='k', alpha=5/255)

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
        distance, index_xy = scipy.spatial.KDTree(ag_data[:,0,:2]).query(pt[:2])
        index_ori = (np.abs(ag_data[index_xy:index_xy+16,0,2] - pt[2])).argmin()
        index = index_xy + index_ori
        ins.plot(dist[:,index], -corr_patch_angle_diff_scaled[index,:], c=color, alpha=.5, linewidth=1)

    labels = [r'-$\pi$', r'-$\pi/2$', '$0$', r'$\pi/2$', r'$\pi$']
    ax1.set_xlim(-20,820)
    ax1.set_yticks(np.arange(-np.pi, np.pi+0.01, np.pi/2))
    ax1.set_yticklabels(labels)
    ax1.set_xlabel('Distance to Patch')
    ax1.set_ylabel('Relative Orientation')

    plt.tight_layout()
    plt.savefig(fr'{save_name}_corr_trajs_patch_{dpi}.png', dpi=dpi)
    # plt.show()
    plt.close()


    ### spatial correlation - init heading - polar hist ###

    fig = plt.figure(figsize=(3,3))
    ax2 = plt.subplot(projection='polar')

    dist = dist.flatten()
    m = np.ma.masked_less(dist, 100)

    corr_init_angle_diff = corr_init_angle_diff_scaled.transpose().flatten()
    corr_init_angle_diff_masked = (1-m.mask)*corr_init_angle_diff
    corr_init_angle_diff_comp = corr_init_angle_diff_masked[corr_init_angle_diff_masked != 0]

    # Visualise by area of bins
    n, bins, patches = circular_hist(ax2, corr_init_angle_diff_comp, bins=100, offset=np.pi/2, colored=True)
    # # Visualise by radius of bins
    # circular_hist(ax[1], corr_init_angle_diff_comp, bins=100, offset=np.pi/2, density=False)

    x_avg = np.mean(np.cos(corr_init_angle_diff_comp))
    y_avg = np.mean(np.sin(corr_init_angle_diff_comp))
    histo_avg_init = np.arctan2(y_avg, x_avg)

    ax2.axvline(histo_avg_init, color='gray', ls='--')

    # wrap before peak finding
    nn = np.concatenate((n,n))
    peaks,_ = scipy.signal.find_peaks(nn, prominence=25000)
    peaks_shift = peaks + len(n)
    pp = np.concatenate((peaks,peaks_shift))
    repeats = [item for item in set(pp) if list(pp).count(item) > 1]
    peaks = np.setdiff1d(peaks, repeats)
    histo_peaks_init = len(peaks)

    plt.tight_layout()
    plt.savefig(fr'{save_name}_corr_polar_init_{dpi}.png', dpi=dpi)
    # plt.show()
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

    n += 1 # avoid log(0)
    e = calc_entropy(n)
    dirent_init = (e_max - e) / (e_max - e_min)


    ### spatial correlation - patch - polar hist ###

    fig = plt.figure(figsize=(3,3))
    ax2 = plt.subplot(projection='polar')

    dist = dist.flatten()
    m = np.ma.masked_less(dist, 100)

    corr_patch_angle_diff = corr_patch_angle_diff.transpose().flatten()
    corr_patch_angle_diff_masked = (1-m.mask)*corr_patch_angle_diff
    corr_patch_angle_diff_comp = corr_patch_angle_diff_masked[corr_patch_angle_diff_masked != 0]

    # Visualise by area of bins
    n, bins, patches = circular_hist(ax2, corr_patch_angle_diff_comp, bins=100, offset=np.pi/2, colored=True)
    # # Visualise by radius of bins
    # circular_hist(ax[1], corr_patch_angleh_diff_comp, bins=100, offset=np.pi/2, density=False)

    x_avg = np.mean(np.cos(corr_patch_angle_diff_comp))
    y_avg = np.mean(np.sin(corr_patch_angle_diff_comp))
    histo_avg_patch = np.arctan2(y_avg, x_avg)

    ax2.axvline(histo_avg_patch, color='gray', ls='--')

    # wrap before peak finding
    nn = np.concatenate((n,n))
    peaks,_ = scipy.signal.find_peaks(nn, prominence=25000)
    peaks_shift = peaks + len(n)
    pp = np.concatenate((peaks,peaks_shift))
    repeats = [item for item in set(pp) if list(pp).count(item) > 1]
    peaks = np.setdiff1d(peaks, repeats)
    histo_peaks_patch = len(peaks)

    plt.tight_layout()
    plt.savefig(fr'{save_name}_corr_polar_patch_{dpi}.png', dpi=dpi)
    # plt.show()
    plt.close()


    ### directedness ###
    n += 1 # avoid log(0)
    e = calc_entropy(n)
    dirent_patch = (e_max - e) / (e_max - e_min)


    # ### action correlation ###

    # fig = plt.figure(figsize=(3,3))
    # ax1 = plt.subplot()

    # dist = np.linalg.norm(disp, axis=2)
    # for r in range(num_runs)[0:1]:
    #     corr = -corr_patch_angle_diff_scaled[r,:]
    #     m = np.ma.masked_less(dist[:,r], 100)

    #     corr_masked = (1-m.mask)*corr
    #     corr_comp = corr_masked[corr_masked != 0]
    #     ax1.plot(corr_comp, action[r,:][corr_masked != 0], 'k', alpha=50/255)

    # # dist = dist.flatten()
    # # corr = -corr_patch_angle_diff_scaled.flatten()
    # # m = np.ma.masked_less(dist, 100)
    # # corr_masked = (1-m.mask)*corr
    # # corr_comp = corr_masked[corr_masked != 0]
    # # action_comp = action.flatten()[corr_masked != 0]

    # # x_bins = np.arange(-np.pi, np.pi+0.01, np.pi/8)
    # # y_bins = np.arange(0, 1.01, 1/8)
    # # # H,_,_ = np.histogram2d(corr_patch_angle_diff_comp.flatten(), action.flatten(), bins=[x_bins, y_bins])
    # # H,_,_ = np.histogram2d(corr_comp, action_comp, bins=[x_bins, y_bins])
    # # # ax1.imshow(hist, cmap='plasma', extent=[-np.pi, np.pi, 0, 1], aspect='auto', origin='lower')
    # # # X,Y = np.meshgrid(x_bins, y_bins)
    # # # im = ax1.pcolormesh(X, Y, H.T, cmap='plasma', norm="log")
    # # # ax1.plot(np.linspace(-np.pi, np.pi, H.shape[0]), np.mean(H, axis=1), 'r')
    # # # ax1.plot(x_bins[1:], np.average(H, axis=1, weights=y_bins[1:]), 'r')

    # # # print(x_bins.shape, y_bins.shape, H.shape)
    # # avgs = []
    # # for i,x in enumerate(x_bins[1:]):
    # #     # avg_per_x = np.average(H[i,:], weights=y_bins[1:])
    # #     avg_per_x = np.average(y_bins[1:], weights=H[i,:])
    # #     avgs.append(avg_per_x)

    # # ax1.plot(x_bins[1:], avgs, 'r')
    # # min_avg = np.min(avgs)
    # # min_avg_idx = np.argmin(avgs)
    # # ax1.plot(x_bins[1:][min_avg_idx], min_avg, 'ro')

    # # # mean_per_ori = np.mean(action, axis=0)
    # # # ax1.plot(np.linspace(-np.pi, np.pi, mean_per_ori.shape[0]), mean_per_ori, 'r')

    # labels = [r'-$\pi$', r'-$\pi/2$', '$0$', r'$\pi/2$', r'$\pi$']
    # ax1.set_xticks(np.arange(-np.pi, np.pi+0.01, np.pi/2))
    # ax1.set_xticklabels(labels)
    # ax1.set_ylim(-0.05,1.05)
    # ax1.set_xlabel('Relative Orientation to Patch')
    # ax1.set_ylabel('Action')

    # plt.tight_layout()
    # # plt.savefig(fr'{save_name}_corr_trajs_patch_{dpi}.png', dpi=dpi)
    # plt.show()
    # # plt.close()


    print(f'decorr: {decorr_time:.2f} // num peaks: {corr_peaks}')
    print(f'histo avg init: {-histo_avg_init:.2f} // histo peaks init: {histo_peaks_init}')
    print(f'histo avg patch: {-histo_avg_patch:.2f} // histo peaks patch: {histo_peaks_patch}')
    print(f'dirent init: {dirent_init:.2f} // dirent patch: {dirent_patch:.2f}')

    return corr_peaks, decorr_time, -histo_avg_init, -histo_avg_patch, histo_peaks_init, histo_peaks_patch, dirent_init, dirent_patch


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



# -------------------------- persistent random walk (null model) -------------------------- #


def agent_traj_from_xyo_PRW(envconf, NN, boundary_endpts, x, y, orient, timesteps, behavior, rot_diff, curve=None, limit=None, bias=None):

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
    
    patch = np.array(eval(envconf["RESOURCE_POS"]))

    if behavior == 'ratchet' or behavior == 'ratchet-biased': curve_acc = np.random.uniform(0,limit)

    traj = np.zeros((timesteps,4))
    for t in range(timesteps):

        agent.gather_self_percep_info()

        if behavior == 'straight':
            action = (2*rot_diff)**.5 * np.random.uniform(-1,1)
        elif behavior == 'curve':
            action = (2*rot_diff)**.5 * np.random.uniform(-1,1) + curve
        elif behavior == 'ratchet':
            action = (2*rot_diff)**.5 * np.random.uniform(-1,1) + curve
            curve_acc += curve
            if curve_acc >= limit:
                action -= curve_acc
                curve_acc = 0
        elif behavior == 'straight-biased':
            disp_from_patch = patch - agent.position
            angle_to_patch = np.arctan2(-disp_from_patch[1], disp_from_patch[0])
            angle_diff = angle_to_patch - agent.orientation
            angle_diff = (angle_diff + np.pi) % (2*np.pi) - np.pi
            angle_diff_scaled = angle_diff / np.pi
            action = (2*rot_diff)**.5 * np.random.uniform(-1,1) + angle_diff_scaled*bias
        elif behavior == 'curve-biased':
            disp_from_patch = patch - agent.position
            angle_to_patch = np.arctan2(-disp_from_patch[1], disp_from_patch[0])
            angle_diff = angle_to_patch - agent.orientation
            angle_diff = (angle_diff + np.pi) % (2*np.pi) - np.pi
            angle_diff_scaled = angle_diff / np.pi
            action = (2*rot_diff)**.5 * np.random.uniform(-1,1) + curve + angle_diff_scaled*bias
        elif behavior == 'ratchet-biased':
            disp_from_patch = patch - agent.position
            angle_to_patch = np.arctan2(-disp_from_patch[1], disp_from_patch[0])
            angle_diff = angle_to_patch - agent.orientation
            angle_diff = (angle_diff + np.pi) % (2*np.pi) - np.pi
            angle_diff_scaled = angle_diff / np.pi
        # print(agent.position, patch, angle_to_patch, angle_diff, angle_diff_scaled, action)

        agent.move(action)

        traj[t,:2] = agent.pt_eye
        traj[t,2] = agent.orientation
        traj[t,3] = np.cos(agent.orientation - orient)
    
    return traj


def build_agent_trajs_parallel_PRW(space_step, orient_step, timesteps, behavior, rot_diff, curve=None, limit=None, bias=None):
    print(f'building {behavior} PRW w {rot_diff} rot_diff, @ {space_step}, {int(np.pi/orient_step)}, {timesteps}, {curve}, {limit}')

    # pull pv + envconf from save folders
    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    exp_name = 'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep0'
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)

    rd_str = str(rot_diff).replace(".","p")
    cv_str = str(curve).replace(".","p")
    lm_str = str(round(limit/np.pi, 2)).replace(".","p") if limit is not None else None
    if bias is None:
        save_name = fr'{data_dir}/traj_matrices/PRW_{behavior}_rd{rd_str}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_cv{cv_str}_lm{lm_str}'
    else:
        b_str = str(bias).replace(".","p")
        save_name = fr'{data_dir}/traj_matrices/PRW_{behavior}_rd{rd_str}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_cv{cv_str}_lm{lm_str}_b{b_str}'
    if os.path.exists(save_name+'.bin'):
        print(f'data already exists')
        return

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
    
    # construct matrix of each traj for each grid pos/dir
    num_inits = len(x_range) * len(y_range) * len(orient_range)
    traj_matrix = np.zeros( (num_inits, timesteps, 4) ) # (pos_x, pos_y, _, _) --> to match self.data_agent format
    print(f'traj matrix shape (# initializations, timesteps, ): {traj_matrix.shape}')

    # pack inputs for multiprocessing map
    mp_inputs = []
    for x in x_range:
        for y in y_range:
            for orient in orient_range:
                mp_inputs.append( (envconf, NN, boundary_endpts, x, y, orient, timesteps, behavior, rot_diff, curve, limit, bias) )
    
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

    with open(save_name+'.bin', 'wb') as f:
        pickle.dump(traj_matrix, f)


def plot_agent_trajs_PRW(space_step, orient_step, timesteps, behavior, rot_diff, curve=None, limit=None, bias=None):
    print(f'plotting {behavior} PRW w {rot_diff} rot_diff, @ {space_step}, {int(np.pi/orient_step)}, {timesteps}, {curve}, {limit}')

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    exp_name = 'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep0'
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)

    rd_str = str(rot_diff).replace(".","p")
    cv_str = str(curve).replace(".","p")
    lm_str = str(round(limit/np.pi, 2)).replace(".","p") if limit is not None else None
    if bias is None:
        save_name = fr'{data_dir}/traj_matrices/PRW_{behavior}_rd{rd_str}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_cv{cv_str}_lm{lm_str}'
    else:
        b_str = str(bias).replace(".","p")
        save_name = fr'{data_dir}/traj_matrices/PRW_{behavior}_rd{rd_str}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_cv{cv_str}_lm{lm_str}_b{b_str}'
    with open(save_name+'.bin', 'rb') as f:
        ag_data = pickle.load(f)
    if os.path.exists(save_name+'_50.bin'):
        print(f'traj plot already exists')
        return

    res_data = np.zeros((1,1,3))
    res_radius = int(envconf["RADIUS_RESOURCE"])
    width, height = tuple(eval(envconf["ENV_SIZE"]))
    x,y = tuple(eval(envconf["RESOURCE_POS"]))

    res_data[0,0,:] = np.array((x, height - y, res_radius))
    traj_plot_data = (ag_data, res_data)

    plot_map_iterative_traj(traj_plot_data, x_max=width, y_max=height, save_name=save_name)


def plot_agent_corr_PRW(space_step, orient_step, timesteps, behavior, rot_diff, curve=None, limit=None, bias=None, dpi=100):
    print(f'plotting {behavior} PRW oricorr w {rot_diff} rot_diff, @ {space_step}, {int(np.pi/orient_step)}, {timesteps}, {curve}, {limit}')

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    rd_str = str(rot_diff).replace(".","p")
    cv_str = str(curve).replace(".","p")
    lm_str = str(round(limit/np.pi, 2)).replace(".","p") if limit is not None else None
    if bias is None:
        save_name = fr'{data_dir}/traj_matrices/PRW_{behavior}_rd{rd_str}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_cv{cv_str}_lm{lm_str}'
    else:
        b_str = str(bias).replace(".","p")
        save_name = fr'{data_dir}/traj_matrices/PRW_{behavior}_rd{rd_str}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_cv{cv_str}_lm{lm_str}_b{b_str}'
    with open(save_name+'.bin', 'rb') as f:
        ag_data = pickle.load(f)

    num_runs,t_len,_ = ag_data.shape
    t = np.linspace(0,t_len,t_len+1)
    # print(f'ag_data shape: {num_runs, len(t)}')

    # auto corr calc - delayed orient_0
    delay = 25
    t = t[:-delay]
    t_len -= delay
    orient = ag_data[:,delay:,2]
    orient_0 = orient[:,0]
    orient_0 = np.tile(orient_0,(t_len,1)).transpose()
    corr = np.cos(orient - orient_0)
    corr = np.insert(corr, 0, 1, axis=1)
    # print(f'corr shape: {corr.shape}')

    fig = plt.figure(figsize=(3,3))
    ax = plt.subplot()

    corr_init_avg = np.mean(corr[:,:], 0)
    ax.plot(t, corr_init_avg, 'k')
    ax.axhline(color='gray', ls='--')

    decorr_idx = np.argmax(corr_init_avg < 0.5)
    decorr_time = t[decorr_idx]
    decorr_val = corr_init_avg[decorr_idx]

    if decorr_time == 0:
        ax.set_title(f'Decorr Time: None')
    else:
        ax.vlines(decorr_time, 0, decorr_val, color='red', ls='--')
        ax.set_title(f'Decorr Time: {decorr_time:.2f}')

    ax.set_xlim(-20,520)
    ax.set_ylim(-1.05,1.05)
    # ax.set_ylim(-0.05,1.05)
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Orientation Correlation')

    plt.tight_layout()
    plt.savefig(fr'{save_name}_corr_auto_delayed.png', dpi=dpi)
    # plt.show()
    plt.close()


def plot_agent_dirent_PRW(space_step, orient_step, timesteps, behavior, rot_diff, curve=None, limit=None, bias=None, dpi=100):
    print(f'plotting {behavior} PRW dirent w {rot_diff} rot_diff, @ {space_step}, {int(np.pi/orient_step)}, {timesteps}, {curve}, {limit}')

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    rd_str = str(rot_diff).replace(".","p")
    cv_str = str(curve).replace(".","p")
    lm_str = str(round(limit/np.pi, 2)).replace(".","p") if limit is not None else None
    if bias is None:
        save_name = fr'{data_dir}/traj_matrices/PRW_{behavior}_rd{rd_str}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_cv{cv_str}_lm{lm_str}'
    else:
        b_str = str(bias).replace(".","p")
        save_name = fr'{data_dir}/traj_matrices/PRW_{behavior}_rd{rd_str}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_cv{cv_str}_lm{lm_str}_b{b_str}'
    with open(save_name+'.bin', 'rb') as f:
        ag_data = pickle.load(f)
    if os.path.exists(save_name+'_hist_dirent.bin'):
        print(f'dirent plot already exists')
        return
    

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    exp_name = 'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep0'
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)

    width, height = tuple(eval(envconf["ENV_SIZE"]))
    x_min, x_max = 0, width
    y_min, y_max = 0, height
    orient_range = np.arange(0, 2*np.pi, orient_step)


    fig, ax = plt.subplots() 
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    h,w = 8,8
    l,r,t,b = fig.subplotpars.left, fig.subplotpars.right, fig.subplotpars.top, fig.subplotpars.bottom
    fig.set_size_inches( float(w)/(r-l) , float(h)/(t-b) )

    # simplify data
    delay = 0
    x = ag_data[::10,delay:,0].flatten()
    y = ag_data[::10,delay:,1].flatten()
    ori = ag_data[::10,delay:,2].flatten()
    num_bins = 51
    x_bins = np.linspace(0, x_max, num_bins)
    y_bins = np.linspace(0, y_max, num_bins)

    # drop into bins + organize
    hitx = np.digitize(x, x_bins)
    hity = np.digitize(y, y_bins)
    hitbins = list(zip(hitx, hity))
    ori_and_bins = list(zip(ori, hitbins))

    h = np.ones(len(orient_range))/10000
    e_max = calc_entropy(h) # random
    h[0] = 1
    e_min = calc_entropy(h) # uniform

    H = np.zeros((num_bins-1, num_bins-1))
    for i in range(num_bins-1):
        for j in range(num_bins-1):
            # print(i,j)
            bin_ori = [ori for (ori,bin) in ori_and_bins if bin == (i,j)]
            if bin_ori:
                h = np.histogram(bin_ori, bins=orient_range)[0]
                e = calc_entropy(h + 1/10000)
                d = (e_max - e) / (e_max - e_min)
                H[i,j] = d
            # else:
            #     print(i,j,'no data')

    X,Y = np.meshgrid(x_bins, y_bins)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    ax.pcolormesh(X, Y, H.T, cmap='plasma', norm=norm)

    # mask edges (100 from each edge) + patch vicinity (100 from center)
    mask = np.zeros([num_bins-1, num_bins-1])
    mask[0:10,:] = 1
    mask[:,0:10] = 1
    mask[89:,:] = 1
    mask[:,89:] = 1
    mask[29:50,49:70] = 1
    H_mask = np.ma.array(H, mask=mask)
    ax.set_title(f'Avg Ent. Directedness: {np.mean(H_mask):.2f}')

    radius = int(envconf["RADIUS_RESOURCE"])
    x,y = tuple(eval(envconf["RESOURCE_POS"]))
    ax.add_patch( plt.Circle((x, height-y), radius, edgecolor='k', fill=False, zorder=1) )

    plt.savefig(fr'{save_name}_hist_dirent.png', dpi=dpi)
    plt.close()



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

    agent.visual_sensing([],[])

    return agent.vis_field


def build_IDM_ori(exp_name, gen_ext, space_step, orient_step, template_orient=0, vis_field_res=32):

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



def plot_IDM_ori(space_step, orient_step, template_orient=0, vis_field_res=32, plot_type='', dpi=50):

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    exp_name = 'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep4'
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
        
        elif plot_type == '_transrot_ori_perf':
            print(f'plotting IDM (transrot - ori - perfect matches only) @ {dpi} dpi')

            save_name = fr'{data_dir}/IDM/IDM_transrot_c{space_step}_o{int(np.pi/orient_step)}_tempori{round(template_orient,2)}_vsres{vis_field_res}'
            with open(save_name+'.bin', 'rb') as f:
                IDM = pickle.load(f)

            norm = mpl.colors.Normalize(vmin=0, vmax=2*np.pi)

            count = IDM[:,:,0]
            ori_mat = IDM[:,:,1]
            nx,ny = count.shape
            x,y,ori = zip(*[ (x_range[i], height-y_range[j], ori_mat[i,j]) for i in range(nx) for j in range(ny) if count[i,j] == 0 ])
            # return
            sc = axes.scatter(x,y, c=ori, cmap='hsv', norm=norm, s=9)
            plt.colorbar(sc, label='Ori of Closest Fit')
            axes.set_title(f'Translation + Rotation to Best Match w. template ori: {round(template_orient,2)} + Scatter at Perfect Match')

        else:
            print('invalid plot type')
            return

        axes.add_patch( plt.Circle((patch_x, height-patch_y), patch_radius, edgecolor='k', fill=False, zorder=1) )
        plt.savefig(fr'{save_name}{plot_type}.png', dpi=dpi)
        plt.close()


def build_agent_views(vis_field_res = 8):

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    exp_name = 'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep4'
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)

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
    window_pad = int(envconf["WINDOW_PAD"])
    agent_radius = int(envconf["RADIUS_AGENT"])

    coll_boundary_thickness = int(envconf["RADIUS_AGENT"])
    x_range = np.linspace(x_min + coll_boundary_thickness, 
                        x_max - coll_boundary_thickness + 1, 
                        int((width - coll_boundary_thickness*2) / space_step))
    y_range = np.linspace(y_min + coll_boundary_thickness, 
                        y_max - coll_boundary_thickness + 1, 
                        int((height - coll_boundary_thickness*2) / space_step))
    orient_range = np.arange(0, 2*np.pi, orient_step)

    views = []
    for x in x_range:
        for y in y_range:
            for ori in orient_range:

                agent = Agent(
                        id=0,
                        position=(x,y),
                        orientation=ori,
                        max_vel=int(envconf["MAXIMUM_VELOCITY"]),
                        FOV=float(envconf['AGENT_FOV']),
                        vis_field_res=vis_field_res,
                        vision_range=int(envconf["VISION_RANGE"]),
                        num_class_elements=4,
                        consumption=1,
                        model=None,
                        boundary_endpts=boundary_endpts,
                        window_pad=window_pad,
                        radius=agent_radius,
                        color=(0,0,0),
                        vis_transform='',
                        percep_angle_noise_std=0,
                    )
                agent.visual_sensing([],[])

                if agent.vis_field not in views:
                    views.append(agent.vis_field)

    views.sort()

    # for v in views:
    #     print(v)
    # print(len(views))

    with open(fr'{data_dir}/views_vfr{vis_field_res}.bin', 'wb') as f:
        pickle.dump(sorted(views), f)

    # with open(fr'{data_dir}/views.bin', 'rb') as f:
    #     views = pickle.load(f)
    
    # print(len(views))
    # for i,v in enumerate(views):
    #     print(i,v)

    return views


def string_one_hot(view):
    onehot = ''
    for x in view:
        if x == 'wall_north': onehot += '0'
        elif x == 'wall_south': onehot += '1'
        elif x == 'wall_east': onehot += '2'
        elif x == 'wall_west': onehot += '3'
        else: print('invalid view')
    return onehot

def build_IDM_view(exp_name, gen_ext, space_step, orient_step, view, vis_field_res=32):

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

    # trans across grid + closest rot view
    IDM = np.zeros((len(x_range), len(y_range), 2))
    print(f'img diff matrix (transrot) shape (x, y, [count/orient]): {IDM.shape}')

    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            counts = []
            for k, orient in enumerate(orient_range):
                trans_pano = agent_pano_from_xyo(envconf, NN, boundary_endpts, x, y, orient, vis_field_res)
                diffcount = sum(1 for i, j in zip(view, trans_pano) if i != j)
                counts.append(diffcount)
            counts = np.array(counts)
            IDM[i,j,0] = np.min(counts)
            IDM[i,j,1] = orient_range[np.argmin(counts)]

    view_onehot = string_one_hot(view)
    with open(fr'{data_dir}/IDM/IDM_transrot_c{space_step}_o{int(np.pi/orient_step)}_vsres{vis_field_res}_view{view_onehot}.bin', 'wb') as f:
        pickle.dump(IDM, f)

    return view_onehot


def plot_IDM_view(space_step, orient_step, view_onehot, vis_field_res=32, plot_type='', dpi=50):

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    exp_name = 'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep4'
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


    fig, axes = plt.subplots() 
    axes.set_xlim(0, x_max)
    axes.set_ylim(0, y_max)
    h,w = 8,8
    l,r,t,b = fig.subplotpars.left, fig.subplotpars.right, fig.subplotpars.top, fig.subplotpars.bottom
    fig.set_size_inches( float(w)/(r-l) , float(h)/(t-b) )

    save_name = fr'{data_dir}/IDM/IDM_transrot_c{space_step}_o{int(np.pi/orient_step)}_vsres{vis_field_res}_view{view_onehot}'
    with open(save_name+'.bin', 'rb') as f:
        IDM = pickle.load(f)

    count_mat = IDM[:,:,0]
    ori_mat = IDM[:,:,1]

    if plot_type == '_transrot_count':
        print(f'plotting IDM (transrot - count) @ {dpi} dpi')

        norm = mpl.colors.Normalize(vmin=0, vmax=np.max(IDM[:,:,0]))
        im = axes.imshow(count_mat.transpose(), cmap='plasma', norm=norm, extent=(x_min, x_max, y_min, y_max), alpha=.6)
        plt.colorbar(im, label='Pixel Diff Count')
        # axes.set_title(f'Translation + Rotation to Best Match w. template ori: {round(template_orient,2)}')
    
    elif plot_type == '_transrot_ori':
        print(f'plotting IDM (transrot - ori) @ {dpi} dpi')
        
        norm = mpl.colors.Normalize(vmin=0, vmax=2*np.pi)
        im = axes.imshow(ori_mat.transpose(), cmap='hsv', norm=norm, extent=(x_min, x_max, y_min, y_max), alpha=.6)
        plt.colorbar(im, label='Ori of Closest Fit')
        # axes.set_title(f'Translation + Rotation to Best Match w. template ori: {round(template_orient,2)} + Scatter at Perfect Match')

        nx,ny = count_mat.shape
        x,y = zip(*[ (x_range[i],height-y_range[j]) for i in range(nx) for j in range(ny) if count_mat[i,j] == 0 ])
        axes.scatter(x,y, edgecolors='none', facecolors='k', s=.5)
    
    elif plot_type == '_transrot_ori_perf':
        print(f'plotting IDM (transrot - ori - perfect matches only) @ {dpi} dpi')

        norm = mpl.colors.Normalize(vmin=0, vmax=2*np.pi)
        # im = axes.imshow(IDM[:,:,1].transpose(), cmap='hsv', norm=norm, extent=(x_min, x_max, y_min, y_max), alpha=.6)
        # plt.colorbar(im, label='Ori of Closest Fit')
        # axes.set_title(f'Translation + Rotation to Best Match w. template ori: {round(template_orient,2)} + Scatter at Perfect Match')

        nx,ny = count_mat.shape
        x,y,ori = zip(*[ (x_range[i], height-y_range[j], ori_mat[i,j]) for i in range(nx) for j in range(ny) if count_mat[i,j] == 0 ])
        axes.scatter(x,y, c=ori, cmap='hsv', norm=norm, s=9)

        print(f'average x/y/ori: {np.mean(x), np.mean(y), np.arctan2(np.mean(np.sin(ori)), np.mean(np.cos(ori)))}')
        # axes.scatter(np.mean(x),np.mean(y), c='k', s=20)
        avgori = np.arctan2(np.mean(np.sin(ori)), np.mean(np.cos(ori)))
        axes.quiver(np.mean(x), np.mean(y), np.cos(avgori), np.sin(avgori))

    else:
        print('invalid plot type')
        return

    axes.add_patch( plt.Circle((patch_x, height-patch_y), patch_radius, edgecolor='k', fill=False, zorder=1) )
    plt.savefig(fr'{save_name}{plot_type}.png', dpi=dpi)
    plt.close()


def agent_action_from_view(envconf, NN, view):

    agent = Agent(
            id=0,
            position=(0,0),
            orientation=0,
            max_vel=int(envconf["MAXIMUM_VELOCITY"]),
            FOV=float(envconf['AGENT_FOV']),
            vis_field_res=int(envconf["VISUAL_FIELD_RESOLUTION"]),
            vision_range=int(envconf["VISION_RANGE"]),
            num_class_elements=4,
            consumption=1,
            model=NN,
            boundary_endpts=(None,None,None,None),
            window_pad=30,
            radius=int(envconf["RADIUS_AGENT"]),
            color=(0,0,0),
            vis_transform='',
            percep_angle_noise_std=0,
        )
    vis_input = agent.encode_one_hot(view)
    agent.action, agent.hidden = agent.model.forward(vis_input, np.array([0]), agent.hidden)
    
    return agent.action

def plot_IDM_avgperfviews(space_step, orient_step, vis_field_res=32, plot_type='', dpi=50):

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    exp_name = 'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep4'
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)

    width, height = tuple(eval(envconf["ENV_SIZE"]))
    x_min, x_max = 0, width
    y_min, y_max = 0, height
    coll_boundary_thickness = int(envconf["RADIUS_AGENT"])
    x_range = np.linspace(x_min + coll_boundary_thickness, 
                        x_max - coll_boundary_thickness + 1, 
                        int((width - coll_boundary_thickness*2) / space_step+1))
    y_range = np.linspace(y_min + coll_boundary_thickness, 
                        y_max - coll_boundary_thickness + 1, 
                        int((height - coll_boundary_thickness*2) / space_step+1))

    patch_radius = int(envconf["RADIUS_RESOURCE"])
    patch_x, patch_y = tuple(eval(envconf["RESOURCE_POS"]))

    with open(fr'{data_dir}/IDM/views_vfr{vis_field_res}.bin', 'rb') as f:
        views = pickle.load(f)

    x_all = np.array([])
    y_all = np.array([])
    ori_all = np.array([])
    avgxyori_per_view = np.zeros((len(views),4))
    for i,v in enumerate(views):
        view_onehot = string_one_hot(v)
        # print(v, view_onehot)

        with open(fr'{data_dir}/IDM/IDM_transrot_c{space_step}_o{int(np.pi/orient_step)}_vsres{vis_field_res}_view{view_onehot}.bin', 'rb') as f:
            IDM = pickle.load(f)

        count_mat = IDM[:,:,0]
        ori_mat = IDM[:,:,1]
        nx,ny = count_mat.shape
        x,y,ori = zip(*[ (x_range[i], height-y_range[j], ori_mat[i,j]) for i in range(nx) for j in range(ny) if count_mat[i,j] == 0 ])
        avgori = np.arctan2(np.mean(np.sin(ori)), np.mean(np.cos(ori)))
        count = len(x)
        avgxyori_per_view[i] = np.array([np.mean(x), np.mean(y), avgori, count])
        x_all = np.append(x_all,x)
        y_all = np.append(y_all,y)
        ori_all = np.append(ori_all,ori)
        # print(f'average x/y/ori: {np.mean(x), np.mean(y), avgori, count}')

    fig, axes = plt.subplots() 
    axes.set_xlim(0, x_max)
    axes.set_ylim(0, y_max)
    h,w = 4,4
    l,r,t,b = fig.subplotpars.left, fig.subplotpars.right, fig.subplotpars.top, fig.subplotpars.bottom
    fig.set_size_inches( float(w)/(r-l) , float(h)/(t-b) )

    if plot_type == '_count':
        X,Y,O,C = avgxyori_per_view.transpose()
        axes.scatter(X,Y, c=C, cmap='plasma', s=25)
        axes.quiver(X,Y, np.cos(O), np.sin(O), C, cmap='plasma', scale=50)

    elif plot_type == '_ori':
        X,Y,O,C = avgxyori_per_view.transpose()
        norm = mpl.colors.Normalize(vmin=-np.pi, vmax=np.pi)
        axes.scatter(X,Y, c=O, cmap='hsv', norm=norm, s=25)
        axes.quiver(X,Y, np.cos(O), np.sin(O), O, cmap='hsv', norm=norm, scale=50)

    elif isinstance(plot_type,tuple):
        exp_name, gen_ext = plot_type
        plot_type = f'_{exp_name}_{gen_ext}'

        with open(fr'{data_dir}/{exp_name}/{gen_ext}_NNcen_pickle.bin','rb') as f:
            pv = pickle.load(f)
        envconf = de.dotenv_values(fr'{data_dir}/{exp_name}/.env')
        NN, arch = reconstruct_NN(envconf, pv)

        acts = np.array([agent_action_from_view(envconf, NN, v) for v in views])
        X,Y,O,C = avgxyori_per_view.transpose()

        # abs val
        acts = np.abs(acts)
        norm = mpl.colors.Normalize(vmin=0, vmax=.25)
        axes.scatter(X,Y, c=acts, cmap='plasma', norm=norm, s=25)
        axes.quiver(X,Y, np.cos(O), np.sin(O), acts, cmap='plasma', norm=norm, scale=50)

        # # norm = mpl.colors.CenteredNorm(vcenter=0) #--> centered @ zero
        # norm = mpl.colors.CenteredNorm(vcenter=0, halfrange=.0001) #--> centered @ zero + clipped
        # axes.scatter(X,Y, c=acts, cmap='coolwarm', norm=norm, s=25)
        # axes.quiver(X,Y, np.cos(O), np.sin(O), acts, cmap='coolwarm', norm=norm, scale=50)

    elif plot_type == '_heatmap_count':
        scale = 1
        x_bins = np.linspace(x_min + coll_boundary_thickness, 
                            x_max - coll_boundary_thickness + 1, 
                            int(scale*(width - coll_boundary_thickness*2) / space_step+1))
        y_bins = np.linspace(y_min + coll_boundary_thickness, 
                            y_max - coll_boundary_thickness + 1, 
                            int(scale*(height - coll_boundary_thickness*2) / space_step+1))
        H,_,_ = np.histogram2d(x_all,y_all, bins=[x_bins, y_bins])
        X,Y = np.meshgrid(x_bins, y_bins)
        im = axes.pcolormesh(X, Y, H.T, cmap='plasma')
        # axes.pcolormesh(X, Y, H.T, cmap='plasma', norm=mpl.colors.Normalize(vmin=110)) # for scale=0.5
        # print(np.max(H), np.min(H))
        # print(np.sort(H.flatten())[2500:3500])
        # from mpl_toolkits.axes_grid1 import make_axes_locatable
        # divider = make_axes_locatable(axes)
        # cax1 = divider.append_axes("right", size="5%", pad=0.1)
        # fig.colorbar(im, cax=cax1, label='Number Unique Views / Bin')

    elif plot_type == '_heatmap_ori':
        num_bins = 100
        x_bins = np.linspace(0, x_max, num_bins)
        y_bins = np.linspace(0, y_max, num_bins)
        H_sin,_,_ = np.histogram2d(x_all,y_all, bins=[x_bins, y_bins], weights=np.sin(ori_all))
        H_cos,_,_ = np.histogram2d(x_all,y_all, bins=[x_bins, y_bins], weights=np.cos(ori_all))
        H = np.arctan2(H_sin, H_cos)
        X,Y = np.meshgrid(x_bins, y_bins)
        axes.pcolormesh(X, Y, H.T, cmap='hsv')
        print(np.max(H.T), np.min(H.T))

    else:
        print('invalid plot type')
        return

    axes.add_patch( plt.Circle((patch_x, height-patch_y), patch_radius, edgecolor='k', fill=False, zorder=1) )
    plt.savefig(fr'{data_dir}/IDM/views_vfr{vis_field_res}_avgstats{plot_type}.png', dpi=dpi)
    # plt.savefig(fr'{data_dir}/IDM/views_vfr{vis_field_res}_avgstats{plot_type}_cen.png', dpi=dpi)
    # plt.savefig(fr'{data_dir}/IDM/views_vfr{vis_field_res}_avgstats{plot_type}_cen_clip.png', dpi=dpi)
    plt.close()
    # plt.show()



# -------------------------- script -------------------------- #

def run_gamut(group_name, names, dpi):

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    with open(fr'{data_dir}/traj_matrices/{group_name}.bin', 'rb') as f:
        data = pickle.load(f)

    rank = 'cen'
    space_step = 25
    timesteps = 500
    eye = True

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
        elif os.path.exists(save_name_trajmap+'.png'):
            print('traj already plotted')
        else:
            plot_agent_trajs(name, gen, space_step, orient_step, timesteps, ex_lines=True, dpi=dpi)

        #if name not in data:
        act_mean, act_min, act_max = plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type='_avgact', dpi=dpi)
        _, _, _ = plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type='_avgori', dpi=dpi)
        len_mean, len_min, len_max = plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type='_avglen', dpi=dpi)
        de_mean, de_min, de_max = plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type='_dirent', dpi=dpi)

        corr_peaks, decorr_time, histo_avg_init, histo_avg_patch, histo_peaks_init, histo_peaks_patch, dirent_init, dirent_patch = plot_agent_orient_corr(name, gen, space_step, orient_step, timesteps, dpi=dpi)
        # angle_medians = plot_agent_valnoise_dists(name, noise_types, dpi=dpi)

        # # action data
        # orient_step = np.pi/32
        # save_name_act = fr'{data_dir}/action_maps/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_action.bin'
        # action_exists = False
        # if os.path.exists(save_name_act):
        #     print('action already built')
        #     action_exists = True
        #     with open(save_name_act, 'rb') as f:
        #         act_matrix = pickle.load(f)
        #     min_action, max_action = act_matrix.min(), act_matrix.max()

        # if name in data:
        #     print('already there + added decorr time')
        #     corr_peaks, histo_avg_init, histo_avg_patch, histo_peaks_init, histo_peaks_patch, dirent_init, dirent_patch, min_action, max_action, avglen_mean, avglen_med, avglen_min, avglen_max, pkf_mean, pkf_med, pkf_min, pkf_max, pkt_mean, pkt_med, pkt_min, pkt_max, def_mean, def_med, def_min, def_max, det_mean, det_med, det_min, det_max, act_mean, act_min, act_max, len_mean, len_min, len_max, de_mean, de_min, de_max = data[name]

        # else:
        #     print('building action map')
        #     min_action, max_action = build_action_matrix(name, gen, space_step, orient_step)

        #     avglen_mean, avglen_med, avglen_min, avglen_max = plot_action_vecfield(name, gen, space_step, orient_step, plot_type='_avg', colored='len', dpi=dpi)
        #     mean, med, min, max = plot_action_vecfield(name, gen, space_step, orient_step, plot_type='_avg', colored='ori', dpi=dpi)
        #     pkf_mean, pkf_med, pkf_min, pkf_max = plot_action_vecfield(name, gen, space_step, orient_step, plot_type='_peaks_fwd', colored='count', ex_lines=True, dpi=dpi)
        #     pkt_mean, pkt_med, pkt_min, pkt_max = plot_action_vecfield(name, gen, space_step, orient_step, plot_type='_peaks_turn', colored='count', ex_lines=True, dpi=dpi)
        #     def_mean, def_med, def_min, def_max = plot_action_vecfield(name, gen, space_step, orient_step, plot_type='_dirent_fwd', dpi=dpi)
        #     det_mean, det_med, det_min, det_max = plot_action_vecfield(name, gen, space_step, orient_step, plot_type='_dirent_turn', dpi=dpi)

        #     if name in data_old:
        #         act_mean, act_min, act_max, len_mean, len_min, len_max, de_mean, de_min, de_max = data_old[name]
        #     else:
        #         pass

        # data[name] = (corr_peaks, decorr_time, histo_avg_init, histo_avg_patch, histo_peaks_init, histo_peaks_patch, dirent_init, dirent_patch, min_action, max_action, avglen_mean, avglen_med, avglen_min, avglen_max, pkf_mean, pkf_med, pkf_min, pkf_max, pkt_mean, pkt_med, pkt_min, pkt_max, def_mean, def_med, def_min, def_max, det_mean, det_med, det_min, det_max, act_mean, act_min, act_max, len_mean, len_min, len_max, de_mean, de_min, de_max)
        data[name] = (corr_peaks, decorr_time, histo_avg_init, histo_avg_patch, histo_peaks_init, histo_peaks_patch, dirent_init, dirent_patch, act_mean, act_min, act_max, len_mean, len_min, len_max, de_mean, de_min, de_max)
        print(f'data dict len: {len(data)}')
        print('')

        with open(fr'{data_dir}/traj_matrices/{group_name}.bin', 'wb') as f:
            pickle.dump(data, f)

        # delete .bin files if not already saved
        if not traj_exists:
            save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o8_t{timesteps}_{rank}_e{int(eye)}.bin'
            if os.path.exists(save_name_traj):
                os.remove(save_name_traj)
        # if not action_exists:
        #     save_name_act = fr'{data_dir}/action_maps/{name}_{gen}_c{space_step}_o32_action.bin'
        #     if os.path.exists(save_name_act):
        #         os.remove(save_name_act)





def analyze_gamut(group_type, data_type):

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    with open(fr'{data_dir}/traj_matrices/gamut_labeled.bin', 'rb') as f:
        data = pickle.load(f)
    print(f'data dict len: {len(data)}')

    vis6 = []
    vis8 = []
    vis10 = []
    vis12 = []
    vis14 = []
    vis16 = []
    vis18 = []
    vis20 = []
    vis24 = []
    vis32 = []
    maxWF = []
    p8WF = []
    p6WF = []
    p5WF = []
    p4WF = []
    p3WF = []
    p2WF = []
    p1WF = []

    for name in data.keys():

        with open(fr'{data_dir}/{name}/val_matrix_cen.bin','rb') as f:
            val_matrix = pickle.load(f)
        fitness = np.mean(val_matrix)

        data_tuple, label = data[name]

        if 'vis6' in name:
            vis6.append((name, data_tuple, fitness, label))
        elif 'vis8' in name and 'dist' not in name and 'CNN12' not in name:
            vis8.append((name, data_tuple, fitness, label))
        elif 'vis10' in name:
            vis10.append((name, data_tuple, fitness, label))
        elif 'vis12' in name:
            vis12.append((name, data_tuple, fitness, label))
        elif 'vis14' in name:
            vis14.append((name, data_tuple, fitness, label))
        elif 'vis16' in name:
            vis16.append((name, data_tuple, fitness, label))
        elif 'vis18' in name:
            vis18.append((name, data_tuple, fitness, label))
        elif 'vis20' in name:
            vis20.append((name, data_tuple, fitness, label))
        elif 'vis24' in name:
            vis24.append((name, data_tuple, fitness, label))
        elif 'vis32' in name:
            vis32.append((name, data_tuple, fitness, label))
        elif 'maxWF' in name:
            maxWF.append((name, data_tuple, fitness, label))
        elif 'p9WF' in name:
            p8WF.append((name, data_tuple, fitness, label))
        elif 'p8WF' in name:
            p6WF.append((name, data_tuple, fitness, label))
        elif 'mlWF' in name:
            p5WF.append((name, data_tuple, fitness, label))
        elif 'mWF' in name:
            p4WF.append((name, data_tuple, fitness, label))
        elif 'msWF' in name:
            p3WF.append((name, data_tuple, fitness, label))
        elif '_sWF' in name:
            p2WF.append((name, data_tuple, fitness, label))
        elif 'ssWF' in name:
            p1WF.append((name, data_tuple, fitness, label))
        else: print(f'{name}, not included')

    if group_type == 'vis':
        print('varying vis res')
        group_list = [vis6, vis8, vis10, vis12, vis14, vis16, vis18, vis20, vis24, vis32]
        group_list_str = ['6', '8', '10', '12', '14', '16', '18', '20', '24', '32']

    elif group_type == 'dist':
        print('varying dist scaling')
        # group_list = [maxWF, p8WF, p6WF, p5WF, p4WF, p3WF, p2WF, p1WF, vis8]
        # group_list_str = ['1', '0.8', '0.6', '0.5', '0.4', '0.3', '0.2', '0.1', '0']
        group_list = [vis8, p1WF, p2WF, p3WF, p4WF, p5WF, p6WF, p8WF, maxWF]
        group_list_str = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.8', '1']

    # data_list_str = ['corr_peaks', 'decorr_time', 'histo_avg_init', 'histo_avg_patch', 'histo_peaks_init', 'histo_peaks_patch', 'dirent_init', 'dirent_patch', 'min_action', 'max_action', 'avglen_mean', 'avglen_med', 'avglen_min', 'avglen_max', 'pkf_mean', 'pkf_med', 'pkf_min', 'pkf_max', 'pkt_mean', 'pkt_med', 'pkt_min', 'pkt_max', 'def_mean', 'def_med', 'def_min', 'def_max', 'det_mean', 'det_med', 'det_min', 'det_max', 'act_mean', 'act_min', 'act_max', 'len_mean', 'len_min', 'len_max', 'de_mean', 'de_min', 'de_max']
    # print(f'listing data_type: {data_list_str[data_type]}')
    # for string, data in zip(group_list_str, group_list):
    #     print(f'{string}: {np.mean([item[data_type] for item in data]).round(2)}')

    # ### plot measurement distributions ###
    # fig, ax1 = plt.subplots(figsize=(6,4)) 
    # cmap = plt.get_cmap('plasma')
    # num_groups = len(group_list)
    # # violin_labs = []
    # # import matplotlib.patches as mpatches

    # for g_num, (group_name, group_data) in enumerate(zip(group_list_str, group_list)):
    #     data = np.array([data[data_type] for data in group_data])
    #     # print(f'{group_name}: {round(np.mean(data),2)}')

    #     l0 = ax1.violinplot(data.flatten(), 
    #                 positions=[g_num],
    #                 widths=1, 
    #                 showmedians=True, 
    #                 showextrema=False,
    #                 )
    #     for part in l0["bodies"]:
    #         part.set_edgecolor(cmap(g_num/num_groups))
    #         part.set_facecolor(cmap(g_num/num_groups))
    #     l0["cmedians"].set_edgecolor(cmap(g_num/num_groups))
    #     # color = l0["bodies"][0].get_facecolor().flatten()
    #     # violin_labs.append((mpatches.Patch(color=color), group_name))

    # # labs = [l.get_label() for l in lns]
    # # ax1.legend(lns, labs, loc='upper left')
    # # ax1.legend(*zip(*violin_labs), loc='upper left')
    # # labs = [group_name for group_name,_ in groups]

    # ax1.set_xticks(np.linspace(0,num_groups-1,num_groups))
    # # ax1.set_xlabel(group_type)
    # ax1.set_xticklabels(group_list_str)
    # ax1.set_ylabel(data_list_str[data_type])
    # if data_list_str[data_type] == 'act_mean':
    #     ax1.set_ylim(0,0.22)
    # elif data_list_str[data_type] == 'len_mean':
    #     ax1.set_ylim(0.5,1)
    # elif data_list_str[data_type] == 'de_mean':
    #     ax1.set_ylim(0,1)
    #     ax1.set_ylabel('Directedness')
    # elif data_list_str[data_type] == 'corr_peaks' or data_list_str[data_type] == 'histo_peaks_init' or data_list_str[data_type] == 'histo_peaks_patch':
    #     ax1.set_ylim(0,6)
    # elif data_list_str[data_type] == 'dirent_init' or data_list_str[data_type] == 'dirent_patch':
    #     ax1.set_ylim(0,0.6)
    # elif data_list_str[data_type] == 'avglen_mean':
    #     ax1.set_ylim(0,1)
    # elif data_list_str[data_type] == 'pkf_mean' or data_list_str[data_type] == 'pkt_mean':
    #     ax1.set_ylim(0,8)
    # elif data_list_str[data_type] == 'def_mean' or data_list_str[data_type] == 'det_mean':
    #     ax1.set_ylim(0,.5)
    # elif data_list_str[data_type] == 'decorr_time':
    #     ax1.set_ylim(20,220)
    #     ax1.set_ylabel('Decorrelation Time')

    # if group_type == 'vis':
    #     ax1.set_xlabel('Visual Resolution')
    # elif group_type == 'dist':
    #     ax1.set_xlabel('Distance Scaling Factor')

    # plt.savefig(fr'{data_dir}/group_traj_dists_{group_type}_{data_list_str[data_type]}.png', dpi=100)
    # plt.close()


    # ### plot measurement statistical difference matrix ###
    # for conf_lvl in [0.05, 0.1]:
    #     fig, ax1 = plt.subplots(figsize=(6,6)) 
    #     num_groups = len(group_list)

    #     M = np.zeros((num_groups,num_groups))
    #     for g_num_1, (group_name_1, group_data_1) in enumerate(zip(group_list_str, group_list)):
    #         data_1 = np.array([data[data_type] for data in group_data_1])
    #         for g_num_2, (group_name_2, group_data_2) in enumerate(zip(group_list_str, group_list)):
    #             data_2 = np.array([data[data_type] for data in group_data_2])

    #             U1, p = scipy.stats.mannwhitneyu(data_1, data_2, alternative='two-sided', method='exact')
    #             # nx, ny = len(data_1), len(data_2)
    #             # U2 = nx*ny - U1
    #             # print(group_name_1, group_name_2, p)
    #             M[g_num_1, g_num_2] = p

    #     im = ax1.imshow(M, cmap='coolwarm', norm=mpl.colors.CenteredNorm(vcenter=conf_lvl, halfrange=conf_lvl/2))
    #     # mask = np.tril(np.ones_like(M, dtype=bool))
    #     # ax1.imshow(mask, cmap='binary')
    #     cbar = ax1.figure.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label='p-value')
    #     ax1.set_xticks(np.linspace(0,num_groups-1,num_groups))
    #     ax1.set_yticks(np.linspace(0,num_groups-1,num_groups))
    #     ax1.set_xticklabels(group_list_str)
    #     ax1.set_yticklabels(group_list_str)
    #     if group_type == 'vis':
    #         ax1.set_xlabel('Visual Resolution')
    #         ax1.set_ylabel('Visual Resolution')
    #     elif group_type == 'dist':
    #         ax1.set_xlabel('Distance Scaling Factor')
    #         ax1.set_ylabel('Distance Scaling Factor')
    #     plt.savefig(fr'{data_dir}/group_traj_dists_{group_type}_{data_list_str[data_type]}_diffmat_conflvl{str(conf_lvl).replace(".","p")}.png', dpi=100)
    #     # plt.show()
    #     plt.close()


    ### plot class fitness distributions ###
    fig, ax1 = plt.subplots(figsize=(6,4)) 
    cmap = plt.get_cmap('Spectral')
    num_groups = len(group_list)

    if group_type == 'vis':
        num_labels = 2
    elif group_type == 'dist':
        num_labels = 3
    width = 1/num_labels

    vlabs = []
    labs_taken = []

    for g_num, (group_name, group_data) in enumerate(zip(group_list_str, group_list)):

        BD,IS,DP,BD_IS,IS_DP,DP_BD = [],[],[],[],[],[]
        labels = ['BD','IS','DP','BD_IS','IS_DP','DP_BD']
        colors = ['cornflowerblue', 'tomato', 'forestgreen', 'darkorchid', 'gold', 'aquamarine']


        for name, data_tuple, fitness, label in group_data:
            if label == 'BD':
                BD.append(int(fitness))
            elif label == 'IS':
                IS.append(int(fitness))
            elif label == 'DP':
                DP.append(int(fitness))
            # elif label == 'BD/IS':
            #     BD_IS.append(fitness)
            # elif label == 'IS/DP':
            #     IS_DP.append(fitness)
            # elif label == 'DP/BD':
            #     DP_BD.append(fitness)
        ax1.axvline(x = g_num - width/2, color='k', linestyle='--', linewidth=0.5)

        for l_num, (data, label, color) in enumerate(zip([BD,IS,DP], labels, colors)):
        # for l_num, (data, label, color) in enumerate(zip([BD,IS,DP,BD_IS,IS_DP,DP_BD], labels, colors)):
            pos = g_num + l_num/num_labels
            print(group_name, label, len(data), data)

            if data:
                if len(data) < 5:
                    # continue
                    l0 = ax1.violinplot(data, 
                                positions=[pos],
                                widths=width, 
                                showmedians=False, 
                                showextrema=False,
                                )
                    for part in l0["bodies"]:
                        part.set_edgecolor(color)
                        part.set_facecolor(color)
                    # l0["cmedians"].set_edgecolor(color)
                    # color = l0["bodies"][0].get_facecolor().flatten()

                    if l_num not in labs_taken:
                        vlabs.append((mpl.patches.Patch(color=color), label))
                        labs_taken.append(l_num)
                    
                    if len(data) > 1:
                        x = beeswarm(data)
                    else:
                        x = 0
                    ax1.scatter(pos + x*width/2, data, c=color, s=.5, alpha=1)

                else:
                    l0 = ax1.violinplot(data, 
                                positions=[pos],
                                widths=width, 
                                showmedians=True, 
                                showextrema=False,
                                )
                    for part in l0["bodies"]:
                        part.set_edgecolor(color)
                        part.set_facecolor(color)
                    l0["cmedians"].set_edgecolor(color)
                    # color = l0["bodies"][0].get_facecolor().flatten()

                    if l_num not in labs_taken:
                        vlabs.append((mpl.patches.Patch(color=color), label))
                        labs_taken.append(l_num)
                    
                    if len(data) > 1:
                        x = beeswarm(data)
                    else:
                        x = 0
                    ax1.scatter(pos + x*width/2, data, c=color, s=.5, alpha=1)

    ax1.axvline(x = g_num - width/2 + 1, color='k', linestyle='--', linewidth=0.5)

    # labs = [l.get_label() for l in lns]
    # ax1.legend(lns, labs, loc='upper left')
    ax1.legend(*zip(*vlabs), loc='upper left')
    # labs = [group_name for group_name,_ in groups]

    # ax1.set_xlabel(group_type)
    ax1.set_xticklabels(group_list_str)
    ax1.set_ylabel('Time Taken to Reach Patch')
    ax1.set_ylim(170,580)

    if group_type == 'vis':
        ax1.set_xlabel('Visual Resolution')
        ax1.set_xticks(np.linspace(0 + width/2, num_groups-1 + width/2, num_groups))
    elif group_type == 'dist':
        ax1.set_xlabel('Distance Scaling Factor')
        ax1.set_xticks(np.linspace(0 + width, num_groups-1 + width, num_groups))

    plt.savefig(fr'{data_dir}/group_traj_dists_{group_type}_fitnessbylabel.png', dpi=100)
    plt.close()
    # plt.show()





def gamut_2d(data_type_x, data_type_y, group, cluster=None, heatmap=None, sc_type=None, dpi=100):

    print(f'plotting {data_type_x} vs {data_type_y} for {group}')
    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    if group == 'gamut':
        with open(fr'{data_dir}/traj_matrices/{group}_labeled.bin', 'rb') as f:
            data = pickle.load(f)
        print(f'data dict len: {len(data)}')

        vis6 = []
        vis8 = []
        vis10 = []
        vis12 = []
        vis14 = []
        vis16 = []
        vis18 = []
        vis20 = []
        vis24 = []
        vis32 = []
        maxWF = []
        p8WF = []
        p6WF = []
        p5WF = []
        p4WF = []
        p3WF = []
        p2WF = []
        p1WF = []

        for name in data.keys():

            data_tuple, label = data[name]

            if 'vis6' in name:
                vis6.append((data_tuple, name, label))
            elif 'vis8' in name and 'dist' not in name and 'CNN12' not in name:
                vis8.append((data_tuple, name, label))
            elif 'vis10' in name:
                vis10.append((data_tuple, name, label))
            elif 'vis12' in name:
                vis12.append((data_tuple, name, label))
            elif 'vis14' in name:
                vis14.append((data_tuple, name, label))
            elif 'vis16' in name:
                vis16.append((data_tuple, name, label))
            elif 'vis18' in name:
                vis18.append((data_tuple, name, label))
            elif 'vis20' in name:
                vis20.append((data_tuple, name, label))
            elif 'vis24' in name:
                vis24.append((data_tuple, name, label))
            elif 'vis32' in name:
                vis32.append((data_tuple, name, label))
            elif 'maxWF' in name:
                maxWF.append((data_tuple, name, label))
            elif 'p9WF' in name:
                p8WF.append((data_tuple, name, label))
            elif 'p8WF' in name:
                p6WF.append((data_tuple, name, label))
            elif 'mlWF' in name:
                p5WF.append((data_tuple, name, label))
            elif 'mWF' in name:
                p4WF.append((data_tuple, name, label))
            elif 'msWF' in name:
                p3WF.append((data_tuple, name, label))
            elif '_sWF' in name:
                p2WF.append((data_tuple, name, label))
            elif 'ssWF' in name:
                p1WF.append((data_tuple, name, label))
            else: print(f'{name}, not included')

        vis_groups = [vis6, vis8, vis10, vis12, vis14, vis16, vis18, vis20, vis24, vis32]
        # vis_groups_str = ['6', '8', '10', '12', '14', '16', '18', '20', '24', '32']
        vis_groups_str = [r'$\sigma = 0, \upsilon = 6$',
                    r'$\sigma = 0, \upsilon = 8$',
                    r'$\sigma = 0, \upsilon = 10$',
                    r'$\sigma = 0, \upsilon = 12$',
                    r'$\sigma = 0, \upsilon = 14$',
                    r'$\sigma = 0, \upsilon = 16$',
                    r'$\sigma = 0, \upsilon = 18$',
                    r'$\sigma = 0, \upsilon = 20$',
                    r'$\sigma = 0, \upsilon = 24$',
                    r'$\sigma = 0, \upsilon = 32$',
                    ]
        dist_groups = [maxWF, p8WF, p6WF, p5WF, p4WF, p3WF, p2WF, p1WF]
        # dist_groups_str = ['1', '0.8', '0.6', '0.5', '0.4', '0.3', '0.2', '0.1']
        dist_groups_str = [r'$\sigma = 1, \upsilon = 8$',
                    r'$\sigma = 0.8, \upsilon = 8$',
                    r'$\sigma = 0.6, \upsilon = 8$',
                    r'$\sigma = 0.5, \upsilon = 8$',
                    r'$\sigma = 0.4, \upsilon = 8$',
                    r'$\sigma = 0.3, \upsilon = 8$',
                    r'$\sigma = 0.2, \upsilon = 8$',
                    r'$\sigma = 0.1, \upsilon = 8$',
                    ]

        group_list = []
        group_list_str = []
        for g, gs in zip(dist_groups, dist_groups_str):
            group_list.append(g)
            group_list_str.append(gs)
        for g, gs in zip(vis_groups, vis_groups_str):
            group_list.append(g)
            group_list_str.append(gs)
        num_groups = len(group_list)

        data_list_str = ['corr_peaks', 'decorr_time', 'histo_avg_init', 'histo_avg_patch', 'histo_peaks_init', 'histo_peaks_patch', 'dirent_init', 'dirent_patch', 'min_action', 'max_action', 'avglen_mean', 'avglen_med', 'avglen_min', 'avglen_max', 'pkf_mean', 'pkf_med', 'pkf_min', 'pkf_max', 'pkt_mean', 'pkt_med', 'pkt_min', 'pkt_max', 'def_mean', 'def_med', 'def_min', 'def_max', 'det_mean', 'det_med', 'det_min', 'det_max', 'act_mean', 'act_min', 'act_max', 'len_mean', 'len_min', 'len_max', 'de_mean', 'de_min', 'de_max']

    elif group == 'gamut_visall_nodist':
        with open(fr'{data_dir}/traj_matrices/{group}_labeled.bin', 'rb') as f:
            data = pickle.load(f)
        print(f'data dict len: {len(data)}')

        vis6 = []
        vis8 = []
        vis10 = []
        vis12 = []
        vis14 = []
        vis16 = []
        vis18 = []
        vis20 = []
        vis24 = []
        vis32 = []

        for name in data.keys():

            data_tuple, label = data[name]

            if 'vis6' in name:
                vis6.append((data_tuple, name, label))
            elif 'vis8' in name and 'dist' not in name and 'CNN12' not in name:
                vis8.append((data_tuple, name, label))
            elif 'vis10' in name:
                vis10.append((data_tuple, name, label))
            elif 'vis12' in name:
                vis12.append((data_tuple, name, label))
            elif 'vis14' in name:
                vis14.append((data_tuple, name, label))
            elif 'vis16' in name:
                vis16.append((data_tuple, name, label))
            elif 'vis18' in name:
                vis18.append((data_tuple, name, label))
            elif 'vis20' in name:
                vis20.append((data_tuple, name, label))
            elif 'vis24' in name:
                vis24.append((data_tuple, name, label))
            elif 'vis32' in name:
                vis32.append((data_tuple, name, label))
            else: print(f'{name}, not included')

        vis_groups = [vis6, vis8, vis10, vis12, vis14, vis16, vis18, vis20, vis24, vis32]
        # vis_groups_str = ['6', '8', '10', '12', '14', '16', '18', '20', '24', '32']
        vis_groups_str = [r'$\sigma = 0, \upsilon = 6$',
                    r'$\sigma = 0, \upsilon = 8$',
                    r'$\sigma = 0, \upsilon = 10$',
                    r'$\sigma = 0, \upsilon = 12$',
                    r'$\sigma = 0, \upsilon = 14$',
                    r'$\sigma = 0, \upsilon = 16$',
                    r'$\sigma = 0, \upsilon = 18$',
                    r'$\sigma = 0, \upsilon = 20$',
                    r'$\sigma = 0, \upsilon = 24$',
                    r'$\sigma = 0, \upsilon = 32$',
                    ]

        group_list = []
        group_list_str = []
        for g, gs in zip(vis_groups, vis_groups_str):
            group_list.append(g)
            group_list_str.append(gs)
        num_groups = len(group_list)

        data_list_str = ['corr_peaks', 'decorr_time', 'histo_avg_init', 'histo_avg_patch', 'histo_peaks_init', 'histo_peaks_patch', 'dirent_init', 'dirent_patch', 'act_mean', 'act_min', 'act_max', 'len_mean', 'len_min', 'len_max', 'de_mean', 'de_min', 'de_max']

    elif group == 'gamut_vis8_dist' or group == 'gamut_vis32_dist':
        with open(fr'{data_dir}/traj_matrices/{group}_labeled.bin', 'rb') as f:
            data = pickle.load(f)
        print(f'data dict len: {len(data)}')

        maxWF = []
        p8WF = []
        p6WF = []
        p5WF = []
        p4WF = []
        p3WF = []
        p2WF = []
        p1WF = []

        for name in data.keys():

            data_tuple, label = data[name]

            if 'maxWF' in name:
                maxWF.append((data_tuple, name, label))
            elif 'p9WF' in name:
                p8WF.append((data_tuple, name, label))
            elif 'p8WF' in name:
                p6WF.append((data_tuple, name, label))
            elif 'mlWF' in name:
                p5WF.append((data_tuple, name, label))
            elif 'mWF' in name:
                p4WF.append((data_tuple, name, label))
            elif 'msWF' in name:
                p3WF.append((data_tuple, name, label))
            elif '_sWF' in name:
                p2WF.append((data_tuple, name, label))
            elif 'ssWF' in name:
                p1WF.append((data_tuple, name, label))
            else: print(f'{name}, not included')

        dist_groups = [maxWF, p8WF, p6WF, p5WF, p4WF, p3WF, p2WF, p1WF]
        # dist_groups_str = ['1', '0.8', '0.6', '0.5', '0.4', '0.3', '0.2', '0.1']
        dist_groups_str = [r'$\sigma = 1, \upsilon = 8$',
                    r'$\sigma = 0.8, \upsilon = 8$',
                    r'$\sigma = 0.6, \upsilon = 8$',
                    r'$\sigma = 0.5, \upsilon = 8$',
                    r'$\sigma = 0.4, \upsilon = 8$',
                    r'$\sigma = 0.3, \upsilon = 8$',
                    r'$\sigma = 0.2, \upsilon = 8$',
                    r'$\sigma = 0.1, \upsilon = 8$',
                    ]

        group_list = []
        group_list_str = []
        for g, gs in zip(dist_groups, dist_groups_str):
            group_list.append(g)
            group_list_str.append(gs)
        num_groups = len(group_list)

        data_list_str = ['corr_peaks', 'decorr_time', 'histo_avg_init', 'histo_avg_patch', 'histo_peaks_init', 'histo_peaks_patch', 'dirent_init', 'dirent_patch', 'act_mean', 'act_min', 'act_max', 'len_mean', 'len_min', 'len_max', 'de_mean', 'de_min', 'de_max']

    elif group == 'gamut_other':
        with open(fr'{data_dir}/traj_matrices/{group}_labeled.bin', 'rb') as f:
            data = pickle.load(f)
        print(f'data dict len: {len(data)}')

        actspacehalf = []
        cnn13 = []
        cnn15 = []
        cnn16 = []
        cnn17 = []
        fnn16 = []
        fnn2x16 = []
        fov35 = []
        fov45 = []

        for name in data.keys():

            data_tuple, label = data[name]

            if 'actspacehalf' in name:
                actspacehalf.append((data_tuple, name, label))
            elif 'CNN13' in name:
                cnn13.append((data_tuple, name, label))
            elif 'CNN15' in name:
                cnn15.append((data_tuple, name, label))
            elif 'CNN16' in name:
                cnn16.append((data_tuple, name, label))
            elif 'CNN17' in name:
                cnn17.append((data_tuple, name, label))
            elif 'FNN16' in name:
                fnn16.append((data_tuple, name, label))
            elif 'FNN2x16' in name:
                fnn2x16.append((data_tuple, name, label))
            elif 'fov35' in name:
                fov35.append((data_tuple, name, label))
            elif 'fov45' in name:
                fov45.append((data_tuple, name, label))
            else: print(f'{name}, not included')

        groups = [actspacehalf, cnn13, cnn15, cnn16, cnn17, fnn16, fnn2x16, fov35, fov45]
        groups_str = ['actspacehalf', 'CNN13', 'CNN15', 'CNN16', 'CNN17', 'FNN16', 'FNN2x16', 'fov35', 'fov45']

        group_list = []
        group_list_str = []
        for g, gs in zip(groups, groups_str):
            group_list.append(g)
            group_list_str.append(gs)
        num_groups = len(group_list)

        data_list_str = ['corr_peaks', 'decorr_time', 'histo_avg_init', 'histo_avg_patch', 'histo_peaks_init', 'histo_peaks_patch', 'dirent_init', 'dirent_patch', 'act_mean', 'act_min', 'act_max', 'len_mean', 'len_min', 'len_max', 'de_mean', 'de_min', 'de_max']

    elif group == 'all':
        with open(fr'{data_dir}/traj_matrices/gamut_visall_nodist_labeled.bin', 'rb') as f:
            data1 = pickle.load(f)
        with open(fr'{data_dir}/traj_matrices/gamut_vis8_dist_labeled.bin', 'rb') as f:
            data2 = pickle.load(f)
        # with open(fr'{data_dir}/traj_matrices/gamut_vis16_dist_labeled.bin', 'rb') as f:
        #     data1 = pickle.load(f)
        with open(fr'{data_dir}/traj_matrices/gamut_vis32_dist_labeled.bin', 'rb') as f:
            data4 = pickle.load(f)
        data = data1 | data2 | data4
        print(f'data dict len: {len(data)}')

        vis6_nodist = []
        vis8_nodist = []
        vis10_nodist = []
        vis12_nodist = []
        vis14_nodist = []
        vis16_nodist = []
        vis18_nodist = []
        vis20_nodist = []
        vis24_nodist = []
        vis32_nodist = []
        vis8_maxWF = []
        vis8_p8WF = []
        vis8_p6WF = []
        vis8_p5WF = []
        vis8_p4WF = []
        vis8_p3WF = []
        vis8_p2WF = []
        vis8_p1WF = []
        vis16_maxWF = []
        vis16_p8WF = []
        vis16_p6WF = []
        vis16_p5WF = []
        vis16_p4WF = []
        vis16_p3WF = []
        vis16_p2WF = []
        vis16_p1WF = []
        vis32_maxWF = []
        vis32_p8WF = []
        vis32_p6WF = []
        vis32_p5WF = []
        vis32_p4WF = []
        vis32_p3WF = []
        vis32_p2WF = []
        vis32_p1WF = []

        for name in data.keys():

            data_tuple, label = data[name]

            if 'vis6' in name:
                vis6_nodist.append((data_tuple, name, label))
            elif 'vis8' in name and 'dist' not in name and 'CNN12' not in name:
                vis8_nodist.append((data_tuple, name, label))
            elif 'vis10' in name:
                vis10_nodist.append((data_tuple, name, label))
            elif 'vis12' in name:
                vis12_nodist.append((data_tuple, name, label))
            elif 'vis14' in name:
                vis14_nodist.append((data_tuple, name, label))
            elif 'vis16' in name and 'dist' not in name:
                vis16_nodist.append((data_tuple, name, label))
            elif 'vis18' in name:
                vis18_nodist.append((data_tuple, name, label))
            elif 'vis20' in name:
                vis20_nodist.append((data_tuple, name, label))
            elif 'vis24' in name:
                vis24_nodist.append((data_tuple, name, label))
            elif 'vis32' in name and 'dist' not in name:
                vis32_nodist.append((data_tuple, name, label))

            elif 'maxWF' in name and 'vis8' in name:
                vis8_maxWF.append((data_tuple, name, label))
            elif 'p9WF' in name and 'vis8' in name:
                vis8_p8WF.append((data_tuple, name, label))
            elif 'p8WF' in name and 'vis8' in name:
                vis8_p6WF.append((data_tuple, name, label))
            elif 'mlWF' in name and 'vis8' in name:
                vis8_p5WF.append((data_tuple, name, label))
            elif 'mWF' in name and 'vis8' in name:
                vis8_p4WF.append((data_tuple, name, label))
            elif 'msWF' in name and 'vis8' in name:
                vis8_p3WF.append((data_tuple, name, label))
            elif '_sWF' in name and 'vis8' in name:
                vis8_p2WF.append((data_tuple, name, label))
            elif 'ssWF' in name and 'vis8' in name:
                vis8_p1WF.append((data_tuple, name, label))

            elif 'maxWF' in name and 'vis16' in name:
                vis16_maxWF.append((data_tuple, name, label))
            elif 'p9WF' in name and 'vis16' in name:
                vis16_p8WF.append((data_tuple, name, label))
            elif 'p8WF' in name and 'vis16' in name:
                vis16_p6WF.append((data_tuple, name, label))
            elif 'mlWF' in name and 'vis16' in name:
                vis16_p5WF.append((data_tuple, name, label))
            elif 'mWF' in name and 'vis16' in name:
                vis16_p4WF.append((data_tuple, name, label))
            elif 'msWF' in name and 'vis16' in name:
                vis16_p3WF.append((data_tuple, name, label))
            elif '_sWF' in name and 'vis16' in name:
                vis16_p2WF.append((data_tuple, name, label))
            elif 'ssWF' in name and 'vis16' in name:
                vis16_p1WF.append((data_tuple, name, label))

            elif 'maxWF' in name and 'vis32' in name:
                vis32_maxWF.append((data_tuple, name, label))
            elif 'p9WF' in name and 'vis32' in name:
                vis32_p8WF.append((data_tuple, name, label))
            elif 'p8WF' in name and 'vis32' in name:
                vis32_p6WF.append((data_tuple, name, label))
            elif 'mlWF' in name and 'vis32' in name:
                vis32_p5WF.append((data_tuple, name, label))
            elif 'mWF' in name and 'vis32' in name:
                vis32_p4WF.append((data_tuple, name, label))
            elif 'msWF' in name and 'vis32' in name:
                vis32_p3WF.append((data_tuple, name, label))
            elif '_sWF' in name and 'vis32' in name:
                vis32_p2WF.append((data_tuple, name, label))
            elif 'ssWF' in name and 'vis32' in name:
                vis32_p1WF.append((data_tuple, name, label))

            else: print(f'{name}, not included')

        vis_groups = [vis6_nodist, vis8_nodist, vis10_nodist, vis12_nodist, vis14_nodist, vis16_nodist, vis18_nodist, vis20_nodist, vis24_nodist, vis32_nodist]
        vis8_dist_groups = [vis8_maxWF, vis8_p8WF, vis8_p6WF, vis8_p5WF, vis8_p4WF, vis8_p3WF, vis8_p2WF, vis8_p1WF]
        vis16_dist_groups = [vis16_maxWF, vis16_p8WF, vis16_p6WF, vis16_p5WF, vis16_p4WF, vis16_p3WF, vis16_p2WF, vis16_p1WF]
        vis32_dist_groups = [vis32_maxWF, vis32_p8WF, vis32_p6WF, vis32_p5WF, vis32_p4WF, vis32_p3WF, vis32_p2WF, vis32_p1WF]
        vis_groups_str = [r'$\sigma = 0, \upsilon = 6$',
                    r'$\sigma = 0, \upsilon = 8$',
                    r'$\sigma = 0, \upsilon = 10$',
                    r'$\sigma = 0, \upsilon = 12$',
                    r'$\sigma = 0, \upsilon = 14$',
                    r'$\sigma = 0, \upsilon = 16$',
                    r'$\sigma = 0, \upsilon = 18$',
                    r'$\sigma = 0, \upsilon = 20$',
                    r'$\sigma = 0, \upsilon = 24$',
                    r'$\sigma = 0, \upsilon = 32$',
                    ]
        vis8_dist_groups_str = [r'$\sigma = 1, \upsilon = 8$',
                    r'$\sigma = 0.8, \upsilon = 8$',
                    r'$\sigma = 0.6, \upsilon = 8$',
                    r'$\sigma = 0.5, \upsilon = 8$',
                    r'$\sigma = 0.4, \upsilon = 8$',
                    r'$\sigma = 0.3, \upsilon = 8$',
                    r'$\sigma = 0.2, \upsilon = 8$',
                    r'$\sigma = 0.1, \upsilon = 8$',
                    ]
        vis16_dist_groups_str = [r'$\sigma = 1, \upsilon = 16$',
                    r'$\sigma = 0.8, \upsilon = 16$',
                    r'$\sigma = 0.6, \upsilon = 16$',
                    r'$\sigma = 0.5, \upsilon = 16$',
                    r'$\sigma = 0.4, \upsilon = 16$',
                    r'$\sigma = 0.3, \upsilon = 16$',
                    r'$\sigma = 0.2, \upsilon = 16$',
                    r'$\sigma = 0.1, \upsilon = 16$',
                    ]
        vis32_dist_groups_str = [r'$\sigma = 1, \upsilon = 32$',
                    r'$\sigma = 0.8, \upsilon = 32$',
                    r'$\sigma = 0.6, \upsilon = 32$',
                    r'$\sigma = 0.5, \upsilon = 32$',
                    r'$\sigma = 0.4, \upsilon = 32$',
                    r'$\sigma = 0.3, \upsilon = 32$',
                    r'$\sigma = 0.2, \upsilon = 32$',
                    r'$\sigma = 0.1, \upsilon = 32$',
                    ]

        group_list = []
        group_list_str = []
        for g, gs in zip(vis_groups, vis_groups_str):
            group_list.append(g)
            group_list_str.append(gs)
        for g, gs in zip(vis8_dist_groups, vis8_dist_groups_str):
            group_list.append(g)
            group_list_str.append(gs)
        # for g, gs in zip(vis16_dist_groups, vis16_dist_groups_str):
        #     group_list.append(g)
        #     group_list_str.append(gs)
        for g, gs in zip(vis32_dist_groups, vis32_dist_groups_str):
            group_list.append(g)
            group_list_str.append(gs)
        num_groups = len(group_list)

        data_list_str = ['corr_peaks', 'decorr_time', 'histo_avg_init', 'histo_avg_patch', 'histo_peaks_init', 'histo_peaks_patch', 'dirent_init', 'dirent_patch', 'act_mean', 'act_min', 'act_max', 'len_mean', 'len_min', 'len_max', 'de_mean', 'de_min', 'de_max']

    else:
        print('group not recognized')
        return


    data_str_x = data_list_str[data_type_x]
    data_str_y = data_list_str[data_type_y]

    # fig, ax1 = plt.subplots(figsize=(6,4)) 
    fig, ax1 = plt.subplots(figsize=(8,5.5)) 
    # fig, ax1 = plt.subplots(figsize=(9,6)) 
    # fig, ax1 = plt.subplots(figsize=(12,8)) 
    cmap = plt.get_cmap('Spectral')

    if cluster is not None:
        full_data = np.array([[],[]]).T
        for group_data in group_list:
            data = np.array([(data[data_type_x],data[data_type_y]) for (data, run_name, label) in group_data])
            full_data = np.vstack((full_data, data))
        # print(full_data.shape)
        if cluster == 'kmeans':
            from sklearn.cluster import KMeans
            model = KMeans(n_clusters=3, n_init=100)
            clusters = model.fit_predict(full_data)
            centers = model.cluster_centers_
        elif cluster == 'gmm':
            from sklearn.mixture import GaussianMixture
            model = GaussianMixture(n_components=3)
            clusters = model.fit_predict(full_data)
            centers = model.means_
        ax1.scatter(full_data[:,0], full_data[:,1], c=clusters, alpha=.5)
        ax1.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
        # ax1.set_title(f'cluster type: {cluster}')

    elif heatmap is not None:
        full_data = np.array([[],[]]).T
        for group_data in group_list:
            data = np.array([(data[data_type_x],data[data_type_y]) for (data, run_name, label) in group_data])
            full_data = np.vstack((full_data, data))
        # print(full_data.shape)

        # x_bins = np.linspace(np.min(full_data[:,0]), np.max(full_data[:,0]), 51)
        # y_bins = np.linspace(np.min(full_data[:,1]), np.max(full_data[:,1]), 51)
        x_bins = np.linspace(18,227, 51)
        y_bins = np.linspace(.3,.9, 51)
        # print(x_bins, y_bins)
        X,Y = np.meshgrid(x_bins, y_bins)
        H,_,_ = np.histogram2d(full_data[:,0], full_data[:,1], bins=[x_bins, y_bins])
        im = ax1.pcolormesh(X, Y, H.T, cmap='plasma')
        # plt.colorbar(im, label='Number Overlapping Runs')

    else:
        data_x, data_y, colors, fitnesses = [],[],[],[]
        points, wedges = [],[]
        for g_num, (group_name, group_data) in enumerate(zip(group_list_str, group_list)):

            BD,IS,DP,BD_IS,IS_DP,DP_BD = 0,0,0,0,0,0
            for (data, run_name, label) in group_data:

                # if np.linalg.norm([data[data_type_x] - 173, data[data_type_y] - .87]) > 5:
                #     continue

                data_x.append(data[data_type_x])
                data_y.append(data[data_type_y])

                # if data[data_type_x] < 90 and data[data_type_y] < 0.6 and label == 'BD/IS':
                #     print(run_name, data[data_type_x], data[data_type_y])
                # if data[data_type_x] > 100 and data[data_type_y] > 0.65 and label == 'BD/IS':
                #     print(run_name, data[data_type_x], data[data_type_y])
                if data[data_type_x] > 155 and data[data_type_y] > 0.65 and label == 'IS/DP':
                    print(run_name, data[data_type_x], data[data_type_y])

                # if data[data_type_x] > 125 and label == 'BD/IS':
                #     print(run_name, data[data_type_x], data[data_type_y])
                # if data[data_type_x] < 150 and label == 'DP':
                #     print(run_name, data[data_type_x], data[data_type_y])
                # if data[data_type_y] < 0.7 and label == 'IS/DP':
                #     print(run_name, data[data_type_x], data[data_type_y])
                # if data[data_type_y] > 0.6 and label == 'DP/BD':
                #     print(run_name, data[data_type_x], data[data_type_y])
                # if data[data_type_y] > 0.57 and label == 'BD':
                #     print(run_name, data[data_type_x], data[data_type_y])
                
                if sc_type == 'label':
                    rot = 45
                    a = 0.5
                    if label == 'BD':
                        ax1.scatter(data[data_type_x], data[data_type_y], color='cornflowerblue', alpha=a)
                        BD += 1
                    elif label == 'IS':
                        ax1.scatter(data[data_type_x], data[data_type_y], color='tomato', alpha=a)
                        IS += 1
                    elif label == 'DP':
                        ax1.scatter(data[data_type_x], data[data_type_y], color='forestgreen', alpha=a)
                        DP += 1
                    elif label == 'BD/IS':
                        # point = (data[data_type_x], data[data_type_y])
                        # # point = (data[data_type_x]/(1+.0021*data[data_type_x])-7, data[data_type_y])
                        # # point = (data[data_type_x]/(1+.0023*data[data_type_x])-6, data[data_type_y])
                        # TL = mpl.patches.Wedge(point, 4, theta1=0+rot, theta2=180+rot, color='cornflowerblue', alpha=.6)
                        # BR = mpl.patches.Wedge(point, 4, theta1=180+rot, theta2=360+rot, color='tomato', alpha=.6)
                        # coll = mpl.collections.PatchCollection(
                        #     [TL,BR], match_original=True, offsets=[point,point],
                        #     transform=mpl.transforms.IdentityTransform(),
                        #     transOffset=ax1.transData
                        #     )
                        # ax1.add_collection(coll)
                        # ax1.scatter(data[data_type_x], data[data_type_y], color='darkorchid', alpha=.6)
                        ax1.scatter(data[data_type_x], data[data_type_y], color='cornflowerblue', marker=MarkerStyle('o', fillstyle='left'), alpha=a)
                        ax1.scatter(data[data_type_x], data[data_type_y], color='tomato', marker=MarkerStyle('o', fillstyle='right'), alpha=a)
                        BD_IS += 1
                    elif label == 'IS/DP':
                        # ax1.scatter(data[data_type_x], data[data_type_y], color='gold', alpha=.6)
                        ax1.scatter(data[data_type_x], data[data_type_y], color='tomato', marker=MarkerStyle('o', fillstyle='left'), alpha=a)
                        ax1.scatter(data[data_type_x], data[data_type_y], color='forestgreen', marker=MarkerStyle('o', fillstyle='right'), alpha=a)
                        IS_DP += 1
                    elif label == 'DP/BD':
                        # ax1.scatter(data[data_type_x], data[data_type_y], color='aquamarine', alpha=.6)
                        ax1.scatter(data[data_type_x], data[data_type_y], color='forestgreen', marker=MarkerStyle('o', fillstyle='left'), alpha=a)
                        ax1.scatter(data[data_type_x], data[data_type_y], color='cornflowerblue', marker=MarkerStyle('o', fillstyle='right'), alpha=a)
                        DP_BD += 1

                elif sc_type == 'fitness':
                    with open(fr'{data_dir}/{run_name}/val_matrix_cen.bin','rb') as f:
                        val_matrix = pickle.load(f)
                    fitnesses.append(np.mean(val_matrix))
                    # print(run_name, label, np.min(fitnesses), np.max(fitnesses))
            
            if sc_type == 'label':
                # print(f'{group_name}: BD,IS,DP,BD_IS,IS_DP,DP_BD: {BD,IS,DP,BD_IS,IS_DP,DP_BD}')
                print(f'{group_name}: BD,BD_IS,IS,IS_DP,DP_BD,DP: {BD,BD_IS,IS,IS_DP,DP_BD,DP}') # plot order
            elif sc_type == 'group':
                ax1.scatter(data_x, data_y, color=cmap(g_num/num_groups), alpha=.5, label=group_name)
                data_x,data_y = [],[]

        # if sc_type == 'label':
        #     ax1.scatter(data_x, data_y, color=colors, alpha=.5)
        if sc_type == 'fitness':
            norm = mpl.colors.Normalize(vmin=180, vmax=500)
            ax1.scatter(data_x, data_y, c=fitnesses, cmap='plasma', norm=norm, alpha=.5)
            print(f'fitness range: {np.min(fitnesses), np.max(fitnesses)}')
            # for x,y,f in zip(data_x, data_y, fitnesses):
            #     print(x,y,f, np.linalg.norm([x-173,y-.87]))

    if (data_type_x,data_type_y) == (1,36):
        ax1.set_xlabel('Decorrelation Time')
        ax1.set_ylabel('Directedness')
    else:
        ax1.set_xlabel(data_str_x)
        ax1.set_ylabel(data_str_y)
    ax1.set_xlim([18,227])
    ax1.set_ylim([.3,.9])
    if sc_type == 'group':
        ax1.legend(loc='lower right', labelspacing=.35)
    elif sc_type == 'label':
        from matplotlib.lines import Line2D
        leg_ele = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='tomato', markersize=7.5, label='Indirect Sequential', alpha=.6),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='cornflowerblue', markersize=7.5, label='Biased Diffusive', alpha=.6),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='forestgreen', markersize=7.5, label='Direct Pathing', alpha=.6)
            # Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', markersize=7.5, label='IS/DP', alpha=.6),
            # Line2D([0], [0], marker='o', color='w', markerfacecolor='darkorchid', markersize=7.5, label='BD/IS', alpha=.6),
            # Line2D([0], [0], marker='o', color='w', markerfacecolor='aquamarine', markersize=7.5, label='DP/BD', alpha=.6),
            ]
        ax1.legend(handles=leg_ele, loc='lower right')

    plt.savefig(fr'{data_dir}/class_2d_{group}_cluster{cluster}_heatmap{heatmap}_sctyp{sc_type}.png', dpi=dpi)
    # plt.show()


# def gamut_2d_iter(data_type_x, data_type_y, sc_type=None, dpi=100):

#     data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
#     with open(fr'{data_dir}/traj_matrices/gamut_labeled.bin', 'rb') as f:
#         data = pickle.load(f)
#     print(f'data dict len: {len(data)}')

#     vis6 = []
#     vis8 = []
#     vis10 = []
#     vis12 = []
#     vis14 = []
#     vis16 = []
#     vis18 = []
#     vis20 = []
#     vis24 = []
#     vis32 = []
#     maxWF = []
#     p8WF = []
#     p6WF = []
#     p5WF = []
#     p4WF = []
#     p3WF = []
#     p2WF = []
#     p1WF = []

#     for name in data.keys():

#         data_tuple, label = data[name]

#         if 'vis6' in name:
#             vis6.append((data_tuple, name, label))
#         elif 'vis8' in name and 'dist' not in name and 'CNN12' not in name:
#             vis8.append((data_tuple, name, label))
#         elif 'vis10' in name:
#             vis10.append((data_tuple, name, label))
#         elif 'vis12' in name:
#             vis12.append((data_tuple, name, label))
#         elif 'vis14' in name:
#             vis14.append((data_tuple, name, label))
#         elif 'vis16' in name:
#             vis16.append((data_tuple, name, label))
#         elif 'vis18' in name:
#             vis18.append((data_tuple, name, label))
#         elif 'vis20' in name:
#             vis20.append((data_tuple, name, label))
#         elif 'vis24' in name:
#             vis24.append((data_tuple, name, label))
#         elif 'vis32' in name:
#             vis32.append((data_tuple, name, label))
#         elif 'maxWF' in name:
#             maxWF.append((data_tuple, name, label))
#         elif 'p9WF' in name:
#             p8WF.append((data_tuple, name, label))
#         elif 'p8WF' in name:
#             p6WF.append((data_tuple, name, label))
#         elif 'mlWF' in name:
#             p5WF.append((data_tuple, name, label))
#         elif 'mWF' in name:
#             p4WF.append((data_tuple, name, label))
#         elif 'msWF' in name:
#             p3WF.append((data_tuple, name, label))
#         elif '_sWF' in name:
#             p2WF.append((data_tuple, name, label))
#         elif 'ssWF' in name:
#             p1WF.append((data_tuple, name, label))
#         else: print(f'{name}, not included')

#     vis_groups = [vis6, vis8, vis10, vis12, vis14, vis16, vis18, vis20, vis24, vis32]
#     vis_groups_str = ['6', '8', '10', '12', '14', '16', '18', '20', '24', '32']
#     dist_groups = [maxWF, p8WF, p6WF, p5WF, p4WF, p3WF, p2WF, p1WF]
#     dist_groups_str = ['1', '0.8', '0.6', '0.5', '0.4', '0.3', '0.2', '0.1']

#     group_list = []
#     group_list_str = []
#     for g, gs in zip(dist_groups, dist_groups_str):
#         group_list.append(g)
#         group_list_str.append(gs)
#     for g, gs in zip(vis_groups, vis_groups_str):
#         group_list.append(g)
#         group_list_str.append(gs)
#     num_groups = len(group_list)

#     data_list_str = ['corr_peaks', 'decorr_time', 'histo_avg_init', 'histo_avg_patch', 'histo_peaks_init', 'histo_peaks_patch', 'dirent_init', 'dirent_patch', 'min_action', 'max_action', 'avglen_mean', 'avglen_med', 'avglen_min', 'avglen_max', 'pkf_mean', 'pkf_med', 'pkf_min', 'pkf_max', 'pkt_mean', 'pkt_med', 'pkt_min', 'pkt_max', 'def_mean', 'def_med', 'def_min', 'def_max', 'det_mean', 'det_med', 'det_min', 'det_max', 'act_mean', 'act_min', 'act_max', 'len_mean', 'len_min', 'len_max', 'de_mean', 'de_min', 'de_max']
#     data_str_x = data_list_str[data_type_x]
#     data_str_y = data_list_str[data_type_y]


#     for g_num, (group_name, group_data) in enumerate(zip(group_list_str, group_list)):

#         fig, ax1 = plt.subplots(figsize=(6,4)) 
#         # fig, ax1 = plt.subplots(figsize=(12,8)) 
#         cmap = plt.get_cmap('Spectral')

#         data_x, data_y, colors, fitnesses = [],[],[],[]
#         BD,IS,DP,BD_IS,IS_DP,DP_BD = 0,0,0,0,0,0
#         for (data, run_name, label) in group_data:
#             data_x.append(data[data_type_x])
#             data_y.append(data[data_type_y])

#             if sc_type == 'label':
#                 if label == 'BD':
#                     colors.append('cornflowerblue')
#                     BD += 1
#                 elif label == 'IS':
#                     colors.append('tomato')
#                     IS += 1
#                 elif label == 'DP':
#                     colors.append('forestgreen')
#                     DP += 1
#                 elif label == 'BD/IS':
#                     colors.append('darkorchid')
#                     BD_IS += 1
#                 elif label == 'IS/DP':
#                     colors.append('black')
#                     IS_DP += 1
#                 elif label == 'DP/BD':
#                     colors.append('cyan')
#                     DP_BD += 1
#                 else: print('label not recognized: ', run_name, label)

#             elif sc_type == 'fitness':
#                 with open(fr'{data_dir}/{run_name}/val_matrix_cen.bin','rb') as f:
#                     val_matrix = pickle.load(f)
#                 fitnesses.append(np.mean(val_matrix))
        
#         # print(f'{group_name}: {round(np.mean(data),2)}')
#         print(f'{group_name}: BD:BD_IS:IS:IS_DP:DP_BD:DP: {BD, BD_IS, IS, IS_DP, DP_BD, DP}')

#         if sc_type == 'group':
#             ax1.scatter(data_x, data_y, color=cmap(g_num/num_groups), alpha=.8, label=group_name)
#             ax1.legend(loc='upper left')
#         elif sc_type == 'label':
#             ax1.scatter(data_x, data_y, color=colors, alpha=.8)
#         elif sc_type == 'fitness':
#             norm = mpl.colors.Normalize(vmin=230, vmax=500)
#             ax1.scatter(data_x, data_y, c=fitnesses, cmap='plasma', norm=norm, alpha=.8)

#         ax1.set_xlabel(data_str_x)
#         ax1.set_ylabel(data_str_y)
#         ax1.set_xlim([15,220])
#         ax1.set_ylim([.29,.91])

#         plt.savefig(fr'{data_dir}/group_traj_dists_2d_{data_str_x}x{data_str_y}_iter_sctyp{sc_type}_{group_name}.png', dpi=dpi)
#         plt.close()


def gamut_table(group, data_type_x=None, data_type_y=None):

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    with open(fr'{data_dir}/traj_matrices/{group}.bin', 'rb') as f:
        data = pickle.load(f)
    print(f'data dict len: {len(data)}')

    # new_data = {}

    for name in data.keys():

        print(name)
        # print(name, data[name])
        # new_data[name] = data[name][0]
        # print(name, len(data[name]))
        # print(name, '\t', int(data[name][1]), '\t', round(data[name][14],2))
        # if type(data[name]) == tuple:
        #     print(f'decorr_time: {round(data[name][1],2)}')
        #     print(f'de_mean: {round(data[name][14],2)}')
        # print('')

    # with open(fr'{data_dir}/traj_matrices/{group}.bin', 'wb') as f:
    #     pickle.dump(new_data, f)


def gamut_label(group):

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'

    if group == 'gamut_visall_nodist':
        with open(fr'{data_dir}/traj_matrices/{group}.bin', 'rb') as f:
            data = pickle.load(f)
        print(f'data dict len: {len(data)}')

        data['sc_CNN12_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep18'] = (data['sc_CNN12_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep18'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep0'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep1'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep5'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep6'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep7'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep10'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep11'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep12'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep14'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep15'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep16'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep17'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep18'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep19'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_seed10k_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_seed10k_rep0'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_seed10k_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_seed10k_rep2'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_seed10k_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_seed10k_rep4'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_seed10k_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_seed10k_rep6'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_seed10k_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_seed10k_rep7'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_seed10k_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_seed10k_rep8'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_seed10k_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_seed10k_rep10'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_seed10k_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_seed10k_rep11'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_seed10k_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_seed10k_rep12'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_seed10k_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_seed10k_rep13'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_seed10k_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_seed10k_rep14'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_seed10k_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_seed10k_rep15'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_seed10k_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_seed10k_rep16'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_seed10k_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_seed10k_rep17'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_seed10k_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_seed10k_rep19'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep0'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep1'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep2'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep3'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep4'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep5'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep6'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep7'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep9'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep10'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep11'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep12'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep13'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep14'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep15'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep17'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep18'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep0'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep1'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep2'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep3'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep4'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep5'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep6'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep7'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep9'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep11'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep13'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep14'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep16'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep17'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep18'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep19'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep0'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep1'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep2'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep3'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep5'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep6'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep7'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep8'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep9'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep10'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep11'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep12'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep13'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep14'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep15'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep16'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep17'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep18'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_seed10k_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_seed10k_rep0'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_seed10k_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_seed10k_rep1'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_seed10k_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_seed10k_rep2'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_seed10k_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_seed10k_rep3'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_seed10k_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_seed10k_rep4'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_seed10k_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_seed10k_rep5'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_seed10k_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_seed10k_rep6'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_seed10k_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_seed10k_rep7'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_seed10k_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_seed10k_rep8'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_seed10k_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_seed10k_rep9'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_seed10k_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_seed10k_rep11'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_seed10k_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_seed10k_rep13'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_seed10k_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_seed10k_rep14'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_seed10k_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_seed10k_rep15'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_seed10k_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_seed10k_rep16'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_seed10k_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_seed10k_rep17'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_seed10k_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_seed10k_rep18'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep0'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep1'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep2'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep3'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep4'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep5'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep6'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep7'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep8'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep9'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep10'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep11'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep12'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep13'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep14'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep15'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep16'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep17'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep18'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep19'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed10k_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed10k_rep1'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed10k_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed10k_rep2'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed10k_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed10k_rep3'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed10k_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed10k_rep4'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed10k_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed10k_rep5'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed10k_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed10k_rep6'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed10k_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed10k_rep7'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed10k_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed10k_rep8'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed10k_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed10k_rep9'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed10k_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed10k_rep10'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed10k_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed10k_rep11'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed10k_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed10k_rep15'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed10k_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed10k_rep16'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed10k_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed10k_rep17'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed10k_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed10k_rep18'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep0'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep1'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep2'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep3'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep4'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep5'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep6'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep7'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep8'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep9'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep10'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep11'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep12'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep13'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep14'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep15'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep17'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep18'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep19'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_seed10k_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_seed10k_rep0'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_seed10k_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_seed10k_rep1'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_seed10k_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_seed10k_rep2'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_seed10k_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_seed10k_rep5'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_seed10k_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_seed10k_rep6'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_seed10k_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_seed10k_rep8'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_seed10k_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_seed10k_rep9'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_seed10k_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_seed10k_rep10'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_seed10k_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_seed10k_rep12'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_seed10k_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_seed10k_rep13'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_seed10k_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_seed10k_rep14'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_seed10k_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_seed10k_rep16'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_seed10k_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_seed10k_rep18'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_seed10k_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_seed10k_rep19'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep1'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep2'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep5'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep6'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep7'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep8'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep10'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep11'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep14'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep15'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep16'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep17'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep18'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_seed10k_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_seed10k_rep0'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_seed10k_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_seed10k_rep2'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_seed10k_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_seed10k_rep3'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_seed10k_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_seed10k_rep5'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_seed10k_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_seed10k_rep6'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_seed10k_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_seed10k_rep7'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_seed10k_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_seed10k_rep8'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_seed10k_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_seed10k_rep9'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_seed10k_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_seed10k_rep10'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_seed10k_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_seed10k_rep12'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_seed10k_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_seed10k_rep14'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_seed10k_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_seed10k_rep15'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_seed10k_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_seed10k_rep18'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_seed10k_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_seed10k_rep19'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_rep0'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_rep2'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_rep3'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_rep5'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_rep6'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_rep7'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_rep8'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_rep9'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_rep11'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_rep12'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_rep13'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_rep16'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_rep17'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_rep18'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_rep19'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_seed10k_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_seed10k_rep1'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_seed10k_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_seed10k_rep2'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_seed10k_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_seed10k_rep3'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_seed10k_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_seed10k_rep5'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_seed10k_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_seed10k_rep6'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_seed10k_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_seed10k_rep8'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_seed10k_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_seed10k_rep12'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_seed10k_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_seed10k_rep13'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_seed10k_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_seed10k_rep14'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_seed10k_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_seed10k_rep15'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_seed10k_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_seed10k_rep16'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_seed10k_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_seed10k_rep18'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_seed10k_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_seed10k_rep19'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_rep0'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_rep3'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_rep4'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_rep5'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_rep8'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_rep10'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_rep11'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_rep13'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_rep14'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_rep17'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_rep18'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_rep19'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_seed10k_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_seed10k_rep0'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_seed10k_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_seed10k_rep1'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_seed10k_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_seed10k_rep4'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_seed10k_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_seed10k_rep6'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_seed10k_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_seed10k_rep7'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_seed10k_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_seed10k_rep8'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_seed10k_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_seed10k_rep9'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_seed10k_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_seed10k_rep10'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_seed10k_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_seed10k_rep11'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_seed10k_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_seed10k_rep12'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_seed10k_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_seed10k_rep13'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_seed10k_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_seed10k_rep14'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_seed10k_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_seed10k_rep15'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_seed10k_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_seed10k_rep16'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_seed10k_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_seed10k_rep18'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_seed10k_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_seed10k_rep19'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep0'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep1'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep2'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep3'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep5'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep6'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep7'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep8'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep9'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep10'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep11'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep13'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep14'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep16'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep18'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep19'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed10k_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed10k_rep1'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed10k_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed10k_rep2'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed10k_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed10k_rep4'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed10k_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed10k_rep7'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed10k_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed10k_rep8'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed10k_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed10k_rep9'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed10k_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed10k_rep10'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed10k_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed10k_rep11'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed10k_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed10k_rep12'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed10k_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed10k_rep13'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed10k_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed10k_rep14'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed10k_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed10k_rep15'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed10k_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed10k_rep16'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed10k_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed10k_rep17'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed10k_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed10k_rep18'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep2'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep4'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep6'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep7'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep8'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep9'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep11'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep12'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep13'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep15'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep17'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep18'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep19'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_seed10k_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_seed10k_rep5'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_seed10k_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_seed10k_rep7'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_seed10k_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_seed10k_rep10'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_seed10k_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_seed10k_rep11'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_seed10k_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_seed10k_rep13'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_seed10k_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_seed10k_rep14'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_seed10k_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_seed10k_rep17'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_seed10k_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_seed10k_rep18'], 'BD')

        with open(fr'{data_dir}/traj_matrices/{group}_labeled.bin', 'wb') as f:
            pickle.dump(data, f)
    
    elif group == 'gamut_vis8_dist':

        with open(fr'{data_dir}/traj_matrices/{group}.bin', 'rb') as f:
            data = pickle.load(f)
        print(f'data dict len: {len(data)}')

        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep0'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep1'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep2'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep3'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep4'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep5'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep6'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep7'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep8'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep9'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep10'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep11'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep12'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep13'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep14'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep15'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep16'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep17'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep18'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep19'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep0'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep1'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep2'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep3'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep4'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep5'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep6'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep7'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep8'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep9'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep10'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep11'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep12'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep13'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep14'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep15'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep16'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep17'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep18'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep0'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep1'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep2'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep3'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep4'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep5'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep6'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep7'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep8'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep9'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep10'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep11'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep12'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep13'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep14'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep15'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep16'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep17'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep18'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep19'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep1'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep2'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep3'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep4'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep5'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep6'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep7'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep8'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep9'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep10'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep11'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep12'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep13'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep14'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep15'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep16'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep17'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep18'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep19'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep0'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep1'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep2'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep3'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep4'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep5'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep6'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep7'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep8'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep9'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep10'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep11'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep12'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep13'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep14'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep15'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep16'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep17'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep18'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep19'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep0'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep1'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep2'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep3'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep4'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep5'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep6'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep7'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep8'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep9'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep10'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep11'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep12'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep13'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep14'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep15'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep16'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep17'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep18'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep19'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep0'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep1'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep2'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep3'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep4'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep5'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep6'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep7'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep8'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep9'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep10'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep11'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep12'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep13'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep14'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep15'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep16'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep17'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep18'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep19'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep0'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep1'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep2'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep3'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep4'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep5'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep6'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep7'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep8'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep9'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep10'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep11'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep12'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep13'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep14'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep15'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep17'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep18'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep19'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep0'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep1'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep2'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep3'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep4'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep5'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep6'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep7'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep8'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep9'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep10'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep11'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep12'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep14'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep15'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep16'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep17'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep18'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep19'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep0'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep1'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep3'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep4'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep5'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep6'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep7'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep8'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep9'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep10'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep11'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep12'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep13'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep14'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep15'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep16'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep17'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep18'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep19'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep0'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep1'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep3'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep5'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep6'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep7'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep8'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep9'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep10'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep12'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep13'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep15'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep16'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep17'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep18'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep19'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep0'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep2'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep3'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep4'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep5'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep6'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep7'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep8'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep9'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep10'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep11'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep12'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep13'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep14'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep15'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep16'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep18'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep19'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep0'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep2'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep3'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep4'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep5'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep6'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep7'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep8'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep11'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep12'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep13'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep14'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep15'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep19'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep5'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep7'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep9'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep10'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep11'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep12'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep13'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep15'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep16'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep18'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep19'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep0'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep1'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep2'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep3'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep6'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep7'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep8'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep9'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep10'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep11'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep12'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep14'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep15'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep16'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep17'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep18'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep19'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep0'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep1'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep2'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep3'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep5'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep7'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep8'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep9'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep11'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep12'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep13'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep14'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep18'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep19'], 'IS')

        with open(fr'{data_dir}/traj_matrices/{group}_labeled.bin', 'wb') as f:
            pickle.dump(data, f)

    elif group == 'gamut_vis32_dist':

        data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
        with open(fr'{data_dir}/traj_matrices/{group}.bin', 'rb') as f:
            data = pickle.load(f)
        print(f'data dict len: {len(data)}')

        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep0'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep1'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep2'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep3'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep4'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep5'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep6'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep7'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep8'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep9'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep10'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep11'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep12'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep13'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep14'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep15'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep16'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep17'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep18'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep19'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep0'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep1'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep2'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep3'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep4'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep5'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep6'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep7'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep8'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep9'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep10'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep11'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep12'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep13'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep14'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep15'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep16'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep17'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep18'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep19'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep0'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep1'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep2'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep3'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep4'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep5'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep6'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep7'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep8'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep9'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep10'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep11'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep12'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep13'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep14'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep15'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep16'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep17'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep18'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep19'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep0'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep1'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep2'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep3'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep4'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep5'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep6'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep7'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep8'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep9'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep10'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep11'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep12'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep13'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep14'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep15'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep16'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep17'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep18'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep19'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep0'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep1'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep2'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep3'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep4'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep5'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep6'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep7'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep8'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep9'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep10'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep11'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep12'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep13'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep14'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep15'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep16'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep17'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep18'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep19'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_rep0'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_rep2'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_rep3'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_rep4'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_rep5'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_rep6'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_rep7'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_rep10'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_rep11'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_rep12'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_rep13'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_rep14'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_rep15'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_rep16'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_rep17'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_rep18'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_rep1'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_rep3'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_rep5'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_rep8'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_rep9'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_rep12'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_rep14'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_rep15'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_rep16'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_rep17'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_rep18'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_rep19'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_rep0'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_rep1'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_rep6'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_rep8'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_rep9'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_rep11'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_rep16'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_rep17'], 'IS')

        with open(fr'{data_dir}/traj_matrices/{group}_labeled.bin', 'wb') as f:
            pickle.dump(data, f)

    elif group == 'gamut_other':

        data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
        with open(fr'{data_dir}/traj_matrices/{group}.bin', 'rb') as f:
            data = pickle.load(f)
        print(f'data dict len: {len(data)}')

        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_rep0'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_rep1'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_rep2'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_rep5'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_rep6'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_rep7'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_rep8'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_rep9'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_rep11'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_rep13'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_rep14'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_rep15'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_rep16'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_rep17'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_rep19'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_seed10k_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_seed10k_rep0'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_seed10k_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_seed10k_rep5'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_seed10k_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_seed10k_rep8'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_seed10k_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_seed10k_rep10'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_seed10k_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_seed10k_rep11'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_seed10k_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_seed10k_rep13'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_seed10k_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_seed10k_rep14'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_seed10k_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_seed10k_rep15'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_seed10k_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_seed10k_rep16'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_seed10k_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_seed10k_rep18'], 'IS')
        data['sc_CNN12_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep18'] = (data['sc_CNN12_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep18'], 'BD/IS')
        data['sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep0'] = (data['sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep0'], 'BD')
        data['sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep1'] = (data['sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep1'], 'BD')
        data['sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep2'] = (data['sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep2'], 'BD')
        data['sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep3'] = (data['sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep3'], 'IS')
        data['sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep5'] = (data['sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep5'], 'BD')
        data['sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep6'] = (data['sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep6'], 'BD')
        data['sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep7'] = (data['sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep7'], 'IS')
        data['sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep8'] = (data['sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep8'], 'BD/IS')
        data['sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep9'] = (data['sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep9'], 'IS')
        data['sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep11'] = (data['sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep11'], 'IS')
        data['sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep12'] = (data['sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep12'], 'BD')
        data['sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep14'] = (data['sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep14'], 'IS')
        data['sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep17'] = (data['sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep17'], 'BD')
        data['sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep18'] = (data['sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep18'], 'IS')
        data['sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep19'] = (data['sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep19'], 'BD')
        data['sc_CNN15_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep0'] = (data['sc_CNN15_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep0'], 'IS')
        data['sc_CNN15_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep1'] = (data['sc_CNN15_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep1'], 'IS')
        data['sc_CNN15_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep2'] = (data['sc_CNN15_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep2'], 'BD/IS')
        data['sc_CNN15_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep4'] = (data['sc_CNN15_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep4'], 'IS')
        data['sc_CNN15_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep5'] = (data['sc_CNN15_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep5'], 'IS')
        data['sc_CNN15_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep6'] = (data['sc_CNN15_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep6'], 'IS')
        data['sc_CNN15_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep8'] = (data['sc_CNN15_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep8'], 'BD/IS')
        data['sc_CNN15_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep9'] = (data['sc_CNN15_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep9'], 'IS')
        data['sc_CNN15_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep10'] = (data['sc_CNN15_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep10'], 'BD')
        data['sc_CNN15_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep11'] = (data['sc_CNN15_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep11'], 'BD')
        data['sc_CNN15_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep12'] = (data['sc_CNN15_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep12'], 'IS')
        data['sc_CNN15_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep14'] = (data['sc_CNN15_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep14'], 'IS')
        data['sc_CNN15_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep16'] = (data['sc_CNN15_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep16'], 'IS')
        data['sc_CNN15_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep17'] = (data['sc_CNN15_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep17'], 'BD/IS')
        data['sc_CNN15_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep18'] = (data['sc_CNN15_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep18'], 'IS')
        data['sc_CNN16_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep1'] = (data['sc_CNN16_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep1'], 'BD')
        data['sc_CNN16_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep2'] = (data['sc_CNN16_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep2'], 'BD')
        data['sc_CNN16_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep3'] = (data['sc_CNN16_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep3'], 'BD/IS')
        data['sc_CNN16_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep4'] = (data['sc_CNN16_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep4'], 'IS')
        data['sc_CNN16_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep5'] = (data['sc_CNN16_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep5'], 'BD')
        data['sc_CNN16_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep7'] = (data['sc_CNN16_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep7'], 'IS')
        data['sc_CNN16_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep9'] = (data['sc_CNN16_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep9'], 'BD')
        data['sc_CNN16_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep10'] = (data['sc_CNN16_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep10'], 'BD')
        data['sc_CNN16_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep11'] = (data['sc_CNN16_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep11'], 'IS')
        data['sc_CNN16_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep13'] = (data['sc_CNN16_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep13'], 'IS')
        data['sc_CNN16_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep15'] = (data['sc_CNN16_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep15'], 'IS')
        data['sc_CNN16_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep17'] = (data['sc_CNN16_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep17'], 'IS')
        data['sc_CNN16_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep18'] = (data['sc_CNN16_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep18'], 'IS')
        data['sc_CNN16_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep19'] = (data['sc_CNN16_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep19'], 'IS')
        data['sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep0'] = (data['sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep0'], 'BD/IS')
        data['sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep1'] = (data['sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep1'], 'BD')
        data['sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep3'] = (data['sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep3'], 'IS')
        data['sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep5'] = (data['sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep5'], 'BD')
        data['sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep6'] = (data['sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep6'], 'IS')
        data['sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep7'] = (data['sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep7'], 'BD')
        data['sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep8'] = (data['sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep8'], 'IS')
        data['sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep9'] = (data['sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep9'], 'BD/IS')
        data['sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep10'] = (data['sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep10'], 'BD')
        data['sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep11'] = (data['sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep11'], 'BD')
        data['sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep12'] = (data['sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep12'], 'BD/IS')
        data['sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep14'] = (data['sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep14'], 'IS')
        data['sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep15'] = (data['sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep15'], 'IS')
        data['sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep16'] = (data['sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep16'], 'IS')
        data['sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep17'] = (data['sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep17'], 'BD/IS')
        data['sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep19'] = (data['sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep19'], 'BD')
        data['sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep0'] = (data['sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep0'], 'IS')
        data['sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep1'] = (data['sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep1'], 'IS')
        data['sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep2'] = (data['sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep2'], 'BD/IS')
        data['sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep3'] = (data['sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep3'], 'IS')
        data['sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep4'] = (data['sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep4'], 'BD/IS')
        data['sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep5'] = (data['sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep5'], 'BD/IS')
        data['sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep6'] = (data['sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep6'], 'IS')
        data['sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep7'] = (data['sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep7'], 'BD')
        data['sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep8'] = (data['sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep8'], 'IS')
        data['sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep9'] = (data['sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep9'], 'IS')
        data['sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep10'] = (data['sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep10'], 'IS')
        data['sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep11'] = (data['sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep11'], 'BD/IS')
        data['sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep12'] = (data['sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep12'], 'IS')
        data['sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep13'] = (data['sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep13'], 'IS')
        data['sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep14'] = (data['sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep14'], 'IS')
        data['sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep15'] = (data['sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep15'], 'IS')
        data['sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep16'] = (data['sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep16'], 'BD/IS')
        data['sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep17'] = (data['sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep17'], 'IS')
        data['sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep19'] = (data['sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep19'], 'IS')
        data['sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep0'] = (data['sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep0'], 'IS')
        data['sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep1'] = (data['sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep1'], 'IS')
        data['sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep2'] = (data['sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep2'], 'IS')
        data['sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep3'] = (data['sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep3'], 'BD/IS')
        data['sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep4'] = (data['sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep4'], 'BD/IS')
        data['sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep5'] = (data['sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep5'], 'IS')
        data['sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep6'] = (data['sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep6'], 'IS')
        data['sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep7'] = (data['sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep7'], 'IS')
        data['sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep8'] = (data['sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep8'], 'IS')
        data['sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep9'] = (data['sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep9'], 'BD')
        data['sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep10'] = (data['sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep10'], 'IS')
        data['sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep11'] = (data['sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep11'], 'IS')
        data['sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep12'] = (data['sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep12'], 'IS')
        data['sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep13'] = (data['sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep13'], 'IS')
        data['sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep14'] = (data['sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep14'], 'BD/IS')
        data['sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep15'] = (data['sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep15'], 'IS')
        data['sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep16'] = (data['sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep16'], 'IS')
        data['sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep18'] = (data['sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep18'], 'BD/IS')
        data['sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep19'] = (data['sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep19'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep1'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep2'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep3'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep5'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep7'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep8'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep9'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep10'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep11'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep12'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep14'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep15'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep16'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep17'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep18'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep19'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov45_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov45_rep0'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov45_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov45_rep1'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov45_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov45_rep3'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov45_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov45_rep4'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov45_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov45_rep5'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov45_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov45_rep6'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov45_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov45_rep7'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov45_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov45_rep8'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov45_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov45_rep9'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov45_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov45_rep12'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov45_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov45_rep13'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov45_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov45_rep14'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov45_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov45_rep15'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov45_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov45_rep17'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov45_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov45_rep18'], 'BD')

        with open(fr'{data_dir}/traj_matrices/{group}_labeled.bin', 'wb') as f:
            pickle.dump(data, f)
    
    else:
        print(f'group {group} not found')



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

    ## traj
    space_step = 25
    orient_step = np.pi/8
    timesteps = 500

    ## quick test
    # space_step = 500
    # orient_step = np.pi/2

    # noise_types = [
    #     (0, 'no_noise'), 
    #     (0.05, 'angle_n05'), 
    #     (0.10, 'angle_n10'),
    #     ]

    names = []

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'

    # names.append('sc_CNN12_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep18')
    # for i in [1,3,4,9,15]:
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{str(i)}')
    # for i in [2,3,4,9,13]:
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep{str(i)}')

    # for i in [2,10,18]: 
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep{str(i)}')
    # for i in [12]: 
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep{str(i)}')


    ### ------ views/phys resolution scaling ------- ###

    # vfr = 8
    # num_views = []
    # vis_res = [2,4,8,12,16,20,24,32,48,64,96,128,256,512]
    # for vfr in vis_res:
    #     views = build_agent_views(vis_field_res=vfr)
    #     num_views.append(len(views))
    #     print(vfr, len(views))
    
    # # plot vis_res vs num_views
    # fig, ax = plt.subplots()
    # ax.plot(vis_res, num_views, marker='o')
    # ax.set_xlabel('Visual Field Resolution')
    # ax.set_ylabel('Number of Views')
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # plt.savefig('views_vs_visres_loglog.png')
    # plt.show()

    # states_per_joint = []
    # joints = [1,2,4,8]
    # positions = [2,4,8]
    # for j in joints:
    #     num_states = []
    #     for p in positions:
    #         states = p**j
    #         num_states.append(states)
    #     states_per_joint.append(num_states)
    # print(states_per_joint)
    # fig, ax = plt.subplots()
    # for i, j in enumerate(joints):
    #     ax.plot(positions, states_per_joint[i], marker='o', label=f'{j} joints')
    # ax.set_xlabel('Physical Resolution')
    # ax.set_ylabel('Number of States')
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.legend()
    # plt.savefig('states_vs_physres_loglog.png')
    # plt.show()


    
    ### ------ IDM ------- ###

    # name = 'sc_CNN12_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep18'
    # gen, valfit = find_top_val_gen(name, 'cen')
    # space_step = 5
    # # vfr = 128
    # orient_step = np.pi/256
    # # # to = 0
    # # for to in [0, np.pi/2, np.pi, 3*np.pi/2, np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]:
    # #     # build_IDM(name, gen, space_step, orient_step, template_orient=to, vis_field_res=vfr)
    # #     # plot_IDM(name, gen, space_step, orient_step, template_orient=to, vis_field_res=vfr, plot_type='_rot', dpi=100)
    # #     # plot_IDM(name, gen, space_step, orient_step, template_orient=to, vis_field_res=vfr, plot_type='_trans', dpi=200)
    # #     # plot_IDM(name, gen, space_step, orient_step, template_orient=to, vis_field_res=vfr, plot_type='_transrot_count', dpi=200)
    # #     # plot_IDM(name, gen, space_step, orient_step, template_orient=to, vis_field_res=vfr, plot_type='_transrot_ori', dpi=200)
    # #     plot_IDM_ori(space_step, orient_step, template_orient=to, vis_field_res=vfr, plot_type='_transrot_ori_perf', dpi=200)

    # # with open(fr'{data_dir}/IDM/views_vfr{vfr}.bin', 'rb') as f:
    # #     views = pickle.load(f)
    # views = [
    #     '00002222',
    #     '00222221',
    #     '11113333',
    #     '13333330',
    #     '22111113',
    #     '22221111',
    #     '30000002',
    #     '33330000'
    # ]
    # for v in views:
    #     print(v)
    #     # print(string_one_hot(v))
    #     # build_IDM_view(name, gen, space_step, orient_step, view=v, vis_field_res=vfr)
    #     # plot_IDM_view(space_step, orient_step, view_onehot=string_one_hot(v), vis_field_res=vfr, plot_type='_transrot_ori_perf', dpi=200)
    #     plot_IDM_view(space_step, orient_step, view_onehot=v, vis_field_res=vfr, plot_type='_transrot_ori_perf', dpi=50)

    # # plot_IDM_avgperfviews(space_step=5, orient_step=np.pi/256, vis_field_res=8, plot_type='_count', dpi=100)
    # # plot_IDM_avgperfviews(space_step=5, orient_step=np.pi/256, vis_field_res=8, plot_type='_ori', dpi=100)
    # # plot_IDM_avgperfviews(space_step=5, orient_step=np.pi/256, vis_field_res=8, plot_type='_heatmap_count', dpi=100)
    # # for name in names:
    # #     gen, valfit = find_top_val_gen(name, 'cen')
    # #     plot_IDM_avgperfviews(space_step=5, orient_step=np.pi/256, vis_field_res=8, plot_type=(name,gen), dpi=100)


    ### ------ vecfield ------- ###

    # names = ['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep1']
    # names = ['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep3']
    # names = ['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep4']
    # names = ['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep9']
    # names = ['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep15']
    # names = ['sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep3']
    # for name in names:
    #     gen, valfit = find_top_val_gen(name, 'cen')
    #     print(f'{name} @ {gen} w {valfit} fitness')

        # build_agent_trajs_parallel(name, gen, space_step, orient_step, timesteps)
        # plot_agent_trajs(name, gen, space_step, orient_step, timesteps, ex_lines=True, dpi=50)
        # plot_agent_trajs(name, gen, space_step, orient_step, timesteps, ex_lines=True, dpi=100)
    #     plot_agent_trajs(name, gen, space_step, orient_step, timesteps, extra='3d')
        # plot_agent_orient_corr(name, gen, space_step, orient_step=np.pi/8, timesteps=500, dpi=100)
        # plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type='_dirent', dpi=100)
        # plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type='_avgori', dpi=50)
        # plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type='_avglen', dpi=50)
        # build_action_matrix(name, gen, space_step, orient_step)
        # plot_action_vecfield(name, gen, space_step, orient_step, plot_type='_avg', colored='ori', dpi=100)
        # plot_action_vecfield(name, gen, space_step, orient_step, plot_type='_avg', colored='len', dpi=100)
    #     angle_medians = plot_agent_valnoise_dists(name, noise_types, dpi=100)


    ### ------ perturbs ------- ###

    # name = 'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep3'
    # gen = 'gen941'
    # plot_agent_trajs(name, gen, space_step, orient_step, timesteps)
    # for e in ['FOV39','FOV41','TLx100','TLy100']:
    #     plot_agent_trajs(name, gen, space_step, orient_step, timesteps, extra=e)
    # for e in ['move75','move125']:
    #     plot_agent_trajs(name, gen, space_step, orient_step, timesteps, extra=e, dpi=50)


    
    ### ------ PRW ------- ###

    # behavior = 'straight'
    # for rot_diff in [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]:
    #     build_agent_trajs_parallel_PRW(space_step, orient_step, timesteps, behavior, rot_diff)
    #     plot_agent_corr_PRW(space_step, orient_step, timesteps, behavior, rot_diff, curve=None, limit=None, dpi=100)
    #     # plot_agent_dirent_PRW(space_step, orient_step, timesteps, behavior, rot_diff, curve=None, limit=None, dpi=100)

    #     rd_str = str(rot_diff).replace(".","p")
    #     cv_str = None
    #     lm_str = None
    #     save_name = fr'{data_dir}/traj_matrices/PRW_{behavior}_rd{rd_str}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_cv{cv_str}_lm{lm_str}'
    #     os.remove(save_name+'.bin')

    # behavior = 'curve'
    # for rot_diff in [0.05, 0.01, 0.005, 0.001]:
    #     for curve in [0.005, 0.01, 0.025, 0.05, 0.1, 0.15]:

    #         rd_str = str(rot_diff).replace(".","p")
    #         cv_str = str(curve).replace(".","p")
    #         lm_str = None
    #         save_name = fr'{data_dir}/traj_matrices/PRW_{behavior}_rd{rd_str}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_cv{cv_str}_lm{lm_str}'
    #         print(save_name)

    #         if not os.path.exists(save_name+'_corr_auto_delayed.png'):
    #             build_agent_trajs_parallel_PRW(space_step, orient_step, timesteps, behavior, rot_diff, curve)
    #             plot_agent_corr_PRW(space_step, orient_step, timesteps, behavior, rot_diff, curve=curve, limit=None, dpi=100)
    #             # plot_agent_dirent_PRW(space_step, orient_step, timesteps, behavior, rot_diff, curve=curve, limit=None, dpi=100)
    #         else:
    #             print(f'already exists')

    #         if os.path.exists(save_name+'.bin'):
    #             os.remove(save_name+'.bin')

    # behavior = 'ratchet'
    # for limit in [np.pi, np.pi*3/4, np.pi/2]:
    #     for rot_diff in [0.01, 0.005, 0.001]:
    #         for curve in [0.005, 0.01, 0.025]:

    #             rd_str = str(rot_diff).replace(".","p")
    #             cv_str = str(curve).replace(".","p")
    #             lm_str = str(round(limit/np.pi, 2)).replace(".","p")
    #             save_name = fr'{data_dir}/traj_matrices/PRW_{behavior}_rd{rd_str}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_cv{cv_str}_lm{lm_str}'
    #             print(save_name)

    #             if not os.path.exists(save_name+'_corr_auto_delayed.png'):
    #                 build_agent_trajs_parallel_PRW(space_step, orient_step, timesteps, behavior, rot_diff, curve, limit)
    #                 plot_agent_corr_PRW(space_step, orient_step, timesteps, behavior, rot_diff, curve=curve, limit=limit, dpi=100)
    #                 # plot_agent_dirent_PRW(space_step, orient_step, timesteps, behavior, rot_diff, curve=curve, limit=limit, dpi=100)
    #             else:
    #                 print(f'already exists')

    #             if os.path.exists(save_name+'.bin'):
    #                 os.remove(save_name+'.bin')

    # behavior = 'straight-biased'
    # for b in [0.1, 0.25, 0.75]:
    #     for rot_diff in [0.01, 0.005, 0.001]:
    #         rd_str = str(rot_diff).replace(".","p")
    #         cv_str = None
    #         lm_str = None
    #         b_str = str(b).replace(".","p")
    #         save_name = fr'{data_dir}/traj_matrices/PRW_{behavior}_rd{rd_str}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_cv{cv_str}_lm{lm_str}_b{b_str}'
    #         print(save_name)

    #         if not os.path.exists(save_name+'_hist_dirent.png'):
    #         # if not os.path.exists(save_name+'_corr_auto_delayed.png'):
    #         # if not os.path.exists(save_name+'_50.png'):
    #             build_agent_trajs_parallel_PRW(space_step, orient_step, timesteps, behavior, rot_diff, bias=b)
    #             plot_agent_corr_PRW(space_step, orient_step, timesteps, behavior, rot_diff, curve=None, limit=None, bias=b, dpi=100)
    #             plot_agent_trajs_PRW(space_step, orient_step, timesteps, behavior, rot_diff, curve=None, limit=None, bias=b)
    #             plot_agent_dirent_PRW(space_step, orient_step, timesteps, behavior, rot_diff, curve=None, limit=None, bias=b, dpi=100)
    #         else:
    #             print(f'already exists')

    #         if os.path.exists(save_name+'.bin'):
    #             os.remove(save_name+'.bin')


    ### ------ gamut ------- ###

    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep{x}' for x in range(20)]:
    #     names.append(name)

    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
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
    # for name in [f'sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)

    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep{x}' for x in range(20)]:
    #     names.append(name)

    # for name in [f'sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_seed10k_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_seed10k_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed10k_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_seed10k_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_seed10k_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_seed10k_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_seed10k_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed10k_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_seed10k_rep{x}' for x in range(20)]:
    #     names.append(name)

    # # # action space
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_rep{x}' for x in range(20)]:
    #    names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_seed10k_rep{x}' for x in range(20)]:
    #    names.append(name)

    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacenarrow_rep{x}' for x in range(20)]:
    #    names.append(name)

    # # # cnn
    # names.append('sc_CNN12_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep18')
    # for name in [f'sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN15_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN16_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)

    # # # fnn
    # for name in [f'sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)

    # for name in [f'sc_CNN14_GRUpara16_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #    names.append(name)

    # # # fov
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov45_rep{x}' for x in range(20)]:
    #     names.append(name)

    # # bound_scale
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_bound1000_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_bound1000_rep{x}' for x in range(20)]:
    #     names.append(name)

    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_bound500_rep{x}' for x in range(20)]:
    #    names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_bound500_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_bound500_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_bound500_rep{x}' for x in range(20)]:
    #     names.append(name)

    # for name in [f'sc_CNN1148_FNN2_p50e20_vis32_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN24_FNN2_p50e20_vis32_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN17_FNN2_p50e20_vis32_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN16_p50e20_vis32_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN27_FNN16_p50e20_vis32_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)]:
    #     names.append(name)

    # for name in [f'sc_CNN24_FNN2_p50e20_vis8_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN27_FNN16_p50e20_vis8_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)]:
    #     names.append(name)

    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)


    # name = 'sc_CNN14_FNN2_p50e20_vis8_2xpinball_rep0'
    # gen, valfit = find_top_val_gen(name, 'cen')
    # plot_agent_trajs(name, gen, space_step, orient_step, timesteps, ex_lines=True, dpi=50)

    # data = {}
    # with open(fr'{data_dir}/traj_matrices/gamut_pinball.bin', 'wb') as f:
    #     pickle.dump(data, f)
    names = []
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_2xpinball_rep{x}' for x in range(19)]:
    #     names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis16_2xpinball_rep{x}' for x in range(17)]:
        names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_maxWF_2xpinball_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN17_FNN16_p50e20_vis16_2xpinball_rep{x}' for x in range(20)]:
    #     names.append(name)
    run_gamut('gamut_pinball', names, dpi=50)

    # data = {}
    # with open(fr'{data_dir}/traj_matrices/gamut_vis16_dist.bin', 'wb') as f:
    #     pickle.dump(data, f)
    names = []
    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep{x+11}' for x in range(9)]:
        names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep{x}' for x in range(20)]:
        names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep{x}' for x in range(20)]:
        names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep{x}' for x in range(20)]:
        names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_rep{x}' for x in range(20)]:
        names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_rep{x}' for x in range(20)]:
        names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_rep{x}' for x in range(20)]:
        names.append(name)
    run_gamut('gamut_vis16_dist', names, dpi=50)


    # analyze_gamut --> input index for desired data type
    # 0-7: corr_peaks, decorr_time, histo_avg_init, histo_avg_patch, histo_peaks_init, histo_peaks_patch, dirent_init, dirent_patch, 
    # 8-13: min_action, max_action, avglen_mean, avglen_med, avglen_min, avglen_max, 
    # 14-21: pkf_mean, pkf_med, pkf_min, pkf_max, pkt_mean, pkt_med, pkt_min, pkt_max, 
    # 22-29: def_mean, def_med, def_min, def_max, det_mean, det_med, det_min, det_max, 
    # 30-38: act_mean, act_min, act_max, len_mean, len_min, len_max, de_mean, de_min, de_max
    
    #for i in [0,1,4,5,6,7]: # corr_peaks, decorr_time, histo_peaks_init, histo_peaks_patch, dirent_init, dirent_patch
    #for i in [10,14,18,22,26]: # avglen_mean, pkf_mean, pkt_mean, def_mean, det_mean
    #for i in [30,33,36]: # act_mean, len_mean, de_mean
    # #for i in [0,1,4,5,6,7,10,14,18,22,26,30,33,36]:
    # for i in [1,36]:
    #     analyze_gamut('vis', i)
        # analyze_gamut('dist', i)
    # analyze_gamut('vis',0)
    # analyze_gamut('dist',0)

    # gamut_table('gamut_visall_nodist')
    # gamut_table('gamut_vis8_dist')
    gamut_table('gamut_vis16_dist')
    # gamut_table('gamut_vis32_dist')
    # gamut_table('gamut_other')
    # gamut_table('gamut_bound')
    # gamut_table('gamut_pinball')

    # gamut_label('gamut_visall_nodist')
    # gamut_label('gamut_vis8_dist')
    # gamut_label('gamut_vis32_dist')
    # gamut_label('gamut_other')

    # gamut_2d(1, 36, 'gamut', sc_type='group')
    # gamut_2d(1, 36, 'gamut', sc_type='label')
    # gamut_2d(1, 36, 'gamut', sc_type='fitness')
    # gamut_2d(1, 36, 'gamut', heatmap=True)
    # gamut_2d(1, 36, 'gamut', cluster='kmeans')
    # gamut_2d(1, 36, 'gamut', cluster='gmm')
    # gamut_2d(1,14, 'gamut_visall_nodist', sc_type='group')
    # gamut_2d(1,14, 'gamut_visall_nodist', sc_type='label')
    # gamut_2d(1,14, 'gamut_visall_nodist', sc_type='fitness')
    # gamut_2d(1,14, 'gamut_visall_nodist', heatmap=True)
    # # gamut_2d(1,14, 'gamut_visall_nodist', cluster='kmeans')
    # # gamut_2d(1,14, 'gamut_visall_nodist', cluster='gmm')
    # gamut_2d(1,14, 'gamut_vis8_dist', sc_type='group')
    # gamut_2d(1,14, 'gamut_vis8_dist', sc_type='label')
    # gamut_2d(1,14, 'gamut_vis8_dist', sc_type='fitness')
    # gamut_2d(1,14, 'gamut_vis8_dist', heatmap=True)
    # # gamut_2d(1,14, 'gamut_vis8_dist', cluster='kmeans')
    # # gamut_2d(1,14, 'gamut_vis8_dist', cluster='gmm')
    # gamut_2d(1,14, 'gamut_vis32_dist', sc_type='group')
    # gamut_2d(1,14, 'gamut_vis32_dist', sc_type='label')
    # gamut_2d(1,14, 'gamut_vis32_dist', sc_type='fitness')
    # gamut_2d(1,14, 'gamut_vis32_dist', heatmap=True)
    # # gamut_2d(1,14, 'gamut_vis32_dist', cluster='kmeans')
    # # gamut_2d(1,14, 'gamut_vis32_dist', cluster='gmm')
    # gamut_2d(1,14, 'all', sc_type='group')
    # gamut_2d(1,14, 'all', sc_type='label')
    # gamut_2d(1,14, 'all', sc_type='fitness')
    # gamut_2d(1,14, 'all', heatmap=True)
    # gamut_2d(1,14, 'all', cluster='kmeans')
    # gamut_2d(1,14, 'all', cluster='gmm')

