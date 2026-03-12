from abm.start_sim import reconstruct_NN, start
from abm.sprites.agent import Agent
from abm.sprites import supcalc
from abm.monitoring.plot_funcs import plot_map_iterative_traj, plot_map_iterative_traj_3d, plot_map_iterative_collisions
from abm.monitoring.util import find_top_val_gen, calc_entropy, calc_KLdiv, calc_JSdiv, beeswarm, name_to_metric

import dotenv as de
from pathlib import Path
import numpy as np
import scipy
import multiprocessing as mp
import _pickle as pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import os, sys, platform
import itertools
import seaborn as sns
from pygam import LinearGAM

# -------------------------- action -------------------------- #

def agent_action_from_xyo(envconf, NN, boundary_endpts, x, y, orient, feat_out=False):

    width, height = tuple(eval(envconf["ENV_SIZE"]))
    agent_radius = int(envconf["RADIUS_AGENT"])
    vis_transform = str(envconf["VIS_TRANSFORM"])
    angl_noise_std = float(envconf["PERCEP_ANGLE_NOISE_STD"])
    dist_noise_std = float(envconf["PERCEP_DIST_NOISE_STD"])
    other_input = int(envconf["RNN_OTHER_INPUT_SIZE"])

    max_dist = np.hypot(width, height)
    min_dist = agent_radius*2

    sim_type = str(envconf["SIM_TYPE"])
    if sim_type == 'walls':
        num_class = 4
    elif sim_type == 'walls, social-RW':
        num_class = 6

    agent = Agent(
            id=0,
            position=(x,y),
            orientation=orient,
            max_vel=int(envconf["MAXIMUM_VELOCITY"]),
            FOV=float(envconf['AGENT_FOV']),
            vis_field_res=int(envconf["VISUAL_FIELD_RESOLUTION"]),
            vision_range=int(envconf["VISION_RANGE"]),
            num_class_elements=num_class,
            consumption=1,
            model=NN,
            boundary_endpts=boundary_endpts,
            window_pad=30,
            radius=int(envconf["RADIUS_AGENT"]),
            color=(0,0,0),
            vis_transform=vis_transform,
            percep_angle_noise_std=angl_noise_std,
            sim_type=sim_type,
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
    elif feat_out:
        agent.action, agent.hidden, vis_feat, RNN_out = agent.model.forward(vis_input, np.array([0]), agent.hidden, feat_out=True)
        output = np.concatenate([vis_feat, RNN_out], axis=-1)
        output = np.insert(output, 0, agent.action)
        return output
    else:
        agent.action, agent.hidden, _, _ = agent.model.forward(vis_input, np.array([0]), agent.hidden)
    
    return agent.action


def build_action_matrix(exp_name, gen_ext, space_step, orient_step, archive=False, feat_out=False):

    save_data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    if archive:
        data_dir = Path(__file__).parent.parent / r'data/simulation_data/archive - ISBDDP/'
    else:
        data_dir = save_data_dir
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
    if not feat_out:
        act_matrix = np.zeros((len(x_range),
                                len(y_range),
                                len(orient_range),
                                ))
        # print(f'act matrix shape (x, y, orient): {act_matrix.shape}')
        for i, x in enumerate(x_range):
            for j, y in enumerate(y_range):
                for k, orient in enumerate(orient_range):
                    act_matrix[i,j,k] = agent_action_from_xyo(envconf, NN, boundary_endpts, x, y, orient)

        with open(fr'{save_data_dir}/action_maps/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_action.bin', 'wb') as f:
            pickle.dump(act_matrix, f)

    else:
        act_matrix = np.zeros((len(x_range),
                                len(y_range),
                                len(orient_range),
                                7 # action + 4x CNN outputs + 2x RNN outputs
                                ))
        for i, x in enumerate(x_range):
            for j, y in enumerate(y_range):
                for k, orient in enumerate(orient_range):
                    act_matrix[i,j,k,:] = agent_action_from_xyo(envconf, NN, boundary_endpts, x, y, orient, feat_out=True)

        with open(fr'{save_data_dir}/action_maps/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_activs.bin', 'wb') as f:
            pickle.dump(act_matrix, f)

    
def plot_action_vecfield(exp_name, gen_ext, space_step, orient_step, plot_type='', colored='count', ex_lines=False, archive=False, dpi=50):
    print(f'plotting action vector field - {exp_name} {plot_type}{colored} @ {dpi} dpi')

    save_data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    if archive:
        data_dir = Path(__file__).parent.parent / r'data/simulation_data/archive - ISBDDP/'
    else:
        data_dir = save_data_dir
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)

    width, height = tuple(eval(envconf["ENV_SIZE"]))
    x_min, x_max = 0, width
    y_min, y_max = 0, height
    coll_boundary_thickness = int(envconf["RADIUS_AGENT"])
    num_bins = int((width - coll_boundary_thickness*2) / space_step)
    x_range = np.linspace(x_min + coll_boundary_thickness, 
                        x_max - coll_boundary_thickness + 1, 
                        num_bins)
    y_range = np.linspace(y_min + coll_boundary_thickness, 
                        y_max - coll_boundary_thickness + 1, 
                        num_bins)
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

    elif plot_type == '_dirent_turn' or plot_type == '_dirent_fwd' or plot_type == '_fwdentropy':
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

    save_name = fr'{save_data_dir}/action_maps/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_action'
    with open(save_name+'.bin', 'rb') as f:
        act_matrix = pickle.load(f)
    act_matrix = np.abs(act_matrix)
    act_matrix = (act_matrix - act_matrix.min()) / (act_matrix.max() - act_matrix.min())

    # inits = [
    #     [700, 200], #BR
    #     [700, 600], #TR
    #     [100, 200], #BL
    #     [200, 700], #TL
    #     [200, 500], #midL
    #     [900, 500], #midR
    #     [500, 500], #mid
    #     [350, 350], #patch
    # ]
    # for x,y in inits:
    #     x_idx = (np.abs(x_range - x)).argmin()
    #     y_idx = (np.abs(y_range - y)).argmin()
    #     print(f'x: {x}, y: {y}')
    #     # print(f'x_idx in range: {x_range[x_idx]}, y_idx in range: {y_range[y_idx]}')
    #     print(f'act_matrix[x_idx,y_idx,ori_idx]: {act_matrix[x_idx,y_idx,:]}')

    if '_tuning' in plot_type:
        if len(plot_type) == 8:
            plot_index = int(plot_type[7])
        elif len(plot_type) == 9:
            act_index = int(plot_type[7])+1 # skip action
            plot_index = int(plot_type[8])

            save_name = fr'{save_data_dir}/action_maps/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_activs'
            with open(save_name+'.bin', 'rb') as f:
                act_matrix = pickle.load(f)
            save_name += str(act_index)

    As,Bs,Cs = [],[],[]
    H_angles, H_activs = [],[]
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
            
            elif '_avgact' in plot_type and '_by_ori' not in plot_type:
                if len(plot_type) == 7:
                    pass
                elif len(plot_type) == 8:
                    act_index = int(plot_type[7])+1 # skip action

                    save_name = fr'{save_data_dir}/action_maps/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_activs'
                    with open(save_name+'.bin', 'rb') as f:
                        act_matrix = pickle.load(f)
                    save_name += str(act_index)
                    # print(act_matrix.shape)

                    act_matrix = (act_matrix - act_matrix.min()) / (act_matrix.max() - act_matrix.min())
                    actions = act_matrix[i,j,:,act_index]

                M[j,i] = np.mean(actions)

            elif '_avgact_by_ori' in plot_type:
                if len(plot_type) == 15:
                    ori_index = int(plot_type[14])
                elif len(plot_type) == 16:
                    act_index = int(plot_type[14])+1 # skip action
                    ori_index = int(plot_type[15])

                    save_name = fr'{save_data_dir}/action_maps/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_activs'
                    with open(save_name+'.bin', 'rb') as f:
                        act_matrix = pickle.load(f)
                    save_name += str(act_index)
                    # print(act_matrix.shape)

                    act_matrix = (act_matrix - act_matrix.min()) / (act_matrix.max() - act_matrix.min())
                    actions = act_matrix[i,j,:,act_index]

                if ori_index == 0:
                    actions = [actions[-1],actions[0]]
                    actions = actions[-4:]
                    actions = np.append(actions, actions[:4], -1)
                else:
                    actions = actions[4+(ori_index-1)*8 : 4+(ori_index-1)*8 + 8]
                M[j,i] = np.mean(actions)

            elif '_avg' in plot_type:
                act_index = int(plot_type[4])+1 # skip action

                save_name = fr'{save_data_dir}/action_maps/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_activs'
                with open(save_name+'.bin', 'rb') as f:
                    act_matrix = pickle.load(f)
                save_name += str(act_index)
                # print(act_matrix.shape)

                act_matrix = (act_matrix - act_matrix.min()) / (act_matrix.max() - act_matrix.min())
                actions = act_matrix[i,j,:,act_index]

                if np.sum((1-actions)) == 0: # if all actions are 1, np.average cannot compute
                    U[j,i] = 0
                    V[j,i] = 0
                else:
                    U[j,i] = np.average(xs, weights=(1-actions))
                    V[j,i] = np.average(ys, weights=(1-actions))

            elif '_tuning' in plot_type:
                actions = act_matrix[i,j,:,act_index]
                # rank actions --> each pos has equal ori dist --> ranking eliminates local activity variations
                # actions = scipy.stats.rankdata(-actions, method='average') # 1 is highest
                # actions = scipy.stats.rankdata(actions, method='average') # 0 is highest
                # actions = (actions - actions.min()) / (actions.max() - actions.min()) # 1 is highest
                # actions = (actions.max() - actions) / (actions.max() - actions.min()) # 0 is highest

                if plot_index == 0: # self-goal vector
                    pt_target = np.array(eval(envconf["RESOURCE_POS"]))
                    x_label = 'Goal-Direction'
                elif plot_index == 1: # self-SW vector
                    pt_target = np.array([0,0])
                    x_label = 'SW Corner'
                elif plot_index == 2: # self-SE vector
                    pt_target = np.array([1000,0])
                    x_label = 'SE Corner'
                elif plot_index == 3: # self-NE vector
                    pt_target = np.array([1000,1000])
                    x_label = 'NE Corner'
                elif plot_index == 4: # self-NW vector
                    pt_target = np.array([0,1000])
                    x_label = 'NW Corner'

                pt_target[1] = 1000 - pt_target[1]
                pt_self = np.array([x,1000 - y])
                # pt_self = np.array([x,y])
                pt_self = np.repeat(pt_self[np.newaxis,:], len(orient_range), axis=0)
                disp = pt_self - pt_target

                # angle_to_target = np.arctan2(disp[:,1], disp[:,0]) # [-pi, pi]
                # As.append(angle_to_target[0])
                # corr_patch_angle_diff = angle_to_target.transpose() - (orient_range-np.pi) # [-2pi, 2pi]
                # Bs.append(corr_patch_angle_diff)
                # corr_patch_angle_diff = corr_patch_angle_diff % (2*np.pi) - np.pi # [-pi, pi]
                # Cs.append(corr_patch_angle_diff)

                angle_to_target = np.arctan2(disp[:,1], disp[:,0]) + np.pi # [0, 2pi]
                # As.append(angle_to_target[0])
                corr_patch_angle_diff = angle_to_target.transpose() - orient_range # [-2pi, 2pi]
                # Bs.append(corr_patch_angle_diff)
                corr_patch_angle_diff = corr_patch_angle_diff % (2*np.pi) - np.pi # [-pi, pi]
                # Cs.append(corr_patch_angle_diff)


                # # top_N_action_ranks = np.argsort(action_ranks)[0] # top rank
                # top_N_action_ranks = np.argsort(action_ranks)[-5:] # top 3 ranks
                # action_ranks = action_ranks[top_N_action_ranks]
                # corr_patch_angle_diff = corr_patch_angle_diff[top_N_action_ranks]


                # dist = np.linalg.norm(disp[0,:])
                # if dist < 100: 
                #     # print(f'x: {x}, y: {y} | dist to {x_label}: {int(dist)}')
                #     continue
                # elif x > 900 or x < 100 or y > 900 or y < 100:
                #     continue
                # else:
                H_angles.append(corr_patch_angle_diff)
                # H_activs.append(actions)
                H_activs.append(actions)

                # if x > 350 and x < 450 and y > 350 and y < 450:
                #     print(x,y)
                #     print(angle_to_target[0])
                #     for ind,ori in enumerate(orient_range):
                #         print(ori, corr_patch_angle_diff[ind], action_ranks[ind])

            else:
                print('invalid plot type')

    # bins_below_thresh = np.sum(M < (M.max() * 0.15)) # count bins below threshold of max*.15
    # print(f'% bins below thresh (max*.15): {bins_below_thresh / M.size * 100}')
    M = scipy.ndimage.gaussian_filter(M, sigma=1)

    if colored == 'ori':
        M = np.arctan2(V, U)
        M = (M + 2*np.pi)%(2*np.pi)
        norm = mpl.colors.Normalize(vmin=0, vmax=2*np.pi)
        # norm = mpl.colors.Normalize(vmin=-np.pi, vmax=np.pi)

        # axes.contourf(X, Y, M, cmap=cmap, norm=norm) # contoured background (issue with 0-2pi discontinuity)
        im = axes.imshow(M, cmap='hsv', norm=norm, extent=(x_min, x_max, y_min, y_max), origin='lower', alpha=.7) # pixelated background
        plt.colorbar(im, label='Mean Action Vector Orientation', 
                     fraction=0.046, pad=0.04,
                    ticks=np.arange(0, 2*np.pi+0.01, np.pi/2),
                    format=mpl.ticker.FixedFormatter(['$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']),)
        # Q = axes.quiver(X,Y, U,V, M, angles='xy', cmap=cmap, norm=norm) # colored arrows
        # Q = axes.streamplot(X,Y, U,-V, color='k', broken_streamlines=False) # streams
        # axes.set_title(f'Avg: {np.mean(M):.2f}, Median: {np.median(M):.2f}, Min: {np.min(M):.2f}, Max: {np.max(M):.2f}')
        axes.quiver(X,Y, U,V)
    
        M_len = np.hypot(U, V)
        flat_argmin = M_len.argmin()
        M_min_x = np.floor_divide(flat_argmin, M_len.shape[0])
        M_min_y = np.remainder(flat_argmin, M_len.shape[0])
        x_basin, y_basin = y_range[M_min_y], x_range[M_min_x]
        # print(M.min() == M[M_min_x, M_min_y])
        # axes.scatter(x_range[M_min_x], y_range[M_min_y], c='k', s=10)
        axes.scatter(x_basin, y_basin, c='k', s=20)

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
        # im = axes.contourf(X,Y,M, cmap='plasma', alpha=.6, extend='max', norm='log')
        plt.colorbar(im, label='Mean Action Vector Magnitude',
                     fraction=0.056, pad=0.04,)
        # axes.set_title(f'Avg: {np.mean(M):.2f}, Median: {np.median(M):.2f}, Min: {np.min(M):.2f}, Max: {np.max(M):.2f}')
        axes.quiver(X,Y, U,V)

        flat_argmin = M.argmin()
        M_min_x = np.floor_divide(flat_argmin, M.shape[0])
        M_min_y = np.remainder(flat_argmin, M.shape[0])
        x_basin, y_basin = y_range[M_min_y], x_range[M_min_x]
        # print(M.min() == M[M_min_x, M_min_y])
        # axes.scatter(x_range[M_min_x], y_range[M_min_y], c='k', s=10)
        axes.scatter(x_basin, y_basin, c='k', s=20)

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

    elif '_avgact' in plot_type:
        # levs = np.linspace(np.min(M), np.max(M), 11)
        # im = axes.contourf(X,Y,M, levs, cmap='plasma', alpha=.6)

        # norm = mpl.colors.Normalize(vmin=M.min(), vmax=M.max())
        M += 0.001 # shift up to avoid log(0)
        norm = mpl.colors.LogNorm(vmin=M.min(), vmax=M.max()) # lognorm by ori
        im = axes.imshow(M, cmap='plasma', norm=norm, extent=(x_min, x_max, y_min, y_max), origin='lower') # pixelated background

        # plt.colorbar(im, label='Mean Action')
        # axes.set_title(f'Avg: {np.mean(M):.2f}, Median: {np.median(M):.2f}, Min: {np.min(M):.2f}, Max: {np.max(M):.2f}')

    elif '_tuning' in plot_type:
        H_angles = np.array(H_angles).flatten()
        H_activs = np.array(H_activs).flatten()

        # As = np.array(As).flatten()
        # Bs = np.array(Bs).flatten()
        # Cs = np.array(Cs).flatten()
        # print(As.min(),As.max())
        # print(Bs.min(),Bs.max())
        # print(Cs.min(),Cs.max())
        # print(H_angles.max(),H_angles.min())

        orient_range = np.linspace(-np.pi, np.pi, 16+1)
        avg_activ = np.histogram(H_angles, bins=orient_range, density=True, weights=H_activs)[0]
        norm_activ = (avg_activ - avg_activ.min()) / (avg_activ.max() - avg_activ.min())

        fig_hist, axes_hist = plt.subplots(figsize=(2,2))
        axes_hist.plot(orient_range[:-1], norm_activ)
        axes_hist.set_xticks(np.linspace(-np.pi,np.pi,3))
        axes_hist.set_xticklabels([r'-$\pi$', '$0$', r'$\pi$'])
        axes_hist.set_yticks([0,1])
        # axes_hist.set_xlabel(x_label)
        axes_hist.set_title(x_label)
        axes_hist.set_ylabel('Normalized Activity')
        plt.tight_layout()
        plt.savefig(fr'{save_name}{plot_type}.png', dpi=dpi)
        plt.close()
        return

    else:
        # Q = axes.quiver(X,Y, U,V)

        # mask outer layer
        mask = np.zeros([num_bins, num_bins])
        mask[0,:] = 1
        mask[:,0] = 1
        mask[-1,:] = 1
        mask[:,-1] = 1
        U = np.ma.masked_where(mask == 1, U)
        V = np.ma.masked_where(mask == 1, V)

        # x_bins = np.linspace(0, x_max, num_bins)
        # y_bins = np.linspace(0, y_max, num_bins)
        # X,Y = np.meshgrid(x_bins, y_bins)

        # # scale down U,V size by half
        # rows, cols = U.shape
        # U_reshaped = U.reshape(rows // 2, 2, cols // 2, 2)
        # V_reshaped = V.reshape(rows // 2, 2, cols // 2, 2)
        # U_scaled = U_reshaped.mean(axis=(1, 3))
        # V_scaled = V_reshaped.mean(axis=(1, 3))
        # U = U_scaled
        # V = V_scaled

        # # scale down U,V size by half, averaging for last (39x39 to 20x20)
        # # pad if odd dimensions
        # rows, cols = U.shape
        # pad_rows = 0 if rows % 2 == 0 else 1
        # pad_cols = 0 if cols % 2 == 0 else 1
        # U_padded = np.pad(U, ((0, pad_rows), (0, pad_cols)), mode='edge')
        # V_padded = np.pad(V, ((0, pad_rows), (0, pad_cols)), mode='edge')
        # X_padded = np.pad(X, ((0, pad_rows), (0, pad_cols)), mode='edge')
        # Y_padded = np.pad(Y, ((0, pad_rows), (0, pad_cols)), mode='edge')
        # # reshape + average
        # padded_rows, padded_cols = U_padded.shape
        # X_reshaped = X_padded.reshape(padded_rows // 4, 4, padded_cols // 4, 4)
        # Y_reshaped = Y_padded.reshape(padded_rows // 4, 4, padded_cols // 4, 4)
        # U_reshaped = U_padded.reshape(padded_rows // 4, 4, padded_cols // 4, 4)
        # V_reshaped = V_padded.reshape(padded_rows // 4, 4, padded_cols // 4, 4)
        # U = U_reshaped.mean(axis=(1, 3))
        # V = V_reshaped.mean(axis=(1, 3))
        # X = X_reshaped.mean(axis=(1, 3))
        # Y = Y_reshaped.mean(axis=(1, 3))

        # rescale from using scipy.ndimage.zoom
        rows, cols = U.shape
        target = 20
        scale_r = target / rows
        scale_c = target / cols

        # use linear interpolation (order=1) to downsample
        U = scipy.ndimage.zoom(U, (scale_r, scale_c), order=1)
        V = scipy.ndimage.zoom(V, (scale_r, scale_c), order=1)
        X = scipy.ndimage.zoom(X, (scale_r, scale_c), order=1)
        Y = scipy.ndimage.zoom(Y, (scale_r, scale_c), order=1)

        H_len = np.hypot(U, V)

        # # mask outer layer
        # mask = np.zeros([num_bins, num_bins])
        # mask[0,:] = 1
        # mask[:,0] = 1
        # mask[-1,:] = 1
        # mask[:,-1] = 1
        # U = np.ma.masked_where(mask == 1, U)
        # V = np.ma.masked_where(mask == 1, V)
        # H_len = np.ma.masked_where(mask == 1, H_len)

        # arrow_len = np.sqrt(U**2 + V**2)
        # U = U/arrow_len
        # V = V/arrow_len
        norm = mpl.colors.Normalize(vmin=H_len.min(), vmax=H_len.max())
        # q = axes.quiver(X, Y, U, V, H_len.T, cmap='gray_r', norm=norm, pivot='middle')
        q = axes.quiver(X, Y, U, V, H_len.T, cmap='Blues', norm=norm, pivot='middle', linewidths=H_len.T.flatten()+.5, edgecolors='royalblue')


    if ex_lines:
        save_name_traj = fr'{save_data_dir}/traj_matrices/{exp_name}_{gen_ext}_c25_o8_t500_cen_e1'
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

    if colored == 'ori' or colored == 'len':
        basin = np.array([x_basin, y_basin])
        patch = np.array([x, y])
        basin_patch_dist = np.linalg.norm(basin - patch)

    axes.set_xticklabels([])
    axes.set_yticklabels([])
    plt.tight_layout()
    if colored == 'count':
        plt.savefig(fr'{save_name}{plot_type}.png', dpi=dpi)
    else:
        plt.savefig(fr'{save_name}{plot_type}_C{colored}.png', dpi=dpi)
    # plt.savefig(fr'{save_name}{plot_type}_C{colored}_colorbar.png', dpi=dpi*4)
    # plt.show()
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


# -------------------------- trajectory -------------------------- #

def build_agent_trajs(exp_name, gen_ext, space_step, orient_step, timesteps, rank='cen', eye=True, extra='', archive=False, feat_out=False):
    print(f'building {exp_name}, {gen_ext}, {space_step}, {int(np.pi/orient_step)}, {timesteps}, {extra}')

    # pull pv + envconf from save folders
    save_data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    if archive:
        data_dir = Path(__file__).parent.parent / r'data/simulation_data/archive - ISBDDP/'
    else:
        data_dir = save_data_dir
    if rank == 'top':   NN_pv_path = fr'{data_dir}/{exp_name}/{gen_ext}_NN0_pickle.bin'
    elif rank == 'cen': NN_pv_path = fr'{data_dir}/{exp_name}/{gen_ext}_NNcen_pickle.bin'
    with open(NN_pv_path,'rb') as f:
        pv = pickle.load(f)
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)

    if extra == '': 
        extra = envconf['N']
        save_extra = ''
    else:
        save_extra = extra

    # every grid position/direction
    width, height = tuple(eval(envconf["ENV_SIZE"]))
    x_min, x_max = 0, width
    y_min, y_max = 0, height
    coll_boundary_thickness = int(envconf["RADIUS_AGENT"])*2
    x_range = np.linspace(x_min + coll_boundary_thickness, 
                        x_max - coll_boundary_thickness, 
                        int((width - coll_boundary_thickness*2) / space_step))
    y_range = np.linspace(y_min + coll_boundary_thickness, 
                        y_max - coll_boundary_thickness, 
                        int((height - coll_boundary_thickness*2) / space_step))
    orient_range = np.arange(0, 2*np.pi, orient_step)
    # print(f'testing ranges (max, min): x[{x_range[0], x_range[-1]}], y[{y_range[0], y_range[-1]}], o[{orient_range[0], orient_range[-1]}]')
    
    # construct matrix of each traj for each grid pos/dir
    num_inits = len(x_range) * len(y_range) * len(orient_range)
    if feat_out:
        if envconf['SIM_TYPE'] == 'walls':
            traj_matrix = np.zeros( (num_inits, timesteps, 10) ) # (pos_x, pos_y, _, _, ...) --> to match self.data_agent format
        elif 'social' in envconf['SIM_TYPE']:
            traj_matrix = np.zeros( (num_inits, timesteps, 24) ) # (pos_x, pos_y, _, _, ...)
    else:
        if envconf['SIM_TYPE'] == 'walls':
            traj_matrix = np.zeros( (num_inits, timesteps, 4) ) # (pos_x, pos_y, ori, act)
        elif 'social' in envconf['SIM_TYPE']:
            # traj_matrix = np.zeros( (num_inits, timesteps, 6) ) # (pos_x, pos_y, ori, act, COM_pos_x, COM_pos_y)
            traj_matrix = np.zeros( (num_inits, timesteps, 3) ) # (pos_x, pos_y, ori)
    # print(f'traj matrix shape (# initializations, timesteps, ): {traj_matrix.shape}')
    
    # pack inputs for multiprocessing map
    mp_inputs = []
    seed = 0
    for x in x_range:
        for y in y_range:
            for orient in orient_range:
                init_info = x, y, orient, timesteps, extra, ''
                mp_inputs.append( (None, pv, None, seed, env_path, init_info, feat_out) ) # model_tuple=None, load_dir=None
                seed += 1

    # run agent NNs in parallel
    with mp.Pool() as pool:
        results = pool.starmap_async( start, mp_inputs )
        pool.close()
        pool.join()

    # unpack results into matrix (y coords transformed for plotting)
    results_list = results.get()

    empties = []
    for n, (time_taken, dist_from_patch, output) in enumerate(results_list):
        traj_matrix[n,:,:] =  output
        if not output[0,:].any():
            empties.append(n) 
    for n in empties:
        traj_matrix = np.delete(traj_matrix, n, axis=0)

    traj_matrix[:,:,1] = y_max - traj_matrix[:,:,1]

    if save_extra == '':
        save_name = fr'{save_data_dir}/traj_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e{int(eye)}.bin'
    else:
        save_name = fr'{save_data_dir}/traj_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e{int(eye)}_{extra}.bin'
    with open(save_name, 'wb') as f:
        pickle.dump(traj_matrix, f)


def plot_agent_trajs(exp_name, gen_ext, space_step, orient_step, timesteps, rank='cen', ellipses=False, eye=True, ex_lines=False, act_arr=False, extra='', archive=False, dpi=25):
    print(f'plotting map - {exp_name}, {gen_ext}, ell{ellipses}, ex_lines{ex_lines}, extra{extra}')

    save_data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    if archive:
        data_dir = Path(__file__).parent.parent / r'data/simulation_data/archive - ISBDDP/'
    else:
        data_dir = save_data_dir
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)

    if extra == '' or extra == '3d' or extra == 'clip' or extra == 'turn':
        save_name = fr'{save_data_dir}/traj_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e{int(eye)}'
    elif extra.startswith('3d'):
        save_name = fr'{save_data_dir}/traj_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e{int(eye)}_{extra[3:]}'
    else:
        save_name = fr'{save_data_dir}/traj_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e{int(eye)}_{extra}'
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
        if 'walls' in sim_type:
            if not act_arr:
                # save_name = fr'{data_dir}/action_maps/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e{int(eye)}'
                plot_map_iterative_traj(traj_plot_data, x_max=width, y_max=height, save_name=save_name, ellipses=ellipses, ex_lines=ex_lines, extra=extra, dpi=dpi)
            else:
                save_name = fr'{save_data_dir}/action_maps/{exp_name}_{gen_ext}_c{space_step}_o32_action' # hardset at 32
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


def plot_traj_vecfield(exp_name, gen_ext, space_step, orient_step, timesteps, plot_type='', extra='', mask_cond='', archive=False, dpi=50):
    print(f'traj vecfield - {plot_type} - {exp_name}{extra} @ {dpi} dpi')

    save_data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    if archive:
        data_dir = Path(__file__).parent.parent / r'data/simulation_data/archive - ISBDDP/'
    else:
        data_dir = save_data_dir

    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)
    width, height = tuple(eval(envconf["ENV_SIZE"]))
    x_min, x_max = 0, width
    y_min, y_max = 0, height
    orient_range = np.arange(0, 2*np.pi, orient_step)

    fig, axes = plt.subplots() 
    axes.set_xlim(0, x_max)
    axes.set_ylim(0, y_max)
    h,w = 5,5
    l,r,t,b = fig.subplotpars.left, fig.subplotpars.right, fig.subplotpars.top, fig.subplotpars.bottom
    fig.set_size_inches( float(w)/(r-l) , float(h)/(t-b) )

    if extra == '':
        save_name = fr'{save_data_dir}/traj_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_cen_e1'
    else:
        save_name = fr'{save_data_dir}/traj_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_cen_e1_{extra}'
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
    # action = ag_data[:,delay:,3].flatten()
    # action = np.abs(action)

    num_bins = 25
    x_bins = np.linspace(0, x_max, num_bins+1)
    y_bins = np.linspace(0, y_max, num_bins+1)

    if plot_type == '_count':
        H,_,_ = np.histogram2d(x, y, bins=[x_bins, y_bins])
 
        X,Y = np.meshgrid(x_bins, y_bins)
        axes.pcolormesh(X, Y, H.T, cmap='plasma')

    # elif plot_type == '_avgact':
    #     H_count,_,_ = np.histogram2d(x, y, bins=[x_bins, y_bins])
    #     H,_,_ = np.histogram2d(x, y, bins=[x_bins, y_bins], weights=action)
    #     H = np.divide(H, H_count, out=np.zeros_like(H), where=H_count!=0)
 
    #     X,Y = np.meshgrid(x_bins, y_bins)
    #     norm = mpl.colors.Normalize(vmin=H.min(), vmax=H.max())
    #     axes.pcolormesh(X, Y, H.T, cmap='plasma', norm=norm)
        
    elif plot_type == '_avgori':
        num_bins = 39
        x_bins = np.linspace(0, x_max, num_bins+1)
        y_bins = np.linspace(0, y_max, num_bins+1)

        H_count,_,_ = np.histogram2d(x, y, bins=[x_bins, y_bins])
        H_x,_,_ = np.histogram2d(x, y, bins=[x_bins, y_bins], weights=np.cos(ori))
        H_x = np.divide(H_x, H_count, out=np.zeros_like(H_x), where=H_count!=0)
        H_y,_,_ = np.histogram2d(x, y, bins=[x_bins, y_bins], weights=np.sin(ori))
        H_y = np.divide(H_y, H_count, out=np.zeros_like(H_y), where=H_count!=0)

        H_x = scipy.ndimage.gaussian_filter(H_x, sigma=2) # smoothen
        H_y = scipy.ndimage.gaussian_filter(H_y, sigma=2) # smoothen

        H_ori = np.arctan2(H_y, H_x)
        H_ori = (H_ori + 2*np.pi)%(2*np.pi)
        H_len = np.hypot(H_x,H_y)

        mask = np.zeros_like(H_count)
        mask[H_count < 100] = 1
        H_len = np.ma.masked_where(mask == 1, H_len)
        H_ori= np.ma.masked_where(mask == 1, H_ori)
        # H_activ = np.ma.filled(H_activ, 0) # backfill masked values with zero
        # H_activ = scipy.ndimage.gaussian_filter(H_activ, sigma=1) # smoothen

        norm = mpl.colors.Normalize(vmin=0, vmax=2*np.pi)
        im = axes.imshow(H_ori.T, cmap='hsv', norm=norm, extent=(x_min, x_max, y_min, y_max), origin='lower', alpha=.7)

        plt.colorbar(im, label='Mean Action Vector Orientation',
                    fraction=0.046, pad=0.04,
                    ticks=np.arange(0, 2*np.pi+0.01, np.pi/2),
                    format=mpl.ticker.FixedFormatter(['$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']),)

        thickness = int(envconf["RADIUS_AGENT"])
        x_bins = np.linspace(thickness, x_max-thickness, num_bins)
        y_bins = np.linspace(thickness, y_max-thickness, num_bins)
        X,Y = np.meshgrid(x_bins, y_bins)

        U = np.cos(H_ori.T)*H_len.T
        V = np.sin(H_ori.T)*H_len.T
        U = np.ma.masked_where(mask.T == 1, U)
        V = np.ma.masked_where(mask.T == 1, V)

        axes.quiver(X,Y,U,V)

        # num_mins = 5
        # H_len_mins = np.argsort(H_len)[:num_mins]
        # # print(H_len_mins, H_len_mins.shape)
        # # H_len(H_len_mins
        # for ind in H_len_mins:
        #     pt = H_len[ind]
        #     print(ind,pt)
        #     M_min_x = np.floor_divide(pt, H_len.shape[0])
        #     M_min_y = np.remainder(pt, H_len.shape[0])
        #     x_basin, y_basin = x_bins[M_min_x], y_bins[M_min_y]
        #     axes.scatter(x_basin, y_basin, c='k', s=20)

        pt = H_len.argmin()
        M_min_x = np.floor_divide(pt, H_len.shape[0])
        M_min_y = np.remainder(pt, H_len.shape[0])
        x_basin, y_basin = x_bins[M_min_x], y_bins[M_min_y]
        axes.scatter(x_basin, y_basin, c='k', s=20)

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

    elif plot_type == '_avgorilen':
        H_count,_,_ = np.histogram2d(x, y, bins=[x_bins, y_bins])
        H_x,_,_ = np.histogram2d(x, y, bins=[x_bins, y_bins], weights=np.cos(ori))
        H_x = np.divide(H_x, H_count, out=np.zeros_like(H_x), where=H_count!=0)
        H_y,_,_ = np.histogram2d(x, y, bins=[x_bins, y_bins], weights=np.sin(ori))
        H_y = np.divide(H_y, H_count, out=np.zeros_like(H_y), where=H_count!=0)

        H_x = scipy.ndimage.gaussian_filter(H_x, sigma=1) # smoothen
        H_y = scipy.ndimage.gaussian_filter(H_y, sigma=1) # smoothen

        H_ori = np.arctan2(H_y, H_x)
        H_ori = (H_ori + 2*np.pi)%(2*np.pi)
        H_len = np.hypot(H_x, H_y)

        mask = np.zeros_like(H_count)
        mask[H_count < 100] = 1
        H_ori = np.ma.masked_where(mask == 1, H_ori)
        H_len = np.ma.masked_where(mask == 1, H_len)

        thickness = int(envconf["RADIUS_AGENT"])
        x_bins = np.linspace(thickness, x_max-thickness, num_bins)
        y_bins = np.linspace(thickness, y_max-thickness, num_bins)
        X,Y = np.meshgrid(x_bins, y_bins)

        U = np.cos(H_ori.T)*H_len.T
        V = np.sin(H_ori.T)*H_len.T
        U = np.ma.masked_where(mask.T == 1, U)
        V = np.ma.masked_where(mask.T == 1, V)

        mask = np.zeros((num_bins,num_bins))
        mask[0:2, :] = 1
        mask[:, 0:2] = 1
        mask[-2:, :] = 1
        mask[:, -2:] = 1
        U = np.ma.masked_where(mask == 1, U)
        V = np.ma.masked_where(mask == 1, V)

        # rescale from using scipy.ndimage.zoom
        rows, cols = U.shape
        target = 20
        scale_r = target / rows
        scale_c = target / cols

        # use linear interpolation (order=1) to downsample
        U = scipy.ndimage.zoom(U, (scale_r, scale_c), order=1)
        V = scipy.ndimage.zoom(V, (scale_r, scale_c), order=1)
        X = scipy.ndimage.zoom(X, (scale_r, scale_c), order=1)
        Y = scipy.ndimage.zoom(Y, (scale_r, scale_c), order=1)

        H_len = np.hypot(U, V)

        # # mask outer layer (now target x target)
        # mask = np.zeros((target, target))
        # mask[0, :] = 1
        # mask[:, 0] = 1
        # mask[-1:, :] = 1
        # mask[:, -1:] = 1
        # U = np.ma.masked_where(mask == 1, U)
        # V = np.ma.masked_where(mask == 1, V)
        # H_len = np.ma.masked_where(mask == 1, H_len)

        norm = mpl.colors.Normalize(vmin=H_len.min(), vmax=H_len.max())
        q = axes.quiver(X, Y, U, V, H_len.T, cmap='Blues', norm=norm, pivot='middle',
                linewidths=H_len.T.flatten() + .5, edgecolors='royalblue')


    elif plot_type == '_dirent':
        # simplify data
        x = ag_data[::10,delay:,0].flatten()
        y = ag_data[::10,delay:,1].flatten()
        ori = ag_data[::10,delay:,2].flatten()
        num_bins = 50
        x_bins = np.linspace(0, x_max, num_bins+1)
        y_bins = np.linspace(0, y_max, num_bins+1)

        # drop into bins + organize
        hitx = np.digitize(x, x_bins[1:]) # digitize counts first bin as to left of initial element
        hity = np.digitize(y, y_bins[1:])
        hitbins = list(zip(hitx, hity))
        ori_and_bins = list(zip(ori, hitbins))

        h = np.ones(len(orient_range))/10000
        e_max = calc_entropy(h) # random
        h[0] = 1
        e_min = calc_entropy(h) # uniform

        H = np.zeros((num_bins, num_bins))
        for i in range(num_bins):
            for j in range(num_bins):
                # print(i,j)
                bin_ori = [ori for (ori,bin) in ori_and_bins if bin == (i,j)]
                if bin_ori:
                    h = np.histogram(bin_ori, bins=orient_range)[0]
                    e = calc_entropy(h + 1/10000)
                    d = (e_max - e) / (e_max - e_min)
                    H[i,j] = d
                # else:
                #     H[i,j] = 0
                #     print(i,j,'no data')

        X,Y = np.meshgrid(x_bins, y_bins)
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        im = axes.pcolormesh(X, Y, H.T, cmap='viridis', norm=norm)
        # cbar = axes.figure.colorbar(im, label='Directedness', ax=axes, fraction=0.046, pad=0.04)

        if mask_cond == '':
            # mask edges (100 from each edge) + patch vicinity (100 from center)
            mask = np.zeros([num_bins, num_bins])
            mask[:5,:] = 1
            mask[:,:5] = 1
            mask[46:,:] = 1
            mask[:,46:] = 1
            mask[16:25,26:35] = 1
            H_mask = np.ma.array(H, mask=mask)
        elif mask_cond == 'no_patch':
            mask = np.zeros([num_bins, num_bins])
            mask[:5,:] = 1
            mask[:,:5] = 1
            mask[46:,:] = 1
            mask[:,46:] = 1
            H_mask = np.ma.array(H, mask=mask)
        elif mask_cond == 'patch_only':
            mask = np.zeros([num_bins, num_bins])
            mask[:16,:] = 1
            mask[25:,:] = 1
            mask[:,:26] = 1
            mask[:,35:] = 1
            H_mask = np.ma.array(H, mask=mask)
        elif mask_cond == 'near_patch':
            mask = np.zeros([num_bins, num_bins])
            mask[:11,:] = 1
            mask[30:,:] = 1
            mask[:,:21] = 1
            mask[:,40:] = 1
            H_mask = np.ma.array(H, mask=mask)
        else:
            print('no valid mask condition specified')
        axes.set_title(f'Avg Ent. Directedness: {np.mean(H_mask):.2f}')
        print(f'{np.mean(H_mask):.2f}, {np.min(H_mask):.2f}, {np.max(H_mask):.2f}')


    elif '_avgactiv' in plot_type and '_by_ori' not in plot_type:
        name_len = 9
        if exp_name.startswith('sc_N'):
            act_index = int(plot_type[name_len:])+4 # skip x/y/o/a
        else:
            act_index = int(plot_type[name_len:])+3 # skip x/y/o
        # print(f'act_index: {act_index}')

        activ = ag_data[:,delay:,act_index].flatten()
        min_activ = activ.min()
        max_activ = activ.max()
        # print(min_activ, max_activ)
        if min_activ == max_activ:
            # print(min_activ, max_activ, 'nothing printed')
            return
        activ = (activ - min_activ) / (max_activ - min_activ) # norm to [0,1]
        # print(activ.mean())

        # num_bins = 50 # indiv nav
        num_bins = 100 # social nav
        x_bins = np.linspace(0, x_max, num_bins+1)
        y_bins = np.linspace(0, y_max, num_bins+1)

        H_count,_,_ = np.histogram2d(x, y, bins=[x_bins, y_bins])
        # H_x,_,_ = np.histogram2d(x, y, bins=[x_bins, y_bins], weights=np.cos(ori))
        # H_x = np.divide(H_x, H_count, out=np.zeros_like(H_x), where=H_count!=0)
        # H_y,_,_ = np.histogram2d(x, y, bins=[x_bins, y_bins], weights=np.sin(ori))
        # H_y = np.divide(H_y, H_count, out=np.zeros_like(H_y), where=H_count!=0)
        # H_ori = np.arctan2(H_y, H_x)
        # H_ori = (H_ori + 2*np.pi)%(2*np.pi)
        # H_len = np.hypot(H_x, H_y)
        H_activ = np.histogram2d(x, y, bins=[x_bins, y_bins], weights=activ)[0]
        H_activ = np.divide(H_activ, H_count, out=np.zeros_like(H_activ), where=H_count!=0)

        mask = np.zeros_like(H_count)
        mask[H_count < 100] = 1
        # H_ori = np.ma.masked_where(mask == 1, H_ori)
        # H_len = np.ma.masked_where(mask == 1, H_len)
        H_activ = np.ma.masked_where(mask == 1, H_activ)
        # bins_below_thresh = np.sum(H_activ < (H_activ.max() * 0.15)) # count bins below threshold of max*.15
        # print(f'% bins below thresh (max*.15): {bins_below_thresh / H_activ.size * 100}')
        H_activ = np.ma.filled(H_activ, 0) # backfill masked values with zero
        H_activ = scipy.ndimage.gaussian_filter(H_activ, sigma=1) # smoothen

        x_bins = np.linspace(0, x_max, num_bins)
        y_bins = np.linspace(0, y_max, num_bins)
        X,Y = np.meshgrid(x_bins, y_bins)

        norm = mpl.colors.Normalize(vmin=0, vmax=H_activ.max())
        # H_activ += 0.001 # shift up to avoid log(0)
        # norm = mpl.colors.LogNorm(vmin=H_activ.min(), vmax=H_activ.max()) # lognorm by ori
        im = axes.pcolormesh(X, Y, H_activ.T, cmap='viridis', norm=norm)
        # print(H_activ.min())

        # U = np.cos(H_ori.T)*H_len.T
        # V = np.sin(H_ori.T)*H_len.T
        # U = np.ma.masked_where(mask.T == 1, U)
        # V = np.ma.masked_where(mask.T == 1, V)
        # norm = mpl.colors.Normalize(vmin=H_len.min(), vmax=H_len.max())
        # q = axes.quiver(X, Y, U, V, H_len.T, cmap='gray_r', norm=norm, pivot='middle')

        # # print where H_len.min() is
        # H_len_min_pos = np.where(H_len == H_len.min())
        # H_len_min_x = x_bins[H_len_min_pos[0][0]]
        # H_len_min_y = y_bins[H_len_min_pos[1][0]]
        # print(f'H_len.min @ ({H_len_min_x:.2f},{H_len_min_y:.2f})')

    elif '_avgactiv_by_ori' in plot_type:
        name_len = 16
        act_index = int(plot_type[name_len])+3 # skip x/y/o
        ori_index = int(plot_type[name_len+1])

        activ = ag_data[:,delay:,act_index].flatten()
        min_activ = activ.min()
        max_activ = activ.max()
        activ = (activ - min_activ) / (max_activ - min_activ) # norm to [0,1]

        num_ori_bins = 8
        ori += 2*np.pi/num_ori_bins/2 # shift data by half bin size --> centers bins on 0
        ori[ori > 2*np.pi] -= 2*np.pi # spin data that were pushed over 360
        o_bins = np.linspace(0, 2*np.pi, num_ori_bins+1)
        ori_mask = (ori >= o_bins[ori_index]) & (ori < o_bins[ori_index+1]) # mask if outside ori bin
        x = x[ori_mask]
        y = y[ori_mask]
        ori = ori[ori_mask]
        activ = activ[ori_mask]
        ori -= np.pi/8 # restore for histo/arrows

        num_bins = 50
        x_bins = np.linspace(0, x_max, num_bins+1)
        y_bins = np.linspace(0, y_max, num_bins+1)

        H_count,_,_ = np.histogram2d(x, y, bins=[x_bins, y_bins])
        # H_x,_,_ = np.histogram2d(x, y, bins=[x_bins, y_bins], weights=np.cos(ori))
        # H_x = np.divide(H_x, H_count, out=np.zeros_like(H_x), where=H_count!=0)
        # H_y,_,_ = np.histogram2d(x, y, bins=[x_bins, y_bins], weights=np.sin(ori))
        # H_y = np.divide(H_y, H_count, out=np.zeros_like(H_y), where=H_count!=0)
        # H_ori = np.arctan2(H_y, H_x)
        # H_ori = (H_ori + 2*np.pi)%(2*np.pi)
        # H_len = np.hypot(H_x, H_y)
        H_activ = np.histogram2d(x, y, bins=[x_bins, y_bins], weights=activ)[0]
        H_activ = np.divide(H_activ, H_count, out=np.zeros_like(H_activ), where=H_count!=0)
        
        mask = np.zeros_like(H_count)
        mask[H_count < 15] = 1
        H_activ = np.ma.masked_where(mask == 1, H_activ)
        H_activ = np.ma.filled(H_activ, 0) # backfill masked values with zero
        H_activ = scipy.ndimage.gaussian_filter(H_activ, sigma=1) # smoothen

        # mask = np.zeros_like(H_count)
        # mask[H_count < 1] = 1
        # H_ori = np.ma.masked_where(mask == 1, H_ori)
        # H_len = np.ma.masked_where(mask == 1, H_len)
        # # H_activ = np.ma.masked_where(mask == 1, H_activ) # no mask since bottom normalized + gaussian smoothened

        x_bins = np.linspace(0, x_max, num_bins)
        y_bins = np.linspace(0, y_max, num_bins)
        X,Y = np.meshgrid(x_bins, y_bins)
        # U = np.cos(H_ori.T)*H_len.T
        # V = np.sin(H_ori.T)*H_len.T
        # U = np.ma.masked_where(mask.T == 1, U)
        # V = np.ma.masked_where(mask.T == 1, V)
        # print(H_activ.min())

        # norm = mpl.colors.Normalize(vmin=0, vmax=H_activ.max()) # norm by ori
        H_activ += 0.001 # shift up to avoid log(0)
        norm = mpl.colors.LogNorm(vmin=H_activ.min(), vmax=H_activ.max()) # lognorm by ori
        # norm = mpl.colors.Normalize(vmin=0, vmax=1) # norm by node
        im = axes.pcolormesh(X, Y, H_activ.T, cmap='plasma', norm=norm)

        # norm = mpl.colors.Normalize(vmin=H_len.min(), vmax=H_len.max())
        # q = axes.quiver(X, Y, U, V, H_len.T, cmap='gray_r', norm=norm, pivot='middle')


    elif '_tuning' in plot_type:
        act_index = int(plot_type[7])+3 # skip x/y/o
        plot_index = int(plot_type[8])

        x = ag_data[:,:,0]
        y = ag_data[:,:,1]
        ori = ag_data[:,:,2]
        activ = ag_data[:,:,act_index]
        orient_range = np.linspace(-np.pi, np.pi, 16+1)
        # print(ori.min(), ori.max())

        # # to flip on/off SE quarter
        # print(x.shape)
        # x_ind_to_mask = np.where(x[:,0] > 500)[0]
        # y_ind_to_mask = np.where(y[:,0] < 500)[0]
        # indices_to_mask = np.union1d(x_ind_to_mask, y_ind_to_mask)
        # # x without those indices (for array[:,0:])
        # x = np.delete(x, indices_to_mask, axis=0)
        # y = np.delete(y, indices_to_mask, axis=0)
        # ori = np.delete(ori, indices_to_mask, axis=0)
        # activ = np.delete(activ, indices_to_mask, axis=0)
        # # # x with only those indices (for array[:,0:])
        # # x = x[indices_to_mask,:]
        # # y = y[indices_to_mask,:]
        # # ori = ori[indices_to_mask,:]
        # # activ = activ[indices_to_mask,:]
        # print(x.shape)

        x = x.flatten()
        y = y.flatten()
        ori = ori.flatten()
        activ = activ.flatten()

        # min_activ = activ.min()
        # max_activ = activ.max()
        # activ = (activ - min_activ) / (max_activ - min_activ) # norm to [0,1]
        activ = scipy.stats.rankdata(activ, method='average') # 0 is highest

        fig_hist, axes_hist = plt.subplots(figsize=(4,4))

        if plot_index == 0:
            pt_target = np.array(eval(envconf["RESOURCE_POS"]))
            x_label = 'Goal-Direction'
        elif plot_index == 1: # self-SW vector
            pt_target = np.array([0,0])
            x_label = 'SW Corner'
        elif plot_index == 2: # self-SE vector
            pt_target = np.array([1000,0])
            x_label = 'SE Corner'
        elif plot_index == 3: # self-NE vector
            pt_target = np.array([1000,1000])
            x_label = 'NE Corner'
        elif plot_index == 4: # self-NW vector
            pt_target = np.array([0,1000])
            x_label = 'NW Corner'

        pt_target[1] = 1000 - pt_target[1]
        pt_self = np.array([x,1000 - y])
        pt_target = np.repeat(pt_target[np.newaxis,:], pt_self.shape[1], axis=0).transpose()
        disp = pt_self - pt_target

        angle_to_target = np.arctan2(disp[1,:], disp[0,:]) + np.pi # [0, 2pi]
        corr_patch_angle_diff = angle_to_target - ori # [-2pi, 2pi]
        corr_patch_angle_diff = corr_patch_angle_diff % (2*np.pi) - np.pi # [-pi, pi]

        # # for only positions > 50 away from target (outside the patch boundary)
        # dist = np.linalg.norm(disp, axis=0)
        # m = np.ma.masked_less(dist, 1000)
        # corr_patch_angle_diff = corr_patch_angle_diff.transpose().flatten()
        # # remove masked values from array
        # activ = activ[~m.mask]
        # corr_patch_angle_diff = corr_patch_angle_diff[~m.mask]

        orient_range = np.linspace(-np.pi, np.pi, 16+1)
        hist_activ = np.histogram(corr_patch_angle_diff, bins=orient_range, weights=activ)[0]
        norm_activ = (hist_activ - hist_activ.min()) / (hist_activ.max() - hist_activ.min())
        # hist_activ = np.histogram(corr_patch_angle_diff, bins=orient_range)[0]
        # norm_activ = hist_activ

        fig_hist, axes_hist = plt.subplots(figsize=(2,2))
        axes_hist.plot(orient_range[:-1], norm_activ)
        axes_hist.set_xticks(np.linspace(-np.pi,np.pi,3))
        axes_hist.set_xticklabels([r'-$\pi$', '$0$', r'$\pi$'])
        axes_hist.set_yticks([0,1])
        # axes_hist.set_xlabel(x_label)
        axes_hist.set_title(x_label)
        axes_hist.set_ylabel('Normalized Activity')
        plt.tight_layout()
        plt.savefig(fr'{save_data_dir}/action_maps/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_traj_hist{plot_type}.png', dpi=dpi)
        plt.close()
        return

    elif '_interfaces' in plot_type :
        num_bins = 50
        x_bins = np.linspace(0, x_max, num_bins+1)
        y_bins = np.linspace(0, y_max, num_bins+1)

        H_count,_,_ = np.histogram2d(x, y, bins=[x_bins, y_bins])
        H_x,_,_ = np.histogram2d(x, y, bins=[x_bins, y_bins], weights=np.cos(ori))
        H_x = np.divide(H_x, H_count, out=np.zeros_like(H_x), where=H_count!=0)
        H_y,_,_ = np.histogram2d(x, y, bins=[x_bins, y_bins], weights=np.sin(ori))
        H_y = np.divide(H_y, H_count, out=np.zeros_like(H_y), where=H_count!=0)
        H_x = scipy.ndimage.gaussian_filter(H_x, sigma=1) # smoothen
        H_y = scipy.ndimage.gaussian_filter(H_y, sigma=1) # smoothen
        H_ori = np.arctan2(H_y, H_x)
        H_ori = (H_ori + 2*np.pi)%(2*np.pi)
        H_len = np.hypot(H_x, H_y)
        # H_activ = np.histogram2d(x, y, bins=[x_bins, y_bins], weights=activ)[0]
        # H_activ = np.divide(H_activ, H_count, out=np.zeros_like(H_activ), where=H_count!=0)

        mask = np.zeros_like(H_count)
        mask[H_count < 100] = 1
        H_ori = np.ma.masked_where(mask == 1, H_ori)
        H_len = np.ma.masked_where(mask == 1, H_len)
        # H_activ = np.ma.filled(H_activ, 0) # backfill masked values with zero
        # H_ori = scipy.ndimage.gaussian_filter(H_ori, sigma=1) # smoothen
        # H_len = scipy.ndimage.gaussian_filter(H_len, sigma=1) # smoothen

        x_bins = np.linspace(0, x_max, num_bins)
        y_bins = np.linspace(0, y_max, num_bins)
        X,Y = np.meshgrid(x_bins, y_bins)

        U = np.cos(H_ori.T)*H_len.T
        V = np.sin(H_ori.T)*H_len.T
        U = np.ma.masked_where(mask.T == 1, U)
        V = np.ma.masked_where(mask.T == 1, V)

        norm = mpl.colors.Normalize(vmin=H_len.min(), vmax=H_len.max())
        # H_len += 0.0001 # shift up to avoid log(0)
        # norm = mpl.colors.LogNorm(vmin=H_len.min(), vmax=H_len.max())

        q = axes.quiver(X, Y, U, V, H_len.T, cmap='gray_r', norm=norm, pivot='middle')

        im = axes.contourf(X,Y, H_len.T, norm=norm, cmap='viridis', alpha=.6)
        # levs = np.linspace(H_len.min(), H_len.max(), 6)
        # im = axes.contourf(X,Y, H_len.T, levs, cmap='viridis', alpha=.6)
        # im = axes.pcolormesh(X, Y, H_len.T, cmap='viridis', norm=norm, alpha=.6)

    elif plot_type == '_finalheat':
        x = ag_data[:,-1,0]
        y = ag_data[:,-1,1]
        
        H,_,_ = np.histogram2d(x, y, bins=[x_bins, y_bins])
        H = scipy.ndimage.gaussian_filter(H, sigma=1)

        X,Y = np.meshgrid(x_bins, y_bins)
        # axes.pcolormesh(X, Y, H.T, cmap='viridis')
        norm = mpl.colors.LogNorm(vmin=1, vmax=H.max())
        axes.pcolormesh(X, Y, H.T+1, cmap='viridis', norm=norm)

    radius = int(envconf["RADIUS_RESOURCE"])
    x,y = tuple(eval(envconf["RESOURCE_POS"]))
    axes.add_patch( plt.Circle((x, height-y), radius, edgecolor='k', fill=False, zorder=1) )

    if extra == '':
        save_name = fr'{save_data_dir}/action_maps/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_traj'
    else:
        save_name = fr'{save_data_dir}/action_maps/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{extra}_traj'

    axes.set_xticklabels([])
    axes.set_yticklabels([])
    plt.tight_layout()
    plt.savefig(fr'{save_name}_hist{plot_type}.png', dpi=dpi)
    plt.close()

    # print(fr'{save_name}_hist{plot_type}.png')

    if plot_type == '_dirent':
        return np.mean(H_mask), np.min(H_mask), np.max(H_mask)


def plot_traj_vecfield_perturb_div(exp_name, gen_ext, space_step, orient_step, timesteps, plot_type='', base_cond='', perturb_cond='', mask_cond='', dpi=50):
    print(f'plotting traj vector field - {exp_name}: {plot_type} + {base_cond}/{perturb_cond}/{mask_cond} @ {dpi} dpi')

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)

    width, height = tuple(eval(envconf["ENV_SIZE"]))
    x_min, x_max = 0, width
    y_min, y_max = 0, height
    orient_range = np.arange(0, 2*np.pi, orient_step)

    fig, axes = plt.subplots() 
    axes.set_xlim(0, x_max)
    axes.set_ylim(0, y_max)
    h,w = 8,8
    l,r,t,b = fig.subplotpars.left, fig.subplotpars.right, fig.subplotpars.top, fig.subplotpars.bottom
    fig.set_size_inches( float(w)/(r-l) , float(h)/(t-b) )

    if base_cond == '':
        save_name_baseline = fr'{data_dir}/traj_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_cen_e1'
    else:
        save_name_baseline = fr'{data_dir}/traj_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_cen_e1_{base_cond}'
    with open(save_name_baseline+'.bin', 'rb') as f:
        ag_data_base = pickle.load(f)
    # print(ag_data.shape)

    delay = 25
    x_base = ag_data_base[::10,delay:,0].flatten()
    y_base = ag_data_base[::10,delay:,1].flatten()
    ori_base = ag_data_base[::10,delay:,2].flatten()

    if perturb_cond == '':
        save_name_perturb = fr'{data_dir}/traj_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_cen_e1'
    else:
        save_name_perturb = fr'{data_dir}/traj_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_cen_e1_{perturb_cond}'
    with open(save_name_perturb+'.bin', 'rb') as f:
        ag_data_perturb = pickle.load(f)
    # print(ag_data.shape)

    x_perturb = ag_data_perturb[::10,delay:,0].flatten()
    y_perturb = ag_data_perturb[::10,delay:,1].flatten()
    ori_perturb = ag_data_perturb[::10,delay:,2].flatten()

    # print(ag_data_base == ag_data_perturb)

    num_bins = 51
    # num_bins = 101
    x_bins = np.linspace(0, x_max, num_bins)
    y_bins = np.linspace(0, y_max, num_bins)
    # organize base
    hitx = np.digitize(x_base, x_bins[1:])
    hity = np.digitize(y_base, y_bins[1:])
    hitbins = list(zip(hitx, hity))
    ori_and_bins_base = list(zip(ori_base, hitbins))
    # organize perturb
    hitx = np.digitize(x_perturb, x_bins[1:])
    hity = np.digitize(y_perturb, y_bins[1:])
    hitbins = list(zip(hitx, hity))
    ori_and_bins_perturb = list(zip(ori_perturb, hitbins))

    H = np.zeros((num_bins-1, num_bins-1))

    if plot_type == 'JS':
        norm = mpl.colors.LogNorm(vmin=.01, vmax=1)
        # norm = mpl.colors.Normalize(vmin=0, vmax=0.5)
        for i in range(num_bins-1):
            for j in range(num_bins-1):
                bin_ori_base = [ori for (ori,bin) in ori_and_bins_base if bin == (i,j)]
                bin_ori_perturb = [ori for (ori,bin) in ori_and_bins_perturb if bin == (i,j)]
                if bin_ori_base and bin_ori_perturb:
                    x = np.histogram(bin_ori_base, bins=orient_range)[0]
                    y = np.histogram(bin_ori_perturb, bins=orient_range)[0]
                    H[i,j] = calc_JSdiv(x,y)
                    # print('y')
                # else:
                #     print(i,j,'no data')
    # elif plot_type == 'KL':
    #     norm = mpl.colors.LogNorm(vmin=.01, vmax=10)
    #     for i in range(num_bins-1):
    #         for j in range(num_bins-1):
    #             bin_ori_base = [ori for (ori,bin) in ori_and_bins_base if bin == (i,j)]
    #             bin_ori_perturb = [ori for (ori,bin) in ori_and_bins_perturb if bin == (i,j)]
    #             if bin_ori_base and bin_ori_perturb:
    #                 x = np.histogram(bin_ori_base, bins=orient_range)[0]
    #                 y = np.histogram(bin_ori_perturb, bins=orient_range)[0]
    #                 H[i,j] = calc_KLdiv(x,y)

    # norm = mpl.colors.Normalize(vmin=0, vmax=1)
    X,Y = np.meshgrid(x_bins, y_bins)
    im = axes.pcolormesh(X, Y, H.T, cmap='viridis', norm=norm)
    # cbar = axes.figure.colorbar(im, ax=axes, fraction=0.046, pad=0.04, extend='both')

    radius = int(envconf["RADIUS_RESOURCE"])
    x,y = tuple(eval(envconf["RESOURCE_POS"]))
    axes.add_patch( plt.Circle((x, height-y), radius, edgecolor='k', fill=False, zorder=1) )

    if mask_cond == '':
        # mask edges (100 from each edge) + patch vicinity (100 from center)
        mask = np.zeros([num_bins-1, num_bins-1])
        mask[:5,:] = 1
        mask[:,:5] = 1
        mask[46:,:] = 1
        mask[:,46:] = 1
        mask[16:25,26:35] = 1
        H_mask = np.ma.array(H, mask=mask)
    elif mask_cond == 'no_patch':
        mask = np.zeros([num_bins-1, num_bins-1])
        mask[:5,:] = 1
        mask[:,:5] = 1
        mask[46:,:] = 1
        mask[:,46:] = 1
        H_mask = np.ma.array(H, mask=mask)
    elif mask_cond == 'patch_only':
        mask = np.zeros([num_bins-1, num_bins-1])
        mask[:16,:] = 1
        mask[25:,:] = 1
        mask[:,:26] = 1
        mask[:,35:] = 1
        H_mask = np.ma.array(H, mask=mask)
    elif mask_cond == 'near_patch':
        mask = np.zeros([num_bins-1, num_bins-1])
        mask[:11,:] = 1
        mask[30:,:] = 1
        mask[:,:21] = 1
        mask[:,40:] = 1
        H_mask = np.ma.array(H, mask=mask)
    else:
        print('no valid mask condition specified')
    axes.set_title(f'Masked + Mean JS Divergence: {np.mean(H_mask):.3f}')
    axes.set_xticklabels([])
    axes.set_yticklabels([])
    plt.tight_layout()

    if base_cond == '': base_cond = 'base'
    if perturb_cond == '': perturb_cond = 'base'
    plt.savefig(fr'{data_dir}/action_maps/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{base_cond}_{perturb_cond}_traj_hist_{plot_type}{mask_cond}.png', dpi=dpi)
    plt.close()

    print(f'min/mean/max: {np.min(H_mask):.3f}/{np.mean(H_mask):.3f}/{np.max(H_mask):.3f}')

    return np.mean(H_mask), np.min(H_mask), np.max(H_mask)


# -------------------------- activfield -------------------------- #

def print_activfield_results():
    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    with open(fr'{data_dir}/traj_matrices/activfield.bin', 'rb') as f:
        data = pickle.load(f)

    ## spatial specificity regardless of ori
    for name in data:
        print(f'Name: {name}')
        for call in data[name]:
            # if '(8' in call:
            if '(0' in call:
                print(f'  {call}: {data[name][call]}')



def plot_activfield_results(plot_type, data_type=''):
    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    with open(fr'{data_dir}/traj_matrices/activfield.bin', 'rb') as f:
        data = pickle.load(f)

    ## ori specificity wrt goal

    # for act_index in range(6)
    # data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    # env_path = fr'{data_dir}/{name}/.env'
    # envconf = de.dotenv_values(env_path)

    # width, height = tuple(eval(envconf["ENV_SIZE"]))
    # x_min, x_max = 0, width
    # y_min, y_max = 0, height
    x_max = 1000
    y_max = 1000
    # orient_range = np.arange(0, 2*np.pi, orient_step)
    patch_x, patch_y = 400, 600 # center of patch
    patch_radius = 50 # radius of patch

    axes_pos = [(0,0), (1,0), (2,0),
                (0,1), (1,1), (2,1),
                (0,2), (1,2), (2,2),
                (0,3), (1,3), (2,3) ]

    if plot_type == 'ori_x_goal':
        for act_index in range(4):
        # for act_index in [4,5]:

            fig, axes = plt.subplots(3,4) 
            h,w = 12,16
            l,r,t,b = fig.subplotpars.left, fig.subplotpars.right, fig.subplotpars.top, fig.subplotpars.bottom
            fig.set_size_inches( float(w)/(r-l) , float(h)/(t-b) )

            ori_colors = np.array(list(range(8)))
            norm = mpl.colors.Normalize(vmin=0, vmax=8)

            for i, name in enumerate(data):
                # print(f'Name: {name}')
                axes[axes_pos[i]].add_patch( plt.Circle((patch_x, patch_y), patch_radius, edgecolor='k', fill=False) )

                pts = []
                for call in data[name]:
                    if f'{act_index})' in call and '(8' not in call:
                        # print(f'  {call}: {data[name][call]}')
                        # print(len(data[name][call]))

                        if data_type == 'max':
                            pts.append((data[name][call][1],data[name][call][2])) # max
                        elif data_type == 'thresh_p15':
                            pts.append((data[name][call][4],data[name][call][5])) # avg above thresh max*.15
                        elif data_type == 'thresh_p25':
                            pts.append((data[name][call][8],data[name][call][9])) # avg above thresh max*.25
                        elif data_type == 'thresh_p50':
                            pts.append((data[name][call][12],data[name][call][13])) # avg above thresh max*.50

                if 'thresh' in data_type:
                    pts = np.array(pts)*1000/50 # scale from bins to grid
                else:
                    pts = np.array(pts)
                axes[axes_pos[i]].scatter(pts[:,0], pts[:,1], c=ori_colors, cmap='hsv', norm=norm, s=100, alpha=1)
                axes[axes_pos[i]].set_xlim(0, x_max)
                axes[axes_pos[i]].set_ylim(0, y_max)
                axes[axes_pos[i]].set_title(name, fontsize=6)
            fig.suptitle(f'Data Type {data_type} || Action {act_index}')

            # axes[0,0].legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
            plt.tight_layout()
            plt.savefig(fr'{data_dir}/action_maps/activfield_results_{plot_type}_{data_type}_a{act_index}.png', dpi=50)
            plt.close()

    elif plot_type == 'histo_dist':

        fig, axes = plt.subplots(3,4) 
        h,w = 12,16
        l,r,t,b = fig.subplotpars.left, fig.subplotpars.right, fig.subplotpars.top, fig.subplotpars.bottom
        fig.set_size_inches( float(w)/(r-l) , float(h)/(t-b) )

        y_maxs = []
        for i, name in enumerate(data):

            dists = []
            for call in data[name]:
                if '(8' not in call:
                    x = float(data[name][call][4])*1000/50 # thresh_p15 + rescale
                    y = float(data[name][call][5])*1000/50
                    dist_to_patch = np.sqrt((x - patch_x)**2 + (y - patch_y)**2)
                    dists.append(dist_to_patch)
            dists = np.array(dists)

            axes[axes_pos[i]].hist(dists, bins=np.linspace(0, 1000, 20), color='blue', alpha=0.5)
            axes[axes_pos[i]].set_xlim(0, 800)
            y_maxs.append(axes[axes_pos[i]].get_ylim()[1])

            median = np.median(dists)
            axes[axes_pos[i]].axvline(median, color='red', linestyle='--', linewidth=1)

            hist, _ = np.histogram(dists, bins=np.linspace(0, 1000, 20), density=True)
            hist = hist[hist > 0]  # ignore empty bins
            entropy = -np.sum(hist * np.log(hist))
            axes[axes_pos[i]].annotate(f'Entropy: {entropy:.2f}', xy=(0.7, 0.9), xycoords='axes fraction', ha='center', fontsize=16)
        
        max_y = max(y_maxs)
        for ax in axes_pos:
            axes[ax].set_ylim(0, max_y)

        plt.tight_layout()
        plt.savefig(fr'{data_dir}/action_maps/activfield_results_{plot_type}.png', dpi=50)
        plt.close()

    elif plot_type == 'histo_ori':

        fig, axes = plt.subplots(3,4) 
        h,w = 12,16
        l,r,t,b = fig.subplotpars.left, fig.subplotpars.right, fig.subplotpars.top, fig.subplotpars.bottom
        fig.set_size_inches( float(w)/(r-l) , float(h)/(t-b) )

        y_maxs = []
        for i, name in enumerate(data):

            oris = []
            for call in data[name]:
                if '(8' not in call:
                    x = float(data[name][call][4])*1000/50 # thresh_p15 + rescale
                    y = float(data[name][call][5])*1000/50
                    ori_to_patch = np.arctan2(y - patch_y, x - patch_x) + np.pi
                    oris.append(ori_to_patch)
            oris = np.array(oris)

            axes[axes_pos[i]].hist(oris, bins=np.linspace(0, 2*np.pi, 8), color='blue', alpha=0.5)
            axes[axes_pos[i]].set_xlim(0, 2*np.pi)
            y_maxs.append(axes[axes_pos[i]].get_ylim()[1])

            hist, _ = np.histogram(oris, bins=np.linspace(0, 2*np.pi, 8), density=True)
            hist = hist[hist > 0]  # ignore empty bins
            entropy = -np.sum(hist * np.log(hist))
            axes[axes_pos[i]].annotate(f'Entropy: {entropy:.2f}', xy=(0.7, 0.9), xycoords='axes fraction', ha='center', fontsize=16)

        # set ylim relative to highest bin of all subplots
        max_y = max(y_maxs)
        for ax in axes_pos:
            axes[ax].set_ylim(0, max_y)

        plt.tight_layout()
        plt.savefig(fr'{data_dir}/action_maps/activfield_results_{plot_type}.png', dpi=50)
        plt.close()




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
    x = (x-np.pi) % (2*np.pi) - np.pi
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



def plot_agent_orient_corr(exp_name, gen_ext, space_step, orient_step, timesteps, rank='cen', eye=True, extra='', archive=False, dpi=None):
    print(f'plotting corr - {exp_name} @ {dpi} dpi')

    save_data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    if archive:
        data_dir = Path(__file__).parent.parent / r'data/simulation_data/archive - ISBDDP/'
    else:
        data_dir = save_data_dir

    if extra == '':
        save_name = fr'{save_data_dir}/traj_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e{int(eye)}'
    else:
        save_name = fr'{save_data_dir}/traj_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e{int(eye)}{extra}'
    if not os.path.exists(save_name+'.bin'):
        print(f'no data found for {save_name}')
        return 0,0,0,0
    with open(save_name+'.bin', 'rb') as f:
        ag_data = pickle.load(f)
    save_name = fr'{save_data_dir}/corrs/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e{int(eye)}'

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
    corr_init_angle_diff_scaled = (corr_init_angle_diff - np.pi) % (2*np.pi) - np.pi # [-pi/2, pi/2]
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
    corr_patch_angle_diff_scaled = (corr_patch_angle_diff - np.pi) % (2*np.pi) - np.pi # [-pi/2, pi/2]
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
    plt.savefig(fr'{save_name}{extra}_corr_orient_{dpi}.png', dpi=dpi)
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
        [800, 200, np.pi], #BR-W
        # [800, 300, 0], #BR-N
        # [800, 900, 3*np.pi/2], #TR-S
        [800, 900, np.pi/2], #TR-W
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
    plt.savefig(fr'{save_name}{extra}_corr_trajs_init_{dpi}.png', dpi=dpi)
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
        [800, 200, np.pi], #BR-W
        # [800, 300, 0], #BR-N
        # [800, 900, 3*np.pi/2], #TR-S
        [800, 900, np.pi/2], #TR-W
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
    plt.savefig(fr'{save_name}{extra}_corr_trajs_patch_{dpi}.png', dpi=dpi)
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
    plt.savefig(fr'{save_name}{extra}_corr_polar_init_{dpi}.png', dpi=dpi)
    # plt.show()
    plt.close()

    ### directedness ###

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
    plt.savefig(fr'{save_name}{extra}_corr_polar_patch_{dpi}.png', dpi=dpi)
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


def log_ray_boundary_collision(agent, action, last_moves, boundary_endpts, phi_angle_diff):
    coll_output = 0
    if action not in last_moves:
        # compare last two observations
        vis_diff = [x != y for x,y in zip(agent.vis_field, agent.last_vis_field)]
        vis_diff_idx = np.where(vis_diff)[0]
        intersect = False

        if len(vis_diff_idx) > 0:
            corner_intersecting_pts = []
            for i,phi in enumerate(agent.phis):
                # print(f'phis: {i}, {phi}')

                for pt_idx, pt in enumerate(boundary_endpts):
                    # print(f'boundary_endpts: {pt_idx}, {pt}')

                    vec_between = pt - agent.pt_eye
                    angle_bw = supcalc.angle_bw_vis(agent.vec_self_dir, vec_between, agent.radius, np.linalg.norm(vec_between))

                    # single ray collision --> find nearest corner
                    if len(vis_diff_idx) == 1 and i == vis_diff_idx[0]:
                        if np.abs(angle_bw-phi) / phi_angle_diff < 1:
                            intersect = True
                            coll_output = (i+1)*10 + (pt_idx+1)

                    # check for ellipse - looser query (single/multi rays), stricter criteria (15% proximity)
                    if np.abs(angle_bw-phi) / phi_angle_diff < 0.15:
                        corner_intersecting_pts.append((i+1)*10 + (pt_idx+1))

            # for 2 corners intersecting --> ellipse
            if len(corner_intersecting_pts) == 2:
                intersect = True
                coll_output = corner_intersecting_pts[0]*100 + corner_intersecting_pts[1]

            # no single ray coll + no ellipse
            elif intersect == False:
                coll_output = 100

    # fading memory of last moves
    last_moves.append(action)
    if len(last_moves) > 2:
        last_moves.pop(0)

    return last_moves, coll_output


def plot_agent_ray_boundary_collision_stats(exp_name, gen_ext, space_step, orient_step, timesteps, dpi=None):
    print(f'plotting collision stats - {exp_name} @ {dpi} dpi')

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)

    load_name = fr'{data_dir}/traj_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_cen_e1'
    if not os.path.exists(load_name+'.bin'):
        print(f'no data found for {load_name}')
        return 0,0,0,0
    else:
        with open(load_name+'.bin', 'rb') as f:
            ag_data = pickle.load(f)
    save_name = fr'{data_dir}/corrs/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_cen_e1'

    # cut init
    delay = 25
    x = ag_data[:,delay:,0]
    y = ag_data[:,delay:,1]
    ori = ag_data[:,delay:,2]
    action = ag_data[:,delay:,3]
    action = np.abs(action)
    coll_output = ag_data[:,delay:,4]

    # print(x.min(), x.max(), y.min(), y.max(), ori.min(), ori.max())

    # cut when reached patch
    patch_radius = int(envconf["RADIUS_RESOURCE"])
    patch_x, patch_y = tuple(eval(envconf["RESOURCE_POS"]))
    patch_y = 1000 - patch_y
    extra_buffer = 25

    spike_counts = {
        'SC': 0,
        # 'SC: 2 Walls': 0,
        # 'SC: 3 Walls': 0,
        'SC: NW': 0,
        'SC: NE': 0,
        'SC: SE': 0,
        'SC: SW': 0,
        'SC: Ray 1': 0,
        'SC: Ray 2': 0,
        'SC: Ray 3': 0,
        'SC: Ray 4': 0,
        'SC: Ray 5': 0,
        'SC: Ray 6': 0,
        'SC: Ray 7': 0,
        'SC: Ray 8': 0,
        'DC': 0,
        'DC-Adj: NW-NE': 0,
        'DC-Adj: NE-SE': 0,
        'DC-Adj: SE-SW': 0,
        'DC-Adj: SW-NW': 0,
        'DC-Opp: NW-SE': 0,
        'DC-Opp: NE-SW': 0,
        'DC: Rays 1 x 2': 0,
        'DC: Rays 1 x 3': 0,
        'DC: Rays 1 x 4': 0,
        'DC: Rays 1 x 5': 0,
        'DC: Rays 1 x 6': 0,
        'DC: Rays 1 x 7': 0,
        'DC: Rays 1 x 8': 0,
        'DC: Rays 2 x 3': 0,
        'DC: Rays 2 x 4': 0,
        'DC: Rays 2 x 5': 0,
        'DC: Rays 2 x 6': 0,
        'DC: Rays 2 x 7': 0,
        'DC: Rays 2 x 8': 0,
        'DC: Rays 3 x 4': 0,
        'DC: Rays 3 x 5': 0,
        'DC: Rays 3 x 6': 0,
        'DC: Rays 3 x 7': 0,
        'DC: Rays 3 x 8': 0,
        'DC: Rays 4 x 5': 0,
        'DC: Rays 4 x 6': 0,
        'DC: Rays 4 x 7': 0,
        'DC: Rays 4 x 8': 0,
        'DC: Rays 5 x 6': 0,
        'DC: Rays 5 x 7': 0,
        'DC: Rays 5 x 8': 0,
        'DC: Rays 6 x 7': 0,
        'DC: Rays 6 x 8': 0,
        'DC: Rays 7 x 8': 0,
        # 'C>2': 0,
    }

    # isi = {
    #     'All': [],
    #     'SC': [],
    #     'SC: NW': [],
    #     'SC: NE': [],
    #     'SC: SE': [],
    #     'SC: SW': [],
    #     'SC: Ray 1': [],
    #     'SC: Ray 2': [],
    #     'SC: Ray 3': [],
    #     'SC: Ray 4': [],
    #     'SC: Ray 5': [],
    #     'SC: Ray 6': [],
    #     'SC: Ray 7': [],
    #     'SC: Ray 8': [],
    #     'DC': [],
    # }

    spike_locs = {
        'SC': [],
        # 'SC: 2 Walls': [],
        # 'SC: 3 Walls': [],
        'SC: NW': [],
        'SC: NE': [],
        'SC: SE': [],
        'SC: SW': [],
        'SC: Ray 1': [],
        'SC: Ray 2': [],
        'SC: Ray 3': [],
        'SC: Ray 4': [],
        'SC: Ray 5': [],
        'SC: Ray 6': [],
        'SC: Ray 7': [],
        'SC: Ray 8': [],
        'DC': [],
        'DC-Adj: NW-NE': [],
        'DC-Adj: NE-SE': [],
        'DC-Adj: SE-SW': [],
        'DC-Adj: SW-NW': [],
        'DC-Opp: NW-SE': [],
        'DC-Opp: NE-SW': [],
        'DC: Rays 1 x 2': [],
        'DC: Rays 1 x 3': [],
        'DC: Rays 1 x 4': [],
        'DC: Rays 1 x 5': [],
        'DC: Rays 1 x 6': [],
        'DC: Rays 1 x 7': [],
        'DC: Rays 1 x 8': [],
        'DC: Rays 2 x 3': [],
        'DC: Rays 2 x 4': [],
        'DC: Rays 2 x 5': [],
        'DC: Rays 2 x 6': [],
        'DC: Rays 2 x 7': [],
        'DC: Rays 2 x 8': [],
        'DC: Rays 3 x 4': [],
        'DC: Rays 3 x 5': [],
        'DC: Rays 3 x 6': [],
        'DC: Rays 3 x 7': [],
        'DC: Rays 3 x 8': [],
        'DC: Rays 4 x 5': [],
        'DC: Rays 4 x 6': [],
        'DC: Rays 4 x 7': [],
        'DC: Rays 4 x 8': [],
        'DC: Rays 5 x 6': [],
        'DC: Rays 5 x 7': [],
        'DC: Rays 5 x 8': [],
        'DC: Rays 6 x 7': [],
        'DC: Rays 6 x 8': [],
        'DC: Rays 7 x 8': [],
        # 'C>2': [],
    }

    if os.path.exists(fr'{save_name}_spikecounts_dict.p'):
        print('exists, skip counting')
        spike_counts = pickle.load(open(fr'{save_name}_spikecounts_dict.p','rb'))
    else:

        unique_counts = []
        num_reach_patch = 0
        runs, timesteps = x.shape
        for run in range(runs):

            # count all
            # spike_counts,spike_locs = count_spikes(x,y,ori,action,coll_output,run,timesteps+1,spike_counts,spike_locs)

            # if x,y within patch radius, mask rest of array, append to accumulating list
            for t in range(timesteps):
                # stop when within patch radius
                if np.linalg.norm([x[run,t]-patch_x, y[run,t]-patch_y]) < patch_radius:
                # stop when within patch radius (+buffer)
                # if np.linalg.norm([x[run,t]-patch_x, y[run,t]-patch_y]) < patch_radius + extra_buffer:
                    spike_counts,spike_locs, unique_count = count_spikes(x,y,ori,action,coll_output,run,t,spike_counts,spike_locs)
                    unique_counts.append(unique_count)
                    num_reach_patch += 1
                    break
                elif t == timesteps-1:
                    spike_counts,spike_locs, unique_count = count_spikes(x,y,ori,action,coll_output,run,t+1,spike_counts,spike_locs)
                    unique_counts.append(unique_count)
                    break
                else:
                    pass
        
        # pickle.dump(spike_counts, open(fr'{save_name}_spikecounts_dict.p', 'wb'))
    # print(spike_counts)
    # import pprint
    # pprint.pprint(spike_counts)
    
    unique_counts = np.array(unique_counts)
    print(f'# unq states to reach patch: {int(np.median(unique_counts)), np.mean(unique_counts).round(2)} / # reach patch: {num_reach_patch} / total: {runs}')

    # transition matrix + markov chain
    tran_matrix, states = estimate_transition_matrix(coll_output)
    visualize_transition_matrix(tran_matrix, states, save_name=fr'{save_name}_tran_matrix.png', dpi=dpi)
    # visualize_markov_chain(tran_matrix, states, save_name=fr'{save_name}_markov.png', dpi=dpi)

    # # trim states not used as often
    # count_matrix, states = labeled_histo(coll_output)
    # states_trimmed = []
    # for s,c in zip(states,count_matrix):
    #     if c > .01:
    #         print(f'{int(s)}: {round(c,4)}')
    #         states_trimmed.append(s)
    # tran_matrix = estimate_transition_matrix_trimmed(coll_output, states_trimmed)
    # visualize_transition_matrix(tran_matrix, states_trimmed, save_name=fr'{save_name}_tran_matrix_trimmed.png', dpi=dpi)
    # # visualize_markov_chain(tran_matrix, states_trimmed, save_name=fr'{save_name}_markov.png', dpi=dpi)

    # spike count histo
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('plasma')
    colors = cmap(np.linspace(0, 1, len(spike_counts)))
    plt.bar(spike_counts.keys(), spike_counts.values(), color=colors, alpha=.7)
    plt.xticks(rotation=45, ha='right')
    plt.title(f'{exp_name} | med # unq states to reach patch: {int(np.median(unique_counts))}')
    plt.xlabel('Spike Type')
    plt.ylabel('Count')
    # for line in [2.5, 6.5, 14.5, 21.5]:
    for line in [.5, 4.5, 13.5, 19.5]:
        plt.axvline(x = line, color='gray', linestyle='--', linewidth=0.5)
    for line in [12.5]:
        plt.axvline(x = line, color='black', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    # plt.show()
    plt.savefig(fr'{save_name}_spike_counts.png', dpi=dpi)
    plt.close()

    # plt.figure(figsize=(10, 6))
    # cmap = plt.get_cmap('plasma')
    # ks,vs = [],[]
    # for k,v in spike_counts.items():
    #     if 'SC' in k:
    #         ks.append(k)
    #         vs.append(v)
    # colors = cmap(np.linspace(0, 1, len(ks)))
    # plt.bar(ks, vs, color=colors, alpha=.7)
    # plt.xticks(rotation=45, ha='right')
    # plt.title(exp_name)
    # plt.xlabel('Spike Type')
    # plt.ylabel('Count')
    # for line in [.5, 4.5]:
    #     plt.axvline(x = line, color='gray', linestyle='--', linewidth=0.5)
    # plt.tight_layout()
    # plt.savefig(fr'{save_name}_spike_counts_onlySC.png', dpi=dpi)
    # plt.close()

    # plt.figure(figsize=(10, 6))
    # cmap = plt.get_cmap('plasma')
    # ks,vs = [],[]
    # for k,v in spike_counts.items():
    #     if 'DC' in k:
    #         ks.append(k)
    #         vs.append(v)
    # colors = cmap(np.linspace(0, 1, len(ks)))
    # plt.bar(ks, vs, color=colors, alpha=.7)
    # plt.xticks(rotation=45, ha='right')
    # plt.title(exp_name)
    # plt.xlabel('Spike Type')
    # plt.ylabel('Count')
    # for line in [.5, 2.5]:
    #     plt.axvline(x = line, color='gray', linestyle='--', linewidth=0.5)
    # plt.tight_layout()
    # plt.savefig(fr'{save_name}_spike_counts_onlyDC.png', dpi=dpi)
    # plt.close()

    # Heatmaps
    res_data = np.zeros((1,1,3))
    width, height = tuple(eval(envconf["ENV_SIZE"]))
    res_data[0,0,:] = np.array((patch_x, patch_y, patch_radius))
    ag_rad = int(envconf["RADIUS_AGENT"])
    plot_map_iterative_collisions(spike_locs, res_data, width, height, ag_rad, save_name=load_name, dpi=dpi)

    # ISI stats
    # for isi_type in isi:
    #     print(f'{isi_type}: {np.mean(isi[isi_type]):.2f}, {np.median(isi[isi_type]):.2f}')
    # pickle.dump(isi, open(fr'{save_name}_isi_dict.p', 'wb'))

    # ISI histo
    # isi_range = (0,100)
    # isi_bins = 50
    # mean_isi = np.mean(isi)
    # median_isi = np.median(isi)
    # std_isi = np.std(isi)
    # min_isi = np.min(isi)
    # max_isi = np.max(isi)
    # print(f'All ISI | mean: {mean_isi:.2f} // med: {median_isi:.2f} // std: {std_isi:.2f} // min-max: {min_isi:.2f}-{max_isi:.2f} // #outliers: {np.sum(np.array(isi) > isi_range[1])}/{len(isi)}')
    # # plt.figure(figsize=(10, 6))
    # # plt.hist(isi, bins=isi_bins, range=isi_range, alpha=.7)
    # # plt.title('Interspike Interval (ISI) Histogram')
    # # plt.xlabel('Interspike Interval (timesteps)')
    # # plt.ylabel('Frequency')
    # # # plt.show()
    # # plt.savefig(fr'{save_name}_isi_{dpi}.png', dpi=dpi)
    # # plt.close()


def count_spikes(x,y,ori,action,coll_output,run,t,spike_counts,spike_locs):
    x_masked = x[run,:t]
    y_masked = y[run,:t]
    ori_masked = ori[run,:t]
    action_masked = action[run,:t]
    coll_output_masked = coll_output[run,:t]

    # count unique states needed to reach patch
    unique_count = len(set(coll_output_masked))-1

    # # find the indices of spikes
    # spike_times = np.where(coll_output_masked > 0)[0]
    # spike_times_sc = np.where(coll_output_masked >= 200)[0]
    # # spike_times_sc_2W = np.where((coll_output_masked >= 200) & (coll_output_masked < 300))[0]
    # # spike_times_sc_3W = np.where(coll_output_masked > 300)[0]
    # spike_times_sc_NW = np.where((coll_output_masked%10 == 1) & (coll_output_masked >= 200))[0]
    # spike_times_sc_NE = np.where((coll_output_masked%10 == 2) & (coll_output_masked >= 200))[0]
    # spike_times_sc_SW = np.where((coll_output_masked%10 == 3) & (coll_output_masked >= 200))[0]
    # spike_times_sc_SE = np.where((coll_output_masked%10 == 4) & (coll_output_masked >= 200))[0]
    # spike_times_sc_r1 = np.where((coll_output_masked-coll_output_masked%10 == 210) | (coll_output_masked-coll_output_masked%10 == 310))[0]
    # spike_times_sc_r2 = np.where((coll_output_masked-coll_output_masked%10 == 220) | (coll_output_masked-coll_output_masked%10 == 320))[0]
    # spike_times_sc_r3 = np.where((coll_output_masked-coll_output_masked%10 == 230) | (coll_output_masked-coll_output_masked%10 == 330))[0]
    # spike_times_sc_r4 = np.where((coll_output_masked-coll_output_masked%10 == 240) | (coll_output_masked-coll_output_masked%10 == 340))[0]
    # spike_times_sc_r5 = np.where((coll_output_masked-coll_output_masked%10 == 250) | (coll_output_masked-coll_output_masked%10 == 350))[0]
    # spike_times_sc_r6 = np.where((coll_output_masked-coll_output_masked%10 == 260) | (coll_output_masked-coll_output_masked%10 == 360))[0]
    # spike_times_sc_r7 = np.where((coll_output_masked-coll_output_masked%10 == 270) | (coll_output_masked-coll_output_masked%10 == 370))[0]
    # spike_times_sc_r8 = np.where((coll_output_masked-coll_output_masked%10 == 280) | (coll_output_masked-coll_output_masked%10 == 380))[0]
    # spike_times_dc = np.where((coll_output_masked > 0) & (coll_output_masked < 100))[0]
    # spike_times_dc_adj_NWNE = np.where((coll_output_masked == 21) | (coll_output_masked == 12))[0]
    # spike_times_dc_adj_NESE = np.where((coll_output_masked == 42) | (coll_output_masked == 24))[0]
    # spike_times_dc_adj_SESW = np.where((coll_output_masked == 43) | (coll_output_masked == 34))[0]
    # spike_times_dc_adj_SWNW = np.where((coll_output_masked == 13) | (coll_output_masked == 31))[0]
    # spike_times_dc_opp_NWSE = np.where((coll_output_masked == 41) | (coll_output_masked == 14))[0]
    # spike_times_dc_opp_NESW = np.where((coll_output_masked == 32) | (coll_output_masked == 23))[0]
    # # spike_times_Cgr2 = np.where(coll_output_masked == 100)[0]

    # find the indices of spikes (SC/DC encoding)
    n = coll_output_masked
    spike_times = np.where(n > 0)[0]
    spike_times_sc = np.where((n > 0) & (n < 100))[0]
    spike_times_sc_NW = np.where((n%10 == 1) & (n < 100))[0]
    spike_times_sc_NE = np.where((n%10 == 2) & (n < 100))[0]
    spike_times_sc_SW = np.where((n%10 == 3) & (n < 100))[0]
    spike_times_sc_SE = np.where((n%10 == 4) & (n < 100))[0]
    spike_times_sc_r1 = np.where((n-n%10 == 10) & (n < 100))[0]
    spike_times_sc_r2 = np.where((n-n%10 == 20) & (n < 100))[0]
    spike_times_sc_r3 = np.where((n-n%10 == 30) & (n < 100))[0]
    spike_times_sc_r4 = np.where((n-n%10 == 40) & (n < 100))[0]
    spike_times_sc_r5 = np.where((n-n%10 == 50) & (n < 100))[0]
    spike_times_sc_r6 = np.where((n-n%10 == 60) & (n < 100))[0]
    spike_times_sc_r7 = np.where((n-n%10 == 70) & (n < 100))[0]
    spike_times_sc_r8 = np.where((n-n%10 == 80) & (n < 100))[0]
    spike_times_dc = np.where(n > 100)[0]
    spike_times_dc_adj_NWNE = np.where(((n%10 == 2) & ((n-n%100)/100%10 == 1)) | ((n%10 == 1) & ((n-n%100)/100%10 == 2)))[0]
    spike_times_dc_adj_NESE = np.where(((n%10 == 4) & ((n-n%100)/100%10 == 2)) | ((n%10 == 2) & ((n-n%100)/100%10 == 4)))[0]
    spike_times_dc_adj_SESW = np.where(((n%10 == 4) & ((n-n%100)/100%10 == 3)) | ((n%10 == 3) & ((n-n%100)/100%10 == 4)))[0]
    spike_times_dc_adj_SWNW = np.where(((n%10 == 1) & ((n-n%100)/100%10 == 3)) | ((n%10 == 3) & ((n-n%100)/100%10 == 1)))[0]
    spike_times_dc_opp_NWSE = np.where(((n%10 == 4) & ((n-n%100)/100%10 == 1)) | ((n%10 == 1) & ((n-n%100)/100%10 == 4)))[0]
    spike_times_dc_opp_NESW = np.where(((n%10 == 3) & ((n-n%100)/100%10 == 2)) | ((n%10 == 2) & ((n-n%100)/100%10 == 3)))[0]
    spike_times_dc_r1_r2 = np.where((((n-n%10)/10%10 == 1) & (n-n%1000 == 2000)) | (((n-n%10)/10%10 == 2) & (n-n%1000 == 1000)))[0]
    spike_times_dc_r1_r3 = np.where((((n-n%10)/10%10 == 1) & (n-n%1000 == 3000)) | (((n-n%10)/10%10 == 3) & (n-n%1000 == 1000)))[0]
    spike_times_dc_r1_r4 = np.where((((n-n%10)/10%10 == 1) & (n-n%1000 == 4000)) | (((n-n%10)/10%10 == 4) & (n-n%1000 == 1000)))[0]
    spike_times_dc_r1_r5 = np.where((((n-n%10)/10%10 == 1) & (n-n%1000 == 5000)) | (((n-n%10)/10%10 == 5) & (n-n%1000 == 1000)))[0]
    spike_times_dc_r1_r6 = np.where((((n-n%10)/10%10 == 1) & (n-n%1000 == 6000)) | (((n-n%10)/10%10 == 6) & (n-n%1000 == 1000)))[0]
    spike_times_dc_r1_r7 = np.where((((n-n%10)/10%10 == 1) & (n-n%1000 == 7000)) | (((n-n%10)/10%10 == 7) & (n-n%1000 == 1000)))[0]
    spike_times_dc_r1_r8 = np.where((((n-n%10)/10%10 == 1) & (n-n%1000 == 8000)) | (((n-n%10)/10%10 == 8) & (n-n%1000 == 1000)))[0]
    spike_times_dc_r2_r3 = np.where((((n-n%10)/10%10 == 2) & (n-n%1000 == 3000)) | (((n-n%10)/10%10 == 3) & (n-n%1000 == 2000)))[0]
    spike_times_dc_r2_r4 = np.where((((n-n%10)/10%10 == 2) & (n-n%1000 == 4000)) | (((n-n%10)/10%10 == 4) & (n-n%1000 == 2000)))[0]
    spike_times_dc_r2_r5 = np.where((((n-n%10)/10%10 == 2) & (n-n%1000 == 5000)) | (((n-n%10)/10%10 == 5) & (n-n%1000 == 2000)))[0]
    spike_times_dc_r2_r6 = np.where((((n-n%10)/10%10 == 2) & (n-n%1000 == 6000)) | (((n-n%10)/10%10 == 6) & (n-n%1000 == 2000)))[0]
    spike_times_dc_r2_r7 = np.where((((n-n%10)/10%10 == 2) & (n-n%1000 == 7000)) | (((n-n%10)/10%10 == 7) & (n-n%1000 == 2000)))[0]
    spike_times_dc_r2_r8 = np.where((((n-n%10)/10%10 == 2) & (n-n%1000 == 8000)) | (((n-n%10)/10%10 == 8) & (n-n%1000 == 2000)))[0]
    spike_times_dc_r3_r4 = np.where((((n-n%10)/10%10 == 3) & (n-n%1000 == 4000)) | (((n-n%10)/10%10 == 4) & (n-n%1000 == 3000)))[0]
    spike_times_dc_r3_r5 = np.where((((n-n%10)/10%10 == 3) & (n-n%1000 == 5000)) | (((n-n%10)/10%10 == 5) & (n-n%1000 == 3000)))[0]
    spike_times_dc_r3_r6 = np.where((((n-n%10)/10%10 == 3) & (n-n%1000 == 6000)) | (((n-n%10)/10%10 == 6) & (n-n%1000 == 3000)))[0]
    spike_times_dc_r3_r7 = np.where((((n-n%10)/10%10 == 3) & (n-n%1000 == 7000)) | (((n-n%10)/10%10 == 7) & (n-n%1000 == 3000)))[0]
    spike_times_dc_r3_r8 = np.where((((n-n%10)/10%10 == 3) & (n-n%1000 == 8000)) | (((n-n%10)/10%10 == 8) & (n-n%1000 == 3000)))[0]
    spike_times_dc_r4_r5 = np.where((((n-n%10)/10%10 == 4) & (n-n%1000 == 5000)) | (((n-n%10)/10%10 == 5) & (n-n%1000 == 4000)))[0]
    spike_times_dc_r4_r6 = np.where((((n-n%10)/10%10 == 4) & (n-n%1000 == 6000)) | (((n-n%10)/10%10 == 6) & (n-n%1000 == 4000)))[0]
    spike_times_dc_r4_r7 = np.where((((n-n%10)/10%10 == 4) & (n-n%1000 == 7000)) | (((n-n%10)/10%10 == 7) & (n-n%1000 == 4000)))[0]
    spike_times_dc_r4_r8 = np.where((((n-n%10)/10%10 == 4) & (n-n%1000 == 8000)) | (((n-n%10)/10%10 == 8) & (n-n%1000 == 4000)))[0]
    spike_times_dc_r5_r6 = np.where((((n-n%10)/10%10 == 5) & (n-n%1000 == 6000)) | (((n-n%10)/10%10 == 6) & (n-n%1000 == 5000)))[0]
    spike_times_dc_r5_r7 = np.where((((n-n%10)/10%10 == 5) & (n-n%1000 == 7000)) | (((n-n%10)/10%10 == 7) & (n-n%1000 == 5000)))[0]
    spike_times_dc_r5_r8 = np.where((((n-n%10)/10%10 == 5) & (n-n%1000 == 8000)) | (((n-n%10)/10%10 == 8) & (n-n%1000 == 5000)))[0]
    spike_times_dc_r6_r7 = np.where((((n-n%10)/10%10 == 6) & (n-n%1000 == 7000)) | (((n-n%10)/10%10 == 7) & (n-n%1000 == 6000)))[0]
    spike_times_dc_r6_r8 = np.where((((n-n%10)/10%10 == 6) & (n-n%1000 == 8000)) | (((n-n%10)/10%10 == 8) & (n-n%1000 == 6000)))[0]
    spike_times_dc_r7_r8 = np.where((((n-n%10)/10%10 == 7) & (n-n%1000 == 8000)) | (((n-n%10)/10%10 == 8) & (n-n%1000 == 7000)))[0]


    ## to check encoding validity

    # sc_corners = [
    #     spike_times_sc_NW,
    #     spike_times_sc_NE,
    #     spike_times_sc_SW,
    #     spike_times_sc_SE,
    #     spike_times_dc
    # ]
    # for i,j in itertools.combinations(sc_corners,2):
    #     if np.any(np.intersect1d(i,j)):
    #         print(f'SC corner overlap: {np.intersect1d(i,j)}')

    # sc_rays = [
    #     spike_times_sc_r1,
    #     spike_times_sc_r2,
    #     spike_times_sc_r3,
    #     spike_times_sc_r4,
    #     spike_times_sc_r5,
    #     spike_times_sc_r6,
    #     spike_times_sc_r7,
    #     spike_times_sc_r8,
    #     spike_times_dc
    # ]
    # for i,j in itertools.combinations(sc_rays,2):
    #     if np.any(np.intersect1d(i,j)):
    #         print(f'SC ray overlap: {np.intersect1d(i,j)}')

    # dc_corners = [
    #     spike_times_dc_adj_NWNE,
    #     spike_times_dc_adj_NESE,
    #     spike_times_dc_adj_SESW,
    #     spike_times_dc_adj_SWNW,
    #     spike_times_dc_opp_NWSE,
    #     spike_times_dc_opp_NESW,
    #     spike_times_sc
    # ]
    # for i,j in itertools.combinations(dc_corners,2):
    #     if np.any(np.intersect1d(i,j)):
    #         print(f'DC corner overlap: {np.intersect1d(i,j)}')

    # dc_rays = [
    #     spike_times_dc_r1_r2,
    #     spike_times_dc_r1_r3,
    #     spike_times_dc_r1_r4,
    #     spike_times_dc_r1_r5,
    #     spike_times_dc_r1_r6,
    #     spike_times_dc_r1_r7,
    #     spike_times_dc_r1_r8,
    #     spike_times_dc_r2_r3,
    #     spike_times_dc_r2_r4,
    #     spike_times_dc_r2_r5,
    #     spike_times_dc_r2_r6,
    #     spike_times_dc_r2_r7,
    #     spike_times_dc_r2_r8,
    #     spike_times_dc_r3_r4,
    #     spike_times_dc_r3_r5,
    #     spike_times_dc_r3_r6,
    #     spike_times_dc_r3_r7,
    #     spike_times_dc_r3_r8,
    #     spike_times_dc_r4_r5,
    #     spike_times_dc_r4_r6,
    #     spike_times_dc_r4_r7,
    #     spike_times_dc_r4_r8,
    #     spike_times_dc_r5_r6,
    #     spike_times_dc_r5_r7,
    #     spike_times_dc_r5_r8,
    #     spike_times_dc_r6_r7,
    #     spike_times_dc_r6_r8,
    #     spike_times_dc_r7_r8,
    #     spike_times_sc
    # ]
    # for i,j in itertools.combinations(dc_rays,2):
    #     if np.any(np.intersect1d(i,j)):
    #         print(f'DC ray overlap: {np.intersect1d(i,j)}')


    # count spike types
    spike_counts['SC'] += len(spike_times_sc)
    # spike_counts['SC: 2 Walls'] += len(spike_times_sc_2W)
    # spike_counts['SC: 3 Walls'] += len(spike_times_sc_3W)
    spike_counts['SC: NW'] += len(spike_times_sc_NW)
    spike_counts['SC: NE'] += len(spike_times_sc_NE)
    spike_counts['SC: SW'] += len(spike_times_sc_SW)
    spike_counts['SC: SE'] += len(spike_times_sc_SE)
    spike_counts['SC: Ray 1'] += len(spike_times_sc_r1)
    spike_counts['SC: Ray 2'] += len(spike_times_sc_r2)
    spike_counts['SC: Ray 3'] += len(spike_times_sc_r3)
    spike_counts['SC: Ray 4'] += len(spike_times_sc_r4)
    spike_counts['SC: Ray 5'] += len(spike_times_sc_r5)
    spike_counts['SC: Ray 6'] += len(spike_times_sc_r6)
    spike_counts['SC: Ray 7'] += len(spike_times_sc_r7)
    spike_counts['SC: Ray 8'] += len(spike_times_sc_r8)
    spike_counts['DC'] += len(spike_times_dc)
    spike_counts['DC-Adj: NW-NE'] += len(spike_times_dc_adj_NWNE)
    spike_counts['DC-Adj: NE-SE'] += len(spike_times_dc_adj_NESE)
    spike_counts['DC-Adj: SE-SW'] += len(spike_times_dc_adj_SESW)
    spike_counts['DC-Adj: SW-NW'] += len(spike_times_dc_adj_SWNW)
    spike_counts['DC-Opp: NW-SE'] += len(spike_times_dc_opp_NWSE)
    spike_counts['DC-Opp: NE-SW'] += len(spike_times_dc_opp_NESW)
    spike_counts['DC: Rays 1 x 2'] += len(spike_times_dc_r1_r2)
    spike_counts['DC: Rays 1 x 3'] += len(spike_times_dc_r1_r3)
    spike_counts['DC: Rays 1 x 4'] += len(spike_times_dc_r1_r4)
    spike_counts['DC: Rays 1 x 5'] += len(spike_times_dc_r1_r5)
    spike_counts['DC: Rays 1 x 6'] += len(spike_times_dc_r1_r6)
    spike_counts['DC: Rays 1 x 7'] += len(spike_times_dc_r1_r7)
    spike_counts['DC: Rays 1 x 8'] += len(spike_times_dc_r1_r8)
    spike_counts['DC: Rays 2 x 3'] += len(spike_times_dc_r2_r3)
    spike_counts['DC: Rays 2 x 4'] += len(spike_times_dc_r2_r4)
    spike_counts['DC: Rays 2 x 5'] += len(spike_times_dc_r2_r5)
    spike_counts['DC: Rays 2 x 6'] += len(spike_times_dc_r2_r6)
    spike_counts['DC: Rays 2 x 7'] += len(spike_times_dc_r2_r7)
    spike_counts['DC: Rays 2 x 8'] += len(spike_times_dc_r2_r8)
    spike_counts['DC: Rays 3 x 4'] += len(spike_times_dc_r3_r4)
    spike_counts['DC: Rays 3 x 5'] += len(spike_times_dc_r3_r5)
    spike_counts['DC: Rays 3 x 6'] += len(spike_times_dc_r3_r6)
    spike_counts['DC: Rays 3 x 7'] += len(spike_times_dc_r3_r7)
    spike_counts['DC: Rays 3 x 8'] += len(spike_times_dc_r3_r8)
    spike_counts['DC: Rays 4 x 5'] += len(spike_times_dc_r4_r5)
    spike_counts['DC: Rays 4 x 6'] += len(spike_times_dc_r4_r6)
    spike_counts['DC: Rays 4 x 7'] += len(spike_times_dc_r4_r7)
    spike_counts['DC: Rays 4 x 8'] += len(spike_times_dc_r4_r8)
    spike_counts['DC: Rays 5 x 6'] += len(spike_times_dc_r5_r6)
    spike_counts['DC: Rays 5 x 7'] += len(spike_times_dc_r5_r7)
    spike_counts['DC: Rays 5 x 8'] += len(spike_times_dc_r5_r8)
    spike_counts['DC: Rays 6 x 7'] += len(spike_times_dc_r6_r7)
    spike_counts['DC: Rays 6 x 8'] += len(spike_times_dc_r6_r8)
    spike_counts['DC: Rays 7 x 8'] += len(spike_times_dc_r7_r8)
    # spike_counts['C>2'] += len(spike_times_Cgr2)

    # # calculate interspike intervals
    # isi['All'].extend(np.diff(spike_times))
    # isi['SC'].extend(np.diff(spike_times_sc))
    # isi['SC: NW'].extend(np.diff(spike_times_sc_NW))
    # isi['SC: NE'].extend(np.diff(spike_times_sc_NE))
    # isi['SC: SW'].extend(np.diff(spike_times_sc_SW))
    # isi['SC: SE'].extend(np.diff(spike_times_sc_SE))
    # isi['SC: Ray 1'].extend(np.diff(spike_times_sc_r1))
    # isi['SC: Ray 2'].extend(np.diff(spike_times_sc_r2))
    # isi['SC: Ray 3'].extend(np.diff(spike_times_sc_r3))
    # isi['SC: Ray 4'].extend(np.diff(spike_times_sc_r4))
    # isi['SC: Ray 5'].extend(np.diff(spike_times_sc_r5))
    # isi['SC: Ray 6'].extend(np.diff(spike_times_sc_r6))
    # isi['SC: Ray 7'].extend(np.diff(spike_times_sc_r7))
    # isi['SC: Ray 8'].extend(np.diff(spike_times_sc_r8))
    # isi['DC'].extend(np.diff(spike_times_dc))

    spike_locs['SC'].extend(np.vstack((x_masked[spike_times_sc], y_masked[spike_times_sc], ori_masked[spike_times_sc])).T)
    # spike_locs['SC: 2 Walls'].extend(np.vstack((x_masked[spike_times_sc_2W], y_masked[spike_times_sc_2W], ori_masked[spike_times_sc_2W])).T)
    # spike_locs['SC: 3 Walls'].extend(np.vstack((x_masked[spike_times_sc_3W], y_masked[spike_times_sc_3W], ori_masked[spike_times_sc_3W])).T)
    spike_locs['SC: NW'].extend(np.vstack((x_masked[spike_times_sc_NW], y_masked[spike_times_sc_NW], ori_masked[spike_times_sc_NW])).T)
    spike_locs['SC: NE'].extend(np.vstack((x_masked[spike_times_sc_NE], y_masked[spike_times_sc_NE], ori_masked[spike_times_sc_NE])).T)
    spike_locs['SC: SW'].extend(np.vstack((x_masked[spike_times_sc_SW], y_masked[spike_times_sc_SW], ori_masked[spike_times_sc_SW])).T)
    spike_locs['SC: SE'].extend(np.vstack((x_masked[spike_times_sc_SE], y_masked[spike_times_sc_SE], ori_masked[spike_times_sc_SE])).T)
    spike_locs['SC: Ray 1'].extend(np.vstack((x_masked[spike_times_sc_r1], y_masked[spike_times_sc_r1], ori_masked[spike_times_sc_r1])).T)
    spike_locs['SC: Ray 2'].extend(np.vstack((x_masked[spike_times_sc_r2], y_masked[spike_times_sc_r2], ori_masked[spike_times_sc_r2])).T)
    spike_locs['SC: Ray 3'].extend(np.vstack((x_masked[spike_times_sc_r3], y_masked[spike_times_sc_r3], ori_masked[spike_times_sc_r3])).T)
    spike_locs['SC: Ray 4'].extend(np.vstack((x_masked[spike_times_sc_r4], y_masked[spike_times_sc_r4], ori_masked[spike_times_sc_r4])).T)
    spike_locs['SC: Ray 5'].extend(np.vstack((x_masked[spike_times_sc_r5], y_masked[spike_times_sc_r5], ori_masked[spike_times_sc_r5])).T)
    spike_locs['SC: Ray 6'].extend(np.vstack((x_masked[spike_times_sc_r6], y_masked[spike_times_sc_r6], ori_masked[spike_times_sc_r6])).T)
    spike_locs['SC: Ray 7'].extend(np.vstack((x_masked[spike_times_sc_r7], y_masked[spike_times_sc_r7], ori_masked[spike_times_sc_r7])).T)
    spike_locs['SC: Ray 8'].extend(np.vstack((x_masked[spike_times_sc_r8], y_masked[spike_times_sc_r8], ori_masked[spike_times_sc_r8])).T)
    spike_locs['DC'].extend(np.vstack((x_masked[spike_times_dc], y_masked[spike_times_dc], ori_masked[spike_times_dc])).T)
    spike_locs['DC-Adj: NW-NE'].extend(np.vstack((x_masked[spike_times_dc_adj_NWNE], y_masked[spike_times_dc_adj_NWNE], ori_masked[spike_times_dc_adj_NWNE])).T)
    spike_locs['DC-Adj: NE-SE'].extend(np.vstack((x_masked[spike_times_dc_adj_NESE], y_masked[spike_times_dc_adj_NESE], ori_masked[spike_times_dc_adj_NESE])).T)
    spike_locs['DC-Adj: SE-SW'].extend(np.vstack((x_masked[spike_times_dc_adj_SESW], y_masked[spike_times_dc_adj_SESW], ori_masked[spike_times_dc_adj_SESW])).T)
    spike_locs['DC-Adj: SW-NW'].extend(np.vstack((x_masked[spike_times_dc_adj_SWNW], y_masked[spike_times_dc_adj_SWNW], ori_masked[spike_times_dc_adj_SWNW])).T)
    spike_locs['DC-Opp: NW-SE'].extend(np.vstack((x_masked[spike_times_dc_opp_NWSE], y_masked[spike_times_dc_opp_NWSE], ori_masked[spike_times_dc_opp_NWSE])).T)
    spike_locs['DC-Opp: NE-SW'].extend(np.vstack((x_masked[spike_times_dc_opp_NESW], y_masked[spike_times_dc_opp_NESW], ori_masked[spike_times_dc_opp_NESW])).T)
    spike_locs['DC: Rays 1 x 2'].extend(np.vstack((x_masked[spike_times_dc_r1_r2], y_masked[spike_times_dc_r1_r2], ori_masked[spike_times_dc_r1_r2])).T)
    spike_locs['DC: Rays 1 x 3'].extend(np.vstack((x_masked[spike_times_dc_r1_r3], y_masked[spike_times_dc_r1_r3], ori_masked[spike_times_dc_r1_r3])).T)
    spike_locs['DC: Rays 1 x 4'].extend(np.vstack((x_masked[spike_times_dc_r1_r4], y_masked[spike_times_dc_r1_r4], ori_masked[spike_times_dc_r1_r4])).T)
    spike_locs['DC: Rays 1 x 5'].extend(np.vstack((x_masked[spike_times_dc_r1_r5], y_masked[spike_times_dc_r1_r5], ori_masked[spike_times_dc_r1_r5])).T)
    spike_locs['DC: Rays 1 x 6'].extend(np.vstack((x_masked[spike_times_dc_r1_r6], y_masked[spike_times_dc_r1_r6], ori_masked[spike_times_dc_r1_r6])).T)
    spike_locs['DC: Rays 1 x 7'].extend(np.vstack((x_masked[spike_times_dc_r1_r7], y_masked[spike_times_dc_r1_r7], ori_masked[spike_times_dc_r1_r7])).T)
    spike_locs['DC: Rays 1 x 8'].extend(np.vstack((x_masked[spike_times_dc_r1_r8], y_masked[spike_times_dc_r1_r8], ori_masked[spike_times_dc_r1_r8])).T)
    spike_locs['DC: Rays 2 x 3'].extend(np.vstack((x_masked[spike_times_dc_r2_r3], y_masked[spike_times_dc_r2_r3], ori_masked[spike_times_dc_r2_r3])).T)
    spike_locs['DC: Rays 2 x 4'].extend(np.vstack((x_masked[spike_times_dc_r2_r4], y_masked[spike_times_dc_r2_r4], ori_masked[spike_times_dc_r2_r4])).T)
    spike_locs['DC: Rays 2 x 5'].extend(np.vstack((x_masked[spike_times_dc_r2_r5], y_masked[spike_times_dc_r2_r5], ori_masked[spike_times_dc_r2_r5])).T)
    spike_locs['DC: Rays 2 x 6'].extend(np.vstack((x_masked[spike_times_dc_r2_r6], y_masked[spike_times_dc_r2_r6], ori_masked[spike_times_dc_r2_r6])).T)
    spike_locs['DC: Rays 2 x 7'].extend(np.vstack((x_masked[spike_times_dc_r2_r7], y_masked[spike_times_dc_r2_r7], ori_masked[spike_times_dc_r2_r7])).T)
    spike_locs['DC: Rays 2 x 8'].extend(np.vstack((x_masked[spike_times_dc_r2_r8], y_masked[spike_times_dc_r2_r8], ori_masked[spike_times_dc_r2_r8])).T)
    spike_locs['DC: Rays 3 x 4'].extend(np.vstack((x_masked[spike_times_dc_r3_r4], y_masked[spike_times_dc_r3_r4], ori_masked[spike_times_dc_r3_r4])).T)
    spike_locs['DC: Rays 3 x 5'].extend(np.vstack((x_masked[spike_times_dc_r3_r5], y_masked[spike_times_dc_r3_r5], ori_masked[spike_times_dc_r3_r5])).T)
    spike_locs['DC: Rays 3 x 6'].extend(np.vstack((x_masked[spike_times_dc_r3_r6], y_masked[spike_times_dc_r3_r6], ori_masked[spike_times_dc_r3_r6])).T)
    spike_locs['DC: Rays 3 x 7'].extend(np.vstack((x_masked[spike_times_dc_r3_r7], y_masked[spike_times_dc_r3_r7], ori_masked[spike_times_dc_r3_r7])).T)
    spike_locs['DC: Rays 3 x 8'].extend(np.vstack((x_masked[spike_times_dc_r3_r8], y_masked[spike_times_dc_r3_r8], ori_masked[spike_times_dc_r3_r8])).T)
    spike_locs['DC: Rays 4 x 5'].extend(np.vstack((x_masked[spike_times_dc_r4_r5], y_masked[spike_times_dc_r4_r5], ori_masked[spike_times_dc_r4_r5])).T)
    spike_locs['DC: Rays 4 x 6'].extend(np.vstack((x_masked[spike_times_dc_r4_r6], y_masked[spike_times_dc_r4_r6], ori_masked[spike_times_dc_r4_r6])).T)
    spike_locs['DC: Rays 4 x 7'].extend(np.vstack((x_masked[spike_times_dc_r4_r7], y_masked[spike_times_dc_r4_r7], ori_masked[spike_times_dc_r4_r7])).T)
    spike_locs['DC: Rays 4 x 8'].extend(np.vstack((x_masked[spike_times_dc_r4_r8], y_masked[spike_times_dc_r4_r8], ori_masked[spike_times_dc_r4_r8])).T)
    spike_locs['DC: Rays 5 x 6'].extend(np.vstack((x_masked[spike_times_dc_r5_r6], y_masked[spike_times_dc_r5_r6], ori_masked[spike_times_dc_r5_r6])).T)
    spike_locs['DC: Rays 5 x 7'].extend(np.vstack((x_masked[spike_times_dc_r5_r7], y_masked[spike_times_dc_r5_r7], ori_masked[spike_times_dc_r5_r7])).T)
    spike_locs['DC: Rays 5 x 8'].extend(np.vstack((x_masked[spike_times_dc_r5_r8], y_masked[spike_times_dc_r5_r8], ori_masked[spike_times_dc_r5_r8])).T)
    spike_locs['DC: Rays 6 x 7'].extend(np.vstack((x_masked[spike_times_dc_r6_r7], y_masked[spike_times_dc_r6_r7], ori_masked[spike_times_dc_r6_r7])).T)
    spike_locs['DC: Rays 6 x 8'].extend(np.vstack((x_masked[spike_times_dc_r6_r8], y_masked[spike_times_dc_r6_r8], ori_masked[spike_times_dc_r6_r8])).T)
    spike_locs['DC: Rays 7 x 8'].extend(np.vstack((x_masked[spike_times_dc_r7_r8], y_masked[spike_times_dc_r7_r8], ori_masked[spike_times_dc_r7_r8])).T)
    # spike_locs['C>2'].extend(np.vstack((x_masked[spike_times_Cgr2], y_masked[spike_times_Cgr2], ori_masked[spike_times_Cgr2])).T)

    return spike_counts, spike_locs, unique_count


def gather_agent_ray_boundary_collision_tran_matrix(exp_name, gen_ext, space_step, orient_step, timesteps):
    # print(f'plotting collision stats - {exp_name} @ {dpi} dpi')

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    load_name = fr'{data_dir}/traj_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_cen_e1'
    if not os.path.exists(load_name+'.bin'):
        print(f'no data found for {load_name}')
        return 0,0,0,0
    else:
        with open(load_name+'.bin', 'rb') as f:
            ag_data = pickle.load(f)

    # cut init
    delay = 25
    # x = ag_data[:,delay:,0]
    # y = ag_data[:,delay:,1]
    # ori = ag_data[:,delay:,2]
    # action = ag_data[:,delay:,3]
    # action = np.abs(action)
    coll_output = ag_data[:,delay:,4]

    count_matrix, states = labeled_histo(coll_output)
    # states_trimmed = []
    # for s,c in zip(states,count_matrix):
    #     if c > .01:
    #         # print(f'{int(s)}: {round(c,4)}')
    #         states_trimmed.append(s)
    # # print(states_trimmed)

    # # output transition matrix
    # tran_matrix = estimate_transition_matrix(coll_output, states_trimmed)
    tran_matrix, states = estimate_transition_matrix(coll_output, states)

    return tran_matrix, states


def labeled_histo(sequence_array):

    # Step 1: Identify unique states
    states = sorted(list(set(sequence_array.flatten())))  # Get unique states and sort them
    states = states[1:] # remove zero (only events)
    # print(states)

    # Step 2: Initialize a count matrix to store transitions
    num_states = len(states)
    count_matrix = np.zeros(num_states, dtype=int)

    # Step 3: Count transitions
    sequence = sequence_array.flatten()
    seq_trimmed = sequence[sequence != 0] # only events

    for i in seq_trimmed:
        state_index = states.index(i)
        count_matrix[state_index] += 1

    # Step 4: Normalize the count matrix to get probabilities
    count_matrix = count_matrix.astype(float)  # Convert to float for division
    count_matrix = count_matrix / count_matrix.sum() # Normalize to total count

    return count_matrix, states


def estimate_transition_matrix(sequence_array):
    """
    Estimate the transition matrix from a sequence of states.

    Parameters:
    - sequence: A list of states (e.g., ["Sunny", "Cloudy", "Rainy", "Sunny", ...]).

    Returns:
    - transition_matrix: A 2D NumPy array representing the transition probabilities.
    - states: A list of unique states in the sequence.
    """
    # Step 1: Identify unique states
    states = sorted(list(set(sequence_array.flatten())))  # Get unique states and sort them
    states = states[1:] # remove zero (only events)
    # print(states)

    # remove symmetries for DC colls?? not sure if needed
    # sequence_array = np.where(sequence_array == 100, 0, sequence_array)

    # Step 2: Initialize a count matrix to store transitions
    # num_states = len(states)
    num_states = len(states)
    count_matrix = np.zeros((num_states, num_states), dtype=int)

    # Step 3: Count transitions
    runs, timesteps = sequence_array.shape
    for run in range(runs):
        sequence = sequence_array[run,:]
        seq_trimmed = sequence[sequence != 0] # only events

        for (i, j) in zip(seq_trimmed[:-1], seq_trimmed[1:]):
            current_state_index = states.index(i)
            next_state_index = states.index(j)
            count_matrix[current_state_index, next_state_index] += 1

    # Step 4: Normalize the count matrix to get probabilities
    transition_matrix = count_matrix.astype(float)  # Convert to float for division
    row_sums = transition_matrix.sum(axis=1, keepdims=True)

    # rows ~ columns (but not exactly, off by ~ # runs, since this isn't one long seq)
    # print(row_sums.T)
    # print(transition_matrix.sum(axis=0, keepdims=True))
    # print(np.sum(np.abs(row_sums.T - transition_matrix.sum(axis=0, keepdims=True)))/2)
    # print(runs, timesteps)

    transition_matrix = transition_matrix / row_sums  # Normalize rows to sum to 1

    # Handle rows with no transitions (replace NaNs with uniform probabilities)
    transition_matrix[np.isnan(transition_matrix)] = 1.0 / num_states

    return transition_matrix, states


def estimate_transition_matrix_trimmed(sequence_array, states_trimmed):
    """
    Estimate the transition matrix from a sequence of states.

    Parameters:
    - sequence: A list of states (e.g., ["Sunny", "Cloudy", "Rainy", "Sunny", ...]).

    Returns:
    - transition_matrix: A 2D NumPy array representing the transition probabilities.
    - states: A list of unique states in the sequence.
    """
    # Step 1: Identify unique states
    states = sorted(list(set(sequence_array.flatten())))  # Get unique states and sort them
    states = states[1:] # remove zero (only events)
    # print(states)

    states_unconsidered = []
    for s in states:
        if s not in states_trimmed:
            states_unconsidered.append(s)

    # Step 2: Initialize a count matrix to store transitions
    num_states = len(states_trimmed)
    count_matrix = np.zeros((num_states, num_states), dtype=int)

    # Step 3: Count transitions
    runs, timesteps = sequence_array.shape
    for run in range(runs):
        sequence = sequence_array[run,:]
        seq_trimmed = sequence[sequence != 0] # only events

        for s in states_unconsidered:
            seq_trimmed = seq_trimmed[seq_trimmed != s]

        for (i, j) in zip(seq_trimmed[:-1], seq_trimmed[1:]):
            current_state_index = states_trimmed.index(i)
            next_state_index = states_trimmed.index(j)
            count_matrix[current_state_index, next_state_index] += 1

    # Step 4: Normalize the count matrix to get probabilities
    transition_matrix = count_matrix.astype(float)  # Convert to float for division
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = transition_matrix / row_sums  # Normalize rows to sum to 1
    # Handle rows with no transitions (replace NaNs with uniform probabilities)
    transition_matrix[np.isnan(transition_matrix)] = 1.0 / num_states

    return transition_matrix


def visualize_transition_matrix(transition_matrix, states, save_name=None, dpi=50):
    """
    Visualize the transition matrix as a heatmap.

    Parameters:
    - transition_matrix: A 2D NumPy array representing the transition probabilities.
    - states: A list of state labels corresponding to the rows/columns of the matrix.
    - title: Title of the plot (optional).
    """
    # Create a mask for zero values: "X" for zero values, empty string otherwise
    zero_mask = transition_matrix == np.min(transition_matrix)
    annotations = np.where(zero_mask, "X", "")

    # Convert states to int
    states = [int(x) for x in states]

    plt.figure(figsize=(16,12))
    # plt.figure(figsize=(8,6))
    sns.heatmap(
        transition_matrix,
        # annot=True,  # Annotate cells with the probability values
        # fmt=".2f",   # Format annotations to 2 decimal places
        annot=annotations,  # Use custom annotations
        fmt="", # strings
        cmap="Blues",  # Color map
        xticklabels=states,
        yticklabels=states,
        cbar=True,   # Show color bar
        linewidths=0.5,  # Add lines between cells
    )
    plt.xlabel("Next State")
    plt.ylabel("Current State")
    plt.tight_layout()

    SC_DC_divider = np.argmax(np.array(states) > 100)
    plt.axvline(x = SC_DC_divider, color='gray', linestyle='--', linewidth=0.5)
    plt.axhline(y = SC_DC_divider, color='gray', linestyle='--', linewidth=0.5)

    if save_name:
        plt.savefig(save_name, dpi=dpi)
        plt.close()
    else:
        plt.show()


def visualize_markov_chain(transition_matrix, states, save_name=None, dpi=50):
    from netgraph import Graph
    # https://github.com/paulbrodersen/netgraph

    sources, targets = np.where(transition_matrix)
    weights = transition_matrix[sources, targets]
    edges = list(zip(sources, targets))
    edge_labels = dict(zip(edges, weights))

    states = [int(x) for x in states]
    nodes = list(range(len(states)))
    node_labels = dict(zip(nodes, states))

    # create a dictionary that maps nodes to the community they belong to
    community = []
    colors = []
    for node in states:
        if node < 100:
            community.append(0)
            colors.append('tab:blue')
        else:
            community.append(1)
            colors.append('tab:red')
    node_community = dict(zip(nodes, community))
    node_colors = dict(zip(nodes, colors))

    # all_src = list(set(sources))
    # print(all_src, len(all_src))
    # print(nodes, len(nodes))
    # print(states, len(states))
    # print(colors, len(colors))

    fig, ax = plt.subplots(figsize=(16,12))
    Graph(edges,
        arrows=True,
        edge_layout='curved',
        edge_layout_kwargs=dict(bundle_parallel_edges=False),
        # edge_layout='bundled',
        # edge_layout_kwargs=dict(k=2000),
        edge_width={(u, v):2*d+.5 for (u, v),d in edge_labels.items()},
        # edge_labels=edge_labels,
        # edge_label_position=0.66,
        # edge_label_fontdict=dict(fontweight='bold'),
        # node_layout=node_positions,
        node_positions=None,
        node_layout='spring',
        # node_layout='community',
        # node_layout_kwargs=dict(node_to_community=node_community),
        node_color=node_colors,
        # node_size=4,
        node_labels=node_labels,
        # node_label_fontdict=dict(size=14,fontweight='bold'),
        # node_label_offset=0.12,
        ax=ax
        )
    if save_name:
        plt.savefig(save_name, dpi=dpi)
        plt.close()
    else:
        plt.show()


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


def build_agent_trajs_PRW(space_step, orient_step, timesteps, behavior, rot_diff, curve=None, limit=None, bias=None):
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
    hitx = np.digitize(x, x_bins[1:])
    hity = np.digitize(y, y_bins[1:])
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


def build_agent_views(space_step=5, orient_step=np.pi/256, vis_field_res = 8):

    print(f'building agent views @ ss{space_step}, os{int(np.pi/orient_step)}, vfr{vis_field_res}')
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
                        sim_type='walls',
                    )
                agent.visual_sensing([],[])

                if agent.vis_field not in views:
                    views.append(agent.vis_field)

    views.sort()

    # for v in views:
    #     print(v)
    # print(len(views))

    with open(fr'{data_dir}/IDM/views_vfr{vis_field_res}.bin', 'wb') as f:
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
    # axes.set_xlim(0, x_max)
    # axes.set_ylim(0, y_max)
    # h,w = 8,8
    # l,r,t,b = fig.subplotpars.left, fig.subplotpars.right, fig.subplotpars.top, fig.subplotpars.bottom
    # fig.set_size_inches( float(w)/(r-l) , float(h)/(t-b) )

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

    elif plot_type == '_transrot_ori_perf_dist':
        print(f'plotting IDM (transrot - ori - perfect matches only) @ {dpi} dpi')

        nx,ny = count_mat.shape
        x,y,ori = zip(*[ (x_range[i], height-y_range[j], ori_mat[i,j]) for i in range(nx) for j in range(ny) if count_mat[i,j] == 0 ])

        ori = np.array(ori)
        ori_shift = (ori + 2*np.pi)%(2*np.pi)
        ori_deg = ori_shift*180/np.pi

        axes.hist(ori_deg, bins=np.arange(0,361,5), density=True)
        axes.set_xticks(np.linspace(0,361,5))
        axes.set_xticklabels(['$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
        axes.set_xlabel('Orientation')
        axes.set_ylabel('Probability Density')

    else:
        print('invalid plot type')
        return

    # axes.add_patch( plt.Circle((patch_x, height-patch_y), patch_radius, edgecolor='k', fill=False, zorder=1) )
    plt.savefig(fr'{save_name}{plot_type}.png', dpi=dpi)
    plt.close()


def agent_action_from_view(envconf, NN, view):

    sim_type = str(envconf["SIM_TYPE"])
    if sim_type == 'walls':
        num_class = 4
    elif sim_type == 'walls, social-RW':
        num_class = 6

    agent = Agent(
            id=0,
            position=(0,0),
            orientation=0,
            max_vel=int(envconf["MAXIMUM_VELOCITY"]),
            FOV=float(envconf['AGENT_FOV']),
            vis_field_res=int(envconf["VISUAL_FIELD_RESOLUTION"]),
            vision_range=int(envconf["VISION_RANGE"]),
            num_class_elements=num_class,
            consumption=1,
            model=NN,
            boundary_endpts=(None,None,None,None),
            window_pad=30,
            radius=int(envconf["RADIUS_AGENT"]),
            color=(0,0,0),
            vis_transform='',
            percep_angle_noise_std=0,
            sim_type=sim_type,
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


def build_IDM_unique_views(exp_name, gen_ext, space_step, orient_step, vis_field_res=32):

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

    # count # unique views for each coord/bin
    IDM = np.zeros((len(x_range), len(y_range)))
    print(f'unique view matrix shape (x, y, count): {IDM.shape}')

    for i, x in enumerate(x_range):
        print(x)
        for j, y in enumerate(y_range):
            views_at_coord = []
            for orient in orient_range:
                trans_pano = agent_pano_from_xyo(envconf, NN, boundary_endpts, x, y, orient, vis_field_res)
                if trans_pano not in views_at_coord:
                    views_at_coord.append(trans_pano)
            IDM[i,j] = len(views_at_coord)

    with open(fr'{data_dir}/IDM/IDM_unq_c{space_step}_o{int(np.pi/orient_step)}_vsres{vis_field_res}.bin', 'wb') as f:
        pickle.dump(IDM, f)


def plot_IDM_unique_views(space_step, orient_step, vis_field_res=32, dpi=50):

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

    with open(fr'{data_dir}/IDM/IDM_unq_c{space_step}_o{int(np.pi/orient_step)}_vsres{vis_field_res}.bin', 'rb') as f:
        H = pickle.load(f)


    fig, axes = plt.subplots() 
    axes.set_xlim(0, x_max)
    axes.set_ylim(0, y_max)
    h,w = 4,4
    l,r,t,b = fig.subplotpars.left, fig.subplotpars.right, fig.subplotpars.top, fig.subplotpars.bottom
    fig.set_size_inches( float(w)/(r-l) , float(h)/(t-b) )

    X,Y = np.meshgrid(x_range, y_range)
    im = axes.pcolormesh(X, Y, H.T, cmap='plasma')
    # axes.pcolormesh(X, Y, H.T, cmap='plasma', norm=mpl.colors.Normalize(vmin=110)) # for scale=0.5
    # print(np.max(H), np.min(H))
    # print(np.sort(H.flatten())[2500:3500])
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(axes)
    cax1 = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax1, label='Number Unique Views / Point')

    axes.add_patch( plt.Circle((patch_x, height-patch_y), patch_radius, edgecolor='k', fill=False, zorder=1) )
    plt.savefig(fr'{data_dir}/IDM/unqviews_c{space_step}_o{int(np.pi/orient_step)}_vsres{vis_field_res}.png', dpi=dpi)
    plt.close()
    # plt.show()


def build_social_views(exp_name, gen_ext, space_step, orient_step, timesteps, extra=''):

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    with open(fr'{data_dir}/IDM/social_view_set.bin', 'rb') as f:
        view_set_all = pickle.load(f)
    view_set = set()

    NN_pv_path = fr'{data_dir}/{exp_name}/{gen_ext}_NNcen_pickle.bin'
    with open(NN_pv_path,'rb') as f:
        pv = pickle.load(f)
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)

    if extra == '': 
        extra = envconf['N']

    # every grid position/direction
    width, height = tuple(eval(envconf["ENV_SIZE"]))
    x_min, x_max = 0, width
    y_min, y_max = 0, height
    coll_boundary_thickness = int(envconf["RADIUS_AGENT"])*2
    x_range = np.linspace(x_min + coll_boundary_thickness, 
                        x_max - coll_boundary_thickness, 
                        int((width - coll_boundary_thickness*2) / space_step))
    y_range = np.linspace(y_min + coll_boundary_thickness, 
                        y_max - coll_boundary_thickness, 
                        int((height - coll_boundary_thickness*2) / space_step))
    orient_range = np.arange(0, 2*np.pi, orient_step)

    # pack inputs for multiprocessing map
    mp_inputs = []
    seed = 0
    for x in x_range:
        for y in y_range:
            for orient in orient_range:
                init_info = x, y, orient, timesteps, extra, 'views'
                mp_inputs.append( (None, pv, None, seed, env_path, init_info, False) ) # model_tuple=None, load_dir=None
                seed += 1

    # run agent NNs in parallel
    with mp.Pool() as pool:
        results = pool.starmap_async( start, mp_inputs )
        pool.close()
        pool.join()

    results_list = results.get()
    for time_taken, dist_from_patch, output in results_list:
        view_set.update(output)
        view_set_all.update(output)

    with open(fr'{data_dir}/IDM/social_view_set.bin', 'wb') as f:
        pickle.dump(view_set_all, f)
    return view_set


def encode_one_hot(vis_field):

        num_class_elements = 6
        len_field = 8
        field_onehot = np.zeros((num_class_elements, len_field))
        for i,x in enumerate(vis_field):
            if x == 'wall_north': field_onehot[0,i] = 1
            elif x == 'wall_south': field_onehot[1,i] = 1
            elif x == 'wall_east': field_onehot[2,i] = 1
            elif x == 'wall_west': field_onehot[3,i] = 1
            elif x == 'agent_explore': field_onehot[4,i] = 1
            elif x == 'agent_exploit': field_onehot[5,i] = 1
            else: 
                print('error - nothing is perceived')

        return field_onehot


def COM_from_view(vis_input):

    # vis_input.shape = (6,8) # types, rays

    explore = vis_input[4,:]
    exploit = vis_input[5,:]
    both = explore + exploit

    if np.all(explore == 0): # no explore
        explore_avg = -1
        explore_only_avg = -1
    elif np.any(exploit != 0): # explore + also exploit
        explore_avg = np.argwhere(explore == 1).mean()
        explore_only_avg = -1
    else: # explore + no exploit
        explore_avg = np.argwhere(explore == 1).mean()
        explore_only_avg = explore_avg

    if np.all(exploit == 0): # no exploit
        exploit_avg = -1
        exploit_only_avg = -1
    elif np.any(explore != 0): # exploit + also explore
        exploit_avg = np.argwhere(exploit == 1).mean()
        exploit_only_avg = -1
    else: # exploit + no explore
        exploit_avg = np.argwhere(exploit == 1).mean()
        exploit_only_avg = exploit_avg

    if np.all(both == 0): # no either
        both_avg = -1
        both_only_avg = -1
        none = 0
    elif np.all(explore == 0) or np.all(exploit == 0): # both
        both_avg = np.argwhere(both == 1).mean()
        both_only_avg = -1
        none = -1
    else: # either
        both_avg = np.argwhere(both == 1).mean()
        both_only_avg = both_avg
        none = -1

    return explore_avg, exploit_avg, both_avg, explore_only_avg, exploit_only_avg, both_only_avg, none
    

def actions_per_view(name, gen_ext, view_set=None):

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    if view_set:
        save_name = f'actions_per_view_{name}_solo'
    else:
        save_name = f'actions_per_view_{name}_all'
        with open(fr'{data_dir}/IDM/social_view_set.bin', 'rb') as f:
            view_set = pickle.load(f)
    view_dict = dict.fromkeys(view_set, 0)
    print(f'actions_per_view {name} @ {len(view_set)} views')

    if os.path.exists(fr'{data_dir}/IDM/{save_name}.png'):
        print('traj already plotted')
        return

    # retrieve model
    gen_ext, valfit = find_top_val_gen(name, 'cen')
    with open(fr'{data_dir}/{name}/{gen_ext}_NNcen_pickle.bin','rb') as f:
        pv = pickle.load(f)
    env_path = fr'{data_dir}/{name}/.env'
    envconf = de.dotenv_values(env_path)
    NN,_ = reconstruct_NN(envconf,pv)

    # vis obs --> action
    angles = []
    avgs = []

    for vis_input in view_dict.keys():
        vis_input = encode_one_hot(vis_input)

        all_avgs = COM_from_view(vis_input)
        avgs.append(all_avgs)

        action,_,_,_ = NN.forward(vis_input, np.array([0]), None)
        angle = action * 90
        angles.append(angle)

    angles = np.array(angles)
    avgs = np.array(avgs)

    fig, axs = plt.subplots(1,3, figsize=(5,3))

    lab_list = ['Explore', 'Exploit', 'Both']
    for i in range(3):
        mask = (avgs[:,i+3] != -1) # take out non-hits
        angles_subset = angles[mask]
        avgs_subset = avgs[:,i+3][mask]
        avgs_subset -= 3.5 # center
        avgs_subset *= 72/3.5 # scale to FOV

        axs[i].hlines(0, -72, 72, 'k', alpha=.1)
        axs[i].vlines(0, -90, 90, 'k', alpha=.1)

        gam = LinearGAM(n_splines=5, verbose=False).gridsearch(avgs_subset, angles_subset, progress=False)
        XX = gam.generate_X_grid(term=0, n=100).flatten()

        # intervals = gam.prediction_intervals(XX, width=0.95)
        # axs[i].plot(XX, intervals, color="b", ls="--")
        # axs[i].fill_between(XX, intervals[:,0], intervals[:,1], color='b', ls='--', alpha=.2)

        axs[i].plot(avgs_subset, angles_subset, 'ko', markeredgecolor='none', markersize=5, alpha=1/255)
        axs[i].plot(XX, gam.predict(XX), 'C0--')

        # non-social obs
        angles_subset = angles[(avgs[:,6] != -1)]
        # print(np.min(angles_subset), np.max(angles_subset), np.median(angles_subset), np.std(angles_subset))
        axs[i].plot(0, np.median(angles_subset), 'C0o', markersize=5, alpha=.8)

        axs[i].set_xlabel(lab_list[i])
        axs[i].set_ylim(-92,92)
        
        axs[i].set_xticks(np.linspace(-72,72,3))
        axs[i].set_yticks(np.linspace(-90,90,7))

    axs[0].set_ylabel('NN Output Angle')

    plt.tight_layout()
    plt.savefig(fr'{data_dir}/IDM/{save_name}.png', dpi=dpi)
    plt.close()


# -------------------------- dists -------------------------- #

def build_patch_dists(exp_name, gen_ext, space_step, orient_step, timesteps, extra=''):
    print(f'building {exp_name} dists')

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    NN_pv_path = fr'{data_dir}/{exp_name}/{gen_ext}_NNcen_pickle.bin'
    with open(NN_pv_path,'rb') as f:
        pv = pickle.load(f)
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)

    if extra == '': 
        extra = envconf['N']

    width, height = tuple(eval(envconf["ENV_SIZE"]))
    x_min, x_max = 0, width
    y_min, y_max = 0, height
    coll_boundary_thickness = int(envconf["RADIUS_AGENT"])*2
    x_range = np.linspace(x_min + coll_boundary_thickness, 
                        x_max - coll_boundary_thickness, 
                        int((width - coll_boundary_thickness*2) / space_step))
    y_range = np.linspace(y_min + coll_boundary_thickness, 
                        y_max - coll_boundary_thickness, 
                        int((height - coll_boundary_thickness*2) / space_step))
    orient_range = np.arange(0, 2*np.pi, orient_step)

    num_inits = len(x_range) * len(y_range) * len(orient_range)
    traj_matrix = np.zeros( (num_inits, timesteps) )

    mp_inputs = []
    seed = 0
    for x in x_range:
        for y in y_range:
            for orient in orient_range:
                init_info = x, y, orient, timesteps, extra, 'dists'
                mp_inputs.append( (None, pv, None, seed, env_path, init_info, False) ) # model_tuple=None, load_dir=None
                seed += 1

    with mp.Pool() as pool:
        results = pool.starmap_async( start, mp_inputs )
        pool.close()
        pool.join()

    results_list = results.get()
    for n, (time_taken, dist_from_patch, output) in enumerate(results_list):
        traj_matrix[n,:] =  output

    with open(fr'{data_dir}/dists/{exp_name}_{gen_ext}.bin', 'wb') as f:
        pickle.dump(traj_matrix, f)


def patch_timeXdist(name, gen_ext, plot_type=''):
    print('plotting patch_timeXdist', name, plot_type)

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    with open(fr'{data_dir}/dists/{name}_{gen_ext}.bin', 'rb') as f:
        data = pickle.load(f)

    fig, axs = plt.subplots(figsize=(3,3))

    num_runs, num_timesteps = data.shape
    time = np.linspace(0,num_timesteps-1,num_timesteps)

    if plot_type == '':
        # lines
        for i in range(num_runs):
            dist_at_run = data[i,:]
            axs.plot(time, dist_at_run, 'k', alpha=1/255)

        # time_all = []
        # dist_all = []
        # for i in range(num_runs):
        #     dist = data[i,:]
        #     axs.plot(time, dist, 'k', alpha=1/255)
        #     time_all.append(time)
        #     dist_all.append(dist)
        # time_all = np.array(time_all).flatten()
        # dist_all = np.array(dist_all).flatten()

        # gam = LinearGAM(n_splines=5, verbose=False).gridsearch(time_all, dist_all, progress=False)
        # XX = gam.generate_X_grid(term=0, n=100).flatten()

        # intervals = gam.prediction_intervals(XX, width=0.95)
        # axs.plot(XX, intervals, 'C0--')
        # axs.fill_between(XX, intervals[:,0], intervals[:,1], color='b', ls='--', alpha=.2)

        # axs.plot(XX, gam.predict(XX), 'r--')

        axs.set_xlim([0,num_timesteps])
        axs.set_ylim([0,800])

        axs.set_xlabel('Time')
        axs.set_ylabel('Distance to Patch')
    
    elif plot_type == '_bars':

        dists_by_category = form_timeXdist_dict(data, thresh=100)

        bottom = np.zeros(num_timesteps)
        for category, num in dists_by_category.items():
            num = np.array(num)
            p = axs.bar(time, num/num_runs, width=1, label=category, bottom=bottom)
            bottom += num/num_runs

            # print(category, round(np.sum(num/num_runs)/num_timesteps, 2))

        axs.set_xticks(np.linspace(0, num_timesteps, 6))
        axs.set_xlabel('Time')
        axs.set_ylabel('Frequency')
        # axs.legend()

    plt.tight_layout()
    plt.savefig(fr'{data_dir}/{name}_{gen_ext}{plot_type}.png', dpi=dpi)
    plt.close()


def form_timeXdist_dict(data, thresh):

    dists_by_category = {
        'near start':[],
        'in transit':[],
        'near patch':[],
        'at patch':[]
    }

    num_runs, num_timesteps = data.shape
    dist_at_start = data[:,0]

    for t in range(num_timesteps):
        dist_at_time = data[:,t]

        num_at_patch = np.sum(dist_at_time == 0)
        num_near_patch = np.sum((dist_at_time <= thresh) & (dist_at_time > 0)) # near + not at patch
        num_near_start = np.sum((dist_at_start - dist_at_time <= thresh) & (dist_at_time > thresh)) # close to start + not near patch
        num_in_transit = num_runs - num_at_patch - num_near_patch - num_near_start

        dists_by_category['near start'].append(num_near_start)
        dists_by_category['in transit'].append(num_in_transit)
        dists_by_category['near patch'].append(num_near_patch)
        dists_by_category['at patch'].append(num_at_patch)
    
    return dists_by_category


# -------------------------- spins -------------------------- #

def build_agent_spins(exp_name, gen_ext, spin_angle, extra=''):
    print(f'building {spin_angle} spin')

    # pull pv + envconf from save folders
    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    exp_name = 'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_rep0' # dummy
    gen_ext = 'gen988' # dummy
    with open(fr'{data_dir}/{exp_name}/{gen_ext}_NNcen_pickle.bin','rb') as f:
        pv = pickle.load(f)
    env_path = fr'{data_dir}/{exp_name}/.env'

    # determines initial positions for perceived agent (focal agent is stationary)
    # z_range = np.linspace(0, 1000, 41)
    # z_range = np.linspace(25, 1000, 40) # +/- 25 noise
    z_range = np.linspace(25, 1200, 48) # +/- 25 noise

    # each run varies by total timesteps but normalized by 1 revolution (spin_angle = 1 --> 90 deg turns)
    # timesteps = int(4 / spin_angle)
    timesteps = int(4 / spin_angle) * 1000 # noise

    # construct matrix of each traj for each grid pos/dir
    traj_matrix = np.zeros( (len(z_range), timesteps, 4) ) # (pos_x, pos_y, ori, detection_count)

    # pack inputs for multiprocessing map
    mp_inputs = []
    seed = 0
    for z in z_range:
        init_info = z, spin_angle, timesteps, extra
        mp_inputs.append( (None, pv, None, seed, env_path, init_info) ) # model_tuple=None, load_dir=None
        seed += 1

    # run agent NNs in parallel
    with mp.Pool() as pool:
        results = pool.starmap_async( start, mp_inputs )
        pool.close()
        pool.join()

    # unpack results into matrix (y coords transformed for plotting)
    results_list = results.get()

    empties = []
    for n, (time_taken, dist_from_patch, output) in enumerate(results_list):
        traj_matrix[n,:,:] =  output
        if not output[0,:].any():
            empties.append(n) 
    for n in empties:
        traj_matrix = np.delete(traj_matrix, n, axis=0)

    # transform to distance X detec prob
    detect_count = traj_matrix[:,:,3]
    detect_avg = np.average(detect_count, axis=1)

    save_name = fr'{data_dir}/spins/angle{spin_angle}.bin'
    with open(save_name, 'wb') as f:
        pickle.dump(detect_avg, f)


def plot_spin_chart(dpi=100):
    print(f'plot spin chart')

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    exp_name = 'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_rep0' # dummy
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)

    # angles = [.0125, .025, .05, .1, .2, .3, .4]
    angles = np.linspace(1, 90, 90)/90
    angles = angles.round(3)

    counts = []
    for angle in angles:
        with open(fr'{data_dir}/spins/angle{angle}.bin', 'rb') as f:
            detect_avg = pickle.load(f) # [z, detection_count]
        counts.append(detect_avg)

    # z_range = np.linspace(0, 1000, 41)
    # z_range = np.linspace(25, 1000, 40)
    z_range = np.linspace(25, 1200, 48)

    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(angles)))

    # fig, axes = plt.subplots(figsize=(4.15,3.15))
    fig, axes = plt.subplots(figsize=(4.25,3.25))
    for i,angle in enumerate(angles):
        deg = int(angle*90)
        axes.plot(z_range, counts[i], label=f'{deg}$^\circ$', c=colors[i], alpha=.25)
        # print(angle, deg)
        # print(z_range[2],'',counts[i][2])
        # print(z_range[7],'',counts[i][7])
        # print(z_range[19],'',counts[i][19])
        # print(z_range[29],'',counts[i][29])
        # print('')

    axes.vlines(67, 0, 0.5, color='grey', linestyle='dashed', alpha=0.5)
    axes.vlines(497, 0, 0.5, color='grey', linestyle='dashed', alpha=0.5)
    axes.annotate('Within 100 Units', xy=(67, 0.125), xytext=(5,0), textcoords='offset points', color='darkslategrey', fontsize=8, rotation=90, va='center')
    axes.annotate('All Map Initialization', xy=(497, 0.3), xytext=(5,0), textcoords='offset points', color='darkslategrey', fontsize=8, rotation=90, va='center')

    # axes.legend(title='Spin Angle', loc='upper right')

    axes.set_xlabel('Distance Between Agents')
    axes.set_ylabel('Detection Probability')
    # axes.set_ylim(-20,1020)

    plt.tight_layout()
    plt.savefig(fr'{data_dir}/spins/detect-prob_all-angles.png', dpi=dpi)
    plt.close()



def plot_social_orient_corr(exp_name, gen_ext, space_step, orient_step, timesteps, rank='cen', eye=True, extra='', archive=False, dpi=None):
    print(f'plotting corr - {exp_name}')

    save_data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    if archive:
        data_dir = Path(__file__).parent.parent / r'data/simulation_data/archive - ISBDDP/'
    else:
        data_dir = save_data_dir

    if extra == '':
        save_name = fr'{save_data_dir}/traj_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e{int(eye)}'
    else:
        save_name = fr'{save_data_dir}/traj_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e{int(eye)}{extra}'
    if not os.path.exists(save_name+'.bin'):
        print(f'no data found for {save_name}')
        return 0,0,0,0
    with open(save_name+'.bin', 'rb') as f:
        ag_data = pickle.load(f)
    save_name = fr'{save_data_dir}/corrs/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e{int(eye)}'

    num_runs,t_len,_ = ag_data.shape
    t = np.linspace(0,t_len,t_len)
    # print(f'ag_data shape: {num_runs, len(t)}')

    # corr to init
    delay = 0
    # t = t[:-delay] # if there is a delay
    t_len -= delay

    x = ag_data[:,delay:,0]
    y = ag_data[:,delay:,1]
    pt_self = np.array([x,y])

    orient = ag_data[:,delay:,2]
    x_COM = ag_data[:,delay:,4]
    y_COM = ag_data[:,delay:,5]
    pt_COM = np.array([x_COM,y_COM])
    disp = pt_self.transpose() - pt_COM.transpose()

    # corr to COM
    angle_to_target = np.arctan2(disp[:,:,1], disp[:,:,0]) + np.pi # shift by pi for [0-2pi]
    # print('max/min angle_to_target: ', np.max(angle_to_target), np.min(angle_to_target))
    corr_COM_angle_diff = angle_to_target.transpose() - orient
    corr_COM_angle_diff_scaled = (corr_COM_angle_diff - np.pi) % (2*np.pi) - np.pi # [-pi/2, pi/2]
    corr_COM = np.cos(corr_COM_angle_diff)

    # distance to patch
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)
    pt_target = np.array(eval(envconf["RESOURCE_POS"]))
    pt_target[1] = 1000 - pt_target[1]
    disp = pt_self.transpose() - pt_target
    dist = np.linalg.norm(disp, axis=2)


    ### temporal correlations ###

    fig = plt.figure(figsize=(3,3))
    ax0 = plt.subplot()

    corr_COM_avg = np.mean(corr_COM, 0)
    ax0.plot(t, corr_COM_avg, 'k')
    ax0.axhline(color='gray', ls='--')

    # decorr_idx = np.argmax(corr_COM_avg < 0.5)
    # decorr_time = t[decorr_idx]
    # decorr_val = corr_COM_avg[decorr_idx]

    # ax0.vlines(decorr_time, 0, decorr_val, color='red', ls='--')
    # # ax0.set_title(f'Decorr Time: {decorr_time:.2f}')

    ax0.set_xlim(-20,520)
    ax0.set_ylim(-1.05,1.05)
    # ax0.set_ylim(-0.05,1.05)
    ax0.set_xlabel('Timesteps')
    ax0.set_ylabel('Orientation Correlation')

    plt.tight_layout()
    plt.savefig(fr'{save_name}{extra}_corr_orient_COM_{dpi}.png', dpi=dpi)
    plt.close()


    ### spatial correlation trajectories ###

    fig = plt.figure(figsize=(3,3))
    ax1 = plt.subplot()

    for r in range(num_runs)[::50]:
        ax1.plot(dist[:,r], -corr_COM_angle_diff_scaled[r,:], c='k', alpha=5/255) # negative to match orientation of polar plot

    ins = ax1.inset_axes([0.7,0.05,0.25,0.25])
    ins.set_yticks([])
    ins.set_xticks([])

    inits = [
        [800, 200, np.pi], #BR-W
        [800, 900, np.pi/2], #TR-W
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

        # ax[0,0].plot(t, corr_init[index,:], c=color, alpha=.5)
        # ax[0,2].plot(dist[:,index], corr_init[index,:], c=color, alpha=.5)
        # ax[0,3].plot(dist[:,index], corr_init_angle_diff[index,:], c=color, alpha=.5)
        # ax[1,0].plot(t, corr_patch[:,index], c=color, alpha=.5)
        # ax[1,2].plot(dist[:,index], corr_patch[:,index], c=color, alpha=.5)
        # ax[1,3].plot(dist[:,index], corr_patch_angle_diff[:,index], c=color, alpha=.5)
        ins.plot(dist[:,index], -corr_COM_angle_diff_scaled[index,:], c=color, alpha=.5, linewidth=1)

    labels = [r'-$\pi$', r'-$\pi/2$', '$0$', r'$\pi/2$', r'$\pi$']

    ax1.set_xlim(-20,820)
    ax1.set_yticks(np.arange(-np.pi, np.pi+0.01, np.pi/2))
    ax1.set_yticklabels(labels)
    ax1.set_xlabel('Distance to Patch')
    ax1.set_ylabel('Relative Orientation')

    plt.tight_layout()
    plt.savefig(fr'{save_name}{extra}_corr_trajs_COM_{dpi}.png', dpi=dpi)
    plt.close()


    ### spatial correlation  - polar hist ###

    fig = plt.figure(figsize=(3,3))
    ax2 = plt.subplot(projection='polar')

    dist = dist.flatten()
    m = np.ma.masked_less(dist, 100)

    corr_COM_angle_diff = corr_COM_angle_diff_scaled.transpose().flatten()
    corr_COM_angle_diff_masked = (1-m.mask)*corr_COM_angle_diff
    corr_COM_angle_diff_comp = corr_COM_angle_diff_masked[corr_COM_angle_diff_masked != 0]

    # Visualise by area of bins
    n, bins, patches = circular_hist(ax2, corr_COM_angle_diff_comp, bins=100, offset=np.pi/2, colored=True)
    # # Visualise by radius of bins
    # circular_hist(ax[1], corr_init_angle_diff_comp, bins=100, offset=np.pi/2, density=False)

    x_avg = np.mean(np.cos(corr_COM_angle_diff_comp))
    y_avg = np.mean(np.sin(corr_COM_angle_diff_comp))
    histo_avg = np.arctan2(y_avg, x_avg)

    ax2.axvline(histo_avg, color='gray', ls='--')

    plt.tight_layout()
    plt.savefig(fr'{save_name}{extra}_corr_polar_COM_{dpi}.png', dpi=dpi)
    plt.close()

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

        orient_step = np.pi/8
        save_name_trajmap = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e{int(eye)}_ex_lines'
        if os.path.exists(save_name_trajmap+'_50.png'):
            print('traj already plotted')
        else:
            # traj data
            orient_step = np.pi/8
            save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e{int(eye)}.bin'
            traj_exists = False
            if os.path.exists(save_name_traj):
                print('traj already built')
                traj_exists = True
            else:
                build_agent_trajs(name, gen, space_step, orient_step, timesteps)

            save_name_trajmap = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e{int(eye)}_ex_lines'
            if os.path.exists(save_name_trajmap+'_100.png'):
                print('traj already plotted at dpi100')
            elif os.path.exists(save_name_trajmap+'_50.png'):
                print('traj already plotted at dpi50')
            elif os.path.exists(save_name_trajmap+'.png'):
                print('traj already plotted')
            else:
                plot_agent_trajs(name, gen, space_step, orient_step, timesteps, ex_lines=True, dpi=dpi)

        # act_mean, act_min, act_max = plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type='_avgact', dpi=dpi)
        # _, _, _ = plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type='_avgori', dpi=dpi)
        # len_mean, len_min, len_max = plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type='_avglen', dpi=dpi)
        # de_mean, de_min, de_max = plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type='_dirent', dpi=dpi)

        # corr_peaks, decorr_time, histo_avg_init, histo_avg_patch, histo_peaks_init, histo_peaks_patch, dirent_init, dirent_patch = plot_agent_orient_corr(name, gen, space_step, orient_step, timesteps, dpi=dpi)

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
        # else:
        #     print('building action map')
        #     min_action, max_action = build_action_matrix(name, gen, space_step, orient_step)

        # avglen_mean, avglen_med, avglen_min, avglen_max, basin_patch_dist = plot_action_vecfield(name, gen, space_step, orient_step, plot_type='_avg', colored='len', dpi=dpi)
        # mean, med, min, max, basin_patch_dist = plot_action_vecfield(name, gen, space_step, orient_step, plot_type='_avg', colored='ori', dpi=dpi)

        # data[name] = (corr_peaks, decorr_time, histo_avg_init, histo_avg_patch, histo_peaks_init, histo_peaks_patch, dirent_init, dirent_patch, act_mean, act_min, act_max, len_mean, len_min, len_max, de_mean, de_min, de_max, basin_patch_dist)
        # print(f'data dict len: {len(data)}')
        # print('')

        # with open(fr'{data_dir}/traj_matrices/{group_name}.bin', 'wb') as f:
        #     pickle.dump(data, f)

        # delete .bin files if not already saved
        # if not traj_exists:
        save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o8_t{timesteps}_{rank}_e{int(eye)}.bin'
        if os.path.exists(save_name_traj):
            os.remove(save_name_traj)
        # if not action_exists:
        #     save_name_act = fr'{data_dir}/action_maps/{name}_{gen}_c{space_step}_o32_action.bin'
        #     if os.path.exists(save_name_act):
        #         os.remove(save_name_act)



def run_gamut_social(group_name, names, dpi):

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    rank = 'cen'
    space_step = 25
    timesteps = 500
    orient_step = np.pi/8

    for name in names:

        gen, valfit = find_top_val_gen(name, rank)
        print(f'{name} @ {gen} w {valfit} fitness')

        # with open(fr'{data_dir}/traj_matrices/{group_name}.bin', 'rb') as f:
        #     data = pickle.load(f)
        # if name in data:
        #     print('already there')
        #     print('')

        #     # save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e1.bin'
        #     # if os.path.exists(save_name_traj):
        #     #     os.remove(save_name_traj)
        #     # save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e1_nosocial.bin'
        #     # if os.path.exists(save_name_traj):
        #     #     os.remove(save_name_traj)
        #     # save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e1_ghost_exploiter.bin'
        #     # if os.path.exists(save_name_traj):
        #     #     os.remove(save_name_traj)
        #     # save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e1_ghost_explorer.bin'
        #     # if os.path.exists(save_name_traj):
        #     #     os.remove(save_name_traj)

        #     continue

        save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e1.bin'
        if not os.path.exists(save_name_traj):
            build_agent_trajs(name, gen, space_step, orient_step, timesteps)
        save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e1_nosocial.bin'
        if not os.path.exists(save_name_traj):
            build_agent_trajs(name, gen, space_step, orient_step, timesteps, extra='nosocial')
        save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e1_ghost_exploiter.bin'
        if not os.path.exists(save_name_traj):
            build_agent_trajs(name, gen, space_step, orient_step, timesteps, extra='ghost_exploiter')
        save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e1_ghost_explorer.bin'
        if not os.path.exists(save_name_traj):
            build_agent_trajs(name, gen, space_step, orient_step, timesteps, extra='ghost_explorer')

        save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e1_ex_lines_50.png'
        if not os.path.exists(save_name_traj):
            plot_agent_trajs(name, gen, space_step, orient_step, timesteps, ex_lines=True, dpi=dpi)
        save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e1_nosocial_ex_lines_50.png'
        if not os.path.exists(save_name_traj):
            plot_agent_trajs(name, gen, space_step, orient_step, timesteps, ex_lines=True, dpi=dpi, extra='nosocial')
        save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e1_ghost_exploiter_ex_lines_50.png'
        if not os.path.exists(save_name_traj):
            plot_agent_trajs(name, gen, space_step, orient_step, timesteps, ex_lines=True, dpi=dpi, extra='ghost_exploiter')
        save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e1_ghost_explorer_ex_lines_50.png'
        if not os.path.exists(save_name_traj):
            plot_agent_trajs(name, gen, space_step, orient_step, timesteps, ex_lines=True, dpi=dpi, extra='ghost_explorer')

        de_mean_OG,_,_ = plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type='_dirent', dpi=dpi)
        de_mean_NS,_,_ = plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type='_dirent', extra='nosocial', mask_cond='', dpi=dpi)
        # de_mean_NS_patchonly,_,_ = plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type='_dirent', extra='nosocial', mask_cond='patch_only', dpi=dpi)
        de_mean_ET,_,_ = plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type='_dirent', extra='ghost_exploiter', mask_cond='', dpi=dpi)
        # de_mean_ET_patchonly,_,_ = plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type='_dirent', extra='nosocial', mask_cond='patch_only', dpi=dpi)
        de_mean_ER,_,_ = plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type='_dirent', dpi=dpi, extra='ghost_explorer')

        JS_mean_OGNS,_,_ = plot_traj_vecfield_perturb_div(name, gen, space_step, orient_step, timesteps, plot_type='JS', dpi=dpi, base_cond='nosocial', perturb_cond='')
        JS_mean_NSET,_,_ = plot_traj_vecfield_perturb_div(name, gen, space_step, orient_step, timesteps, plot_type='JS', base_cond='nosocial', perturb_cond='ghost_exploiter', mask_cond='', dpi=dpi)
        # JS_mean_NSET_patchonly,_,_ = plot_traj_vecfield_perturb_div(name, gen, space_step, orient_step, timesteps, plot_type='JS', base_cond='nosocial', perturb_cond='ghost_exploiter', mask_cond='patch_only', dpi=dpi)
        JS_mean_NSER,_,_ = plot_traj_vecfield_perturb_div(name, gen, space_step, orient_step, timesteps, plot_type='JS', dpi=dpi, base_cond='nosocial', perturb_cond='ghost_explorer')
        JS_mean_ETER,_,_ = plot_traj_vecfield_perturb_div(name, gen, space_step, orient_step, timesteps, plot_type='JS', dpi=dpi, base_cond='ghost_exploiter', perturb_cond='ghost_explorer')

        with open(fr'{data_dir}/traj_matrices/{group_name}.bin', 'rb') as f:
            data = pickle.load(f)
        # print(f'data dict len: {len(data)}')

        data[name] = (
            de_mean_OG, de_mean_NS, de_mean_ET, de_mean_ER,
            JS_mean_OGNS, JS_mean_NSET, JS_mean_NSER, JS_mean_ETER,
            )
        # data[name] = (
        #     de_mean_OG, de_mean_NS, de_mean_ET, de_mean_ER,
        #     JS_mean_OGNS, JS_mean_NSET, JS_mean_NSER, JS_mean_ETER,
        #     de_mean_NS_patchonly, de_mean_ET_patchonly, JS_mean_NSET_patchonly
        #     )

        with open(fr'{data_dir}/traj_matrices/{group_name}.bin', 'wb') as f:
            pickle.dump(data, f)

        # delete .bin files if not already saved
        save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e1.bin'
        if os.path.exists(save_name_traj):
            os.remove(save_name_traj)
        save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e1_nosocial.bin'
        if os.path.exists(save_name_traj):
            os.remove(save_name_traj)
        save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e1_ghost_exploiter.bin'
        if os.path.exists(save_name_traj):
            os.remove(save_name_traj)
        save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e1_ghost_explorer.bin'
        if os.path.exists(save_name_traj):
            os.remove(save_name_traj)




def run_gamut_social_extra(group_name, names, dpi):

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    rank = 'cen'
    space_step = 25
    timesteps = 500
    orient_step = np.pi/8

    with open(fr'{data_dir}/traj_matrices/{group_name}.bin', 'rb') as f:
        data = pickle.load(f)

    for name in names:

        gen, valfit = find_top_val_gen(name, rank)
        # print(f'{name} @ {gen} w {valfit} fitness')

        # if len(data[name]) == 8:
        #     # print(f'{name} @ {gen} w {valfit} fitness')
        #     continue
        # elif len(data[name]) == 11:
        #     print(f'skip {name}, already done')
        #     continue
        # else:
        #     print(f'remove {name}, do again')
        #     # del data[name]
        #     # with open(fr'{data_dir}/traj_matrices/{group_name}.bin', 'wb') as f:
        #     #     pickle.dump(data, f)
        #     continue
        if name in data:
            print('already there')
            continue

        save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e1_nosocial.bin'
        if not os.path.exists(save_name_traj):
            build_agent_trajs(name, gen, space_step, orient_step, timesteps, extra='nosocial')
        save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e1_ghost_exploiter.bin'
        if not os.path.exists(save_name_traj):
            build_agent_trajs(name, gen, space_step, orient_step, timesteps, extra='ghost_exploiter')

        de_mean_NS_new,_,_ = plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type='_dirent', extra='nosocial', mask_cond='', dpi=dpi)
        de_mean_NS_patchonly,_,_ = plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type='_dirent', extra='nosocial', mask_cond='patch_only', dpi=dpi)
        de_mean_ET_new,_,_ = plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type='_dirent', extra='ghost_exploiter', mask_cond='', dpi=dpi)
        de_mean_ET_patchonly,_,_ = plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type='_dirent', extra='ghost_exploiter', mask_cond='patch_only', dpi=dpi)

        JS_mean_NSET_new,_,_ = plot_traj_vecfield_perturb_div(name, gen, space_step, orient_step, timesteps, plot_type='JS', dpi=dpi, 
                                                            base_cond='nosocial', perturb_cond='ghost_exploiter', mask_cond='')
        JS_mean_NSET_patchonly,_,_ = plot_traj_vecfield_perturb_div(name, gen, space_step, orient_step, timesteps, plot_type='JS', dpi=dpi, 
                                                            base_cond='nosocial', perturb_cond='ghost_exploiter', mask_cond='patch_only')

        # with open(fr'{data_dir}/traj_matrices/{group_name}.bin', 'rb') as f:
        with open(fr'{data_dir}/traj_matrices/gamut_social.bin', 'rb') as f:
            data = pickle.load(f)

        de_mean_OG, de_mean_NS, de_mean_ET, de_mean_ER, JS_mean_OGNS, JS_mean_NSET, JS_mean_NSER, JS_mean_ETER = data[name]
        data[name] = de_mean_OG, de_mean_NS_new, de_mean_ET_new, de_mean_ER, JS_mean_OGNS, JS_mean_NSET_new, JS_mean_NSER, JS_mean_ETER, de_mean_NS_patchonly, de_mean_ET_patchonly, JS_mean_NSET_patchonly

        with open(fr'{data_dir}/traj_matrices/{group_name}.bin', 'wb') as f:
            pickle.dump(data, f)

        print(f'de_mean_NS-og: {de_mean_NS}, de_mean_NS-new: {de_mean_NS_new}, -patchonly: {de_mean_NS_patchonly}')
        print(f'de_mean_ET-og: {de_mean_ET}, de_mean_ET-new: {de_mean_ET_new}, -patchonly: {de_mean_ET_patchonly}')
        print(f'JS_mean_NSET-og: {JS_mean_NSET}, JS_mean_NSET-new: {JS_mean_NSET_new}, -patchonly: {JS_mean_NSET_patchonly}')

        save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e1_nosocial.bin'
        if os.path.exists(save_name_traj):
            os.remove(save_name_traj)
        save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e1_ghost_exploiter.bin'
        if os.path.exists(save_name_traj):
            os.remove(save_name_traj)


def run_gamut_social_redo(group_name, names, dpi):

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    rank = 'cen'
    space_step = 25
    timesteps = 500
    orient_step = np.pi/8

    for name in names:

        gen, valfit = find_top_val_gen(name, rank)
        print(f'{name} @ {gen} w {valfit} fitness')

        save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e1.bin'
        if not os.path.exists(save_name_traj):
            build_agent_trajs(name, gen, space_step, orient_step, timesteps)
        save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e1_nosocial.bin'
        if not os.path.exists(save_name_traj):
            build_agent_trajs(name, gen, space_step, orient_step, timesteps, extra='nosocial')
        save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e1_ghost_exploiter.bin'
        if not os.path.exists(save_name_traj):
            build_agent_trajs(name, gen, space_step, orient_step, timesteps, extra='ghost_exploiter')
        save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e1_ghost_explorer.bin'
        if not os.path.exists(save_name_traj):
            build_agent_trajs(name, gen, space_step, orient_step, timesteps, extra='ghost_explorer')

        plot_agent_trajs(name, gen, space_step, orient_step, timesteps, ex_lines=True, dpi=dpi)
        plot_agent_trajs(name, gen, space_step, orient_step, timesteps, ex_lines=True, dpi=dpi, extra='nosocial')
        plot_agent_trajs(name, gen, space_step, orient_step, timesteps, ex_lines=True, dpi=dpi, extra='ghost_exploiter')
        plot_agent_trajs(name, gen, space_step, orient_step, timesteps, ex_lines=True, dpi=dpi, extra='ghost_explorer')

        de_mean_OG,_,_ = plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type='_dirent', dpi=dpi)
        de_mean_NS,_,_ = plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type='_dirent', extra='nosocial', mask_cond='', dpi=dpi)
        de_mean_ET,_,_ = plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type='_dirent', extra='ghost_exploiter', mask_cond='', dpi=dpi)
        de_mean_ER,_,_ = plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type='_dirent', dpi=dpi, extra='ghost_explorer')

        JS_mean_OGNS,_,_ = plot_traj_vecfield_perturb_div(name, gen, space_step, orient_step, timesteps, plot_type='JS', dpi=dpi, base_cond='nosocial', perturb_cond='')
        JS_mean_NSET,_,_ = plot_traj_vecfield_perturb_div(name, gen, space_step, orient_step, timesteps, plot_type='JS', base_cond='nosocial', perturb_cond='ghost_exploiter', mask_cond='', dpi=dpi)
        JS_mean_NSER,_,_ = plot_traj_vecfield_perturb_div(name, gen, space_step, orient_step, timesteps, plot_type='JS', dpi=dpi, base_cond='nosocial', perturb_cond='ghost_explorer')
        JS_mean_ETER,_,_ = plot_traj_vecfield_perturb_div(name, gen, space_step, orient_step, timesteps, plot_type='JS', dpi=dpi, base_cond='ghost_exploiter', perturb_cond='ghost_explorer')

        update = (
            de_mean_OG, de_mean_NS, de_mean_ET, de_mean_ER,
            JS_mean_OGNS, JS_mean_NSET, JS_mean_NSER, JS_mean_ETER,
            )
        with open(fr'{data_dir}/traj_matrices/{group_name}.bin', 'rb') as f:
            data = pickle.load(f)
        print(f'same? {update == data[name]}')
        if not update == data[name]:
            for x,(i,j) in enumerate(zip(update, data[name])):
                print(f'same? {i == j}')
        data[name] = update

        with open(fr'{data_dir}/traj_matrices/{group_name}.bin', 'wb') as f:
            pickle.dump(data, f)

        with open(fr'{data_dir}/dups_alt.bin', 'rb') as f:
            dups = pickle.load(f)
        dups.remove(name)
        with open(fr'{data_dir}/dups_alt.bin', 'wb') as f:
            pickle.dump(dups, f)

        save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e1.bin'
        if os.path.exists(save_name_traj):
            os.remove(save_name_traj)
        save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e1_nosocial.bin'
        if os.path.exists(save_name_traj):
            os.remove(save_name_traj)
        save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e1_ghost_exploiter.bin'
        if os.path.exists(save_name_traj):
            os.remove(save_name_traj)
        save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e1_ghost_explorer.bin'
        if os.path.exists(save_name_traj):
            os.remove(save_name_traj)


def analyze_gamut(group_type, plot_type):

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    # with open(fr'{data_dir}/traj_matrices/gamut_labeled.bin', 'rb') as f:
    #     data = pickle.load(f)
    # print(f'data dict len: {len(data)}')
    with open(fr'{data_dir}/traj_matrices/archive - ISBDDP/gamut_visall_nodist_labeled.bin', 'rb') as f:
        data1 = pickle.load(f)
    with open(fr'{data_dir}/traj_matrices/archive - ISBDDP/gamut_vis8_dist_labeled.bin', 'rb') as f:
        data2 = pickle.load(f)
    data = data1|data2
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

        with open(fr'{data_dir}/archive - ISBDDP/{name}/val_matrix_cen.bin','rb') as f:
            val_matrix = pickle.load(f)
        fitness = np.mean(val_matrix)

        if len(data[name][0]) != 18:
            print(f'skipping {name}')
            continue
        bpdist = list(data[name][0])[17]

        data_tuple, label = data[name]

        if 'vis6' in name:
            vis6.append((name, data_tuple, fitness, bpdist, label))
        elif 'vis8' in name and 'dist' not in name and 'CNN12' not in name:
            vis8.append((name, data_tuple, fitness, bpdist, label))
        elif 'vis10' in name:
            vis10.append((name, data_tuple, fitness, bpdist, label))
        elif 'vis12' in name:
            vis12.append((name, data_tuple, fitness, bpdist, label))
        elif 'vis14' in name:
            vis14.append((name, data_tuple, fitness, bpdist, label))
        elif 'vis16' in name:
            vis16.append((name, data_tuple, fitness, bpdist, label))
        elif 'vis18' in name:
            vis18.append((name, data_tuple, fitness, bpdist, label))
        elif 'vis20' in name:
            vis20.append((name, data_tuple, fitness, bpdist, label))
        elif 'vis24' in name:
            vis24.append((name, data_tuple, fitness, bpdist, label))
        elif 'vis32' in name:
            vis32.append((name, data_tuple, fitness, bpdist, label))
        elif 'maxWF' in name:
            maxWF.append((name, data_tuple, fitness, bpdist, label))
        elif 'p9WF' in name:
            p8WF.append((name, data_tuple, fitness, bpdist, label))
        elif 'p8WF' in name:
            p6WF.append((name, data_tuple, fitness, bpdist, label))
        elif 'mlWF' in name:
            p5WF.append((name, data_tuple, fitness, bpdist, label))
        elif 'mWF' in name:
            p4WF.append((name, data_tuple, fitness, bpdist, label))
        elif 'msWF' in name:
            p3WF.append((name, data_tuple, fitness, bpdist, label))
        elif '_sWF' in name:
            p2WF.append((name, data_tuple, fitness, bpdist, label))
        elif 'ssWF' in name:
            p1WF.append((name, data_tuple, fitness, bpdist, label))
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
    cmap_points = plt.get_cmap('plasma')
    norm = plt.Normalize(0,500)
    num_groups = len(group_list)

    if group_type == 'vis':
        num_labels = 2
    elif group_type == 'dist':
        num_labels = 3
    width = 1/num_labels

    vlabs = []
    labs_taken = []

    for g_num, (group_name, group_data) in enumerate(zip(group_list_str, group_list)):

        BD_fit,IS_fit,DP_fit = [],[],[]
        BD_bpd,IS_bpd,DP_bpd = [],[],[]
        labels = ['BD','IS','DP','BD_IS','IS_DP','DP_BD']
        colors = ['cornflowerblue', 'tomato', 'forestgreen', 'darkorchid', 'gold', 'aquamarine']

        for name, data_tuple, fitness, bpdist, label in group_data:

            if label == 'BD':
                BD_fit.append(int(fitness))
                BD_bpd.append(int(bpdist))
            elif label == 'IS':
                IS_fit.append(int(fitness))
                IS_bpd.append(int(bpdist))
            elif label == 'DP':
                DP_fit.append(int(fitness))
                DP_bpd.append(int(bpdist))
        ax1.axvline(x = g_num - width/2, color='k', linestyle='--', linewidth=0.5)

        for l_num, (fit, bpd, label, color) in enumerate(zip([BD_fit,IS_fit,DP_fit], [BD_bpd,IS_bpd,DP_bpd], labels, colors)):
            pos = g_num + l_num/num_labels

            if plot_type == 'fitness': 
                data = fit
                other = bpd
            elif plot_type == 'basin_patch_dist': 
                data = bpd
                other = fit
            print(group_name, label, len(data), data)

            if data:
                if len(data) < 5:
                    # # continue
                    # l0 = ax1.violinplot(data, 
                    #             positions=[pos],
                    #             widths=width, 
                    #             showmedians=False, 
                    #             showextrema=False,
                    #             )
                    # for part in l0["bodies"]:
                    #     part.set_edgecolor(color)
                    #     part.set_facecolor(color)
                    # # l0["cmedians"].set_edgecolor(color)
                    # # color = l0["bodies"][0].get_facecolor().flatten()

                    # if l_num not in labs_taken:
                    #     vlabs.append((mpl.patches.Patch(color=color), label))
                    #     labs_taken.append(l_num)
                    
                    if len(data) > 1:
                        x = beeswarm(data)
                    else:
                        x = 0
                    ax1.scatter(pos + x*width/2, data, c=color, s=1, alpha=1, clip_on=False, zorder=10)

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
                    ax1.scatter(pos + x*width/2, data, c=color, s=1, alpha=1, clip_on=False, zorder=10)
                    # print(list(x))
                    # ax1.scatter(pos + x*width/2, data, c=cmap_points(norm(other)), s=1, alpha=1, clip_on=False, zorder=10)

                    # ax1.scatter(pos + x[]*width/2, data, exs_vals[:-1], facecolors='none', edgecolors='k', s=5, alpha=1, clip_on=False, zorder=10)

    # # examples

    # if plot_type == 'fitness':
    #     exs_PT = np.array([
    #         [256,-.5],
    #         [394,0],
    #         [184,.5],
    #     ])
    # elif plot_type == 'basin_patch_dist':
    #     exs_PT = np.array([
    #         [161,0.91666667],
    #         [40,0],
    #         [28,0.06944444],
    #     ])

    # if group_type == 'vis':
    #     exs_GT = np.array([
    #         1+1/num_labels,
    #         1+0/num_labels,
    #     ])
    #     ax1.scatter(exs_GT + exs_PT[:-1,1]*width/2, exs_PT[:-1,0], facecolors='none', edgecolors='k', s=5, alpha=1, clip_on=False, zorder=10)
    # elif group_type == 'dist':
    #     exs_GT = np.array([
    #         0+1/num_labels,
    #         0+0/num_labels,
    #         8+2/num_labels,
    #     ])
    #     ax1.scatter(exs_GT + exs_PT[:,1]*width/2, exs_PT[:,0], facecolors='none', edgecolors='k', s=5, alpha=1, clip_on=False, zorder=10)

    ax1.axvline(x = g_num - width/2 + 1, color='k', linestyle='--', linewidth=0.5)

    # labs = [l.get_label() for l in lns]
    # ax1.legend(lns, labs, loc='upper left')
    if group_type == 'vis': 
        vlabs.append(vlabs.pop(0))
    elif group_type == 'dist': 
        # vlabs.append(vlabs.pop(1))
        # vlabs.append(vlabs.pop(0))
        vlabs.insert(0,vlabs.pop(1))
    # ax1.legend(*zip(*vlabs))
    plt.legend(*zip(*vlabs), loc='upper left').set_zorder(11)
    # labs = [group_name for group_name,_ in groups]

    # ax1.set_xlabel(group_type)
    ax1.set_xticklabels(group_list_str)
    ax1.set_xlim(-width/2, g_num - width/2 + 1)
    # ax1.set_xlim(-width/2 - width/4, g_num - width/2 + 1 + width/4)

    if group_type == 'vis':
        ax1.set_xlabel('Visual Resolution')
        ax1.set_xticks(np.linspace(0 + width/2, num_groups-1 + width/2, num_groups))
    elif group_type == 'dist':
        ax1.set_xlabel('Distance Scaling')
        ax1.set_xticks(np.linspace(0 + width, num_groups-1 + width, num_groups))

    if plot_type == 'fitness':
        ax1.set_ylabel('Time Taken to Reach Patch')
        ax1.set_ylim(170,580)
        plt.savefig(fr'{data_dir}/group_traj_dists_{group_type}_fitnessbylabel.png', dpi=100)
    elif plot_type == 'basin_patch_dist':
        ax1.set_ylabel('Basin to Patch Distance')
        ax1.set_ylim(-25,850)
        plt.savefig(fr'{data_dir}/group_traj_dists_{group_type}_basinpatch_dist.png', dpi=100)

    # plt.close()
    plt.show()



def bpd_by_fit(plot_type='scatter'):

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    with open(fr'{data_dir}/traj_matrices/gamut_visall_nodist_labeled.bin', 'rb') as f:
        data1 = pickle.load(f)
    with open(fr'{data_dir}/traj_matrices/gamut_vis8_dist_labeled.bin', 'rb') as f:
        data2 = pickle.load(f)
    data = data1 | data2

    BD_bpds_fits = []
    BDIS_bpds_fits = []
    IS_bpds_fits = []
    ISDP_bpds_fits = []
    DP_bpds_fits = []
    DPBD_bpds_fits = []

    for name in data.keys():

        if len(data[name][0]) != 18:
            print(f'skipping {name}')
            continue
        bpd = list(data[name][0])[17]

        with open(fr'{data_dir}/{name}/val_matrix_cen.bin','rb') as f:
            val_matrix = pickle.load(f)
        fit = np.mean(val_matrix)

        data_tuple, label = data[name]
        if label == 'BD':
            if bpd > 300:
                print(name,bpd,fit)
            BD_bpds_fits.append([bpd,fit])
        elif label == 'BD/IS':
            BDIS_bpds_fits.append([bpd,fit])
        elif label == 'IS':
            IS_bpds_fits.append([bpd,fit])
        elif label == 'IS/DP':
            ISDP_bpds_fits.append([bpd,fit])
        elif label == 'DP':
            DP_bpds_fits.append([bpd,fit])
        elif label == 'DP/BD':
            DPBD_bpds_fits.append([bpd,fit])

    BD_bpds_fits = np.array(BD_bpds_fits)
    BDIS_bpds_fits = np.array(BDIS_bpds_fits)
    IS_bpds_fits = np.array(IS_bpds_fits)
    ISDP_bpds_fits = np.array(ISDP_bpds_fits)
    DP_bpds_fits = np.array(DP_bpds_fits)
    DPBD_bpds_fits = np.array(DPBD_bpds_fits)

    if plot_type.startswith('scatter'):
        fig, ax1 = plt.subplots(figsize=(7,4)) 

        size = 10
        alpha = 0.3
        ax1.scatter(IS_bpds_fits[:,0], IS_bpds_fits[:,1], c='tomato', s=size, alpha=alpha, label='Indirect Sequential')
        # ax1.scatter(BDIS_bpds_fits[:,0], BDIS_bpds_fits[:,1], c='cornflowerblue', s=size, alpha=alpha, marker=MarkerStyle('o', fillstyle='left'))
        # ax1.scatter(BDIS_bpds_fits[:,0], BDIS_bpds_fits[:,1], c='tomato', s=size, alpha=alpha, marker=MarkerStyle('o', fillstyle='right'))
        ax1.scatter(BD_bpds_fits[:,0], BD_bpds_fits[:,1], c='cornflowerblue', s=size, alpha=alpha, label='Biased Diffusive')
        ax1.scatter(DP_bpds_fits[:,0], DP_bpds_fits[:,1], c='forestgreen', s=size, alpha=alpha, label='Direct Pathing')

        if plot_type == 'scatter+line':
            sns.regplot(x=IS_bpds_fits[:,0], y=IS_bpds_fits[:,1], color='tomato', ax=ax1, robust=True, line_kws=dict(alpha=.5), scatter_kws=dict(s=0))
            # sns.regplot(x=BDIS_bpds_fits[:,0], y=BDIS_bpds_fits[:,1], color='grey', ax=ax1, scatter_kws=dict(s=0))
            sns.regplot(x=BD_bpds_fits[:,0], y=BD_bpds_fits[:,1], color='cornflowerblue', ax=ax1, robust=True, line_kws=dict(alpha=.5), scatter_kws=dict(s=0))
            sns.regplot(x=DP_bpds_fits[:,0], y=DP_bpds_fits[:,1], color='forestgreen', ax=ax1, robust=True, line_kws=dict(alpha=.5), scatter_kws=dict(s=0))

        ax1.set_xlabel('Basin Patch Distance')
        ax1.set_ylabel('Time Taken to Reach Patch')

        from matplotlib.lines import Line2D
        leg_ele = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='tomato', markersize=7.5, label='Indirect Sequential', alpha=.6),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='cornflowerblue', markersize=7.5, label='Biased Diffusive', alpha=.6),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='forestgreen', markersize=7.5, label='Direct Pathing', alpha=.6),
            ]
        ax1.legend(handles=leg_ele, loc='upper right')

        if plot_type == 'scatter': plt.savefig(fr'{data_dir}/group_traj_dists_basinpatchdist_x_fitness.png', dpi=100)
        elif plot_type == 'scatter+line': plt.savefig(fr'{data_dir}/group_traj_dists_basinpatchdist_x_fitness_w_lineartrends.png', dpi=100)
        plt.close()

    elif plot_type == 'heatmap':

        fig, ax1 = plt.subplots(figsize=(7,4)) 
        x_bins = np.linspace(0,650,25)
        y_bins = np.linspace(150,600,25)
        X,Y = np.meshgrid(x_bins, y_bins)
        H,_,_ = np.histogram2d(BD_bpds_fits[:,0], BD_bpds_fits[:,1], bins=[x_bins, y_bins])
        im = ax1.pcolormesh(X, Y, H.T, cmap='plasma')
        ax1.set_xlabel('Basin Patch Distance')
        ax1.set_ylabel('Time Taken to Reach Patch')
        plt.savefig(fr'{data_dir}/group_traj_dists_basinpatchdist_x_fitness_BDhm.png', dpi=100)

        fig, ax1 = plt.subplots(figsize=(7,4)) 
        x_bins = np.linspace(0,650,25)
        y_bins = np.linspace(150,600,25)
        X,Y = np.meshgrid(x_bins, y_bins)
        H,_,_ = np.histogram2d(IS_bpds_fits[:,0], IS_bpds_fits[:,1], bins=[x_bins, y_bins])
        im = ax1.pcolormesh(X, Y, H.T, cmap='plasma')
        ax1.set_xlabel('Basin Patch Distance')
        ax1.set_ylabel('Time Taken to Reach Patch')
        plt.savefig(fr'{data_dir}/group_traj_dists_basinpatchdist_x_fitness_IShm.png', dpi=100)
    
    elif plot_type == 'violin':

        fig, ax1 = plt.subplots(figsize=(4,3)) 
        # colors = ['tomato', 'grey', 'cornflowerblue']
        # data = [IS_bpds_fits[:,0], BDIS_bpds_fits[:,0], BD_bpds_fits[:,0]]
        # labels = ['IS', 'BD/IS', 'BD']
        colors = ['tomato', 'cornflowerblue', 'forestgreen']
        data = [IS_bpds_fits[:,0], BD_bpds_fits[:,0], DP_bpds_fits[:,0], ]
        labels = ['IS', 'BD', 'DP']
        # colors = ['tomato', 'grey', 'cornflowerblue', 'grey', 'forestgreen', 'grey']
        # data = [IS_bpds_fits[:,0], BDIS_bpds_fits[:,0], BD_bpds_fits[:,0], DPBD_bpds_fits[:,0], DP_bpds_fits[:,0], ISDP_bpds_fits[:,0]]
        # labels = ['IS', 'BD/IS', 'BD', 'DP/BD', 'DP', 'IS/DP']

        for i,(c,d) in enumerate(zip(colors,data)):
            l0 = ax1.violinplot(d, 
                        positions=[i],
                        widths=1, 
                        showmedians=True,
                        showextrema=False,
                        )
            for p in l0['bodies']:
                p.set_facecolor(c)
                p.set_edgecolor(c)
            l0['cmedians'].set_edgecolor(c)

        plt.xticks(np.arange(0, len(labels), 1))
        ax1.xaxis.set_ticklabels(labels)
        # ax1.set_xticks([])
        # ax1.set_xticklabels(labels)
        ax1.set_xlabel('Class Types')
        ax1.set_ylabel('Basin Patch Distance')
        # ax1.set_ylim(-20,1020)

        plt.tight_layout()
        plt.savefig(fr'{data_dir}/group_traj_dists_basinpatchdist_x_fitness_violin.png', dpi=100)
        plt.close()



def gamut_2d(data_type_x, data_type_y, group, cluster=None, heatmap=None, sc_type=None, dpi=100):

    print(f'plotting {data_type_x} vs {data_type_y} for {group}')
    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    # if group == 'main_fig':
    if group == 'gamut_noise' or group == 'gamut_pinball' or group == 'gamut_bound':
        with open(fr'{data_dir}/traj_matrices/gamut_visall_nodist_labeled.bin', 'rb') as f:
            data = pickle.load(f)
        # with open(fr'{data_dir}/traj_matrices/{group}.bin', 'rb') as f:
        #     data = pickle.load(f)
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
            # data_tuple = data[name]
            # label = None

            if 'vis6' in name:
                vis6.append((data_tuple, name, label))
            elif 'vis8' in name and 'WF' not in name and 'CNN12' not in name:
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
        # data_list_str = ['corr_peaks', 'decorr_time', 'histo_avg_init', 'histo_avg_patch', 'histo_peaks_init', 'histo_peaks_patch', 'dirent_init', 'dirent_patch', 'act_mean', 'act_min', 'act_max', 'len_mean', 'len_min', 'len_max', 'de_mean', 'de_min', 'de_max']

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

    elif group == 'gamut_vis8_dist' or group == 'gamut_vis16_dist' or group == 'gamut_vis32_dist':
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

    elif group == 'all' or group == 'main_fig':

        with open(fr'{data_dir}/traj_matrices/archive - ISBDDP/gamut_visall_nodist_labeled.bin', 'rb') as f:
            data1 = pickle.load(f)
        with open(fr'{data_dir}/traj_matrices/archive - ISBDDP/gamut_vis8_dist_labeled.bin', 'rb') as f:
            data2 = pickle.load(f)
        if group == 'all':
            with open(fr'{data_dir}/traj_matrices/gamut_vis16_dist_labeled.bin', 'rb') as f:
                data3 = pickle.load(f)
            with open(fr'{data_dir}/traj_matrices/gamut_vis32_dist_labeled.bin', 'rb') as f:
                data4 = pickle.load(f)
            data = data1 | data2 | data3 | data4
        else:
            data = data1 | data2
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
        if group == 'main_fig':
            for g, gs in zip(vis8_dist_groups, vis8_dist_groups_str):
                group_list.append(g)
                group_list_str.append(gs)
            for g, gs in zip(vis_groups, vis_groups_str):
                group_list.append(g)
                group_list_str.append(gs)
        elif group == 'all':
            for g, gs in zip(vis_groups, vis_groups_str):
                group_list.append(g)
                group_list_str.append(gs)
            for g, gs in zip(vis8_dist_groups, vis8_dist_groups_str):
                group_list.append(g)
                group_list_str.append(gs)
            for g, gs in zip(vis16_dist_groups, vis16_dist_groups_str):
                group_list.append(g)
                group_list_str.append(gs)
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

        norm = mpl.colors.Normalize(vmin=0, vmax=5)

        im = ax1.pcolormesh(X, Y, H.T, cmap='plasma', norm=norm)
        # plt.colorbar(im, label='Number Overlapping Runs')

    else:
        data_x, data_y, fitnesses = [],[],[]
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
                # if data[data_type_x] > 155 and data[data_type_y] > 0.65 and label == 'IS/DP':
                #     print(run_name, data[data_type_x], data[data_type_y])

                # if data[data_type_x] < 115 and data[data_type_x] > 70 and data[data_type_y] < 0.65 and label == 'IS':
                #     print(run_name, data[data_type_x], data[data_type_y])
                # if data[data_type_x] < 140 and label == 'DP':
                #     print(run_name, data[data_type_x], data[data_type_y])
                # if data[data_type_y] < 0.7 and label == 'IS/DP':
                #     print(run_name, data[data_type_x], data[data_type_y])
                # if data[data_type_y] > 0.6 and label == 'DP/BD':
                #     print(run_name, data[data_type_x], data[data_type_y])
                # if data[data_type_y] > 0.57 and label == 'BD':
                #     print(run_name, data[data_type_x], data[data_type_y])
                
                if sc_type == 'label':
                    if label == 'BD':
                        ax1.scatter(data[data_type_x], data[data_type_y], color='cornflowerblue', alpha=.5)
                        BD += 1
                    elif label == 'IS':
                        ax1.scatter(data[data_type_x], data[data_type_y], color='tomato', alpha=.5)
                        IS += 1
                    elif label == 'DP':
                        ax1.scatter(data[data_type_x], data[data_type_y], color='forestgreen', alpha=.5)
                        DP += 1
                    elif label == 'BD/IS':
                        ax1.scatter(data[data_type_x], data[data_type_y], color='cornflowerblue', marker=MarkerStyle('o', fillstyle='left'), alpha=.5)
                        ax1.scatter(data[data_type_x], data[data_type_y], color='tomato', marker=MarkerStyle('o', fillstyle='right'), alpha=.5)
                        BD_IS += 1
                    elif label == 'IS/DP':
                        ax1.scatter(data[data_type_x], data[data_type_y], color='tomato', marker=MarkerStyle('o', fillstyle='left'), alpha=.5)
                        ax1.scatter(data[data_type_x], data[data_type_y], color='forestgreen', marker=MarkerStyle('o', fillstyle='right'), alpha=.5)
                        IS_DP += 1
                    elif label == 'DP/BD':
                        ax1.scatter(data[data_type_x], data[data_type_y], color='forestgreen', marker=MarkerStyle('o', fillstyle='left'), alpha=.5)
                        ax1.scatter(data[data_type_x], data[data_type_y], color='cornflowerblue', marker=MarkerStyle('o', fillstyle='right'), alpha=.5)
                        DP_BD += 1

                elif sc_type == 'fitness':
                    with open(fr'{data_dir}/{run_name}/val_matrix_cen.bin','rb') as f:
                        val_matrix = pickle.load(f)
                    fitnesses.append(np.mean(val_matrix))
                    # print(run_name, label, np.min(fitnesses), np.max(fitnesses))
            
            if sc_type == 'label':
                print(f'{group_name}: BD,BD_IS,IS,IS_DP,DP_BD,DP: {BD,BD_IS,IS,IS_DP,DP_BD,DP}') # plot order for occurrence bar
            elif sc_type == 'group':
                ax1.scatter(data_x, data_y, color=cmap(g_num/num_groups), alpha=.5, label=group_name)
                data_x,data_y = [],[]

        if sc_type == 'fitness':
            norm = mpl.colors.Normalize(vmin=180, vmax=500)
            ax1.scatter(data_x, data_y, c=fitnesses, cmap='plasma', norm=norm, alpha=.5)
            print(f'fitness range: {np.min(fitnesses), np.max(fitnesses)}')
            # for x,y,f in zip(data_x, data_y, fitnesses):
            #     print(x,y,f, np.linalg.norm([x-173,y-.87]))
    
    # main_fig_exs = np.array([
    #     [121,.76],
    #     [31,.39],
    #     [173,.87]
    # ])
    # ax1.scatter(main_fig_exs[:,0], main_fig_exs[:,1], facecolors='none', edgecolors='k')

    if ((data_type_x,data_type_y) == (1,36) and group == 'gamut') or ((data_type_x,data_type_y) == (1,14)):
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


def gamut_table(group):

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    with open(fr'{data_dir}/traj_matrices/{group}.bin', 'rb') as f:
        data = pickle.load(f)
    print(f'data dict len: {len(data)}')

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
        # if len(data[name]) < 18:
        #     print(name, 'incomplete')
        #     continue

        # with open(fr'{data_dir}/{name}/val_matrix_cen.bin','rb') as f:
        #     val_matrix = pickle.load(f)
        # fit = np.mean(val_matrix)

        # print(name, int(data[name][1]), round(data[name][14],2), int(data[name][17]), int(fit))


    # with open(fr'{data_dir}/traj_matrices/{group}_rerun.bin', 'rb') as f:
    #     data_extra = pickle.load(f)

    # new_data = {}
    # for name in data.keys():
    #     if name not in data_extra:
    #         print(name, 'no data_extra')
    #         new_data[name] = data[name]
    #         continue
    #     new_entry = list(data[name])
    #     new_entry.append(data_extra[name])
    #     new_data[name] = tuple(new_entry)
    #     print(name, len(data[name]), data_extra[name], len(new_data[name]))

    # print(f'data dict len: {len(new_data)}')

    # with open(fr'{data_dir}/traj_matrices/{group}.bin', 'wb') as f:
    #     pickle.dump(new_data, f)


def gamut_label(group):

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'

    with open(fr'{data_dir}/traj_matrices/{group}.bin', 'rb') as f:
        data = pickle.load(f)
    print(f'data dict len: {len(data)}')

    if group == 'gamut_visall_nodist' or group == 'gamut':

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
    
    if group == 'gamut_vis8_dist' or group == 'gamut':

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

    if group == 'gamut_vis16_dist':

        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep0'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep1'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep2'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep3'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep4'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep5'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep6'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep7'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep8'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep9'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep10'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep11'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep12'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep13'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep14'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep15'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep16'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep18'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep19'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep0'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep1'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep2'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep3'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep4'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep5'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep6'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep7'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep8'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep9'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep10'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep11'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep12'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep13'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep14'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep15'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep16'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep17'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep18'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep19'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep0'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep1'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep2'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep3'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep4'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep5'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep6'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep7'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep8'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep9'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep10'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep11'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep12'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep13'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep14'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep15'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep16'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep17'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep18'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep19'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep0'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep1'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep2'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep3'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep4'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep5'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep6'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep7'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep8'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep9'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep10'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep11'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep12'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep13'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep14'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep15'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep16'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep17'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep18'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep19'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep0'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep1'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep2'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep3'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep4'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep5'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep6'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep7'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep8'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep9'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep10'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep11'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep12'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep13'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep14'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep15'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep16'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep17'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep19'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep0'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep1'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep2'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep3'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep4'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep5'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep6'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep7'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep8'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep9'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep10'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep11'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep12'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep13'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep14'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep15'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep16'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep17'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep18'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep19'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep0'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep1'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep2'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep3'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep4'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep5'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep6'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep7'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep8'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep9'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep10'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep11'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep12'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep13'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep14'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep15'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep16'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep17'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep18'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep19'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep0'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep1'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep2'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep3'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep4'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep5'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep6'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep7'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep8'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep9'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep10'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep11'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep12'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep13'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep14'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep15'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep16'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep17'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep18'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep19'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep0'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep1'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep2'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep3'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep4'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep5'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep6'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep7'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep8'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep9'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep10'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep11'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep12'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep13'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep14'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep15'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep16'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep17'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep18'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep19'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep0'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep1'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep2'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep3'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep4'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep5'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep6'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep7'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep8'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep9'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep10'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep11'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep12'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep13'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep14'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep15'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep16'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep17'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep18'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep19'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_rep1'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_rep2'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_rep3'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_rep4'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_rep5'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_rep6'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_rep7'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_rep8'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_rep9'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_rep11'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_rep12'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_rep13'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_rep14'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_rep15'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_rep17'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_rep18'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep0'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep1'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep2'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep3'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep4'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep5'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep6'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep7'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep8'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep9'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep10'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep11'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep12'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep13'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep14'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep15'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep16'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep17'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep18'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep19'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_rep3'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_rep4'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_rep5'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_rep6'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_rep7'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_rep8'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_rep9'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_rep11'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_rep12'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_rep13'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_rep15'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_rep17'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_rep18'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_rep19'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep0'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep1'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep2'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep3'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep4'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep5'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep6'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep7'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep8'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep9'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep10'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep11'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep13'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep14'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep15'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep18'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep19'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_rep0'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_rep1'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_rep5'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_rep9'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_rep10'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_rep11'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_rep12'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_rep14'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_rep15'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_rep16'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep0'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep2'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep5'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep9'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep10'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep11'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep12'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep13'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep16'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep17'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep18'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep19'], 'BD')

    if group == 'gamut_vis32_dist':

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
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep0'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep1'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep2'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep3'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep4'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep5'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep6'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep7'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep8'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep9'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep10'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep11'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep12'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep13'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep14'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep15'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep16'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep17'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep18'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep19'], 'DP')
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
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep0'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep1'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep2'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep3'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep4'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep5'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep6'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep7'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep8'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep9'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep10'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep11'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep12'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep13'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep14'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep15'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep16'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep17'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep18'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep19'], 'DP')
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
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep0'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep1'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep2'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep3'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep4'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep5'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep6'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep7'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep8'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep9'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep10'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep11'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep12'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep13'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep14'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep15'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep16'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep17'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep18'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep19'], 'DP')
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
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep0'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep1'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep2'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep3'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep4'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep5'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep6'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep7'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep8'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep9'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep10'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep11'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep12'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep13'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep14'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep15'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep16'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep17'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep18'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep19'], 'DP')
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
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep0'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep1'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep2'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep3'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep4'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep5'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep6'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep7'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep8'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep9'], 'IS/DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep10'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep11'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep12'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep13'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep14'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep15'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep16'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep17'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep18'], 'DP')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep19'], 'DP/BD')
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
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep0'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep1'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep2'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep2'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep3'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep4'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep5'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep6'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep7'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep8'], 'DP/BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep9'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep10'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep11'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep12'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep13'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep14'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep15'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep15'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep16'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep17'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep18'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep19'], 'IS')
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
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep0'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep1'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep6'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep7'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep7'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep8'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep11'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep12'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep12'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep13'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep13'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep14'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep16'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep19'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep19'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_rep0'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_rep0'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_rep1'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_rep1'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_rep6'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_rep8'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_rep9'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_rep11'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_rep11'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_rep16'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_rep17'], 'IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep3'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep3'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep4'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep4'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep5'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep5'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep6'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep6'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep8'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep8'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep9'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep9'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep10'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep10'], 'BD/IS')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep14'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep14'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep16'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep16'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep17'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep17'], 'BD')
        data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep18'] = (data['sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep18'], 'BD')

    if group == 'gamut_other':

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


if __name__ == '__main__':

    ## traj
    space_step = 25
    # space_step = 50 # for log_ray_boundary
    orient_step = np.pi/8
    timesteps = 500
    dpi = 100

    # # med test
    # space_step = 50
    # orient_step = np.pi/4

    # # quick test
    # space_step = 500
    # orient_step = np.pi/2

    # noise_types = [
    #     (0, 'no_noise'), 
    #     (0.05, 'angle_n05'), 
    #     (0.10, 'angle_n10'),
    #     ]

    names = []

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'


    ### ------ final figure update ------- ###

    # for i in [1,3,4,15]:
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{str(i)}')
    # for i in [3,4]:
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep{str(i)}')
    # for i in [10,18]:
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep{str(i)}')
    # names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep12')
    # for i in [7,11]:
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep{str(i)}')
    # names.append('sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep2')

    # names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep3')
    # names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep15')
    # names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep10')

    # names.append('sc_N3_NRW0_ND2_CNN14_FNN16_vis8_rep4') # pure follower - exploit or explore
    # names.append('sc_N3_NRW0_ND2_CNN14_FNN16_vis8_rep6') # follower + nav - BD
    # names.append('sc_N3_NRW0_ND2_CNN14_FNN16_vis8_rep1') # follower + nav - IS
    # names.append('sc_N3_NRW0_ND2_CNN14_FNN16_vis8_rep14') # pure navigator - BD
    # names.append('sc_N3_NRW0_ND2_CNN14_FNN16_vis8_rep2') # pure navigator - IS

    # names.append('sc_N4_NRW1_ND2_CNN14_FNN16_vis8_rep12')

    # names = [
    #     'sc_CNN18_FNN2x64_p50e20_vis32_fov97_rep0', # actually were trained with N=2, N_RAND=1 --> rerun
    #     'sc_CNN18_FNN2x64_p50e20_vis32_fov97_rep1', 
    #     'sc_CNN18_FNN2x64_p50e20_vis32_fov97_rep2',
    # ]


    # for name in names:
    #     # gen, valfit = find_top_val_gen(name, 'cen')
    #     gen, valfit = find_top_val_gen(name, 'cen', archive=True)
    #     # print(f'{name} @ {gen} w {valfit} fitness')

        # orient_step = np.pi/8
        # save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_cen_e1.bin'
        # print(f'save as: {save_name_traj}')
        # traj_exists = False
        # if os.path.exists(save_name_traj):
        #     print('traj already built')
        #     traj_exists = True
        # else:
        #     # build_agent_trajs(name, gen, space_step, orient_step, timesteps, feat_out=True)
        #     build_agent_trajs(name, gen, space_step, orient_step, timesteps, archive=True)

    #     save_name_trajmap = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_cen_e1_ex_lines'
    #     # if os.path.exists(save_name_trajmap+'_100.png'):
    #     #     print('traj already plotted at dpi100')
    #     if os.path.exists(save_name_trajmap+'_50.png'):
    #         print('traj already plotted at dpi50')
    #     # elif os.path.exists(save_name_trajmap+'.png'):
    #     #     print('traj already plotted')
    #     else:
    #         plot_agent_trajs(name, gen, space_step, orient_step, timesteps, ex_lines=True, dpi=50)
    #     # plot_agent_trajs(name, gen, space_step, orient_step, timesteps, ex_lines=True, archive=True, dpi=50)
    #     # plot_agent_trajs(name, gen, space_step, orient_step, timesteps, ex_lines=True, archive=True, dpi=100)
    
        # plot_agent_trajs(name, gen, space_step, orient_step, timesteps, extra='3d', dpi=50)

        # plot_agent_orient_corr(name, gen, space_step, orient_step, timesteps, archive=True, dpi=dpi)


        # plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type='_dirent', archive=True, dpi=dpi)
        # plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type='_dirent', dpi=dpi, extra='nosocial')
        # plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type='_dirent', dpi=dpi, extra='ghost_exploiter')
        # plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type='_dirent', dpi=dpi, extra='ghost_explorer')

        # plot_traj_vecfield_perturb_div(name, gen, space_step, orient_step, timesteps, plot_type='JS', dpi=dpi, base_cond='nosocial', perturb_cond='')
        # plot_traj_vecfield_perturb_div(name, gen, space_step, orient_step, timesteps, plot_type='JS', dpi=dpi, base_cond='nosocial', perturb_cond='ghost_exploiter')
        # plot_traj_vecfield_perturb_div(name, gen, space_step, orient_step, timesteps, plot_type='JS', dpi=dpi, base_cond='nosocial', perturb_cond='ghost_explorer')
        # plot_traj_vecfield_perturb_div(name, gen, space_step, orient_step, timesteps, plot_type='JS', dpi=dpi, base_cond='ghost_exploiter', perturb_cond='ghost_explorer')

        # plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type=f'_count', archive=True, dpi=dpi)
        # plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type=f'_avgact', archive=True, dpi=dpi)
        # plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type=f'_avgori', archive=True, dpi=dpi)

        # # activs = 4
        # # for a in range(activs):
        # # for a in [0,1,2,3,5]:
        # for a in [5]:
        #     # plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type=f'_avgactiv{a}', archive=True, dpi=dpi)
        #     plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type=f'_tuning{a}0', archive=True, dpi=dpi)
        #     # plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type=f'_tuning{a}1', archive=True, dpi=dpi)
        #     # plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type=f'_tuning{a}2', archive=True, dpi=dpi)
        #     # plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type=f'_tuning{a}3', archive=True, dpi=dpi)
        #     # plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type=f'_tuning{a}4', archive=True, dpi=dpi)
        #     # oris = 8
        #     # for o in range(oris):
        #     #     plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type=f'_avgactiv_by_ori{a}{o}', archive=True, dpi=dpi)
        # # plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type=f'_avgorilen', archive=True, dpi=dpi) # quiver only




        # orient_step = np.pi/32
        # save_name_act = fr'{data_dir}/action_maps/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_action.bin'
        # print(f'save as: {save_name_act}')
        # if os.path.exists(save_name_act):
        #     print('action matrix already built')
        #     action_exists = True
        #     with open(save_name_act, 'rb') as f:
        #         act_matrix = pickle.load(f)
        #     min_action, max_action = act_matrix.min(), act_matrix.max()
        # else:
        #     build_action_matrix(name, gen, space_step, orient_step, archive=True)
        # # build_action_matrix(name, gen, space_step, orient_step, archive=True, feat_out=True)

        # plot_action_vecfield(name, gen, space_step, orient_step=np.pi/32, plot_type='_avg', colored='len', archive=True, dpi=dpi)
        # plot_action_vecfield(name, gen, space_step, orient_step=np.pi/32, plot_type='_avg', colored='ori', archive=True, dpi=dpi)

        # # activs = 6
        # # for a in range(activs)
        # for a in [0,1,2,3,5]:
        # for a in [5]:
            # plot_action_vecfield(name, gen, space_step, orient_step=np.pi/32, plot_type=f'_tuning{a}0', archive=True, dpi=dpi) # CNN outputs - goal
            # for i in [1,2,3,4]:
            #     plot_action_vecfield(name, gen, space_step, orient_step=np.pi/32, plot_type=f'_tuning{a}{i}', archive=True, dpi=dpi) # CNN outputs


        # plot_action_vecfield(name, gen, space_step, orient_step=np.pi/32, plot_type='_avgact', archive=True, dpi=dpi)
        # for ind in range(8):
        #     plot_action_vecfield(name, gen, space_step, orient_step=np.pi/32, plot_type=f'_avgact_by_ori{ind}', archive=True, dpi=dpi)
        # a = 5
        # plot_action_vecfield(name, gen, space_step, orient_step=np.pi/32, plot_type=f'_avgact{a}', archive=True, dpi=dpi)
        # for ind in range(8):
        #     plot_action_vecfield(name, gen, space_step, orient_step=np.pi/32, plot_type=f'_avgact_by_ori{a}{ind}', archive=True, dpi=dpi)
        # plot_action_vecfield(name, gen, space_step, orient_step=np.pi/32, plot_type=f'_avg5', archive=True, dpi=dpi) # quiver only


    names = []
    # names.append('sc_N6_NRW5_ND0_CNN14_FNN16_vis8_rep37') # no perf + no spatial
    # names.append('sc_N4_NRW2_ND1_CNN14_FNN16_vis8_rep18') # spatial only
    # names.append('sc_N6_NRW1_ND4_CNN14_FNN16_vis8_rep35') # perf only (BD)
    # names.append('sc_N6_NRW2_ND3_CNN14_FNN16_vis8_rep33') # perf + weak spatial (hybrid)
    # names.append('sc_N5_NRW2_ND2_CNN14_FNN16_vis8_rep31') # perf + strong spatial + discernment (only exploiter) --> not in list
    # names.append('sc_N3_NRW0_ND2_CNN14_FNN16_vis8_rep18') # perf + strong spatial + some discernment (explorer also, but less)

    # # names.append('sc_N6_NRW1_ND4_CNN14_FNN16_vis8_nocoll_rep14') # perf only --> rescinded
    # names.append('sc_N6_NRW1_ND4_CNN14_FNN16_vis8_nocoll_rep30') # perf + low spatial
    # names.append('sc_N6_NRW0_ND5_CNN14_FNN16_vis8_nocoll_rep33') # perf + med spatial
    # names.append('sc_N6_NRW3_ND2_CNN14_FNN16_vis8_nocoll_rep25') # perf + strong spatial

    # # names.append('sc_N6_NRW0_ND5_CNN14_FNN16_vis8_SinitAg100_rep1') # redacted
    # # names.append('sc_N6_NRW0_ND5_CNN14_FNN16_vis8_SinitAg100_rep33') # redacted
    # names.append('sc_N5_NRW2_ND2_CNN14_FNN16_vis8_SinitAg100_rep9') # perf + low spatial
    # names.append('sc_N6_NRW1_ND4_CNN14_FNN16_vis8_SinitAg100_rep4') # perf + med spatial
    # names.append('sc_N6_NRW0_ND5_CNN14_FNN16_vis8_SinitAg100_rep35') # perf + strong spatial

    # names.append('sc_N5_NRW1_ND3_CNN14_FNN16_vis8_collinput_rep34') # perf + low spatial
    # names.append('sc_N5_NRW0_ND4_CNN14_FNN16_vis8_collinput_rep16') # perf + med spatial
    # names.append('sc_N5_NRW0_ND4_CNN14_FNN16_vis8_collinput_rep31') # perf + strong spatial

    # names.append('sc_N6_NRW0_ND5_CNN14_FNN16_vis8_rep0')
    # # names.append('sc_N6_NRW1_ND4_CNN14_FNN16_vis8_rep0')
    # # names.append('sc_N6_NRW2_ND3_CNN14_FNN16_vis8_rep0')
    # # names.append('sc_N6_NRW3_ND2_CNN14_FNN16_vis8_rep0')
    # # names.append('sc_N6_NRW4_ND1_CNN14_FNN16_vis8_rep0')
    # names.append('sc_N6_NRW5_ND0_CNN14_FNN16_vis8_rep0')
    # # names.append('sc_N5_NRW0_ND4_CNN14_FNN16_vis8_rep0')
    # # names.append('sc_N5_NRW1_ND3_CNN14_FNN16_vis8_rep0')
    # # names.append('sc_N5_NRW2_ND2_CNN14_FNN16_vis8_rep0')
    # # names.append('sc_N5_NRW3_ND1_CNN14_FNN16_vis8_rep0')
    # # names.append('sc_N5_NRW4_ND0_CNN14_FNN16_vis8_rep0')
    # # names.append('sc_N4_NRW0_ND3_CNN14_FNN16_vis8_rep0')
    # # names.append('sc_N4_NRW1_ND2_CNN14_FNN16_vis8_rep0')
    # # names.append('sc_N4_NRW2_ND1_CNN14_FNN16_vis8_rep0')
    # # names.append('sc_N4_NRW3_ND0_CNN14_FNN16_vis8_rep0')
    # # names.append('sc_N3_NRW0_ND2_CNN14_FNN16_vis8_rep0')
    # # names.append('sc_N3_NRW1_ND1_CNN14_FNN16_vis8_rep0')
    # # names.append('sc_N3_NRW2_ND0_CNN14_FNN16_vis8_rep0')
    # names.append('sc_N2_NRW0_ND1_CNN14_FNN16_vis8_rep0')
    # names.append('sc_N2_NRW1_ND0_CNN14_FNN16_vis8_rep0')

    # names.append('sc_N6_NRW0_ND5_CNN14_FNN16_vis8_SinitAg100_rep0')
    # # names.append('sc_N6_NRW1_ND4_CNN14_FNN16_vis8_SinitAg100_rep0')
    # # names.append('sc_N6_NRW2_ND3_CNN14_FNN16_vis8_SinitAg100_rep0')
    # # names.append('sc_N6_NRW3_ND2_CNN14_FNN16_vis8_SinitAg100_rep0')
    # # names.append('sc_N6_NRW4_ND1_CNN14_FNN16_vis8_SinitAg100_rep0')
    # names.append('sc_N6_NRW5_ND0_CNN14_FNN16_vis8_SinitAg100_rep0')
    # # names.append('sc_N5_NRW0_ND4_CNN14_FNN16_vis8_SinitAg100_rep0')
    # # names.append('sc_N5_NRW1_ND3_CNN14_FNN16_vis8_SinitAg100_rep0')
    # # names.append('sc_N5_NRW2_ND2_CNN14_FNN16_vis8_SinitAg100_rep0')
    # # names.append('sc_N5_NRW3_ND1_CNN14_FNN16_vis8_SinitAg100_rep0')
    # # names.append('sc_N5_NRW4_ND0_CNN14_FNN16_vis8_SinitAg100_rep0')
    # # names.append('sc_N4_NRW0_ND3_CNN14_FNN16_vis8_SinitAg100_rep0')
    # # names.append('sc_N4_NRW1_ND2_CNN14_FNN16_vis8_SinitAg100_rep0')
    # # names.append('sc_N4_NRW2_ND1_CNN14_FNN16_vis8_SinitAg100_rep0')
    # # names.append('sc_N4_NRW3_ND0_CNN14_FNN16_vis8_SinitAg100_rep0')
    # # names.append('sc_N3_NRW0_ND2_CNN14_FNN16_vis8_SinitAg100_rep0')
    # # names.append('sc_N3_NRW1_ND1_CNN14_FNN16_vis8_SinitAg100_rep0')
    # # names.append('sc_N3_NRW2_ND0_CNN14_FNN16_vis8_SinitAg100_rep0')
    # names.append('sc_N2_NRW0_ND1_CNN14_FNN16_vis8_SinitAg100_rep0')
    # names.append('sc_N2_NRW1_ND0_CNN14_FNN16_vis8_SinitAg100_rep0')

    # idx = int(sys.argv[1])-1
    # name = names[idx]
    # print(f'running name index {idx} on {platform.node()}')
    # gen, valfit = find_top_val_gen(name, 'cen')
    # for angle in [.5,.1,.05,.01]:
    #     build_agent_trajs(name, gen, space_step, orient_step, timesteps, extra=f'spin-{angle}')

    # metrics = ['avg-init100', 'avg-init200']
    # plots = ['heatmap']
    # types = ['explore', 'exploit']

    # comb_tuples = list(itertools.product(metrics, plots, types))
    # comb_strings = [f'{metric}-{soctype}-{plot}' for metric, plot, soctype in comb_tuples]


    # for name in names:
    #     gen, valfit = find_top_val_gen(name, 'cen')

        # save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_cen_e1.bin'
        # if not os.path.exists(save_name_traj):
        #     build_agent_trajs(name, gen, space_step, orient_step, timesteps)
        #     # build_agent_trajs(name, gen, space_step, orient_step, timesteps, feat_out=True)
        # save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_cen_e1_nosocial.bin'
        # if not os.path.exists(save_name_traj):
        #     # build_agent_trajs(name, gen, space_step, orient_step, timesteps, extra='nosocial')
        #     build_agent_trajs(name, gen, space_step, orient_step, timesteps, extra='nosocial', feat_out=True)
        # save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_cen_e1_ghost_exploiter.bin'
        # if not os.path.exists(save_name_traj):
        #     # build_agent_trajs(name, gen, space_step, orient_step, timesteps, extra='ghost_exploiter')
        #     build_agent_trajs(name, gen, space_step, orient_step, timesteps, extra='ghost_exploiter', feat_out=True)
        # save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_cen_e1_ghost_explorer.bin'
        # if not os.path.exists(save_name_traj):
        #     # build_agent_trajs(name, gen, space_step, orient_step, timesteps, extra='ghost_explorer')
        #     build_agent_trajs(name, gen, space_step, orient_step, timesteps, extra='ghost_explorer', feat_out=True)
        # build_agent_trajs(name, gen, space_step, orient_step, timesteps, feat_out=True)
        # build_agent_trajs(name, gen, space_step, orient_step, timesteps, extra='nosocial', feat_out=True)
        # build_agent_trajs(name, gen, space_step, orient_step, timesteps, extra='ghost_exploiter', feat_out=True)
        # build_agent_trajs(name, gen, space_step, orient_step, timesteps, extra='ghost_explorer', feat_out=True)

    #     for test in ['','nosocial','ghost_exploiter','ghost_explorer']:
    #         plot_agent_trajs(name, gen, space_step, orient_step, timesteps, ex_lines=True, dpi=50, extra=test)
    #         for a in range(16):
    #             plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type=f'_avgactiv{a+4}', dpi=50, extra=test)

    #     JS_mean_OGNS,_,_ = plot_traj_vecfield_perturb_div(name, gen, space_step, orient_step, timesteps, plot_type='JS', dpi=50, base_cond='nosocial', perturb_cond='')
    #     JS_mean_NSET,_,_ = plot_traj_vecfield_perturb_div(name, gen, space_step, orient_step, timesteps, plot_type='JS', dpi=50, base_cond='nosocial', perturb_cond='ghost_exploiter', mask_cond='')
    #     JS_mean_NSER,_,_ = plot_traj_vecfield_perturb_div(name, gen, space_step, orient_step, timesteps, plot_type='JS', dpi=50, base_cond='nosocial', perturb_cond='ghost_explorer')
    #     JS_mean_ETER,_,_ = plot_traj_vecfield_perturb_div(name, gen, space_step, orient_step, timesteps, plot_type='JS', dpi=50, base_cond='ghost_exploiter', perturb_cond='ghost_explorer')

    #     # for plot in ['_interfaces']: # _avgorilen
    #     # # for plot in ['_finalheat']: 
    #     # # for plot in ['_interfaces','_finalheat']:
    #     for plot in ['_dirent','_interfaces','_finalheat']:
    #         for test in ['','nosocial','ghost_exploiter','ghost_explorer']:
    #             plot_traj_vecfield(name, gen, space_step, orient_step, timesteps, plot_type=plot, dpi=50, extra=test)

        # save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_cen_e1.bin'
        # if os.path.exists(save_name_traj):
        #     os.remove(save_name_traj)
        # save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_cen_e1_nosocial.bin'
        # if os.path.exists(save_name_traj):
        #     os.remove(save_name_traj)
        # save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_cen_e1_ghost_exploiter.bin'
        # if os.path.exists(save_name_traj):
        #     os.remove(save_name_traj)
        # save_name_traj = fr'{data_dir}/traj_matrices/{name}_{gen}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_cen_e1_ghost_explorer.bin'
        # if os.path.exists(save_name_traj):
        #     os.remove(save_name_traj)

    # name = 'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_rep0'
    # # angles = [.0125, .025, .05, .1, .2, .3, .4]
    # # angles = [.15, .25]
    # # angles = np.linspace(1, 90, 90)/90
    # # angles = angles.round(3)
    # for angle in angles:
    #     gen, valfit = find_top_val_gen(name, 'cen')
    #     build_agent_spins(name, gen, angle)
    # plot_spin_chart()


    ### ------ views/phys resolution scaling ------- ###

    # vfr = 8
    # num_views = []
    # vis_res = [6,10,12,14,18,20,24,32]
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
    # vfr = 8
    # orient_step = np.pi/256
    # # # # to = 0
    # # # for to in [0, np.pi/2, np.pi, 3*np.pi/2, np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]:
    # # #     # build_IDM(name, gen, space_step, orient_step, template_orient=to, vis_field_res=vfr)
    # # #     # plot_IDM(name, gen, space_step, orient_step, template_orient=to, vis_field_res=vfr, plot_type='_rot', dpi=100)
    # # #     # plot_IDM(name, gen, space_step, orient_step, template_orient=to, vis_field_res=vfr, plot_type='_trans', dpi=200)
    # # #     # plot_IDM(name, gen, space_step, orient_step, template_orient=to, vis_field_res=vfr, plot_type='_transrot_count', dpi=200)
    # # #     # plot_IDM(name, gen, space_step, orient_step, template_orient=to, vis_field_res=vfr, plot_type='_transrot_ori', dpi=200)
    # # #     plot_IDM_ori(space_step, orient_step, template_orient=to, vis_field_res=vfr, plot_type='_transrot_ori_perf', dpi=200)

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
    #     # plot_IDM_view(space_step, orient_step, view_onehot=v, vis_field_res=vfr, plot_type='_transrot_ori_perf', dpi=50)
    #     plot_IDM_view(space_step, orient_step, view_onehot=v, vis_field_res=vfr, plot_type='_transrot_ori_perf_dist', dpi=50)

    # # plot_IDM_avgperfviews(space_step=5, orient_step=np.pi/256, vis_field_res=8, plot_type='_count', dpi=100)
    # # plot_IDM_avgperfviews(space_step=5, orient_step=np.pi/256, vis_field_res=8, plot_type='_ori', dpi=100)
    # # plot_IDM_avgperfviews(space_step=5, orient_step=np.pi/256, vis_field_res=8, plot_type='_heatmap_count', dpi=100)
    # # for name in names:
    # #     gen, valfit = find_top_val_gen(name, 'cen')
    # #     plot_IDM_avgperfviews(space_step=5, orient_step=np.pi/256, vis_field_res=8, plot_type=(name,gen), dpi=100)
    
    # for vfr in [8]:
    # for vfr in [16]:
    #     build_IDM_unique_views(name, gen, space_step, orient_step, vis_field_res=vfr)
        # plot_IDM_unique_views(space_step, orient_step, vis_field_res=vfr, dpi=100)



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
    #     build_agent_trajs_PRW(space_step, orient_step, timesteps, behavior, rot_diff)
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
    #             build_agent_trajs_PRW(space_step, orient_step, timesteps, behavior, rot_diff, curve)
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
    #                 build_agent_trajs_PRW(space_step, orient_step, timesteps, behavior, rot_diff, curve, limit)
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
    #             build_agent_trajs_PRW(space_step, orient_step, timesteps, behavior, rot_diff, bias=b)
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


    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_rep{x}' for x in range(20)]:
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

    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep{x}' for x in range(20)]:
    #     names.append(name)

    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep{x}' for x in range(20)]:
    #     names.append(name)

    # names = []
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_proprio_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_proprio_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_proprio_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_GRU16_p50e20_vis8_PGPE_ss20_mom8_proprio_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_GRUpara16_p50e20_vis8_PGPE_ss20_mom8_proprio_rep{x}' for x in range(20)]:
    #     names.append(name)
    # names.append('sc_CNN14_GRUpara64_p50e20_vis8_PGPE_ss20_mom8_rep0')
    # names.append('sc_CNN14_GRUpara64_p50e20_vis8_PGPE_ss20_mom8_proprio_rep0')
    # names.append('sc_CNN14_GRUparanoise64_p50e20_vis8_PGPE_ss20_mom8_rep0')
    # names.append('sc_CNN14_GRUparanoise64_p50e20_vis8_PGPE_ss20_mom8_proprio_rep0')
    # run_gamut('gamut_proprio', names, dpi=50)

    # n = 10
    # for name in [f'sc_CNN14_FNN64_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN64_p50e20_vis16_PGPE_ss20_mom8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN64_p50e20_vis32_PGPE_ss20_mom8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2x64_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2x64_p50e20_vis16_PGPE_ss20_mom8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2x64_p50e20_vis32_PGPE_ss20_mom8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_GRU64_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_GRU64_p50e20_vis16_PGPE_ss20_mom8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_GRU64_p50e20_vis32_PGPE_ss20_mom8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_GRUpara64_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_GRUpara64_p50e20_vis16_PGPE_ss20_mom8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_GRUpara64_p50e20_vis32_PGPE_ss20_mom8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # run_gamut('gamut_FNN64', names, dpi=50)


    n = 40
    names = []

    # for name in [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_rep{x+40}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW1_ND4_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW2_ND3_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW3_ND2_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW4_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW5_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N5_NRW0_ND4_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N5_NRW1_ND3_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N5_NRW2_ND2_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N5_NRW3_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N5_NRW4_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N4_NRW0_ND3_CNN14_FNN16_vis8_rep{x+40}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N4_NRW1_ND2_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N4_NRW2_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N4_NRW3_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N3_NRW0_ND2_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N3_NRW1_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N3_NRW2_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N2_NRW1_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)

    # for name in [f'sc_N1_NRW0_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)


    # for name in [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_SinitAg200_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_SinitAg300_rep{x}' for x in range(n)]:
    #     names.append(name)

    # for name in [f'sc_N1_NRW0_ND0_CNN14_FNN16_vis8_ghost_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_SinitRes100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_SinitRes200_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_SinitRes300_rep{x}' for x in range(n)]:
    #     names.append(name)

    # for name in [f'sc_N3_NRW0_ND2_CNN14_FNN16_vis8_SinitRes100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N3_NRW0_ND2_CNN14_FNN16_vis8_SinitRes200_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N3_NRW0_ND2_CNN14_FNN16_vis8_SinitRes300_rep{x}' for x in range(n)]:
    #     names.append(name)

    # for name in [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_SinitAg200_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_SinitAg300_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_SinitRes100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_SinitRes200_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_SinitRes300_rep{x}' for x in range(n)]:
    #     names.append(name)

    # for name in [f'sc_N11_NRW10_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N16_NRW15_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N21_NRW20_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)

    for name in [f'sc_N11_NRW0_ND10_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
        names.append(name)
    for name in [f'sc_N16_NRW0_ND15_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
        names.append(name)
    for name in [f'sc_N21_NRW0_ND20_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
        names.append(name)

    # for name in [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW1_ND4_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW2_ND3_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW3_ND2_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW4_ND1_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW5_ND0_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N5_NRW0_ND4_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N5_NRW1_ND3_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N5_NRW2_ND2_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N5_NRW3_ND1_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N5_NRW4_ND0_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N4_NRW0_ND3_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N4_NRW1_ND2_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N4_NRW2_ND1_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N4_NRW3_ND0_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N3_NRW0_ND2_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N3_NRW1_ND1_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N3_NRW2_ND0_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N2_NRW1_ND0_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)

    # for name in [f'sc_N11_NRW10_ND0_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N16_NRW15_ND0_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N21_NRW20_ND0_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)

    # for name in [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_nocollpatch_rep{x+20}' for x in range(20)]:
    #     names.append(name)

    # for name in [f'sc_N6_NRW0_ND5_CNN18_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis12_rep{x}' for x in range(n)]:
    #     names.append(name)

    # for name in [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW1_ND4_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW2_ND3_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW3_ND2_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW4_ND1_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW5_ND0_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N5_NRW0_ND4_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N5_NRW1_ND3_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N5_NRW2_ND2_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N5_NRW3_ND1_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N5_NRW4_ND0_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N4_NRW0_ND3_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N4_NRW1_ND2_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N4_NRW2_ND1_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N4_NRW3_ND0_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N3_NRW0_ND2_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N3_NRW1_ND1_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N3_NRW2_ND0_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N2_NRW1_ND0_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)

    # for name in [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_SinitAg100_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)

    # for name in [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW1_ND4_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW2_ND3_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW3_ND2_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW4_ND1_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW5_ND0_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N5_NRW0_ND4_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N5_NRW1_ND3_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N5_NRW2_ND2_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N5_NRW3_ND1_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N5_NRW4_ND0_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N4_NRW0_ND3_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N4_NRW1_ND2_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N4_NRW2_ND1_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N4_NRW3_ND0_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N3_NRW0_ND2_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N3_NRW1_ND1_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N3_NRW2_ND0_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N2_NRW1_ND0_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)

    with open(fr'{data_dir}/traj_matrices/gamut_social.bin', 'rb') as f:
        data = pickle.load(f)

    # for name in names:
    #     print(name, len(data[name]))

    size = 2
    idx = int(sys.argv[1])-1
    names = names[idx*size:(idx+1)*size]
    print(f'running name indices {idx*size}:{(idx+1)*size}')
    print(f'on {platform.node()}')

    run_gamut_social('gamut_social', names, dpi=25)
    # # run_gamut_social('gamut_social_extra', names, dpi=50)
    # run_gamut_social_extra('gamut_social', names, dpi=50)
    # run_gamut_social_redo('gamut_social', names, dpi=25)

    # with open(fr'{data_dir}/traj_matrices/gamut_social.bin', 'rb') as f:
    # # with open(fr'{data_dir}/traj_matrices/gamut_social_extra.bin', 'rb') as f:
    #     data = pickle.load(f)
    # count = 0
    # run = []
    # for k,v in sorted(data.items()):
    #     if k in names:
    #         # print(k)
    #         # for vv in v:
    #         #     print(f'  {round(vv,3)}')
    #         count += 1
    #         run.append(k)
    #     # if len(v) != 11:
    #     #     print(k, len(v))
    # for n in names:
    #     if n not in run:
    #         print(f'missing: {n}')
    # print(f'count: {count} / {len(names)}')
    # print(len(data.items()))


    # with open(fr'{data_dir}/traj_matrices/gamut_social.bin', 'rb') as f:
    #     data1 = pickle.load(f)
    # with open(fr'{data_dir}/traj_matrices/gamut_social_extra.bin', 'rb') as f:
    #     data2 = pickle.load(f)

    # for k,v in sorted(data1.items()):
    #     # print(k)
    #     if k in data2:
    #         v1 = data1[k]
    #         v2 = data2[k]
    #         if v1 != v2:
    #             print(k, 'diff')
    #             print(v1)
    #             print(v2)
    #             # # pass
    #         # else:
    #         #     print(f'match in {k}')
    #     else:
    #         print(f'missing')
    #     # print('')

    # for k,v in sorted(data2.items()):
    #     if k in names:
    #         if len(v) == 8:
    #             de_mean_OG, de_mean_NS, de_mean_ET, de_mean_ER, JS_mean_OGNS, JS_mean_NSET, JS_mean_NSER, JS_mean_ETER = data2[k]
    #             data1[k] = de_mean_OG, de_mean_NS, de_mean_ET, de_mean_ER, JS_mean_OGNS, JS_mean_NSET, JS_mean_NSER, JS_mean_ETER
    #         elif len(v) == 11:
    #             de_mean_OG, de_mean_NS, de_mean_ET, de_mean_ER, JS_mean_OGNS, JS_mean_NSET, JS_mean_NSER, JS_mean_ETER, de_mean_NS_patchonly, de_mean_ET_patchonly, JS_mean_NSET_patchonly = data2[k]
    #             data1[k] = de_mean_OG, de_mean_NS, de_mean_ET, de_mean_ER, JS_mean_OGNS, JS_mean_NSET, JS_mean_NSER, JS_mean_ETER
    #         else:
    #             print(k,len(v))

    # with open(fr'{data_dir}/traj_matrices/gamut_social.bin', 'wb') as f:
    #     pickle.dump(data1, f)



    # with open(fr'{data_dir}/traj_matrices/gamut_social.bin', 'rb') as f:
    #     data_dict = pickle.load(f)

    # # metric_type1 = 'dist_shift_NSET'
    # # metric_type2 = 'JS_mean_NSET'
    # metric_type1 = 'dist_shift_OGNS'
    # metric_type2 = 'JS_mean_OGNS'
    # metric_type2_list = [
    #         'de_mean_OG', 'de_mean_NS', 'de_mean_ET', 'de_mean_ER',
    #         'JS_mean_OGNS', 'JS_mean_NSET', 'JS_mean_NSER', 'JS_mean_ETER'
    #         ]
    # index = metric_type2_list.index(metric_type2)

    # for name in names:
    #     gen, valfit = find_top_val_gen(name, 'cen')
    #     # plot_social_orient_corr(name, gen, space_step, orient_step, timesteps, dpi=dpi)
    #     if name in data_dict.keys():
    #         dist1 = name_to_metric(name, metric_type1)
    #         dist2 = data_dict[name][index]
    #         if dist1 > 625 and dist2 > .21:
    #             # of the followers for each group:
    #             gen, valfit = find_top_val_gen(name, 'cen')
    #             # view_set = build_social_views(name, gen, space_step, orient_step, timesteps)
    #             # actions_per_view(name, gen, view_set)
    #             actions_per_view(name, gen)


    # names = []
    # names.append('sc_N6_NRW5_ND0_CNN14_FNN16_vis8_rep37') # no perf + no spatial
    # names.append('sc_N4_NRW2_ND1_CNN14_FNN16_vis8_rep18') # spatial only
    # names.append('sc_N6_NRW1_ND4_CNN14_FNN16_vis8_rep35') # perf only (BD)
    # names.append('sc_N6_NRW2_ND3_CNN14_FNN16_vis8_rep33') # perf + weak spatial (hybrid)
    # names.append('sc_N5_NRW2_ND2_CNN14_FNN16_vis8_rep31') # perf + strong spatial + discernment (only exploiter) --> not in list
    # names.append('sc_N3_NRW0_ND2_CNN14_FNN16_vis8_rep18') # perf + strong spatial + some discernment (explorer also, but less)

    # for name in names:
        
    #     gen, valfit = find_top_val_gen(name, 'cen')
    #     # build_patch_dists(name, gen, space_step, orient_step, timesteps)
    #     # patch_timeXdist(name, gen)
    #     patch_timeXdist(name, gen, plot_type='_bars')


    # analyze_gamut --> input index for desired data type
    # 0-7: corr_peaks, decorr_time, histo_avg_init, histo_avg_patch, histo_peaks_init, histo_peaks_patch, dirent_init, dirent_patch, 
    # 8-13: min_action, max_action, avglen_mean, avglen_med, avglen_min, avglen_max, 
    # 14-21: pkf_mean, pkf_med, pkf_min, pkf_max, pkt_mean, pkt_med, pkt_min, pkt_max, 
    # 22-29: def_mean, def_med, def_min, def_max, det_mean, det_med, det_min, det_max, 
    # 30-38: act_mean, act_min, act_max, len_mean, len_min, len_max, de_mean, de_min, de_max

    # gamut_table('gamut_visall_nodist')
    # gamut_table('gamut_vis8_dist')
    # gamut_table('gamut_vis16_dist')
    # gamut_table('gamut_vis32_dist')
    # gamut_table('gamut_other')
    # gamut_table('gamut_bound')
    # gamut_table('gamut_pinball')

    # gamut_label('gamut')
    # gamut_label('gamut_visall_nodist')
    # gamut_label('gamut_vis8_dist')
    # gamut_label('gamut_visall_nodist')
    # gamut_label('gamut_vis8_dist')
    # gamut_label('gamut_vis16_dist')
    # gamut_label('gamut_vis32_dist')
    # gamut_label('gamut_other')

    # analyze_gamut('vis','fitness')
    # analyze_gamut('dist','fitness')
    # analyze_gamut('vis','basin_patch_dist')
    # analyze_gamut('dist','basin_patch_dist')
    # bpd_by_fit(plot_type='scatter')
    # bpd_by_fit(plot_type='scatter+line')
    # bpd_by_fit(plot_type='violin')

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
    # gamut_2d(1,14, 'gamut_vis16_dist', sc_type='group')
    # gamut_2d(1,14, 'gamut_vis16_dist', sc_type='label')
    # gamut_2d(1,14, 'gamut_vis16_dist', sc_type='fitness')
    # gamut_2d(1,14, 'gamut_vis16_dist', heatmap=True)
    # gamut_2d(1,14, 'gamut_vis16_dist', cluster='kmeans')
    # gamut_2d(1,14, 'gamut_vis16_dist', cluster='gmm')
    # gamut_2d(1,14, 'gamut_vis32_dist', sc_type='group')
    # gamut_2d(1,14, 'gamut_vis32_dist', sc_type='label')
    # gamut_2d(1,14, 'gamut_vis32_dist', sc_type='fitness')
    # gamut_2d(1,14, 'gamut_vis32_dist', heatmap=True)
    # gamut_2d(1,14, 'gamut_vis32_dist', cluster='kmeans')
    # gamut_2d(1,14, 'gamut_vis32_dist', cluster='gmm')
    # gamut_2d(1,14, 'all', sc_type='group')
    # gamut_2d(1,14, 'all', sc_type='label')
    # gamut_2d(1,14, 'all', sc_type='fitness')
    # gamut_2d(1,14, 'all', heatmap=True)
    # # gamut_2d(1,14, 'all', cluster='kmeans')
    # # gamut_2d(1,14, 'all', cluster='gmm')
    # gamut_2d(1,14, 'main_fig', sc_type='group')
    # gamut_2d(1,14, 'main_fig', sc_type='label')
    # gamut_2d(1,14, 'main_fig', sc_type='fitness')
    # gamut_2d(1,14, 'main_fig', heatmap=True)
    # gamut_2d(1,14, 'main_fig', cluster='kmeans')
    # gamut_2d(1,14, 'main_fig', cluster='gmm')

    # gamut_table('gamut_noise')
    # gamut_2d(1,14, 'gamut_noise', sc_type='group')
    # gamut_2d(1,14, 'gamut_noise', sc_type='fitness')

    # gamut_table('gamut_pinball')
    # gamut_2d(1,14, 'gamut_pinball', sc_type='group')
    # gamut_2d(1,14, 'gamut_pinball', sc_type='fitness')