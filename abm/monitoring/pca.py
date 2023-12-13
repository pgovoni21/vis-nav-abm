from abm.monitoring.agent_vis_matrices import find_top_val_gen

from pathlib import Path
import numpy as np

from sklearn.preprocessing import scale, ICA
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import _pickle as pickle


def plot_CNN_analysis(exp_name, gen_ext, space_step, orient_step, timesteps, n_CNN_filters, analysis='pca'):

    print(f'plotting {exp_name}, {gen_ext}, {space_step}, {int(np.pi/orient_step)}, {timesteps}')
    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    with open(fr'{data_dir}/trajall_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}.bin', 'rb') as f:
        data = pickle.load(f)
    traj_data, res_data = data
    print('Raw data shape: (agents, time points, neurons): ', traj_data.shape)

    # slice relevant data out
    vis_field_res = 8

    plots = [
        ('input', 3, 3+vis_field_res),
        ('CNN', 3+vis_field_res, 3+vis_field_res + n_CNN_filters),
        ('output', 3+vis_field_res, -1),
        ]
    
    for name, start, end in plots:
        print(f'Plot name: {name}')
        sliced_data = traj_data[:,:,start:end]

        # reshape/flatten trajectories
        n_ag, n_tp, n_feat = sliced_data.shape
        sliced_data = sliced_data.reshape((n_ag*n_tp, n_feat))
        print('Neural activity shape: (time points, neurons): ', sliced_data.shape)

        # scale + fit analysis according to number of input dimensions+
        if analysis == 'pca':
            analyzed_data = PCA().fit(scale(sliced_data))
            expl_var = analyzed_data.explained_variance_ratio_
        elif analysis == 'ica':
            analyzed_data = ICA().fit(scale(sliced_data))
            expl_var = analyzed_data.pca_explained_variance_ratio_
        # print('Explained variance by component: ', pca.explained_variance_)

        # explained variance as cumulative line + 95% cutoff + individual contribution bar
        plt.plot(np.cumsum(analyzed_data.explained_variance_ratio_))
        plt.hlines(.95,0,n_feat, linestyles='dashed')

        plt.bar(range(n_feat), analyzed_data.explained_variance_ratio_, alpha=.5)
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance');

        save_name = fr'{data_dir}/trajall_matrices/{exp_name}_{gen_ext}_traj_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{analysis}_{name}'
        plt.savefig(fr'{save_name}.png')
        plt.close()


    # # Project each trial and visualize activity

    # # Plot all trials in ax1, plot fewer trials in ax2
    # fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(6, 3))

    # for i in range(100):
    #     # Transform and plot each trial
    #     activity_pc = pca.transform(activity_dict[i])  # (Time points, PCs)

    #     trial = trial_infos[i]
    #     color = 'red' if trial['ground_truth'] == 0 else 'blue'
        
    #     _ = ax1.plot(activity_pc[:, 0], activity_pc[:, 1], 'o-', color=color)
    #     if i < 3:
    #         _ = ax2.plot(activity_pc[:, 
    #                                 0], activity_pc[:, 1], 'o-', color=color)
            
    #     # Plot the beginning of a trial with a special symbol
    #     _ = ax1.plot(activity_pc[0, 0], activity_pc[0, 1], '^', color='black')

    # ax1.set_title('{:d} Trials'.format(100))
    # ax2.set_title('{:d} Trials'.format(3))
    # ax1.set_xlabel('PC 1')
    # ax1.set_ylabel('PC 2')



if __name__ == '__main__':

    ## traj / Nact
    space_step = 25
    orient_step = np.pi/8
    timesteps = 500

    names = []

    names.append(('sc_CNN12_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep18', 2, 'cen'))

    # names.append(('sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep1', 3, 'cen'))
    # names.append(('sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep9', 3, 'cen'))

    # names.append(('sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep1', 4, 'cen'))
    # names.append(('sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep3', 4, 'cen'))
    # names.append(('sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep10', 4, 'cen'))
    # names.append(('sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep11', 4, 'cen'))
    # names.append(('sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep14', 4, 'cen'))

    # names.append(('sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep9', 4, 'cen'))
    # names.append(('sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_rep3', 4, 'top'))
    # names.append(('sc_CNN1124_FNN2_p100e20_vis8_rep1', 4, 'top'))

    # names.append(('sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov6_rep0', 4, 'cen'))
    # names.append(('sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov6_rep5', 4, 'cen'))
    # names.append(('sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov6_rep8', 4, 'cen'))


    for name, n_CNN_filters, rank in names:
        gen, valfit = find_top_val_gen(name, rank)
        print(f'build/plot matrix for: {name} @ {gen} w {valfit} fitness')

        plot_CNN_analysis(name, gen, space_step, orient_step, timesteps, n_CNN_filters, 'ica')