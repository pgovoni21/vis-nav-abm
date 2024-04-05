from abm.monitoring.trajs import find_top_val_gen

from pathlib import Path
import numpy as np

from sklearn.preprocessing import scale
from sklearn.decomposition import PCA, FastICA

import matplotlib.pyplot as plt
import _pickle as pickle


def comp_analysis(exp_name, gen_ext, space_step, orient_step, timesteps, 
                      vis_field_res, n_CNN_filters, analysis='pca'):

    print(f'analyzing {exp_name}, {gen_ext}, {space_step}, {int(np.pi/orient_step)}, {timesteps}')
    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    with open(fr'{data_dir}/trajall_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}.bin', 'rb') as f:
        data = pickle.load(f)
    traj_data, res_data = data
    print('Raw data shape: (agents, time points, neurons): ', traj_data.shape)

    # slice relevant data out for each plot
    plots = [
        ('input', 3, 3+vis_field_res),
        ('CNN', 3+vis_field_res, 3+vis_field_res + n_CNN_filters),
        # ('output', 3+vis_field_res, None),
        ]

    for name, start, end in plots:
        print(f'Plot name: {name}')
        sliced_data = traj_data[:,:,start:end]

        # reshape/flatten trajectories
        n_ag, n_tp, n_feat = sliced_data.shape
        sliced_data = sliced_data.reshape((n_ag*n_tp, n_feat))
        print('Neural activity shape: (time points, features): ', sliced_data.shape)

        # scale + fit analysis according to number of input dimensions
        if analysis == 'pca':
            est = PCA().fit(scale(sliced_data))
            expl_var = est.explained_variance_ratio_
            projected_data = est.transform(scale(sliced_data))

        elif analysis == 'ica':
            est = FastICA(whiten="arbitrary-variance").fit(scale(sliced_data))
            projected_data = est.transform(scale(sliced_data))

        reshap_proj_data = projected_data.reshape((n_ag, n_tp, n_feat))
        print('Projected data shape: (agents, time points, features): ', reshap_proj_data.shape)
        new_traj_data = np.concatenate( (traj_data[:,:,:3], reshap_proj_data), axis=2 )
        print('Traj data shape: (agents, time points, features): ', new_traj_data.shape)
        new_data = new_traj_data, res_data
        # print('Explained variance by component: ', pca.explained_variance_)

        # save projected array
        with open(fr'{data_dir}/trajall_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{analysis}_{name}_projdata.bin', 'wb') as f:
            pickle.dump(new_data, f)

        if analysis == 'pca':
            # explained variance as cumulative line + 95% cutoff + individual contribution bar
            plt.plot(np.cumsum(expl_var))
            plt.hlines(.95,0,n_feat, linestyles='dashed')

            plt.bar(range(n_feat), expl_var, alpha=.5)
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

    names.append(('sc_CNN12_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep18', 8, 2, 'cen'))

    names.append(('sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep1', 8, 3, 'cen'))
    names.append(('sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep9', 8, 3, 'cen'))

    names.append(('sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep1', 8, 4, 'cen'))
    names.append(('sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep3', 8, 4, 'cen'))
    names.append(('sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep10', 8, 4, 'cen'))
    names.append(('sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep11', 8, 4, 'cen'))
    names.append(('sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep14', 8, 4, 'cen'))

    names.append(('sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep9', 8, 4, 'cen'))
    names.append(('sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_rep3', 8, 4, 'top'))
    names.append(('sc_CNN1124_FNN2_p100e20_vis8_rep1', 8, 4, 'top'))

    names.append(('sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov6_rep0', 8, 4, 'cen'))
    names.append(('sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov6_rep5', 8, 4, 'cen'))
    names.append(('sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov6_rep8', 8, 4, 'cen'))


    for name, vis_field_res, n_CNN_filters, rank in names:
        gen, valfit = find_top_val_gen(name, rank)
        print(f'build/plot matrix for: {name} @ {gen} w {valfit} fitness')

        # comp_analysis(name, gen, space_step, orient_step, timesteps, vis_field_res, n_CNN_filters, 'pca')
        comp_analysis(name, gen, space_step, orient_step, timesteps, vis_field_res, n_CNN_filters, 'ica')