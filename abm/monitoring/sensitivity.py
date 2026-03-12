import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import pickle
import dotenv as de
from abm.monitoring.trajs import find_top_val_gen, agent_action_from_view
from abm.start_sim import reconstruct_NN

def gather_views(data_dir, basis_vfr):

    # gather non-social views
    with open(fr'{data_dir}/IDM/views_vfr{basis_vfr}.bin', 'rb') as f:
        views = pickle.load(f)

    views_list = views.copy() # start with set without viewable agent

    # 1 ag
    for x in range(basis_vfr): # set with 1 explorer x each visual perturb
        for view in views:
            v = view.copy()
            v[x] = 'agent_explore'
            views_list.append(v)
    for x in range(basis_vfr): # set with 1 exploiter x each visual perturb
        for view in views:
            v = view.copy()
            v[x] = 'agent_exploit'
            views_list.append(v)

    # 2 ags
    for x in range(basis_vfr-1): # set with 2 explorer x each visual perturb
        for view in views:
            v = view.copy()
            v[x] = 'agent_explore'
            v[x+1] = 'agent_explore'
            views_list.append(v)
    for x in range(basis_vfr-1): # set with 2 exploiter x each visual perturb
        for view in views:
            v = view.copy()
            v[x] = 'agent_exploit'
            v[x+1] = 'agent_exploit'
            views_list.append(v)
    
    # 3 ags
    for x in range(basis_vfr-2): # set with 3 explorer x each visual perturb
        for view in views:
            v = view.copy()
            v[x] = 'agent_explore'
            v[x+1] = 'agent_explore'
            v[x+2] = 'agent_explore'
            views_list.append(v)
    for x in range(basis_vfr-2): # set with 3 exploiter x each visual perturb
        for view in views:
            v = view.copy()
            v[x] = 'agent_exploit'
            v[x+1] = 'agent_exploit'
            v[x+2] = 'agent_exploit'
            views_list.append(v)
    
    # 4 ags
    for x in range(basis_vfr-3): # set with 4 explorer x each visual perturb
        for view in views:
            v = view.copy()
            v[x] = 'agent_explore'
            v[x+1] = 'agent_explore'
            v[x+2] = 'agent_explore'
            v[x+3] = 'agent_explore'
            views_list.append(v)
    for x in range(basis_vfr-3): # set with 4 exploiter x each visual perturb
        for view in views:
            v = view.copy()
            v[x] = 'agent_exploit'
            v[x+1] = 'agent_exploit'
            v[x+2] = 'agent_exploit'
            v[x+3] = 'agent_exploit'
            views_list.append(v)
    
    # 5 ags
    for x in range(basis_vfr-4): # set with 5 explorer x each visual perturb
        for view in views:
            v = view.copy()
            v[x] = 'agent_explore'
            v[x+1] = 'agent_explore'
            v[x+2] = 'agent_explore'
            v[x+3] = 'agent_explore'
            v[x+4] = 'agent_explore'
            views_list.append(v)
    for x in range(basis_vfr-4): # set with 5 exploiter x each visual perturb
        for view in views:
            v = view.copy()
            v[x] = 'agent_exploit'
            v[x+1] = 'agent_exploit'
            v[x+2] = 'agent_exploit'
            v[x+3] = 'agent_exploit'
            v[x+4] = 'agent_exploit'
            views_list.append(v)
    
    # 6 ags
    for x in range(basis_vfr-5): # set with 6 explorer x each visual perturb
        for view in views:
            v = view.copy()
            v[x] = 'agent_explore'
            v[x+1] = 'agent_explore'
            v[x+2] = 'agent_explore'
            v[x+3] = 'agent_explore'
            v[x+4] = 'agent_explore'
            v[x+5] = 'agent_explore'
            views_list.append(v)
    for x in range(basis_vfr-5): # set with 6 exploiter x each visual perturb
        for view in views:
            v = view.copy()
            v[x] = 'agent_exploit'
            v[x+1] = 'agent_exploit'
            v[x+2] = 'agent_exploit'
            v[x+3] = 'agent_exploit'
            v[x+4] = 'agent_exploit'
            v[x+5] = 'agent_exploit'
            views_list.append(v)
    
    # 7 ags
    for x in range(basis_vfr-6): # set with 7 explorer x each visual perturb
        for view in views:
            v = view.copy()
            v[x] = 'agent_explore'
            v[x+1] = 'agent_explore'
            v[x+2] = 'agent_explore'
            v[x+3] = 'agent_explore'
            v[x+4] = 'agent_explore'
            v[x+5] = 'agent_explore'
            v[x+6] = 'agent_explore'
            views_list.append(v)
    for x in range(basis_vfr-6): # set with 7 exploiter x each visual perturb
        for view in views:
            v = view.copy()
            v[x] = 'agent_exploit'
            v[x+1] = 'agent_exploit'
            v[x+2] = 'agent_exploit'
            v[x+3] = 'agent_exploit'
            v[x+4] = 'agent_exploit'
            v[x+5] = 'agent_exploit'
            v[x+6] = 'agent_exploit'
            views_list.append(v)
    
    # 8 ags
    for view in views:
        v = view.copy()
        v = ['agent_explore','agent_explore','agent_explore','agent_explore','agent_explore','agent_explore','agent_explore','agent_explore']
        views_list.append(v)
    for view in views:
        v = view.copy()
        v = ['agent_exploit','agent_exploit','agent_exploit','agent_exploit','agent_exploit','agent_exploit','agent_exploit','agent_exploit']
        views_list.append(v)

    return views_list


def plot_sensitivity(names, plot_type='abs(action)'):
    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    exp_name = names[0]
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)

    basis_vfr = 8
    views = gather_views(data_dir, basis_vfr)
    num_views = len(views[0])
    num_conditions = int(len(views)/num_views)

    # iterate over runs
    for name in names:
        print(f'plot {name} - {plot_type} sensitivity')
        gen, valfit = find_top_val_gen(name, 'cen')
        with open(fr'{data_dir}/{name}/{gen}_NNcen_pickle.bin','rb') as f:
            pv = pickle.load(f)
        envconf = de.dotenv_values(fr'{data_dir}/{name}/.env')
        NN, arch = reconstruct_NN(envconf, pv)

        acts = [agent_action_from_view(envconf, NN, v) for v in views] # iters over spatial/social views (only works for single vfr)
        # acts = np.abs(acts)

        # reorder into array
        act_array = np.zeros([num_views, num_conditions])
        # print(act_array.shape, np.any(act_array==0))
        # print(act_array[0,:])

        x = 0
        borders = []
        for i in range(num_views):
            if plot_type == 'act-abs':          act_array[i,0] = np.abs(acts[x])
            elif plot_type == 'act-diff-abs':   act_array[i,0] = acts[x]
            x+=1
        last_loop = 1
        borders.append(last_loop)

        for a in range(basis_vfr):
            for j in range(basis_vfr - a):
                for i in range(num_views):
                    if plot_type == 'act-abs':          act_array[i,last_loop+j] = np.abs(acts[x])
                    elif plot_type == 'act-diff-abs':   act_array[i,last_loop+j] = np.abs(acts[x] - act_array[i,0])
                    x+=1
            last_loop = last_loop+j+1
            borders.append(last_loop)
            for j in range(basis_vfr - a):
                for i in range(num_views):
                    if plot_type == 'act-abs':          act_array[i,last_loop+j] = np.abs(acts[x])
                    elif plot_type == 'act-diff-abs':   act_array[i,last_loop+j] = np.abs(acts[x] - act_array[i,0])
                    x+=1
            last_loop = last_loop+j+1
            borders.append(last_loop)

        # print(act_array.shape, np.any(act_array==0))
        # print(act_array[0,:])
        # print(act_array[:,0])


        # heatmap
        fig, axes = plt.subplots(figsize=(15,2*basis_vfr - .12*basis_vfr**2)) # scaling

        act_array = act_array.T # uno reverse
        # print(act_array.shape, np.any(act_array==0))
        # print(act_array[:,0])

        axes.set_xlim(0, num_views)
        axes.set_ylim(0, num_conditions)
        axes.set_yticks(np.arange(0,num_conditions))

        if plot_type == 'act-diff-abs': # take blank out (spatial only)
            act_array = act_array[1:,:]
            borders = borders[1:]
            axes.set_ylim(0, num_conditions-1)
            axes.set_yticks(np.arange(0,num_conditions-1))

        norm = mpl.colors.Normalize(vmin=act_array.min(), vmax=act_array.max())
        im = axes.imshow(act_array, cmap='plasma', extent=(0,num_views, 0,num_conditions), norm=norm, alpha=.6) 
        # plt.colorbar(im, label='abs(action)', 
        #                 # fraction=0.046, pad=0.04,
        #             # ticks=np.arange(0, 2*np.pi+0.01, np.pi/2),
        #             # format=mpl.ticker.FixedFormatter(['$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']),
        #             )
        borders = [num_conditions-i for i in borders] # uno reverse
        axes.hlines(borders, *axes.get_xlim(), color='k', linewidth=1)

        if plot_type == 'act-abs': labs = ['blank']
        elif plot_type == 'act-diff-abs': labs = []
        labs.extend(['', '', '', '1 Ag Explore', '', '', '', '', 
                '', '', '', '1 Ag Exploit', '', '', '', '', ])
        if basis_vfr >= 2:
            labs.extend(['', '', '', '2 Ag Explore', '', '', '',
                         '', '', '', '2 Ag Exploit', '', '', '',])
        if basis_vfr >= 3:
            labs.extend(['', '', '3 Ag Explore', '', '', '',
                         '', '', '3 Ag Exploit', '', '', '',])
        if basis_vfr >= 4:
            labs.extend(['', '', '4 Ag Explore', '', '',
                         '', '', '4 Ag Exploit', '', '',])
        if basis_vfr >= 5:
            labs.extend(['', '5 Ag Explore', '', '',
                         '', '5 Ag Exploit', '', '',])
        if basis_vfr >= 6:
            labs.extend(['', '6 Ag Explore', '',
                         '', '6 Ag Exploit', '',])
        if basis_vfr >= 7:
            labs.extend(['7 Ag Explore', '',
                         '7 Ag Exploit', '',])
        if basis_vfr >= 8:
            labs.extend(['8 Ag Explore',
                         '8 Ag Exploit',])
        axes.set_yticklabels(labs[::-1]) # uno reverse
        axes.set_xticks([])
        axes.set_xticklabels([])
        axes.set_xlabel('132 Spatial Views')

        fig.tight_layout()
        plt.savefig(fr'{data_dir}/sensitivity/{plot_type}_{basis_vfr}_{name}.png', dpi=100)
        plt.close()


# def action_vs_COM():
#     data_dir = Path(__file__).parent.parent / r'data/simulation_data/'

#     # exp_name = names[0]
#     # env_path = fr'{data_dir}/{exp_name}/.env'
#     # envconf = de.dotenv_values(env_path)

#     basis_vfr = 8
#     views = gather_views(data_dir, basis_vfr)
#     num_views = len(views[0])
#     num_conditions = int(len(views)/num_views)

#     # for v in views:
#     #     print(v)


if __name__ == "__main__":

    # n = 20
    # name_list = []
    # for name in [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     name_list.append(name)
    # for name in [f'sc_N6_NRW1_ND4_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     name_list.append(name)
    # for name in [f'sc_N6_NRW2_ND3_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     name_list.append(name)
    # for name in [f'sc_N6_NRW3_ND2_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     name_list.append(name)
    # for name in [f'sc_N6_NRW4_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     name_list.append(name)
    # for name in [f'sc_N6_NRW5_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     name_list.append(name)
    # labels = np.arange(0,len(name_list))

    # plot_sensitivity(name_list, basis_vfr=1)
    # for i in range(1,8+1):
    #     plot_sensitivity(name_list, basis_vfr=i)
    # plot_sensitivity(name_list, basis_vfr=8, plot_type='act-abs')
    # plot_sensitivity(name_list, basis_vfr=8, plot_type='act-diff-abs')

    action_vs_COM()

