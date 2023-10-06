import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import pickle

def plot_map(plot_data, x_max, y_max, cbt, w=4, h=4, save_name=None):

    ag_data, res_data = plot_data

    fig, axes = plt.subplots() 
    axes.set_xlim(0, x_max)
    axes.set_ylim(0, y_max)

    # rescale plotting area to square
    l,r,t,b = fig.subplotpars.left, fig.subplotpars.right, fig.subplotpars.top, fig.subplotpars.bottom
    fig.set_size_inches( float(w)/(r-l) , float(h)/(t-b) )

    # collision boundaries via rectangles (drawn as 2*thickness to coincide with agent center pos when collided)
    walls = [
        ((0, y_max - cbt*2), x_max, cbt*2),
        ((0, 0), x_max, cbt*2),
        ((x_max - cbt*2, 0), cbt*2, y_max),
        ((0, 0), cbt*2, y_max),
    ]
    for (x,y),w,h in walls:
        axes.add_patch( plt.Rectangle((x,y), w, h, color='lightgray', zorder=0) )

    # resource patches via circles
    N_res = res_data.shape[0]
    for res in range(N_res):
        
        # unpack data array
        pos_x = res_data[res,0,0]
        pos_y = res_data[res,0,1]
        radius = res_data[res,0,2]

        axes.add_patch( plt.Circle((pos_x,pos_y), radius, color='lightgray', zorder=0) )
    
    # agent trajectories as arrows/points/events
    N_ag = ag_data.shape[0]
    for agent in range(N_ag):

        # unpack data array
        pos_x = ag_data[agent,:,0]
        pos_y = ag_data[agent,:,1]
        mode_nums = ag_data[agent,:,2]

        # agent start/end points
        axes.plot(pos_x[0], pos_y[0], 'wo', ms=10, markeredgecolor='k', zorder=4, clip_on=False)
        axes.plot(pos_x[-1], pos_y[-1], 'ko', ms=10, zorder=4, clip_on=False)

        # build arrays according to agent mode (~5x faster plot runtime than using compressed save_data)
        traj_explore, traj_exploit, traj_collide = [],[],[]
        for x, y, mode_num in zip(pos_x, pos_y, mode_nums):
            if mode_num == 0: traj_explore.append([x,y])
            elif mode_num == 1: traj_exploit.append([x,y])
            elif mode_num == 2: traj_collide.append([x,y])
        traj_explore, traj_exploit, traj_collide = np.array(traj_explore), np.array(traj_exploit), np.array(traj_collide)
        
        # add agent directional trajectory via arrows 
        arrows(axes, traj_explore[:,0], traj_explore[:,1])

        # add agent positional trajectory + mode via points (every 10 ts for explore, every ts for exploit/collisions)
        axes.plot(traj_explore[::10,0], traj_explore[::10,1],'o', color='royalblue', ms=.5, zorder=2)
        if traj_exploit.size: axes.plot(traj_exploit[:,0], traj_exploit[:,1],'o', color='green', ms=5, zorder=3)
        if traj_collide.size: axes.plot(traj_collide[:,0], traj_collide[:,1],'o', color='red', ms=5, zorder=3, clip_on=False)

    if save_name:
        # line added to sidestep backend memory issues in matplotlib 3.5+
        # if not used, (sometimes) results in tk.call Runtime Error: main thread is not in main loop
        # though this line results in blank first plot, not sure what to do here
        matplotlib.use('Agg')

        # # toggle for indiv runs
        # root_dir = Path(__file__).parent.parent.parent
        # # save_name = Path(root_dir, 'abm/data/simulation_data', f'{save_name[-5:]}')
        # save_name = Path(root_dir, 'abm/data/simulation_data', f'{save_name[-12:]}')

        plt.savefig(fr'{save_name}.png')
        plt.close()
    else:
        plt.show()


def plot_map_iterative_traj(plot_data, x_max, y_max, w=8, h=8, save_name=None):

    from cycler import cycler

    ag_data, res_data = plot_data

    fig, axes = plt.subplots() 
    axes.set_xlim(0, x_max)
    axes.set_ylim(0, y_max)

    # rescale plotting area to square
    l,r,t,b = fig.subplotpars.left, fig.subplotpars.right, fig.subplotpars.top, fig.subplotpars.bottom
    fig.set_size_inches( float(w)/(r-l) , float(h)/(t-b) )

    # agent trajectories as arrows/points/events
    cmap = plt.get_cmap('Blues')
    timesteps = ag_data.shape[1]
    axes.set_prop_cycle(cycler(color=[cmap(1-i/timesteps) for i in range(timesteps)])) # light to dark

    N_ag = ag_data.shape[0]
    for agent in range(N_ag):

        # unpack data array
        pos_x = ag_data[agent,:,0]
        pos_y = ag_data[agent,:,1]

        axes.plot(pos_x, pos_y, linewidth=.1, alpha=0.2, zorder=0)
        # axes.plot(pos_x, pos_y, color='royalblue', linewidth=.1)

    # resource patches via circles
    N_res = res_data.shape[0]
    for res in range(N_res):
        
        # unpack data array
        pos_x = res_data[res,0,0]
        pos_y = res_data[res,0,1]
        radius = res_data[res,0,2]

        axes.add_patch( plt.Circle((pos_x, pos_y), radius, edgecolor='k', fill=False, zorder=1) )

    if save_name:
        plt.savefig(fr'{save_name}.png')
        plt.close()
    else:
        plt.show()


def arrows(axes, x, y, ahl=6, ahw=3):
    # from here: https://stackoverflow.com/questions/8247973/how-do-i-specify-an-arrow-like-linestyle-in-matplotlib

    # r is the distance spanned between pairs of points
    r = [0]
    for i in range(1,len(x)):
        dx = x[i]-x[i-1]
        dy = y[i]-y[i-1]
        r.append(np.sqrt(dx*dx+dy*dy))
    r = np.array(r)

    # set arrow spacing
    num_arrows = int(len(x) / 50)
    aspace = r.sum() / num_arrows
    
    # set inital arrow position at first space
    arrowPos = 0
    
    # rtot is a cumulative sum of r, it's used to save time
    rtot = []
    for i in range(len(r)):
        rtot.append(r[0:i].sum())
    rtot.append(r.sum())

    arrowData = [] # will hold tuples of x,y,theta for each arrow

    ndrawn = 0
    rcount = 1 
    while arrowPos < r.sum() and ndrawn < num_arrows:
        x1, x2 = x[rcount-1], x[rcount]
        y1, y2 = y[rcount-1], y[rcount]
        da = arrowPos - rtot[rcount]
        theta = np.arctan2((x2-x1),(y2-y1))
        ax = np.sin(theta)*da + x1
        ay = np.cos(theta)*da + y1
        arrowData.append((ax,ay,theta))
        ndrawn += 1
        arrowPos += aspace
        while arrowPos > rtot[rcount+1]: 
            rcount += 1
            if arrowPos > rtot[-1]:
                break

    for ax,ay,theta in arrowData[1:]:
        # use aspace as a guide for size and length of things
        # scaling factors were chosen by experimenting a bit

        dx0 = np.sin(theta)*ahl/2. + ax
        dy0 = np.cos(theta)*ahl/2. + ay
        dx1 = -1.*np.sin(theta)*ahl/2. + ax
        dy1 = -1.*np.cos(theta)*ahl/2. + ay

        axes.annotate('', xy=(dx0, dy0), xytext=(dx1, dy1),
                arrowprops=dict( headwidth=ahw, headlength=ahl, ec='royalblue', fc='royalblue', zorder=1))
        

def plot_EA_trend_violin(trend_data, save_dir=False):

    # convert to array
    trend_data = np.array(trend_data)

    # transpose from shape: (number of generations, population size)
    #             to shape: (population size, number of generations)
    data_per_gen_tp = trend_data.transpose()

    # plot population distributions + means of fitnesses for each generation
    plt.violinplot(data_per_gen_tp, widths=1, showmeans=True, showextrema=False)

    if save_dir: 
        plt.savefig(fr'{save_dir}/fitness_spread_violin.png')
        plt.close()
    else: 
        plt.show()


def plot_mult_EA_trends(names, save_name=None):

    # establish load directory
    root_dir = Path(__file__).parent.parent
    data_dir = Path(root_dir, r'data/simulation_data')

    # # init plot details
    fig, ax1 = plt.subplots(figsize=(15,10)) 
    # # ax2 = ax1.twinx()
    cmap = plt.get_cmap('hsv')
    cmap_range = len(names)
    lns = []
    violin_labs = []
    
    # iterate over each file
    for i, name in enumerate(names):

        run_data_exists = False
        if Path(fr'{data_dir}/{name}/run_data.bin').is_file():
            run_data_exists = True
            with open(fr'{data_dir}/{name}/run_data.bin','rb') as f:
                mean_pv, std_pv, time = pickle.load(f)
            print(f'{name}, time taken: {time}')

            # trend_data = std_pv.transpose()
            # # trend_data = mean_pv.transpose()

            # l0 = ax1.violinplot(trend_data, 
            #                widths=1, 
            #                showmeans=True, 
            #                showextrema=True,
            #                )
            # color = l0["bodies"][0].get_facecolor().flatten()
            # violin_labs.append((mpatches.Patch(color=color), name))


        with open(fr'{data_dir}/{name}/fitness_spread_per_generation.bin','rb') as f:
            trend_data = pickle.load(f)
        trend_data = np.array(trend_data)

        # top_trend_data = np.min(trend_data, axis=1) # min : top
        top_trend_data = np.max(trend_data, axis=1) # max : top
        
        # for i,t in enumerate(top_trend_data):
        #     print(f'{i}: {t}')

        # top_5_ind = np.argsort(top_trend_data)[:5] # min : top
        top_5_ind = np.argsort(top_trend_data)[-1:-6:-1] # max : top
        top_5_fit = [top_trend_data[i] for i in top_5_ind]
        for g,f in zip(top_5_ind, top_5_fit):
            print(f'gen {g}: fit {f}')

        l1 = ax1.plot(top_trend_data, 
                        label = f'max {name}',
                        # label = f'avg {name} | t: {time} sec',
                        color=cmap(i/cmap_range), 
                        # linestyle='dotted'
                        )
        lns.append(l1[0])

        avg_trend_data = np.mean(trend_data, axis=1)
        if run_data_exists:
            l2 = ax1.plot(avg_trend_data, 
                            # label = f'avg {name}',
                            label = f'avg {name} | t: {int(time)} sec',
                            color=cmap(i/cmap_range), 
                            linestyle='dotted'
                            )
        else:
            l2 = ax1.plot(avg_trend_data, 
                            label = f'avg {name}',
                            # label = f'avg {name} | t: {int(time)} sec',
                            color=cmap(i/cmap_range), 
                            linestyle='dotted'
                            )
        lns.append(l2[0])
    
    ax1.set_xlabel('Generation')

    labs = [l.get_label() for l in lns]
    # ax1.legend(lns, labs, loc='upper right')
    # ax1.legend(lns, labs, loc='lower left')
    ax1.legend(lns, labs, loc='upper left')

    # ax1.set_ylabel('Time to Find Patch')
    # # ax1.set_ylim(-20,1020)
    # ax1.set_ylim(1900,5000)

    ax1.set_ylabel('# Patches Found')
    ax1.set_ylim(0,8)

    # ax1.legend(*zip(*violin_labs), loc='upper left')
    # ax1.set_ylabel('Parameter')
    # # ax1.set_ylim(-1.25,1.25)

    if save_name: 
        plt.savefig(fr'{data_dir}/{save_name}.png')
    plt.show()



def plot_mult_EA_trends_groups(groups, save_name=None):

    # establish load directory
    root_dir = Path(__file__).parent.parent
    data_dir = Path(root_dir, r'data/simulation_data')

    # # init plot details
    fig, ax1 = plt.subplots(figsize=(15,10)) 
    cmap = plt.get_cmap('hsv')
    cmap_range = len(groups)
    lns = []
    
    # iterate over each file
    for g_num, (group_name, run_names) in enumerate(groups):

        num_runs = len(run_names)
        top_stack = np.zeros((num_runs,1000))
        avg_stack = np.zeros((num_runs,1000))
        time_stack = np.zeros(num_runs)

        for r_num, name in enumerate(run_names):

            with open(fr'{data_dir}/{name}/fitness_spread_per_generation.bin','rb') as f:
                trend_data = pickle.load(f)
            trend_data = np.array(trend_data)

            # top_trend_data = np.min(trend_data, axis=1) # min : top
            top_trend_data = np.max(trend_data, axis=1) # max : top
            top_stack[r_num,:] = top_trend_data

            avg_trend_data = np.mean(trend_data, axis=1)
            avg_stack[r_num,:] = avg_trend_data

            with open(fr'{data_dir}/{name}/run_data.bin','rb') as f:
                mean_pv, std_pv, time = pickle.load(f)
            time_stack[r_num] = time

        l1 = ax1.plot(np.mean(top_stack, axis=0), 
                        label = f'{group_name}: top individual (avg of {num_runs} runs, time: {int(np.mean(time_stack)/60)} min)',
                        color=cmap(g_num/cmap_range), 
                        # linestyle='dotted'
                        )
        lns.append(l1[0])

        l2 = ax1.plot(np.mean(avg_stack, axis=0), 
                        label = f'{group_name}: population average (avg of {num_runs} runs, time: {int(np.mean(time_stack)/60)} min)',
                        color=cmap(g_num/cmap_range), 
                        linestyle='dotted'
                        )
        lns.append(l2[0])
    
    ax1.set_xlabel('Generation')

    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left')

    ax1.set_ylabel('# Patches Found')
    ax1.set_ylim(0,8)

    if save_name: 
        plt.savefig(fr'{data_dir}/{save_name}.png')
    plt.show()



def plot_seeded_trajs(names, num_NNs=5, num_seeds=3):

    from abm.start_sim import start

    # establish load directory
    root_dir = Path(__file__).parent.parent
    data_dir = Path(root_dir, r'data/simulation_data')

    # iterate over each file
    for name in names:

        print(f'agent: {name}')

        env_path = fr'{data_dir}/{name}/.env'
        
        with open(fr'{data_dir}/{name}/fitness_spread_per_generation.bin','rb') as f:
            trend_data = pickle.load(f)
        trend_data = np.array(trend_data)

        # top_trend_data = np.min(trend_data, axis=1) # min : top
        top_trend_data = np.max(trend_data, axis=1) # max : top

        # top_ind = np.argsort(top_trend_data)[:num_NNs] # min : top
        top_ind = np.argsort(top_trend_data)[-1:-num_NNs-1:-1] # max : top
        top_fit = [top_trend_data[i] for i in top_ind]

        for g,f in zip(top_ind, top_fit):
            print(f'gen {g}: fit {f}')

            NN_pv_path = fr'{data_dir}/{name}/gen{g}/NN0_af{int(f)}/NN_pickle.bin'
            with open(NN_pv_path,'rb') as f:
                pv = pickle.load(f)

            for s in range(num_seeds):
                start(pv=pv, env_path=env_path, save_ext=f'top{num_seeds}_{name}_gen{g}_seed{s}', seed=s)



if __name__ == '__main__':


### ----------pop runs----------- ###

    names = []

    # plot_mult_EA_trends([f'doublepoint_CNN18_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(5)], 'doublepoint_vis8_dirfit_GRU_18')
    # plot_mult_EA_trends([f'doublepoint_CNN14_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(4)], 'doublepoint_vis8_dirfit_GRU_14')

    # plot_mult_EA_trends([f'doublepoint_CNN1128_FNN2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(4)], 'doublepoint_vis8_dirfit_FNN')
    # plot_mult_EA_trends([f'doublepoint_CNN1128_CTRNN2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(4)], 'doublepoint_vis8_dirfit_CTRNN')
    # plot_mult_EA_trends([f'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(4)], 'doublepoint_vis8_dirfit_GRU')

    # plot_mult_EA_trends([f'doublepoint_CNN1128_FNN2_p25e5g1000_sig0p1_vis8_dirfit_other0_rep{x}' for x in range(4)], 'doublepoint_vis8_dirfit_FNN_other0')
    # plot_mult_EA_trends([f'doublepoint_CNN1128_CTRNN2_p25e5g1000_sig0p1_vis8_dirfit_other0_rep{x}' for x in range(4)], 'doublepoint_vis8_dirfit_CTRNN_other0')
    # plot_mult_EA_trends([f'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_other0_rep{x}' for x in range(4)], 'doublepoint_vis8_dirfit_GRU_other0')
    # plot_mult_EA_trends([f'doublepoint_CNN1128_GRU1_p25e5g1000_sig0p1_vis8_dirfit_other0_rep{x}' for x in range(4)], 'doublepoint_vis8_dirfit_GRU_other0_1unit')

    # plot_mult_EA_trends([f'doublepoint_CNN1128_FNN2_p25e5g1000_sig0p1_vis8_dirfit_simp_rep{x}' for x in range(6)], 'doublepoint_vis8_dirfit_simp_FNN')
    # plot_mult_EA_trends([f'doublepoint_CNN1128_CTRNN2_p25e5g1000_sig0p1_vis8_dirfit_simp_rep{x}' for x in range(6)], 'doublepoint_vis8_dirfit_simp_CTRNN')
    # plot_mult_EA_trends([f'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_simp_rep{x}' for x in range(6)], 'doublepoint_vis8_dirfit_simp_GRU')

    # plot_mult_EA_trends([f'doublepoint_CNN1126_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(5)], 'doublepoint_vis8_dirfit_GRU_1126')
    # plot_mult_EA_trends([f'doublepoint_CNN1124_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(5)], 'doublepoint_vis8_dirfit_GRU_1124')
    # plot_mult_EA_trends([f'doublepoint_CNN1122_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(5)], 'doublepoint_vis8_dirfit_GRU_1122')
    # plot_mult_EA_trends([f'doublepoint_CNN1118_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(5)], 'doublepoint_vis8_dirfit_GRU_1118')
    # plot_mult_EA_trends([f'doublepoint_CNN1116_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(5)], 'doublepoint_vis8_dirfit_GRU_1116')
    # plot_mult_EA_trends([f'doublepoint_CNN1114_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(5)], 'doublepoint_vis8_dirfit_GRU_1114')
    # plot_mult_EA_trends([f'doublepoint_CNN1112_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(5)], 'doublepoint_vis8_dirfit_GRU_1112')

    # plot_mult_EA_trends([f'doublepoint_CNN111124_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(4)], 'doublepoint_vis8_dirfit_GRU_111124')
    # plot_mult_EA_trends([f'doublepoint_CNN111248_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(5)], 'doublepoint_vis8_dirfit_GRU_111248')

    # plot_mult_EA_trends([f'doublepoint_CNN1128_GRU1_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(5)], 'doublepoint_vis8_dirfit_GRU1')
    # plot_mult_EA_trends([f'doublepoint_CNN1128_GRU3_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(5)], 'doublepoint_vis8_dirfit_GRU3')



    # plot_mult_EA_trends([f'doublepoint_CNN11210_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(5)], 'doublepoint_vis8_dirfit_GRU_11210')
    # plot_mult_EA_trends([f'doublepoint_CNN11212_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(5)], 'doublepoint_vis8_dirfit_GRU_11212')
    # plot_mult_EA_trends([f'doublepoint_CNN11110_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(5)], 'doublepoint_vis8_dirfit_GRU_11110')
    # plot_mult_EA_trends([f'doublepoint_CNN11112_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(5)], 'doublepoint_vis8_dirfit_GRU_11112')



    # plot_mult_EA_trends([f'symdoublepoint_CNN1128_GRU2_rep{x}' for x in range(100)], 'symdoublepoint_vis8_dirfit_GRU_other3')
    # plot_mult_EA_trends([f'symdoublepoint_CNN1128_FNN2_rep{x}' for x in range(67)], 'symdoublepoint_vis8_dirfit_FNN_other3')


### ----------group pop runs----------- ###

    groups = []

    # groups.append(('GRU_14',[f'doublepoint_CNN14_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(5)]))
    # groups.append(('GRU_18',[f'doublepoint_CNN18_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(5)]))
    # plot_mult_EA_trends_groups(groups, 'doublepoint_vis8_dirfit_GRU_1x_groups')

    # groups.append(('FNN',[f'doublepoint_CNN1128_FNN2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(4)]))
    # groups.append(('CTRNN',[f'doublepoint_CNN1128_CTRNN2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(4)]))
    # groups.append(('GRU',[f'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(4)]))
    # plot_mult_EA_trends_groups(groups, 'doublepoint_vis8_dirfit_groups')

    # groups.append(('FNN',[f'doublepoint_CNN1128_FNN2_p25e5g1000_sig0p1_vis8_dirfit_other0_rep{x}' for x in range(4)]))
    # groups.append(('CTRNN',[f'doublepoint_CNN1128_CTRNN2_p25e5g1000_sig0p1_vis8_dirfit_other0_rep{x}' for x in range(4)]))
    # groups.append(('GRU',[f'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_other0_rep{x}' for x in range(4)]))
    # groups.append(('GRU_1',[f'doublepoint_CNN1128_GRU1_p25e5g1000_sig0p1_vis8_dirfit_other0_rep{x}' for x in range(4)]))
    # plot_mult_EA_trends_groups(groups, 'doublepoint_vis8_dirfit_other0_groups')



    # groups.append(('FNN',[f'doublepoint_CNN1128_FNN2_p25e5g1000_sig0p1_vis8_dirfit_other0_rep{x}' for x in range(20)]))
    # # groups.append(('CTRNN',[f'doublepoint_CNN1128_CTRNN2_p25e5g1000_sig0p1_vis8_dirfit_other0_rep{x}' for x in range(4)]))
    # groups.append(('GRU',[f'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_other0_rep{x}' for x in range(20)]))
    # # groups.append(('GRU_1',[f'doublepoint_CNN1128_GRU1_p25e5g1000_sig0p1_vis8_dirfit_other0_rep{x}' for x in range(4)]))
    # plot_mult_EA_trends_groups(groups, 'doublepoint_vis8_dirfit_other0_groups')

    groups = []
    groups.append(('FNN_other0',[f'symdoublepoint_CNN1128_FNN2_other0_rep{x}' for x in range(20)]))
    groups.append(('GRU_other0',[f'symdoublepoint_CNN1128_GRU2_other0_rep{x}' for x in range(20)]))
    plot_mult_EA_trends_groups(groups, 'symdoublepoint_other0')



    # groups.append(('FNN',[f'doublepoint_CNN1128_FNN2_p25e5g1000_sig0p1_vis8_dirfit_simp_rep{x}' for x in range(6)]))
    # groups.append(('CTRNN',[f'doublepoint_CNN1128_CTRNN2_p25e5g1000_sig0p1_vis8_dirfit_simp_rep{x}' for x in range(6)]))
    # groups.append(('GRU',[f'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_simp_rep{x}' for x in range(6)]))
    # plot_mult_EA_trends_groups(groups, 'doublepoint_vis8_dirfit_simp_groups')

    # groups = []
    # groups.append(('11212',[f'doublepoint_CNN11212_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(10)]))
    # groups.append(('11210',[f'doublepoint_CNN11210_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(10)]))
    # groups.append(('1128',[f'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(100)]))
    # groups.append(('1126',[f'doublepoint_CNN1126_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(5)]))
    # groups.append(('1124',[f'doublepoint_CNN1124_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(5)]))
    # groups.append(('1122',[f'doublepoint_CNN1122_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(5)]))
    # plot_mult_EA_trends_groups(groups, 'doublepoint_vis8_dirfit_GRU_112x_groups')

    # groups = []
    # groups.append(('11112',[f'doublepoint_CNN11112_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(10)]))
    # groups.append(('11110',[f'doublepoint_CNN11110_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(10)]))
    # groups.append(('1118',[f'doublepoint_CNN1118_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(5)]))
    # groups.append(('1116',[f'doublepoint_CNN1116_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(5)]))
    # groups.append(('1114',[f'doublepoint_CNN1114_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(5)]))
    # groups.append(('1112',[f'doublepoint_CNN1112_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(5)]))
    # plot_mult_EA_trends_groups(groups, 'doublepoint_vis8_dirfit_GRU_111x_groups')

    # groups = []
    # groups.append(('111124',[f'doublepoint_CNN111124_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(4)]))
    # groups.append(('111248',[f'doublepoint_CNN111248_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(10)]))
    # plot_mult_EA_trends_groups(groups, 'doublepoint_vis8_dirfit_GRU_111xxx_groups')


    # groups.append(('GRU 1',[f'doublepoint_CNN1128_GRU1_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(5)]))
    # groups.append(('GRU 2',[f'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(4)]))
    # groups.append(('GRU 3',[f'doublepoint_CNN1128_GRU3_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(5)]))
    # groups.append(('GRU 1 / other0',[f'doublepoint_CNN1128_GRU1_p25e5g1000_sig0p1_vis8_dirfit_other0_rep{x}' for x in range(4)]))
    # plot_mult_EA_trends_groups(groups, 'doublepoint_vis8_dirfit_GRUx_groups')


    # groups.append(('p25e5', [f'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x+5}' for x in range(95)]))
    # groups.append(('p50e5', [f'doublepoint_CNN1128_GRU2_p50e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(20)]))
    # groups.append(('p100e5', [f'doublepoint_CNN1128_GRU2_p100e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(20)]))
    # groups.append(('p25e10', [f'doublepoint_CNN1128_GRU2_p25e10g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(20)]))
    # groups.append(('p25e20', [f'doublepoint_CNN1128_GRU2_p25e20g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(20)]))
    # plot_mult_EA_trends_groups(groups, 'doublepoint_CNN1128_GRU2_popepvar')


    # groups.append(('CNN1128_GRU2', [f'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(100)]))
    # groups.append(('CNN1128_GRU2', [f'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(50)]))
    # groups.append(('CNN1128_GRU2', [f'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(25)]))
    # groups.append(('CNN1128_GRU2', [f'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(10)]))
    # groups.append(('CNN1128_GRU2', [f'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(5)]))
    # plot_mult_EA_trends_groups(groups, 'doublepoint_CNN1128_GRU2_repvar')
    # groups.append(('CNN1128_GRU2', [f'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(100)]))
    # groups.append(('CNN1128_GRU2', [f'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x+25}' for x in range(50)]))
    # groups.append(('CNN1128_GRU2', [f'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x+25}' for x in range(25)]))
    # groups.append(('CNN1128_GRU2', [f'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x+25}' for x in range(10)]))
    # groups.append(('CNN1128_GRU2', [f'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x+25}' for x in range(5)]))
    # plot_mult_EA_trends_groups(groups, 'doublepoint_CNN1128_GRU2_repvar_p25')
    # groups.append(('CNN1128_GRU2', [f'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(100)]))
    # groups.append(('CNN1128_GRU2', [f'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x+50}' for x in range(50)]))
    # groups.append(('CNN1128_GRU2', [f'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x+50}' for x in range(25)]))
    # groups.append(('CNN1128_GRU2', [f'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x+50}' for x in range(10)]))
    # groups.append(('CNN1128_GRU2', [f'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x+50}' for x in range(5)]))
    # plot_mult_EA_trends_groups(groups, 'doublepoint_CNN1128_GRU2_repvar_p50')
    # groups.append(('CNN1128_GRU2', [f'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(100)]))
    # groups.append(('CNN1128_GRU2', [f'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(50)]))
    # groups.append(('CNN1128_GRU2', [f'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x+75}' for x in range(25)]))
    # groups.append(('CNN1128_GRU2', [f'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x+75}' for x in range(10)]))
    # groups.append(('CNN1128_GRU2', [f'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x+75}' for x in range(5)]))
    # plot_mult_EA_trends_groups(groups, 'doublepoint_CNN1128_GRU2_repvar_p75')


    # groups.append(('CNN1128_GRU2_sig0p1', [f'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(100)]))
    # groups.append(('CNN1128_GRU2_sig1', [f'doublepoint_CNN1128_GRU2_p25e5g1000_sig10_vis8_dirfit_rep{x}' for x in range(11)]))
    # groups.append(('CNN1128_GRU2_sig10', [f'doublepoint_CNN1128_GRU2_p25e5g1000_sig10_vis8_dirfit_rep{x}' for x in range(20)]))
    # plot_mult_EA_trends_groups(groups, 'doublepoint_CNN1128_GRU2_sigvar')

    # groups = []
    # groups.append(('FNN_other3', [f'symdoublepoint_CNN1128_FNN2_rep{x}' for x in range(100)]))
    # groups.append(('GRU_other3', [f'symdoublepoint_CNN1128_GRU2_rep{x}' for x in range(100)]))
    # plot_mult_EA_trends_groups(groups, 'symdoublepoint')
    
    # groups.append(('GRU_other3', [f'symdoublepoint_CNN1128_GRU2_rep{x}' for x in range(100)]))
    # groups.append(('GRU_other3', [f'symdoublepoint_CNN1128_GRU2_rep{x}' for x in range(50)]))
    # groups.append(('GRU_other3', [f'symdoublepoint_CNN1128_GRU2_rep{x}' for x in range(25)]))
    # groups.append(('GRU_other3', [f'symdoublepoint_CNN1128_GRU2_rep{x}' for x in range(10)]))
    # groups.append(('GRU_other3', [f'symdoublepoint_CNN1128_GRU2_rep{x}' for x in range(5)]))
    # plot_mult_EA_trends_groups(groups, 'symdoublepoint_GRU_repvar')
    # groups.append(('GRU_other3', [f'symdoublepoint_CNN1128_GRU2_rep{x}' for x in range(100)]))
    # groups.append(('GRU_other3', [f'symdoublepoint_CNN1128_GRU2_rep{x+50}' for x in range(50)]))
    # groups.append(('GRU_other3', [f'symdoublepoint_CNN1128_GRU2_rep{x+50}' for x in range(25)]))
    # groups.append(('GRU_other3', [f'symdoublepoint_CNN1128_GRU2_rep{x+50}' for x in range(10)]))
    # groups.append(('GRU_other3', [f'symdoublepoint_CNN1128_GRU2_rep{x+50}' for x in range(5)]))
    # plot_mult_EA_trends_groups(groups, 'symdoublepoint_GRU_repvar_p50')
    # groups.append(('GRU_other3', [f'symdoublepoint_CNN1128_GRU2_rep{x}' for x in range(100)]))
    # groups.append(('GRU_other3', [f'symdoublepoint_CNN1128_GRU2_rep{x+25}' for x in range(50)]))
    # groups.append(('GRU_other3', [f'symdoublepoint_CNN1128_GRU2_rep{x+25}' for x in range(25)]))
    # groups.append(('GRU_other3', [f'symdoublepoint_CNN1128_GRU2_rep{x+25}' for x in range(10)]))
    # groups.append(('GRU_other3', [f'symdoublepoint_CNN1128_GRU2_rep{x+25}' for x in range(5)]))
    # plot_mult_EA_trends_groups(groups, 'symdoublepoint_GRU_repvar_p25')
    # groups.append(('GRU_other3', [f'symdoublepoint_CNN1128_GRU2_rep{x}' for x in range(100)]))
    # groups.append(('GRU_other3', [f'symdoublepoint_CNN1128_GRU2_rep{x}' for x in range(50)]))
    # groups.append(('GRU_other3', [f'symdoublepoint_CNN1128_GRU2_rep{x+75}' for x in range(25)]))
    # groups.append(('GRU_other3', [f'symdoublepoint_CNN1128_GRU2_rep{x+75}' for x in range(10)]))
    # groups.append(('GRU_other3', [f'symdoublepoint_CNN1128_GRU2_rep{x+75}' for x in range(5)]))
    # plot_mult_EA_trends_groups(groups, 'symdoublepoint_GRU_repvar_p75')


    # # groups.append(('FNN_other3', [f'symdoublepoint_CNN1128_FNN2_rep{x}' for x in range(100)]))
    # groups.append(('FNN_other3', [f'symdoublepoint_CNN1128_FNN2_rep{x}' for x in range(50)]))
    # groups.append(('FNN_other3', [f'symdoublepoint_CNN1128_FNN2_rep{x}' for x in range(25)]))
    # groups.append(('FNN_other3', [f'symdoublepoint_CNN1128_FNN2_rep{x}' for x in range(10)]))
    # groups.append(('FNN_other3', [f'symdoublepoint_CNN1128_FNN2_rep{x}' for x in range(5)]))
    # plot_mult_EA_trends_groups(groups, 'symdoublepoint_FNN_repvar')
    # # groups.append(('FNN_other3', [f'symdoublepoint_CNN1128_FNN2_rep{x}' for x in range(100)]))
    # groups.append(('FNN_other3', [f'symdoublepoint_CNN1128_FNN2_rep{x}' for x in range(50)]))
    # groups.append(('FNN_other3', [f'symdoublepoint_CNN1128_FNN2_rep{x+25}' for x in range(25)]))
    # groups.append(('FNN_other3', [f'symdoublepoint_CNN1128_FNN2_rep{x+25}' for x in range(10)]))
    # groups.append(('FNN_other3', [f'symdoublepoint_CNN1128_FNN2_rep{x+25}' for x in range(5)]))
    # plot_mult_EA_trends_groups(groups, 'symdoublepoint_FNN_repvar+p25')
    

### ----------singles----------- ###

    # # names.append(f'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_other0_rep0')
    # names.append(f'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_other0_rep2')
    # # names.append(f'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_other0_rep3')
    # plot_seeded_trajs(names, num_NNs=5, num_seeds=20)


### ----------violins----------- ###
    
    # plot_mult_EA_trends(names, fr'stationarycorner_violin_mean')
    # plot_mult_EA_trends(names, fr'stationarycorner_violin_stdv')

    # name = 'stationarycorner_CNN11_FNN8_p25e5g100_sig0p1'
    # name = 'stationarycorner_CNN12_FNN8_p25e5g100_sig0p1'
    # name = 'stationarycorner_CNN14_FNN8_p25e5g100_sig0p1'
    # name = 'stationarycorner_CNN18_FNN8_p25e5g100_sig0p1'
    # name = 'stationarycorner_CNN24_FNN8_p25e5g100_sig0p1'
    # name = 'stationarycorner_CNN12_FNN1_p25e5g100_sig0p1'
    # name = 'stationarycorner_CNN12_FNN1_p25e5g100_sig0p1_other0'
    # name = 'stationarycorner_CNN12_FNN4_p25e5g100_sig0p1_other0'
    # plot_mult_EA_trends([name], fr'{name}/violin_mean')