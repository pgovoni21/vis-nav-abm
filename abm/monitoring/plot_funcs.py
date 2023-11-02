import matplotlib as mpl
import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
from cycler import cycler
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
        axes.plot(traj_explore[::8,0], traj_explore[::8,1],'o', color='royalblue', ms=.5, zorder=2)
        if traj_exploit.size: axes.plot(traj_exploit[:,0], traj_exploit[:,1],'o', color='green', ms=5, zorder=3)
        if traj_collide.size: axes.plot(traj_collide[:,0], traj_collide[:,1],'o', color='red', ms=5, zorder=3, clip_on=False)

    if save_name:
        # line added to sidestep backend memory issues in matplotlib 3.5+
        # if not used, (sometimes) results in tk.call Runtime Error: main thread is not in main loop
        # though this line results in blank first plot, not sure what to do here
        mpl.use('Agg')

        # # toggle for indiv runs
        # root_dir = Path(__file__).parent.parent.parent
        # # save_name = Path(root_dir, 'abm/data/simulation_data', f'{save_name[-5:]}')
        # save_name = Path(root_dir, 'abm/data/simulation_data', f'{save_name[-12:]}')

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
    num_arrows = int(len(x) / 40)
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


def plot_map_iterative_traj(plot_data, x_max, y_max, w=8, h=8, save_name=None):

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


def plot_map_iterative_trajall(plot_data, x_max, y_max, w=8, h=8, save_name=None, var_pos=-1, inv=False, change=True):

    ag_data, res_data = plot_data
    print(f'traj matrix {ag_data.shape}; var_pos [{var_pos}]; inv [{inv}]')

    fig, axes = plt.subplots() 
    axes.set_xlim(0, x_max)
    axes.set_ylim(0, y_max)

    # rescale plotting area to square
    l,r,t,b = fig.subplotpars.left, fig.subplotpars.right, fig.subplotpars.top, fig.subplotpars.bottom
    fig.set_size_inches( float(w)/(r-l) , float(h)/(t-b) )

    # configure coloring
    if var_pos == -1: # action --> narrow around zero (straight)
        cmap = mpl.cm.bwr
        norm = mpl.colors.Normalize(vmin=-.1, vmax=.1) 
    if var_pos == 3: # sensory input change --> binary
        if change: cmap = mpl.colors.ListedColormap(['w','k']) # black : changing 
        else:      cmap = mpl.colors.ListedColormap(['k','w']) # black : no change
        norm = mpl.colors.Normalize()
    else: # Nact --> cmap limits as broad as outermost value
        cmap = mpl.cm.bwr
        max = np.max(ag_data[:,:,var_pos])
        min = np.min(ag_data[:,:,var_pos])
        lim = round(np.maximum(abs(max), abs(min)),2)
        norm = mpl.colors.Normalize(vmin=-lim, vmax=lim)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes)

    # loop over iterations, plotting variable of interest
    N_ag = ag_data.shape[0]
    # for agent in range(0,N_ag,100):
    for agent in range(N_ag):

        # unpack data array
        pos_x = ag_data[agent,:,0]
        pos_y = ag_data[agent,:,1]

        # flip sign if func calls for inverse
        if inv: var = -ag_data[agent,:,var_pos]
        else: var = ag_data[agent,:,var_pos]

        if var_pos == 3: # sensory input change
            var = ag_data[agent,:,3:11]                 # gather matrix (timesteps, vis_field_res)
            var = abs(np.diff(var, axis=0))             # take abs(diff) along timestep axis
            var = np.any(var>0, axis=1).astype(int)     # 0 if no change, 1 if change
            pos_x, pos_y = pos_x[1:], pos_y[1:]         # cut first pos point
        
        # plot variable, whether sensory input or neural activity
        axes.scatter(pos_x, pos_y, c=var, 
                    s=1, alpha=0.05, cmap=cmap, norm=norm) 

    # add resource patches via circles
    N_res = res_data.shape[0]
    for res in range(N_res):
        x,y,radius = res_data[res,0,:]
        axes.add_patch( plt.Circle((x, y), radius, edgecolor='k', fill=False, zorder=1) )

    if save_name:
        plt.savefig(fr'{save_name}.png')
        plt.close()
    else:
        plt.show()


def plot_EA_trend_violin(data, est_method='mean', save_dir=False):

    if est_method == 'mean':
        data_genxpop = np.mean(data, axis=2)
    else:
        data_genxpop = np.median(data, axis=2)

    # transpose from shape: (number of generations, population size)
    #             to shape: (population size, number of generations)
    data_popxgen = data_genxpop.transpose()

    # plot population distributions + means of fitnesses for each generation
    plt.violinplot(data_popxgen, widths=1, showmeans=True, showextrema=False)

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
                        # label = f'top {name}',
                        label = f'top {name} | t: {time} sec',
                        color=cmap(i/cmap_range), 
                        # alpha=0.2
                        )
        lns.append(l1[0])

        # avg_trend_data = np.mean(trend_data, axis=1)
        # if run_data_exists:
        #     l2 = ax1.plot(avg_trend_data, 
        #                     # label = f'avg {name}',
        #                     label = f'avg {name} | t: {int(time)} sec',
        #                     color=cmap(i/cmap_range), 
        #                     linestyle='dotted'
        #                     )
        # else:
        #     l2 = ax1.plot(avg_trend_data, 
        #                     label = f'avg {name}',
        #                     # label = f'avg {name} | t: {int(time)} sec',
        #                     color=cmap(i/cmap_range), 
        #                     linestyle='dotted'
        #                     )
        # lns.append(l2[0])
    
    ax1.set_xlabel('Generation')

    labs = [l.get_label() for l in lns]
    # ax1.legend(lns, labs, loc='upper right')
    # ax1.legend(lns, labs, loc='lower left')
    ax1.legend(lns, labs, loc='upper left')

    # ax1.set_ylabel('Time to Find Patch')
    # # ax1.set_ylim(-20,1020)
    # ax1.set_ylim(1900,5000)

    ax1.set_ylabel('# Patches Found')
    ax1.set_ylim(0,3)

    # ax1.legend(*zip(*violin_labs), loc='upper left')
    # ax1.set_ylabel('Parameter')
    # # ax1.set_ylim(-1.25,1.25)

    if save_name: 
        plt.savefig(fr'{data_dir}/{save_name}.png')
    plt.show()


def plot_mult_EA_trends_np(names, est_method='mean', save_name=None):

    # establish load directory
    root_dir = Path(__file__).parent.parent
    data_dir = Path(root_dir, r'data/simulation_data')

    # # init plot details
    fig, ax1 = plt.subplots(figsize=(15,10)) 
    # # ax2 = ax1.twinx()
    cmap = plt.get_cmap('hsv')
    cmap_range = len(names)
    lns = []
    
    # iterate over each file
    for i, name in enumerate(names):
        print(name)

        # run_data_exists = False
        # if Path(fr'{data_dir}/{name}/run_data.bin').is_file():
        #     run_data_exists = True
        #     with open(fr'{data_dir}/{name}/run_data.bin','rb') as f:
        #         mean_pv, std_pv, time = pickle.load(f)
        #     print(f'{name}, time taken: {int(time/60)} min')


        with open(fr'{data_dir}/{name}/fitness_spread_per_generation.bin','rb') as f:
            data = pickle.load(f)
        
        
        if est_method == 'mean':
            data_genxpop = np.mean(data, axis=2)
        else:
            data_genxpop = np.median(data, axis=2)

        top_data = np.min(data_genxpop, axis=1) # min : top
        # top_data = np.max(data_genxpop, axis=1) # max : top
        top_ind = np.argsort(top_data)[:3] # min : top
        # top_ind = np.argsort(top_data)[-1:-4:-1] # max : top
        top_fit = [top_data[i] for i in top_ind]
        for g,f in zip(top_ind, top_fit):
            print(f'trn | gen {int(g)}: fit {int(f)}')

        l1 = ax1.plot(top_data, 
                        label = f'top {name}',
                        # label = f'top {name} | t: {int(time/60)} min',
                        color=cmap(i/cmap_range), 
                        alpha=0.5
                        )
        lns.append(l1[0])

        avg_trend_data = np.mean(data_genxpop, axis=1)
        l2 = ax1.plot(avg_trend_data, 
                        label = f'avg {name}',
                        # label = f'avg {name} | t: {int(time)} sec',
                        color=cmap(i/cmap_range), 
                        linestyle='dotted',
                        alpha=0.5
                        )
        lns.append(l2[0])

        # parse val results text file if exists
        if Path(fr'{data_dir}/{name}/val_results.txt').is_file():
            with open(fr'{data_dir}/{name}/val_results.txt') as f:
                lines = f.readlines()

                val_data = np.zeros((len(lines)-1, 3))
                for i, line in enumerate(lines[1:]):
                    data = [item.strip() for item in line.split(' ')]
                    val_data[i,0] = data[1] # generation
                    val_data[i,1] = data[4] # train fitness
                    val_data[i,2] = data[7] # val fitness

                top_ind = np.argsort(val_data[:,2])[:3] # min : top
                # top_ind = np.argsort(top_data, axis=2)[-1:-4:-1] # max : top

                top_gen = [val_data[i,0] for i in top_ind]
                top_valfit = [val_data[i,2] for i in top_ind]
                for g,f in zip(top_gen, top_valfit):
                    print(f'val | gen {int(g)}: fit {int(f)}')

            ax1.vlines(val_data[:,0], val_data[:,1], val_data[:,2],
                    color='black'
                    )
    
    ax1.set_xlabel('Generation')

    labs = [l.get_label() for l in lns]
    # ax1.legend(lns, labs, loc='upper right')
    # ax1.legend(lns, labs, loc='lower left')
    ax1.legend(lns, labs, loc='upper left')

    ax1.set_ylabel('Time to Find Patch')
    ax1.set_ylim(-20,1720)
    # ax1.set_ylim(1900,5000)

    # ax1.set_ylabel('# Patches Found')
    # ax1.set_ylim(0,8)

    if save_name: 
        plt.savefig(fr'{data_dir}/{save_name}.png')
    plt.show()


def plot_mult_EA_violins_np(names, save_name=None):

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
    import matplotlib.patches as mpatches
    
    # iterate over each file
    for i, name in enumerate(names):

        with open(fr'{data_dir}/{name}/run_data.bin','rb') as f:
            mean_pv, std_pv, time = pickle.load(f)
        print(f'{name}, time taken: {int(time/60)} min')

        trend_data = std_pv.transpose()
        # trend_data = mean_pv.transpose()

        print(trend_data, trend_data.shape)
        print('')
        print(std_pv, std_pv.shape)

        l0 = ax1.violinplot(trend_data, 
                    widths=1, 
                    showmeans=True, 
                    showextrema=True,
                    )
        color = l0["bodies"][0].get_facecolor().flatten()
        violin_labs.append((mpatches.Patch(color=color), name))
    
    ax1.set_xlabel('Generation')

    labs = [l.get_label() for l in lns]
    # ax1.legend(lns, labs, loc='upper right')
    # ax1.legend(lns, labs, loc='lower left')
    ax1.legend(lns, labs, loc='upper left')

    ax1.legend(*zip(*violin_labs), loc='upper left')
    ax1.set_ylabel('Parameter')
    # ax1.set_ylim(-1.25,1.25)

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

            # with open(fr'{data_dir}/{name}/run_data.bin','rb') as f:
            #     mean_pv, std_pv, time = pickle.load(f)
            # time_stack[r_num] = time

        l1 = ax1.plot(np.mean(top_stack, axis=0), 
                        # label = f'{group_name}: top individual (avg of {num_runs} runs, time: {int(np.mean(time_stack)/60)} min)',
                        label = f'{group_name}: top individual (avg of {num_runs} runs',
                        color=cmap(g_num/cmap_range), 
                        # linestyle='dotted'
                        )
        lns.append(l1[0])

        l2 = ax1.plot(np.mean(avg_stack, axis=0), 
                        # label = f'{group_name}: population average (avg of {num_runs} runs, time: {int(np.mean(time_stack)/60)} min)',
                        label = f'{group_name}: population average (avg of {num_runs} runs',
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


def plot_mult_EA_trends_groups_np(groups, save_name=None):

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
                data = pickle.load(f)

            data_genxpop = np.mean(data, axis=2)
            top_data = np.min(data_genxpop, axis=1) # min : top
            # top_data = np.max(data_genxpop, axis=1) # max : top
            top_stack[r_num,:] = top_data

            avg_trend_data = np.mean(data_genxpop, axis=1)
            avg_stack[r_num,:] = avg_trend_data

            # with open(fr'{data_dir}/{name}/run_data.bin','rb') as f:
            #     mean_pv, std_pv, time = pickle.load(f)
            # time_stack[r_num] = time

        l1 = ax1.plot(np.mean(top_stack, axis=0), 
                        # label = f'{group_name}: top individual (avg of {num_runs} runs, time: {int(np.mean(time_stack)/60)} min)',
                        label = f'{group_name}: top individual (avg of {num_runs} runs',
                        color=cmap(g_num/cmap_range), 
                        alpha=0.5
                        )
        lns.append(l1[0])

        l2 = ax1.plot(np.mean(avg_stack, axis=0), 
                        # label = f'{group_name}: population average (avg of {num_runs} runs, time: {int(np.mean(time_stack)/60)} min)',
                        label = f'{group_name}: population average (avg of {num_runs} runs',
                        color=cmap(g_num/cmap_range), 
                        linestyle='dotted', 
                        alpha=0.5
                        )
        lns.append(l2[0])
    
    ax1.set_xlabel('Generation')

    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left')

    # ax1.set_ylabel('# Patches Found')
    # ax1.set_ylim(0,8)
    ax1.set_ylabel('Time to Find Patch')
    ax1.set_ylim(-20,1720)

    if save_name: 
        plt.savefig(fr'{data_dir}/{save_name}.png')
    plt.show()


if __name__ == '__main__':


### ----------pop runs----------- ###

    # plot_mult_EA_trends([f'doublecorner_exp_CNN18_FNN2_e10p25_rep{x}' for x in range(20)], 'doublecorner_exp_CNN18_FNN2_e10p25')
    # plot_mult_EA_trends([f'doublecorner_exp_CNN18_FNN2_e15p25_rep{x}' for x in range(10)], 'doublecorner_exp_CNN18_FNN2_e15p25')
    # plot_mult_EA_trends([f'doublecorner_exp_CNN18_FNN1_p25e5_rep{x}' for x in range(9)], 'doublecorner_exp_CNN18_FNN1_p25e5')
    # plot_mult_EA_trends([f'doublecorner_exp_CNN1128_FNN1_p25e5_rep{x}' for x in range(20)], 'doublecorner_exp_CNN1128_FNN1_p25e5')

    # plot_mult_EA_trends_np([f'doublecorner_exp_CNN1124_FNN2_p25e10_mean_rep{x}' for x in range(1)], 'mean', 'doublecorner_exp_CNN1124_FNN2_p25e10')
    # plot_mult_EA_trends_np([f'doublecorner_exp_CNN1124_FNN2_p25e5_mean_rep{x}' for x in range(1)], 'mean', 'doublecorner_exp_CNN1124_FNN2_p25e5')
    # plot_mult_EA_trends_np([f'doublecorner_exp_CNN1128_FNN2_p25e10_median_rep{x}' for x in range(13)], 'doublecorner_exp_CNN1128_FNN2_p25e10_median')
    # plot_mult_EA_trends_np([f'doublecorner_exp_CNN1124_FNN2_p25e5_mean_resTRspwn_rep{x}' for x in range(1)], 'mean', 'doublecorner_exp_CNN1124_FNN2_p25e5_mean_resTRspwn')
    # plot_mult_EA_trends_np([f'singlecorner_exp_CNN1124_FNN2_p25e5_mean_rep{x}' for x in range(2)], 'mean', 'singlecorner_exp_CNN1124_FNN2_p25e5_mean')

    # plot_mult_EA_trends_np([f'singlecorner_exp_CNN1124_FNN2_p25e5_vis8_rep{x}' for x in range(20)], 'mean', 'singlecorner_exp_CNN1124_FNN2_vis8_p25e5')
    # plot_mult_EA_trends_np([f'singlecorner_exp_CNN1124_FNN2_p25e5_vis48_rep{x}' for x in range(20)], 'mean', 'singlecorner_exp_CNN1124_FNN2_vis48_p25e5')
    # plot_mult_EA_trends_np([f'singlecorner_exp_CNN14_FNN2_p25e5_vis8_rep{x}' for x in range(20)], 'mean', 'singlecorner_exp_CNN14_FNN2_vis8_p25e5')
    # plot_mult_EA_trends_np([f'singlecorner_exp_CNN14_FNN2_p25e5_vis48_rep{x}' for x in range(20)], 'mean', 'singlecorner_exp_CNN14_FNN2_vis48_p25e5')

    # plot_mult_EA_trends_np([f'singlecorner_exp_CNN1124_FNN2_p25e15_vis8_rep{x}' for x in range(3)], 'mean', 'singlecorner_exp_CNN1124_FNN2_vis8_p25e15')
    # plot_mult_EA_trends_np([f'singlecorner_exp_CNN1124_FNN2_p50e15_vis8_rep{x}' for x in range(5)], 'mean', 'singlecorner_exp_CNN1124_FNN2_vis8_p50e15')
    # plot_mult_EA_trends_np([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_rep{x}' for x in range(7)], 'mean', True, 'singlecorner_exp_CNN1124_FNN2_vis8_p50e20')

    # plot_mult_EA_trends_np([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_nodist_rep{x}' for x in range(2)], 'mean', 'singlecorner_exp_CNN1124_FNN2_vis8_p50e20_nodist')

    # plot_mult_EA_trends_np([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_wh500_rep{x}' for x in range(4)], 'mean', 'singlecorner_exp_CNN1124_FNN2_vis8_p50e20_wh500')
    # plot_mult_EA_trends_np([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_wcoll_rep{x}' for x in range(2)], 'mean', False, 'singlecorner_exp_CNN1124_FNN2_vis8_p50e20_wcoll')

    # plot_mult_EA_trends_np([f'singlecorner_exp_CNN1124_FNN16_p50e20_vis8_rep{x}' for x in range(5)], 'mean', True, 'singlecorner_exp_CNN1124_FNN16_vis8_p50e20')

    # plot_mult_EA_trends_np([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_sn05afCNN_rep{x}' for x in range(5)], 'mean', True, 'singlecorner_exp_CNN1124_FNN2_vis8_p50e20_sn05afCNN')
    # plot_mult_EA_trends_np([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_sn025afCNN_rep{x}' for x in range(5)], 'mean', True, 'singlecorner_exp_CNN1124_FNN2_vis8_p50e20_sn025afCNN')
    # plot_mult_EA_trends_np([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_sn05b4CNN_rep{x}' for x in range(2)], 'mean', True, 'singlecorner_exp_CNN1124_FNN2_vis8_p50e20_sn05b4CNN')

    # plot_mult_EA_trends_np([f'singlecorner_exp_CNN1128_FNN2_p50e20_vis8_rep{x}' for x in range(5)], 'mean', True, 'singlecorner_exp_CNN1128_FNN2_vis8_p50e20')
    # plot_mult_EA_trends_np([f'singlecorner_exp_CNN18_FNN2_p50e20_vis8_rep{x}' for x in range(5)], 'mean', True, 'singlecorner_exp_CNN18_FNN2_vis8_p50e20')
    # plot_mult_EA_trends_np([f'singlecorner_exp_CNN14_FNN2_p50e20_vis8_rep{x}' for x in range(5)], 'mean', True, 'singlecorner_exp_CNN14_FNN2_vis8_p50e20')

    # plot_mult_EA_trends_np(['singlecorner_exp_CNN1124_FNN2_p50e20_vis8_gen50k'], 'mean', False, 'singlecorner_exp_CNN1124_FNN2_vis8_p50e20_gen50k')

    # plot_mult_EA_trends_np([f'singlecorner_exp_CNN1124_FNN2_p100e20_vis8_rep{x}' for x in range(3)], 'mean', True, 'singlecorner_exp_CNN1124_FNN2_vis8_p100e20')
    # plot_mult_EA_trends_np([f'singlecorner_exp_CNN1124_FNN2_p200e20_vis8_rep{x}' for x in range(3)], 'mean', True, 'singlecorner_exp_CNN1124_FNN2_vis8_p200e20')

    # plot_mult_EA_trends_np([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss075_rep{x}' for x in range(5)], save_name='singlecorner_exp_CNN1124_FNN2_vis8_PGPE_ss075_p50e20')
    # plot_mult_EA_trends_np([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_rep{x}' for x in range(4)], save_name='singlecorner_exp_CNN1124_FNN2_vis8_PGPE_ss10_p50e20')
    # plot_mult_EA_trends_np([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_rep{x}' for x in range(5)], save_name='singlecorner_exp_CNN1124_FNN2_vis8_PGPE_ss15_p50e20')

### ----------group pop runs----------- ###

    groups = []

    # groups.append(('CNN14_FNN2_p25e5_vis8',[f'singlecorner_exp_CNN14_FNN2_p25e5_vis8_rep{x}' for x in range(20)]))
    # groups.append(('CNN14_FNN2_p25e5_vis48',[f'singlecorner_exp_CNN14_FNN2_p25e5_vis48_rep{x}' for x in range(20)]))
    # groups.append(('CNN1124_FNN2_p25e5_vis8',[f'singlecorner_exp_CNN1124_FNN2_p25e5_vis8_rep{x}' for x in range(20)]))
    # groups.append(('CNN1124_FNN2_p25e5_vis48',[f'singlecorner_exp_CNN1124_FNN2_p25e5_vis48_rep{x}' for x in range(20)]))
    # # groups.append(('CNN1124_FNN2_p25e5_vis8_noisevis5',[f'singlecorner_exp_CNN1124_FNN24_p25e5_vis8_noisevis5_rep{x}' for x in range(8)]))
    # plot_mult_EA_trends_groups_np(groups, 'groups_singlecorner_CNN-vis')


    # groups.append(('CNN1124_FNN2_p25e5_vis8',[f'singlecorner_exp_CNN1124_FNN2_p25e5_vis8_rep{x}' for x in range(20)]))
    # groups.append(('CNN1124_FNN2_p25e15_vis8',[f'singlecorner_exp_CNN1124_FNN2_p25e15_vis8_rep{x}' for x in range(3)]))
    # groups.append(('CNN1124_FNN2_p50e15_vis8',[f'singlecorner_exp_CNN1124_FNN2_p50e15_vis8_rep{x}' for x in range(5)]))
    # groups.append(('CNN1124_FNN2_p50e20_vis8',[f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_rep{x}' for x in range(5)]))
    # plot_mult_EA_trends_groups_np(groups, 'groups_singlecorner_pop_ep')

    # groups.append(('CNN1124_FNN2_p50e20_vis8',[f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_rep{x}' for x in range(5)]))
    # groups.append(('CNN1124_FNN16_p50e20_vis8', [f'singlecorner_exp_CNN1124_FNN16_p50e20_vis8_rep{x}' for x in range(5)]))
    # plot_mult_EA_trends_groups_np(groups, 'groups_singlecorner_FNN')

    # groups.append(('CNN1124_FNN2_p50e20_vis8',[f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_rep{x}' for x in range(5)]))
    # groups.append(('CNN1124_FNN2_p50e20_vis8_sn05afCNN', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_sn05afCNN_rep{x}' for x in range(5)]))
    # groups.append(('CNN1124_FNN2_p50e20_vis8_sn025afCNN', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_sn025afCNN_rep{x}' for x in range(5)]))
    # groups.append(('CNN1124_FNN2_p50e20_vis8_sn05b4CNN', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_sn05b4CNN_rep{x}' for x in range(2)]))
    # plot_mult_EA_trends_groups_np(groups, 'groups_singlecorner_noise')
    
    # groups.append(('CNN14_FNN2_p50e20_vis8', [f'singlecorner_exp_CNN14_FNN2_p50e20_vis8_rep{x}' for x in range(5)]))
    # groups.append(('CNN18_FNN2_p50e20_vis8', [f'singlecorner_exp_CNN18_FNN2_p50e20_vis8_rep{x}' for x in range(5)]))
    # groups.append(('CNN1124_FNN2_p50e20_vis8',[f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_rep{x}' for x in range(4)]))
    # groups.append(('CNN1128_FNN2_p50e20_vis8', [f'singlecorner_exp_CNN1128_FNN2_p50e20_vis8_rep{x}' for x in range(5)]))
    # plot_mult_EA_trends_groups_np(groups, 'groups_singlecorner_CNN')

    groups.append(('CMA-ES',[f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_rep{x}' for x in range(7)]))
    groups.append(('PGPE - ss075', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss075_rep{x}' for x in range(5)]))
    groups.append(('PGPE - ss10', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_rep{x}' for x in range(4)]))
    groups.append(('PGPE - ss15', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_rep{x}' for x in range(5)]))
    plot_mult_EA_trends_groups_np(groups, 'groups_singlecorner_optim')


### ----------violins----------- ###


    # plot_mult_EA_violins_np(['singlecorner_exp_CNN1124_FNN2_p50e20_vis8_gen50k'], 'singlecorner_exp_CNN1124_FNN2_vis8_p50e20_gen50k_stdev_violin')
    
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

    
    # root_dir = Path(__file__).parent.parent
    # data_dir = Path(root_dir, r'data/simulation_data')
    # name = 'doublecorner_exp_CNN1124_FNN2_p25e10_mean_rep0'

    # with open(fr'{data_dir}/{name}/fitness_spread_per_generation.bin','rb') as f:
    #     data = pickle.load(f)

    # # plot_EA_trend_violin(data, mean=True)
    # plot_EA_trend_violin(data, mean=True, save_dir=fr'{data_dir}/{name}')