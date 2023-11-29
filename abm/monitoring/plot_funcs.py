import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import collections as mc
# import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import pickle
from collections import deque
from itertools import islice

# ------------------------------- single sim map ---------------------------------------- #

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


def sliding_window(iterable, n):
  """
  sliding_window('ABCDEFG', 4) -> ABCD BCDE CDEF DEFG
  [recipe from python docs]
  """
  it = iter(iterable)
  window = deque(islice(it, n), maxlen=n)
  if len(window) == n:
      yield tuple(window)
  for x in it:
      window.append(x)
      yield tuple(window)

def color_gradient(x, y):
  """
  Creates a line collection with a gradient from colors c1 to c2
  https://stackoverflow.com/questions/8500700/how-to-plot-a-gradient-color-line [nog642]
  """
  n = len(x)
  if len(y) != n:
    raise ValueError('x and y data lengths differ')
  
  cm = plt.get_cmap('plasma')
  cm_disc = cm(np.linspace(0, 1, n-1, endpoint=False))
#   cm_disc = cm(np.linspace(1, 0, n-1, endpoint=False)) # flipped for some cmaps
#   cm_disc[:,-1] = np.linspace(0.2, 1, n-1, endpoint=False) # start with lower alpha

  return mc.LineCollection(sliding_window(zip(x, y), 2),
                        #    colors=np.linspace(c1, c2, n - 1),
                           colors=cm_disc,
                           linewidth=.1, alpha=.3, zorder=0)


# ------------------------------- iterative trajectory maps ---------------------------------------- #

def plot_map_iterative_traj(plot_data, x_max, y_max, w=8, h=8, save_name=None, ellipses=False):

    ag_data, res_data = plot_data

    fig, axes = plt.subplots() 
    axes.set_xlim(0, x_max)
    axes.set_ylim(0, y_max)

    # rescale plotting area to square
    l,r,t,b = fig.subplotpars.left, fig.subplotpars.right, fig.subplotpars.top, fig.subplotpars.bottom
    fig.set_size_inches( float(w)/(r-l) , float(h)/(t-b) )

    # agent trajectories as gradient lines
    N_ag = ag_data.shape[0]
    for agent in range(N_ag):
        pos_x = ag_data[agent,:,0]
        pos_y = ag_data[agent,:,1]
        axes.add_collection(color_gradient(ag_data[agent,:,0], ag_data[agent,:,1]))

    # resource patches via circles
    N_res = res_data.shape[0]
    for res in range(N_res):
        pos_x = res_data[res,0,0]
        pos_y = res_data[res,0,1]
        radius = res_data[res,0,2]
        axes.add_patch( plt.Circle((pos_x, pos_y), radius, edgecolor='k', fill=False, zorder=1) )
    
    if ellipses:
        pts = get_ellipses()

        # norm = mpl.colors.Normalize(vmin=0, vmax=360)
        # sc = plt.scatter(pts[:,0], pts[:,1], c=pts[:,2], cmap='plasma', norm=norm, alpha=.1, s=10)
        # cb = plt.colorbar(sc)
        # cb.solids.set(alpha=.7)
        plt.scatter(pts[:,0], pts[:,1], c='black', alpha=.1, s=10)

    if save_name:
        if ellipses:
            plt.savefig(fr'{save_name}_ellipses.png')
        else:
            plt.savefig(fr'{save_name}.png')
        plt.close()
    else:
        plt.show()


def get_ellipses(fov=.4, grid_length=1000):

    from itertools import combinations
    from math import isclose

    phis = np.linspace(-fov*np.pi, fov*np.pi, 8)
    positions = np.arange(0, grid_length, 1)

    pts = []

    # east wall
    angles = np.arange(0, 90+1, .1)
    angles = np.append(angles, np.arange(270, 360, .1))

    for angle in angles:
        view = angle * np.pi / 180 - phis

        for x_diff in positions:
            for ray1, ray2 in combinations(view, 2):

                y1 = np.tan(ray1) * x_diff
                if y1 < 0: # first (arbitratily) should be positive
                    continue

                y2 = np.tan(ray2) * x_diff
                if y2 > 0: # should be opposite sign
                    continue
                
                y_sep = y1 - y2
                if isclose(y_sep, grid_length, abs_tol=.2):
                    pts.append((grid_length-x_diff, grid_length-y1, angle))
    
    
    # north wall
    angles = np.arange(.1, 180, .1)

    for angle in angles:
        view = angle * np.pi / 180 - phis

        for y_diff in positions:
            for ray1, ray2 in combinations(view, 2):

                x1 = np.tan(ray1 - np.pi/2) * y_diff
                if x1 < 0: # first (arbitratily) should be positive
                    continue

                x2 = np.tan(ray2 - np.pi/2) * y_diff
                if x2 > 0: # should be opposite sign
                    continue

                y_sep = x1 - x2
                if isclose(y_sep, grid_length, abs_tol=.2):
                    pts.append((x1, grid_length-y_diff, angle))
    
    # west wall
    angles = np.arange(90.1, 270, .1)

    for angle in angles:
        view = angle * np.pi / 180 - phis
        for x_diff in positions:
            for ray1, ray2 in combinations(view, 2):

                y1 = np.tan(ray1) * x_diff
                if y1 < 0: # first (arbitratily) should be positive
                    continue

                y2 = np.tan(ray2) * x_diff
                if y2 > 0: # should be opposite sign
                    continue
                
                y_sep = y1 - y2
                if isclose(y_sep, grid_length, abs_tol=.2):
                    pts.append((x_diff, y1, angle))
    
    # south wall
    angles = np.arange(180.1, 360, .1)

    for angle in angles:
        view = angle * np.pi / 180 - phis
        for y_diff in positions:
            for ray1, ray2 in combinations(view, 2):

                x1 = np.tan(ray1 + np.pi/2) * y_diff
                if x1 < 0: # first (arbitratily) should be positive
                    continue

                x2 = np.tan(ray2 + np.pi/2) * y_diff
                if x2 > 0: # should be opposite sign
                    continue
                
                y_sep = x1 - x2
                if isclose(y_sep, grid_length, abs_tol=.2):
                    pts.append((grid_length-x1, y_diff, angle))
    
    return np.array(pts)


def plot_map_iterative_trajall(plot_data, x_max, y_max, w=8, h=8, save_name=None, var_pos=-1, inv=False, change=True, wall=0):

    ag_data, res_data = plot_data
    print(f'traj matrix {ag_data.shape}; var_pos [{var_pos}]; inv [{inv}]; change[{change}]; wall [{wall}]')

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

        if var_pos == 3 and wall == 0: # sensory input change
            sens = ag_data[agent,:,3:3+8]                 # gather matrix (timesteps, vis_field_res)
            var = abs(np.diff(sens, axis=0))              # take abs(diff) along timestep axis
            var = np.any(var>0, axis=1).astype(int)      # 0 if no change, 1 if change
            pos_x, pos_y = pos_x[1:], pos_y[1:]          # cut first pos point

        elif var_pos == 3 and wall != 0: # sensory input change
            sens = ag_data[agent,:,3:3+8]                 # gather matrix (timesteps, vis_field_res)
            var = np.diff(sens, axis=0)                   # take diff) along timestep axis
            var = sens[:-1,:]*(var != 0)                  # pass mask over sensory input for varying elements
            var = np.any(var==wall, axis=1).astype(int)   # 0 if wall input did not change during timestep, 1 if change
            pos_x, pos_y = pos_x[:-1], pos_y[:-1]           # cut last pos point
        
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

# ------------------------------- violins ---------------------------------------- #

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


def plot_EA_mult_trend_violin(names, save_name=False, gap=None, scale=None):

    root_dir = Path(__file__).parent.parent
    data_dir = Path(root_dir, r'data/simulation_data')

    group = np.array(())

    for name in names:

        with open(fr'{data_dir}/{name}/fitness_spread_per_generation.bin','rb') as f:
            data = pickle.load(f)

        # if est_method == 'mean':
        #     data_genxpop = np.mean(data, axis=2)
        # else:
        #     data_genxpop = np.median(data, axis=2)
        data_genxpop = data.reshape((1000,1000))

        # transpose from shape: (number of generations, population size)
        #             to shape: (population size, number of generations)
        data_popxgen = data_genxpop.transpose()
	
        if group.shape[0] > 0:
            group = np.concatenate((group,data_popxgen), axis=0)
        else:
            group = data_popxgen

    # take out gap if exists
    if gap:
        group[group > 1000] -= gap
    if scale:
        group[group > 1000] *= scale
        group[group > 1000] -= 1000

    # plot population distributions + means of fitnesses for each generation
    plt.violinplot(group, widths=1, showmeans=True, showextrema=False)
    plt.ylim([-50,2050])

    if save_name: 
        plt.savefig(fr'{data_dir}/{save_name}.png')
        plt.close()
    else: 
        plt.show()

def plot_mult_EA_param_violins(names, data='mean', save_name=None):

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

        if data == 'mean':
            trend_data = mean_pv.transpose()
        else:
            trend_data = std_pv.transpose()

        l0 = ax1.violinplot(trend_data[:,::10], 
                    widths=1, 
                    # showmeans=True, 
                    showextrema=False,
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


# ------------------------------- EA trends ---------------------------------------- #

def plot_mult_EA_trends(names, inter=False, val=None, group_est='mean', save_name=None):

    # establish load directory
    root_dir = Path(__file__).parent.parent
    data_dir = Path(root_dir, r'data/simulation_data')

    # # init plot details
    fig, ax1 = plt.subplots(figsize=(15,10)) 
    # # ax2 = ax1.twinx()
    cmap = plt.get_cmap('hsv')
    cmap_range = len(names)
    lns = []
    val_avgs = []
    val_diffs = []
    group_avg = []
    group_top = []
    
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

        data_genxpop = np.mean(data, axis=2)
        if inter: data_genxpop = np.ma.masked_equal(data_genxpop, 0)

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
                        alpha=0.2
                        )
        lns.append(l1[0])
        group_top.append(top_data)

        avg_trend_data = np.mean(data_genxpop, axis=1)
        l2 = ax1.plot(avg_trend_data, 
                        label = f'avg {name}',
                        # label = f'avg {name} | t: {int(time)} sec',
                        color=cmap(i/cmap_range), 
                        linestyle='dotted',
                        alpha=0.2
                        )
        # lns.append(l2[0])
        group_avg.append(avg_trend_data)

        # parse val results text file if exists
        if val == 'top':
            if Path(fr'{data_dir}/{name}/val_results.txt').is_file():
                with open(fr'{data_dir}/{name}/val_results.txt') as f:
                    lines = f.readlines()

                    val_data = np.zeros((len(lines)-1, 3))
                    for n, line in enumerate(lines[1:]):
                        data = [item.strip() for item in line.split(' ')]
                        val_data[n,0] = data[1] # generation
                        val_data[n,1] = data[4] # train fitness
                        val_data[n,2] = data[7] # val fitness

                    top_ind = np.argsort(val_data[:,2])[:3] # min : top
                    # top_ind = np.argsort(top_data, axis=2)[-1:-4:-1] # max : top

                    top_gen = [val_data[i,0] for i in top_ind]
                    top_valfit = [val_data[i,2] for i in top_ind]
                    for g,f in zip(top_gen, top_valfit):
                        print(f'val | gen {int(g)}: fit {int(f)}')

                ax1.vlines(val_data[:,0], val_data[:,1], val_data[:,2],
                        color='black'
                        )
                
                avg_val = np.mean(val_data[:,2])
                val_avgs.append(avg_val)

                ax1.hlines(avg_val, i*5, data_genxpop.shape[0] + i*5,
                        color=cmap(i/cmap_range),
                        # linestyle='dashed',
                        alpha=0.5
                        )

        elif val == 'cen':
            if Path(fr'{data_dir}/{name}/val_results_cen.txt').is_file():
                with open(fr'{data_dir}/{name}/val_results_cen.txt') as f:
                    lines = f.readlines()

                    val_data = np.zeros((len(lines)-1, 3))
                    for n, line in enumerate(lines[1:]):
                        data = [item.strip() for item in line.split(' ')]
                        val_data[n,0] = data[1] # generation
                        val_data[n,1] = data[4] # train fitness
                        val_data[n,2] = data[7] # val fitness

                    top_ind = np.argsort(val_data[:,2])[:3] # min : top
                    # top_ind = np.argsort(top_data, axis=2)[-1:-4:-1] # max : top

                    top_gen = [val_data[i,0] for i in top_ind]
                    top_valfit = [val_data[i,2] for i in top_ind]
                    for g,f in zip(top_gen, top_valfit):
                        print(f'val | gen {int(g)}: fit {int(f)}')

                val_diffs.append(np.mean(val_data[:,2] - val_data[:,1]))

                ax1.vlines(val_data[:,0], val_data[:,1], val_data[:,2],
                        color='black',
                        alpha=0.5
                        )
                ax1.scatter(val_data[:,0], val_data[:,2], color=cmap(i/cmap_range), edgecolor='black')
                
                avg_val = np.mean(val_data[:,2])
                val_avgs.append(avg_val)

                ax1.hlines(avg_val, i*5, data_genxpop.shape[0] + i*5,
                        color=cmap(i/cmap_range),
                        linestyle='dashed',
                        alpha=0.5
                        )
    
    group_top = np.array(group_top)
    if group_est == 'mean':
        est_trend = np.mean(group_top, axis=0)
    elif group_est == 'median':
        est_trend = np.median(group_top, axis=0)
    lt = ax1.plot(est_trend, 
                    label = f'{group_est} of group top',
                    color='k', 
                    alpha=.5
                    )
    lns.append(lt[0])
    
    group_avg = np.array(group_avg)
    if group_est == 'mean':
        est_trend = np.mean(group_avg, axis=0)
    elif group_est == 'median':
        est_trend = np.median(group_avg, axis=0)
    la = ax1.plot(est_trend, 
                    label = f'{group_est} of group avg',
                    color='k', 
                    linestyle='dotted',
                    alpha=.5
                    )
    lns.append(la[0])

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

    if val is not None:
        ax1.set_title(f'overall validated run avg: {int(np.mean(val_avgs))} | val diffs: {int(np.mean(val_diffs))}')

    if save_name: 
        plt.savefig(fr'{data_dir}/{save_name}.png')
    plt.show()


def plot_mult_EA_trends_groups(groups, inter=False, save_name=None):

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
            if inter: data_genxpop = np.ma.masked_equal(data_genxpop, 0)
            top_data = np.min(data_genxpop, axis=1) # min : top
            # top_data = np.max(data_genxpop, axis=1) # max : top
            top_stack[r_num,:] = top_data

            avg_trend_data = np.mean(data_genxpop, axis=1)
            avg_stack[r_num,:] = avg_trend_data

            # with open(fr'{data_dir}/{name}/run_data.bin','rb') as f:
            #     mean_pv, std_pv, time = pickle.load(f)
            # time_stack[r_num] = time

        l1 = ax1.plot(np.median(top_stack, axis=0), 
                        # label = f'{group_name}: top individual (avg of {num_runs} runs, time: {int(np.mean(time_stack)/60)} min)',
                        label = f'{group_name}: top individual (avg of {num_runs} runs)',
                        color=cmap(g_num/cmap_range), 
                        alpha=0.5
                        )
        lns.append(l1[0])

        l2 = ax1.plot(np.median(avg_stack, axis=0), 
                        # label = f'{group_name}: population average (avg of {num_runs} runs, time: {int(np.mean(time_stack)/60)} min)',
                        label = f'{group_name}: population average (avg of {num_runs} runs)',
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

    # plot_mult_EA_trends([f'doublecorner_exp_CNN1124_FNN2_p25e10_mean_rep{x}' for x in range(1)], 'mean', 'doublecorner_exp_CNN1124_FNN2_p25e10')
    # plot_mult_EA_trends([f'doublecorner_exp_CNN1124_FNN2_p25e5_mean_rep{x}' for x in range(1)], 'mean', 'doublecorner_exp_CNN1124_FNN2_p25e5')
    # plot_mult_EA_trends([f'doublecorner_exp_CNN1128_FNN2_p25e10_median_rep{x}' for x in range(13)], 'doublecorner_exp_CNN1128_FNN2_p25e10_median')
    # plot_mult_EA_trends([f'doublecorner_exp_CNN1124_FNN2_p25e5_mean_resTRspwn_rep{x}' for x in range(1)], 'mean', 'doublecorner_exp_CNN1124_FNN2_p25e5_mean_resTRspwn')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p25e5_mean_rep{x}' for x in range(2)], 'mean', 'singlecorner_exp_CNN1124_FNN2_p25e5_mean')

    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p25e5_vis8_rep{x}' for x in range(20)], 'mean', 'singlecorner_exp_CNN1124_FNN2_vis8_p25e5')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p25e5_vis48_rep{x}' for x in range(20)], 'mean', 'singlecorner_exp_CNN1124_FNN2_vis48_p25e5')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN14_FNN2_p25e5_vis8_rep{x}' for x in range(20)], 'mean', 'singlecorner_exp_CNN14_FNN2_vis8_p25e5')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN14_FNN2_p25e5_vis48_rep{x}' for x in range(20)], 'mean', 'singlecorner_exp_CNN14_FNN2_vis48_p25e5')

    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p25e15_vis8_rep{x}' for x in range(4)], save_name='singlecorner_exp_CNN1124_FNN2_vis8_p25e15')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p50e15_vis8_rep{x}' for x in range(5)], save_name='singlecorner_exp_CNN1124_FNN2_vis8_p50e15')

    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_nodist_rep{x}' for x in range(2)], 'mean', 'singlecorner_exp_CNN1124_FNN2_vis8_p50e20_nodist')

    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_wh500_rep{x}' for x in range(4)], save_name='singlecorner_exp_CNN1124_FNN2_vis8_p50e20_wh500')

    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_sn05afCNN_rep{x}' for x in range(5)], 'mean', True, 'singlecorner_exp_CNN1124_FNN2_vis8_p50e20_sn05afCNN')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_sn025afCNN_rep{x}' for x in range(5)], 'mean', True, 'singlecorner_exp_CNN1124_FNN2_vis8_p50e20_sn025afCNN')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_sn05b4CNN_rep{x}' for x in range(2)], 'mean', True, 'singlecorner_exp_CNN1124_FNN2_vis8_p50e20_sn05b4CNN')

    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN16_p50e20_vis8_rep{x}' for x in range(5)], save_name='singlecorner_exp_CNN1124_FNN16_vis8_p50e20')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN1128_FNN2_p50e20_vis8_rep{x}' for x in range(5)], save_name='singlecorner_exp_CNN1128_FNN2_vis8_p50e20')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN18_FNN2_p50e20_vis8_rep{x}' for x in range(5)], save_name='singlecorner_exp_CNN18_FNN2_vis8_p50e20')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN14_FNN2_p50e20_vis8_rep{x}' for x in range(5)], save_name='singlecorner_exp_CNN14_FNN2_vis8_p50e20')

    # plot_mult_EA_trends(['singlecorner_exp_CNN1124_FNN2_p50e20_vis8_gen50k'], 'mean', False, 'singlecorner_exp_CNN1124_FNN2_vis8_p50e20_gen50k')

    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_rep{x}' for x in range(8)], save_name='singlecorner_exp_CNN1124_FNN2_vis8_p50e20')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p100e20_vis8_rep{x}' for x in range(3)], save_name='singlecorner_exp_CNN1124_FNN2_vis8_p100e20')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p200e20_vis8_rep{x}' for x in range(3)], save_name='singlecorner_exp_CNN1124_FNN2_vis8_p200e20')

    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss075_rep{x}' for x in range(5)], save_name='singlecorner_exp_CNN1124_FNN2_vis8_PGPE_ss075_p50e20')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_rep{x}' for x in range(5)], save_name='singlecorner_exp_CNN1124_FNN2_vis8_PGPE_ss10_p50e20')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_rep{x}' for x in range(5)], save_name='singlecorner_exp_CNN1124_FNN2_vis8_PGPE_ss15_p50e20')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom9_rep{x}' for x in range(5)], save_name='singlecorner_exp_CNN1124_FNN2_vis8_PGPE_ss20_mom9_p50e20')

    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_mom8_rep{x}' for x in range(5)], save_name='singlecorner_exp_CNN1124_FNN2_vis8_PGPE_ss10_mom8_p50e20')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom8_rep{x}' for x in range(14)], save_name='singlecorner_exp_CNN1124_FNN2_vis8_PGPE_ss15_mom8_p50e20')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(5)], save_name='singlecorner_exp_CNN1124_FNN2_vis8_PGPE_ss20_mom8_p50e20')

    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom7_rep{x}' for x in range(5)], save_name='singlecorner_exp_CNN1124_FNN2_vis8_PGPE_ss15_mom7_p50e20')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_mom7_rep{x}' for x in range(4)], save_name='singlecorner_exp_CNN1124_FNN2_vis8_PGPE_ss10_mom7_p50e20')

    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom6_rep{x}' for x in range(1)], save_name='singlecorner_exp_CNN1124_FNN2_vis8_PGPE_ss15_mom6_p50e20')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom5_rep{x}' for x in range(1)], save_name='singlecorner_exp_CNN1124_FNN2_vis8_PGPE_ss15_mom5_p50e20')


    # plot_mult_EA_trends([f'singlecorner_exp_CNN18_FNN2_p50e20_vis8_PGPE_ss15_mom8_rep{x}' for x in range(3)], save_name='singlecorner_exp_CNN18_FNN2_vis8_PGPE_ss15_mom8_p50e20')


    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom6_rep{x}' for x in range(20)], save_name='singlecorner_exp_CNN1124_FNN2_vis8_PGPE_ss20_mom6_p50e20')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom7_rep{x}' for x in range(10)], save_name='singlecorner_exp_CNN1124_FNN2_vis8_PGPE_ss20_mom7_p50e20')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(53)], save_name='singlecorner_exp_CNN1124_FNN2_vis8_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom9_rep{x}' for x in range(10)], save_name='singlecorner_exp_CNN1124_FNN2_vis8_PGPE_ss20_mom9_p50e20')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_gap200_rep{x}' for x in range(20)], save_name='singlecorner_exp_CNN1124_FNN2_vis8_PGPE_ss20_mom8_gap200_p50e20')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_scalehalf_rep{x}' for x in range(20)], save_name='singlecorner_exp_CNN1124_FNN2_vis8_PGPE_ss20_mom8_scalehalf_p50e20')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_gap200_scalshalf_rep{x}' for x in range(2)], save_name='singlecorner_exp_CNN1124_FNN2_vis8_PGPE_ss20_mom8_gap200_scalehalf_p50e20')

    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep{x}' for x in range(20)], save_name='singlecorner_exp_CNN1124_FNN2_vis6_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(53)], save_name='singlecorner_exp_CNN1124_FNN2_vis8_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis9_PGPE_ss20_mom8_rep{x}' for x in range(20)], save_name='singlecorner_exp_CNN1124_FNN2_vis9_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep{x}' for x in range(26)], save_name='singlecorner_exp_CNN1124_FNN2_vis10_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN1122_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(15)], save_name='singlecorner_exp_CNN1122_FNN2_vis8_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)], save_name='singlecorner_exp_CNN14_FNN2_vis8_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)], save_name='singlecorner_exp_CNN13_FNN2_vis8_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN12_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)], save_name='singlecorner_exp_CNN12_FNN2_vis8_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_GRU4_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)], save_name='singlecorner_exp_CNN1124_GRU4_vis8_PGPE_ss20_mom8_p50e20')


    # plot_mult_EA_trends([f'singlecorner_exp_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_block52_rep{x}' for x in range(20)], save_name='singlecorner_exp_CNN14_FNN2_vis8_block52_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_block52_rep{x}' for x in range(20)], save_name='singlecorner_exp_CNN14_FNN2_vis10_block52_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep{x}' for x in range(9)], save_name='singlecorner_exp_CNN14_FNN2_vis10_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN14_FNN1_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)], save_name='singlecorner_exp_CNN14_FNN1_vis8_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN13_FNN32_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(9)], save_name='singlecorner_exp_CNN13_FNN32_vis8_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN13_FNN1_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)], save_name='singlecorner_exp_CNN13_FNN1_vis8_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov2_rep{x}' for x in range(20)], save_name='singlecorner_exp_CNN14_FNN2_vis8_fov2_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep{x}' for x in range(9)], save_name='singlecorner_exp_CNN14_FNN2_vis8_fov35_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov45_rep{x}' for x in range(11)], save_name='singlecorner_exp_CNN14_FNN2_vis8_fov45_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov6_rep{x}' for x in range(20)], save_name='singlecorner_exp_CNN14_FNN2_vis8_fov6_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov875_rep{x}' for x in range(20)], save_name='singlecorner_exp_CNN14_FNN2_vis8_fov875_PGPE_ss20_mom8_p50e20')


    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='singlecorner_exp_CNN1124_FNN2_vis6_PGPE_ss20_mom8_p50e20_valcen')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='singlecorner_exp_CNN1124_FNN2_vis8_PGPE_ss20_mom8_p50e20_valcen') # 53 total
    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis9_PGPE_ss20_mom8_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='singlecorner_exp_CNN1124_FNN2_vis9_PGPE_ss20_mom8_p50e20_valcen')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='singlecorner_exp_CNN1124_FNN2_vis10_PGPE_ss20_mom8_p50e20_valcen') # 26 total

    # plot_mult_EA_trends([f'singlecorner_exp_CNN1122_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(15)], val='cen', 
    #                     save_name='singlecorner_exp_CNN1122_FNN2_vis8_PGPE_ss20_mom8_p50e20_valcen')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='singlecorner_exp_CNN14_FNN2_vis8_PGPE_ss20_mom8_p50e20_valcen')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='singlecorner_exp_CNN13_FNN2_vis8_PGPE_ss20_mom8_p50e20_valcen')

    # plot_mult_EA_trends([f'singlecorner_exp_CNN14_FNN1_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='singlecorner_exp_CNN14_FNN1_vis8_PGPE_ss20_mom8_p50e20_valcen')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN13_FNN1_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='singlecorner_exp_CNN13_FNN1_vis8_PGPE_ss20_mom8_p50e20_valcen')
    
    # plot_mult_EA_trends([f'singlecorner_exp_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov2_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='singlecorner_exp_CNN14_FNN2_vis8_PGPE_ss20_mom8_fov2_p50e20_valcen')
    plot_mult_EA_trends([f'singlecorner_exp_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep{x}' for x in range(9)], val='cen', 
                        save_name='singlecorner_exp_CNN14_FNN2_vis8_PGPE_ss20_mom8_fov35_p50e20_valcen')
    plot_mult_EA_trends([f'singlecorner_exp_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov45_rep{x}' for x in range(11)], val='cen', 
                        save_name='singlecorner_exp_CNN14_FNN2_vis8_PGPE_ss20_mom8_fov45_p50e20_valcen')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov6_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='singlecorner_exp_CNN14_FNN2_vis8_PGPE_ss20_mom8_fov6_p50e20_valcen')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov875_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='singlecorner_exp_CNN14_FNN2_vis8_PGPE_ss20_mom8_fov875_p50e20_valcen')

    # plot_mult_EA_trends([f'singlecorner_exp_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_block52_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='singlecorner_exp_CNN14_FNN2_vis8_PGPE_ss20_mom8_block52_p50e20_valcen')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_block52_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='singlecorner_exp_CNN14_FNN2_vis10_PGPE_ss20_mom8_block52_p50e20_valcen')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_block52_rep{x}' for x in range(9)], val='cen', 
    #                     save_name='singlecorner_exp_CNN14_FNN2_vis10_PGPE_ss20_mom8_block52_p50e20_valcen')

    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_GRU4_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='singlecorner_exp_CNN1124_GRU4_vis8_PGPE_ss20_mom8_p50e20_valcen')



    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN1_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(1)], save_name='singlecorner_exp_CNN1124_FNN1_p50e20_vis8_PGPE_ss20_mom8')

### ----------pop runs >> num_inter ----------- ###

    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_mom9_inter_rep{x}' for x in range(11)], inter=True, save_name='singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_mom9_inter')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_mom10_inter_rep{x}' for x in range(10)], inter=True, save_name='singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_mom10_inter')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom9_inter_rep{x}' for x in range(7)], inter=True, save_name='singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom9_inter')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom8_inter_rep{x}' for x in range(10)], inter=True, save_name='singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom8_inter')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_inter_rep{x}' for x in range(7)], inter=True, save_name='singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_inter')
    # plot_mult_EA_trends([f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom7_inter_rep{x}' for x in range(4)], inter=True, save_name='singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom7_inter')

    # groups = []
    # # groups.append(('ss10_mom9', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_mom9_inter_rep{x}' for x in range(11)]))
    # # groups.append(('ss10_mom10', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_mom10_inter_rep{x}' for x in range(9)]))
    # # groups.append(('ss15_mom7', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom7_inter_rep{x}' for x in range(3)]))
    # groups.append(('ss15_mom9', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom8_inter_rep{x}' for x in range(10)]))
    # # groups.append(('ss15_mom9', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom9_inter_rep{x}' for x in range(6)]))
    # # groups.append(('ss20_mom8', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_inter_rep{x}' for x in range(6)]))
    # plot_mult_EA_trends_groups(groups, inter=True, save_name='groups_singlecorner_PGPE_inter_mom8')

### ----------group pop runs----------- ###

    groups = []

    # groups.append(('CNN14_FNN2_p25e5_vis8',[f'singlecorner_exp_CNN14_FNN2_p25e5_vis8_rep{x}' for x in range(20)]))
    # groups.append(('CNN14_FNN2_p25e5_vis48',[f'singlecorner_exp_CNN14_FNN2_p25e5_vis48_rep{x}' for x in range(20)]))
    # groups.append(('CNN1124_FNN2_p25e5_vis8',[f'singlecorner_exp_CNN1124_FNN2_p25e5_vis8_rep{x}' for x in range(20)]))
    # groups.append(('CNN1124_FNN2_p25e5_vis48',[f'singlecorner_exp_CNN1124_FNN2_p25e5_vis48_rep{x}' for x in range(20)]))
    # # groups.append(('CNN1124_FNN2_p25e5_vis8_noisevis5',[f'singlecorner_exp_CNN1124_FNN24_p25e5_vis8_noisevis5_rep{x}' for x in range(8)]))
    # plot_mult_EA_trends_groups(groups, save_name='groups_singlecorner_CNN-vis')

    # groups.append(('CNN1124_FNN2_p25e5_vis8',[f'singlecorner_exp_CNN1124_FNN2_p25e5_vis8_rep{x}' for x in range(20)]))
    # groups.append(('CNN1124_FNN2_p25e15_vis8',[f'singlecorner_exp_CNN1124_FNN2_p25e15_vis8_rep{x}' for x in range(3)]))
    # groups.append(('CNN1124_FNN2_p50e15_vis8',[f'singlecorner_exp_CNN1124_FNN2_p50e15_vis8_rep{x}' for x in range(5)]))
    # groups.append(('CNN1124_FNN2_p50e20_vis8',[f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_rep{x}' for x in range(5)]))
    # plot_mult_EA_trends_groups(groups, save_name='groups_singlecorner_pop_ep')

    # groups.append(('CNN1124_FNN2_p50e20_vis8',[f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_rep{x}' for x in range(5)]))
    # groups.append(('CNN1124_FNN16_p50e20_vis8', [f'singlecorner_exp_CNN1124_FNN16_p50e20_vis8_rep{x}' for x in range(5)]))
    # plot_mult_EA_trends_groups(groups, save_name='groups_singlecorner_FNN')

    # groups.append(('CNN1124_FNN2_p50e20_vis8',[f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_rep{x}' for x in range(5)]))
    # groups.append(('CNN1124_FNN2_p50e20_vis8_sn05afCNN', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_sn05afCNN_rep{x}' for x in range(5)]))
    # groups.append(('CNN1124_FNN2_p50e20_vis8_sn025afCNN', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_sn025afCNN_rep{x}' for x in range(5)]))
    # groups.append(('CNN1124_FNN2_p50e20_vis8_sn05b4CNN', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_sn05b4CNN_rep{x}' for x in range(2)]))
    # plot_mult_EA_trends_groups(groups, save_name='groups_singlecorner_noise')
    
    # groups.append(('CNN14_FNN2_p50e20_vis8', [f'singlecorner_exp_CNN14_FNN2_p50e20_vis8_rep{x}' for x in range(5)]))
    # groups.append(('CNN18_FNN2_p50e20_vis8', [f'singlecorner_exp_CNN18_FNN2_p50e20_vis8_rep{x}' for x in range(5)]))
    # groups.append(('CNN1124_FNN2_p50e20_vis8',[f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_rep{x}' for x in range(4)]))
    # groups.append(('CNN1128_FNN2_p50e20_vis8', [f'singlecorner_exp_CNN1128_FNN2_p50e20_vis8_rep{x}' for x in range(5)]))
    # plot_mult_EA_trends_groups(groups, save_name='groups_singlecorner_CNN')

    # groups.append(('CMA-ES - pop50',[f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_rep{x}' for x in range(7)]))
    # groups.append(('CMA-ES - pop100',[f'singlecorner_exp_CNN1124_FNN2_p100e20_vis8_rep{x}' for x in range(3)]))
    # groups.append(('CMA-ES - pop200',[f'singlecorner_exp_CNN1124_FNN2_p200e20_vis8_rep{x}' for x in range(3)]))
    # groups.append(('PGPE - pop50 - ss075 - mom9', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss075_rep{x}' for x in range(5)]))
    # groups.append(('PGPE - pop50 - ss10 - mom9', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_rep{x}' for x in range(5)]))
    # groups.append(('PGPE - pop50 - ss15 - mom9', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_rep{x}' for x in range(5)]))
    # groups.append(('PGPE - pop50 - ss10 - mom8', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_mom8_rep{x}' for x in range(5)]))
    # groups.append(('PGPE - pop50 - ss15 - mom8', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom8_rep{x}' for x in range(5)]))
    # groups.append(('PGPE - pop50 - ss20 - mom8', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(5)]))
    # groups.append(('PGPE - pop50 - ss10 - mom7', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_mom7_rep{x}' for x in range(4)]))
    # groups.append(('PGPE - pop50 - ss15 - mom7', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom7_rep{x}' for x in range(5)]))
    # plot_mult_EA_trends_groups(groups, save_name='groups_singlecorner_optim')

    # groups.append(('ss10 - mom7', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_mom7_rep{x}' for x in range(4)]))
    # groups.append(('ss15 - mom7', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom7_rep{x}' for x in range(5)]))
    # groups.append(('ss20 - mom7', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom7_rep{x}' for x in range(5)]))
    # groups.append(('ss10 - mom8', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_mom8_rep{x}' for x in range(5)]))
    # groups.append(('ss15 - mom8', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom8_rep{x}' for x in range(14)]))
    # groups.append(('ss20 - mom8', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(11)]))
    # groups.append(('ss075 - mom9', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss075_rep{x}' for x in range(5)]))
    # groups.append(('ss10 - mom9', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_rep{x}' for x in range(5)]))
    # groups.append(('ss15 - mom9', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_rep{x}' for x in range(5)]))
    # groups.append(('ss20 - mom9', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom9_rep{x}' for x in range(9)]))
    # plot_mult_EA_trends_groups(groups, save_name='groups_singlecorner_optim_by_mom')

    # groups.append(('ss075 - mom9', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss075_rep{x}' for x in range(5)]))
    # groups.append(('ss10 - mom7', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_mom7_rep{x}' for x in range(4)]))
    # groups.append(('ss10 - mom8', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_mom8_rep{x}' for x in range(5)]))
    # groups.append(('ss10 - mom9', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_rep{x}' for x in range(5)]))
    # groups.append(('ss15 - mom7', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom7_rep{x}' for x in range(5)]))
    # groups.append(('ss15 - mom8', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom8_rep{x}' for x in range(14)]))
    # groups.append(('ss15 - mom9', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_rep{x}' for x in range(5)]))
    # groups.append(('ss20 - mom6', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom6_rep{x}' for x in range(3)]))
    # groups.append(('ss20 - mom7', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom7_rep{x}' for x in range(16)]))
    # groups.append(('ss20 - mom8', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('ss20 - mom9', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom9_rep{x}' for x in range(16)]))
    # plot_mult_EA_trends_groups(groups, save_name='groups_singlecorner_optim_by_ss')

    # groups.append(('ss20 - mom6', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom6_rep{x}' for x in range(20)]))
    # groups.append(('ss20 - mom7', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom7_rep{x}' for x in range(20)]))
    # groups.append(('ss20 - mom8', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('ss20 - mom9', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom9_rep{x}' for x in range(20)]))
    # plot_mult_EA_trends_groups(groups, save_name='groups_singlecorner_optim_by_ss20')

    # groups.append(('ss20 - mom8', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('ss20 - mom8 - gap200', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_gap200_rep{x}' for x in range(20)]))
    # groups.append(('ss20 - mom8 - scalehalf', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_scalehalf_rep{x}' for x in range(20)]))
    # groups.append(('ss20 - mom8 - gap200 + scalehalf', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_gap200_scalehalf_rep{x}' for x in range(7)]))
    # plot_mult_EA_trends_groups(groups, save_name='groups_singlecorner_optim_by_ss20_mom8_shaping')

    # groups.append(('CMA-ES - pop50',[f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_rep{x}' for x in range(7)]))
    # groups.append(('CMA-ES - pop100',[f'singlecorner_exp_CNN1124_FNN2_p100e20_vis8_rep{x}' for x in range(3)]))
    # groups.append(('CMA-ES - pop200',[f'singlecorner_exp_CNN1124_FNN2_p200e20_vis8_rep{x}' for x in range(3)]))
    # plot_mult_EA_trends_groups(groups, save_name='groups_singlecorner_optim_CMAES')


    # groups.append(('ss20 - mom7', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom7_rep{x}' for x in range(10)]))
    # groups.append(('ss20 - mom8', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('ss20 - mom9', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom9_rep{x}' for x in range(10)]))
    # plot_mult_EA_trends_groups(groups, save_name='groups_singlecorner_optim_by_ss20')



    # groups.append(('vis 6', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('vis 8', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(53)]))
    # groups.append(('vis 9', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis9_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('vis 10', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep{x}' for x in range(26)]))
    # plot_mult_EA_trends_groups(groups, save_name='groups_singlecorner_vis')

    # groups.append(('CNN 12', [f'singlecorner_exp_CNN12_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('CNN 13', [f'singlecorner_exp_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('CNN 14', [f'singlecorner_exp_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('CNN 1122', [f'singlecorner_exp_CNN1122_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(15)]))
    # groups.append(('CNN 1124', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(53)]))
    # plot_mult_EA_trends_groups(groups, save_name='groups_singlecorner_CNN')

    # groups.append(('FNN 2', [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(53)]))
    # groups.append(('GRU 4', [f'singlecorner_exp_CNN1124_GRU4_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # plot_mult_EA_trends_groups(groups, save_name='groups_singlecorner_integrator_memory')

    # groups = []
    # groups.append(('block31 - vis8', [f'singlecorner_exp_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('block31 - vis10', [f'singlecorner_exp_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep{x}' for x in range(9)]))
    # groups.append(('block52 - vis8', [f'singlecorner_exp_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_block52_rep{x}' for x in range(20)]))
    # groups.append(('block52 - vis10', [f'singlecorner_exp_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_block52_rep{x}' for x in range(20)]))
    # plot_mult_EA_trends_groups(groups, save_name='groups_singlecorner_CNN_kernels')

    # groups = []
    # groups.append(('CNN14 - FNN 1', [f'singlecorner_exp_CNN14_FNN1_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('CNN14 - FNN 2', [f'singlecorner_exp_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('CNN13 - FNN 1', [f'singlecorner_exp_CNN13_FNN1_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('CNN13 - FNN 2', [f'singlecorner_exp_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # # groups.append(('CNN13 - FNN 32', [f'singlecorner_exp_CNN13_FNN32_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(8)]))
    # plot_mult_EA_trends_groups(groups, save_name='groups_singlecorner_integrator_size')

    # groups = []
    # groups.append(('fov2', [f'singlecorner_exp_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov2_rep{x}' for x in range(20)]))
    # groups.append(('fov35', [f'singlecorner_exp_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(9)]))
    # groups.append(('fov4', [f'singlecorner_exp_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('fov45', [f'singlecorner_exp_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(11)]))
    # groups.append(('fov6', [f'singlecorner_exp_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov6_rep{x}' for x in range(20)]))
    # groups.append(('fov875', [f'singlecorner_exp_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov875_rep{x}' for x in range(20)]))
    # plot_mult_EA_trends_groups(groups, save_name='groups_singlecorner_FOV')


### ----------violins----------- ###

    # name = 'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom6_rep0'
    # plot_mult_EA_violins([name], 'stdev', name+'_stdev_violin')
    # plot_mult_EA_violins([name], 'mean', name+'_mean_violin')

    # name = 'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_rep0'
    # plot_mult_EA_violins([name], 'stdev', name+'_stdev_violin')
    # plot_mult_EA_violins([name], 'mean', name+'_mean_violin')

    # names = [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom6_rep{x}' for x in range(20)]
    # plot_EA_mult_trend_violin(names, 'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom6_violin')

    # names = [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom7_rep{x}' for x in range(20)]
    # plot_EA_mult_trend_violin(names, 'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom7_violin')

    #names = [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]
    #plot_EA_mult_trend_violin(names, 'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_violin')

    # names = [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom9_rep{x}' for x in range(20)]
    # plot_EA_mult_trend_violin(names, 'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom9_violin')

    # names = [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_gap200_rep{x}' for x in range(20)]
    # plot_EA_mult_trend_violin(names, 'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_gap200_violin')
    # plot_EA_mult_trend_violin(names, 'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_gap200_violin_rescale', gap=200)

    # names = [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_scalehalf_rep{x}' for x in range(20)]
    # plot_EA_mult_trend_violin(names, 'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_scalehalf_violin')
    # plot_EA_mult_trend_violin(names, 'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_scalehalf_violin_rescale', scale=2)