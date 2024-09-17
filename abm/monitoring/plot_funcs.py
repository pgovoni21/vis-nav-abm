import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import collections as mc
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.patches as mpatches
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
        r.append(np.sqrt(dx**2 + dy**2))
    r = np.array(r)

    # set arrow spacing
    num_arrows = int(len(x) / 40)
    aspace = r.sum() / num_arrows
    
    # rtot is a cumulative sum of r, it's used to save time
    rtot = []
    for i in range(len(r)):
        rtot.append(r[0:i].sum())
    rtot.append(r.sum())

    arrowData = [] # will hold tuples of x,y,theta for each arrow
    arrowPos = 0 # set inital arrow position at first space
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


def arrows_3d(axes, x, y, z, ahl=6, ahw=3):

    # number of line segments per interval
    ds = 1 # length
    Ns = np.round(np.sqrt( (x[1:]-x[:-1])**2 + (y[1:]-y[:-1])**2 + (z[1:]-z[:-1])**2 ) / ds).astype(int)

    # sub-divide intervals w.r.t. Ns
    subdiv = lambda x, Ns=Ns: np.concatenate([ np.linspace(x[ii], x[ii+1], Ns[ii]) for ii, _ in enumerate(x[:-1]) ])
    x, y, z = subdiv(x), subdiv(y), subdiv(z)

    axes.quiver(x[:-1], y[:-1], z[:-1], 
            x[1:]-x[:-1], y[1:]-y[:-1], z[1:]-z[:-1], 
            #  scale_units='xyz', angles='xyz', 
            #  scale=1, width=.004, headlength=4, headwidth=4
            # length=.5,
            # arrow_length_ratio=1,
            color='black', alpha=0.2,
            )

    # n_pts = x.shape[0]
    # stack = np.vstack((x,y,z))

    # print(stack.shape)

    # r = np.zeros(n_pts) # distance spanned between pairs of points
    # rtot = [0.] # cum sum of r
    # for i in range(n_pts-1):
    #     r[i+1] = np.linalg.norm(stack[:,i+1] - stack[:,i])
    #     rtot.append(r[0:i+1].sum())
    # rtot.append(r.sum())

    # print(r.shape, rtot.shape)

    # # set arrow spacing
    # num_arrows = int(len(x) / 40)
    # aspace = r.sum() / num_arrows

    # arrowData = [] # will hold tuples of x,y,theta for each arrow
    # arrowPos = 0 # set inital arrow position at first space
    # ndrawn = 0
    # rcount = 1 

    # while arrowPos < r.sum() and ndrawn < num_arrows:
    #     x1, x2 = x[rcount-1], x[rcount]
    #     y1, y2 = y[rcount-1], y[rcount]
    #     z1, z2 = z[rcount-1], z[rcount]

    #     theta_xy = np.arctan2((x2-x1),(y2-y1))
    #     theta_xz = np.arctan2((x2-x1),(z2-z1))
    #     da = arrowPos - rtot[rcount]

    #     ax = np.sin(theta_xy)*da + x1
    #     ay = np.cos(theta_xy)*da + y1
    #     az = np.cos(theta_xz)*da + y1

    #     arrowData.append((ax, ay, az, theta_xy, theta_xz))

    #     ndrawn += 1
    #     arrowPos += aspace
    #     while arrowPos > rtot[rcount+1]: 
    #         rcount += 1
    #         if arrowPos > rtot[-1]:
    #             break

    # print(len(arrowData))

    # for ax, ay, az, theta_xy, theta_xz in arrowData[1:]:
    #     # use aspace as a guide for size and length of things
    #     # scaling factors were chosen by experimenting a bit

    #     dx0 = np.sin(theta_xy)*ahl/2. + ax
    #     dy0 = np.cos(theta_xy)*ahl/2. + ay
    #     dz0 = np.cos(theta_xz)*ahl/2. + az

    #     dx1 = -1.*np.sin(theta_xy)*ahl/2. + ax
    #     dy1 = -1.*np.cos(theta_xy)*ahl/2. + ay
    #     dz1 = -1.*np.cos(theta_xz)*ahl/2. + az

    #     axes.annotate('', xy=(dx0, dy0, dz0), xytext=(dx1, dy1, dz1),
    #             arrowprops=dict( headwidth=ahw, headlength=ahl, ec='black', fc='black', zorder=1))


def sliding_window(iterable, n):
  """
  sliding_window('ABCDEFG', 4) -> ABCD BCDE CDEF DEFG
  [recipe from python docs: https://docs.python.org/3/library/itertools.html]
  """
  it = iter(iterable)
  window = deque(islice(it, n-1), maxlen=n)
  for x in it:
      window.append(x)
      yield tuple(window)

def sliding_window_ori(iterable, n):
  """
  sliding_window('ABCDEFG', 4) -> ABCD BCDE CDEF DEFG
  [recipe from python docs: https://docs.python.org/3/library/itertools.html]
  + polar (cyclic) boundary conditions on third element (orientation)
  """
  it = iter(iterable)
  window = deque(islice(it, n-1), maxlen=n)
  last = ''

  for x in it:
      window.append(x)
  
      ptA,ptB = tuple(window)

      if ptB[2] - ptA[2] < -3:
        last = 'topout'
        yield (ptA, (ptA[0], ptA[1], 2*np.pi))
        
      elif ptB[2] - ptA[2] > 3:
        last = 'bottomout'
        yield (ptA, (ptA[0], ptA[1], 0))

      else:

        if last == 'topout':
          last = ''
          yield ((ptA[0], ptA[1], 0), ptA)
          yield tuple(window)

        elif last == 'bottomout':
          last = ''
          yield ((ptA[0], ptA[1], 2*np.pi), ptA)
          yield tuple(window)

        else:
          yield tuple(window)


def color_gradient(x, y, lw=.1, alp=.3):
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
                           colors=cm_disc,
                           linewidth=lw, alpha=alp, zorder=0)

def color_gradient_3d(x, y, z, lw=.1, alp=.1):
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

  return Line3DCollection(sliding_window_ori(zip(x, y, z), 2),
                           colors=cm_disc,
                           capstyle="round",
                           linewidth=lw, alpha=alp, zorder=0)


# ------------------------------- iterative trajectory maps ---------------------------------------- #

def plot_map_iterative_traj(plot_data, x_max, y_max, w=8, h=8, save_name=None, ellipses=False, ex_lines=False, act_mat=False, envconf=None, extra='', landmarks=(), dpi=50):

    ag_data, res_data = plot_data

    fig, axes = plt.subplots() 
    axes.set_xlim(0, x_max)
    axes.set_ylim(0, y_max)

    # rescale plotting area to square
    l,r,t,b = fig.subplotpars.left, fig.subplotpars.right, fig.subplotpars.top, fig.subplotpars.bottom
    fig.set_size_inches( float(w)/(r-l) , float(h)/(t-b) )

    if extra == 'turn':
        turn = abs(ag_data[:,:,2])
        min_turn, max_turn = 0, np.pi/2
        turn = (turn - min_turn) / (max_turn - min_turn)
        ag_data[:,:,2] = turn

    # agent trajectories as gradient lines
    N_ag = ag_data.shape[0]
    delay = 25
    for agent in range(N_ag):
    # for agent in range(N_ag)[::100]:
        pos_x = ag_data[agent,:,0]
        pos_y = ag_data[agent,:,1]

        if extra == 'turn':
            turn = ag_data[agent,:,2]
            # norm = mpl.colors.Normalize(vmin=0, vmax=np.pi/8)
            # axes.add_collection(plt.scatter(pos_x, pos_y, 
            #                             c=turn, cmap='plasma_r', norm=norm, alpha=.01, s=1))
            axes.scatter(pos_x[delay:], pos_y[delay:], c=turn[delay:], cmap='Blues', alpha=turn[delay:]*.1, s=5)
        else:
            axes.add_collection(color_gradient(pos_x[delay:], pos_y[delay:]))

    # resource patches via circles
    N_res = res_data.shape[0]
    for res in range(N_res):
        pos_x, pos_y, radius = res_data[res,0,:]
        axes.add_patch( plt.Circle((pos_x, pos_y), radius, edgecolor='k', fill=False, zorder=1) )
    
    # landmarks via circles
    if landmarks:
        lm_radius, pts = landmarks
        for pos in pts:
            axes.add_patch( plt.Circle(pos, lm_radius, edgecolor='k', fill=False, zorder=1) )
    
    if ellipses:
        pts = get_ellipses()
        plt.scatter(pts[:,0], pts[:,1], c='black', alpha=.1, s=10)

    if ex_lines:
        from scipy import spatial

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
            distance, index_xy = spatial.KDTree(ag_data[:,0,:2]).query(pt[:2])
            # search locally for best ori
            array = ag_data[index_xy:index_xy+16,0,2]
            ori = pt[2]
            index_ori = (np.abs(array - ori)).argmin()
            # combine + find traj
            index = index_xy + index_ori
            pos_x = ag_data[index,:,0]
            pos_y = ag_data[index,:,1]
            # ori = ag_data[index,:,2]
            # turn = ag_data[index,:3]

            axes.plot(pos_x, pos_y, color)
            axes.plot(pos_x, pos_y, 'k:')
            axes.plot(pos_x[0], pos_y[0], marker='o', c=color, markeredgecolor='k', ms=10)
    
    if isinstance(act_mat, np.ndarray):
        from scipy import spatial

        # inits = [ # CNN12
        #     [150, 650, np.pi/2, [.75,.725,.2,.2]],
        #     [200, 750, np.pi/2, [.75,.5,.2,.2]],
        #     [430, 320, np.pi, [.75,.275,.2,.2]],
        #     [380, 332, np.pi, [.75,.05,.2,.2]],
        # ]
        inits = [ # CNN14, vis24, rep18
            [400, 300, np.pi/2, 0, [.75,.725,.2,.2]],
            [600, 250, np.pi/2, 0, [.75,.5,.2,.2]],
            [400, 300, np.pi/2, 62, [.75,.275,.2,.2]],
            [600, 250, np.pi/2, 52, [.75,.05,.2,.2]],
        ]

        colors = [
            'cornflowerblue',
            'tomato',
            'forestgreen',
            'gold',
        ]

        width, height = tuple(eval(envconf["ENV_SIZE"]))
        x_min, x_max = 0, width
        y_min, y_max = 0, height
        coll_boundary_thickness = int(envconf["RADIUS_AGENT"])
        space_step = 25

        x_range = np.linspace(x_min + coll_boundary_thickness, 
                            x_max - coll_boundary_thickness + 1, 
                            int((width - coll_boundary_thickness*2) / space_step))
        y_range = np.linspace(y_min + coll_boundary_thickness, 
                            y_max - coll_boundary_thickness + 1, 
                            int((height - coll_boundary_thickness*2) / space_step))

        for (x,y,ori,t,inset_loc),color in zip(inits,colors):

            distance, index_xy = spatial.KDTree(ag_data[:,0,:2]).query(np.array([x,y]))
            array = ag_data[index_xy:index_xy+16,0,2]
            index_ori = (np.abs(array - ori)).argmin()
            index = index_xy + index_ori
            pos_x = ag_data[index,:,0]
            pos_y = ag_data[index,:,1]
            # ori = ag_data[index,:,2]
            # turn = ag_data[index,:3]
            # print(ag_data[index,0,:])

            # axes.plot(pos_x, pos_y, color)
            axes.plot(pos_x, pos_y, 'k:')
            # axes.plot(pos_x[0], pos_y[0], marker='o', c=color, markeredgecolor='k', ms=10)
            axes.plot(pos_x[t], pos_y[t], marker='o', c=color, markeredgecolor='k', ms=10)


            # find action array
            x,y = pos_x[t], pos_y[t]
            x_idx = (np.abs(x_range - x)).argmin()
            y_idx = (np.abs(y_range - y)).argmin()
            act_arr = act_mat[x_idx,y_idx,:]
            act_len = len(act_arr)

            print(x,y,ori)
            print(act_arr)
            # axes.plot(x, y, marker='x', c='k', markeredgecolor=None, ms=5)

            ins = axes.inset_axes(inset_loc, polar=True)
            ins.set_yticks([])
            ins.set_xticks([])
            # labels = ['$0$', r'$\pi/4$',  r'$\pi/2$', r'$3\pi/4$', r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$', ]
            # ins.set_xticks(ins.get_xticks())
            # ins.set_xticklabels(labels)
            ins.spines[:].set_color(color)

            orient_range = np.arange(0, 2*np.pi, 2*np.pi/act_len)
            widths = np.ones(act_len)*(2*np.pi/act_len)

            # calc radii, where fwd/turn : area
            area = act_arr / act_len
            radius = (area / np.pi) ** .5

            my_cmap = plt.get_cmap('plasma')
            rescale = lambda z: (z - np.min(z)) / (np.max(z) - np.min(z))
            ins.bar(orient_range, radius, align='edge', width=widths, 
                    edgecolor=my_cmap(act_arr), fill=False, linewidth=1, alpha=.7)
            # ins.set_theta_offset(offset=0)


    if save_name:
        if ellipses:
            plt.savefig(fr'{save_name}_ellipses_{dpi}.png', dpi=dpi)
        elif extra == 'turn':
            plt.savefig(fr'{save_name}_turn_{dpi}.png', dpi=dpi)
        elif ex_lines:
            plt.savefig(fr'{save_name}_ex_lines_{dpi}.png', dpi=dpi)
        elif isinstance(act_mat, np.ndarray):
            plt.savefig(fr'{save_name}_act_arr_{dpi}.png', dpi=dpi)
        else:
            plt.savefig(fr'{save_name}_{dpi}.png', dpi=dpi)
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
    angles = np.arange(0, 90, .1)
    angles = np.append(angles, np.arange(270.1, 360, .1))

    for angle in angles:
        view = angle * np.pi / 180 - phis

        for x_diff in positions:
            for ray1, ray2 in combinations(view, 2):

                y1 = np.tan(ray1) * x_diff
                if y1 < 0: # should be positive
                    continue

                y2 = np.tan(ray2) * x_diff
                if y2 > 0: # should negative
                    continue

                # if np.cos(ray1) < 0: # should be positive
                #     continue
                # if np.cos(ray2) < 0: # should be positive
                #     continue
                
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
                if x1 < 0: # should be positive
                    continue

                x2 = np.tan(ray2 - np.pi/2) * y_diff
                if x2 > 0: # should negative
                    continue

                # if np.cos(ray1) < 0: # should be positive
                #     continue
                # if np.cos(ray2) < 0: # should be positive
                #     continue

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
                if y1 < 0: # should be positive
                    continue

                y2 = np.tan(ray2) * x_diff
                if y2 > 0: # should negative
                    continue

                # if np.cos(ray1) < 0: # should be positive
                #     continue
                # if np.cos(ray2) < 0: # should be positive
                #     continue
                
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
                if x1 < 0: # should be positive
                    continue

                x2 = np.tan(ray2 + np.pi/2) * y_diff
                if x2 > 0: # should negative
                    continue

                # if np.cos(ray1) < 0: # should be positive
                #     continue
                # if np.cos(ray2) < 0: # should be positive
                #     continue
                
                y_sep = x1 - x2
                if isclose(y_sep, grid_length, abs_tol=.2):
                    pts.append((grid_length-x1, y_diff, angle))
    
    return np.array(pts)


def plot_map_iterative_traj_3d(plot_data, x_max, y_max, w=8, h=8, save_name=None, plt_type='scatter', var='turn'):
    print(f'plotting 3d: {plt_type} | {var}')

    ag_data, res_data = plot_data

    fig = plt.figure(
        # figsize=(25, 25), 
        )
    axes = fig.add_subplot(projection='3d')
    axes.set_xlim3d(0, x_max)
    axes.set_ylim3d(0, y_max)
    axes.set_zlim3d(0, np.pi*2)

    # pull out first X timesteps
    ag_data = ag_data[:,30:,:]

    # print(f'init pos_x: {ag_data[:,0,0].min()}, {ag_data[:,0,0].max()}')
    # print(f'init pos_y: {ag_data[:,0,1].min()}, {ag_data[:,0,1].max()}')
    # print(f'init ori: {ag_data[:,0,2].min()}, {ag_data[:,0,2].max()}')
    # # print(f'init turn: {ag_data[:,0,3].min()}, {ag_data[:,0,3].max()}')

    # print(f'pos_x: {ag_data[:,:,0].min()}, {ag_data[:,:,0].max()}')
    # print(f'pos_y: {ag_data[:,:,1].min()}, {ag_data[:,:,1].max()}')
    # print(f'ori: {ag_data[:,:,2].min()}, {ag_data[:,:,2].max()}')
    # # print(f'turn: {ag_data[:,:,3].min()}, {ag_data[:,:,3].max()}')

    if plt_type == 'scatter':
        # flatten
        N_ag, N_ts, N_feat = ag_data.shape
        ag_data_flat = ag_data.reshape((N_ag*N_ts, N_feat))

        # pull apart flattened feature vectors
        pos_x = ag_data_flat[:,0]
        pos_y = ag_data_flat[:,1]
        ori = ag_data_flat[:,2]
        turn = ag_data_flat[:,3]

        # color turning only
        if var == 'cturn':
            axes.scatter(pos_x, pos_y, ori, c=turn, cmap='plasma', alpha=.025, s=.1)

        # differentiate straight manifolds (colored via ori) vs turning (black)
        elif var == 'str_manif':
            min_turn = np.min(abs(turn))
            max_turn = np.max(abs(turn))
            turn_morevis_norm = (abs(turn) - min_turn) / (max_turn - min_turn) # turns more visible
            turn_lessvis_norm = -(abs(turn) - max_turn) / (max_turn - min_turn) # turns less visible

            axes.scatter(pos_x, pos_y, ori, c='k', alpha=turn_morevis_norm*.025, s=turn_morevis_norm*.01)
            axes.scatter(pos_x, pos_y, ori, c=ori, cmap='plasma', alpha=turn_lessvis_norm*.025, s=turn_lessvis_norm*.1)


    # agent trajectories as gradient lines
    elif plt_type == 'lines':

        if var == 'ctime_arrows_only':
            pass
        else:
            N_ag = ag_data.shape[0]
            for agent in range(N_ag):
                pos_x = ag_data[agent,:,0]
                pos_y = ag_data[agent,:,1]
                ori = ag_data[agent,:,2]
                turn = ag_data[agent,:,3]

                cm = plt.get_cmap('plasma')
                cm_disc = cm(np.linspace(0, 1, len(pos_x)-1, endpoint=False)) # edges bw points, thus 1 less

                if var == 'cturn':
                    line = Line3DCollection(sliding_window_ori(zip(pos_x, pos_y, ori), 2),
                                            cmap='plasma_r',
                                            array=turn,
                                            capstyle="round",
                                            linewidth=.1, alpha=.1, zorder=0)

                elif var == 'ctime' or var == 'ctime_arrows':
                    line = Line3DCollection(sliding_window_ori(zip(pos_x, pos_y, ori), 2),
                                            colors=cm_disc,
                                            capstyle="round",
                                            linewidth=.1, alpha=.1, zorder=0)

                elif var == 'ctime_flat':
                    line = Line3DCollection(sliding_window_ori(zip(pos_x, pos_y, ori/20), 2),
                                            colors=cm_disc,
                                            capstyle="round",
                                            linewidth=.1, alpha=.1, zorder=0)

                axes.add_collection(line)

        
        if var == 'ctime_arrows':
            per_ag = 2000
            per_ts = 1
            ag_data = ag_data[::per_ag,::per_ts,:]

            for agent in range(ag_data.shape[0]):
                pos_x = ag_data[agent,:,0]
                pos_y = ag_data[agent,:,1]
                ori = ag_data[agent,:,2]
                turn = ag_data[agent,:3]

                # arrows_3d(axes, pos_x, pos_y, ori) # --> blurry
                line = Line3DCollection(sliding_window_ori(zip(pos_x, pos_y, ori), 2),
                                        colors='black',
                                        capstyle='round',
                                        linewidth=1, alpha=.5, zorder=1)
                axes.add_collection(line)
        
        elif var == 'ctime_arrows_only':
            per_ag = 2000
            per_ts = 1
            ag_data = ag_data[::per_ag,::per_ts,:]

            for agent in range(ag_data.shape[0]):
                pos_x = ag_data[agent,:,0]
                pos_y = ag_data[agent,:,1]
                ori = ag_data[agent,:,2]
                turn = ag_data[agent,:3]

                # test if within patch + clip traj
                center_x, center_y = 400, 600
                radius = 50
                for i, (x,y) in enumerate(zip(pos_x,pos_y)):
                    if (x - center_x)**2 + (y - center_y)**2 <= radius**2:
                        break

                # before patch
                sw = sliding_window_ori(zip(pos_x[:i], pos_y[:i], ori[:i]),2)
                line = Line3DCollection(sw,
                                        cmap=plt.get_cmap('plasma'),
                                        norm=plt.Normalize(0,np.pi*2),
                                        capstyle='round',
                                        linewidth=1, alpha=.5, zorder=1)
                sw = sliding_window_ori(zip(pos_x[:i], pos_y[:i], ori[:i]),2) # reinitate generator for coloring
                ori_array = np.array([seg[0][2] for seg in sw]) # array of orientation of first point in segment
                line.set_array(ori_array) # --> use for ori cmap
                # line.set_array(np.linspace(0, i/len(ori_array), i-1, endpoint=False)) # --> use for time cmap
                axes.add_collection(line)

                # after reaching patch
                line = Line3DCollection(sliding_window_ori(zip(pos_x[i:], pos_y[i:], ori[i:]), 2),
                                        colors='black',
                                        capstyle='round',
                                        linewidth=1, alpha=.5, zorder=1)
                axes.add_collection(line)
    
    # axes.set_zticks(np.arange(0, 2*np.pi+0.01, np.pi/2))
    # labels = ['$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']
    # axes.set_zticklabels(labels)
    axes.set_xticklabels([])
    axes.set_yticklabels([])
    axes.set_zticklabels([])

    if save_name:
        plt.savefig(fr'{save_name}_{plt_type}_{var}.png', dpi=100)
        plt.close()

    else:
        plt.show()


def plot_map_iterative_trajall(plot_data, x_max, y_max, w=8, h=8, save_name=None, var_pos=-1, inv=False, change=True, wall=0):

    ag_data, res_data = plot_data
    print(f'traj matrix {ag_data.shape}; var_pos [{var_pos}]; inv [{inv}]; change[{change}]; wall [{wall}]')

    vis_field_res = 8

    fig, axes = plt.subplots() 
    axes.set_xlim(0, x_max)
    axes.set_ylim(0, y_max)

    # rescale plotting area to square
    l,r,t,b = fig.subplotpars.left, fig.subplotpars.right, fig.subplotpars.top, fig.subplotpars.bottom
    fig.set_size_inches( float(w)/(r-l) , float(h)/(t-b) )

    # configure coloring
    if var_pos == -1: # action --> narrow around zero (straight)
        ag_data[:,:,-1] = abs(ag_data[:,:,-1])
        
        local_max = np.max(ag_data[:,:,-1])
        local_min = np.min(ag_data[:,:,-1])
        print(f'local max: {round(local_max,2)}, local min: {round(local_min,2)}')
        cmap = 'Blues'
        norm = mpl.colors.Normalize(vmin=0, vmax=local_max)
    elif var_pos == 0: # sensory input change --> binary
        if change: cmap = mpl.colors.ListedColormap(['w','k']) # black : changing 
        else:      cmap = mpl.colors.ListedColormap(['k','w']) # black : no change
        norm = mpl.colors.Normalize()
    else: # Nact --> cmap limits as broad as outermost value
        cmap = mpl.cm.bwr
        # CNN_output_size = 2
        # global_max = np.max(ag_data[:,:, 3+vis_field_res : 3+vis_field_res+CNN_output_size])
        # global_min = np.min(ag_data[:,:, 3+vis_field_res : 3+vis_field_res+CNN_output_size])
        # print(f'global max: {round(global_max,2)}, global min: {round(global_min,2)}')
        local_max = np.max(ag_data[:,:, var_pos])
        local_min = np.min(ag_data[:,:, var_pos])
        print(f'local max: {round(local_max,2)}, local min: {round(local_min,2)}')
        # lim = round(np.maximum(abs(max), abs(min)),2)
        # norm = mpl.colors.Normalize(vmin=-lim, vmax=lim)
        norm = mpl.colors.Normalize(vmin=local_min, vmax=local_max)
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

        if var_pos == 0 and wall == 0: # sensory input change
            sens = ag_data[agent,:,3:3+vis_field_res]    # gather matrix (timesteps, vis_field_res)
            var = abs(np.diff(sens, axis=0))             # take abs(diff) along timestep axis
            var = np.any(var>0, axis=1).astype(int)      # 0 if no change, 1 if change
            pos_x, pos_y = pos_x[1:], pos_y[1:]          # cut first pos point

        elif var_pos == 0 and wall != 0: # wall input change
            sens = ag_data[agent,:, 3 : 3+vis_field_res]     # gather matrix (timesteps, vis_field_res)
            var = np.diff(sens, axis=0)                   # take diff along timestep axis
            var = sens[:-1,:]*(var != 0)                  # pass mask over sensory input for varying elements
            var = np.any(var==wall, axis=1).astype(int)   # 0 if wall input did not change during timestep, 1 if change
            pos_x, pos_y = pos_x[:-1], pos_y[:-1]         # cut last pos point
        
        # plot variable, whether sensory input or neural activity
        if var_pos == -1 or var_pos == 0:
            axes.scatter(pos_x[25:], pos_y[25:], c=var[25:], alpha=var[25:]*.1,
                        s=1, cmap=cmap, norm=norm)
        else:
            axes.scatter(pos_x[25:], pos_y[25:], c=var[25:], alpha=.01,
                        s=1, cmap=cmap, norm=norm) 

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
    top_vals_overall = np.zeros((3,0))
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

        # avg_trend_data = np.mean(data_genxpop, axis=1)
        # l2 = ax1.plot(avg_trend_data, 
        #                 label = f'avg {name}',
        #                 # label = f'avg {name} | t: {int(time)} sec',
        #                 color=cmap(i/cmap_range), 
        #                 linestyle='dotted',
        #                 alpha=0.2
        #                 )
        # # lns.append(l2[0])
        # group_avg.append(avg_trend_data)

        # parse val results text file if exists
        if val is not None:
            if val == 'top': filename = 'val_results'
            elif val == 'cen': filename = 'val_results_cen'

            if Path(fr'{data_dir}/{name}/{filename}.txt').is_file():
                with open(fr'{data_dir}/{name}/{filename}.txt') as f:
                    lines = f.readlines()

                    val_data = np.zeros((len(lines)-1, 3))
                    for n, line in enumerate(lines[1:]):
                        data = [item.strip() for item in line.split(' ')]
                        val_data[n,0] = data[1] # generation
                        val_data[n,1] = data[4] # train fitness
                        val_data[n,2] = data[7] # val fitness

                    print(i, val_data[:,2])

                    top_ind = np.argsort(val_data[:,2])[:3] # min : top
                    # top_ind = np.argsort(top_data, axis=2)[-1:-4:-1] # max : top

                    top_gen = [val_data[i,0] for i in top_ind]
                    top_valfit = [val_data[i,2] for i in top_ind]
                    for g,f in zip(top_gen, top_valfit):
                        print(f'val | gen {int(g)}: fit {int(f)}')

                    top_vals_current = np.array(([i], [top_gen[0]], [top_valfit[0]]))
                    top_vals_overall = np.hstack((top_vals_overall, top_vals_current))

                val_diffs.append(np.mean((val_data[:,2] - val_data[:,1])**2))

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
            else:
                print(fr'{data_dir}/{name}/{filename}.txt is not a file')
    
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
    
    # group_avg = np.array(group_avg)
    # if group_est == 'mean':
    #     est_trend = np.mean(group_avg, axis=0)
    # elif group_est == 'median':
    #     est_trend = np.median(group_avg, axis=0)
    # la = ax1.plot(est_trend, 
    #                 label = f'{group_est} of group avg',
    #                 color='k', 
    #                 linestyle='dotted',
    #                 alpha=.5
    #                 )
    # lns.append(la[0])

    ax1.set_xlabel('Generation')

    # labs = [l.get_label() for l in lns]
    # ax1.legend(lns, labs, loc='upper right')
    # ax1.legend(lns, labs, loc='lower left')
    # ax1.legend(lns, labs, loc='upper left')

    ax1.set_ylabel('Time to Find Patch')
    # ax1.set_ylim(-20,1720)
    ax1.set_ylim(-20,1520)
    # ax1.set_ylim(1900,5000)

    # ax1.set_ylabel('# Patches Found')
    # ax1.set_ylim(0,8)

    if val is not None:
        top_val_inds = np.argsort(top_vals_overall[2,:])
        top_reps = top_vals_overall[0,:][top_val_inds]
        top_gens = top_vals_overall[1,:][top_val_inds]
        top_vals = top_vals_overall[2,:][top_val_inds]
        
        ax1.set_title(f'overall validated run avg: {int(np.mean(val_avgs))} | val var: {int((np.mean(val_diffs))**.5)} | top val run: {int(top_vals[0])}')

        top_num = 20
        for rep, gen, val_fit in zip(top_reps[:top_num], top_gens[:top_num], top_vals[:top_num]):
            print(f'overall val | rep {int(rep)} | gen {int(gen)} | fit {int(val_fit)}')

    if save_name: 
        plt.savefig(fr'{data_dir}/{save_name}.png')
    plt.show()

def plot_mult_EA_trends_new(names, inter=False, val=None, group_est='mean', save_name=None):

    # establish load directory
    root_dir = Path(__file__).parent.parent
    data_dir = Path(root_dir, r'data/simulation_data')

    # # init plot details
    fig, ax1 = plt.subplots(figsize=(15,10)) 
    # # ax2 = ax1.twinx()
    cmap = plt.get_cmap('hsv')
    cmap_range = len(names)
    lns = []
    
    num_runs = len(names)
    top_stack = np.zeros((num_runs,1000))
    avg_stack = np.zeros((num_runs,1000))
    val_stack = np.zeros(num_runs)

    for r_num, name in enumerate(names):

        with open(fr'{data_dir}/{name}/fitness_spread_per_generation.bin','rb') as f:
            data = pickle.load(f)

        data_genxpop = np.mean(data, axis=2)
        if inter: data_genxpop = np.ma.masked_equal(data_genxpop, 0)
        top_data = np.min(data_genxpop, axis=1) # min : top
        # top_data = np.max(data_genxpop, axis=1) # max : top
        top_stack[r_num,:] = top_data

        avg_trend_data = np.mean(data_genxpop, axis=1)
        avg_stack[r_num,:] = avg_trend_data

        if val is not None:
            if val == 'top': filename = 'val_results'
            elif val == 'cen': filename = 'val_results_cen'

            if Path(fr'{data_dir}/{name}/{filename}.txt').is_file():
                with open(fr'{data_dir}/{name}/{filename}.txt') as f:
                    lines = f.readlines()

                    val_data = np.zeros((len(lines)-1, 3))
                    for n, line in enumerate(lines[1:]):
                        data = [item.strip() for item in line.split(' ')]
                        val_data[n,0] = data[1] # generation
                        val_data[n,1] = data[4] # train fitness
                        val_data[n,2] = data[7] # val fitness

                avg_val = np.mean(val_data[:,2])
                val_stack[r_num] = avg_val

        l1 = ax1.plot(top_stack[r_num,:], 
                        # label = f'{group_name}: top individual (avg of {num_runs} runs)',
                        label = f'{name}',
                        color=cmap(r_num/cmap_range), 
                        alpha=0.05,
                        linewidth=1,
                        )
        lns.append(l1[0])

        # l2 = ax1.plot(avg_stack[r_num,:], 
        #                 # label = f'{group_name}: population average (avg of {num_runs} runs)',
        #                 color=cmap(r_num/cmap_range), 
        #                 # linestyle='dotted', 
        #                 alpha=0.05,
        #                 linewidth=1,
        #                 )
        # lns.append(l2[0])

    if val is not None:
        if group_est == 'mean':
            val_group = np.mean(val_stack, axis=0)
        elif group_est == 'median':
            val_group = np.median(val_stack, axis=0)
        print(f'avg val: {int(val_group)}')
        if val_group != 0:
            ax1.hlines(val_group, 0, data_genxpop.shape[0],
                    color='k',
                    linestyle='dashed',
                    alpha=0.5
                    )

    if group_est == 'mean':
        est_trend = np.mean(top_stack, axis=0)
    elif group_est == 'median':
        est_trend = np.median(top_stack, axis=0)
    la = ax1.plot(est_trend, 
                    label = f'{group_est} of group avg',
                    color='k', 
                    # linestyle='dotted',
                    alpha=.5
                    )
    lns.append(la[0])

    ax1.set_xlabel('Generation')

    # labs = [l.get_label() for l in lns]
    # ax1.legend(lns, labs, loc='upper left')

    ax1.set_ylabel('Time to Find Patch')
    ax1.set_ylim(-20,1480)

    if save_name: 
        plt.savefig(fr'{data_dir}/{save_name}.png')
    plt.show()


def plot_mult_EA_trends_groups(groups, inter=False, val=None, group_est='mean', save_name=None):

    # establish load directory
    root_dir = Path(__file__).parent.parent
    data_dir = Path(root_dir, r'data/simulation_data')

    # # init plot details
    # fig, ax1 = plt.subplots(figsize=(15,10)) 
    fig, ax1 = plt.subplots(figsize=(6,4)) 
    cmap = plt.get_cmap('plasma')
    cmap_range = len(groups)
    lns = []

    # add perfect trajectory
    # # lns.append(ax1.plot([], [], 'k', label='perfect')[0])
    #with open(fr'{data_dir}/nonNN/perfect.bin','rb') as f:
    #    data = pickle.load(f)
    #if group_est == 'mean':
    #    perf = np.mean(data)
    #elif group_est == 'median':
    #    perf = np.median(data)
    #l = ax1.hlines(perf, -5, 1000-5,
    #        label='perfect',
    #        color='k',
    #        linestyle='dashed',
    #        alpha=0.5
    #        )
    #lns.append(l)
    
    # iterate over each file
    for g_num, (group_name, run_names) in enumerate(groups):

        num_runs = len(run_names)
        top_stack = np.zeros((num_runs,1000))
        avg_stack = np.zeros((num_runs,1000))
        val_stack = np.zeros(num_runs)

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

            if val is not None:
                if val == 'top': filename = 'val_matrix'
                elif val == 'cen': filename = 'val_matrix_cen'

                with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                    data = pickle.load(f)

                avg = np.mean(data)
                val_stack[r_num] = avg

                if Path(fr'{data_dir}/{name}/val_results_cen.txt').is_file():
                    with open(fr'{data_dir}/{name}/val_results_cen.txt') as f:
                        lines = f.readlines()

                        val_data = np.zeros((len(lines)-1, 3))
                        for n, line in enumerate(lines[1:]):
                            data = [item.strip() for item in line.split(' ')]
                            val_data[n,0] = data[1] # generation
                            val_data[n,1] = data[4] # train fitness
                            val_data[n,2] = data[7] # val fitness

                    avg_val = np.mean(val_data[:,2])
                    val_stack[r_num] = avg_val

                    # sort according to val fitness
                    top_ind = np.argsort(val_data[:,2])[0] 
                    top_gen = int(val_data[top_ind,0])
                    top_valfit = int(val_data[top_ind,2])
                    if top_valfit < 500:
                        print(f'{name}, gen {top_gen}: fit {top_valfit}')

        #top_indices = np.argsort(val_stack)[:10]
        #for i in top_indices:
        #    print(int(val_stack[i]), run_names[i])

        if group_est == 'mean':
            est_trend = np.mean(top_stack, axis=0)
        elif group_est == 'median':
            est_trend = np.median(top_stack, axis=0)
        la = ax1.plot(est_trend, 
                        # label = f'{group_name}: top individual (avg of {num_runs} runs)',
                        label = f'{group_name}',
                        color=cmap(g_num/cmap_range), 
                        alpha=0.4
                        )
        lns.append(la[0])

        # l2 = ax1.plot(np.mean(avg_stack, axis=0), 
        #                 label = f'{group_name}: population average (avg of {num_runs} runs)',
        #                 color=cmap(g_num/cmap_range), 
        #                 linestyle='dotted', 
        #                 alpha=0.5
        #                 )
        # # lns.append(l2[0])
    
        if val is not None:
            if group_est == 'mean':
                val_group = np.mean(val_stack, axis=0)
            elif group_est == 'median':
                val_group = np.median(val_stack, axis=0)
            print(f'{group_name} | avg val: {int(val_group)}')
            if val_group != 0:
                ax1.hlines(val_group, g_num*5, data_genxpop.shape[0] + g_num*5,
                        color=cmap(g_num/cmap_range),
                        linestyle='dashed',
                        alpha=0.5
                        )

    ax1.set_xlabel('Generation')

    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper right')

    ax1.set_ylabel('Performance')
    ax1.set_ylim(-20,1350)

    if save_name: 
        # plt.savefig(fr'{data_dir}/{save_name}.png')
        plt.savefig(fr'{data_dir}/{save_name}.png', dpi=100)
        # plt.savefig(fr'{data_dir}/{save_name}_500.png', dpi=500)
    plt.show()


def plot_mult_EA_trends_groups_endonly(groups, val=None, save_name=None):

    # establish load directory
    root_dir = Path(__file__).parent.parent
    data_dir = Path(root_dir, r'data/simulation_data')

    # # init plot details
    # fig, ax1 = plt.subplots(figsize=(15,10)) 
    fig, ax1 = plt.subplots(figsize=(6,4)) 
    cmap = plt.get_cmap('plasma')
    cmap_range = len(groups)
    violin_labs = []
    
    # iterate over each file
    for g_num, (group_name, run_names) in enumerate(groups):

        data_group = []        
        for r_num, name in enumerate(run_names):

            if val == 'top': filename = 'val_matrix'
            elif val == 'cen': filename = 'val_matrix_cen'

            with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                data = pickle.load(f)
            
            data_group.append(data.flatten())

            # print(data, data.shape)
            # mean_vals = np.mean(data, axis=1)
            # print(mean_vals, mean_vals.shape)
        
            # l0 = ax1.violinplot(data.flatten(), 
            #             positions=[r_num],
            #             widths=1, 
            #             showmeans=False, 
            #             showextrema=True,
            #             )
            # color = l0["bodies"][0].get_facecolor().flatten()
            # violin_labs.append((mpatches.Patch(color=color), name))
    
        data = np.array(data_group)
        l0 = ax1.violinplot(data.flatten(), 
                    positions=[g_num],
                    widths=1, 
                    showmedians=True, 
                    showextrema=False,
                    )
        for part in l0["bodies"]:
            part.set_edgecolor(cmap(g_num/cmap_range))
            part.set_facecolor(cmap(g_num/cmap_range))
        l0["cmedians"].set_edgecolor(cmap(g_num/cmap_range))
        color = l0["bodies"][0].get_facecolor().flatten()
        violin_labs.append((mpatches.Patch(color=color), group_name))
        # violin_labs.append((mpatches.Patch(color=color), labs[g_num]))

        print(f'{group_name}: {int(np.mean(data))}')

    ax1.legend(*zip(*violin_labs), loc='upper left')
    # labs = [group_name for group_name,_ in groups]

    # ax1.set_xticks(np.linspace(0,len(groups)-1,len(groups)))
    # labs = [1000, 10000, 20000, 30000, 40000]
    # ax1.set_xlabel('Starting Seed')
    # labs = [2,3,4,5,6,7]
    # ax1.set_xlabel('# CNN Outputs')
    # labs = [6,8,10,12,14,16,18,20,24,32]
    # ax1.set_xlabel('Visual Resolution')
    # labs = [1,2,4,16,'2x2','2x4','2x16']
    # ax1.set_xlabel('FNN Size')
    # labs = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.875]
    # ax1.set_xlabel('Field of Vision (% of total)')
    # labs = [1,0.8,0.6,0.5,0.4,0.3,0.2,0.1,0]
    # labs = [1,0.5,0.4,0.3,0.2,0.1,0]
    # labs = [1,0.8,0.5,0.2,0]
    # ax1.set_xlabel('Distance Scaling Factor')
    # ax1.set_xticklabels(labs)
    ax1.set_xticks([])
    ax1.set_ylabel('Time to Find Patch')
    # ax1.set_ylim(-20,1020)

    if save_name: 
        # plt.savefig(fr'{data_dir}/{save_name}.png')
        plt.savefig(fr'{data_dir}/{save_name}.png', dpi=100)
    plt.show()


def plot_mult_EA_trends_groups_endonly_perfect(groups, val=None, save_name=None):

    # establish load directory
    root_dir = Path(__file__).parent.parent
    data_dir = Path(root_dir, r'data/simulation_data')

    # # init plot details
    # fig, ax1 = plt.subplots(figsize=(15,10)) 
    fig, ax1 = plt.subplots(figsize=(6,4)) 
    cmap = plt.get_cmap('plasma')
    cmap_range = len(groups)+1
    # violin_labs = []

    with open(fr'{data_dir}/nonNN/perfect.bin','rb') as f:
        data = pickle.load(f)
    # print(f'average perfect performance: {np.mean(data.flatten())}')

    l0 = ax1.violinplot(data.flatten(), 
                positions=[0],
                widths=1, 
                showmedians=True, 
                showextrema=False,
                )
    for part in l0["bodies"]:
        part.set_edgecolor('k')
        part.set_facecolor('k')
    l0["cmedians"].set_edgecolor('k')
    color = l0["bodies"][0].get_facecolor().flatten()
    violin_labs = [(mpatches.Patch(color=color), 'perfect')]
    
    # iterate over each file
    for g_num, (group_name, run_names) in enumerate(groups):

        data_group = []        
        for r_num, name in enumerate(run_names):

            if val == 'top': filename = 'val_matrix'
            elif val == 'cen': filename = 'val_matrix_cen'

            with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                data = pickle.load(f)
            
            data_group.append(data.flatten())
    
        data = np.array(data_group)
        l0 = ax1.violinplot(data.flatten(), 
                    positions=[g_num+1],
                    widths=1, 
                    showmedians=True, 
                    showextrema=False,
                    )
        for part in l0["bodies"]:
            part.set_edgecolor(cmap((g_num)/cmap_range))
            part.set_facecolor(cmap((g_num)/cmap_range))
        l0["cmedians"].set_edgecolor(cmap((g_num)/cmap_range))
        color = l0["bodies"][0].get_facecolor().flatten()
        violin_labs.append((mpatches.Patch(color=color), group_name))

        print(f'{group_name}: {int(np.mean(data))}')

    # ax1.legend(*zip(*violin_labs), loc='upper left')
    # labs = [group_name for group_name,_ in groups]

    # ax1.xaxis.set_ticklabels([])
    ax1.set_xticks([])
    # labs = ['','perfect', 'WF * 0.5','WF * 0.4','WF * 0.3','WF * 0.2','WF * 0.1', 'no dist scaling']
    # ax1.set_xlabel('Field of Vision (% of total)')
    # ax1.set_xticklabels(labs)
    ax1.set_ylabel('Time to Find Patch')
    # ax1.set_ylim(-20,1020)

    if save_name: 
        # plt.savefig(fr'{data_dir}/{save_name}.png')
        plt.savefig(fr'{data_dir}/{save_name}_perfect.png', dpi=150)
    plt.show()


def plot_mult_EA_trends_valnoise(run_names, noise, val=None, save_name=None):

    # establish load directory
    root_dir = Path(__file__).parent.parent
    data_dir = Path(root_dir, r'data/simulation_data')

    if val == 'top': filename = 'val_matrix'
    elif val == 'cen': filename = 'val_matrix_cen'

    # init plot details
    fig, ax1 = plt.subplots(figsize=(15,10)) 
    cmap = plt.get_cmap('hsv')
    cmap_range = 20

    noise_name, noise_types = noise

    # iterate over each file
    for r_num, name in enumerate(run_names):

        for n_num, noise_type in enumerate(noise_types):

            if noise_type == 'no_noise':
                with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                    data = pickle.load(f)
            else:
                with open(fr'{data_dir}/{name}/{filename}_{noise_type}_noise.bin','rb') as f:
                    data = pickle.load(f)

            # data = data.flatten()
            # x = beeswarm(data)
            # # x = beeswarm2(data)
            # ax1.scatter(r_num + x, data, c=cmap(r_num/cmap_range), alpha=0.01)

            # print(r_num + n_num/len(noise_types))

            l0 = ax1.violinplot(data.flatten(), 
                        positions=[r_num + n_num/len(noise_types)*.8],
                        widths=1/len(noise_types)*.8, 
                        showmedians=True, 
                        showextrema=False,
                        )
            for p in l0['bodies']:
                p.set_facecolor(cmap(r_num/cmap_range))
                p.set_edgecolor('black')
            l0['cmedians'].set_edgecolor('black')

            # color = l0["bodies"][0].get_facecolor().flatten()
            # violin_labs.append((mpatches.Patch(color=color), group_name))

    title = '~'
    for x in noise_types:
        title += f' {x} ~'
    ax1.set_title(title)

    # plt.grid(axis = 'x')
    plt.xticks(np.arange(0, r_num+1, 1))
    ax1.xaxis.set_ticklabels([])
    ax1.set_ylabel('Time to Find Patch')

    if save_name: 
        plt.savefig(fr'{data_dir}/{save_name}_{noise_name}.png')
    plt.show()
    plt.close()


def beeswarm(y, nbins=None):
    """
    Returns x coordinates for the points in ``y``, so that plotting ``x`` and
    ``y`` results in a bee swarm plot.
    https://stackoverflow.com/questions/36153410/how-to-create-a-swarm-plot-with-matplotlib
    """
    y = np.asarray(y)
    if nbins is None:
        nbins = len(y) // 2
        if nbins == 0:
            nbins = 1

    # Get upper bounds of bins
    x = np.zeros(len(y))
    ylo = np.min(y)
    yhi = np.max(y)
    dy = (yhi - ylo) / nbins
    ybins = np.linspace(ylo + dy, yhi - dy, nbins - 1)

    # Divide indices into bins
    i = np.arange(len(y))
    ibs = [0] * nbins
    ybs = [0] * nbins
    nmax = 0
    for j, ybin in enumerate(ybins):
        f = y <= ybin
        ibs[j], ybs[j] = i[f], y[f]
        nmax = max(nmax, len(ibs[j]))
        f = ~f
        i, y = i[f], y[f]
    ibs[-1], ybs[-1] = i, y
    nmax = max(nmax, len(ibs[-1]))

    # Assign x indices
    dx = 1 / (nmax // 2)
    for i, y in zip(ibs, ybs):
        if len(i) > 1:
            j = len(i) % 2
            i = i[np.argsort(y)]
            a = i[j::2]
            b = i[j+1::2]
            x[a] = (0.5 + j / 3 + np.arange(len(b))) * dx
            x[b] = (0.5 + j / 3 + np.arange(len(b))) * -dx

    return x



def plot_mult_EA_trends_randomwalk(run_names, save_name=None):

    # establish load directory
    root_dir = Path(__file__).parent.parent
    data_dir = Path(root_dir, r'data/simulation_data/nonNN')

    # init plot details
    fig, ax1 = plt.subplots(figsize=(15,10)) 
    # cmap = plt.get_cmap('hsv')
    # cmap_range = len(run_names)
    # violin_labs = []

    diff_coeffs = [0.001,0.005,0.01,0.05,0.1,0.5]

    plt.xticks(np.arange(0, len(run_names)+2, 1))
    fig.canvas.draw()
    labels = [item.get_text() for item in ax1.get_xticklabels()]

    # iterate over each file
    for r_num, name in enumerate(run_names):

        with open(fr'{data_dir}/{name}.bin','rb') as f:
            data = pickle.load(f)

        # print(r_num + n_num/len(noise_types))

        l0 = ax1.violinplot(data.flatten(), 
                    positions=[r_num],
                    widths=1, 
                    showmeans=True, 
                    showextrema=False,
                    )
        # for p in l0['bodies']:
        #     p.set_facecolor(cmap(r_num/cmap_range))
        #     p.set_edgecolor('black')
        # l0['cmedians'].set_edgecolor('black')

        # color = l0["bodies"][0].get_facecolor().flatten()
        # violin_labs.append((mpatches.Patch(color=color), name))

        labels[r_num] = diff_coeffs[r_num]
    
    # also add perfect traj
    with open(fr'{data_dir}/perfect.bin','rb') as f:
        data = pickle.load(f)
    
    print(f'average perfect performance: {np.mean(data.flatten())}')

    l0 = ax1.violinplot(data.flatten(), 
                positions=[r_num+1],
                widths=1, 
                showmeans=True, 
                showextrema=False,
                )
    labels[r_num+1] = 'perfect'



    # ax1.legend(*zip(*violin_labs), loc='lower right')
    # plt.grid(axis = 'x')
    
    # ax1.xaxis.set_ticklabels([])
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('Time to Find Patch')
    ax1.set_xlabel('Rotational Diffusion Coefficient')

    plt.savefig(fr'{data_dir}/random.png')
    plt.show()
    plt.close()


def plot_LM_percep(lm_radius, vis_res, FOV, x_max=1000, y_max=1000, w=8, h=8, save_name=None):

    fig, axes = plt.subplots() 
    axes.set_xlim(0, x_max)
    axes.set_ylim(0, y_max)

    # rescale plotting area to square
    l,r,t,b = fig.subplotpars.left, fig.subplotpars.right, fig.subplotpars.top, fig.subplotpars.bottom
    fig.set_size_inches( float(w)/(r-l) , float(h)/(t-b) )

    # resource patch via circle
    pos_x = 400
    pos_y = 600
    radius = 50
    axes.add_patch( plt.Circle((pos_x, pos_y), radius, edgecolor='k', fill=False, zorder=1) )
    
    # landmarks via circles
    pts = [
        (0,0),
        (0,y_max),
        (x_max,0),
        (x_max,y_max),
    ]
    for pos in pts:
        axes.add_patch( plt.Circle(pos, lm_radius, edgecolor='k', fill=True, color='gray', zorder=1) )

    deg_bw_rays = 360 * FOV / vis_res
    rays = np.arange(1, vis_res+1)
    dists = lm_radius / np.tan(deg_bw_rays/2 * np.pi/180 * rays)

    cm = plt.get_cmap('plasma_r')
    cm_disc = cm(np.linspace(.1, .8, len(rays), endpoint=False))

    for i,d in enumerate(dists):
        for pos in pts:
            axes.add_patch( plt.Circle(pos, lm_radius + d, edgecolor=cm_disc[i], fill=False, zorder=1) )

    if save_name:
        plt.savefig(fr'{save_name}.png')
        plt.show()
        plt.close()
    else:
        plt.show()


def relative_occurence_stacked_bars(dpi):

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'

    # increase hatch linewidth
    mpl.rcParams['hatch.linewidth'] = 3

    ### visual resolution ###
        
    category_by_runtype = {
        'BD': (8,7,14,13,9,14,13,20,18,15),
        'BD/IS': (2,7,6,9,10,5,8,5,7,4),
        'IS': (19,19,15,13,14,8,7,3,6,2),
    }
    runtype_by_category = {
        '6': (8,2,19),
        '8': (7,7,19),
        '10': (14,6,15),
        '12': (13,9,13),
        '14': (9,10,14),
        '16': (14,5,8),
        '18': (13,8,7),
        '20': (20,5,3),
        '24': (18,7,6),
        '32': (15,4,2),
    }
    category_colors = [
        ['cornflowerblue'],
        ['tomato','cornflowerblue'],
        ['tomato'],
    ]

    fig,ax = plt.subplots(figsize=(5,4))

    run_types = [x for x,_ in runtype_by_category.items()]
    categories = [x for x,_ in category_by_runtype.items()]
    run_sums = [np.sum(np.array(runtype_count)) for _,runtype_count in runtype_by_category.items()]

    bottom = np.zeros(len(run_types))
    for i, (category, category_count) in enumerate(category_by_runtype.items()):

        category_count = np.array(category_count)
        category_count_normalized = category_count / run_sums

        colors = category_colors[i]

        if len(colors) == 1:
            ax.bar(run_types, category_count_normalized, label=category, bottom=bottom, facecolor=colors[0], edgecolor=colors[0])
            bottom += category_count_normalized
        if len(colors) == 2:
            ax.bar(run_types, category_count_normalized, label=category, bottom=bottom, facecolor=colors[0], edgecolor=colors[1], hatch=r'\\')
            bottom += category_count_normalized

    ax.set_ylabel('Relative Occurence')
    var = r'$\upsilon$'
    ax.set_xlabel(f'Visual Resolution ({var})')
    ax.set_ylim(0,1.05)
    ax.legend(loc='upper left', reverse=True)
    # ax.legend(loc=(.85,.7))

    plt.tight_layout()
    plt.savefig(fr'{data_dir}/relative_occurence_vis_{dpi}.png', dpi=dpi)
    plt.show()
    # plt.close()


    ### distance scaling ###

    category_by_runtype = {
        'BD': (7,6,4,5,3,0,1,0,0),
        'BD/IS': (7,6,3,0,1,1,0,0,0),
        'IS': (19,19,17,24,14,6,2,0,0),
        'IS/DP': (0,0,1,5,9,9,5,0,0),
        'DP/BD': (0,0,0,0,4,7,5,3,2),
        'DP': (0,0,0,0,7,16,27,36,37),
    }
    runtype_by_category = {
        '0': (7,7,19,0,0,0),
        '0.1': (6,6,19,0,0,0),
        '0.2': (4,3,17,1,0,0),
        '0.3': (5,0,24,5,0,0),
        '0.4': (3,1,14,9,4,7),
        '0.5': (0,1,6,9,7,16),
        '0.6': (1,0,2,5,5,27),
        '0.8': (0,0,0,0,3,36),
        '1': (0,0,0,0,2,37),
    }
    category_colors = [
        ['cornflowerblue'],
        ['tomato','cornflowerblue'],
        ['tomato'],
        ['tomato','forestgreen'],
        ['cornflowerblue','forestgreen'],
        ['forestgreen'],
    ]

    fig,ax = plt.subplots(figsize=(5,4))

    run_types = [x for x,_ in runtype_by_category.items()]
    categories = [x for x,_ in category_by_runtype.items()]
    run_sums = [np.sum(np.array(runtype_count)) for _,runtype_count in runtype_by_category.items()]

    bottom = np.zeros(len(run_types))
    for i, (category, category_count) in enumerate(category_by_runtype.items()):

        category_count = np.array(category_count)
        category_count_normalized = category_count / run_sums
        
        colors = category_colors[i]

        if len(colors) == 1:
            ax.bar(run_types, category_count_normalized, label=category, bottom=bottom, facecolor=colors[0], edgecolor=colors[0])
            bottom += category_count_normalized
        if len(colors) == 2:
            ax.bar(run_types, category_count_normalized, label=category, bottom=bottom, facecolor=colors[0], edgecolor=colors[1], hatch=r'\\')
            bottom += category_count_normalized

    ax.set_ylabel('Relative Occurence')
    var = r'$\sigma$'
    ax.set_xlabel(f'Distance Scaling ({var})')
    ax.set_ylim(0,1.05)
    plt.legend(loc='upper right', reverse=True)
    # ax.legend(loc=(.74,.55))

    plt.tight_layout()
    plt.savefig(fr'{data_dir}/relative_occurence_dist_{dpi}.png', dpi=dpi)
    plt.show()
    # plt.close()


    ### other ###

    category_by_runtype = {
        'BD': (7,6,8,2,5,6,1,1,9,5),
        'BD/IS': (7,5,1,3,1,4,5,4,1,5),
        'IS': (19,14,6,10,8,6,13,14,6,5),
        # 'IS/DP': (0,0,0,0,0,0,0,0,0,0),
        # 'DP/BD': (0,0,0,0,0,0,0,0,0,0),
        # 'DP': (0,0,0,0,0,0,0,0,0,0),
    }
    runtype_by_category = {
        # 'Original': (7,7,19,0,0,0),
        # 'act=pi/4': (6,5,14,0,0,0),
        # 'CNN13': (8,1,6,0,0,0),
        # 'CNN15': (2,3,10,0,0,0),
        # 'CNN16': (5,1,8,0,0,0),
        # 'CNN17': (6,4,6,0,0,0),
        # 'FNN16': (1,5,13,0,0,0),
        # 'FNN2x16': (1,4,14,0,0,0),
        # 'FOV35': (9,1,6,0,0,0),
        # 'FOV45': (5,5,5,0,0,0),
        'Original': (7,7,19),
        'act=pi/4': (6,5,14),
        'CNN13': (8,1,6),
        'CNN15': (2,3,10),
        'CNN16': (5,1,8),
        'CNN17': (6,4,6),
        'FNN16': (1,5,13),
        'FNN2x16': (1,4,14),
        'FOV35': (9,1,6),
        'FOV45': (5,5,5),
    }
    category_colors = [
        ['cornflowerblue'],
        ['tomato','cornflowerblue'],
        ['tomato'],
        # ['tomato','forestgreen'],
        # ['cornflowerblue','forestgreen'],
        # ['forestgreen'],
    ]

    fig,ax = plt.subplots(figsize=(10,8))

    run_types = [x for x,_ in runtype_by_category.items()]
    categories = [x for x,_ in category_by_runtype.items()]
    run_sums = [np.sum(np.array(runtype_count)) for _,runtype_count in runtype_by_category.items()]

    bottom = np.zeros(len(run_types))
    for i, (category, category_count) in enumerate(category_by_runtype.items()):

        category_count = np.array(category_count)
        category_count_normalized = category_count / run_sums
        
        colors = category_colors[i]

        if len(colors) == 1:
            ax.bar(run_types, category_count_normalized, label=category, bottom=bottom, facecolor=colors[0], edgecolor=colors[0])
            bottom += category_count_normalized
        if len(colors) == 2:
            ax.bar(run_types, category_count_normalized, label=category, bottom=bottom, facecolor=colors[0], edgecolor=colors[1], hatch=r'\\')
            bottom += category_count_normalized

    ax.set_ylabel('Relative Occurence')
    # ax.set_xlabel('Distance Scaling')
    ax.set_ylim(0,1.05)
    plt.legend(loc='upper right', reverse=True)
    # ax.legend(loc=(.74,.55))

    plt.tight_layout()
    plt.savefig(fr'{data_dir}/relative_occurence_other_{dpi}.png', dpi=dpi)
    plt.show()
    # plt.close()


def plot_hsv_dir(w=8, h=8, dpi=50):

    fig, axes = plt.subplots() 
    axes.set_xlim(-1.5, 1.5)
    axes.set_ylim(-1.5, 1.5)

    # rescale plotting area to square
    l,r,t,b = fig.subplotpars.left, fig.subplotpars.right, fig.subplotpars.top, fig.subplotpars.bottom
    fig.set_size_inches( float(w)/(r-l) , float(h)/(t-b) )

    num_angs = 8
    angs = np.linspace(0, 2*np.pi, num_angs+1)

    pts = np.zeros([num_angs,3])
    x,y = 0,0
    for i,a in enumerate(angs[:-1]):
        pts[i,0] = x + np.cos(a)
        pts[i,1] = y + np.sin(a)
        pts[i,2] = a

    pos_x = pts[:,0]
    pos_y = pts[:,1]
    ori = pts[:,2]

    norm = mpl.colors.Normalize(vmin=0, vmax=2*np.pi)
    axes.scatter(pos_x, pos_y, c=ori, cmap='hsv', norm=norm, s=5000, alpha=0.5)
    axes.quiver(pos_x, pos_y, np.cos(ori), np.sin(ori), pivot='mid')

    plt.savefig(fr'hsv_dir_{dpi}.png', dpi=dpi)
    # plt.show()


if __name__ == '__main__':

    # plot_hsv_dir(dpi=100)
    
    # plot_LM_percep(lm_radius=100, vis_res=8, FOV=.4, save_name='landmarks_vis8_lm100')
    # plot_LM_percep(lm_radius=100, vis_res=10, FOV=.4, save_name='landmarks_vis10_lm100')
    # plot_LM_percep(lm_radius=100, vis_res=12, FOV=.4, save_name='landmarks_vis12_lm100')
    # plot_LM_percep(lm_radius=100, vis_res=16, FOV=.4, save_name='landmarks_vis16_lm100')

    # relative_occurence_stacked_bars(dpi=100)


### ----------pop runs----------- ###

    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='sc_CNN14_FNN2_vis6_PGPE_ss20_mom8_p50e20_valcen') 
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='sc_CNN14_FNN2_vis10_PGPE_ss20_mom8_p50e20_valcen') 
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='sc_CNN14_FNN2_vis12_PGPE_ss20_mom8_p50e20_valcen') 
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='sc_CNN14_FNN2_vis14_PGPE_ss20_mom8_p50e20_valcen') 
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='sc_CNN14_FNN2_vis16_PGPE_ss20_mom8_p50e20_valcen') 
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis24_PGPE_ss20_mom8_p50e20_valcen')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis32_PGPE_ss20_mom8_p50e20_valcen')

    # plot_mult_EA_trends([f'sc_CNN12_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='sc_CNN12_FNN2_vis8_PGPE_ss20_mom8_p50e20_valcen')
    # plot_mult_EA_trends([f'sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='sc_CNN13_FNN2_vis8_PGPE_ss20_mom8_p50e20_valcen')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='sc_CNN14_FNN2_vis8_PGPE_ss20_mom8_p50e20_valcen')
    # plot_mult_EA_trends([f'sc_CNN15_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='sc_CNN15_FNN2_vis8_PGPE_ss20_mom8_p50e20_valcen')
    # plot_mult_EA_trends([f'sc_CNN16_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='sc_CNN16_FNN2_vis8_PGPE_ss20_mom8_p50e20_valcen')
    # plot_mult_EA_trends([f'sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='sc_CNN17_FNN2_vis8_PGPE_ss20_mom8_p50e20_valcen')
    
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov2_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='sc_CNN14_FNN2_vis8_PGPE_ss20_mom8_fov2_p50e20_valcen')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov3_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='sc_CNN14_FNN2_vis8_PGPE_ss20_mom8_fov3_p50e20_valcen')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='sc_CNN14_FNN2_vis8_PGPE_ss20_mom8_fov35_p50e20_valcen')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov45_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='sc_CNN14_FNN2_vis8_PGPE_ss20_mom8_fov45_p50e20_valcen')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov5_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='sc_CNN14_FNN2_vis8_PGPE_ss20_mom8_fov5_p50e20_valcen')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov6_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='sc_CNN14_FNN2_vis8_PGPE_ss20_mom8_fov6_p50e20_valcen')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov7_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='sc_CNN14_FNN2_vis8_PGPE_ss20_mom8_fov7_p50e20_valcen')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov8_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='sc_CNN14_FNN2_vis8_PGPE_ss20_mom8_fov8_p50e20_valcen')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov875_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='sc_CNN14_FNN2_vis8_PGPE_ss20_mom8_fov875_p50e20_valcen')

    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_minmax_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_dist_minmax_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_far_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_dist_far_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_far_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN1124_FNN2_vis8_dist_far_PGPE_ss20_mom8_p50e20')

    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_dist_WF_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_n1_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_dist_WF_n1_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_n2_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_dist_WF_n2_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_n3_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_dist_WF_n3_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_n4_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_dist_WF_n4_PGPE_ss20_mom8_p50e20')

    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_dist_mlWF_n0_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_dist_mWF_n0_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_dist_msWF_n0_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_dist_sWF_n0_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n1_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_dist_sWF_n1_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep{x}' for x in range(20)], 
    #                     save_name='sc_CNN14_FNN2_vis8_dist_ssWF_n0_PGPE_ss20_mom8_p50e20')

    
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_angl_n05_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_angl_n05_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_angl_n10_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_angl_n10_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_angl_n15_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_angl_n15_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_angl_n20_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_angl_n20_PGPE_ss20_mom8_p50e20')

    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_act_n05_rep{x}' for x in range(20)], 
    #                     save_name='sc_CNN14_FNN2_vis8_act_n05_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_act_n10_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_act_n10_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_act_n15_rep{x}' for x in range(20)], 
    #                     save_name='sc_CNN14_FNN2_vis8_act_n15_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_act_n20_rep{x}' for x in range(20)],
    #                     save_name='sc_CNN14_FNN2_vis8_act_n20_PGPE_ss20_mom8_p50e20')

    
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_seed10k_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed20k_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_seed20k_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed30k_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_seed30k_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed40k_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_seed40k_PGPE_ss20_mom8_p50e20')


    # plot_mult_EA_trends([f'sc_lm_CNN14_FNN2_p50e20_vis8_lm100_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_lm_CNN14_FNN2_p50e20_vis8_lm100')
    # plot_mult_EA_trends([f'sc_lm_CNN14_FNN2_p50e20_vis10_lm100_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_lm_CNN14_FNN2_p50e20_vis10_lm100')
    # plot_mult_EA_trends([f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_lm_CNN14_FNN2_p50e20_vis12_lm100')
    # plot_mult_EA_trends([f'sc_lm_CNN14_FNN2_p50e20_vis16_lm100_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_lm_CNN14_FNN2_p50e20_vis16_lm100')
    # plot_mult_EA_trends([f'sc_lm_CNN14_FNN2_p50e20_vis24_lm100_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_lm_CNN14_FNN2_p50e20_vis24_lm100')
    # plot_mult_EA_trends([f'sc_lm_CNN14_FNN2_p50e20_vis32_lm100_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_lm_CNN14_FNN2_p50e20_vis32_lm100')
    # plot_mult_EA_trends([f'sc_lm_CNN14_FNN2_p50e20_vis64_lm100_rep{x}' for x in range(20)], 
    #                     save_name='sc_lm_CNN14_FNN2_p50e20_vis64_lm100')

    # plot_mult_EA_trends([f'sc_lm_CNN14_FNN2_p50e20_vis8_lm100_angl_n10_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_lm_CNN14_FNN2_p50e20_vis8_lm100_angl_n10')
    # plot_mult_EA_trends([f'sc_lm_CNN14_FNN2_p50e20_vis10_lm100_angl_n10_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_lm_CNN14_FNN2_p50e20_vis10_lm100_angl_n10')

    # plot_mult_EA_trends([f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_angl_n05_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_lm_CNN14_FNN2_p50e20_vis12_lm100_angl_n05')
    # plot_mult_EA_trends([f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_angl_n10_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_lm_CNN14_FNN2_p50e20_vis12_lm100_angl_n10')
    # plot_mult_EA_trends([f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_angl_n25_rep{x}' for x in range(20)], 
    #                     save_name='sc_lm_CNN14_FNN2_p50e20_vis12_lm100_angl_n25')
    # plot_mult_EA_trends([f'sc_lm_CNN14_FNN2_p50e20_vis12_selfangl_n50_rep{x}' for x in range(20)], 
    #                     save_name='sc_lm_CNN14_FNN2_p50e20_vis12_selfangl_n50')

    # plot_mult_EA_trends([f'sc_lm_CNN14_FNNn8_p50e20_vis8_lm100_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_lm_CNN14_FNNn8_p50e20_vis8_lm100')
    # plot_mult_EA_trends([f'sc_lm_CNN14_FNNn8_p50e20_vis10_lm100_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_lm_CNN14_FNNn8_p50e20_vis10_lm100')
    # plot_mult_EA_trends([f'sc_lm_CNN14_FNNn8_p50e20_vis12_lm100_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_lm_CNN14_FNNn8_p50e20_vis12_lm100')
    
    # # # plot_mult_EA_trends([f'sc_lm_CNN14_FNN2_p50e20_vis8_lm300_rep{x}' for x in range(20)], val='cen',
    # # #                     save_name='sc_lm_CNN14_FNN2_p50e20_vis8_lm300')

    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)], val='cen',)
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep{x}' for x in range(20)], val='cen',)

    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)], val='cen',)
    # plot_mult_EA_trends([f'sc_CNN24_FNN2_p50e20_vis8_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)], val='cen',)
    # plot_mult_EA_trends([f'sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)], val='cen',)
    # plot_mult_EA_trends([f'sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)], val='cen',)
    # plot_mult_EA_trends([f'sc_CNN27_FNN16_p50e20_vis8_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)], val='cen',)

    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)], val='cen',)
    # plot_mult_EA_trends([f'sc_CNN24_FNN2_p50e20_vis32_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)], val='cen',)
    # plot_mult_EA_trends([f'sc_CNN17_FNN2_p50e20_vis32_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)], val='cen',)
    # plot_mult_EA_trends([f'sc_CNN14_FNN16_p50e20_vis32_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)], val='cen',)
    # plot_mult_EA_trends([f'sc_CNN27_FNN16_p50e20_vis32_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)], val='cen',)
    plot_mult_EA_trends([f'sc_CNN1148_FNN2_p50e20_vis32_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)], val='cen',)

    # names = []
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}')
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep{x}')
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed20k_rep{x}')
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed30k_rep{x}')
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed40k_rep{x}')
    # plot_mult_EA_trends_new(names, val='cen', group_est='median', save_name='sc_CNN14_FNN2_vis8_PGPE_ss20_mom8_p50e20')

    # names = []
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep{x}')
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed10k_rep{x}')
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed20k_rep{x}')
    # for x in range(16):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed30k_rep{x}')
    # # for x in range(20):
    # #     names.append(f'sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed40k_rep{x}')
    # plot_mult_EA_trends_new(names, group_est='median', save_name='sc_CNN14_FNN2_vis24_PGPE_ss20_mom8_p50e20')


    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_n0_rep{x}')
    # plot_mult_EA_trends_new(names, val='cen', group_est='median', save_name='sc_CNN14_FNN2_vis8_PGPE_ss20_mom8_p50e20_dist_WF')

    # names = []
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep{x}')
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep{x}')
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed20k_rep{x}')
    # for x in range(10):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed30k_rep{x}')
    # # for x in range(20):
    # #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed40k_rep{x}')
    # plot_mult_EA_trends_new(names, group_est='median', save_name='sc_CNN14_FNN2_vis8_PGPE_ss20_mom8_p50e20_dist_sWF')

    # names = []
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep{x}')
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep{x}')
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed20k_rep{x}')
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed30k_rep{x}')
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed40k_rep{x}')
    # plot_mult_EA_trends(names, val='cen', group_est='median', save_name='sc_CNN14_FNN2_vis8_PGPE_ss20_mom8_p50e20_dist_mlWF')
    # names = []
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep{x}')
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep{x}')
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed20k_rep{x}')
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed30k_rep{x}')
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed40k_rep{x}')
    # plot_mult_EA_trends(names, val='cen', group_est='median', save_name='sc_CNN14_FNN2_vis8_PGPE_ss20_mom8_p50e20_dist_sWF')
    # names = []
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep{x}')
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed10k_rep{x}')
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed20k_rep{x}')
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed30k_rep{x}')
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed40k_rep{x}')
    # plot_mult_EA_trends(names, val='cen', group_est='median', save_name='sc_CNN14_FNN2_vis24_PGPE_ss20_mom8_p50e20')
    # names = []
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep{x}')
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed10k_rep{x}')
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed20k_rep{x}')
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed30k_rep{x}')
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed40k_rep{x}')
    # plot_mult_EA_trends(names, val='cen', group_est='median', save_name='sc_CNN14_FNN2_vis12_PGPE_ss20_mom8_p50e20')
    # names = []
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep{x}')
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed10k_rep{x}')
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed20k_rep{x}')
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed30k_rep{x}')
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed40k_rep{x}')
    # plot_mult_EA_trends(names, val='cen', group_est='median', save_name='sc_CNN14_FNN2_vis12_PGPE_ss20_mom8_p50e20')


    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_PGPE_ss20_mom8_p50e20_dist_maxWF')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep{x}' for x in range(20)], 
    #                     save_name='sc_CNN14_FNN2_vis8_PGPE_ss20_mom8_p50e20_dist_p9WF')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep{x}' for x in range(20)], 
    #                     save_name='sc_CNN14_FNN2_vis8_PGPE_ss20_mom8_p50e20_dist_p8WF')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_PGPE_ss20_mom8_p50e20_dist_mlF')



### ----------group pop runs----------- ###

    # ### SEED ###
    # groups = []
    # groups.append(('seed 1k', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('seed 10k', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep{x}' for x in range(20)]))
    # groups.append(('seed 20k', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed20k_rep{x}' for x in range(20)]))
    # groups.append(('seed 30k', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed30k_rep{x}' for x in range(20)]))
    # groups.append(('seed 40k', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed40k_rep{x}' for x in range(20)]))
    # # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_singlecorner_seed')
    # plot_mult_EA_trends_groups_endonly(groups, val='cen', save_name='groups_endonly_singlecorner_seed')


    # ### FEAT ###
    # groups = []
    # # groups.append(('CNN 1122', [f'sc_CNN1122_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(15)]))
    # # groups.append(('CNN 1124', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(53)]))
    # groups.append(('CNN 2', [f'sc_CNN12_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('CNN 3', [f'sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('CNN 4', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('CNN 5', [f'sc_CNN15_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('CNN 6', [f'sc_CNN16_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('CNN 7', [f'sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_singlecorner_CNN')
    # plot_mult_EA_trends_groups_endonly(groups, val='cen', save_name='groups_endonly_singlecorner_CNN')


    # ## VIS ###
    # groups = []
    # groups.append(('vis 6', [f'sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('vis 8', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('vis 10', [f'sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('vis 12', [f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('vis 14', [f'sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('vis 16', [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('vis 18', [f'sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('vis 20', [f'sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('vis 24', [f'sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('vis 32', [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # # plot_mult_EA_trends_groups(groups, val='cen', group_est='median', save_name='groups_singlecorner_vis')
    # plot_mult_EA_trends_groups_endonly(groups, val='cen', save_name='groups_endonly_singlecorner_vis')

    groups = []
    names = []
    for x in range(20):
        names.append(f'sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep{x}')
        names.append(f'sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_seed10k_rep{x}')
    groups.append(('vis 6', names))
    names = []
    for x in range(20):
        names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}')
        names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep{x}')
    groups.append(('vis 8', names))
    names = []
    for x in range(20):
        names.append(f'sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep{x}')
        names.append(f'sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_seed10k_rep{x}')
    groups.append(('vis 10', names))
    names = []
    for x in range(20):
        names.append(f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep{x}')
        names.append(f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed10k_rep{x}')
    groups.append(('vis 12', names))
    names = []
    for x in range(20):
        names.append(f'sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep{x}')
        names.append(f'sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_seed10k_rep{x}')
    groups.append(('vis 14', names))
    names = []
    for x in range(20):
        names.append(f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep{x}')
        names.append(f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_seed10k_rep{x}')
    groups.append(('vis 16', names))
    names = []
    for x in range(20):
        names.append(f'sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_rep{x}')
        names.append(f'sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_seed10k_rep{x}')
    groups.append(('vis 18', names))
    names = []
    for x in range(20):
        names.append(f'sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_rep{x}')
        names.append(f'sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_seed10k_rep{x}')
    groups.append(('vis 20', names))
    names = []
    for x in range(20):
        names.append(f'sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep{x}')
        names.append(f'sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed10k_rep{x}')
    groups.append(('vis 24', names))
    names = []
    for x in range(20):
        names.append(f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep{x}')
        names.append(f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_seed10k_rep{x}')
    groups.append(('vis 32', names))

    # plot_mult_EA_trends_groups(groups, val='cen', group_est='median', save_name='groups_singlecorner_vis_s10')
    # plot_mult_EA_trends_groups_endonly(groups, val='cen', save_name='groups_endonly_singlecorner_vis_s10')

    # groups = []
    # groups.append(('vis 6', [f'sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('vis 8', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('vis 10', [f'sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('vis 12', [f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('vis 14', [f'sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('vis 16', [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('vis 18', [f'sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('vis 20', [f'sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('vis 24', [f'sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('vis 32', [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('vis 64', [f'sc_CNN14_FNN2_p50e20_vis64_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('vis 128', [f'sc_CNN14_FNN2_p50e20_vis128_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # plot_mult_EA_trends_groups_endonly(groups, val='cen', save_name='groups_endonly_singlecorner_vis_extra')

    # groups = []
    # groups.append(('vis 128, CNN14', [f'sc_CNN14_FNN2_p50e20_vis128_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('vis 128, CNN24', [f'sc_CNN24_FNN2_p50e20_vis128_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('vis 128, CNN34', [f'sc_CNN34_FNN2_p50e20_vis128_PGPE_ss20_mom8_rep{x}' for x in range(19)]))
    # groups.append(('vis 128, CNN1124', [f'sc_CNN1124_FNN2_p50e20_vis128_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # plot_mult_EA_trends_groups_endonly(groups, val='cen', save_name='groups_endonly_singlecorner_vis_128')

    # ### FNN SIZE ###
    groups = []
    groups.append(('FNN 1', [f'sc_CNN14_FNN1_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    groups.append(('FNN 2', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('FNN 3', [f'sc_CNN14_FNN3_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    groups.append(('FNN 4', [f'sc_CNN14_FNN4_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('FNN 8', [f'sc_CNN14_FNN8_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    groups.append(('FNN 16', [f'sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    groups.append(('FNN 2x2', [f'sc_CNN14_FNN2x2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('FNN 2x3', [f'sc_CNN14_FNN2x3_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    groups.append(('FNN 2x4', [f'sc_CNN14_FNN2x4_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('FNN 2x8', [f'sc_CNN14_FNN2x8_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    groups.append(('FNN 2x16', [f'sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_singlecorner_integrator_size')
    # plot_mult_EA_trends_groups_endonly(groups, val='cen', save_name='groups_endonly_singlecorner_integrator_size')


    # ### FOV ###
    # groups = []
    # groups.append(('fov2', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov2_rep{x}' for x in range(20)]))
    # groups.append(('fov3', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov3_rep{x}' for x in range(20)]))
    # # groups.append(('fov35', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep{x}' for x in range(20)]))
    # groups.append(('fov4', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # # groups.append(('fov45', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov45_rep{x}' for x in range(20)]))
    # groups.append(('fov5', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov5_rep{x}' for x in range(20)]))
    # groups.append(('fov6', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov6_rep{x}' for x in range(20)]))
    # groups.append(('fov7', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov7_rep{x}' for x in range(20)]))
    # groups.append(('fov8', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov8_rep{x}' for x in range(20)]))
    # groups.append(('fov875', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov875_rep{x}' for x in range(20)]))
    # # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_singlecorner_FOV')
    # plot_mult_EA_trends_groups_endonly(groups, val='cen', save_name='groups_endonly_singlecorner_FOV')


    ### OTHER ###
    # groups = []
    # groups.append(('FNN 2, relu', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('FNN 2, silu', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_silu_rep{x}' for x in range(20)]))
    # groups.append(('GRU 8, relu', [f'sc_CNN14_FNN8_p50e20_vis8_PGPE_ss20_mom8_gru_rep{x}' for x in range(20)]))
    # groups.append(('FNN 16x16, act disc 8', [f'sc_CNN14_FNN16x16_p50e20_vis8_PGPE_ss20_mom8_act8_rep{x}' for x in range(20)]))
    # groups.append(('FNN 64x64, act disc 32', [f'sc_CNN14_FNN64x64_p50e20_vis8_PGPE_ss20_mom8_act32_rep{x}' for x in range(20)]))
    # # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_singlecorner_other')
    # plot_mult_EA_trends_groups_endonly(groups, val='cen', save_name='groups_endonly_singlecorner_other')


    ### DIST ###
    # groups = []
    # # groups.append(('min-max', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_minmax_rep{x}' for x in range(20)]))

    # # groups.append(('WF ~ {0.00, 1.00}', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep{x}' for x in range(20)]))
    # # groups.append(('WF ~ {0.10, 0.90}', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep{x}' for x in range(20)]))
    # # groups.append(('WF ~ {0.20, 0.80}', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep{x}' for x in range(20)]))
    # # # groups.append(('WF ~ {0.20, 0.90}', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_n0_rep{x}' for x in range(20)]))
    # # groups.append(('WF ~ {0.25, 0.75}', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep{x}' for x in range(20)]))
    # # groups.append(('WF ~ {0.30, 0.70}', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep{x}' for x in range(20)]))
    # # groups.append(('WF ~ {0.45, 0.65}', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep{x}' for x in range(20)]))
    # # groups.append(('WF ~ {0.40, 0.60}', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep{x}' for x in range(20)]))
    # # groups.append(('WF scaling ~ {0.45, 0.55}', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep{x}' for x in range(20)]))

    # groups.append(('WF', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep{x}' for x in range(20)]))
    # # groups.append(('WF * 0.8', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep{x}' for x in range(20)]))
    # # groups.append(('WF * 0.6', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep{x}' for x in range(20)]))
    # groups.append(('WF * 0.5', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep{x}' for x in range(20)]))
    # groups.append(('WF * 0.4', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep{x}' for x in range(20)]))
    # groups.append(('WF * 0.3', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep{x}' for x in range(20)]))
    # groups.append(('WF * 0.2', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep{x}' for x in range(20)]))
    # groups.append(('WF * 0.1', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep{x}' for x in range(20)]))

    # # groups.append(('no distance scaling', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('no distance scaling', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # # groups.append(('no distance scaling', names))
    # # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_singlecorner_WF_scaling')
    # # plot_mult_EA_trends_groups_endonly(groups, val='cen', save_name='groups_endonly_singlecorner_WF_scaling')
    # plot_mult_EA_trends_groups_endonly_perfect(groups, val='cen', save_name='groups_endonly_singlecorner_WF_scaling')

    groups = []
    names = []
    for x in range(20):
        names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep{x}')
        names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep{x}')
    groups.append((r'$\sigma = 1$', names))
    # names = []
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep{x}')
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_seed10k_rep{x}')
    # groups.append(('WF * 0.8', names))
    # names = []
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep{x}')
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_seed10k_rep{x}')
    # groups.append(('WF * 0.6', names))
    names = []
    for x in range(20):
        names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep{x}')
        names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep{x}')
    groups.append((r'$\sigma = 0.5$', names))
    names = []
    for x in range(20):
        names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep{x}')
        names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep{x}')
    groups.append((r'$\sigma = 0.4$', names))
    names = []
    for x in range(20):
        names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep{x}')
        names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep{x}')
    groups.append((r'$\sigma = 0.3$', names))
    names = []
    for x in range(20):
        names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep{x}')
        names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep{x}')
    groups.append((r'$\sigma = 0.2$', names))
    names = []
    for x in range(20):
        names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep{x}')
        names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep{x}')
    groups.append((r'$\sigma = 0.1$', names))
    names = []
    for x in range(20):
        names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}')
        names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep{x}')
    groups.append((r'$\sigma = 0$', names))

    # plot_mult_EA_trends_groups(groups, val='cen', group_est='median', save_name='groups_singlecorner_WF_scaling_s10')
    # # plot_mult_EA_trends_groups_endonly(groups, val='cen', save_name='groups_endonly_singlecorner_WF_scaling_s10')
    # plot_mult_EA_trends_groups_endonly_perfect(groups, val='cen', save_name='groups_endonly_singlecorner_WF_scaling_s10')


    # groups = []
    # groups.append(('full WF, no noise', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_rep{x}' for x in range(20)]))
    # groups.append(('full WF, 0.1 std noise', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_n1_rep{x}' for x in range(20)]))
    # groups.append(('full WF, 0.2 std noise', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_n2_rep{x}' for x in range(20)]))
    # groups.append(('full WF, 0.3 std noise', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_n3_rep{x}' for x in range(20)]))
    # groups.append(('full WF, 0.4 std noise', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_n4_rep{x}' for x in range(20)]))
    # # groups.append(('sWF_n0', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep{x}' for x in range(20)]))
    # # groups.append(('sWF_n1', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n1_rep{x}' for x in range(20)]))
    # groups.append(('no dist scaling', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_singlecorner_WF_noise')
    # plot_mult_EA_trends_groups_endonly(groups, val='cen', save_name='groups_endonly_singlecorner_WF_noise')

    ### RES POS ###
    # groups = []
    # groups.append(('res55', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res55_rep{x}' for x in range(20)]))
    # groups.append(('res54', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res54_rep{x}' for x in range(20)]))
    # groups.append(('res53', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res53_rep{x}' for x in range(20)]))
    # groups.append(('res52', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res52_rep{x}' for x in range(20)]))
    # groups.append(('res51', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res51_rep{x}' for x in range(20)]))
    # # groups.append(('res50', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res50_rep{x}' for x in range(20)]))
    # groups.append(('res44', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('res43', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res43_rep{x}' for x in range(20)]))
    # groups.append(('res42', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res42_rep{x}' for x in range(20)]))
    # groups.append(('res41', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res41_rep{x}' for x in range(20)]))
    # # groups.append(('res40', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res40_rep{x}' for x in range(20)]))
    # groups.append(('res33', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res33_rep{x}' for x in range(20)]))
    # groups.append(('res32', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res32_rep{x}' for x in range(20)]))
    # groups.append(('res31', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res31_rep{x}' for x in range(20)]))
    # # groups.append(('res30', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res30_rep{x}' for x in range(20)]))
    # groups.append(('res22', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res22_rep{x}' for x in range(20)]))
    # groups.append(('res21', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res21_rep{x}' for x in range(20)]))
    # # groups.append(('res20', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res20_rep{x}' for x in range(20)]))
    # groups.append(('res11', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res11_rep{x}' for x in range(20)]))
    # # groups.append(('res10', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res10_rep{x}' for x in range(20)]))
    # groups.append(('res00', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res0_rep{x}' for x in range(20)]))
    # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_singlecorner_res_pos')

    # groups = []
    # groups.append(('res55', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res55_rep{x}' for x in range(20)]))
    # groups.append(('res44', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('res33', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res33_rep{x}' for x in range(20)]))
    # groups.append(('res22', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res22_rep{x}' for x in range(20)]))
    # groups.append(('res11', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res11_rep{x}' for x in range(20)]))
    # groups.append(('res00', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res0_rep{x}' for x in range(20)]))
    # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_singlecorner_res_pos_diag')

    # groups = []
    # groups.append(('res55', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res55_rep{x}' for x in range(20)]))
    # groups.append(('res54', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res54_rep{x}' for x in range(20)]))
    # groups.append(('res53', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res53_rep{x}' for x in range(20)]))
    # groups.append(('res52', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res52_rep{x}' for x in range(20)]))
    # groups.append(('res51', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res51_rep{x}' for x in range(20)]))
    # # groups.append(('res50', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res50_rep{x}' for x in range(20)]))
    # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_singlecorner_res_pos_5s')
    # groups = []
    # groups.append(('res44', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('res43', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res43_rep{x}' for x in range(20)]))
    # groups.append(('res42', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res42_rep{x}' for x in range(20)]))
    # groups.append(('res41', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res41_rep{x}' for x in range(20)]))
    # # groups.append(('res40', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res40_rep{x}' for x in range(20)]))
    # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_singlecorner_res_pos_4s')
    # groups = []
    # groups.append(('res33', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res33_rep{x}' for x in range(20)]))
    # groups.append(('res32', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res32_rep{x}' for x in range(20)]))
    # groups.append(('res31', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res31_rep{x}' for x in range(20)]))
    # # groups.append(('res30', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res30_rep{x}' for x in range(20)]))
    # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_singlecorner_res_pos_3s')
    # groups = []
    # groups.append(('res22', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res22_rep{x}' for x in range(20)]))
    # groups.append(('res21', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res21_rep{x}' for x in range(20)]))
    # # groups.append(('res20', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res20_rep{x}' for x in range(20)]))
    # groups.append(('res11', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res11_rep{x}' for x in range(20)]))
    # # groups.append(('res10', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res10_rep{x}' for x in range(20)]))
    # groups.append(('res00', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res0_rep{x}' for x in range(20)]))
    # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_singlecorner_res_pos_210s')


    ### NOISE ###
    # groups = []
    # groups.append(('no noise', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('angl n05', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_angl_n05_rep{x}' for x in range(20)]))
    # groups.append(('angl n10', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_angl_n10_rep{x}' for x in range(20)]))
    # groups.append(('angl n15', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_angl_n15_rep{x}' for x in range(20)]))
    # groups.append(('angl n20', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_angl_n20_rep{x}' for x in range(20)]))
    # # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_singlecorner_angl_noise')
    # plot_mult_EA_trends_groups_endonly(groups, val='cen', save_name='groups_singlecorner_angl_noise')

    # groups = []
    # groups.append(('no noise', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('act n05', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_act_n05_rep{x}' for x in range(20)]))
    # groups.append(('act n15', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_act_n15_rep{x}' for x in range(20)]))
    # groups.append(('act n20', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_act_n20_rep{x}' for x in range(20)]))
    # # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_singlecorner_angl_noise')
    # plot_mult_EA_trends_groups_endonly(groups, val='cen', save_name='groups_singlecorner_angl_noise')

    ## BOUNDARY_SCALE
    groups = []
    # groups.append(('VIS 8', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('VIS 8, BS 500', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_bound500_rep{x}' for x in range(20)]))
    # groups.append(('VIS 8, BS 1000', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)]))
    # groups.append(('VIS 8, maxWF', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep{x}' for x in range(20)]))
    # groups.append(('VIS 8, maxWF, BS 500', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_bound500_rep{x}' for x in range(20)]))
    # groups.append(('VIS 8, maxWF, BS 1000', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_bound1000_rep{x}' for x in range(20)]))
    # groups.append(('VIS 8, mlWF', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep{x}' for x in range(20)]))
    # groups.append(('VIS 8, mlWF, BS 500', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_bound500_rep{x}' for x in range(20)]))
    # groups.append(('VIS 8, mlWF, BS 1000', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_bound1000_rep{x}' for x in range(20)]))
    # groups.append(('VIS 32', [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('VIS 32, BS 500', [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_bound500_rep{x}' for x in range(20)]))
    groups.append(('VIS 32, BS 1000', [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)]))
    groups.append(('VIS 32, CNN24, BS 1000', [f'sc_CNN24_FNN2_p50e20_vis32_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(17)]))
    groups.append(('VIS 32, CNN17, BS 1000', [f'sc_CNN17_FNN2_p50e20_vis32_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)]))
    groups.append(('VIS 32, FNN16, BS 1000', [f'sc_CNN14_FNN16_p50e20_vis32_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)]))
    groups.append(('VIS 32, CNN27, FNN16, BS 1000', [f'sc_CNN27_FNN16_p50e20_vis32_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)]))
    groups.append(('VIS 32, CNN1148, FNN2, BS 1000', [f'sc_CNN1148_FNN2_p50e20_vis32_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)]))
    # groups.append(('VIS 8, CNN24, BS 1000', [f'sc_CNN24_FNN2_p50e20_vis8_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(17)]))
    # groups.append(('VIS 8, CNN17, BS 1000', [f'sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)]))
    # groups.append(('VIS 8, FNN16, BS 1000', [f'sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)]))
    # groups.append(('VIS 8, CNN27, FNN16, BS 1000', [f'sc_CNN27_FNN16_p50e20_vis8_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)]))
    # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_singlecorner_integrator_size')
    plot_mult_EA_trends_groups_endonly(groups, val='cen', save_name='groups_endonly_singlecorner_integrator_size')



    ### LANDMARKS ###
    # groups = []
    # groups.append(('resolution = 8', [f'sc_lm_CNN14_FNN2_p50e20_vis8_lm100_rep{x}' for x in range(20)]))
    # groups.append(('resolution = 10', [f'sc_lm_CNN14_FNN2_p50e20_vis10_lm100_rep{x}' for x in range(20)]))
    # groups.append(('resolution = 12', [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_rep{x}' for x in range(20)]))
    # groups.append(('resolution = 16', [f'sc_lm_CNN14_FNN2_p50e20_vis16_lm100_rep{x}' for x in range(20)]))
    # groups.append(('resolution = 24', [f'sc_lm_CNN14_FNN2_p50e20_vis24_lm100_rep{x}' for x in range(20)]))
    # groups.append(('resolution = 32', [f'sc_lm_CNN14_FNN2_p50e20_vis32_lm100_rep{x}' for x in range(20)]))
    # groups.append(('resolution = 64', [f'sc_lm_CNN14_FNN2_p50e20_vis64_lm100_rep{x}' for x in range(20)]))
    # # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_singlecorner_landmarks')
    # plot_mult_EA_trends_groups_endonly(groups, val='cen', save_name='groups_endonly_landmarks')

    # groups = []
    # groups.append(('vis12', [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_rep{x}' for x in range(20)]))
    # # groups.append(('vis12 + angl n05', [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_angl_n05_rep{x}' for x in range(20)]))
    # groups.append(('vis12 + angl n10', [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_angl_n10_rep{x}' for x in range(20)]))
    # groups.append(('vis12 + angl n25', [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_angl_n25_rep{x}' for x in range(20)]))
    # groups.append(('vis12 + angl n50', [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_angl_n50_rep{x}' for x in range(20)]))
    # groups.append(('vis12 + angl n100', [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_angl_n100_rep{x}' for x in range(20)]))
    # # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_singlecorner_landmarks_vis12_angl_noise')
    # plot_mult_EA_trends_groups_endonly(groups, val='cen', save_name='groups_endonly_landmarks_vis12_angl_noise')

    # groups = []
    # groups.append(('vis8', [f'sc_lm_CNN14_FNN2_p50e20_vis8_lm100_rep{x}' for x in range(20)]))
    # groups.append(('vis16', [f'sc_lm_CNN14_FNN2_p50e20_vis16_lm100_rep{x}' for x in range(20)]))
    # groups.append(('vis8 + LM s300', [f'sc_lm_CNN14_FNN2_p50e20_vis8_lm300_rep{x}' for x in range(20)]))
    # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_singlecorner_landmarks_size')

    # groups = []
    # groups.append(('vis12', [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_rep{x}' for x in range(20)]))
    # # groups.append(('vis12 + LM dist n50, pre LM angle calc', [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmdist_n050_rep{x}' for x in range(20)]))
    # # groups.append(('vis12 + LM dist n100, pre LM angle calc', [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmdist_n100_rep{x}' for x in range(20)]))
    # # groups.append(('vis12 + LM dist n50, post LM angle calc', [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmdistpost_n50_rep{x}' for x in range(20)]))
    # groups.append(('vis12 + LM dist n100', [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmdistpost_n100_rep{x}' for x in range(20)]))
    # # groups.append(('vis12 + LM dist n250, post LM angle calc', [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmdistpost_n250_rep{x}' for x in range(20)]))
    # groups.append(('vis12 + LM dist n250', [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmdistpost_clip_n250_rep{x}' for x in range(20)]))
    # # groups.append(('vis12 + LM dist n500, post LM angle calc', [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmdistpost_n500_rep{x}' for x in range(20)]))
    # groups.append(('vis12 + LM dist n500', [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmdistpost_clip_n500_rep{x}' for x in range(20)]))
    # # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_singlecorner_landmarks_vis12_lmdist_noise')
    # plot_mult_EA_trends_groups_endonly(groups, val='cen', save_name='groups_endonly_landmarks_vis12_lmdist_noise')

    # groups = []
    # groups.append(('vis12', [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_rep{x}' for x in range(20)]))
    # # groups.append(('vis12 + LM angle n05', [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmangle_n05_rep{x}' for x in range(20)]))
    # groups.append(('vis12 + LM angle n10', [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmangle_n10_rep{x}' for x in range(20)]))
    # groups.append(('vis12 + LM angle n25', [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmangle_n25_rep{x}' for x in range(20)]))
    # groups.append(('vis12 + LM angle n50', [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmangle_n50_rep{x}' for x in range(20)]))
    # # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_singlecorner_landmarks_vis12_lmangle_noise')
    # plot_mult_EA_trends_groups_endonly(groups, val='cen', save_name='groups_endonly_landmarks_vis12_lmangle_noise')

    # groups = []
    # groups.append(('vis12', [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_rep{x}' for x in range(20)]))
    # groups.append(('vis12 + LM radius n50', [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmradius_n50_rep{x}' for x in range(20)]))
    # groups.append(('vis12 + LM radius n100', [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmradius_n100_rep{x}' for x in range(20)]))
    # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_singlecorner_landmarks_vis12_lmradius_noise')
    # plot_mult_EA_trends_groups_endonly(groups, val='cen', save_name='groups_endonly_landmarks_vis12_lmradius_noise')

    # groups = []
    # groups.append(('vis12', [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_rep{x}' for x in range(20)]))
    # # groups.append(('vis12 + LM angle n05 + LM dist 50', [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmangle_n05_lmdist_n50_rep{x}' for x in range(20)]))
    # groups.append(('vis12 + LM angle n10 + LM dist 100', [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmangle_n10_lmdist_n100_rep{x}' for x in range(20)]))
    # groups.append(('vis12 + LM angle n25 + LM dist 250', [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmangle_n25_lmdist_n250_rep{x}' for x in range(20)]))
    # groups.append(('vis12 + LM angle n50 + LM dist 500', [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmangle_n50_lmdist_n500_rep{x}' for x in range(20)]))
    # # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_singlecorner_landmarks_vis12_lmanglepdist_noise')
    # plot_mult_EA_trends_groups_endonly(groups, val='cen', save_name='groups_endonly_landmarks_vis12_lmanglepdist_noise')


### ----------val noise----------- ###

    groups = []
    # noise = ('self_dist', ['no_noise', 'dist_n025', 'dist_n05'])
    # # noise = ('self_dist', ['no_noise', 'dist_n05', 'dist_n10'])

    noise = ('self_angle', ['no_noise', 'angle_n05', 'angle_n10'])
    # # noise = ('dist', ['no_noise', 'dist_n50', 'dist_n100'])
    # # # noise = ('self_angle', ['no_noise', 'angle_n05', 'angle_n10', 'angle_n20'])
    # # # noise = ('dist', ['no_noise', 'dist_n50', 'dist_n100', 'dist_n200'])
    # # # # noise = ('LM_angle', ['no_noise', 'lmangle_n05', 'lmangle_n10'])
    # # # # noise = ('ang_plus_dist', ['no_noise', 'angle_n05_dist_n50', 'angle_n10_dist_n100'])

    # # # # groups.append(('vis8', [f'sc_lm_CNN14_FNN2_p50e20_vis8_lm100_rep{x}' for x in range(20)]))
    # # # # groups.append(('vis10', [f'sc_lm_CNN14_FNN2_p50e20_vis10_lm100_rep{x}' for x in range(20)]))
    # # # # groups.append(('vis12', [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_rep{x}' for x in range(20)]))
    # # # # groups.append(('vis16', [f'sc_lm_CNN14_FNN2_p50e20_vis16_lm100_rep{x}' for x in range(20)]))
    # # # # groups.append(('vis24', [f'sc_lm_CNN14_FNN2_p50e20_vis24_lm100_rep{x}' for x in range(20)]))
    # # # # groups.append(('vis32', [f'sc_lm_CNN14_FNN2_p50e20_vis32_lm100_rep{x}' for x in range(20)]))

    # # # for tag, names in groups:
    # # #     plot_mult_EA_trends_valnoise(names, noise, val='cen', save_name=f'valnoise_lm_{tag}')

    # # # groups.append(('vis6', [f'sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # # # groups.append(('vis8', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # # # groups.append(('vis10', [f'sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # # # groups.append(('vis12', [f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # # # groups.append(('vis14', [f'sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # # # groups.append(('vis16', [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # # # groups.append(('vis24', [f'sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep{x}' for x in range(20)]))

    # # groups.append(('seed10k', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep{x}' for x in range(20)]))
    # # groups.append(('seed20k', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed20k_rep{x}' for x in range(20)]))
    # # groups.append(('seed30k', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed30k_rep{x}' for x in range(20)]))

    # groups.append(('min-max', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_minmax_rep{x}' for x in range(20)]))
    # groups.append(('WF {0.00, 0.00}', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep{x}' for x in range(20)]))
    # groups.append(('WF {0.20, 0.90}', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_rep{x}' for x in range(20)]))
    # groups.append(('WF {0.25, 0.75}', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep{x}' for x in range(20)]))
    # groups.append(('WF {0.30, 0.70}', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep{x}' for x in range(20)]))
    # groups.append(('WF {0.35, 0.65}', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep{x}' for x in range(20)]))
    # groups.append(('WF {0.40, 0.60}', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep{x}' for x in range(20)]))

    # # groups.append(('full WF, 0.1 std noise', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_n1_rep{x}' for x in range(20)]))
    # # groups.append(('full WF, 0.2 std noise', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_n2_rep{x}' for x in range(20)]))
    # # groups.append(('full WF, 0.3 std noise', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_n3_rep{x}' for x in range(20)]))
    # # groups.append(('full WF, 0.4 std noise', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_n4_rep{x}' for x in range(20)]))

    # for tag, names in groups:
    #     plot_mult_EA_trends_valnoise(names, noise, val='cen', save_name=f'valnoise_{tag}')


    # names = [
    #     'rotdiff_0p001',
    #     'rotdiff_0p005',
    #     'rotdiff_0p01',
    #     'rotdiff_0p05',
    #     'rotdiff_0p10',
    #     'rotdiff_0p50',
    # ]
    # plot_mult_EA_trends_randomwalk(names)


### ----------violins----------- ###

    # name = 'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom6_rep0'
    # plot_mult_EA_violins([name], 'stdev', name+'_stdev_violin')
    # plot_mult_EA_violins([name], 'mean', name+'_mean_violin')

    # name = 'sc_CNN1124_FNN2_p50e20_vis8_rep0'
    # plot_mult_EA_violins([name], 'stdev', name+'_stdev_violin')
    # plot_mult_EA_violins([name], 'mean', name+'_mean_violin')

    # names = [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom6_rep{x}' for x in range(20)]
    # plot_EA_mult_trend_violin(names, 'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom6_violin')

    # names = [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom7_rep{x}' for x in range(20)]
    # plot_EA_mult_trend_violin(names, 'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom7_violin')

    #names = [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]
    #plot_EA_mult_trend_violin(names, 'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_violin')

    # names = [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom9_rep{x}' for x in range(20)]
    # plot_EA_mult_trend_violin(names, 'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom9_violin')

    # names = [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_gap200_rep{x}' for x in range(20)]
    # plot_EA_mult_trend_violin(names, 'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_gap200_violin')
    # plot_EA_mult_trend_violin(names, 'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_gap200_violin_rescale', gap=200)

    # names = [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_scalehalf_rep{x}' for x in range(20)]
    # plot_EA_mult_trend_violin(names, 'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_scalehalf_violin')
    # plot_EA_mult_trend_violin(names, 'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_scalehalf_violin_rescale', scale=2)
