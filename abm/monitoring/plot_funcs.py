import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import collections as mc
from mpl_toolkits.mplot3d.art3d import Line3DCollection
# import matplotlib.patches as mpatches
from matplotlib import animation
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

def plot_map_iterative_traj(plot_data, x_max, y_max, w=8, h=8, save_name=None, ellipses=False, extra='', landmarks=()):

    ag_data, res_data = plot_data

    fig, axes = plt.subplots() 
    axes.set_xlim(0, x_max)
    axes.set_ylim(0, y_max)

    # rescale plotting area to square
    l,r,t,b = fig.subplotpars.left, fig.subplotpars.right, fig.subplotpars.top, fig.subplotpars.bottom
    fig.set_size_inches( float(w)/(r-l) , float(h)/(t-b) )

    if extra == 'turn':
        turn = abs(ag_data[:,:,2])

        # min_turn = np.min(turn)
        # max_turn = np.max(turn)
        min_turn = 0
        max_turn = np.pi/2
        turn = (turn - min_turn) / (max_turn - min_turn)

        ag_data[:,:,2] = turn

    # agent trajectories as gradient lines
    # ag_data = ag_data[::103,25:100,:]
    # ag_data = ag_data[::25,25:100,:]
    N_ag = ag_data.shape[0]
    # print(N_ag)
    for agent in range(N_ag):
        pos_x = ag_data[agent,:,0]
        pos_y = ag_data[agent,:,1]

        if extra == 'turn':
            turn = ag_data[agent,:,2]
            # norm = mpl.colors.Normalize(vmin=0, vmax=np.pi/8)
            # axes.add_collection(plt.scatter(pos_x, pos_y, 
            #                             c=turn, cmap='plasma_r', norm=norm, alpha=.01, s=1))
            
            axes.add_collection(plt.scatter(pos_x[25:], pos_y[25:], c=turn[25:], cmap='Blues', alpha=turn[25:]*.01, s=1))
        else:
            axes.add_collection(color_gradient(pos_x, pos_y))
            # axes.add_collection(color_gradient(pos_x, pos_y, lw=.3, alp=.8))

    # resource patches via circles
    N_res = res_data.shape[0]
    for res in range(N_res):
        pos_x = res_data[res,0,0]
        pos_y = res_data[res,0,1]
        radius = res_data[res,0,2]
        axes.add_patch( plt.Circle((pos_x, pos_y), radius, edgecolor='k', fill=False, zorder=1) )
    
    # landmarks via circles
    if landmarks:
        lm_radius, pts = landmarks
        for pos in pts:
            axes.add_patch( plt.Circle(pos, lm_radius, edgecolor='k', fill=False, zorder=1) )
    
    if ellipses:
        pts = get_ellipses()
        plt.scatter(pts[:,0], pts[:,1], c='black', alpha=.1, s=10)

    # plt.show()
    if save_name:
        if ellipses:
            plt.savefig(fr'{save_name}_ellipses.png')
        elif extra == 'turn':
            plt.savefig(fr'{save_name}_turn.png')
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


def plot_map_iterative_traj_3d(plot_data, x_max, y_max, w=8, h=8, save_name=None, plt_type='scatter', var='turn', sv_typ='plot'):
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

    if save_name and sv_typ == 'plot':
        plt.savefig(fr'{save_name}_{plt_type}_{var}.png', dpi=400)
        plt.close()

    elif save_name and sv_typ == 'anim':
        def animate(frame):
            axes.view_init(30, frame/4)
            plt.pause(.001)
            return fig

        anim = animation.FuncAnimation(fig, animate, frames=200, interval=50)
        anim.savefig(fr'{save_name}_{plt_type}_{var}_anim.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

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


def plot_mult_EA_trends_groups(groups, inter=False, val=None, save_name=None):

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

        l1 = ax1.plot(np.mean(top_stack, axis=0), 
                        # label = f'{group_name}: top individual (avg of {num_runs} runs)',
                        label = f'{group_name}',
                        color=cmap(g_num/cmap_range), 
                        alpha=0.5
                        )
        lns.append(l1[0])

        l2 = ax1.plot(np.mean(avg_stack, axis=0), 
                        label = f'{group_name}: population average (avg of {num_runs} runs)',
                        color=cmap(g_num/cmap_range), 
                        linestyle='dotted', 
                        alpha=0.5
                        )
        # lns.append(l2[0])
    
        if val is not None:
            val_group_median = np.mean(val_stack, axis=0)
            print(f'{group_name} | avg val: {int(val_group_median)}')
            if val_group_median != 0:
                ax1.hlines(val_group_median, g_num*5, data_genxpop.shape[0] + g_num*5,
                        color=cmap(g_num/cmap_range),
                        linestyle='dashed',
                        alpha=0.5
                        )

    ax1.set_xlabel('Generation')

    labs = [l.get_label() for l in lns]
    # ax1.legend(lns, labs, loc='upper right')

    # ax1.set_ylabel('# Patches Found')
    # ax1.set_ylim(0,8)
    ax1.set_ylabel('Mean Fitness')
    # ax1.set_ylim(-20,1720)
    ax1.set_ylim(-20,1520)

    if save_name: 
        plt.savefig(fr'{data_dir}/{save_name}.png')
        # plt.savefig(fr'{data_dir}/{save_name}_300.png', dpi=300)
        # plt.savefig(fr'{data_dir}/{save_name}_500.png', dpi=500)
    plt.show()


def plot_mult_EA_trends_groups_endonly(groups, val=None, save_name=None):

    import matplotlib.patches as mpatches

    # establish load directory
    root_dir = Path(__file__).parent.parent
    data_dir = Path(root_dir, r'data/simulation_data')

    # # init plot details
    fig, ax1 = plt.subplots(figsize=(15,10)) 
    cmap = plt.get_cmap('hsv')
    cmap_range = len(groups)
    lns = []
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
                    showmeans=True, 
                    showextrema=False,
                    )
        color = l0["bodies"][0].get_facecolor().flatten()
        violin_labs.append((mpatches.Patch(color=color), group_name))
        # violin_labs.append((mpatches.Patch(color=color), labs[g_num]))

        print(f'{name}: {np.min(data), np.mean(data), np.max(data)}')

    # labs = [l.get_label() for l in lns]
    # ax1.legend(lns, labs, loc='upper left')
    ax1.legend(*zip(*violin_labs), loc='upper left')
    # labs = [group_name for group_name,_ in groups]

    ax1.xaxis.set_ticklabels([])
    # ax1.set_xticklabels(labs)
    ax1.set_ylabel('Time to Find Patch')
    # ax1.set_ylim(-20,1020)

    if save_name: 
        plt.savefig(fr'{data_dir}/{save_name}.png')
    plt.show()



def plot_mult_EA_trends_valnoise(run_names, noise, val=None, save_name=None):

    # import matplotlib.patches as mpatches

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
                        showmeans=True, 
                        showextrema=False,
                        )
            for p in l0['bodies']:
                p.set_facecolor(cmap(r_num/cmap_range))
                p.set_edgecolor('black')
            l0['cmeans'].set_edgecolor('black')

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
        nbins = len(y) // 6

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

    import matplotlib.patches as mpatches

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
        # l0['cmeans'].set_edgecolor('black')

        # color = l0["bodies"][0].get_facecolor().flatten()
        # violin_labs.append((mpatches.Patch(color=color), name))

        labels[r_num] = diff_coeffs[r_num]
    
    # also add perfect traj
    with open(fr'{data_dir}/perfect.bin','rb') as f:
        data = pickle.load(f)

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



if __name__ == '__main__':
    

    # plot_LM_percep(lm_radius=100, vis_res=8, FOV=.4, save_name='landmarks_vis8_lm100')
    # plot_LM_percep(lm_radius=100, vis_res=10, FOV=.4, save_name='landmarks_vis10_lm100')
    # plot_LM_percep(lm_radius=100, vis_res=12, FOV=.4, save_name='landmarks_vis12_lm100')
    # plot_LM_percep(lm_radius=100, vis_res=16, FOV=.4, save_name='landmarks_vis16_lm100')


### ----------pop runs----------- ###

    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p50e20_vis8_nodist_rep{x}' for x in range(2)], 'mean', 'sc_CNN1124_FNN2_vis8_p50e20_nodist')

    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p50e20_vis8_wh500_rep{x}' for x in range(4)], save_name='sc_CNN1124_FNN2_vis8_p50e20_wh500')

    # plot_mult_EA_trends([f'sc_CNN1124_FNN16_p50e20_vis8_rep{x}' for x in range(5)], save_name='sc_CNN1124_FNN16_vis8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN1128_FNN2_p50e20_vis8_rep{x}' for x in range(5)], save_name='sc_CNN1128_FNN2_vis8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN18_FNN2_p50e20_vis8_rep{x}' for x in range(5)], save_name='sc_CNN18_FNN2_vis8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_rep{x}' for x in range(5)], save_name='sc_CNN14_FNN2_vis8_p50e20')

    # plot_mult_EA_trends(['sc_CNN1124_FNN2_p50e20_vis8_gen50k'], 'mean', False, 'sc_CNN1124_FNN2_vis8_p50e20_gen50k')

    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p50e20_vis8_rep{x}' for x in range(8)], save_name='sc_CNN1124_FNN2_vis8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p100e20_vis8_rep{x}' for x in range(3)], save_name='sc_CNN1124_FNN2_vis8_p100e20')
    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p200e20_vis8_rep{x}' for x in range(3)], save_name='sc_CNN1124_FNN2_vis8_p200e20')

    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss075_rep{x}' for x in range(5)], save_name='sc_CNN1124_FNN2_vis8_PGPE_ss075_p50e20')
    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_rep{x}' for x in range(5)], save_name='sc_CNN1124_FNN2_vis8_PGPE_ss10_p50e20')
    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_rep{x}' for x in range(5)], save_name='sc_CNN1124_FNN2_vis8_PGPE_ss15_p50e20')
    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom9_rep{x}' for x in range(5)], save_name='sc_CNN1124_FNN2_vis8_PGPE_ss20_mom9_p50e20')

    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_mom8_rep{x}' for x in range(5)], save_name='sc_CNN1124_FNN2_vis8_PGPE_ss10_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom8_rep{x}' for x in range(14)], save_name='sc_CNN1124_FNN2_vis8_PGPE_ss15_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(5)], save_name='sc_CNN1124_FNN2_vis8_PGPE_ss20_mom8_p50e20')

    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom7_rep{x}' for x in range(5)], save_name='sc_CNN1124_FNN2_vis8_PGPE_ss15_mom7_p50e20')
    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_mom7_rep{x}' for x in range(4)], save_name='sc_CNN1124_FNN2_vis8_PGPE_ss10_mom7_p50e20')

    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom6_rep{x}' for x in range(1)], save_name='sc_CNN1124_FNN2_vis8_PGPE_ss15_mom6_p50e20')
    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom5_rep{x}' for x in range(1)], save_name='sc_CNN1124_FNN2_vis8_PGPE_ss15_mom5_p50e20')


    # plot_mult_EA_trends([f'sc_CNN18_FNN2_p50e20_vis8_PGPE_ss15_mom8_rep{x}' for x in range(3)], save_name='sc_CNN18_FNN2_vis8_PGPE_ss15_mom8_p50e20')


    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom6_rep{x}' for x in range(20)], save_name='sc_CNN1124_FNN2_vis8_PGPE_ss20_mom6_p50e20')
    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom7_rep{x}' for x in range(10)], save_name='sc_CNN1124_FNN2_vis8_PGPE_ss20_mom7_p50e20')
    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(53)], save_name='sc_CNN1124_FNN2_vis8_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom9_rep{x}' for x in range(10)], save_name='sc_CNN1124_FNN2_vis8_PGPE_ss20_mom9_p50e20')
    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_gap200_rep{x}' for x in range(20)], save_name='sc_CNN1124_FNN2_vis8_PGPE_ss20_mom8_gap200_p50e20')
    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_scalehalf_rep{x}' for x in range(20)], save_name='sc_CNN1124_FNN2_vis8_PGPE_ss20_mom8_scalehalf_p50e20')
    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_gap200_scalshalf_rep{x}' for x in range(2)], save_name='sc_CNN1124_FNN2_vis8_PGPE_ss20_mom8_gap200_scalehalf_p50e20')

    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep{x}' for x in range(20)], save_name='sc_CNN1124_FNN2_vis6_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(53)], save_name='sc_CNN1124_FNN2_vis8_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p50e20_vis9_PGPE_ss20_mom8_rep{x}' for x in range(20)], save_name='sc_CNN1124_FNN2_vis9_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep{x}' for x in range(26)], save_name='sc_CNN1124_FNN2_vis10_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN1122_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(15)], save_name='sc_CNN1122_FNN2_vis8_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN1124_GRU4_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)], save_name='sc_CNN1124_GRU4_vis8_PGPE_ss20_mom8_p50e20')


    # plot_mult_EA_trends([f'sc_CNN12_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)], save_name='sc_CNN12_FNN2_vis8_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)], save_name='sc_CNN13_FNN2_vis8_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)], save_name='sc_CNN14_FNN2_vis8_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN15_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)], save_name='sc_CNN15_FNN2_vis8_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN16_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(13)], save_name='sc_CNN16_FNN2_vis8_PGPE_ss20_mom8_p50e20')

    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep{x}' for x in range(9)], save_name='sc_CNN14_FNN2_vis10_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep{x}' for x in range(16)], save_name='sc_CNN14_FNN2_vis12_PGPE_ss20_mom8_p50e20')

    # plot_mult_EA_trends([f'sc_CNN14_FNN1_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)], save_name='sc_CNN14_FNN1_vis8_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN13_FNN32_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(9)], save_name='sc_CNN13_FNN32_vis8_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN13_FNN1_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)], save_name='sc_CNN13_FNN1_vis8_PGPE_ss20_mom8_p50e20')

    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov2_rep{x}' for x in range(20)], save_name='sc_CNN14_FNN2_vis8_fov2_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov3_rep{x}' for x in range(20)], save_name='sc_CNN14_FNN2_vis8_fov3_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep{x}' for x in range(20)], save_name='sc_CNN14_FNN2_vis8_fov35_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov45_rep{x}' for x in range(20)], save_name='sc_CNN14_FNN2_vis8_fov45_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov5_rep{x}' for x in range(19)], save_name='sc_CNN14_FNN2_vis8_fov5_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov6_rep{x}' for x in range(20)], save_name='sc_CNN14_FNN2_vis8_fov6_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov7_rep{x}' for x in range(20)], save_name='sc_CNN14_FNN2_vis8_fov7_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov8_rep{x}' for x in range(20)], save_name='sc_CNN14_FNN2_vis8_fov8_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov875_rep{x}' for x in range(20)], save_name='sc_CNN14_FNN2_vis8_fov875_PGPE_ss20_mom8_p50e20')


    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_block52_rep{x}' for x in range(20)], save_name='sc_CNN14_FNN2_vis8_block52_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_block52_rep{x}' for x in range(20)], save_name='sc_CNN14_FNN2_vis10_block52_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_block52_rep{x}' for x in range(20)], save_name='sc_CNN14_FNN2_vis12_block52_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_head44_rep{x}' for x in range(9)], save_name='sc_CNN14_FNN2_vis8_head44_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_block52_head44_rep{x}' for x in range(20)], save_name='sc_CNN14_FNN2_vis16_block52_head44_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_block52_head44_rep{x}' for x in range(20)], save_name='sc_CNN14_FNN2_vis8_block52_head44_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_block52_head44_rep{x}' for x in range(13)], save_name='sc_CNN14_FNN2_vis12_block52_head44_PGPE_ss20_mom8_p50e20')

    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_vestib_rep{x}' for x in range(20)], save_name='sc_CNN14_FNN2_vis8_vestib_PGPE_ss20_mom8_p50e20')

    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_far_rep{x}' for x in range(17)], save_name='sc_CNN1124_FNN2_vis8_dist_far_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_close_rep{x}' for x in range(20)], save_name='sc_CNN1124_FNN2_vis8_dist_close_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_far_rep{x}' for x in range(20)], save_name='sc_CNN14_FNN2_vis8_dist_far_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_close_rep{x}' for x in range(20)], save_name='sc_CNN14_FNN2_vis8_dist_close_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_minmax_rep{x}' for x in range(20)], save_name='sc_CNN14_FNN2_vis8_dist_minmax_PGPE_ss20_mom8_p50e20')

    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='sc_CNN1124_FNN2_vis6_PGPE_ss20_mom8_p50e20_valcen')
    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='sc_CNN1124_FNN2_vis8_PGPE_ss20_mom8_p50e20_valcen') # 53 total
    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p50e20_vis9_PGPE_ss20_mom8_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='sc_CNN1124_FNN2_vis9_PGPE_ss20_mom8_p50e20_valcen')
    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='sc_CNN1124_FNN2_vis10_PGPE_ss20_mom8_p50e20_valcen') # 26 total

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

    # plot_mult_EA_trends([f'sc_CNN1122_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(15)], val='cen', 
    #                     save_name='sc_CNN1122_FNN2_vis8_PGPE_ss20_mom8_p50e20_valcen')
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

    # plot_mult_EA_trends([f'sc_CNN14_FNN1_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='sc_CNN14_FNN1_vis8_PGPE_ss20_mom8_p50e20_valcen')
    # plot_mult_EA_trends([f'sc_CNN13_FNN1_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='sc_CNN13_FNN1_vis8_PGPE_ss20_mom8_p50e20_valcen')
    
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

    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_block52_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='sc_CNN14_FNN2_vis8_PGPE_ss20_mom8_block52_p50e20_valcen')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_block52_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='sc_CNN14_FNN2_vis10_PGPE_ss20_mom8_block52_p50e20_valcen')

    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_block52_head44_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='sc_CNN14_FNN2_vis8_PGPE_ss20_mom8_block52_head44_p50e20_valcen')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_block52_head44_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='sc_CNN14_FNN2_vis10_PGPE_ss20_mom8_block52_head44_p50e20_valcen')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_block52_head44_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='sc_CNN14_FNN2_vis12_PGPE_ss20_mom8_block52_head44_p50e20_valcen')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_block52_head44_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='sc_CNN14_FNN2_vis14_PGPE_ss20_mom8_block52_head44_p50e20_valcen')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_block52_head44_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='sc_CNN14_FNN2_vis16_PGPE_ss20_mom8_block52_head44_p50e20_valcen')

    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_head44_rep{x}' for x in range(9)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_head44_PGPE_ss20_mom8_p50e20')

    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_vestib_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_vestib_PGPE_ss20_mom8_p50e20')

    # plot_mult_EA_trends([f'sc_CNN1124_GRU4_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)], val='cen', 
    #                     save_name='sc_CNN1124_GRU4_vis8_PGPE_ss20_mom8_p50e20_valcen')

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


    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res55_rep{x}' for x in range(20)], val='cen', #done
    #                     save_name='sc_CNN14_FNN2_vis8_res55_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res54_rep{x}' for x in range(20)], val='cen', #done
    #                     save_name='sc_CNN14_FNN2_vis8_res54_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res53_rep{x}' for x in range(20)], val='cen', #done
    #                     save_name='sc_CNN14_FNN2_vis8_res53_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res52_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_res52_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res51_rep{x}' for x in range(20)], val='cen', #done
    #                     save_name='sc_CNN14_FNN2_vis8_res51_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res50_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_res50_PGPE_ss20_mom8_p50e20')

    ## plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res44_rep{x}' for x in range(20)], val='cen', 
    ##                     save_name='sc_CNN14_FNN2_vis8_res44_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res43_rep{x}' for x in range(20)], val='cen', #done
    #                     save_name='sc_CNN14_FNN2_vis8_res43_PGPE_ss20_mom8_p50e20') 
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res42_rep{x}' for x in range(20)], val='cen', #done
    #                     save_name='sc_CNN14_FNN2_vis8_res42_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res41_rep{x}' for x in range(20)], val='cen', #done
    #                     save_name='sc_CNN14_FNN2_vis8_res41_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res40_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_res40_PGPE_ss20_mom8_p50e20')

    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res33_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_res33_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res32_rep{x}' for x in range(20)], val='cen', #done
    #                     save_name='sc_CNN14_FNN2_vis8_res32_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res31_rep{x}' for x in range(20)], val='cen', #done
    #                     save_name='sc_CNN14_FNN2_vis8_res31_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res30_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_res30_PGPE_ss20_mom8_p50e20')
                        
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res22_rep{x}' for x in range(20)], val='cen', #done
    #                     save_name='sc_CNN14_FNN2_vis8_res22_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res21_rep{x}' for x in range(20)], val='cen', #done
    #                     save_name='sc_CNN14_FNN2_vis8_res21_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res20_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_res20_PGPE_ss20_mom8_p50e20')
                        
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res11_rep{x}' for x in range(20)], val='cen', #done
    #                     save_name='sc_CNN14_FNN2_vis8_res11_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res10_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_res10_PGPE_ss20_mom8_p50e20')
                        
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res0_rep{x}' for x in range(20)], val='cen', #done
    #                     save_name='sc_CNN14_FNN2_vis8_res0_PGPE_ss20_mom8_p50e20')

    
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


    # plot_mult_EA_trends([f'sc_CNN14_FNN3_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN3_vis8_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN4_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)], 
    #                     save_name='sc_CNN14_FNN4_vis8_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN8_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN8_vis8_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)], 
    #                     save_name='sc_CNN14_FNN16_vis8_PGPE_ss20_mom8_p50e20')


    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_silu_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_PGPE_ss20_mom8_silu_p50e20')


    # plot_mult_EA_trends([f'sc_lm_CNN14_FNN2_p50e20_vis8_lm100_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_lm_CNN14_FNN2_p50e20_vis8_lm100')
    # plot_mult_EA_trends([f'sc_lm_CNN14_FNN2_p50e20_vis10_lm100_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_lm_CNN14_FNN2_p50e20_vis10_lm100')
    plot_mult_EA_trends([f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_rep{x}' for x in range(20)], val='cen',
                        save_name='sc_lm_CNN14_FNN2_p50e20_vis12_lm100')
    # plot_mult_EA_trends([f'sc_lm_CNN14_FNN2_p50e20_vis16_lm100_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_lm_CNN14_FNN2_p50e20_vis16_lm100')
    # plot_mult_EA_trends([f'sc_lm_CNN14_FNN2_p50e20_vis24_lm100_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_lm_CNN14_FNN2_p50e20_vis24_lm100')
    plot_mult_EA_trends([f'sc_lm_CNN14_FNN2_p50e20_vis32_lm100_rep{x}' for x in range(20)], val='cen',
                        save_name='sc_lm_CNN14_FNN2_p50e20_vis32_lm100')
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

    # plot_mult_EA_trends([f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmdist_n050_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmdist_n050')
    # plot_mult_EA_trends([f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmdist_n100_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmdist_n100')
    # plot_mult_EA_trends([f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmdistpost_n50_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmdistpost_n50')
    # plot_mult_EA_trends([f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmdistpost_n100_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmdistpost_n100')
    # plot_mult_EA_trends([f'sc_lm_CNN14_FNN2_p50e20_vis12_LMdist_n500_rep{x}' for x in range(20)], 
    #                     save_name='sc_lm_CNN14_FNN2_p50e20_vis12_LMdist_n500')

    # plot_mult_EA_trends([f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmangle_n05_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmangle_n05')
    # plot_mult_EA_trends([f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmangle_n10_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmangle_n10')
    # plot_mult_EA_trends([f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmangle_n50_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmangle_n50')

    # plot_mult_EA_trends([f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmradius_n50_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmradius_n50')
    # plot_mult_EA_trends([f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmradius_n100_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmradius_n100')

    # plot_mult_EA_trends([f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmangle_n05_lmdist_n50_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmangle_n05_lmdist_n50')
    # plot_mult_EA_trends([f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmangle_n10_lmdist_n100_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmangle_n10_lmdist_n100')
    # plot_mult_EA_trends([f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmangle_n50_lmdist_n500_rep{x}' for x in range(20)], 
    #                     save_name='sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmangle_n50_lmdist_n500')


### ----------pop runs >> num_inter ----------- ###

    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_mom9_inter_rep{x}' for x in range(11)], inter=True, save_name='sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_mom9_inter')
    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_mom10_inter_rep{x}' for x in range(10)], inter=True, save_name='sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_mom10_inter')
    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom9_inter_rep{x}' for x in range(7)], inter=True, save_name='sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom9_inter')
    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom8_inter_rep{x}' for x in range(10)], inter=True, save_name='sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom8_inter')
    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_inter_rep{x}' for x in range(7)], inter=True, save_name='sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_inter')
    # plot_mult_EA_trends([f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom7_inter_rep{x}' for x in range(4)], inter=True, save_name='sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom7_inter')

    # groups = []
    # # groups.append(('ss10_mom9', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_mom9_inter_rep{x}' for x in range(11)]))
    # # groups.append(('ss10_mom10', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_mom10_inter_rep{x}' for x in range(9)]))
    # # groups.append(('ss15_mom7', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom7_inter_rep{x}' for x in range(3)]))
    # groups.append(('ss15_mom9', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom8_inter_rep{x}' for x in range(10)]))
    # # groups.append(('ss15_mom9', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom9_inter_rep{x}' for x in range(6)]))
    # # groups.append(('ss20_mom8', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_inter_rep{x}' for x in range(6)]))
    # plot_mult_EA_trends_groups(groups, inter=True, save_name='groups_singlecorner_PGPE_inter_mom8')

### ----------group pop runs----------- ###

    groups = []

    # groups.append(('CNN14_FNN2_p25e5_vis8',[f'sc_CNN14_FNN2_p25e5_vis8_rep{x}' for x in range(20)]))
    # groups.append(('CNN14_FNN2_p25e5_vis48',[f'sc_CNN14_FNN2_p25e5_vis48_rep{x}' for x in range(20)]))
    # groups.append(('CNN1124_FNN2_p25e5_vis8',[f'sc_CNN1124_FNN2_p25e5_vis8_rep{x}' for x in range(20)]))
    # groups.append(('CNN1124_FNN2_p25e5_vis48',[f'sc_CNN1124_FNN2_p25e5_vis48_rep{x}' for x in range(20)]))
    # # groups.append(('CNN1124_FNN2_p25e5_vis8_noisevis5',[f'sc_CNN1124_FNN24_p25e5_vis8_noisevis5_rep{x}' for x in range(8)]))
    # plot_mult_EA_trends_groups(groups, save_name='groups_singlecorner_CNN-vis')

    # groups.append(('CNN1124_FNN2_p25e5_vis8',[f'sc_CNN1124_FNN2_p25e5_vis8_rep{x}' for x in range(20)]))
    # groups.append(('CNN1124_FNN2_p25e15_vis8',[f'sc_CNN1124_FNN2_p25e15_vis8_rep{x}' for x in range(3)]))
    # groups.append(('CNN1124_FNN2_p50e15_vis8',[f'sc_CNN1124_FNN2_p50e15_vis8_rep{x}' for x in range(5)]))
    # groups.append(('CNN1124_FNN2_p50e20_vis8',[f'sc_CNN1124_FNN2_p50e20_vis8_rep{x}' for x in range(5)]))
    # plot_mult_EA_trends_groups(groups, save_name='groups_singlecorner_pop_ep')

    # groups.append(('CNN1124_FNN2_p50e20_vis8',[f'sc_CNN1124_FNN2_p50e20_vis8_rep{x}' for x in range(5)]))
    # groups.append(('CNN1124_FNN16_p50e20_vis8', [f'sc_CNN1124_FNN16_p50e20_vis8_rep{x}' for x in range(5)]))
    # plot_mult_EA_trends_groups(groups, save_name='groups_singlecorner_FNN')

    # groups.append(('CNN1124_FNN2_p50e20_vis8',[f'sc_CNN1124_FNN2_p50e20_vis8_rep{x}' for x in range(5)]))
    # groups.append(('CNN1124_FNN2_p50e20_vis8_sn05afCNN', [f'sc_CNN1124_FNN2_p50e20_vis8_sn05afCNN_rep{x}' for x in range(5)]))
    # groups.append(('CNN1124_FNN2_p50e20_vis8_sn025afCNN', [f'sc_CNN1124_FNN2_p50e20_vis8_sn025afCNN_rep{x}' for x in range(5)]))
    # groups.append(('CNN1124_FNN2_p50e20_vis8_sn05b4CNN', [f'sc_CNN1124_FNN2_p50e20_vis8_sn05b4CNN_rep{x}' for x in range(2)]))
    # plot_mult_EA_trends_groups(groups, save_name='groups_singlecorner_noise')
    
    # groups.append(('CNN14_FNN2_p50e20_vis8', [f'sc_CNN14_FNN2_p50e20_vis8_rep{x}' for x in range(5)]))
    # groups.append(('CNN18_FNN2_p50e20_vis8', [f'sc_CNN18_FNN2_p50e20_vis8_rep{x}' for x in range(5)]))
    # groups.append(('CNN1124_FNN2_p50e20_vis8',[f'sc_CNN1124_FNN2_p50e20_vis8_rep{x}' for x in range(4)]))
    # groups.append(('CNN1128_FNN2_p50e20_vis8', [f'sc_CNN1128_FNN2_p50e20_vis8_rep{x}' for x in range(5)]))
    # plot_mult_EA_trends_groups(groups, save_name='groups_singlecorner_CNN')

    # groups.append(('CMA-ES - pop50',[f'sc_CNN1124_FNN2_p50e20_vis8_rep{x}' for x in range(7)]))
    # groups.append(('CMA-ES - pop100',[f'sc_CNN1124_FNN2_p100e20_vis8_rep{x}' for x in range(3)]))
    # groups.append(('CMA-ES - pop200',[f'sc_CNN1124_FNN2_p200e20_vis8_rep{x}' for x in range(3)]))
    # groups.append(('PGPE - pop50 - ss075 - mom9', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss075_rep{x}' for x in range(5)]))
    # groups.append(('PGPE - pop50 - ss10 - mom9', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_rep{x}' for x in range(5)]))
    # groups.append(('PGPE - pop50 - ss15 - mom9', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_rep{x}' for x in range(5)]))
    # groups.append(('PGPE - pop50 - ss10 - mom8', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_mom8_rep{x}' for x in range(5)]))
    # groups.append(('PGPE - pop50 - ss15 - mom8', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom8_rep{x}' for x in range(5)]))
    # groups.append(('PGPE - pop50 - ss20 - mom8', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(5)]))
    # groups.append(('PGPE - pop50 - ss10 - mom7', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_mom7_rep{x}' for x in range(4)]))
    # groups.append(('PGPE - pop50 - ss15 - mom7', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom7_rep{x}' for x in range(5)]))
    # plot_mult_EA_trends_groups(groups, save_name='groups_singlecorner_optim')

    # groups.append(('ss10 - mom7', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_mom7_rep{x}' for x in range(4)]))
    # groups.append(('ss15 - mom7', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom7_rep{x}' for x in range(5)]))
    # groups.append(('ss20 - mom7', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom7_rep{x}' for x in range(5)]))
    # groups.append(('ss10 - mom8', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_mom8_rep{x}' for x in range(5)]))
    # groups.append(('ss15 - mom8', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom8_rep{x}' for x in range(14)]))
    # groups.append(('ss20 - mom8', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(11)]))
    # groups.append(('ss075 - mom9', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss075_rep{x}' for x in range(5)]))
    # groups.append(('ss10 - mom9', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_rep{x}' for x in range(5)]))
    # groups.append(('ss15 - mom9', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_rep{x}' for x in range(5)]))
    # groups.append(('ss20 - mom9', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom9_rep{x}' for x in range(9)]))
    # plot_mult_EA_trends_groups(groups, save_name='groups_singlecorner_optim_by_mom')

    # groups.append(('ss075 - mom9', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss075_rep{x}' for x in range(5)]))
    # groups.append(('ss10 - mom7', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_mom7_rep{x}' for x in range(4)]))
    # groups.append(('ss10 - mom8', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_mom8_rep{x}' for x in range(5)]))
    # groups.append(('ss10 - mom9', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_rep{x}' for x in range(5)]))
    # groups.append(('ss15 - mom7', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom7_rep{x}' for x in range(5)]))
    # groups.append(('ss15 - mom8', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom8_rep{x}' for x in range(14)]))
    # groups.append(('ss15 - mom9', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_rep{x}' for x in range(5)]))
    # groups.append(('ss20 - mom6', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom6_rep{x}' for x in range(3)]))
    # groups.append(('ss20 - mom7', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom7_rep{x}' for x in range(16)]))
    # groups.append(('ss20 - mom8', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('ss20 - mom9', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom9_rep{x}' for x in range(16)]))
    # plot_mult_EA_trends_groups(groups, save_name='groups_singlecorner_optim_by_ss')

    # groups.append(('ss20 - mom6', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom6_rep{x}' for x in range(20)]))
    # groups.append(('ss20 - mom7', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom7_rep{x}' for x in range(20)]))
    # groups.append(('ss20 - mom8', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('ss20 - mom9', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom9_rep{x}' for x in range(20)]))
    # plot_mult_EA_trends_groups(groups, save_name='groups_singlecorner_optim_by_ss20')

    # groups.append(('ss20 - mom8', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('ss20 - mom8 - gap200', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_gap200_rep{x}' for x in range(20)]))
    # groups.append(('ss20 - mom8 - scalehalf', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_scalehalf_rep{x}' for x in range(20)]))
    # groups.append(('ss20 - mom8 - gap200 + scalehalf', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_gap200_scalehalf_rep{x}' for x in range(7)]))
    # plot_mult_EA_trends_groups(groups, save_name='groups_singlecorner_optim_by_ss20_mom8_shaping')

    # groups.append(('CMA-ES - pop50',[f'sc_CNN1124_FNN2_p50e20_vis8_rep{x}' for x in range(7)]))
    # groups.append(('CMA-ES - pop100',[f'sc_CNN1124_FNN2_p100e20_vis8_rep{x}' for x in range(3)]))
    # groups.append(('CMA-ES - pop200',[f'sc_CNN1124_FNN2_p200e20_vis8_rep{x}' for x in range(3)]))
    # plot_mult_EA_trends_groups(groups, save_name='groups_singlecorner_optim_CMAES')

    # groups.append(('ss20 - mom7', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom7_rep{x}' for x in range(10)]))
    # groups.append(('ss20 - mom8', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('ss20 - mom9', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom9_rep{x}' for x in range(10)]))
    # plot_mult_EA_trends_groups(groups, save_name='groups_singlecorner_optim_by_ss20')


    ### FEAT ###
    # # groups.append(('CNN 1122', [f'sc_CNN1122_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(15)]))
    # # groups.append(('CNN 1124', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(53)]))
    # groups.append(('CNN 12', [f'sc_CNN12_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('CNN 13', [f'sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('CNN 14', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('CNN 15', [f'sc_CNN15_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('CNN 16', [f'sc_CNN16_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('CNN 17', [f'sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_singlecorner_CNN')
    # plot_mult_EA_trends_groups_endonly(groups, val='cen', save_name='groups_endonly_CNN')

    # groups.append(('FNN 2', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(53)]))
    # groups.append(('GRU 4', [f'sc_CNN1124_GRU4_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # plot_mult_EA_trends_groups(groups, save_name='groups_singlecorner_integrator_memory')


    # ## VIS ###
    # groups.append(('vis 6', [f'sc_CNN1124_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('vis 8', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(53)]))
    # groups.append(('vis 9', [f'sc_CNN1124_FNN2_p50e20_vis9_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('vis 10', [f'sc_CNN1124_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep{x}' for x in range(26)]))
    # plot_mult_EA_trends_groups(groups, save_name='groups_singlecorner_vis')
    # groups.append(('vis 6', [f'sc_CNN1124_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('vis 8', [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('vis 9', [f'sc_CNN1124_FNN2_p50e20_vis9_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('vis 10', [f'sc_CNN1124_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_singlecorner_vis')
    # groups = []
    # groups.append(('vis 6', [f'sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('vis 8', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('vis 10', [f'sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('vis 12', [f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('vis 14', [f'sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('vis 16', [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('vis 20', [f'sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('vis 24', [f'sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('vis 32', [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_singlecorner_vis')
    # plot_mult_EA_trends_groups_endonly(groups, val='cen', save_name='groups_endonly_vis')
    # groups = []
    # groups.append(('block52 - vis8', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_block52_rep{x}' for x in range(20)]))
    # groups.append(('block52 - vis10', [f'sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_block52_rep{x}' for x in range(20)]))
    # groups.append(('block52 - vis12', [f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_block52_rep{x}' for x in range(20)]))
    # plot_mult_EA_trends_groups(groups, save_name='groups_singlecorner_CNN_kernels_block52')
    # groups = []
    # groups.append(('block52/head44 - vis8', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_block52_head44_rep{x}' for x in range(20)]))
    # groups.append(('block52/head44 - vis10', [f'sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_block52_head44_rep{x}' for x in range(20)]))
    # groups.append(('block52/head44 - vis12', [f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_block52_head44_rep{x}' for x in range(20)]))
    # groups.append(('block52/head44 - vis14', [f'sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_block52_head44_rep{x}' for x in range(20)]))
    # groups.append(('block52/head44 - vis16', [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_block52_head44_rep{x}' for x in range(20)]))
    # # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_singlecorner_block52head44_vis')
    # plot_mult_EA_trends_groups_endonly(groups, val='cen', save_name='groups_endonly_block52head44_vis')

    # groups = []
    # groups.append(('CNN13 - FNN 1', [f'sc_CNN13_FNN1_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('CNN13 - FNN 2', [f'sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # # groups.append(('CNN13 - FNN 32', [f'sc_CNN13_FNN32_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(8)]))
    # groups.append(('CNN14 - FNN 1', [f'sc_CNN14_FNN1_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('CNN14 - FNN 2', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))

    ### FNN SIZE ###
    # groups = []
    # groups.append(('FNN 1', [f'sc_CNN14_FNN1_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('FNN 2', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('FNN 3', [f'sc_CNN14_FNN3_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('FNN 4', [f'sc_CNN14_FNN4_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('FNN 8', [f'sc_CNN14_FNN8_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('FNN 16', [f'sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('FNN 2x2', [f'sc_CNN14_FNN2x2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('FNN 2x3', [f'sc_CNN14_FNN2x3_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('FNN 2x4', [f'sc_CNN14_FNN2x4_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # # groups.append(('FNN 2x8', [f'sc_CNN14_FNN2x8_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('FNN 2x16', [f'sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_singlecorner_integrator_size')
    # plot_mult_EA_trends_groups_endonly(groups, val='cen', save_name='groups_endonly_integrator_size')


    ### FOV ###
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
    # plot_mult_EA_trends_groups_endonly(groups, val='cen', save_name='groups_endonly_FOV')


    ### DIST ###
    # groups = []
    # groups.append(('min-max', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_minmax_rep{x}' for x in range(20)]))
    # groups.append(('WF scaling ~ {0.20, 0.90}', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_rep{x}' for x in range(20)]))
    # groups.append(('WF scaling ~ {0.25, 0.75}', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep{x}' for x in range(20)]))
    # groups.append(('WF scaling ~ {0.30, 0.70}', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep{x}' for x in range(20)]))
    # groups.append(('WF scaling ~ {0.45, 0.65}', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep{x}' for x in range(20)]))
    # groups.append(('WF scaling ~ {0.40, 0.60}', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep{x}' for x in range(20)]))
    # groups.append(('WF scaling ~ {0.45, 0.55}', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep{x}' for x in range(20)]))
    # groups.append(('no dist scaling', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_singlecorner_WF_scaling')
    # plot_mult_EA_trends_groups_endonly(groups, val='cen', save_name='groups_endonly_singlecorner_WF_scaling')

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

    ### ACTIV FUNC ###
    # groups = []
    # # groups.append(('relu', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('relu + angl noise n10', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_angl_n10_rep{x}' for x in range(20)]))
    # groups.append(('relu + angl noise n10 -- seed 40k', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed40k_angl_n10_rep{x+2}' for x in range(18)]))
    # groups.append(('silu + angl noise n10', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_silu_angl_n10_rep{x}' for x in range(20)]))
    # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_singlecorner_activ_func')


    ### SEED ###
    # groups = []
    # groups.append(('seed 1k', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('seed 10k', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep{x}' for x in range(20)]))
    # groups.append(('seed 20k', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed20k_rep{x}' for x in range(20)]))
    # groups.append(('seed 30k', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed30k_rep{x}' for x in range(20)]))
    # # groups.append(('seed 40k', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed40k_rep{x}' for x in range(20)]))
    # # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_singlecorner_seed')
    # plot_mult_EA_trends_groups_endonly(groups, val='cen', save_name='groups_endonly_seed')


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

    noise = ('self_dist', ['no_noise', 'dist_n025', 'dist_n05'])
    # noise = ('self_dist', ['no_noise', 'dist_n05', 'dist_n10'])

    # noise = ('self_angle', ['no_noise', 'angle_n05', 'angle_n10'])
    # noise = ('dist', ['no_noise', 'dist_n50', 'dist_n100'])
    # # noise = ('self_angle', ['no_noise', 'angle_n05', 'angle_n10', 'angle_n20'])
    # # noise = ('dist', ['no_noise', 'dist_n50', 'dist_n100', 'dist_n200'])
    # # # noise = ('LM_angle', ['no_noise', 'lmangle_n05', 'lmangle_n10'])
    # # # noise = ('ang_plus_dist', ['no_noise', 'angle_n05_dist_n50', 'angle_n10_dist_n100'])

    # # # groups.append(('vis8', [f'sc_lm_CNN14_FNN2_p50e20_vis8_lm100_rep{x}' for x in range(20)]))
    # # # groups.append(('vis10', [f'sc_lm_CNN14_FNN2_p50e20_vis10_lm100_rep{x}' for x in range(20)]))
    # # # groups.append(('vis12', [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_rep{x}' for x in range(20)]))
    # # # groups.append(('vis16', [f'sc_lm_CNN14_FNN2_p50e20_vis16_lm100_rep{x}' for x in range(20)]))
    # # # groups.append(('vis24', [f'sc_lm_CNN14_FNN2_p50e20_vis24_lm100_rep{x}' for x in range(20)]))
    # # # groups.append(('vis32', [f'sc_lm_CNN14_FNN2_p50e20_vis32_lm100_rep{x}' for x in range(20)]))

    # # for tag, names in groups:
    # #     plot_mult_EA_trends_valnoise(names, noise, val='cen', save_name=f'valnoise_lm_{tag}')

    # # groups.append(('vis6', [f'sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # # groups.append(('vis8', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # # groups.append(('vis10', [f'sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # # groups.append(('vis12', [f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # # groups.append(('vis14', [f'sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # # groups.append(('vis16', [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # # groups.append(('vis24', [f'sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep{x}' for x in range(20)]))

    # groups.append(('seed10k', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep{x}' for x in range(20)]))
    # groups.append(('seed20k', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed20k_rep{x}' for x in range(20)]))
    # groups.append(('seed30k', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed30k_rep{x}' for x in range(20)]))

    groups.append(('min-max', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_minmax_rep{x}' for x in range(20)]))
    groups.append(('WF {0.20, 0.90}', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_rep{x}' for x in range(20)]))
    groups.append(('WF {0.25, 0.75}', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep{x}' for x in range(20)]))
    groups.append(('WF {0.30, 0.70}', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep{x}' for x in range(20)]))
    groups.append(('WF {0.35, 0.65}', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep{x}' for x in range(20)]))
    groups.append(('WF {0.40, 0.60}', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep{x}' for x in range(20)]))

    # groups.append(('full WF, 0.1 std noise', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_n1_rep{x}' for x in range(20)]))
    # groups.append(('full WF, 0.2 std noise', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_n2_rep{x}' for x in range(20)]))
    # groups.append(('full WF, 0.3 std noise', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_n3_rep{x}' for x in range(20)]))
    # groups.append(('full WF, 0.4 std noise', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_n4_rep{x}' for x in range(20)]))

    for tag, names in groups:
        plot_mult_EA_trends_valnoise(names, noise, val='cen', save_name=f'valnoise_{tag}')


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