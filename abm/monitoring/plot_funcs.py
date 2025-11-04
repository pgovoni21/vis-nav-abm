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
import seaborn as sns

# ------------------------------- tools ---------------------------------------- #

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


def beeswarm(y, nbins=None, scaling=2.25):
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
    # print(int(ylo),int(np.median(y)),int(yhi),len(ybins))

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
    if nmax == 1:
        nmax = 2
    dx = 1 / (nmax // 2)
    for i, y in zip(ibs, ybs):
        if len(i) > 1:
            j = len(i) % 2
            i = i[np.argsort(y)]
            a = i[j::2]
            b = i[j+1::2]
            x[a] = (0.5 + j / 3 + np.arange(len(b))) * dx / scaling
            x[b] = (0.5 + j / 3 + np.arange(len(b))) * -dx / scaling

    return x


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

            axes.plot(pos_x, pos_y, color, linewidth=3)
            axes.plot(pos_x, pos_y, 'k:', linewidth=3)
            axes.plot(pos_x[0], pos_y[0], marker='o', c=color, markeredgecolor='k', markeredgewidth=2, ms=15)
    
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



def plot_map_iterative_collisions(spike_locs, res_data, x_max, y_max, ag_rad, save_name=None, dpi=50):

    for i,key in enumerate(spike_locs.keys()):

        if not (key == 'SC' or key == 'DC'):
            continue

        print(f'plotting {i} - {key}')

        ag_data = np.array(spike_locs[key])
        x = ag_data[:,0]
        y = ag_data[:,1]
        ori = ag_data[:,2]


        ### scatter

        fig, axes = plt.subplots() 
        axes.set_xlim(0, x_max)
        axes.set_ylim(0, y_max)
        h,w = 8,8
        l,r,t,b = fig.subplotpars.left, fig.subplotpars.right, fig.subplotpars.top, fig.subplotpars.bottom
        fig.set_size_inches( float(w)/(r-l) , float(h)/(t-b) )

        # print(x.min(), x.max(), y.min(), y.max(), ori.min(), ori.max())

        # collisions at each loc + colored via ori
        sc = axes.scatter(x, y, c=ori, cmap='hsv', alpha=0.01, s=5)
        # cbar = axes.figure.colorbar(sc, cax=axes, fraction=0.046, pad=0.04,
        #             ticks=np.arange(0, 2*np.pi+0.01, np.pi/2),
        #             format=mpl.ticker.FixedFormatter([r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']),
        #             )
        # cbar.solids.set(alpha=1)
        axes.set_title(key)

        # resource patches via circles
        N_res = res_data.shape[0]
        for res in range(N_res):
            patch_x, patch_y, patch_rad = res_data[res,0,:]
            axes.add_patch( plt.Circle((patch_x, patch_y), patch_rad, edgecolor='k', fill=False, zorder=1) )

        fig.tight_layout()
        plt.savefig(fr'{save_name}_plot{i}_scatter_{dpi}.png', dpi=dpi)
        plt.close()


        ### heatmap

        fig, axes = plt.subplots() 
        axes.set_xlim(0, x_max)
        axes.set_ylim(0, y_max)
        h,w = 8,8
        l,r,t,b = fig.subplotpars.left, fig.subplotpars.right, fig.subplotpars.top, fig.subplotpars.bottom
        fig.set_size_inches( float(w)/(r-l) , float(h)/(t-b) )

        coll_boundary_thickness = ag_rad
        scale = 1
        space_step = 25
        x_bins = np.linspace(coll_boundary_thickness, 
                            x_max - coll_boundary_thickness + 1, 
                            int(scale*(x_max - coll_boundary_thickness*2) / space_step+1))
        y_bins = np.linspace(coll_boundary_thickness, 
                            y_max - coll_boundary_thickness + 1, 
                            int(scale*(y_max - coll_boundary_thickness*2) / space_step+1))
        X,Y = np.meshgrid(x_bins, y_bins)

        H,_,_ = np.histogram2d(x,y, bins=[x_bins, y_bins])

        # im = axes.pcolormesh(X, Y, H.T, cmap='plasma')
        min = H.min()
        max = H.max()

        if key == 'SC' or key == 'DC':
            cbar_scale = .1
        else:
            cbar_scale = .1

        norm = mpl.colors.Normalize(vmin=min, vmax=(max-min)*cbar_scale+min)
        im = axes.pcolormesh(X, Y, H.T, cmap='plasma', norm=norm)

        cbar = axes.figure.colorbar(im, ax=axes, fraction=0.046, pad=0.04, label='Count', extend='max')
        axes.add_patch( plt.Circle((patch_x, patch_y), patch_rad, edgecolor='k', fill=False, zorder=1) )

        fig.tight_layout()
        plt.savefig(fr'{save_name}_plot{i}_heatmap_{dpi}.png', dpi=dpi)
        plt.close()



# ------------------------------- social tables ---------------------------------------- #


def plot_social_table_DR(metric_type, metric_thresh=None, coll=True, dpi=50):

    fig, ax = plt.subplots()
    h,w = 4,4.5
    l,r,t,b = fig.subplotpars.left, fig.subplotpars.right, fig.subplotpars.top, fig.subplotpars.bottom
    fig.set_size_inches( float(w)/(r-l) , float(h)/(t-b) )
    plt.box(False)

    x_max = 5
    y_max = 5
    x = np.linspace(0, x_max, num=x_max+1)
    y = np.linspace(0, y_max, num=y_max+1)

    metric_all = []
    n = 40

    if coll:
        metric_row = [
            names_to_metric([f'sc_N1_NRW0_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)], metric_type, metric_thresh),
            names_to_metric([f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)], metric_type, metric_thresh),
            names_to_metric([f'sc_N3_NRW0_ND2_CNN14_FNN16_vis8_rep{x}' for x in range(n)], metric_type, metric_thresh),
            names_to_metric([f'sc_N4_NRW0_ND3_CNN14_FNN16_vis8_rep{x+40}' for x in range(n)], metric_type, metric_thresh), #-40 is sig higher
            names_to_metric([f'sc_N5_NRW0_ND4_CNN14_FNN16_vis8_rep{x}' for x in range(n)], metric_type, metric_thresh), #+40 is same
            names_to_metric([f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_rep{x+40}' for x in range(n)], metric_type, metric_thresh), #-40 is sig higher
        ]
        metric_all.append(metric_row)

        metric_row = [
            names_to_metric([f'sc_N2_NRW1_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)], metric_type, metric_thresh),
            names_to_metric([f'sc_N3_NRW1_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)], metric_type, metric_thresh),
            names_to_metric([f'sc_N4_NRW1_ND2_CNN14_FNN16_vis8_rep{x}' for x in range(n)], metric_type, metric_thresh),
            names_to_metric([f'sc_N5_NRW1_ND3_CNN14_FNN16_vis8_rep{x}' for x in range(n)], metric_type, metric_thresh),
            names_to_metric([f'sc_N6_NRW1_ND4_CNN14_FNN16_vis8_rep{x}' for x in range(n)], metric_type, metric_thresh),
            np.nan
        ]
        metric_all.append(metric_row)

        metric_row = [
            names_to_metric([f'sc_N3_NRW2_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)], metric_type, metric_thresh),
            names_to_metric([f'sc_N4_NRW2_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)], metric_type, metric_thresh),
            names_to_metric([f'sc_N5_NRW2_ND2_CNN14_FNN16_vis8_rep{x}' for x in range(n)], metric_type, metric_thresh),
            names_to_metric([f'sc_N6_NRW2_ND3_CNN14_FNN16_vis8_rep{x}' for x in range(n)], metric_type, metric_thresh), #+40 is same
            np.nan,
            np.nan
        ]
        metric_all.append(metric_row)

        metric_row = [
            names_to_metric([f'sc_N4_NRW3_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)], metric_type, metric_thresh),
            names_to_metric([f'sc_N5_NRW3_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)], metric_type, metric_thresh),
            names_to_metric([f'sc_N6_NRW3_ND2_CNN14_FNN16_vis8_rep{x}' for x in range(n)], metric_type, metric_thresh),
            np.nan,
            np.nan,
            np.nan
        ]
        metric_all.append(metric_row)

        metric_row = [
            names_to_metric([f'sc_N5_NRW4_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)], metric_type, metric_thresh),
            names_to_metric([f'sc_N6_NRW4_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)], metric_type, metric_thresh),
            np.nan,
            np.nan,
            np.nan,
            np.nan
        ]
        metric_all.append(metric_row)

        metric_row = [
            names_to_metric([f'sc_N6_NRW5_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)], metric_type, metric_thresh),
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan
        ]
        metric_all.append(metric_row)

    else:
        metric_row = [
            names_to_metric([f'sc_N1_NRW0_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)], metric_type, metric_thresh), # same as above
            names_to_metric([f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)], metric_type, metric_thresh),
            names_to_metric([f'sc_N3_NRW0_ND2_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)], metric_type, metric_thresh),
            names_to_metric([f'sc_N4_NRW0_ND3_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)], metric_type, metric_thresh),
            names_to_metric([f'sc_N5_NRW0_ND4_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)], metric_type, metric_thresh),
            names_to_metric([f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)], metric_type, metric_thresh),
        ]
        metric_all.append(metric_row)

        metric_row = [
            names_to_metric([f'sc_N2_NRW1_ND0_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)], metric_type, metric_thresh),
            names_to_metric([f'sc_N3_NRW1_ND1_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)], metric_type, metric_thresh),
            names_to_metric([f'sc_N4_NRW1_ND2_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)], metric_type, metric_thresh),
            names_to_metric([f'sc_N5_NRW1_ND3_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)], metric_type, metric_thresh),
            names_to_metric([f'sc_N6_NRW1_ND4_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)], metric_type, metric_thresh),
            np.nan
        ]
        metric_all.append(metric_row)

        metric_row = [
            names_to_metric([f'sc_N3_NRW2_ND0_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)], metric_type, metric_thresh),
            names_to_metric([f'sc_N4_NRW2_ND1_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)], metric_type, metric_thresh),
            names_to_metric([f'sc_N5_NRW2_ND2_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)], metric_type, metric_thresh),
            names_to_metric([f'sc_N6_NRW2_ND3_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)], metric_type, metric_thresh),
            np.nan,
            np.nan
        ]
        metric_all.append(metric_row)

        metric_row = [
            names_to_metric([f'sc_N4_NRW3_ND0_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)], metric_type, metric_thresh),
            names_to_metric([f'sc_N5_NRW3_ND1_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)], metric_type, metric_thresh),
            names_to_metric([f'sc_N6_NRW3_ND2_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)], metric_type, metric_thresh),
            np.nan,
            np.nan,
            np.nan
        ]
        metric_all.append(metric_row)

        metric_row = [
            names_to_metric([f'sc_N5_NRW4_ND0_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)], metric_type, metric_thresh),
            names_to_metric([f'sc_N6_NRW4_ND1_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)], metric_type, metric_thresh),
            np.nan,
            np.nan,
            np.nan,
            np.nan
        ]
        metric_all.append(metric_row)

        metric_row = [
            names_to_metric([f'sc_N6_NRW5_ND0_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)], metric_type, metric_thresh),
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan
        ]
        metric_all.append(metric_row)

    Z = np.asarray(metric_all)
    # print(Z.shape)

    if 'fit' in metric_type:
        min,max = 250,500
    elif 'JSspatial' in metric_type and metric_thresh is None:
        min,max = 0.05,0.15
    elif 'meanshift' in metric_type and metric_thresh is None:
        min,max = -50,300
        # min,max = 0,100
    elif 'learning_time' in metric_type:
        min,max = 50,350
    elif 'dirent' in metric_type:
        min,max = 0.2,0.8
    else:
        min,max = 0,0.6

    im = ax.pcolormesh(x, y, Z, 
                edgecolors='w', linewidths=0.5, 
                vmin=min, vmax=max,
                )

    if 'fit_og' in metric_type: label_type = 'Median of Means, OG'
    elif 'fit_nosoc' in metric_type: label_type = 'Median of Means, No-Social'
    elif 'fit_exploiter' in metric_type: label_type = 'Median of Means, Beacon-Exploiter'
    elif 'fit_explorer' in metric_type: label_type = 'Median of Means, Beacon-Explorer'
    elif 'meanshift_OGNS' in metric_type: label_type = f'Mean Shift, OG-NoSocial, Threshold @ {metric_thresh}'
    elif 'meanshift_NSET' in metric_type: label_type = f'Performance Difference' ###
    elif 'meanshift_NSER' in metric_type: label_type = f'Mean Shift, NoSocial-BeaconExplorer, Threshold @ {metric_thresh}'
    elif 'meanshift_ETER' in metric_type: label_type = f'Mean Shift, Beacon Exploiter-Explorer, Threshold @ {metric_thresh}'
    elif 'JS_soc' in metric_type: label_type = f'JS Divergence, OG-NoSocial, Threshold @ {metric_thresh}'
    elif 'JS_exp' in metric_type: label_type = f'JS Divergence, Beacon Exploiter-Explorer, Threshold @ {metric_thresh}'
    elif 'JSspatial_OGNS' in metric_type: label_type = f'Spatial JS Divergence, OG-NoSocial'
    elif 'JSspatial_NSET' in metric_type: label_type = f'Directional Divergence' ###
    elif 'JSspatial_NSER' in metric_type: label_type = f'Spatial JS Divergence, NoSocial-BeaconExplorer'
    elif 'JSspatial_ETER' in metric_type: label_type = f'Spatial JS Divergence, Beacon:Explorer-Exploiter'
    elif 'learning_time' in metric_type: label_type = f'Learning Time @ {metric_thresh}'
    elif 'dirent_OG' in metric_type: label_type = f'Directional Entropy, OG'
    elif 'dirent_NS' in metric_type: label_type = f'Directedness (No-Social)' ###
    elif 'dirent_ET' in metric_type: label_type = f'Directional Entropy, BeaconExploiter'
    elif 'dirent_ER' in metric_type: label_type = f'Directional Entropy, BeaconExplorer'
    elif 'meanshift_Nd+2' in metric_type: label_type = f'Mean Shift, OG-OG+Nd2'
    elif 'meanshift_Nr+2' in metric_type: label_type = f'Mean Shift, OG-OG+Nr2'
    # elif 'meanshift_SinitAg100-1' in metric_type: label_type = f'Mean Shift, OG-OG+Nd2'

    ax.set_xlabel('# Direct')
    ax.set_ylabel('# Random')
    

    if 'meanshift' in metric_type and metric_thresh is None:
        cbar = ax.figure.colorbar(im, ax=ax, label=label_type, extend='both')
    else:
        cbar = ax.figure.colorbar(im, ax=ax, label=label_type, extend='max')

    for i,row in enumerate(Z):
        for j,value in enumerate(row):
            if not np.isnan(value):
                if 'fit' in metric_type:
                    plt.text(j, i, int(value), ha='center', va='center', color='white')
                elif 'meanshift' in metric_type and metric_thresh is None:
                    plt.text(j, i, int(value), ha='center', va='center', color='white')
                else:
                    plt.text(j, i, round(value,2), ha='center', va='center', color='white')

    fig.tight_layout()
    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    if coll:
        plt.savefig(fr'{data_dir}/social_table_DR_{metric_type}_thresh{metric_thresh}_{dpi}.png', dpi=dpi)
    else:
        plt.savefig(fr'{data_dir}/social_table_DR_nocoll_{metric_type}_thresh{metric_thresh}_{dpi}.png', dpi=dpi)
    plt.close()



def names_to_metric(names, metric_type, metric_thresh, std=False):

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    bin_range = np.arange(0,1001,10)

    row = []
    for name in names:
        # print(name)

        if Path(fr'{data_dir}/{name}/val_matrix_best_nosocial_perturb.bin').is_file():
            with open(fr'{data_dir}/{name}/val_matrix_best.bin','rb') as f:
                data_og = pickle.load(f)
            with open(fr'{data_dir}/{name}/val_matrix_best_nosocial_perturb.bin','rb') as f:
                data_nosoc = pickle.load(f)
        else:
            # print(name)
            with open(fr'{data_dir}/{name}/val_matrix_best_ghostexploiter_perturb.bin','rb') as f:
                data_og = pickle.load(f)
            with open(fr'{data_dir}/{name}/val_matrix_best.bin','rb') as f:
                data_og = pickle.load(f)
            with open(fr'{data_dir}/{name}/val_matrix_best.bin','rb') as f:
                data_nosoc = pickle.load(f)
        with open(fr'{data_dir}/{name}/val_matrix_best_ghostexploiter_perturb.bin','rb') as f:
            data_exploiter = pickle.load(f)
        # with open(fr'{data_dir}/{name}/val_matrix_best_ghostexplorer_perturb.bin','rb') as f:
        #     data_explorer = pickle.load(f)

        # # skip poor performers
        # if np.mean(data_og) > 500:
        #     metric = None

        if 'og' in metric_type:
            metric = np.mean(data_og)
        elif 'nosoc' in metric_type:
            metric = np.mean(data_nosoc)
        elif 'exploiter' in metric_type:
            metric = np.mean(data_exploiter)
        elif 'explorer' in metric_type:
            metric = np.mean(data_explorer)

        elif 'shift_OGNS' in metric_type:
            metric = np.mean(data_nosoc) - np.mean(data_og)
        elif 'shift_NSET' in metric_type:
            metric = np.mean(data_nosoc) - np.mean(data_exploiter)
        elif 'shift_NSER' in metric_type:
            metric = np.mean(data_nosoc) - np.mean(data_explorer)
        elif 'shift_ETER' in metric_type:
            metric = np.mean(data_explorer) - np.mean(data_exploiter)
        elif 'shift_Nd+2' in metric_type:
            with open(fr'{data_dir}/{name}/val_matrix_best_Nd+2_perturb.bin','rb') as f:
                data_Nd2 = pickle.load(f)
            metric = np.mean(data_Nd2) - np.mean(data_og)
        elif 'shift_Nr+2' in metric_type:
            with open(fr'{data_dir}/{name}/val_matrix_best_Nr+2_perturb.bin','rb') as f:
                data_Nr2 = pickle.load(f)
            metric = np.mean(data_Nr2) - np.mean(data_og)
        elif 'shift_SinitAg100-1' in metric_type:
            with open(fr'{data_dir}/{name}/val_matrix_best_SinitAg100-1_perturb.bin','rb') as f:
                data_SinitAg = pickle.load(f)
            metric = np.mean(data_SinitAg) - np.mean(data_exploiter)

        elif 'norm_NSET' in metric_type:
            metric = (np.mean(data_nosoc) - np.mean(data_exploiter)) / np.mean(data_exploiter)

        elif 'JS_soc' in metric_type:
            h_og = np.histogram(data_og, bins=bin_range)[0]
            h_nosoc = np.histogram(data_nosoc, bins=bin_range)[0]
            metric = calc_JSdiv(h_og, h_nosoc)
        elif 'JS_exp' in metric_type:
            h_explorer = np.histogram(data_explorer, bins=bin_range)[0]
            h_exploiter = np.histogram(data_exploiter, bins=bin_range)[0]
            metric = calc_JSdiv(h_explorer, h_exploiter)

        elif 'JSspatial' in metric_type or 'dirent' in metric_type:
            # spatial_metric_list = [
            #         'de_mean_OG', 'de_mean_NS', 'de_mean_ET', 'de_mean_ER',
            #         'JS_mean_OGNS', 'JS_mean_NSET', 'JS_mean_NSER', 'JS_mean_ETER'
            #         ]
            if 'OGNS' in metric_type:
                index = 4
            elif 'NSET' in metric_type:
                index = 5
                # index = 10 # for social_extra
            elif 'NSER' in metric_type:
                index = 6
            elif 'ETER' in metric_type:
                index = 7
            elif 'OG' in metric_type:
                index = 0
            elif 'NS' in metric_type:
                index = 1
            elif 'ET' in metric_type:
                index = 2
            elif 'ER' in metric_type:
                index = 3

            with open(fr'{data_dir}/traj_matrices/gamut_social.bin', 'rb') as f:
            # with open(fr'{data_dir}/traj_matrices/gamut_social_extra.bin', 'rb') as f:
                data_dict = pickle.load(f)

            # print(name, metric_thresh, index, data_dict[name][index])
            # print(name, np.mean(data_nosoc), np.mean(data_exploiter), np.mean(data_nosoc) - np.mean(data_exploiter), data_dict[name][index])

            metric = data_dict[name][index]
            # if metric_thresh is None:
            #     metric = data_dict[name][index]
            # else:
            #     if np.mean(data_nosoc) - np.mean(data_exploiter) > metric_thresh:
            #         metric = data_dict[name][index]
            #     else:
            #         continue

        elif 'learning_time' in metric_type:
            with open(fr'{data_dir}/{name}/fitness_spread_per_generation.bin','rb') as f:
                data = pickle.load(f)

            data_genxpop = np.mean(data, axis=2)
            top_data = np.min(data_genxpop, axis=1)

            if np.min(top_data) <= metric_thresh:
                metric = int(np.argwhere(top_data <= metric_thresh)[0][0])
            else:
                metric = 1000

        elif 'distance' in metric_type:
            with open(fr'{data_dir}/{name}/val_matrix_best_nosocial-dist_perturb.bin','rb') as f:
                data = pickle.load(f)
            metric = np.mean(data)

        else:
            print(f'{metric_type} not valid metric type -type1')

        row.append(metric)

    # row = [x for x in metric_row if x is not None]
    row = np.asarray(row)
    if 'fit' in metric_type:
        return np.median(row)
    elif 'meanshift' in metric_type:
        if metric_thresh is not None:
            return (row > metric_thresh).sum()/len(row)
        else:
            return np.median(row)
    elif 'JS_' in metric_type:
        return (row > metric_thresh).sum()/len(row)
    elif 'dist' in metric_type:
        return row
    elif 'JSspatial' in metric_type:
        if metric_thresh is not None:
            return (row > metric_thresh).sum()/len(row)
        else:
            return np.median(row)
    elif 'learning_time' in metric_type:
        return np.median(row)
    elif 'dirent' in metric_type:
        return np.median(row)
    else:
        print(f'{metric_type} not valid metric type -type2y')



# ------------------------------- EA trends ---------------------------------------- #

def plot_mult_EA_trends(names, inter=False, val=None, group_est=None, order='min', scoring='time', num_agents=1, val_perturb=None, max=None, save_name=None):

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

        if not Path(fr'{data_dir}/{name}/fitness_spread_per_generation.bin').is_file():
            print(f'{data_dir}/{name}/fitness_spread_per_generation.bin is not a file')
            continue

        with open(fr'{data_dir}/{name}/fitness_spread_per_generation.bin','rb') as f:
            data = pickle.load(f)

        data_genxpop = np.mean(data, axis=2)
        if inter: data_genxpop = np.ma.masked_equal(data_genxpop, 0)

        # score per agent
        data_genxpop /= num_agents

        if order == 'min':
            top_data = np.min(data_genxpop, axis=1) # min : top
            top_ind = np.argsort(top_data)[:3] # min : top
        elif order == 'max':
            top_data = np.max(data_genxpop, axis=1) # max : top
            top_ind = np.argsort(top_data)[-1:-4:-1] # max : top
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

        if top_data.shape[0] > 1000:
            top_data = top_data[-1000:]

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

                    # print(i, val_data[:,2])

                    # score per agent
                    val_data[:,1:] /= num_agents

                    if order == 'min':
                        top_ind = np.argsort(val_data[:,2])[:3] # min : top
                    elif order == 'max':
                        top_ind = np.argsort(val_data[:,2])[-1:-4:-1] # max : top

                    top_gen = [val_data[i,0] for i in top_ind]
                    top_valfit = [val_data[i,2] for i in top_ind]
                    for g,f in zip(top_gen, top_valfit):
                        print(f'val | gen {int(g)}: fit {int(f)}')

                    top_vals_current = np.array(([i], [top_gen[0]], [top_valfit[0]]))
                    top_vals_overall = np.hstack((top_vals_overall, top_vals_current))

                val_diff = np.mean((val_data[:,2] - val_data[:,1])**2)
                print(f'mean sq val diff: {int(val_diff)}')
                val_diffs.append(val_diff)

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

    
        if val_perturb is not None:
            filename = 'val_matrix_cen'+'_'+val_perturb+'_perturb'

            if Path(fr'{data_dir}/{name}/{filename}.bin').is_file():
                with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                    data = pickle.load(f)
                data /= num_agents
                
                top_perturb = np.mean(data[top_ind,:], axis=1)
                for g,f in zip(top_gen, top_perturb):
                    print(f'{val_perturb} | gen {int(g)}: fit {int(f)}')

                avg_perturb = np.mean(data)
                ax1.hlines(avg_perturb, i*5, data_genxpop.shape[0] + i*5,
                        color=cmap(i/cmap_range),
                        linestyle='solid',
                        alpha=0.5
                        )
            else:
                print(fr'{data_dir}/{name}/{filename}.txt is not a file')
    
    group_top = np.array(group_top)
    if group_est == 'mean':
        est_trend = np.mean(group_top, axis=0)
        lt = ax1.plot(est_trend, 
                        label = f'{group_est} of group top',
                        color='k', 
                        alpha=.5
                        )
        lns.append(lt[0])
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

    labs = [l.get_label() for l in lns]
    # ax1.legend(lns, labs, loc='upper right')
    ax1.legend(lns, labs, loc='lower right')
    # ax1.legend(lns, labs, loc='upper left')

    if scoring == 'time':
        ax1.set_ylabel('Time to Find Patch')
    elif scoring == 'res':
        ax1.set_ylabel('Resources Collected per Agent')
    # ax1.set_ylim(-20,1520)
    if max is not None:
        ax1.set_ylim(-20,max)

    # ax1.set_ylabel('# Patches Found')
    # ax1.set_ylim(0,8)

    if val is not None:
        top_val_inds = np.argsort(top_vals_overall[2,:])
        top_reps = top_vals_overall[0,:][top_val_inds]
        top_gens = top_vals_overall[1,:][top_val_inds]
        top_vals = top_vals_overall[2,:][top_val_inds]
        
        # ax1.set_title(f'overall validated run avg: {int(np.mean(val_avgs))} | val var: {int((np.mean(val_diffs))**.5)} | top val run: {int(top_vals[0])}')

        top_num = 50
        if order == 'min':
            for rep, gen, val_fit in zip(top_reps[:top_num], top_gens[:top_num], top_vals[:top_num]):
                print(f'overall val | rep {names[int(rep)]} | gen {int(gen)} | fit {int(val_fit)}')
        elif order == 'max':
            for rep, gen, val_fit in zip(top_reps[-1:-top_num-1:-1], top_gens[-1:-top_num-1:-1], top_vals[-1:-top_num-1:-1]):
                print(f'overall val | rep {names[int(rep)]} | gen {int(gen)} | fit {int(val_fit)}')

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


def plot_mult_EA_trends_groups(groups, inter=False, val=None, group_est='mean', order='min', scoring='dist', num_agents=1, val_perturb=None, max=None, save_name=None):

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
        val_perturb_stack = np.zeros(num_runs)

        for r_num, name in enumerate(run_names):

            if not Path(fr'{data_dir}/{name}/fitness_spread_per_generation.bin').is_file():
                print(f'{data_dir}/{name}/fitness_spread_per_generation.bin is not a file')
                continue

            with open(fr'{data_dir}/{name}/fitness_spread_per_generation.bin','rb') as f:
                data = pickle.load(f)

            data_genxpop = np.mean(data, axis=2)
            if inter: data_genxpop = np.ma.masked_equal(data_genxpop, 0)

            # score per agent
            data_genxpop /= num_agents

            if order == 'min':
                top_data = np.min(data_genxpop, axis=1) # min : top
            elif order == 'max':
                top_data = np.max(data_genxpop, axis=1) # max : top
            top_stack[r_num,:] = top_data

            avg_trend_data = np.mean(data_genxpop, axis=1)
            avg_stack[r_num,:] = avg_trend_data

            if val is not None:
                if val == 'top': filename = 'val_matrix'
                elif val == 'cen': filename = 'val_matrix_cen'
                if 'ghost' in name: filename = 'val_matrix_cen_ghostexploiter_perturb' # override for these guys

                with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                    data = pickle.load(f)

                avg = np.mean(data)
                avg /= num_agents
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

                    # score per agent
                    val_data[:,1:] /= num_agents

                    # avg_val = np.mean(val_data[:,2])
                    # val_stack[r_num] = avg_val

                    # find top val fitness
                    if order == 'min':
                        top_ind = np.argsort(val_data[:,2])[0] # min : top
                    elif order == 'max':
                        top_ind = np.argsort(val_data[:,2])[-1] # max : top
                    top_gen = int(val_data[top_ind,0])
                    top_valfit = int(val_data[top_ind,2])
                    # if top_valfit < 500:
                    print(f'{name}, gen {top_gen}: fit {top_valfit}')


            if val_perturb is not None:
                filename = 'val_matrix_cen'+'_'+val_perturb+'_perturb'
                with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                    data = pickle.load(f)

                data /= num_agents
                avg = np.mean(data)
                val_perturb_stack[r_num] = avg

                top_perturb = np.mean(data[top_ind,:])
                print(f'{name}, gen {top_gen}: {val_perturb} perturb {top_valfit}')

            # la = ax1.plot(top_data, 
            #                 # label = f'{group_name}: top individual (avg of {num_runs} runs)',
            #                 label = f'{group_name}',
            #                 color=cmap(g_num/cmap_range), 
            #                 alpha=.01,
            #                 linewidth=.1,
            #                 )
            # lns.append(la[0])

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
                val_group = np.mean(val_stack)
            elif group_est == 'median':
                val_group = np.median(val_stack)
            print(f'{group_name} | avg val: {int(val_group)}')
            if val_group != 0:
                ax1.hlines(val_group, g_num*5, data_genxpop.shape[0] + g_num*5,
                        color=cmap(g_num/cmap_range),
                        linestyle='dashed',
                        alpha=0.5
                        )

        if val_perturb is not None:
            if group_est == 'mean':
                val_perturb_group = np.mean(val_perturb_stack)
            elif group_est == 'median':
                val_perturb_group = np.median(val_perturb_stack)
            print(f'{group_name} | avg {val_perturb} perturb: {int(val_perturb_group)}')
            if val_perturb_group != 0:
                ax1.hlines(val_perturb_group, g_num*5, data_genxpop.shape[0] + g_num*5,
                        color=cmap(g_num/cmap_range),
                        linestyle='solid',
                        alpha=0.5
                        )

    ax1.set_xlabel('Generation')

    labs = [l.get_label() for l in lns]
    # ax1.legend(lns, labs, loc='upper right')
    ax1.legend(lns, labs, loc='lower right')

    # ax1.set_ylabel('Performance')
    if scoring == 'time':
        ax1.set_ylabel('Time to Find Patch')
    elif scoring == 'res':
        ax1.set_ylabel('Resources Collected per Agent')
    
    if max is not None:
        ax1.set_ylim(-20,max)

    if save_name: 
        # plt.savefig(fr'{data_dir}/{save_name}.png')
        plt.savefig(fr'{data_dir}/{save_name}.png', dpi=100)
        # plt.savefig(fr'{data_dir}/{save_name}_500.png', dpi=500)
    plt.show()


def plot_mult_EA_trends_groups_endonly(groups, val=None, scoring='dist', num_agents=1, max=None, bees=False, trunc=False, title=None, save_name=None):

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
            if 'ghost' in name: filename = 'val_matrix_cen_ghostexploiter_perturb' # override for these guys

            with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                data = pickle.load(f)
            data /= num_agents
            
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
    
        data = np.array(data_group).flatten()
        l0 = ax1.violinplot(data, 
                    positions=[g_num],
                    widths=1, # KDE plot area proportional to navigator ratio 
                    showmedians=True, 
                    showextrema=False,
                    # bw_method='silverman',
                    # bw_method=.25,
                    )
        for part in l0["bodies"]:
            part.set_edgecolor(cmap(g_num/cmap_range))
            part.set_facecolor(cmap(g_num/cmap_range))
        l0["cmedians"].set_edgecolor(cmap(g_num/cmap_range))
        # color = l0["bodies"][0].get_facecolor().flatten()
        # violin_labs.append((mpatches.Patch(color=color), group_name))
        # violin_labs.append((mpatches.Patch(color=color), labs[g_num]))
        print(f'{group_name}: {int(np.mean(data))}')

    # ax1.legend(*zip(*violin_labs), loc='upper left')
    # ax1.legend(*zip(*violin_labs), bbox_to_anchor=(1.1, 1.05))
    labs = [group_name for group_name,_ in groups]

    ax1.set_xticks(np.linspace(0,len(groups)-1,len(groups)))
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
    # ax1.set_xticklabels([fr'{lab}({rat})' for lab,rat in zip(labs, ratio_missed)])
    ax1.set_xticklabels(labs)
    # ax1.set_xticks([])
    if scoring == 'dist':
        ax1.set_ylabel('Time to Find Patch')
    elif scoring == 'res':
        ax1.set_ylabel('Resources Collected per Agent')
    
    if max is not None:
        ax1.set_ylim(-20,max)
    
    if title is not None:
        ax1.set_title(title)

    if save_name: 
        # plt.savefig(fr'{data_dir}/{save_name}.png')
        plt.savefig(fr'{data_dir}/{save_name}.png', dpi=100)
    plt.show()



def plot_mult_EA_trends_groups_endonly_split(groups, val=None, scoring='dist', num_agents=1, max=None, bees=False, trunc=False, title=None, save_name=None):

    # establish load directory
    root_dir = Path(__file__).parent.parent
    data_dir = Path(root_dir, r'data/simulation_data')

    # # init plot details
    # fig, ax1 = plt.subplots(figsize=(15,10)) 
    fig, ax1 = plt.subplots(figsize=(6,4)) 
    cmap = plt.get_cmap('plasma')
    cmap_range = len(groups)
    violin_labs = []
    ratio_missed = []
    
    # iterate over each file
    for g_num, (group_name, run_names) in enumerate(groups):

        data_group = []        
        for r_num, name in enumerate(run_names):

            if val == 'top': filename = 'val_matrix'
            elif val == 'cen': filename = 'val_matrix_cen'
            if 'ghost' in name: filename = 'val_matrix_cen_ghostexploiter_perturb' # override for these guys

            with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                data = pickle.load(f)
            data /= num_agents
            
            data_group.append(data.flatten())
    
        data = np.array(data_group).flatten()
        median = np.median(data)
        ratio = np.round( np.count_nonzero(data==1000)/np.size(data), 2)
        # ratio_missed.append(ratio)
        ax1.text(g_num-.27, 1005, 1-ratio, size=10)

        if bees:
            data_bees = data.flatten()
            # randomly strip
            strip_count = len(data_bees)*.8
            data_bees = np.delete(data_bees, np.random.choice(len(data_bees), int(strip_count), replace=False))

            data_below = np.delete(data_bees, np.argwhere(data_bees == 1000))
            x = beeswarm(data_below, scaling=2)
            ax1.scatter(g_num + x, data_below, color=cmap(g_num/cmap_range), alpha=1/255)

            data_edge = np.delete(data_bees, np.argwhere(data_bees < 1000))
            from scipy.stats import truncnorm # jitter
            a_trunc, b_trunc, loc, scale = 0, 1000, 1000, 250*ratio
            a, b = (a_trunc - loc) / scale, (b_trunc - loc) / scale
            rv = truncnorm(a,b,loc,scale)
            data_edge = rv.rvs(len(data_edge))
            x = beeswarm(data_edge, scaling=2)
            ax1.scatter(g_num + x, data_edge, color=cmap(g_num/cmap_range), alpha=1/255)

            print(len(data.flatten()), len(data_bees), len(data_below), len(data_edge))

        if trunc:
            data = np.delete(data, np.argwhere(data == 1000))

            # circle represents proportion missed
            # r = np.sqrt(ratio*2 / np.pi) # area (2 x KDE) to radius
            # ell = mpatches.Ellipse((g_num, 1150), width=r, height=r*200, angle=0, color=cmap(g_num/cmap_range), alpha=.3)
            # ax1.add_patch(ell)

        # fit area to proportion found by iterating width
        # from shapely.geometry import Polygon
        # target_area = 400 * (1-ratio)
        # width = 1-ratio # init guess
        # error = 51 # init above
        # l0 = None
        # while error > 25:
        #     # print(width, error)

        #     if l0 is not None:
        #         # overwrite by fading out previous
        #         for part in l0["bodies"]:
        #             part.set_alpha(0)
        #         l0["cmedians"].set_alpha(0)

        #     l0 = ax1.violinplot(data, 
        #                 positions=[g_num],
        #                 widths=width, # KDE plot area proportional to navigator ratio 
        #                 showmedians=True, 
        #                 showextrema=False,
        #                 # bw_method='silverman',
        #                 # bw_method=.25,
        #                 )

        #     paths = l0["bodies"][0].get_paths()
        #     area = Polygon(paths[0].vertices[:-1]).area
        #     error = target_area - area
        #     if error > 0: width += .025
        #     else: width -= .025
        #     error = abs(error)
        # print(width, error, int(area), 1-ratio, r)

        l0 = ax1.violinplot(data, 
                    positions=[g_num],
                    widths=1-ratio, # KDE width proportional to navigator ratio 
                    showmedians=True, 
                    showextrema=False,
                    # bw_method='silverman',
                    # bw_method=.25,
                    )

        for part in l0["bodies"]:
            part.set_edgecolor(cmap(g_num/cmap_range))
            part.set_facecolor(cmap(g_num/cmap_range))
        l0["cmedians"].set_edgecolor(cmap(g_num/cmap_range))

        # shift median line to account for those that did not make it to patch (deleted data)
        median_segment = l0["cmedians"].get_segments()[0]
        median_segment[:,0] = g_num - .5/2, g_num + .5/2
        median_segment[:,-1] = median,median
        l0["cmedians"].set_segments([median_segment])

        print(f'{group_name}: {int(np.mean(data))}')

    labs = [group_name for group_name,_ in groups]
    ax1.set_xticks(np.linspace(0,len(groups)-1,len(groups)))
    ax1.set_xticklabels(labs)
    # ax1.set_yticks([0,200,400,600,800,1000,1150])
    # ax1.set_yticklabels([0,200,400,600,800,1000,'Prop Found'])
    ax1.set_yticks([0,200,400,600,800,1000])
    ax1.set_yticklabels([0,200,400,600,800,1000])
    if scoring == 'dist':
        ax1.set_ylabel('Time to Find Patch')
    elif scoring == 'res':
        ax1.set_ylabel('Resources Collected per Agent')
    
    if max is not None:
        ax1.set_ylim(-20,max)
    
    if title is not None:
        ax1.set_title(title)

    if save_name: 
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


def plot_mult_EA_trends_randomwalk(run_names, social=False, bees=False, norm_outside=False, clean_outside=False, save_name=None):

    # establish load directory
    root_dir = Path(__file__).parent.parent
    data_dir = Path(root_dir, r'data/simulation_data/nonNN')

    # init plot details
    fig, ax1 = plt.subplots(figsize=(7,4)) 
    # cmap = plt.get_cmap('hsv')
    # cmap_range = len(run_names)
    # violin_labs = []

    diff_coeffs = [0.0005,0.001,0.005,0.01,0.05,0.1,0.5]
    N_RWs = [1,5,10,15,20]
    N_RWs = [
        '1 Rand',
        '5 Rand',
        '10 Rand',
        '15 Rand',
        '20 Rand',
    ]

    plt.xticks(np.arange(0, len(run_names)+2, 1))
    fig.canvas.draw()
    labels = [item.get_text() for item in ax1.get_xticklabels()]

    # iterate over each file
    for r_num, name in enumerate(run_names):

        with open(fr'{data_dir}/{name}.bin','rb') as f:
            data = pickle.load(f)
        data = data[0,:] # remove copies

        if clean_outside:
            data = np.delete(data, np.argwhere(data == 1000)) # clean array

        if bees:
            data_bees = data.flatten()

            if norm_outside:
                num_outside = len(np.argwhere(data_bees == 1000))
                data_bees = np.delete(data_bees, np.argwhere(data_bees == 1000)) # clean array
                
                # build array of normal distributed noise for num outside
                stdev = 50
                noise = np.random.randn(num_outside)*stdev + 1000
                data_bees = np.append(data_bees, noise)
                # print(data_bees.shape)

            x = beeswarm(data_bees)
            ax1.scatter(r_num + x, data_bees, alpha=.4)
            # sns.swarmplot(data=data.flatten(), ax=ax1)

        l0 = ax1.violinplot(data.flatten(), 
                    positions=[r_num],
                    widths=1, 
                    showmeans=True, 
                    showmedians=True, 
                    showextrema=False,
                    )
        l0['cmeans'].set_linestyle('dashed')
        l0['cmedians'].set_edgecolor('black')

        if social:
            labels[r_num] = N_RWs[r_num]
        else:
            labels[r_num] = diff_coeffs[r_num]
    
    if social:

        # also add perfect traj
        with open(fr'{data_dir}/perfect.bin','rb') as f:
            data = pickle.load(f)
        print(f'average direct performance: {np.mean(data.flatten())}')

        if bees:
            data_bees = data.flatten()
            x = beeswarm(data_bees)
            ax1.scatter(r_num+1 + x, data_bees, alpha=.4)

        l0 = ax1.violinplot(data.flatten(), 
                    positions=[r_num+1],
                    widths=1, 
                    showmeans=True, 
                    showmedians=True, 
                    showextrema=False,
                    )
        l0['cmeans'].set_linestyle('dashed')
        l0['cmedians'].set_edgecolor('black')
        labels[r_num+1] = '1 Direct'

        # also add trained agent
        data_dir = Path(root_dir, r'data/simulation_data')

        # # all agents (2.5 inits for 40 agents = 100, round up to 3 each for 120)
        # num_inits = 3
        # data_all = []
        # perfs = []
        # for name in [f'sc_N1_NRW0_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(40)]:
        #     with open(fr'{data_dir}/{name}/val_matrix_best.bin','rb') as f:
        #         data = pickle.load(f)
        #     print(name, np.mean(data))
        #     perfs.append(np.mean(data))
        #     data_all.append(data[:3])
        # data = np.array(data_all)
        # print(np.array(perfs))
        # print(np.median(np.array(perfs)))

        # # only best
        # name = 'sc_N1_NRW0_ND0_CNN14_FNN16_vis8_rep17'
        # with open(fr'{data_dir}/{name}/val_matrix_best.bin','rb') as f:
        #     data = pickle.load(f)
        # data = data[:100]
        # print(f'average trained-best performance: {np.mean(data.flatten())}')

        # only median
        name = 'sc_N1_NRW0_ND0_CNN14_FNN16_vis8_rep28'
        with open(fr'{data_dir}/{name}/val_matrix_best.bin','rb') as f:
            data = pickle.load(f)
        data = data[:100]
        print(f'average trained-best performance: {np.mean(data.flatten())}')

        if bees:
            data_bees = data.flatten()
            x = beeswarm(data_bees)
            ax1.scatter(r_num+2 + x, data_bees, alpha=.4)

        l0 = ax1.violinplot(data.flatten(), 
                    positions=[r_num+2],
                    widths=1, 
                    showmeans=True, 
                    showmedians=True, 
                    showextrema=False,
                    )
        l0['cmeans'].set_linestyle('dashed')
        l0['cmedians'].set_edgecolor('black')
        labels[r_num+2] = '1 Trained'

    ax1.set_xticklabels(labels)
    ax1.set_ylabel('Time (for the First Agent) to Find Patch')

    if social:
        # ax1.set_xlabel('Number of Random Walkers')
        data_dir = Path(root_dir, r'data/simulation_data/nonNN')
    else:
        ax1.set_xlabel('Rotational Diffusion Coefficient')


    plt.savefig(fr'{data_dir}/random{save_name}.png')
    plt.show()
    plt.close()


# ------------------------------- social specific ---------------------------------------- #


def plot_mult_EA_trends_groups_endonly_social_rand(groups, max=None, bees=False, save_name=None):

    # establish load directory
    root_dir = Path(__file__).parent.parent
    data_dir = Path(root_dir, r'data/simulation_data')

    # # init plot details
    fig, ax1 = plt.subplots(figsize=(6,4)) 
    cmap = plt.get_cmap('plasma')
    cmap_range = len(groups)
    violin_labs = []
    
    # iterate over each file
    for g_num, (group_name, run_names) in enumerate(groups):

        data_group = []
        data_group_allRW = []
        data_group_RWp1 = []
        data_group_allD = []
        data_group_Dp1 = []
        data_group_nosocial = []
        data_group_selfsocial = []
        data_group_ghostexploiter = []
        data_group_ghostexplorer = []
        data_group_N2exploitexploiter = []
        data_group_N2exploitexplorer = []
        for r_num, name in enumerate(run_names):

            filename = 'val_matrix_cen'

            with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                dataOG = pickle.load(f)
            data_group.append(dataOG.flatten())

            # also load perturb data
            if 'RWp1' in save_name:
                filename = 'val_matrix_cen_RWp1_perturb'
                if Path(fr'{data_dir}/{name}/{filename}.bin').is_file():
                    with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                        data = pickle.load(f)
                    data_group_RWp1.append(data.flatten())
            if 'Dp1' in save_name:
                filename = 'val_matrix_cen_Dp1_perturb'
                if Path(fr'{data_dir}/{name}/{filename}.bin').is_file():
                    with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                        data = pickle.load(f)
                    data_group_Dp1.append(data.flatten())
            if 'allRW' in save_name:
                filename = 'val_matrix_cen_allRW_perturb'
                if Path(fr'{data_dir}/{name}/{filename}.bin').is_file():
                    with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                        data = pickle.load(f)
                    data_group_allRW.append(data.flatten())
            if 'allD' in save_name:
                filename = 'val_matrix_cen_allD_perturb'
                if Path(fr'{data_dir}/{name}/{filename}.bin').is_file():
                    with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                        data = pickle.load(f)
                    data_group_allD.append(data.flatten())
            if 'nosocial' in save_name:
                filename = 'val_matrix_cen_nosocial_perturb'
                if Path(fr'{data_dir}/{name}/{filename}.bin').is_file():
                    with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                        data = pickle.load(f)
                    data_group_nosocial.append(data.flatten())
            if 'selfsocial' in save_name:
                filename = 'val_matrix_cen_selfsocial_perturb'
                if Path(fr'{data_dir}/{name}/{filename}.bin').is_file():
                    with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                        data = pickle.load(f)
                    data_group_selfsocial.append(data.flatten())
            if '_ghost' in save_name:
                filename = 'val_matrix_cen_ghostexploiter_perturb'
                if Path(fr'{data_dir}/{name}/{filename}.bin').is_file():
                    with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                        data1 = pickle.load(f)
                    data_group_ghostexploiter.append(data1.flatten())
                filename = 'val_matrix_cen_ghostexplorer_perturb'
                if Path(fr'{data_dir}/{name}/{filename}.bin').is_file():
                    with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                        data = pickle.load(f)
                    data_group_ghostexplorer.append(data.flatten())
            if 'N2exploit' in save_name:
                filename = 'val_matrix_cen_N2-ghostexploiter_perturb'
                if Path(fr'{data_dir}/{name}/{filename}.bin').is_file():
                    with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                        data = pickle.load(f)
                    data_group_N2exploitexploiter.append(data.flatten())
                filename = 'val_matrix_cen_N2-ghostexplorer_perturb'
                if Path(fr'{data_dir}/{name}/{filename}.bin').is_file():
                    with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                        data = pickle.load(f)
                    data_group_N2exploitexplorer.append(data.flatten())
            
            # print(f'{g_num}: {r_num}: {int(np.median(dataOG))} | {int(np.median(data1))} | {int(np.median(data))} || {(int(np.median(dataOG)) - int(np.median(data))) - (int(np.median(dataOG)) - int(np.median(data1)))}')

        data = np.array(data_group)
        if bees:
            data = data.flatten()
            data = np.delete(data, np.argwhere(data == 1000))
            x = beeswarm(data)
            ax1.scatter(g_num+x, data, color=cmap(g_num/cmap_range), alpha=1/255)
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

        # perturbs
        if data_group_allRW:
            data = np.array(data_group_allRW)
            if bees:
                data = data.flatten()
                data = np.delete(data, np.argwhere(data == 1000))
                x = beeswarm(data)
                ax1.scatter(g_num+.25+x, data, color=cmap(g_num/cmap_range), alpha=1/255)
            l0 = ax1.violinplot(data.flatten(), 
                        positions=[g_num+.25],
                        widths=.5, 
                        showmedians=True, 
                        showextrema=False,
                        )
            for part in l0["bodies"]:
                part.set_edgecolor(cmap(g_num/cmap_range))
                part.set_facecolor(cmap(g_num/cmap_range))
                part.set_alpha(.7)
            l0["cmedians"].set_edgecolor(cmap(g_num/cmap_range))
            l0["cmedians"].set_alpha(.7)

        if data_group_allD:
            data = np.array(data_group_allD)
            if bees:
                data = data.flatten()
                data = np.delete(data, np.argwhere(data == 1000))
                x = beeswarm(data)
                ax1.scatter(g_num+.25+x, data, color=cmap(g_num/cmap_range), alpha=1/255)
            l0 = ax1.violinplot(data.flatten(), 
                        positions=[g_num+.25],
                        widths=.5, 
                        showmedians=True, 
                        showextrema=False,
                        )
            for part in l0["bodies"]:
                part.set_edgecolor(cmap(g_num/cmap_range))
                part.set_facecolor(cmap(g_num/cmap_range))
                part.set_alpha(.7)
            l0["cmedians"].set_edgecolor(cmap(g_num/cmap_range))
            l0["cmedians"].set_alpha(.7)

        if data_group_selfsocial:
            data = np.array(data_group_selfsocial)
            if bees:
                data = data.flatten()
                data = np.delete(data, np.argwhere(data == 1000))
                x = beeswarm(data)
                ax1.scatter(g_num+.25+x, data, color=cmap(g_num/cmap_range), alpha=1/255)
            l0 = ax1.violinplot(data.flatten(), 
                        positions=[g_num+.25],
                        widths=.5, 
                        showmedians=True, 
                        showextrema=False,
                        )
            for part in l0["bodies"]:
                part.set_edgecolor(cmap(g_num/cmap_range))
                part.set_facecolor(cmap(g_num/cmap_range))
                part.set_alpha(.7)
            l0["cmedians"].set_edgecolor(cmap(g_num/cmap_range))
            l0["cmedians"].set_alpha(.7)

        if data_group_nosocial:
            data = np.array(data_group_nosocial)
            if bees:
                data = data.flatten()
                data = np.delete(data, np.argwhere(data == 1000))
                x = beeswarm(data)
                ax1.scatter(g_num+.25+x, data, color=cmap(g_num/cmap_range), alpha=1/255)
            l0 = ax1.violinplot(data.flatten(), 
                        positions=[g_num+.25],
                        widths=.5, 
                        showmedians=True, 
                        showextrema=False,
                        )
            for part in l0["bodies"]:
                part.set_edgecolor(cmap(g_num/cmap_range))
                part.set_facecolor(cmap(g_num/cmap_range))
                part.set_alpha(.7)
            l0["cmedians"].set_edgecolor(cmap(g_num/cmap_range))
            l0["cmedians"].set_alpha(.7)

        if data_group_ghostexplorer:
            data = np.array(data_group_ghostexplorer)
            l0 = ax1.violinplot(data.flatten(), 
                        positions=[g_num+.25],
                        widths=.5, 
                        showmedians=True, 
                        showextrema=False,
                        )
            for part in l0["bodies"]:
                part.set_edgecolor(cmap(g_num/cmap_range))
                part.set_facecolor(cmap(g_num/cmap_range))
                part.set_alpha(.6)
            l0["cmedians"].set_edgecolor(cmap(g_num/cmap_range))
            l0["cmedians"].set_alpha(.6)
        if data_group_ghostexploiter:
            data = np.array(data_group_ghostexploiter)
            l0 = ax1.violinplot(data.flatten(), 
                        positions=[g_num+.5],
                        widths=.5, 
                        showmedians=True, 
                        showextrema=False,
                        )
            for part in l0["bodies"]:
                part.set_edgecolor(cmap(g_num/cmap_range))
                part.set_facecolor(cmap(g_num/cmap_range))
                part.set_alpha(.7)
            l0["cmedians"].set_edgecolor(cmap(g_num/cmap_range))
            l0["cmedians"].set_alpha(.7)

        if data_group_N2exploitexplorer:
            data = np.array(data_group_N2exploitexplorer)
            l0 = ax1.violinplot(data.flatten(), 
                        positions=[g_num+.25],
                        widths=.5, 
                        showmedians=True, 
                        showextrema=False,
                        )
            for part in l0["bodies"]:
                part.set_edgecolor(cmap(g_num/cmap_range))
                part.set_facecolor(cmap(g_num/cmap_range))
                part.set_alpha(.6)
            l0["cmedians"].set_edgecolor(cmap(g_num/cmap_range))
            l0["cmedians"].set_alpha(.6)
        if data_group_N2exploitexploiter:
            data = np.array(data_group_N2exploitexploiter)
            l0 = ax1.violinplot(data.flatten(), 
                        positions=[g_num+.5],
                        widths=.5, 
                        showmedians=True, 
                        showextrema=False,
                        )
            for part in l0["bodies"]:
                part.set_edgecolor(cmap(g_num/cmap_range))
                part.set_facecolor(cmap(g_num/cmap_range))
                part.set_alpha(.7)
            l0["cmedians"].set_edgecolor(cmap(g_num/cmap_range))
            l0["cmedians"].set_alpha(.7)


        # print(f'{group_name}: {int(np.mean(data))} / {int(np.mean(data_allRW))} / {int(np.mean(data_nosocial))}')
        # print(f'{group_name}: {int(np.mean(data))} / {int(np.mean(data_allRW))}')
        # print(f'{group_name}: {int(np.mean(data))} / {int(np.mean(data_nosocial))}')
        # print(f'{group_name}: {int(np.mean(data))}')

    # ax1.legend(*zip(*violin_labs), loc='upper left')
    # ax1.legend(*zip(*violin_labs), bbox_to_anchor=(1.1, 1.05))
    # ax1.set_xticks([])
    
    ax1.set_xticks(np.linspace(0,len(groups)-1,len(groups)))
    # labs = [group_name for group_name,_ in groups]
    if 'N6' in save_name:
        labs = [0,1,2,3,4,5]
    elif 'N3' in save_name:
        labs = [0,1,2]
    elif 'N11' in save_name:
        labs = [0,5,10]
    elif 'N21' in save_name:
        labs = [0,10,20]
    elif 'NX' in save_name:
        labs = [0,1,2,3,4,5,10,20]

    if 'N3' in save_name or 'N6' in save_name or 'N11' in save_name or 'N21' in save_name:
        ax1.set_xlabel('# Random Other Agents (& n-1 # Direct)')
    elif 'NX' in save_name:
        ax1.set_xlabel('# Other Agents (All Direct)')
    ax1.set_xticklabels(labs)

    ax1.set_ylabel('Time Taken to Find Patch')
    if max is not None:
        ax1.set_ylim(-20,max)

    ax1.set_title('Light: as trained | Dark: no others')
    # ax1.set_title('Light: as trained | Dark: all RW perturb')
    # ax1.set_title('Light: as trained | Dark: all D perturb')
    # ax1.set_title('Light: as trained | Med: all RW perturb | Dark: no others perturb')
    # ax1.set_title('Light: as trained | Med: all D perturb | Dark: no others perturb')
    # ax1.set_title('Light: as trained | Dark: others=self')
    # ax1.set_title('Light: as trained | Med: others=self | Dark: no others')
    # ax1.set_title('Light: as trained | Dark: other agents are self')
    # ax1.set_title('Light: as trained | LM: all RW | MD: others=self | Dark: no others')
    # ax1.set_title('Light: as trained | Med: ghost-explorer | Dark: ghost-exploiter')
    # ax1.set_title('Light: as trained | Med: N=2 ghost-explorer | Dark: N=2 ghost-exploiter')

    if save_name: 
        # plt.savefig(fr'{data_dir}/{save_name}.png')
        plt.savefig(fr'{data_dir}/{save_name}.png', dpi=100)
    plt.show()


def plot_mult_EA_trends_groups_endonly_social_rand_indivperturbs(groups, max=None, title=None, save_name=None):

    # establish load directory
    root_dir = Path(__file__).parent.parent
    data_dir = Path(root_dir, r'data/simulation_data')

    # # init plot details
    fig, ax1 = plt.subplots(figsize=(6,4)) 

    # iterate over each file
    data_all = []
    for g_num, (group_name, run_names) in enumerate(groups):

        data_group = []
        for r_num, name in enumerate(run_names):

            data_indiv = []

            filename = 'val_matrix_cen'
            with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                data = pickle.load(f)
            data_indiv.append(np.median(data))
            # filename = 'val_matrix_cen_RWp1_perturb'
            # if Path(fr'{data_dir}/{name}/{filename}.bin').is_file():
            #     with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
            #         data = pickle.load(f)
            #     data_indiv.append(np.median(data))
            # filename = 'val_matrix_cen_Dp1_perturb'
            # if Path(fr'{data_dir}/{name}/{filename}.bin').is_file():
            #     with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
            #         data = pickle.load(f)
            #     data_indiv.append(np.median(data))
            filename = 'val_matrix_cen_allRW_perturb'
            if Path(fr'{data_dir}/{name}/{filename}.bin').is_file():
                with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                    data = pickle.load(f)
                data_indiv.append(np.median(data))
            filename = 'val_matrix_cen_allD_perturb'
            if Path(fr'{data_dir}/{name}/{filename}.bin').is_file():
                with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                    data = pickle.load(f)
                data_indiv.append(np.median(data))
            filename = 'val_matrix_cen_nosocial_perturb'
            if Path(fr'{data_dir}/{name}/{filename}.bin').is_file():
                with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                    data = pickle.load(f)
                data_indiv.append(np.median(data))
            filename = 'val_matrix_cen_selfsocial_perturb'
            if Path(fr'{data_dir}/{name}/{filename}.bin').is_file():
                with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                    data = pickle.load(f)
                data_indiv.append(np.median(data))
            filename = 'val_matrix_cen_ghostexplorer_perturb'
            if Path(fr'{data_dir}/{name}/{filename}.bin').is_file():
                with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                    data = pickle.load(f)
                data_indiv.append(np.median(data))
            filename = 'val_matrix_cen_ghostexploiter_perturb'
            if Path(fr'{data_dir}/{name}/{filename}.bin').is_file():
                with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                    data = pickle.load(f)
                data_indiv.append(np.median(data))
            filename = 'val_matrix_cen_N2-ghostexplorer_perturb'
            if Path(fr'{data_dir}/{name}/{filename}.bin').is_file():
                with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                    data = pickle.load(f)
                data_indiv.append(np.median(data))
            filename = 'val_matrix_cen_N2-ghostexploiter_perturb'
            if Path(fr'{data_dir}/{name}/{filename}.bin').is_file():
                with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                    data = pickle.load(f)
                data_indiv.append(np.median(data))
            
            data_group.append(data_indiv)
        data_all.append(data_group)

    data = np.array(data_all)
    num_groups, num_runs, num_perturbs = data.shape

    print(data.shape)

    total_runs = num_groups * num_runs
    color_range = np.linspace(0,1,total_runs)
    perturb_range = np.arange(1,num_perturbs+1)
    perturb_labels = ['As Trained', 'All RW', 'All D', 'No Social', 'Self Social', 'Ghost Explorer', 'Ghost Exploiter', 'N=2 Ghost Explorer', 'N=2 Ghost Exploiter']

    for g_num in range(num_groups):
        for r_num in range(num_runs):
            color = mpl.cm.plasma(color_range[g_num*num_runs + r_num])
            ax1.plot(perturb_range, data[g_num,r_num,:], 
                     color=color, alpha=0.5,
                     label=f'{groups[g_num][0]}',
                     )
    
    ax1.set_xlabel('Perturbation Type')
    ax1.set_xticks(perturb_range)
    ax1.set_xticklabels(perturb_labels, rotation=15)
    ax1.set_ylabel('Time Taken to Find Patch')
    ax1.set_ylim(180,1020)

    # ax1.legend()
    # ax1.legend(*zip(*violin_labs), loc='upper left')
    # ax1.legend(*zip(*violin_labs), bbox_to_anchor=(1.1, 1.05))

    if title is not None:
        ax1.set_title(title)

    plt.tight_layout()
    if save_name: 
        plt.savefig(fr'{data_dir}/{save_name}.png', dpi=100)
    plt.show()


def plot_mult_EA_trends_groups_endonly_ghost(groups, val_type='res', num_agents=1, max=None, save_name=None):

    # establish load directory
    root_dir = Path(__file__).parent.parent
    data_dir = Path(root_dir, r'data/simulation_data')

    # # init plot details
    fig, ax1 = plt.subplots(figsize=(6,4)) 
    cmap = plt.get_cmap('plasma')
    cmap_range = len(groups)
    violin_labs = []
    
    # iterate over each file
    for g_num, (group_name, run_names) in enumerate(groups):

        data_group = []
        data_group_ghostexplorer = []
        data_group_ghostexploiter = []
        for r_num, name in enumerate(run_names):

            if val_type == 'res': filename = 'val_matrix_cen'
            elif val_type == 'time': filename = 'val_matrix_cen_time'

            with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                data = pickle.load(f)
            if val_type == 'res':
                data /= num_agents
            data_group.append(data.flatten())

            # also load ghost data
            if val_type == 'res': filename = 'val_matrix_cen_ghostexplorer_perturb'
            elif val_type == 'time': filename = 'val_matrix_cen_ghostexplorer_time_perturb'
            if Path(fr'{data_dir}/{name}/{filename}.bin').is_file():
                with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                    data = pickle.load(f)
                if val_type == 'res': data /= num_agents
                data_group_ghostexplorer.append(data.flatten())
            if val_type == 'res': filename = 'val_matrix_cen_ghostexploiter_perturb'
            elif val_type == 'time': filename = 'val_matrix_cen_ghostexploiter_time_perturb'
            if Path(fr'{data_dir}/{name}/{filename}.bin').is_file():
                with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                    data = pickle.load(f)
                if val_type == 'res': data /= num_agents
                data_group_ghostexploiter.append(data.flatten())

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

        # ghost
        if data_group_ghostexplorer:
            data_ghostexplorer = np.array(data_group_ghostexplorer)
            l0 = ax1.violinplot(data_ghostexplorer.flatten(), 
                        positions=[g_num+.25],
                        widths=.5, 
                        showmedians=True, 
                        showextrema=False,
                        )
            for part in l0["bodies"]:
                part.set_edgecolor(cmap(g_num/cmap_range))
                part.set_facecolor(cmap(g_num/cmap_range))
                part.set_alpha(.5)
            l0["cmedians"].set_edgecolor(cmap(g_num/cmap_range))
            l0["cmedians"].set_alpha(.5)
            # color = l0["bodies"][0].get_facecolor().flatten()
            # violin_labs.append((mpatches.Patch(color=color), group_name))

        if data_group_ghostexploiter:
            data_ghostexploiter = np.array(data_group_ghostexploiter)
            l0 = ax1.violinplot(data_ghostexploiter.flatten(), 
                        positions=[g_num+.5],
                        widths=.5, 
                        showmedians=True, 
                        showextrema=False,
                        )
            for part in l0["bodies"]:
                part.set_edgecolor(cmap(g_num/cmap_range))
                part.set_facecolor(cmap(g_num/cmap_range))
                part.set_alpha(.7)
            l0["cmedians"].set_edgecolor(cmap(g_num/cmap_range))
            l0["cmedians"].set_alpha(.7)
            # color = l0["bodies"][0].get_facecolor().flatten()
            # violin_labs.append((mpatches.Patch(color=color), group_name))

        if data_group_ghostexploiter and data_group_ghostexplorer:
            print(f'{group_name}: {int(np.mean(data))} / {int(np.mean(data_ghostexplorer))} / {int(np.mean(data_ghostexploiter))}')
        elif data_group_ghostexplorer:
            print(f'{group_name}: {int(np.mean(data))} / {int(np.mean(data_ghostexplorer))}')
        elif data_group_ghostexploiter:
            print(f'{group_name}: {int(np.mean(data))} / {int(np.mean(data_ghostexploiter))}')
        else:
            print(f'{group_name}: {int(np.mean(data))}')

    ax1.legend(*zip(*violin_labs), loc='upper left')

    ax1.set_xticks([])
    if val_type == 'res': ax1.set_ylabel('Resources Collected per Agent')
    elif val_type == 'time': ax1.set_ylabel('Time for 1st Agent to Find Patch')
    if max is not None:
        ax1.set_ylim(-20,max)

    if save_name: 
        # plt.savefig(fr'{data_dir}/{save_name}.png')
        plt.savefig(fr'{data_dir}/{save_name}.png', dpi=100)
    plt.show()



def plot_mult_EA_trends_groups_endonly_sums(groups, bees=False, highlight=None, save_name=None):

    # establish load directory
    root_dir = Path(__file__).parent.parent
    data_dir = Path(root_dir, r'data/simulation_data')

    # # init plot details
    if highlight is None:
        fig, ax1 = plt.subplots(figsize=(6,4))
    else:
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4))

    cmap = plt.get_cmap('plasma')
    cmap_range = len(groups)
    violin_labs = []
    highlight_probs = []
    
    # iterate over each file
    for g_num, (group_name, run_names) in enumerate(groups):
        # print(group_name)

        data_group = []
        data_group_perturb = []
        for r_num, name in enumerate(run_names):
            # print(name)


            with open(fr'{data_dir}/{name}/val_matrix_best.bin','rb') as f:
                data_og = pickle.load(f)
            if 'soc' in save_name:
                if Path(fr'{data_dir}/{name}/val_matrix_best_nosocial_perturb.bin').is_file():
                    with open(fr'{data_dir}/{name}/val_matrix_best_nosocial_perturb.bin','rb') as f:
                        data_nosoc = pickle.load(f)
                else:
                    with open(fr'{data_dir}/{name}/val_matrix_best.bin','rb') as f:
                        data_nosoc = pickle.load(f)
                    with open(fr'{data_dir}/{name}/val_matrix_best_ghostexploiter_perturb.bin','rb') as f:
                        data_og = pickle.load(f)
            elif 'exp' in save_name:
                with open(fr'{data_dir}/{name}/val_matrix_best_ghostexploiter_perturb.bin','rb') as f:
                    data_exploiter = pickle.load(f)
                with open(fr'{data_dir}/{name}/val_matrix_best_ghostexplorer_perturb.bin','rb') as f:
                    data_explorer = pickle.load(f)





            filename = 'val_matrix_best'
            with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                data = pickle.load(f)
            num_pts = np.prod(data.shape)

            if 'median' in save_name:
                sum_type = 'median'
                data_group.append(np.median(data))
                prob_evol_text = 1050
            if 'mean' in save_name:
                sum_type = 'mean'
                data_group.append(np.mean(data))
                prob_evol_text = 1050
            if 'numfound' in save_name:
                sum_type = 'numfound'
                data_group.append((data<1000).sum()/num_pts)
                prob_evol_text = 1.1

            # also load perturb data
            if 'nosoc' in save_name:
                filename = 'val_matrix_best_nosocial_perturb'
                if Path(fr'{data_dir}/{name}/{filename}.bin').is_file():
                    with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                        data = pickle.load(f)
                    if sum_type == 'median': data_group_perturb.append(np.median(data))
                    elif sum_type == 'mean': data_group_perturb.append(np.mean(data))
                    elif sum_type == 'numfound': data_group_perturb.append((data<1000).sum()/num_pts)
            elif 'exploit' in save_name:
                filename = 'val_matrix_best_ghostexploiter_perturb'
                if Path(fr'{data_dir}/{name}/{filename}.bin').is_file():
                    with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                        data = pickle.load(f)
                    if sum_type == 'median': data_group_perturb.append(np.median(data))
                    elif sum_type == 'mean': data_group_perturb.append(np.mean(data))
                    elif sum_type == 'numfound': data_group_perturb.append((data<1000).sum()/num_pts)
            elif 'explore' in save_name:
                filename = 'val_matrix_best_ghostexplorer_perturb'
                if Path(fr'{data_dir}/{name}/{filename}.bin').is_file():
                    with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                        data = pickle.load(f)
                    if sum_type == 'median': data_group_perturb.append(np.median(data))
                    elif sum_type == 'mean': data_group_perturb.append(np.mean(data))
                    elif sum_type == 'numfound': data_group_perturb.append((data<1000).sum()/num_pts)

        data_og = np.array(data_group).flatten()
        if bees:
            # data = np.delete(data, np.argwhere(data == 1000))
            x_og = beeswarm(data_og)
            ax1.scatter(g_num*2+x_og, data_og, color=cmap(g_num/cmap_range), alpha=.3)
        if data_group_perturb:
            l0 = ax1.violinplot(data_og, 
                        positions=[g_num*2],
                        widths=0.5, 
                        showmedians=True, 
                        showextrema=False,
                        )
        else:
            l0 = ax1.violinplot(data_og, 
                        positions=[g_num*2],
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

        # perturbs
        if data_group_perturb:
            data_perturb = np.array(data_group_perturb)
            if bees:
                # data = np.delete(data, np.argwhere(data == 1000))
                x_nosoc = beeswarm(data_perturb)
                ax1.scatter(g_num*2+1+x_nosoc, data_perturb, color=cmap(g_num/cmap_range), alpha=.5)
            l0 = ax1.violinplot(data_perturb, 
                        positions=[g_num*2+1],
                        widths=.5, 
                        showmedians=True, 
                        showextrema=False,
                        )
            for part in l0["bodies"]:
                part.set_edgecolor(cmap(g_num/cmap_range))
                part.set_facecolor(cmap(g_num/cmap_range))
                part.set_alpha(.7)
            l0["cmedians"].set_edgecolor(cmap(g_num/cmap_range))
            l0["cmedians"].set_alpha(.7)

            # strings bw dists
            if highlight is None:
                ax1.plot([g_num*2+x_og, g_num*2+1+x_nosoc], [data_og, data_perturb], color=cmap(g_num/cmap_range), alpha=.2, linewidth=1)
            else:
                if 'nosocial' in save_name: # highlight strings + count
                    if highlight == 'pure_follow':
                        count = 0
                        for i in range(len(x_og)):
                            if sum_type == 'median' or sum_type == 'mean':
                                if data_perturb[i] > 950:
                                    ax1.plot([g_num*2+x_og[i], g_num*2+1+x_nosoc[i]], [data_og[i], data_perturb[i]], color='k', alpha=.3, linewidth=2)
                                    count += 1
                            elif sum_type == 'numfound':
                                if data_perturb[i] < 0.1:
                                    ax1.plot([g_num*2+x_og[i], g_num*2+1+x_nosoc[i]], [data_og[i], data_perturb[i]], color='k', alpha=.3, linewidth=2)
                                    count += 1
                        prob = count/len(x_og)
                        highlight_probs.append(prob)
                        ax1.text(g_num*2, prob_evol_text, f'{round(prob,2)}', color='k', fontsize=10)
                    elif highlight == 'pure_nav':
                        count = 0
                        for i in range(len(x_og)):
                            if sum_type == 'median' or sum_type == 'mean':
                                if data_og[i] - data_perturb[i] < 50 and data_og[i] - data_perturb[i] > -50 and data_perturb[i] < 950:
                                    ax1.plot([g_num*2+x_og[i], g_num*2+1+x_nosoc[i]], [data_og[i], data_perturb[i]], color='k', alpha=.3, linewidth=2)
                                    count += 1
                            elif sum_type == 'numfound':
                                if data_og[i] - data_perturb[i] < 0.1 and data_og[i] - data_perturb[i] > -0.1 and data_perturb[i] > 0.1:
                                    ax1.plot([g_num*2+x_og[i], g_num*2+1+x_nosoc[i]], [data_og[i], data_perturb[i]], color='k', alpha=.3, linewidth=2)
                                    count += 1
                        prob = count/len(x_og)
                        highlight_probs.append(prob)
                        ax1.text(g_num*2, prob_evol_text, f'{round(prob,2)}', color='k', fontsize=10)
                    elif highlight == 'nav_follhurts':
                        count = 0
                        for i in range(len(x_og)):
                            if sum_type == 'median' or sum_type == 'mean':
                                if data_og[i] - data_perturb[i] > 50:
                                    ax1.plot([g_num*2+x_og[i], g_num*2+1+x_nosoc[i]], [data_og[i], data_perturb[i]], color='k', alpha=.3, linewidth=2)
                                    count += 1
                            elif sum_type == 'numfound':
                                if data_og[i] - data_perturb[i] < -0.1:
                                    ax1.plot([g_num*2+x_og[i], g_num*2+1+x_nosoc[i]], [data_og[i], data_perturb[i]], color='k', alpha=.3, linewidth=2)
                                    count += 1
                        prob = count/len(x_og)
                        highlight_probs.append(prob)
                        ax1.text(g_num*2, prob_evol_text, f'{round(prob,2)}', color='k', fontsize=10)
                    elif highlight == 'nav_follhelps':
                        count = 0
                        for i in range(len(x_og)):
                            if sum_type == 'median' or sum_type == 'mean':
                                if data_og[i] - data_perturb[i] < -50 and not data_perturb[i] > 950:
                                    ax1.plot([g_num*2+x_og[i], g_num*2+1+x_nosoc[i]], [data_og[i], data_perturb[i]], color='k', alpha=.3, linewidth=2)
                                    count += 1
                            elif sum_type == 'numfound':
                                if data_og[i] - data_perturb[i] > 0.1 and not (data_perturb[i] < 0.1):
                                    ax1.plot([g_num*2+x_og[i], g_num*2+1+x_nosoc[i]], [data_og[i], data_perturb[i]], color='k', alpha=.3, linewidth=2)
                                    count += 1
                        prob = count/len(x_og)
                        highlight_probs.append(prob)
                        ax1.text(g_num*2, prob_evol_text, f'{round(prob,2)}', color='k', fontsize=10)

                    # print(f'{group_name} count: {count}, prob: {prob}')

                elif 'N2exploit' in save_name:
                    if highlight == 'pure_follow':
                        count = 0
                        for i in range(len(x_og)):
                            if data_perturb[i] < 0.1:
                                ax1.plot([g_num*2+x_og[i], g_num*2+1+x_nosoc[i]], [data_og[i], data_perturb[i]], color='k', alpha=.3, linewidth=2)
                                count += 1
                        prob = count/len(x_og)
                        ax1.text(g_num*2, prob_evol_text, f'{round(prob,2)}', color='k', fontsize=10)
            
    # ax1.set_xticks(np.linspace(0,len(groups)*2-2,len(groups))+0.5)
    # labs = [group_name for group_name,_ in groups]
    # if 'N6' in save_name:
    #     labs = [0,1,2,3,4,5]
    # elif 'N3' in save_name:
    #     labs = [0,1,2]
    # elif 'N11' in save_name:
    #     labs = [0,5,10]
    # elif 'N21' in save_name:
    #     labs = [0,10,20]
    # if 'N3' in save_name or 'N6' in save_name or 'N11' in save_name or 'N21' in save_name:
    #     ax1.set_xlabel('# Random Other Agents (& n-1 # Direct)')

    if 'NRW0' in save_name:
        ax1.set_xlabel('# Other Agents (All Direct)')
        if 'nosocial' not in save_name and 'N2exploit' not in save_name:
            labs = [0,1,2,3,4,5,10,20]
        else:
            labs = [1,2,3,4,5,10,20]
            if highlight is not None:
                ax1.text(-3, prob_evol_text, 'Prob evol:', color='k', fontsize=10)

    elif 'ND1' in save_name:
        ax1.set_xlabel('# Other Agents (1 Direct)')
        labs = [1,2,3,4,5]
        if highlight is not None:
            ax1.text(-2, prob_evol_text, 'Prob evol:', color='k', fontsize=10)

    elif 'ND2' in save_name:
        ax1.set_xlabel('# Other Agents (2 Direct)')
        labs = [2,3,4,5]
        if highlight is not None:
            ax1.text(-2, prob_evol_text, 'Prob evol:', color='k', fontsize=10)

    elif 'ghost' in save_name:
        ax1.set_xlabel('# Other Agents (All Direct)')
        labs = [0,'0-ghost',1,'1-ghost',5,'5-ghost']
        if highlight is not None:
            ax1.text(-2, prob_evol_text, 'Prob evol:', color='k', fontsize=10)

    elif 'init' in save_name:
        ax1.set_xlabel('# Direct // (Hundred Radial Units from Patch Center)')
        labs = ['ND1','A0-O4','A0-O2','A4-O4','A4-O2','ND5','A0-O4','A0-O2','A4-O4','A4-O2']
        if highlight is not None:
            ax1.text(-3, prob_evol_text, 'Prob evol:', color='k', fontsize=10)
        else:
            ax1.set_title('Varying Where (A)gent/(O)thers Spawn')

    elif 'NRWX' in save_name:
        ax1.set_xlabel('# Random Agents (& n-1 Direct)')
        labs = [0,1,2,3,4,5]
        if highlight is not None:
            ax1.text(-3, prob_evol_text, 'Prob evol:', color='k', fontsize=10)

    elif 'dist' in save_name:
        ax1.set_xlabel(r'$\sigma$ (Distance Scaling Factor)')
        labs = [1,.5,.4,.3,.2,.1,0]
        if highlight is not None:
            ax1.text(-3, prob_evol_text, 'Prob evol:', color='k', fontsize=10)

    elif 'vis_' in save_name:
        ax1.set_xlabel(r'$\upsilon$ (Visual Resolution)')
        labs = [6,8,10,12,14,16,18,20,24,32]
        if highlight is not None:
            ax1.text(-3, prob_evol_text, 'Prob evol:', color='k', fontsize=10)

    elif 'fov' in save_name:
        ax1.set_xlabel('Field of Vision')
        labs = [.2,.3,.4,.5,.6,.7,.8,.875]
        if highlight is not None:
            ax1.text(-3, prob_evol_text, 'Prob evol:', color='k', fontsize=10)


    if 'nosocial' not in save_name and 'N2exploit' not in save_name:
        ax1.set_xticks(np.linspace(0,len(labs)*2-2,len(labs)))
    else:
        ax1.set_xticks(np.linspace(0,len(labs)*2-2,len(labs))+0.5)
    ax1.set_xticklabels(labs)

    if sum_type == 'median' or sum_type == 'mean':
        ax1.set_ylabel('Time Taken to Reach Patch')
        ax1.set_ylim(-20,1020)
    elif sum_type == 'numfound':
        ax1.set_ylabel('Probability Finding Patch by t=1000')
        ax1.set_ylim(-.05,1.05)

    if highlight is not None:
        # print(np.linspace(0,len(labs)*2-2, len(labs))+0.5, highlight_probs)
        if len(labs) == len(highlight_probs):
            ax2.plot(np.linspace(0,len(labs)*2-2, len(labs))+0.5, highlight_probs, '--ko')
        else: # when including non-social trained agents (all agents must be navigators)
            if highlight == 'nav': 
                ax2.plot(np.linspace(0,len(labs)*2-2, len(labs))+0.5, [1]+highlight_probs, '--ko')
            else:
                ax2.plot(np.linspace(0,len(labs)*2-2, len(labs))+0.5, [0]+highlight_probs, '--ko')
        ax2.set_ylim(-.05,1.05)
        ax2.set_ylabel('Probability Evolving')
        ax2.set_xticks(ax1.get_xticks())
        ax2.set_xticklabels(ax1.get_xticklabels())

    plt.tight_layout()
    if save_name: 
        # plt.savefig(fr'{data_dir}/{save_name}.png')
        plt.savefig(fr'{data_dir}/{save_name}.png', dpi=100)
    plt.show()


def plot_mult_EA_trends_groups_endonly_divs(groups, sideplot=False, save_name=None):

    root_dir = Path(__file__).parent.parent
    data_dir = Path(root_dir, r'data/simulation_data')

    if sideplot is False:
        fig, ax1 = plt.subplots(figsize=(6,4))
    else:
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4))

    # bin possible fitness range for entropy dists
    bin_range = np.arange(0,1001,10)
    foll_divs_all = []
    foll_shifts_all = []
    foll_found_all = []
    foll_pure_all = []

    # print(f'----plotting {save_name}')
    # print(f'-+- plot -+-')
    for g_num, (group_name, run_names) in enumerate(groups):
        # print(f'-- group --')

        div_group = []
        shift_group = []
        found_group = []
        pure_group = []
        for r_num, name in enumerate(run_names):

            if 'nosocial' in save_name:
                if Path(fr'{data_dir}/{name}/val_matrix_best_nosocial_perturb.bin').is_file():
                    with open(fr'{data_dir}/{name}/val_matrix_best_nosocial_perturb.bin','rb') as f:
                        data_og = pickle.load(f)
                else:
                    with open(fr'{data_dir}/{name}/val_matrix_best.bin','rb') as f: # for NX_ghost
                        data_og = pickle.load(f)
                h_og = np.histogram(data_og, bins=bin_range)[0]
                with open(fr'{data_dir}/{name}/val_matrix_best.bin','rb') as f:
                    data_perturb = pickle.load(f)
                h_perturb = np.histogram(data_perturb, bins=bin_range)[0]
            elif 'ghostexploiter' in save_name:
                if Path(fr'{data_dir}/{name}/val_matrix_best_nosocial_perturb.bin').is_file():
                    with open(fr'{data_dir}/{name}/val_matrix_best_nosocial_perturb.bin','rb') as f:
                        data_og = pickle.load(f)
                else:
                    with open(fr'{data_dir}/{name}/val_matrix_best.bin','rb') as f: # for NX_ghost
                        data_og = pickle.load(f)
                h_og = np.histogram(data_og, bins=bin_range)[0]
                with open(fr'{data_dir}/{name}/val_matrix_best_ghostexploiter_perturb.bin','rb') as f:
                    data_perturb = pickle.load(f)
                h_perturb = np.histogram(data_perturb, bins=bin_range)[0]
            elif 'ghostexplorer' in save_name:
                if Path(fr'{data_dir}/{name}/val_matrix_best_nosocial_perturb.bin').is_file():
                    with open(fr'{data_dir}/{name}/val_matrix_best_nosocial_perturb.bin','rb') as f:
                        data_og = pickle.load(f)
                else:
                    with open(fr'{data_dir}/{name}/val_matrix_best.bin','rb') as f: # for NX_ghost
                        data_og = pickle.load(f)
                h_og = np.histogram(data_og, bins=bin_range)[0]
                with open(fr'{data_dir}/{name}/val_matrix_best_ghostexplorer_perturb.bin','rb') as f:
                    data_perturb = pickle.load(f)
                h_perturb = np.histogram(data_perturb, bins=bin_range)[0]
            elif 'expexp' in save_name:
                with open(fr'{data_dir}/{name}/val_matrix_best_ghostexplorer_perturb.bin','rb') as f:
                    data_og = pickle.load(f)
                h_og = np.histogram(data_og, bins=bin_range)[0]
                with open(fr'{data_dir}/{name}/val_matrix_best_ghostexploiter_perturb.bin','rb') as f:
                    data_perturb = pickle.load(f)
                h_perturb = np.histogram(data_perturb, bins=bin_range)[0]

            if 'KL' in save_name:
                div = calc_KLdiv(h_og, h_perturb)
            elif 'JS' in save_name:
                div = calc_JSdiv(h_og, h_perturb)
            div_group.append(div)

            mean_shift = np.mean(data_og) - np.mean(data_perturb)
            shift_group.append(mean_shift)

            # found_shift = ((data_og<1000).sum() - (data_perturb<1000).sum())/len(run_names)
            # found_group.append(found_shift)
            # foll_pure = np.mean(data_og)
            # pure_group.append(foll_pure)


        divs = np.array(div_group)
        shifts = np.array(shift_group)
        # founds = np.array(found_group)
        # pures = np.array(pure_group)
        # print(divs, np.mean(divs)-np.std(divs), np.mean(divs)+np.std(divs))
        # print(np.log(divs))
        # num_zeros = len(divs[divs == 0])

        # handle perfect matches (=zero, throws off log scale --> include at bottom of plot)
        zero_inds = np.where(divs == 0)[0]
        divs[zero_inds] = 1/1000

        if 'KL' in save_name:
            x = beeswarm(np.log(divs+1/10000))
        elif 'JS' in save_name:
            x = beeswarm(np.log(divs+1/10000))
            # x = beeswarm(divs)

        if 'bygroup' in save_name:
            ax1.scatter(g_num*2+x, divs, c=shifts, cmap='berlin', alpha=.3)
        elif 'twoslope' in save_name:
            norm = mpl.colors.TwoSlopeNorm(vmin=-50, vcenter=0, vmax=400)
            ax1.scatter(g_num*2+x, divs, c=shifts, cmap='berlin_r', norm=norm, alpha=.3)
        

        # if 'JS' in save_name and 'ghostexploiter' in save_name:
        # #     print(f'{np.min(divs):.3f}, {np.max(divs):.3f}')
        #     print(int(shifts.min()), int(shifts.mean()-shifts.std()), int(shifts.mean()+shifts.std()), int(shifts.max()))


        l0 = ax1.violinplot(divs, 
                    positions=[g_num*2],
                    widths=0.5, 
                    showmedians=True, 
                    showextrema=False,
                    )
        for part in l0["bodies"]:
            part.set_edgecolor('k')
            part.set_facecolor('k')
            part.set_alpha(.05)
        l0["cmedians"].set_edgecolor('k')
        # l0["cmedians"].set_alpha(.05)

        # for d,s,n in zip(divs,shifts,run_names):
        #     print(f'{n}: {d:.2f}, {s:.2f}')

        num_foll_divs = (divs>0.1).sum()/len(divs)
        num_foll_shifts = (shifts>150).sum()/len(shifts)
        # num_foll_found = (founds>0.1).sum()/len(founds)
        # num_foll_pure = (pures>950).sum()/len(pures)
        foll_divs_all.append(num_foll_divs)
        foll_shifts_all.append(num_foll_shifts)
        # foll_found_all.append(num_foll_found)
        # foll_pure_all.append(num_foll_pure)
        # num_nav_divs = (divs>0.1).sum()/len(divs)
        # num_nav_shifts = (shifts>80).sum()/len(shifts)
        # print(name, num_foll_divs, num_foll_shifts)
        if sideplot is False:
            ax1.text(g_num*2 - .5, 1.1, f'{num_foll_divs:.2f}/{num_foll_shifts:.2f}', color='k', fontsize=10)


    if 'NRW0' in save_name:
        ax1.set_xlabel('# Other Agents (All Direct)')
        # labs = [1,2,3,4,5,10,20]
        labs = [1,2,3,4,5]
    elif 'NRW1' in save_name:
        ax1.set_xlabel('# Other Agents (1 Random, n-1 Direct)')
        labs = [1,2,3,4]
    elif 'NRW2' in save_name:
        ax1.set_xlabel('# Other Agents (2 Random, n-2 Direct)')
        labs = [1,2,3]
    elif 'NRW3' in save_name:
        ax1.set_xlabel('# Other Agents (3 Random, n-3 Direct)')
        labs = [1,2]

    elif 'ND1' in save_name:
        ax1.set_xlabel('# Other Agents (1 Direct, n-1 Random)')
        labs = [1,2,3,4,5]
    elif 'ND2' in save_name:
        ax1.set_xlabel('# Other Agents (2 Direct, n-2 Random)')
        labs = [2,3,4,5]
    elif 'ND3' in save_name:
        ax1.set_xlabel('# Other Agents (3 Direct, n-3 Random)')
        labs = [3,4,5]
    elif 'ND4' in save_name:
        ax1.set_xlabel('# Other Agents (4 Direct, n-4 Random)')
        labs = [4,5]

    elif 'NRWX' in save_name:
        # ax1.set_xlabel('# Random Agents (& n-1 Direct)')
        # labs = [0,1,2,3,4,5]
        ax1.set_xlabel('# Direct Agents (5-n Random Others)')
        labs = [0,1,2,3,4,5]

    elif 'ghost_' in save_name:
        ax1.set_xlabel('# Other Agents (All Direct)')
        labs = ['0-ghost',1,'1-ghost',5,'5-ghost']

    elif 'N1-init' in save_name:
        ax1.set_xlabel('# Radial Units from Patch Center (1 Direct)')
        labs = ['All Map','400','200','0']
        ax1.set_title('Varying Where Other Agents Spawn')

    elif 'N5-init' in save_name:
        ax1.set_xlabel('# Radial Units from Patch Center (5 Direct)')
        labs = ['All Map','400','200','0']
        ax1.set_title('Varying Where Other Agents Spawn')

    elif 'init' in save_name:
        ax1.set_xlabel('# Direct // (Hundred Radial Units from Patch Center)')
        labs = ['ND1','A0-O4','A0-O2','A4-O4','A4-O2','ND5','A0-O4','A0-O2','A4-O4','A4-O2']
        ax1.set_title('Varying Where (A)gent/(O)thers Spawn')

    elif 'dist' in save_name:
        ax1.set_xlabel(r'$\sigma$ (Distance Scaling Factor)')
        labs = [1,.5,.4,.3,.2,.1,0]

    elif 'vis_' in save_name:
        ax1.set_xlabel(r'$\upsilon$ (Visual Resolution)')
        labs = [6,8,10,12,14,16,18,20,24,32]

    elif 'fov' in save_name:
        ax1.set_xlabel('Field of Vision')
        labs = [.2,.3,.4,.5,.6,.7,.8,.875]

    ax1.set_xticks(np.linspace(0,len(labs)*2-2,len(labs)))
    ax1.set_xticklabels(labs)

    if 'KL' in save_name:
        ax1.set_ylabel('KL Divergence')
        ax1.set_yscale('log')
        # ax1.set_ylim(.02,20)
        ax1.set_ylim(.007,20)
    elif 'JS' in save_name:
        ax1.set_ylabel('JS Divergence')
        # ax1.set_ylim(-.02,.72)
        ax1.set_yscale('log')
        ax1.set_ylim(.001,1)

    if sideplot is True:
        # print(np.linspace(0,len(labs)*2-2, len(labs))+0.5, highlight_probs)
        # if len(labs) == len(highlight_probs):
        ax2.plot(np.linspace(0,len(labs)*2-2, len(labs)), np.array(foll_divs_all), '-ko', alpha=.5, label='JS Divergence > 0.1')
        ax2.plot(np.linspace(0,len(labs)*2-2, len(labs)), np.array(foll_shifts_all), '--ko', alpha=.5, label='Mean Shift > 160')
        # else: # when including non-social trained agents (all agents must be navigators)
        #     if highlight == 'nav': 
        #         ax2.plot(np.linspace(0,len(labs)*2-2, len(labs))+0.5, [1]+highlight_probs, '--ko')
        #     else:
        #         ax2.plot(np.linspace(0,len(labs)*2-2, len(labs))+0.5, [0]+highlight_probs, '--ko')
        ax2.set_ylim(-.05,1.05)
        ax2.set_ylabel('Probability Evolving')
        ax2.set_xticks(ax1.get_xticks())
        ax2.set_xticklabels(ax1.get_xticklabels())
        ax2.set_xlim(left=-.5)
        ax2.legend(loc='upper left')
    else:
        text_pos = -int(len(labs)/2)
        # print(len(labs), text_pos)
        ax1.text(text_pos, 1.75, 'Num Foll', color='k', fontsize=10)
        ax1.text(text_pos, 1.25, '(divs/shifts)', color='k', fontsize=10)

    plt.tight_layout()
    if save_name: 
        # plt.savefig(fr'{data_dir}/{save_name}.png')
        plt.savefig(fr'{data_dir}/{save_name}.png', dpi=100)
    plt.show()


def calc_KLdiv(x,y):
    x = x + 1/10000
    y = y + 1/10000
    x_norm = x / np.sum(x)
    y_norm = y / np.sum(y)
    KL = np.sum( x_norm*np.log(x_norm/y_norm) )
    return KL

def calc_JSdiv(x,y):
    x = x + 1/10000
    y = y + 1/10000
    x_norm = x / np.sum(x)
    y_norm = y / np.sum(y)
    mix = (x_norm + y_norm)/2
    KL_x_mix = np.sum( x_norm*np.log(x_norm/mix) )
    KL_y_mix = np.sum( y_norm*np.log(y_norm/mix) )
    JS = KL_x_mix/2 + KL_y_mix/2
    return JS


def plot_mult_EA_trends_groups_endonly_means(groups, metric_type=None, metric_thresh=None, cmap='berlin', save_name=None):

    root_dir = Path(__file__).parent.parent
    data_dir = Path(root_dir, r'data/simulation_data')

    # fig, ax1 = plt.subplots(figsize=(10,4))
    # fig, ax1 = plt.subplots(figsize=(6,4))
    fig, ax1 = plt.subplots(figsize=(5,4))

    dist_all = []
    labs_all = []

    for g_num, (group_name, run_names) in enumerate(groups):

        labs_all.append(group_name)
        dists = np.array(names_to_metric(run_names, metric_type, metric_thresh))
        dist_all.append(dists)

        x = beeswarm(dists)

        # # print all names with corresponding dists
        # for name, dist in zip(run_names, dists):
        #     print(f"{name}: {dist}")

        if cmap == 'berlin' and 'shift' in metric_type:
            norm = mpl.colors.TwoSlopeNorm(vmin=-50, vcenter=0, vmax=50)
            im = ax1.scatter(g_num*2+x, dists, c=dists, cmap='berlin_r', norm=norm, alpha=.3)
        elif cmap == 'berlin' and 'norm' in metric_type:
            norm = mpl.colors.TwoSlopeNorm(vmin=-.2, vcenter=0, vmax=.2)
            im = ax1.scatter(g_num*2+x, dists, c=dists, cmap='berlin_r', norm=norm, alpha=.3)
        # elif 'dirent' in metric_type:
        #     ax1.scatter(g_num*2+x, dists, c=dists, cmap='plasma', vmin=.2, vmax=.8, alpha=.3)
        elif cmap == 'fit_og':
            data = np.array(names_to_metric(run_names, 'dist_og', None))
            im = ax1.scatter(g_num*2+x, dists, c=data, cmap='plasma', vmin=200, vmax=500, alpha=.3)
        elif cmap == 'fit_ET':
            data = np.array(names_to_metric(run_names, 'dist_exploiter', None))
            im = ax1.scatter(g_num*2+x, dists, c=data, cmap='plasma', vmin=200, vmax=500, alpha=.3)
        elif cmap == 'dist_shift_NSET':
            data = np.array(names_to_metric(run_names, cmap, None))
            im = ax1.scatter(g_num*2+x, dists, c=data, cmap='viridis', vmin=0, vmax=300, alpha=.3)
        elif cmap == 'dist_JSspatial_NSET':
            data = np.array(names_to_metric(run_names, cmap, None))
            im = ax1.scatter(g_num*2+x, dists, c=data, cmap='viridis', vmin=0, vmax=.2, alpha=.3)

        l0 = ax1.violinplot(dists, 
                    positions=[g_num*2],
                    widths=0.5, 
                    showmedians=True, 
                    showextrema=False,
                    # quantiles=[.25,.75],
                    )
        for part in l0["bodies"]:
            part.set_edgecolor('k')
            part.set_facecolor('k')
            part.set_alpha(.05)
        l0["cmedians"].set_edgecolor('k')

    if 'Nall' in save_name:
        ax1.set_xlabel('# Direct x # Random')
        labs = labs_all
        divisions = np.array([5.5,10.5,14.5,17.5,19.5])*2
    
    elif 'init' in save_name:
        ax1.set_xlabel('Hundred Radial Units from Patch Center // # Direct x # Random')
        labs = labs_all
        divisions = np.array([3.5])*2
    
    elif 'ag_' in save_name:
        ax1.set_xlabel('Initialization Diameter from Agent (1 Direct x 0 Random)')
        labs = labs_all
    elif 'agNd5' in save_name:
        ax1.set_xlabel('Initialization Diameter from Agent (5 Direct x 0 Random)')
        labs = labs_all
    elif 'res_' in save_name:
        ax1.set_xlabel('Initialization Diameter from Patch (1 Direct x 0 Random)')
        labs = labs_all
    elif 'resNd2' in save_name:
        ax1.set_xlabel('Initialization Diameter from Patch (2 Direct x 0 Random)')
        labs = labs_all
    elif 'resNd5' in save_name:
        ax1.set_xlabel('Initialization Diameter from Patch (5 Direct x 0 Random)')
        labs = labs_all

    elif 'Nd0' in save_name:
        ax1.set_xlabel('# Random')
        labs = labs_all
    
    else:
        ax1.set_xlabel('# Direct x # Random')
        labs = labs_all

    ax1.set_xticks(np.linspace(0,len(labs)*2-2,len(labs)))
    ax1.set_xticklabels(labs)

    if 'dist_shift' in metric_type:
        ax1.set_ylabel('Performance Difference')
        ax1.yaxis.label.set_color('forestgreen')
        dist_all = np.array(dist_all)
        ax1.set_ylim(-200,800)
        # ax1.set_ylim(-600,300)
        # ax1.hlines(50,-1,20*2+1, linestyles='dotted', colors='black', alpha=.3)
        # ax1.hlines(100,-1,20*2+1, linestyles='dotted', colors='black', alpha=.5)
        # ax1.hlines(200,-1,20*2+1, linestyles='dotted', colors='black', alpha=.7)
    elif 'dist_JS' in metric_type:
        ax1.set_ylabel('Directional Divergence')
        ax1.yaxis.label.set_color('red')
        ax1.set_ylim(-.01,0.43)
    elif 'dist_dirent' in metric_type:
        ax1.set_ylabel('Directional Entropy')
        ax1.set_ylim(0,0.85)
    elif 'dist_norm' in metric_type:
        ax1.set_ylabel('Normalized Performance Difference')
        dist_all = np.array(dist_all)
        # ax1.set_ylim(-200,800)
    elif 'dist_learning_time' in metric_type:
        # ax1.set_ylabel('Mean Time Taken to Reach Patch')
        ax1.set_ylabel('Time to 500 Performance')
        ax1.set_ylim(0,1000)
    elif 'dist' in metric_type:
        # ax1.set_ylabel('Mean Time Taken to Reach Patch')
        ax1.set_ylabel('Performance (Trained Env)')
        ax1.set_ylim(200,900)

    if 'all' in save_name:
        ax1.vlines(divisions,np.min(dist_all),np.max(dist_all), linestyles='dashed', alpha=.5)

    # # cbar = ax1.figure.colorbar(im)
    cbar = ax1.figure.colorbar(im, label='Performance Difference', extend='both')
    # # cbar = ax1.figure.colorbar(im, label='Directional Divergence')
    cbar.solids.set(alpha=.7)

    plt.tight_layout()
    if save_name: 
        plt.savefig(fr'{data_dir}/{save_name}_{metric_type}_{cmap}.png', dpi=100)
    plt.show()


def plot_mult_EA_trends_groups_2D(groups, metric_type1=None, metric_type2=None, color_type=None, cbar=True, save_name=None):

    root_dir = Path(__file__).parent.parent
    data_dir = Path(root_dir, r'data/simulation_data')

    with open(fr'{data_dir}/traj_matrices/gamut_social.bin', 'rb') as f:
        data_dict = pickle.load(f)

    fig, ax1 = plt.subplots(figsize=(6,6))

    labs_all = []

    metric_type2_list = [
            'de_mean_OG', 'de_mean_NS', 'de_mean_ET', 'de_mean_ER',
            'JS_mean_OGNS', 'JS_mean_NSET', 'JS_mean_NSER', 'JS_mean_ETER'
            ]
    index = metric_type2_list.index(metric_type2)
    # print(metric_type1, metric_type2_list[index])

    dists1_all = []
    dists2_all = []
    xs,ys = [],[]
    for g_num, (group_name, run_names) in enumerate(groups):
        labs_all.append(group_name)

        dist1_group = []
        dist2_group = []
        for r_num, name in enumerate(run_names):
            # if int(group_name[2:]) < 1:
            # if int(group_name[2:]) < 2:
            #     continue

            if name in data_dict.keys():
                # if int(name_to_metric(name, 'dist_og')) > 500:
                #     print(f'{name} not included')
                #     dist1_group.append(None)
                #     dist2_group.append(None)
                #     continue
                dist1 = name_to_metric(name, metric_type1)
                dist2 = data_dict[name][index]

                # names = [
                #     'sc_N6_NRW5_ND0_CNN14_FNN16_vis8_rep37', # no perf + no spatial
                #     'sc_N4_NRW2_ND1_CNN14_FNN16_vis8_rep18', # spatial only
                #     'sc_N6_NRW1_ND4_CNN14_FNN16_vis8_rep35', # perf only (BD)
                #     'sc_N6_NRW2_ND3_CNN14_FNN16_vis8_rep33', # perf + weak spatial (hybrid)
                #     'sc_N6_NRW4_ND1_CNN14_FNN16_vis8_rep29', # perf + weak spatial (hybrid)
                #     'sc_N5_NRW2_ND2_CNN14_FNN16_vis8_rep31', # perf + strong spatial + discernment
                # ]
    
                # if name in names:
                #     print(name, dist1, dist2)

                # print(f'{name}: {metric_type1}: {int(dist1)} | {metric_type2}: {round(dist2,1)}')
                # print(f'{name}: {[round(n,2) for n in data_dict[name]]} --> {data_dict[name][index]}')

                # if dist1 > -100 and dist1 < 50 and dist2 < 0.03: # NSET - no effect
                # if dist1 > -100 and dist1 < 50 and dist2 > 0.13 and dist2 < 0.15: # NSET - spatial effect only (no perf)
                # if dist1 > 600 and dist2 > 0.34: # NSET - follow - near-direct
                # if dist1 > 720 and dist2 < 0.36: # NSET - follow - messier
                # if dist1 > 575 and dist2 < 0.2: # NSET - poor self-nav
                # if dist1 > 550 and dist2 < 0.12: # NSET - ok self-nav
                # if dist1 > 400 and dist2 < 0.06: # NSET - approx self-nav - perf effect only (no spatial)
                # if dist1 < 600 and dist2 > 0.24: # NSET - slow self-nav
                # if dist2 > 0.2 and int(name_to_metric(name, 'shift_Nd+2')) < 80 and int(name_to_metric(name, 'shift_ETER')) < 50: # NSET - perf/spatial effect but agent/density invariant
                    # print(name, int(dist1), dist2.round(2))
                    # print(name, int(dist1), dist2.round(2), int(name_to_metric(name, 'dist_og')))
                    # print(name, int(dist1), dist2.round(2), int(name_to_metric(name, 'shift_Nd+2')), int(name_to_metric(name, 'shift_ETER')), )
                    # print(name, int(name_to_metric(name, 'dist_nosoc')), int(name_to_metric(name, 'dist_exploiter')))
                    # print(name, int(dist1), dist2.round(2))

                dist1_group.append(dist1)
                dist2_group.append(dist2)
            else:
                print(f'{name} not in dict')
                dist1_group.append(None)
                dist2_group.append(None)

        # key_pts = [
        #     ('i',   -37,    0.029),
        #     ('ii',    0,    0.143),
        #     ('iii', 475,    0.039),
        #     ('iv',  734,    0.210),
        #     ('v',   768,    0.393),
        # ]
        # for n,x,y in key_pts:
        #     ax1.scatter(x,y, alpha=.5, s=15, edgecolor='darkslategrey', facecolor='None')
        #     if n == 'i' or n == 'ii':
        #         ax1.annotate(n,(x,y), color='darkslategrey', fontsize=12, xytext=(-10,-5), textcoords='offset points')
        #     else:
        #         ax1.annotate(n,(x,y), color='darkslategrey', fontsize=12, xytext=(5,-5), textcoords='offset points')

        # print(group_name, len(dist_group))
        if len(dist1_group) == 0:
            # dists = np.array([])
            # dist_all.append(np.array([]))
            pass
        else:
            dists1 = np.array(dist1_group)
            dists2 = np.array(dist2_group)

            # ax1.scatter(dists1, dists2, c='k', alpha=.3)
            # print(group_name,int(group_name[0]))
            label = ''
            if color_type == '':
                im = ax1.scatter(dists1, dists2, c='k', alpha=.5, s=15)
            elif color_type == 'heatmap':
                dists1_all.extend(dists1)
                dists2_all.extend(dists2)
            elif color_type == 'COM':
                x = np.median([x for x in dists1 if x is not None])
                y = np.median([x for x in dists2 if x is not None])
                if 'ag' in save_name:
                    if group_name[0] == 'A':
                        im = ax1.scatter(x,y, c=4, cmap='plasma', vmin=0, vmax=4, alpha=.7, s=20)
                    else:
                        im = ax1.scatter(x,y, c=int(group_name[0]), cmap='plasma', vmin=0, vmax=4, alpha=.7, s=20)
                    ax1.annotate(group_name,(x,y), alpha=.3)
                    label = 'Init Diameter from Agent'
                elif 'res' in save_name:
                    if group_name[0] == 'A':
                        im = ax1.scatter(x,y, c=4, cmap='plasma', vmin=0, vmax=4, alpha=.7, s=20)
                    else:
                        im = ax1.scatter(x,y, c=int(group_name[0]), cmap='plasma', vmin=0, vmax=4, alpha=.7, s=20)
                    ax1.annotate(group_name,(x,y), alpha=.3)
                    label = 'Init Diameter from Patch'
                elif 'varycoll' in save_name:
                    im = ax1.scatter(x,y, c=int(group_name[-1]), cmap='plasma', vmin=0, vmax=2, alpha=.7, s=20)
                    ax1.annotate(group_name[:-1],(x,y), alpha=.3)
                    label = ''
                else:
                    im = ax1.scatter(x,y, c=int(group_name[0]), cmap='plasma', vmin=0, vmax=5, alpha=.7, s=20) #dir
                    ax1.annotate(group_name,(x,y), alpha=.3)
                    label = '# Direct'
                    # im = ax1.scatter(x,y, c=int(group_name), cmap='plasma', vmin=0, vmax=20, alpha=.7, s=20) #rand
                    # im = ax1.scatter(x,y, c=int(group_name[2]), cmap='plasma', vmin=0, vmax=5, alpha=.7, s=20) #rand
                    # ax1.annotate(group_name,(x,y), alpha=.3)
                    # label = '# Random'
            elif color_type == 'num_direct':
                im = ax1.scatter(dists1, dists2, c=[int(group_name[0])]*len(run_names), cmap='viridis', vmin=0, vmax=5, alpha=.3, s=10)
                label = '# Direct'
            elif color_type == 'num_rand':
                im = ax1.scatter(dists1, dists2, c=[int(group_name[2:])]*len(run_names), cmap='plasma', vmin=0, vmax=5, alpha=.3, s=10)
                # im = ax1.scatter(dists1, dists2, c=[int(group_name[2:])]*len(run_names), cmap='plasma', vmin=0, vmax=20, alpha=.3, s=10)
                label = '# Random'
            elif color_type == 'num_total':
                im = ax1.scatter(dists1, dists2, c=[int(group_name[0])+int(group_name[2])]*len(run_names), cmap='plasma', vmin=0, vmax=5, alpha=.3, s=10)
            elif color_type == 'num_direct_split':
                norm = mpl.colors.TwoSlopeNorm(vmin=1, vcenter=1.5, vmax=2)
                im = ax1.scatter(dists1, dists2, c=[int(group_name[0])]*len(run_names), cmap='bwr', norm=norm, alpha=.3, s=10)
            elif color_type == 'num_direct_COM':
                im = ax1.scatter(dists1, dists2, c=[int(group_name[0])]*len(run_names), cmap='plasma', vmin=0, vmax=5, alpha=.3, s=10)
                x = np.median([x for x in dists1 if x is not None])
                y = np.median([x for x in dists2 if x is not None])
                # im = ax1.scatter(x,y, c=int(group_name[0]), cmap='plasma', vmin=0, vmax=5, alpha=.7, s=20)
                im = ax1.scatter(x,y, c=int(group_name[0]), cmap='plasma', vmin=0, vmax=5, alpha=.7, s=20, edgecolor='black')
                xs.append(x)
                ys.append(y)
                label = '# Direct'
            elif color_type == 'num_rand_COM':
                im = ax1.scatter(dists1, dists2, c=[int(group_name[2])]*len(run_names), cmap='plasma', vmin=0, vmax=5, alpha=.3, s=10)
                x = np.median([x for x in dists1 if x is not None])
                y = np.median([x for x in dists2 if x is not None])
                # im = ax1.scatter(x,y, c=int(group_name[2]), cmap='plasma', vmin=0, vmax=5, alpha=.7, s=20)
                im = ax1.scatter(x,y, c=int(group_name[2]), cmap='plasma', vmin=0, vmax=5, alpha=.7, s=20, edgecolor='black')
                xs.append(x)
                ys.append(y)
                label = '# Random'
            elif color_type == 'learning_time':
                data = np.array(names_to_metric(run_names, 'dist_learning_time', 500))
                # print(group_name,data.min(),data.mean(),data.max())
                im = ax1.scatter(dists1, dists2, c=data, cmap='plasma', vmin=0, vmax=500, alpha=.3, s=10)
            elif color_type == 'dirent_OG':
                data = np.array(names_to_metric(run_names, 'dist_dirent_OG', None))
                im = ax1.scatter(dists1, dists2, c=data, cmap='plasma', vmin=.2, vmax=.8, alpha=.3, s=10)
            elif color_type == 'dirent_NS':
                data = np.array(names_to_metric(run_names, 'dist_dirent_NS', None))
                im = ax1.scatter(dists1, dists2, c=data, cmap='plasma', vmin=.2, vmax=.8, alpha=.3, s=10)
                label = 'Directedness (No-Social)'
            elif color_type == 'dirent_ET':
                data = np.array(names_to_metric(run_names, 'dist_dirent_ET', None))
                im = ax1.scatter(dists1, dists2, c=data, cmap='plasma', vmin=.2, vmax=.8, alpha=.3, s=10)
            elif color_type == 'dirent_ER':
                data = np.array(names_to_metric(run_names, 'dist_dirent_ER', None))
                im = ax1.scatter(dists1, dists2, c=data, cmap='plasma', vmin=.2, vmax=.8, alpha=.3, s=10)
            elif color_type == 'fit_OG':
                data = np.array(names_to_metric(run_names, 'dist_og', None))
                # im = ax1.scatter(dists1, dists2, c=data, cmap='plasma', vmin=300, vmax=450, alpha=.3, s=10) # coll
                # im = ax1.scatter(dists1, dists2, c=data, cmap='plasma', vmin=225, vmax=275, alpha=.3, s=10) # nocoll
                im = ax1.scatter(dists1, dists2, c=data, cmap='plasma', vmin=300, vmax=450, alpha=.7, s=20) # for indiv plots
                # im = ax1.scatter(dists1, dists2, c=data, cmap='plasma', vmin=225, vmax=275, alpha=.7, s=20) # for indiv plots + nocoll
                label = 'Performance (Trained Env)'
            elif color_type == 'fit_NS':
                data = np.array(names_to_metric(run_names, 'dist_nosoc', None))
                im = ax1.scatter(dists1, dists2, c=data, cmap='plasma', vmin=200, vmax=500, alpha=.3, s=10)
            elif color_type == 'fit_ET':
                data = np.array(names_to_metric(run_names, 'dist_exploiter', None))
                im = ax1.scatter(dists1, dists2, c=data, cmap='plasma', vmin=200, vmax=500, alpha=.3, s=10)
            elif color_type == 'fit_ER':
                data = np.array(names_to_metric(run_names, 'dist_explorer', None))
                im = ax1.scatter(dists1, dists2, c=data, cmap='plasma', vmin=200, vmax=500, alpha=.3, s=10)
            elif color_type == 'Sinit_dist':
                if 'Ag' in group_name:
                    im = ax1.scatter(dists1, dists2, c=[int(group_name[6])]*len(run_names), cmap='plasma', vmin=1, vmax=4, alpha=.3, s=10)
                elif 'Res' in group_name:
                    im = ax1.scatter(dists1, dists2, c=[int(group_name[7])]*len(run_names), cmap='plasma', vmin=1, vmax=4, alpha=.3, s=10)
            elif color_type == 'dist_shift_OGNS':
                data = np.array(names_to_metric(run_names, color_type, None))
                norm = mpl.colors.TwoSlopeNorm(vmin=-50, vcenter=0, vmax=50)
                im = ax1.scatter(dists1, dists2, c=data, cmap='berlin_r', norm=norm, alpha=.3, s=10)
            elif color_type == 'dist_shift_NSER':
                data = np.array(names_to_metric(run_names, color_type, None))
                im = ax1.scatter(dists1, dists2, c=data, cmap='plasma', vmin=0, vmax=400, alpha=.3, s=10)
                label = 'Performance Difference (NS - BEr)'
            elif color_type == 'dist_shift_ETER':
                data = np.array(names_to_metric(run_names, color_type, None))
                im = ax1.scatter(dists1, dists2, c=data, cmap='plasma', vmin=0, vmax=400, alpha=.3, s=10)
                label = 'Performance Difference (BEr - BEt)'
            elif color_type == 'dist_JSspatial_ETER':
                data = np.array(names_to_metric(run_names, color_type, None))
                im = ax1.scatter(dists1, dists2, c=data, cmap='plasma', vmin=0, vmax=0.15, alpha=.3, s=10)
                label = 'Spatial Divergence (BEr - BEt)'
            elif color_type == 'dist_shift_Nd+2':
                data = np.array(names_to_metric(run_names, color_type, None))
                im = ax1.scatter(dists1, dists2, c=data, cmap='plasma', vmin=0, vmax=150, alpha=.3, s=10)
                label = 'Performance Difference (ND+2 - OG)'
            elif color_type == 'dist_shift_Nr+2':
                data = np.array(names_to_metric(run_names, color_type, None))
                im = ax1.scatter(dists1, dists2, c=data, cmap='plasma', vmin=0, vmax=150, alpha=.3, s=10)
                label = 'Performance Difference (NR+2 - OG)'
            elif color_type == 'distance':
                data = np.array(names_to_metric(run_names, color_type, None))
                im = ax1.scatter(dists1, dists2, c=data, cmap='plasma', vmin=0, vmax=100, alpha=.3, s=10)

    # ax1.set_xlabel(metric_type1)
    # ax1.set_ylabel(metric_type2)
    ax1.set_xlabel('Performance Difference')
    ax1.set_ylabel('Directional Divergence')
    ax1.xaxis.label.set_color('forestgreen')
    ax1.yaxis.label.set_color('red')

    if color_type =='heatmap':
        dists1_all = [x for x in dists1_all if x is not None]
        dists2_all = [x for x in dists2_all if x is not None]
        dists1 = np.array(dists1_all)
        dists2 = np.array(dists2_all)
        # print(dists1.shape, dists2.shape)
        # print(np.min(dists1), ' | ', np.max(dists1))
        # print(np.min(dists2), ' | ', np.max(dists2))
        num_bins = 51
        x_bins = np.linspace(-500, 850, num_bins)
        y_bins = np.linspace(-.01, 0.43, num_bins)
        H,_,_ = np.histogram2d(dists1, dists2, bins=[x_bins, y_bins])
        X,Y = np.meshgrid(x_bins, y_bins)
        # print(H.shape, X.shape, Y.shape)
        im = ax1.pcolormesh(X, Y, H.T, vmax=5, cmap='plasma')

        cbar = ax1.figure.colorbar(im, label='Count')
        cbar.solids.set(alpha=1)
    else:
        # scatter plots
        ax1.set_xlim(-500,850)
        # ax1.set_xlim(200,1010)
        ax1.set_ylim(-.01,.43)
        if cbar:
            if label == '':
                cbar = ax1.figure.colorbar(im, label=color_type)
            else:
                cbar = ax1.figure.colorbar(im, label=label)
            cbar.solids.set(alpha=.7)

        if color_type == 'num_direct_COM':
            ax1.plot(xs,ys, color='grey', linestyle='--', linewidth=1, zorder=0)

    plt.tight_layout()
    if save_name: 
        plt.savefig(fr'{data_dir}/{save_name}_{metric_type1}_x_{metric_type2}_{color_type}.png', dpi=100)
    plt.close()


def name_to_metric(name, metric_type):

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'

    if Path(fr'{data_dir}/{name}/val_matrix_best_nosocial_perturb.bin').is_file():
        with open(fr'{data_dir}/{name}/val_matrix_best.bin','rb') as f:
            data_og = pickle.load(f)
        with open(fr'{data_dir}/{name}/val_matrix_best_nosocial_perturb.bin','rb') as f:
            data_nosoc = pickle.load(f)
    else:
        with open(fr'{data_dir}/{name}/val_matrix_best.bin','rb') as f:
            data_og = pickle.load(f)
        with open(fr'{data_dir}/{name}/val_matrix_best.bin','rb') as f:
            data_nosoc = pickle.load(f)
    with open(fr'{data_dir}/{name}/val_matrix_best_ghostexploiter_perturb.bin','rb') as f:
        data_exploiter = pickle.load(f)
    # with open(fr'{data_dir}/{name}/val_matrix_best_ghostexplorer_perturb.bin','rb') as f:
    #     data_explorer = pickle.load(f)

    if 'og' in metric_type:
        metric = np.mean(data_og)
    elif 'nosoc' in metric_type:
        metric = np.mean(data_nosoc)
    elif 'exploiter' in metric_type:
        metric = np.mean(data_exploiter)
    elif 'explorer' in metric_type:
        metric = np.mean(data_explorer)

    elif 'shift_OGNS' in metric_type:
        metric = np.mean(data_nosoc) - np.mean(data_og)
    elif 'shift_NSET' in metric_type:
        metric = np.mean(data_nosoc) - np.mean(data_exploiter)
    elif 'shift_NSER' in metric_type:
        metric = np.mean(data_nosoc) - np.mean(data_explorer)
    elif 'shift_ETER' in metric_type:
        metric = np.mean(data_explorer) - np.mean(data_exploiter)
    elif 'shift_Nd+2' in metric_type:
        with open(fr'{data_dir}/{name}/val_matrix_best_Nd+2_perturb.bin','rb') as f:
            data_Nd2 = pickle.load(f)
        metric = np.mean(data_Nd2) - np.mean(data_og)

    elif 'JS_soc' in metric_type:
        h_og = np.histogram(data_og, bins=np.arange(0,1001,10))[0]
        h_nosoc = np.histogram(data_nosoc, bins=np.arange(0,1001,10))[0]
        metric = calc_JSdiv(h_og, h_nosoc)
    elif 'JS_exp' in metric_type:
        h_explorer = np.histogram(data_explorer, bins=np.arange(0,1001,10))[0]
        h_exploiter = np.histogram(data_exploiter, bins=np.arange(0,1001,10))[0]
        metric = calc_JSdiv(h_explorer, h_exploiter)

    else:
        print(f'{metric_type} not valid metric type')
    
    return metric



def plot_mult_EA_trends_multievo(names, val=None, save_name=None):

    # establish load directory
    root_dir = Path(__file__).parent.parent
    data_dir = Path(root_dir, r'data/simulation_data')

    # init plot details
    fig, ax1 = plt.subplots(figsize=(15,10)) 
    cmap = plt.get_cmap('hsv')
    cmap_range = len(names)
    lns = []
    val_avgs = []
    val_diffs = []
    top_vals_overall = np.zeros((3,0))
    group_top = []
    
    # iterate over each file
    for i, name in enumerate(names):
        print(name)

        with open(fr'{data_dir}/{name}/fitness_spread_per_generation.bin','rb') as f:
            data = pickle.load(f)
        num_steps,num_gen,num_eps,num_indivs = data.shape
        data_genxpop = np.mean(data, axis=2)
        top_data = np.min(data_genxpop, axis=1) # min : top
        avg_data = np.mean(data_genxpop, axis=1)
        avg_data_summed_across_indivs = np.sum(avg_data, axis=1) # sum bw each agent
        top_ind = np.argsort(avg_data_summed_across_indivs)[:1] # min : top
        avg_fit = [avg_data[i,:].round(0) for i in top_ind]
        for g,f in zip(top_ind, avg_fit):
            print(f'trn | gen {int(g)}: fit {f}')

        for indiv in range(num_indivs):
            l1 = ax1.plot(avg_data[:,indiv], 
                            label = f'avg {name}, ag{indiv}',
                            color=cmap(i/cmap_range), 
                            alpha=0.2
                            )
            lns.append(l1[0])

            l2 = ax1.plot(top_data[:,indiv], 
                            label = f'top {name}, ag{indiv}',
                            color=cmap(i/cmap_range), 
                            linestyle='dashed',
                            alpha=0.2
                            )
            lns.append(l2[0])

        # if top_data.shape[0] > 1000:
        #     top_data = top_data[-1000:]

        group_top.append(avg_data)

        # parse val results text file if exists
        if val is not None:
            if val == 'top': filename = 'val_results'
            elif val == 'cen': filename = 'val_results_cen'

            if Path(fr'{data_dir}/{name}/{filename}.txt').is_file():
                with open(fr'{data_dir}/{name}/{filename}.txt') as f:
                    lines = f.readlines()

                    val_data = np.zeros((len(lines)-1, 1 + num_indivs*2))
                    for n, line in enumerate(lines[1:]):
                        data = [item.strip() for item in line.split(' ')]

                        val_data[n,0] = data[1] # generation

                        data_raw = ''.join(data[4 : 4 + num_indivs])[1:-1]
                        val_data[n,1:1+num_indivs] = list(map(float,data_raw.split(','))) # train fitness

                        data_raw = ''.join(data[6 + num_indivs : 6 + num_indivs*2])[1:-1]
                        val_data[n,1+num_indivs:1+2*num_indivs] = list(map(float,data_raw.split(','))) # val fitness

                    top_ind = np.argsort(val_data[:,2])[:3] # min : top
                    top_gen = [val_data[i,0] for i in top_ind]
                    top_valfit = [val_data[i,1+num_indivs:1+2*num_indivs] for i in top_ind]
                    for g,f in zip(top_gen, top_valfit):
                        print(f'val | gen {int(g)}: fit {f}')

                    # print(top_gen)
                    # print(top_valfit)
                    # top_vals_current = np.array(([i], [top_gen[0]], [top_valfit[0]]))
                    # top_vals_overall = np.hstack((top_vals_overall, top_vals_current))

                train_fits = val_data[:,1:1+num_indivs]
                val_fits = val_data[:,1+num_indivs:1+2*num_indivs]

                val_diff = np.mean((val_fits - train_fits)**2)
                print(f'mean sq val diff: {val_diff}')
                val_diffs.append(val_diff)

                for indiv in range(num_indivs):
                    ax1.vlines(val_data[:,0], train_fits[:,indiv], val_fits[:,indiv],
                            color='black',
                            alpha=0.5
                            )
                    ax1.scatter(val_data[:,0], val_fits[:,indiv], color=cmap(i/cmap_range), edgecolor='black')

                    avg_val = np.mean(val_fits[:,indiv])
                    val_avgs.append(avg_val)

                    ax1.hlines(avg_val, i*5, data_genxpop.shape[0] + i*5,
                            color=cmap(i/cmap_range),
                            linestyle='dashed',
                            alpha=0.5
                            )
            else:
                print(fr'{data_dir}/{name}/{filename}.txt is not a file')
    
    # group_top = np.array(group_top)
    # est_trend = np.median(group_top, axis=0)
    # lt = ax1.plot(est_trend, 
    #                 label = f'Median of group top',
    #                 color='k', 
    #                 alpha=.5
    #                 )
    # lns.append(lt[0])

    ax1.set_xlabel('Generation')

    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='lower right')

    ax1.set_ylabel('Time to Find Patch')
    ax1.set_ylim(175,1475)

    # if val is not None:
    #     top_val_inds = np.argsort(top_vals_overall[2,:])
    #     top_reps = top_vals_overall[0,:][top_val_inds]
    #     top_gens = top_vals_overall[1,:][top_val_inds]
    #     top_vals = top_vals_overall[2,:][top_val_inds]

    #     top_num = 50
    #     for rep, gen, val_fit in zip(top_reps[:top_num], top_gens[:top_num], top_vals[:top_num]):
    #         print(f'overall val | rep {names[int(rep)]} | gen {int(gen)} | fit {int(val_fit)}')

    if save_name: 
        plt.savefig(fr'{data_dir}/{save_name}.png')
    plt.show()



# ------------------------------- relative occurence ---------------------------------------- #


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


    ### distance scaling, vis8 ###

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


    ### distance scaling, vis32 ###

    category_by_runtype = {
        'BD': (15,6,9,5,5,0,1,0,0),
        'BD/IS': (4,1,1,3,3,0,0,0,0),
        'IS': (2,1,2,7,0,0,0,0,0),
        'IS/DP': (0,0,0,0,2,5,5,2,3),
        'DP/BD': (0,0,0,1,5,4,8,2,0),
        'DP': (0,0,0,0,5,11,6,16,17),
    }
    runtype_by_category = {
        '0': (15,4,2,0,0,0),
        '0.1': (6,1,1,0,0,0),
        '0.2': (9,1,2,0,0,0),
        '0.3': (5,3,7,0,1,0),
        '0.4': (5,3,0,2,5,5),
        '0.5': (0,0,0,5,4,11),
        '0.6': (1,0,0,5,8,6),
        '0.8': (0,0,0,2,2,16),
        '1': (0,0,0,3,0,17),
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
    plt.savefig(fr'{data_dir}/relative_occurence_dist_vis32_{dpi}.png', dpi=dpi)
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


# ------------------------------- auxiliary illustration ---------------------------------------- #

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

def plot_colorbar(label, T, scheme):
    # Create a figure and a colorbar
    fig, ax = plt.subplots(figsize=(6, .5))
    fig.subplots_adjust(bottom=0.5)

    if scheme == 'dark':
        # Set the background color to black
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        nonface_color = 'white'
    else:
        nonface_color = 'black'

    # Create a colormap
    cmap = plt.cm.plasma

    # Create a norm object to scale the data values to the colormap
    norm = plt.Normalize(vmin=0, vmax=T)

    # Create a colorbar
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal')
    cb.set_label(label, fontsize=20, color=nonface_color)  # Increase font size for label and set color to white
    cb.ax.tick_params(labelsize=20, colors=nonface_color)  # Increase font size for ticks and set color to white

    # Set the color of the colorbar ticks and label to white
    cb.outline.set_edgecolor(nonface_color)
    plt.setp(plt.getp(cb.ax.axes, 'xticklabels'), color=nonface_color)

    plt.savefig(fr'colorbar_{label}_{scheme}.png', dpi=100)
    plt.show()


if __name__ == '__main__':

    # plot_hsv_dir(dpi=100)
    # plot_colorbar(label='Simulation Time', T = 100, scheme='light')
    
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

    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_PGPE_ss20_mom8_p50e20_dist_maxWF')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_PGPE_ss20_mom8_p50e20_dist_p9WF')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_PGPE_ss20_mom8_p50e20_dist_p8WF')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_PGPE_ss20_mom8_p50e20_dist_mlF')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_dist_mWF_n0_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_dist_msWF_n0_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_dist_sWF_n0_PGPE_ss20_mom8_p50e20')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_CNN14_FNN2_vis8_dist_ssWF_n0_PGPE_ss20_mom8_p50e20')

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
    # plot_mult_EA_trends([f'sc_lm_CNN14_FNN2_p50e20_vis64_lm100_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_lm_CNN14_FNN2_p50e20_vis64_lm100')

    # plot_mult_EA_trends([f'sc_lm_CNN14_FNNn8_p50e20_vis8_lm100_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_lm_CNN14_FNNn8_p50e20_vis8_lm100')
    # plot_mult_EA_trends([f'sc_lm_CNN14_FNNn8_p50e20_vis10_lm100_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_lm_CNN14_FNNn8_p50e20_vis10_lm100')
    # plot_mult_EA_trends([f'sc_lm_CNN14_FNNn8_p50e20_vis12_lm100_rep{x}' for x in range(20)], val='cen',
    #                     save_name='sc_lm_CNN14_FNNn8_p50e20_vis12_lm100')

    # names = []
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}')
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep{x}')
    # plot_mult_EA_trends_new(names, val='cen', group_est='median', save_name='sc_CNN14_FNN2_vis8_PGPE_ss20_mom8_p50e20')

    # names = []
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep{x}')
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed10k_rep{x}')
    # for x in range(20):
    # plot_mult_EA_trends_new(names, group_est='median', save_name='sc_CNN14_FNN2_vis24_PGPE_ss20_mom8_p50e20')

    # names = []
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep{x}')
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep{x}')
    # plot_mult_EA_trends_new(names, group_est='median', save_name='sc_CNN14_FNN2_vis8_PGPE_ss20_mom8_p50e20_dist_sWF')

    # names = []
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep{x}')
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep{x}')
    # plot_mult_EA_trends(names, val='cen', group_est='median', save_name='sc_CNN14_FNN2_vis8_PGPE_ss20_mom8_p50e20_dist_mlWF')
    # names = []
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep{x}')
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep{x}')
    # plot_mult_EA_trends(names, val='cen', group_est='median', save_name='sc_CNN14_FNN2_vis8_PGPE_ss20_mom8_p50e20_dist_sWF')
    # names = []
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep{x}')
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed10k_rep{x}')
    # plot_mult_EA_trends(names, val='cen', group_est='median', save_name='sc_CNN14_FNN2_vis24_PGPE_ss20_mom8_p50e20')
    # names = []
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep{x}')
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed10k_rep{x}')
    # plot_mult_EA_trends(names, val='cen', group_est='median', save_name='sc_CNN14_FNN2_vis12_PGPE_ss20_mom8_p50e20')
    # names = []
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep{x}')
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed10k_rep{x}')
    # plot_mult_EA_trends(names, val='cen', group_est='median', save_name='sc_CNN14_FNN2_vis12_PGPE_ss20_mom8_p50e20')

    
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_2xpinball_rep{x}' for x in range(20)], save_name='sc_CNN14_FNN2_p50e20_vis8_2xpinball')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis16_2xpinball_rep{x}' for x in range(17)], save_name='sc_CNN14_FNN2_p50e20_vis16_2xpinball')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2_p50e20_vis8_maxWF_2xpinball_rep{x}' for x in range(20)], save_name='sc_CNN14_FNN2_p50e20_vis8_maxWF_2xpinball')
    # plot_mult_EA_trends([f'sc_CNN17_FNN16_p50e20_vis16_2xpinball_rep{x}' for x in range(20)], save_name='sc_CNN17_FNN16_p50e20_vis16_2xpinball')


    # plot_mult_EA_trends([f'sc_CNN14_FNN2gaussian_vis8_rep{x}' for x in range(5)], val='cen', save_name='sc_CNN14_FNN2gaussian_vis8')
    # plot_mult_EA_trends([f'sc_CNN14_FNN2gaussian_vis8_proprio_rep{x}' for x in range(5)], val='cen', save_name='sc_CNN14_FNN2gaussian_vis8_proprio')
    # plot_mult_EA_trends([f'sc_CNN14_FNN16gaussian_vis8_rep{x}' for x in range(5)], val='cen', save_name='sc_CNN14_FNN16gaussian_vis8')

    # plot_mult_EA_trends([f'nowall_N5_CNN14_FNN16_vis8_proprio_rep{x}' for x in range(5)], val='cen', order='max', scoring='res', num_agents=5, max=600,)
    # plot_mult_EA_trends([f'nowall_N5_CNN14_FNN16gaussian_vis8_e20_rep{x}' for x in range(3)], order='max', scoring='res', num_agents=5, max=600,)
    # plot_mult_EA_trends([f'nowall_N5_CNN14_GRU16_vis8_g5k_rep{x}' for x in range(2)], val='cen', order='max', scoring='res', num_agents=5, max=600,)

    # names = [
    #     'sc_CNN18_FNN2x64_p50e20_vis32_fov97_rep0',
    #     'sc_CNN18_FNN2x64_p50e20_vis32_fov97_rep1',
    #     'sc_CNN18_FNN2x64_p50e20_vis32_fov97_rep2',
    # ]
    # plot_mult_EA_trends(names, save_name='CNN18_FNN2x64_vis32_fov97')

    names = [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(40)]
    plot_mult_EA_trends(names, val='cen', save_name='5x0-collinput')


    # groups = []
    # groups.append(('no walls, vis8', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('no walls, vis16', [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('vis8', [f'sc_CNN14_FNN2_p50e20_vis8_2xpinball_rep{x}' for x in range(20)]))
    # groups.append(('vis16', [f'sc_CNN14_FNN2_p50e20_vis16_2xpinball_rep{x}' for x in range(17)]))
    # groups.append(('maxWF', [f'sc_CNN14_FNN2_p50e20_vis8_maxWF_2xpinball_rep{x}' for x in range(20)]))
    # groups.append(('vis16, CNN17, FNN16', [f'sc_CNN17_FNN16_p50e20_vis16_2xpinball_rep{x}' for x in range(20)]))
    # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_singlecorner_2xpinball')
    # plot_mult_EA_trends_groups_endonly(groups, val='cen', save_name='groups_endonly_singlecorner_2xpinball')

    # names = []
    # n = 5
    # for name in [f'nowall_N5_CNN14_FNN2_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'nowall_N5_CNN14_FNN2_vis16_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'nowall_N5_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'nowall_N5_CNN14_FNN2_vis8_e40_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'nowall_N5_CNN14_FNN2_vis16_e40_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'nowall_N5_CNN14_FNN16_vis8_e40_rep{x}' for x in range(n)]:
    #     names.append(name)
    # n = 5
    # for name in [f'nowall_N5_CNN14_FNN16_vis8_proprio_rep{x}' for x in range(n)]:
    #     names.append(name)
    # n = 4
    # for name in [f'nowall_N5_CNN14_FNN16_vis8_fov875_e40_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'nowall_N5_CNN14_FNN16_vis16_fov94_e40_rep{x}' for x in range(n)]:
    #     names.append(name)
    # n = 2
    # for name in [f'nowall_N5_CNN14_GRU16_vis8_e40_rep{x}' for x in range(n)]:
    #     names.append(name)
    # n = 2
    # for name in [f'nowall_N5_CNN14_GRU16_vis8_g5k_rep{x}' for x in range(n)]:
    #     names.append(name)
    # plot_mult_EA_trends(names, order='max', scoring='res', num_agents=5, max=600, save_name='nowalls_N5')
    # plot_mult_EA_trends(names, val='cen', order='max', scoring='res', num_agents=5, max=600, save_name='nowalls_N5')
    # plot_mult_EA_trends(names, val='cen', order='max', scoring='res', num_agents=5, val_perturb='ghostexploiter', max=600, save_name='nowalls_N5')


    n = 40
    # names = []
    # for name in [f'sc_N6_NRW0_ND5_CNN18_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW0_ND5_CNN14_FNN64_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW0_ND5_CNN14_FNN16x2_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis12_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW0_ND5_CNN18_FNN16x2_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW0_ND5_CNN18_FNN64x2_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # plot_mult_EA_trends(names, save_name='groups_Nd5_extra')

    # groups = []
    # groups.append(('og',[f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('CNN18',[f'sc_N6_NRW0_ND5_CNN18_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('FNN64',[f'sc_N6_NRW0_ND5_CNN14_FNN64_vis8_rep{x}' for x in range(n)]))
    # groups.append(('FNN16x2',[f'sc_N6_NRW0_ND5_CNN14_FNN16x2_vis8_rep{x}' for x in range(n)]))
    # groups.append(('vis12',[f'sc_N6_NRW0_ND5_CNN14_FNN16_vis12_rep{x}' for x in range(n)]))
    # groups.append(('CNN18/FNN16x2',[f'sc_N6_NRW0_ND5_CNN18_FNN16x2_vis8_rep{x}' for x in range(n)]))
    # groups.append(('CNN18/FNN64x2',[f'sc_N6_NRW0_ND5_CNN18_FNN64x2_vis8_rep{x}' for x in range(n)]))
    # plot_mult_EA_trends_groups(groups, save_name='groups_Nd5_extra')

    # names = []
    # n = 3
    # for name in [f'nowall_N10_CNN14_FNN2_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'nowall_N10_CNN14_FNN2_vis16_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'nowall_N10_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # # plot_mult_EA_trends(names, order='max', scoring='res', num_agents=10, max=600, save_name='nowalls_N10')
    # plot_mult_EA_trends(names, val='cen', order='max', scoring='res', num_agents=10, max=600, save_name='nowalls_N10')
    # # # plot_mult_EA_trends(names, val='cen', order='max', scoring='res', num_agents=10, val_perturb='ghostexploiter', max=600, save_name='nowalls_N10')


    # groups = []
    # groups.append(('N5, FNN2, vis8', [f'nowall_N5_CNN14_FNN2_vis8_rep{x}' for x in range(5)]))
    # groups.append(('N5, FNN2, vis16', [f'nowall_N5_CNN14_FNN2_vis16_rep{x}' for x in range(5)]))
    # groups.append(('N5, FNN16, vis8', [f'nowall_N5_CNN14_FNN16_vis8_rep{x}' for x in range(5)]))
    # groups.append(('N5, FNN2, vis8, e40', [f'nowall_N5_CNN14_FNN2_vis8_e40_rep{x}' for x in range(5)]))
    # groups.append(('N5, FNN2, vis16, e40', [f'nowall_N5_CNN14_FNN2_vis16_e40_rep{x}' for x in range(5)]))
    # groups.append(('N5, FNN16, vis8, e40', [f'nowall_N5_CNN14_FNN16_vis8_e40_rep{x}' for x in range(5)]))
    # groups.append(('N5, FNN16, vis8, FOV875', [f'nowall_N5_CNN14_FNN16_vis8_fov875_e40_rep{x}' for x in range(4)]))
    # groups.append(('N5, FNN16, vis16, FOV94', [f'nowall_N5_CNN14_FNN16_vis16_fov94_e40_rep{x}' for x in range(4)]))
    # names = []
    # for name in [f'nowall_N5_CNN14_FNN2_vis8_rep{x}' for x in range(5)]:
    #     names.append(name)
    # for name in [f'nowall_N5_CNN14_FNN2_vis8_e40_rep{x}' for x in range(5)]:
    #     names.append(name)
    # groups.append(('N5, FNN2, vis8', names))
    # names = []
    # for name in [f'nowall_N5_CNN14_FNN2_vis16_rep{x}' for x in range(5)]:
    #     names.append(name)
    # for name in [f'nowall_N5_CNN14_FNN2_vis16_e40_rep{x}' for x in range(5)]:
    #     names.append(name)
    # groups.append(('N5, FNN2, vis16', names))
    # names = []
    # for name in [f'nowall_N5_CNN14_FNN16_vis8_rep{x}' for x in range(5)]:
    #     names.append(name)
    # for name in [f'nowall_N5_CNN14_FNN16_vis8_e40_rep{x}' for x in range(5)]:
    #     names.append(name)
    # groups.append(('N5, FNN16, vis8', names))
    # plot_mult_EA_trends_groups(groups, val='cen', order='max', scoring='res', num_agents=5, max=500, save_name='groups_nowalls_N5')
    # plot_mult_EA_trends_groups(groups, val='cen', order='max', scoring='res', num_agents=5, val_perturb='ghostexploiter', max=500, save_name='groups_nowalls_N5_valghost_exploiter')
    # plot_mult_EA_trends_groups(groups, val='cen', order='max', scoring='res', num_agents=5, val_perturb='ghostexplorer', max=500, save_name='groups_nowalls_N5_valghost_explorer')
    # plot_mult_EA_trends_groups_endonly(groups, val='cen', scoring='res', num_agents=5, max=920, save_name='groups_nowalls_N5_endonly')
    # plot_mult_EA_trends_groups_endonly_ghost(groups, val_type='res', num_agents=5, max=920, save_name='groups_nowalls_N5_valghost_endonly')
    # plot_mult_EA_trends_groups_endonly_ghost(groups, val_type='time', num_agents=5, max=1050, save_name='groups_nowalls_N5_valghost_endonly_time')
    # n = 4
    # groups.append(('N5, FNN2, vis8, ghost', [f'nowall_N5_CNN14_FNN16_vis8_e40_ghost_rep{x}' for x in range(n)]))
    # plot_mult_EA_trends_groups(groups, val='cen', order='max', scoring='res', num_agents=5, max=800, save_name='groups_nowalls_N5_ghosttrained')
    # plot_mult_EA_trends_groups_endonly(groups, val='cen', scoring='res', num_agents=5, max=920, save_name='groups_nowalls_N5_ghosttrained_endonly')
    # plot_mult_EA_trends_groups_endonly_ghost(groups, val_type='res', num_agents=5, max=920, save_name='groups_nowalls_N5_valghost_ghosttrained_endonly')
    # plot_mult_EA_trends_groups_endonly_ghost(groups, val_type='time', num_agents=5, max=1050, save_name='groups_nowalls_N5_valghost_ghosttrained_endonly_time')

    # groups = []
    # n = 4
    # groups.append(('N10, FNN2, vis8', [f'nowall_N10_CNN14_FNN2_vis8_rep{x}' for x in range(n)]))
    # groups.append(('N10, FNN2, vis16', [f'nowall_N10_CNN14_FNN2_vis16_rep{x}' for x in range(n)]))
    # groups.append(('N10, FNN16, vis8', [f'nowall_N10_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # plot_mult_EA_trends_groups(groups, val='cen', order='max', scoring='res', num_agents=10, max=500, save_name='groups_nowalls_N10')
    # plot_mult_EA_trends_groups(groups, val='cen', order='max', scoring='res', num_agents=10, val_perturb='ghostexploiter', max=500, save_name='groups_nowalls_N10_valghost_exploiter')
    # plot_mult_EA_trends_groups(groups, val='cen', order='max', scoring='res', num_agents=10, val_perturb='ghostexplorer', max=500, save_name='groups_nowalls_N10_valghost_explorer')
    # plot_mult_EA_trends_groups_endonly(groups, val='cen', scoring='res', num_agents=10, max=920, save_name='groups_nowalls_N10_endonly')
    # plot_mult_EA_trends_groups_endonly_ghost(groups, val_type='res', num_agents=10, max=920, save_name='groups_nowalls_N10_valghost_endonly')
    # plot_mult_EA_trends_groups_endonly_ghost(groups, val_type='time', num_agents=10, max=1050, save_name='groups_nowalls_N10_valghost_endonly_time')
    # n = 4
    # groups.append(('N10, FNN2, vis8, ghost', [f'nowall_N10_CNN14_FNN16_vis8_e20_ghost_rep{x}' for x in range(n)]))
    # plot_mult_EA_trends_groups(groups, val='cen', order='max', scoring='res', num_agents=10, max=800, save_name='groups_nowalls_N10_ghosttrained')
    # plot_mult_EA_trends_groups_endonly(groups, val='cen', scoring='res', num_agents=10, max=920, save_name='groups_nowalls_N10_ghosttrained_endonly')
    # plot_mult_EA_trends_groups_endonly_ghost(groups, val_type='res', num_agents=10, max=920, save_name='groups_nowalls_N10_valghost_ghosttrained_endonly')
    # plot_mult_EA_trends_groups_endonly_ghost(groups, val_type='time', num_agents=10, max=1050, save_name='groups_nowalls_N10_valghost_ghosttrained_endonly_time')



    # names = []
    # n = 4
    # for name in [f'nowall_N5_CNN14_FNN16_vis8_e40_ghost_rep{x}' for x in range(n)]:
    #     names.append(name)
    # plot_mult_EA_trends(names, val='cen', order='max', scoring='res', num_agents=5, max=800, save_name='nowalls_N5_ghost')

    # groups = []
    # n = 5
    # groups.append(('N5, FNN2, vis8, ghost', [f'nowall_N5_CNN14_FNN16_vis8_e40_ghost_rep{x}' for x in range(n)]))
    # plot_mult_EA_trends_groups(groups, val='cen', order='max', scoring='res', num_agents=5, max=800, save_name='groups_nowalls_N5_ghost')



    # groups = []
    # n = 20
    # groups.append(('NR0_ND2', [f'sc_N3_NRW0_ND2_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('NR1_ND1', [f'sc_N3_NRW1_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('NR2_ND0', [f'sc_N3_NRW2_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_sc_N3_NRWX_CNN14_FNN16_vis8')
    # plot_mult_EA_trends_groups_endonly(groups, val='cen', save_name='groups_sc_N3_NRWX_CNN14_FNN16_vis8_endonly')
    # plot_mult_EA_trends_groups_endonly_social_rand(groups, save_name='groups_sc_N3_NRWX_CNN14_FNN16_vis8_endonly_nosocial')
    # plot_mult_EA_trends_groups_endonly_social_rand(groups, save_name='groups_sc_N3_NRWX_CNN14_FNN16_vis8_endonly_allRW')
    # plot_mult_EA_trends_groups_endonly_social_rand(groups, save_name='groups_sc_N3_NRWX_CNN14_FNN16_vis8_endonly_allD')
    # plot_mult_EA_trends_groups_endonly_social_rand(groups, save_name='groups_sc_N3_NRWX_CNN14_FNN16_vis8_endonly_selfsocial')
    # plot_mult_EA_trends_groups_endonly_social_rand(groups, save_name='groups_sc_N3_NRWX_CNN14_FNN16_vis8_endonly_ghost')
    # plot_mult_EA_trends_groups_endonly_social_rand(groups, save_name='groups_sc_N3_NRWX_CNN14_FNN16_vis8_endonly_N2exploit')

    # groups = []
    # n = 20
    # groups.append(('NR0_ND10', [f'sc_N11_NRW0_ND10_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('NR5_ND5', [f'sc_N11_NRW5_ND5_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('NR10_ND0', [f'sc_N11_NRW10_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_sc_N11_NRWX_CNN14_FNN16_vis8')
    # plot_mult_EA_trends_groups_endonly(groups, val='cen', save_name='groups_sc_N11_NRWX_CNN14_FNN16_vis8_endonly')
    # plot_mult_EA_trends_groups_endonly_social_rand(groups, save_name='groups_sc_N11_NRWX_CNN14_FNN16_vis8_endonly_nosocial')
    # plot_mult_EA_trends_groups_endonly_social_rand(groups, save_name='groups_sc_N11_NRWX_CNN14_FNN16_vis8_endonly_allRW')
    # plot_mult_EA_trends_groups_endonly_social_rand(groups, save_name='groups_sc_N11_NRWX_CNN14_FNN16_vis8_endonly_allD')
    # plot_mult_EA_trends_groups_endonly_social_rand(groups, save_name='groups_sc_N11_NRWX_CNN14_FNN16_vis8_endonly_selfsocial')
    # plot_mult_EA_trends_groups_endonly_social_rand(groups, save_name='groups_sc_N11_NRWX_CNN14_FNN16_vis8_endonly_ghost')
    # plot_mult_EA_trends_groups_endonly_social_rand(groups, save_name='groups_sc_N11_NRWX_CNN14_FNN16_vis8_endonly_N2exploit')


    # groups = []
    # n = 20
    # groups.append(('NR0_ND20', [f'sc_N21_NRW0_ND20_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('NR10_ND10', [f'sc_N21_NRW10_ND10_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('NR20_ND0', [f'sc_N21_NRW20_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_sc_N21_NRWX_CNN14_FNN16_vis8')
    # plot_mult_EA_trends_groups_endonly(groups, val='cen', save_name='groups_sc_N21_NRWX_CNN14_FNN16_vis8_endonly')
    # plot_mult_EA_trends_groups_endonly_social_rand(groups, save_name='groups_sc_N21_NRWX_CNN14_FNN16_vis8_endonly_nosocial')
    # plot_mult_EA_trends_groups_endonly_social_rand(groups, save_name='groups_sc_N21_NRWX_CNN14_FNN16_vis8_endonly_allRW')
    # plot_mult_EA_trends_groups_endonly_social_rand(groups, save_name='groups_sc_N21_NRWX_CNN14_FNN16_vis8_endonly_allD')
    # plot_mult_EA_trends_groups_endonly_social_rand(groups, save_name='groups_sc_N21_NRWX_CNN14_FNN16_vis8_endonly_selfsocial')
    # plot_mult_EA_trends_groups_endonly_social_rand(groups, save_name='groups_sc_N21_NRWX_CNN14_FNN16_vis8_endonly_ghost')
    # plot_mult_EA_trends_groups_endonly_social_rand(groups, save_name='groups_sc_N21_NRWX_CNN14_FNN16_vis8_endonly_N2exploit')



    # groups = []
    # n = 40
    # groups.append(('NR5_ND0', [f'sc_N6_NRW5_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('NR4_ND1', [f'sc_N6_NRW4_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('NR3_ND2', [f'sc_N6_NRW3_ND2_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('NR2_ND3', [f'sc_N6_NRW2_ND3_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('NR1_ND4', [f'sc_N6_NRW1_ND4_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('NR0_ND5', [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # plot_mult_EA_trends_groups(groups, val='cen')
    # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_sc_N6_NRWX_CNN14_FNN16_vis8')
    # groups.append(('NR0_ND5', [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]))
    # groups.append(('NR1_ND4', [f'sc_N6_NRW1_ND4_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]))
    # groups.append(('NR2_ND3', [f'sc_N6_NRW2_ND3_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]))
    # groups.append(('NR3_ND2', [f'sc_N6_NRW3_ND2_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]))
    # groups.append(('NR4_ND1', [f'sc_N6_NRW4_ND1_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]))
    # groups.append(('NR5_ND0', [f'sc_N6_NRW5_ND0_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]))
    # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_sc_N6_NRWX_CNN14_FNN16_vis8_nocoll')
    # # plot_mult_EA_trends_groups_endonly(groups, val='cen', save_name='groups_sc_N6_NRWX_CNN14_FNN16_vis8_endonly')
    # # # plot_mult_EA_trends_groups_endonly_social_rand(groups, save_name='groups_sc_N6_NRWX_CNN14_FNN16_vis8_endonly_nosocial')
    # # # plot_mult_EA_trends_groups_endonly_social_rand(groups, save_name='groups_sc_N6_NRWX_CNN14_FNN16_vis8_endonly_RWp1')
    # # # plot_mult_EA_trends_groups_endonly_social_rand(groups, save_name='groups_sc_N6_NRWX_CNN14_FNN16_vis8_endonly_Dp1')
    # # # plot_mult_EA_trends_groups_endonly_social_rand(groups, save_name='groups_sc_N6_NRWX_CNN14_FNN16_vis8_endonly_allRW')
    # # # plot_mult_EA_trends_groups_endonly_social_rand(groups, save_name='groups_sc_N6_NRWX_CNN14_FNN16_vis8_endonly_allD')
    # # # plot_mult_EA_trends_groups_endonly_social_rand(groups, save_name='groups_sc_N6_NRWX_CNN14_FNN16_vis8_endonly_RWp1_allRW')
    # # # plot_mult_EA_trends_groups_endonly_social_rand(groups, save_name='groups_sc_N6_NRWX_CNN14_FNN16_vis8_endonly_Dp1_allD')
    # # # plot_mult_EA_trends_groups_endonly_social_rand(groups, save_name='groups_sc_N6_NRWX_CNN14_FNN16_vis8_endonly_selfsocial')
    # # # plot_mult_EA_trends_groups_endonly_social_rand(groups, save_name='groups_sc_N6_NRWX_CNN14_FNN16_vis8_endonly_selfsocial_nosocial')
    # # # plot_mult_EA_trends_groups_endonly_social_rand(groups, save_name='groups_sc_N6_NRWX_CNN14_FNN16_vis8_endonly_ghost')
    # # # plot_mult_EA_trends_groups_endonly_social_rand(groups, save_name='groups_sc_N6_NRWX_CNN14_FNN16_vis8_endonly_N2exploit')
    # # for group in groups:
    # #     plot_mult_EA_trends_groups_endonly_social_rand_indivperturbs([group], max=None, title=group[0], 
    # #                                 save_name=fr'groups_sc_N6_NRWX_CNN14_FNN16_vis8_endonly_indivperturbs_{group[0]}')
    # # plot_mult_EA_trends_groups_endonly_social_rand_indivperturbs(groups, max=None, title='All N6', 
    # #                             save_name=fr'groups_sc_N6_NRWX_CNN14_FNN16_vis8_endonly_indivperturbs_allN6')

    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,save_name='groups_sc_N6_NRWX_CNN14_FNN16_vis8_endonly_numfound_bees')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,save_name='groups_sc_N6_NRWX_CNN14_FNN16_vis8_endonly_mean_bees')

    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,save_name='groups_sc_N6_NRWX_CNN14_FNN16_vis8_endonly_nosocial_numfound')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='pure_follow', save_name='groups_sc_N6_NRWX_CNN14_FNN16_vis8_endonly_nosocial_numfound_foll')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='pure_nav', save_name='groups_sc_N6_NRWX_CNN14_FNN16_vis8_endonly_nosocial_numfound_nav')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='nav_follhurts', save_name='groups_sc_N6_NRWX_CNN14_FNN16_vis8_endonly_nosocial_numfound_nav_fhurts')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='nav_follhelps', save_name='groups_sc_N6_NRWX_CNN14_FNN16_vis8_endonly_nosocial_numfound_nav_fhelps')

    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,save_name='groups_sc_N6_NRWX_CNN14_FNN16_vis8_endonly_nosocial_mean')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='pure_follow',save_name='groups_sc_N6_NRWX_CNN14_FNN16_vis8_endonly_nosocial_mean_foll')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='pure_nav',save_name='groups_sc_N6_NRWX_CNN14_FNN16_vis8_endonly_nosocial_mean_nav')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='nav_follhurts',save_name='groups_sc_N6_NRWX_CNN14_FNN16_vis8_endonly_nosocial_mean_nav_fhurts')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='nav_follhelps',save_name='groups_sc_N6_NRWX_CNN14_FNN16_vis8_endonly_nosocial_mean_nav_fhelps')

    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,save_name='groups_sc_N6_NRWX_CNN14_FNN16_vis8_endonly_N2exploit_numfound')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,save_name='groups_sc_N6_NRWX_CNN14_FNN16_vis8_endonly_N2exploit_mean')

    # # plot_mult_EA_trends_groups_endonly_divs(groups,save_name='groups_sc_N6_NRWX_CNN14_FNN16_vis8_endonly_nosocial_KL_twoslope')
    # # plot_mult_EA_trends_groups_endonly_divs(groups,save_name='groups_sc_N6_NRWX_CNN14_FNN16_vis8_endonly_ghostexploiter_KL_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_N6_NRWX_CNN14_FNN16_vis8_endonly_nosocial_JS_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_N6_NRWX_CNN14_FNN16_vis8_endonly_ghostexploiter_JS_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_N6_NRWX_CNN14_FNN16_vis8_endonly_ghostexplorer_JS_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_N6_NRWX_CNN14_FNN16_vis8_endonly_expexp_JS_twoslope')


    # groups = []
    # names = []
    # # for x in range(20):
    # #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}')
    # #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep{x}')
    # # groups.append(('ND0', names))
    # n = 40
    # groups.append(('ND1', [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('ND2', [f'sc_N3_NRW0_ND2_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('ND3', [f'sc_N4_NRW0_ND3_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('ND4', [f'sc_N5_NRW0_ND4_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('ND5', [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # # groups.append(('ND3', [f'sc_N4_NRW0_ND3_CNN14_FNN16_vis8_rep{x+40}' for x in range(n)]))
    # # groups.append(('ND4', [f'sc_N5_NRW0_ND4_CNN14_FNN16_vis8_rep{x+40}' for x in range(n)]))
    # # groups.append(('ND5', [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_rep{x+40}' for x in range(n)]))
    # # n = 20
    # # groups.append(('ND10', [f'sc_N11_NRW0_ND10_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # # groups.append(('ND20', [f'sc_N21_NRW0_ND20_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))

    # # plot_mult_EA_trends_groups(groups, val='cen', group_est='median', save_name='groups_sc_NX_NRW0_CNN14_FNN16_vis8')
    # # plot_mult_EA_trends_groups_endonly(groups, val='cen', title='NR0', save_name='groups_sc_NX_NRW0_CNN14_FNN16_vis8_endonly')
    # # plot_mult_EA_trends_groups_endonly_split(groups, val='cen', title='NR0', trunc=True, save_name='groups_sc_NX_NRW0_CNN14_FNN16_vis8_endonly_split')
    # # plot_mult_EA_trends_groups_endonly_social_rand(groups, save_name='groups_sc_NX_NRW0_CNN14_FNN16_vis8_endonly_nosocial')

    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=False,save_name='groups_sc_NX_NRW0_CNN14_FNN16_vis8_endonly_numfound')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,save_name='groups_sc_NX_NRW0_CNN14_FNN16_vis8_endonly_numfound_bees')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=False,save_name='groups_sc_NX_NRW0_CNN14_FNN16_vis8_endonly_mean')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,save_name='groups_sc_NX_NRW0_CNN14_FNN16_vis8_endonly_mean_bees')

    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,save_name='groups_sc_NX_NRW0_CNN14_FNN16_vis8_endonly_nosocial_numfound')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='pure_follow', save_name='groups_sc_NX_NRW0_CNN14_FNN16_vis8_endonly_nosocial_numfound_foll')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='pure_nav', save_name='groups_sc_NX_NRW0_CNN14_FNN16_vis8_endonly_nosocial_numfound_nav')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='nav_follhurts', save_name='groups_sc_NX_NRW0_CNN14_FNN16_vis8_endonly_nosocial_numfound_nav_fhurts')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='nav_follhelps', save_name='groups_sc_NX_NRW0_CNN14_FNN16_vis8_endonly_nosocial_numfound_nav_fhelps')

    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,save_name='groups_sc_NX_NRW0_CNN14_FNN16_vis8_endonly_nosocial_mean')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='pure_follow',save_name='groups_sc_NX_NRW0_CNN14_FNN16_vis8_endonly_nosocial_mean_foll')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='pure_nav',save_name='groups_sc_NX_NRW0_CNN14_FNN16_vis8_endonly_nosocial_mean_nav')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='nav_follhurts',save_name='groups_sc_NX_NRW0_CNN14_FNN16_vis8_endonly_nosocial_mean_nav_fhurts')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='nav_follhelps',save_name='groups_sc_NX_NRW0_CNN14_FNN16_vis8_endonly_nosocial_mean_nav_fhelps')

    # # plot_mult_EA_trends_groups_endonly_divs(groups,save_name='groups_sc_NX_NRW0_CNN14_FNN16_vis8_endonly_nosocial_KL_twoslope')
    # # plot_mult_EA_trends_groups_endonly_divs(groups,save_name='groups_sc_NX_NRW0_CNN14_FNN16_vis8_endonly_ghostexploiter_KL_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_NX_NRW0_CNN14_FNN16_vis8_endonly_nosocial_JS_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_NX_NRW0_CNN14_FNN16_vis8_endonly_ghostexploiter_JS_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_NX_NRW0_CNN14_FNN16_vis8_endonly_ghostexplorer_JS_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_NX_NRW0_CNN14_FNN16_vis8_endonly_expexp_JS_twoslope')


    # groups = []
    # names = []
    # n = 40
    # groups.append(('N2', [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('N3', [f'sc_N3_NRW1_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('N4', [f'sc_N4_NRW2_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('N5', [f'sc_N5_NRW3_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('N6', [f'sc_N6_NRW4_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))

    # # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,save_name='groups_sc_NX_ND1_CNN14_FNN16_vis8_endonly_numfound_bees')
    # # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,save_name='groups_sc_NX_ND1_CNN14_FNN16_vis8_endonly_mean_bees')

    # # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,save_name='groups_sc_NX_ND1_CNN14_FNN16_vis8_endonly_nosocial_numfound')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='pure_follow', save_name='groups_sc_NX_ND1_CNN14_FNN16_vis8_endonly_nosocial_numfound_foll')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='pure_nav', save_name='groups_sc_NX_ND1_CNN14_FNN16_vis8_endonly_nosocial_numfound_nav')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='nav_follhurts', save_name='groups_sc_NX_ND1_CNN14_FNN16_vis8_endonly_nosocial_numfound_nav_fhurts')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='nav_follhelps', save_name='groups_sc_NX_ND1_CNN14_FNN16_vis8_endonly_nosocial_numfound_nav_fhelps')

    # # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,save_name='groups_sc_NX_ND1_CNN14_FNN16_vis8_endonly_nosocial_mean')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='pure_follow',save_name='groups_sc_NX_ND1_CNN14_FNN16_vis8_endonly_nosocial_mean_foll')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='pure_nav',save_name='groups_sc_NX_ND1_CNN14_FNN16_vis8_endonly_nosocial_mean_nav')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='nav_follhurts',save_name='groups_sc_NX_ND1_CNN14_FNN16_vis8_endonly_nosocial_mean_nav_fhurts')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='nav_follhelps',save_name='groups_sc_NX_ND1_CNN14_FNN16_vis8_endonly_nosocial_mean_nav_fhelps')

    # # plot_mult_EA_trends_groups_endonly_divs(groups,save_name='groups_sc_NX_ND1_CNN14_FNN16_vis8_endonly_nosocial_KL_twoslope')
    # # plot_mult_EA_trends_groups_endonly_divs(groups,save_name='groups_sc_NX_ND1_CNN14_FNN16_vis8_endonly_ghostexploiter_KL_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_NX_ND1_CNN14_FNN16_vis8_endonly_nosocial_JS_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_NX_ND1_CNN14_FNN16_vis8_endonly_ghostexploiter_JS_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_NX_ND1_CNN14_FNN16_vis8_endonly_ghostexplorer_JS_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_NX_ND1_CNN14_FNN16_vis8_endonly_expexp_JS_twoslope')


    # groups = []
    # names = []
    # n = 40
    # groups.append(('N4', [f'sc_N3_NRW0_ND2_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('N5', [f'sc_N4_NRW1_ND2_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('N5', [f'sc_N5_NRW2_ND2_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('N6', [f'sc_N6_NRW3_ND2_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))

    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,save_name='groups_sc_NX_ND2_CNN14_FNN16_vis8_endonly_numfound_bees')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,save_name='groups_sc_NX_ND2_CNN14_FNN16_vis8_endonly_mean_bees')

    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,save_name='groups_sc_NX_ND2_CNN14_FNN16_vis8_endonly_nosocial_numfound')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='pure_follow', save_name='groups_sc_NX_ND2_CNN14_FNN16_vis8_endonly_nosocial_numfound_foll')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='pure_nav', save_name='groups_sc_NX_ND2_CNN14_FNN16_vis8_endonly_nosocial_numfound_nav')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='nav_follhurts', save_name='groups_sc_NX_ND2_CNN14_FNN16_vis8_endonly_nosocial_numfound_nav_fhurts')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='nav_follhelps', save_name='groups_sc_NX_ND2_CNN14_FNN16_vis8_endonly_nosocial_numfound_nav_fhelps')

    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,save_name='groups_sc_NX_ND2_CNN14_FNN16_vis8_endonly_nosocial_mean')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='pure_follow',save_name='groups_sc_NX_ND2_CNN14_FNN16_vis8_endonly_nosocial_mean_foll')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='pure_nav',save_name='groups_sc_NX_ND2_CNN14_FNN16_vis8_endonly_nosocial_mean_nav')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='nav_follhurts',save_name='groups_sc_NX_ND2_CNN14_FNN16_vis8_endonly_nosocial_mean_nav_fhurts')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='nav_follhelps',save_name='groups_sc_NX_ND2_CNN14_FNN16_vis8_endonly_nosocial_mean_nav_fhelps')

    # # plot_mult_EA_trends_groups_endonly_divs(groups,save_name='groups_sc_NX_ND2_CNN14_FNN16_vis8_endonly_nosocial_KL_twoslope')
    # # plot_mult_EA_trends_groups_endonly_divs(groups,save_name='groups_sc_NX_ND2_CNN14_FNN16_vis8_endonly_ghostexploiter_KL_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_NX_ND2_CNN14_FNN16_vis8_endonly_nosocial_JS_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_NX_ND2_CNN14_FNN16_vis8_endonly_ghostexploiter_JS_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_NX_ND2_CNN14_FNN16_vis8_endonly_ghostexplorer_JS_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_NX_ND2_CNN14_FNN16_vis8_endonly_expexp_JS_twoslope')


    # groups = []
    # names = []
    # n = 40
    # groups.append(('N4', [f'sc_N4_NRW0_ND3_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('N5', [f'sc_N5_NRW1_ND3_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('N6', [f'sc_N6_NRW2_ND3_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))

    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_NX_ND3_CNN14_FNN16_vis8_endonly_nosocial_JS_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_NX_ND3_CNN14_FNN16_vis8_endonly_ghostexploiter_JS_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_NX_ND3_CNN14_FNN16_vis8_endonly_ghostexplorer_JS_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_NX_ND3_CNN14_FNN16_vis8_endonly_expexp_JS_twoslope')

    # groups = []
    # names = []
    # n = 40
    # groups.append(('N5', [f'sc_N5_NRW0_ND4_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('N6', [f'sc_N6_NRW1_ND4_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))

    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_NX_ND4_CNN14_FNN16_vis8_endonly_nosocial_JS_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_NX_ND4_CNN14_FNN16_vis8_endonly_ghostexploiter_JS_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_NX_ND4_CNN14_FNN16_vis8_endonly_ghostexplorer_JS_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_NX_ND4_CNN14_FNN16_vis8_endonly_expexp_JS_twoslope')

    # groups = []
    # names = []
    # n = 40
    # groups.append(('N3', [f'sc_N3_NRW1_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('N4', [f'sc_N4_NRW1_ND2_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('N5', [f'sc_N5_NRW1_ND3_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('N6', [f'sc_N6_NRW1_ND4_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))

    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_NX_NRW1_CNN14_FNN16_vis8_endonly_nosocial_JS_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_NX_NRW1_CNN14_FNN16_vis8_endonly_ghostexploiter_JS_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_NX_NRW1_CNN14_FNN16_vis8_endonly_ghostexplorer_JS_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_NX_NRW1_CNN14_FNN16_vis8_endonly_expexp_JS_twoslope')

    # groups = []
    # names = []
    # n = 40
    # groups.append(('N4', [f'sc_N4_NRW2_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('N5', [f'sc_N5_NRW2_ND2_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('N6', [f'sc_N6_NRW2_ND3_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))

    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_NX_NRW2_CNN14_FNN16_vis8_endonly_nosocial_JS_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_NX_NRW2_CNN14_FNN16_vis8_endonly_ghostexploiter_JS_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_NX_NRW2_CNN14_FNN16_vis8_endonly_ghostexplorer_JS_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_NX_NRW2_CNN14_FNN16_vis8_endonly_expexp_JS_twoslope')

    # groups = []
    # names = []
    # n = 40
    # groups.append(('N5', [f'sc_N5_NRW3_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('N6', [f'sc_N6_NRW3_ND2_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))

    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_NX_NRW3_CNN14_FNN16_vis8_endonly_nosocial_JS_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_NX_NRW3_CNN14_FNN16_vis8_endonly_ghostexploiter_JS_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_NX_NRW3_CNN14_FNN16_vis8_endonly_ghostexplorer_JS_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_NX_NRW3_CNN14_FNN16_vis8_endonly_expexp_JS_twoslope')



    # groups = []
    # names = []
    # # for x in range(20):
    # #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}')
    # #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep{x}')
    # # groups.append(('ND0', names))
    # groups.append(('ND0-ghost', [f'sc_N1_NRW0_ND0_CNN14_FNN16_vis8_ghost_rep{x}' for x in range(20)]))
    # groups.append(('ND1', [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(40)]))
    # groups.append(('ND1-ghost', [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_ghost_rep{x}' for x in range(20)]))
    # groups.append(('ND5', [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_rep{x}' for x in range(40)]))
    # groups.append(('ND5-ghost', [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_ghost_rep{x}' for x in range(20)]))

    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,save_name='groups_sc_NX_ghost_CNN14_FNN16_vis8_endonly_numfound_bees')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,save_name='groups_sc_NX_ghost_CNN14_FNN16_vis8_endonly_mean_bees')

    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,save_name='groups_sc_NX_ghost_CNN14_FNN16_vis8_endonly_nosocial_numfound')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='pure_follow', save_name='groups_sc_NX_ghost_CNN14_FNN16_vis8_endonly_nosocial_numfound_foll')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='pure_nav', save_name='groups_sc_NX_ghost_CNN14_FNN16_vis8_endonly_nosocial_numfound_nav')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='nav_follhurts', save_name='groups_sc_NX_ghost_CNN14_FNN16_vis8_endonly_nosocial_numfound_nav_fhurts')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='nav_follhelps', save_name='groups_sc_NX_ghost_CNN14_FNN16_vis8_endonly_nosocial_numfound_nav_fhelps')

    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,save_name='groups_sc_NX_ghost_CNN14_FNN16_vis8_endonly_nosocial_mean')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='pure_follow',save_name='groups_sc_NX_ghost_CNN14_FNN16_vis8_endonly_nosocial_mean_foll')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='pure_nav',save_name='groups_sc_NX_ghost_CNN14_FNN16_vis8_endonly_nosocial_mean_nav')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='nav_follhurts',save_name='groups_sc_NX_ghost_CNN14_FNN16_vis8_endonly_nosocial_mean_nav_fhurts')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='nav_follhelps',save_name='groups_sc_NX_ghost_CNN14_FNN16_vis8_endonly_nosocial_mean_nav_fhelps')

    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,save_name='groups_sc_NX_ghost_CNN14_FNN16_vis8_endonly_N2exploit_numfound')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,save_name='groups_sc_NX_ghost_CNN14_FNN16_vis8_endonly_N2exploit_mean')

    # # plot_mult_EA_trends_groups_endonly_divs(groups,save_name='groups_sc_NX_ghost_CNN14_FNN16_vis8_endonly_ghostexploiter_KL_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_NX_ghost_CNN14_FNN16_vis8_endonly_ghostexploiter_JS_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_NX_ghost_CNN14_FNN16_vis8_endonly_ghostexplorer_JS_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_NX_ghost_CNN14_FNN16_vis8_endonly_expexp_JS_twoslope')


    # groups = []
    # names = []
    # n = 20
    # groups.append(('OG', [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('Ainit0_Sinit400', [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_Ainit0_Sinit400_rep{x}' for x in range(n)]))
    # groups.append(('Ainit0_Sinit200', [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_Ainit0_Sinit200_rep{x}' for x in range(n)]))
    # groups.append(('Ainit400_Sinit400', [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_Ainit400_Sinit400_rep{x}' for x in range(n)]))
    # groups.append(('Ainit400_Sinit200', [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_Ainit400_Sinit200_rep{x}' for x in range(n)]))
    # groups.append(('OG', [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('Ainit0_Sinit400', [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_Ainit0_Sinit400_rep{x}' for x in range(n)]))
    # groups.append(('Ainit0_Sinit200', [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_Ainit0_Sinit200_rep{x}' for x in range(n)]))
    # groups.append(('Ainit400_Sinit400', [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_Ainit400_Sinit400_rep{x}' for x in range(n)]))
    # groups.append(('Ainit400_Sinit200', [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_Ainit400_Sinit200_rep{x}' for x in range(n)]))

    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,save_name='groups_sc_NX_init_CNN14_FNN16_vis8_endonly_numfound_bees')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,save_name='groups_sc_NX_init_CNN14_FNN16_vis8_endonly_mean_bees')

    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,save_name='groups_sc_NX_init_CNN14_FNN16_vis8_endonly_nosocial_numfound')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='pure_follow', save_name='groups_sc_NX_init_CNN14_FNN16_vis8_endonly_nosocial_numfound_foll')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='pure_nav', save_name='groups_sc_NX_init_CNN14_FNN16_vis8_endonly_nosocial_numfound_nav')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='nav_follhurts', save_name='groups_sc_NX_init_CNN14_FNN16_vis8_endonly_nosocial_numfound_nav_fhurts')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='nav_follhelps', save_name='groups_sc_NX_init_CNN14_FNN16_vis8_endonly_nosocial_numfound_nav_fhelps')

    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,save_name='groups_sc_NX_init_CNN14_FNN16_vis8_endonly_nosocial_mean')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='pure_follow',save_name='groups_sc_NX_init_CNN14_FNN16_vis8_endonly_nosocial_mean_foll')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='pure_nav',save_name='groups_sc_NX_init_CNN14_FNN16_vis8_endonly_nosocial_mean_nav')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='nav_follhurts',save_name='groups_sc_NX_init_CNN14_FNN16_vis8_endonly_nosocial_mean_nav_fhurts')
    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='nav_follhelps',save_name='groups_sc_NX_init_CNN14_FNN16_vis8_endonly_nosocial_mean_nav_fhelps')

    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,save_name='groups_sc_NX_init_CNN14_FNN16_vis8_endonly_N2exploit_numfound')
    # # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='pure_follow', save_name='groups_sc_NX_init_CNN14_FNN16_vis8_endonly_N2exploit_numfound_foll')
    # # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='pure_nav', save_name='groups_sc_NX_init_CNN14_FNN16_vis8_endonly_N2exploit_numfound_nav')
    # # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='nav_follhurts', save_name='groups_sc_NX_init_CNN14_FNN16_vis8_endonly_N2exploit_numfound_nav_fhurts')
    # # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='nav_follhelps', save_name='groups_sc_NX_init_CNN14_FNN16_vis8_endonly_N2exploit_numfound_nav_fhelps')

    # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,save_name='groups_sc_NX_init_CNN14_FNN16_vis8_endonly_N2exploit_mean')
    # # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='pure_follow',save_name='groups_sc_NX_init_CNN14_FNN16_vis8_endonly_N2exploit_mean_foll')
    # # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='pure_nav',save_name='groups_sc_NX_init_CNN14_FNN16_vis8_endonly_N2exploit_mean_nav')
    # # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='nav_follhurts',save_name='groups_sc_NX_init_CNN14_FNN16_vis8_endonly_N2exploit_mean_nav_fhurts')
    # # # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,highlight='nav_follhelps',save_name='groups_sc_NX_init_CNN14_FNN16_vis8_endonly_N2exploit_mean_nav_fhelps')

    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_NX_init_CNN14_FNN16_vis8_endonly_nosocial_JS_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_NX_init_CNN14_FNN16_vis8_endonly_ghostexploiter_JS_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_NX_init_CNN14_FNN16_vis8_endonly_ghostexplorer_JS_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_NX_init_CNN14_FNN16_vis8_endonly_expexp_JS_twoslope')


    # groups = []
    # names = []
    # n = 20
    # groups.append(('OG', [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('Ainit0_Sinit400', [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_Ainit0_Sinit400_rep{x}' for x in range(n)]))
    # groups.append(('Ainit0_Sinit200', [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_Ainit0_Sinit200_rep{x}' for x in range(n)]))
    # groups.append(('ND0-ghost', [f'sc_N1_NRW0_ND0_CNN14_FNN16_vis8_ghost_rep{x}' for x in range(n)]))
    # # groups.append(('ND1-ghost', [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_ghost_rep{x}' for x in range(n)])) # 2 other agents - 1 on patch + 1 off
    # # groups.append(('Ainit400_Sinit400', [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_Ainit400_Sinit400_rep{x}' for x in range(n)]))
    # # groups.append(('Ainit400_Sinit200', [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_Ainit400_Sinit200_rep{x}' for x in range(n)]))
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_N1-init_CNN14_FNN16_vis8_endonly_nosocial_JS_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_N1-init_CNN14_FNN16_vis8_endonly_ghostexploiter_JS_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_N1-init_CNN14_FNN16_vis8_endonly_ghostexplorer_JS_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_N1-init_CNN14_FNN16_vis8_endonly_expexp_JS_twoslope')

    # groups = []
    # names = []
    # n = 20
    # groups.append(('OG', [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('Ainit0_Sinit400', [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_Ainit0_Sinit400_rep{x}' for x in range(n)]))
    # groups.append(('Ainit0_Sinit200', [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_Ainit0_Sinit200_rep{x}' for x in range(n)]))
    # groups.append(('ND0-ghost', [f'sc_N1_NRW0_ND0_CNN14_FNN16_vis8_ghost_rep{x}' for x in range(n)])) # 1 other agent
    # # groups.append(('ND5-ghost', [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_ghost_rep{x}' for x in range(n)])) # 6 other agents - 5 off (anywhere in map)
    # # groups.append(('Ainit400_Sinit400', [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_Ainit400_Sinit400_rep{x}' for x in range(n)]))
    # # groups.append(('Ainit400_Sinit200', [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_Ainit400_Sinit200_rep{x}' for x in range(n)]))
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_N5-init_CNN14_FNN16_vis8_endonly_nosocial_JS_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_N5-init_CNN14_FNN16_vis8_endonly_ghostexploiter_JS_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_N5-init_CNN14_FNN16_vis8_endonly_ghostexplorer_JS_twoslope')
    # plot_mult_EA_trends_groups_endonly_divs(groups,sideplot=True,save_name='groups_sc_N5-init_CNN14_FNN16_vis8_endonly_expexp_JS_twoslope')




    # plot_social_table_DR(metric_type='fit_og', dpi=100)
    # # plot_social_table_DR(metric_type='fit_nosoc', dpi=100)
    # # plot_social_table_DR(metric_type='fit_exploiter', dpi=100)
    # # plot_social_table_DR(metric_type='fit_explorer', dpi=100)
    # # plot_social_table_DR(metric_type='meanshift_OGNS', metric_thresh=100, dpi=100)
    # # plot_social_table_DR(metric_type='meanshift_NSET', metric_thresh=100, dpi=100)
    # plot_social_table_DR(metric_type='meanshift_ETER', metric_thresh=100, dpi=100)
    # plot_social_table_DR(metric_type='meanshift_OGNS', dpi=100)
    # plot_social_table_DR(metric_type='meanshift_NSET', dpi=100)
    # plot_social_table_DR(metric_type='meanshift_NSER', dpi=100)
    # plot_social_table_DR(metric_type='meanshift_ETER', dpi=100)
    # plot_social_table_DR(metric_type='JSspatial_OGNS', dpi=100)
    # plot_social_table_DR(metric_type='JSspatial_NSET', dpi=100)
    # plot_social_table_DR(metric_type='JSspatial_NSER', dpi=100)
    # plot_social_table_DR(metric_type='JSspatial_ETER', dpi=100)
    # plot_social_table_DR(metric_type='JSspatial_NSET', metric_thresh=100, dpi=100)
    # plot_social_table_DR(metric_type='JSspatial_ETER', metric_thresh=0.1, dpi=100)
    # plot_social_table_DR(metric_type='learning_time', metric_thresh=500, dpi=100)
    # plot_social_table_DR(metric_type='dirent_OG', dpi=100)
    # plot_social_table_DR(metric_type='dirent_NS', dpi=100)
    # plot_social_table_DR(metric_type='dirent_ET', dpi=100)
    # plot_social_table_DR(metric_type='dirent_ER', dpi=100)
    # plot_social_table_DR(metric_type='meanshift_Nd+2', dpi=100)
    # plot_social_table_DR(metric_type='meanshift_Nr+2', dpi=100)

    # plot_social_table_DR(metric_type='fit_og', coll=False, dpi=100)
    # plot_social_table_DR(metric_type='meanshift_NSET', coll=False, dpi=100)
    # plot_social_table_DR(metric_type='meanshift_NSER', coll=False, dpi=100)
    # plot_social_table_DR(metric_type='meanshift_ETER', coll=False, dpi=100)
    # plot_social_table_DR(metric_type='meanshift_ETER', coll=False, metric_thresh=100, dpi=100)
    # plot_social_table_DR(metric_type='JSspatial_NSET', coll=False, dpi=100)
    # plot_social_table_DR(metric_type='JSspatial_NSER', coll=False, dpi=100)
    # plot_social_table_DR(metric_type='JSspatial_ETER', coll=False, dpi=100)
    # plot_social_table_DR(metric_type='JSspatial_ETER', coll=False, metric_thresh=0.1, dpi=100)
    # plot_social_table_DR(metric_type='learning_time', coll=False, metric_thresh=500, dpi=100)
    # plot_social_table_DR(metric_type='dirent_NS', coll=False, dpi=100)
    # plot_social_table_DR(metric_type='meanshift_Nd+2', coll=False, dpi=100)
    # plot_social_table_DR(metric_type='meanshift_Nr+2', coll=False, dpi=100)


    groups = []
    names = []
    n = 40

    # groups.append(('0x0', [f'sc_N1_NRW0_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('1x0', [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('2x0', [f'sc_N3_NRW0_ND2_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('3x0', [f'sc_N4_NRW0_ND3_CNN14_FNN16_vis8_rep{x+40}' for x in range(n)]))
    # groups.append(('4x0', [f'sc_N5_NRW0_ND4_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('5x0', [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_rep{x+40}' for x in range(n)])) #
    # groups.append(('0x1', [f'sc_N2_NRW1_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('1x1', [f'sc_N3_NRW1_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('2x1', [f'sc_N4_NRW1_ND2_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('3x1', [f'sc_N5_NRW1_ND3_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('4x1', [f'sc_N6_NRW1_ND4_CNN14_FNN16_vis8_rep{x}' for x in range(n)])) #
    # groups.append(('0x2', [f'sc_N3_NRW2_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('1x2', [f'sc_N4_NRW2_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('2x2', [f'sc_N5_NRW2_ND2_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('3x2', [f'sc_N6_NRW2_ND3_CNN14_FNN16_vis8_rep{x}' for x in range(n)])) #
    # groups.append(('0x3', [f'sc_N4_NRW3_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('1x3', [f'sc_N5_NRW3_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('2x3', [f'sc_N6_NRW3_ND2_CNN14_FNN16_vis8_rep{x}' for x in range(n)])) #
    # groups.append(('0x4', [f'sc_N5_NRW4_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('1x4', [f'sc_N6_NRW4_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)])) #
    # groups.append(('0x5', [f'sc_N6_NRW5_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)])) #

    # groups.append(('0x5', [f'sc_N6_NRW5_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)])) #
    # groups.append(('1x4', [f'sc_N6_NRW4_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)])) #
    # groups.append(('2x3', [f'sc_N6_NRW3_ND2_CNN14_FNN16_vis8_rep{x}' for x in range(n)])) #
    # groups.append(('3x2', [f'sc_N6_NRW2_ND3_CNN14_FNN16_vis8_rep{x}' for x in range(n)])) #
    # groups.append(('4x1', [f'sc_N6_NRW1_ND4_CNN14_FNN16_vis8_rep{x}' for x in range(n)])) #
    # groups.append(('5x0', [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_rep{x+40}' for x in range(n)])) #

    # groups.append(('All-Map', [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)])) # All-Map for _means
    # # groups.append(('1x0xAg400', [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)])) # 4xx used as max Sinit_dist
    # groups.append(('300', [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_SinitAg300_rep{x}' for x in range(n)]))
    # groups.append(('200', [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_SinitAg200_rep{x}' for x in range(n)]))
    # groups.append(('100', [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]))

    # groups.append(('All-Map', [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('300', [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_SinitAg300_rep{x}' for x in range(n)]))
    # groups.append(('200', [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_SinitAg200_rep{x}' for x in range(n)]))
    # groups.append(('100', [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]))

    # groups.append(('All-Map', [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)])) # All-Map for _means
    # # groups.append(('1x0xRes4xx', [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)])) # 4xx used as max Sinit_dist
    # groups.append(('300', [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_SinitRes300_rep{x}' for x in range(n)]))
    # groups.append(('200', [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_SinitRes200_rep{x}' for x in range(n)]))
    # groups.append(('100', [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_SinitRes100_rep{x}' for x in range(n)]))
    # # groups.append(('1x0xRes0', [f'sc_N1_NRW0_ND0_CNN14_FNN16_vis8_ghost_rep{x}' for x in range(20)]))

    # groups.append(('All-Map', [f'sc_N3_NRW0_ND2_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # # groups.append(('2x0xRes4xx', [f'sc_N3_NRW0_ND2_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('300', [f'sc_N3_NRW0_ND2_CNN14_FNN16_vis8_SinitRes300_rep{x}' for x in range(n)]))
    # groups.append(('200', [f'sc_N3_NRW0_ND2_CNN14_FNN16_vis8_SinitRes200_rep{x}' for x in range(n)]))
    # groups.append(('100', [f'sc_N3_NRW0_ND2_CNN14_FNN16_vis8_SinitRes100_rep{x}' for x in range(n)]))

    # groups.append(('All-Map', [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('300', [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_SinitRes300_rep{x}' for x in range(n)]))
    # groups.append(('200', [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_SinitRes200_rep{x}' for x in range(n)]))
    # groups.append(('100', [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_SinitRes100_rep{x}' for x in range(n)]))


    # # groups.append(('0x0', [f'sc_N1_NRW0_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # # groups.append(('0x1', [f'sc_N2_NRW1_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # # # groups.append(('0x2', [f'sc_N3_NRW2_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # # # groups.append(('0x3', [f'sc_N4_NRW3_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # # # groups.append(('0x4', [f'sc_N5_NRW4_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # # groups.append(('0x5', [f'sc_N6_NRW5_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)])) #
    # # groups.append(('0x10', [f'sc_N11_NRW10_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # # groups.append(('0x15', [f'sc_N16_NRW15_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # # groups.append(('0x20', [f'sc_N21_NRW20_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # # # groups.append(('5x5', [f'sc_N11_NRW5_ND5_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('0', [f'sc_N1_NRW0_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('1', [f'sc_N2_NRW1_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # # groups.append(('0x2', [f'sc_N3_NRW2_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # # groups.append(('0x3', [f'sc_N4_NRW3_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # # groups.append(('0x4', [f'sc_N5_NRW4_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('5', [f'sc_N6_NRW5_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)])) #
    # groups.append(('10', [f'sc_N11_NRW10_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('15', [f'sc_N16_NRW15_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('20', [f'sc_N21_NRW20_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # # groups.append(('5x5', [f'sc_N11_NRW5_ND5_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))

    # groups.append(('0x0', [f'sc_N1_NRW0_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('1x0', [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]))
    # groups.append(('2x0', [f'sc_N3_NRW0_ND2_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]))
    # groups.append(('3x0', [f'sc_N4_NRW0_ND3_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]))
    # groups.append(('4x0', [f'sc_N5_NRW0_ND4_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]))
    # groups.append(('5x0', [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)])) #
    # groups.append(('0x1', [f'sc_N2_NRW1_ND0_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]))
    # groups.append(('1x1', [f'sc_N3_NRW1_ND1_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]))
    # groups.append(('2x1', [f'sc_N4_NRW1_ND2_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]))
    # groups.append(('3x1', [f'sc_N5_NRW1_ND3_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]))
    # groups.append(('4x1', [f'sc_N6_NRW1_ND4_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)])) #
    # groups.append(('0x2', [f'sc_N3_NRW2_ND0_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]))
    # groups.append(('1x2', [f'sc_N4_NRW2_ND1_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]))
    # groups.append(('2x2', [f'sc_N5_NRW2_ND2_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]))
    # groups.append(('3x2', [f'sc_N6_NRW2_ND3_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)])) #
    # groups.append(('0x3', [f'sc_N4_NRW3_ND0_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]))
    # groups.append(('1x3', [f'sc_N5_NRW3_ND1_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]))
    # groups.append(('2x3', [f'sc_N6_NRW3_ND2_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)])) #
    # groups.append(('0x4', [f'sc_N5_NRW4_ND0_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]))
    # groups.append(('1x4', [f'sc_N6_NRW4_ND1_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)])) #
    # groups.append(('0x5', [f'sc_N6_NRW5_ND0_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)])) #

    # groups.append(('0x5', [f'sc_N6_NRW5_ND0_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)])) #
    # groups.append(('1x4', [f'sc_N6_NRW4_ND1_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)])) #
    # groups.append(('2x3', [f'sc_N6_NRW3_ND2_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)])) #
    # groups.append(('3x2', [f'sc_N6_NRW2_ND3_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)])) #
    # groups.append(('4x1', [f'sc_N6_NRW1_ND4_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)])) #
    # groups.append(('5x0', [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)])) #

    # groups.append(('0', [f'sc_N1_NRW0_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('1', [f'sc_N2_NRW1_ND0_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]))
    # groups.append(('5', [f'sc_N6_NRW5_ND0_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]))
    # groups.append(('10', [f'sc_N11_NRW10_ND0_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]))
    # groups.append(('15', [f'sc_N16_NRW15_ND0_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]))
    # groups.append(('20', [f'sc_N21_NRW20_ND0_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]))
    
    # # groups.append(('2x0', [f'sc_N3_NRW0_ND2_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('2x1', [f'sc_N4_NRW1_ND2_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('2x2', [f'sc_N5_NRW2_ND2_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('2x3', [f'sc_N6_NRW3_ND2_CNN14_FNN16_vis8_rep{x}' for x in range(n)])) #

    # # groups.append(('2x0', [f'sc_N3_NRW0_ND2_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]))
    # groups.append(('2x1', [f'sc_N4_NRW1_ND2_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]))
    # groups.append(('2x2', [f'sc_N5_NRW2_ND2_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]))
    # groups.append(('2x3', [f'sc_N6_NRW3_ND2_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)])) #

    # groups.append(('coll0', [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('nocollpatch1', [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_nocollpatch_rep{x}' for x in range(n)]))
    # groups.append(('nocoll2', [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]))


    # groups.append(('og',[f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('CNN18',[f'sc_N6_NRW0_ND5_CNN18_FNN16_vis8_rep{x}' for x in range(n)]))
    # # groups.append(('FNN64',[f'sc_N6_NRW0_ND5_CNN14_FNN64_vis8_rep{x}' for x in range(n)]))
    # # groups.append(('FNN16x2',[f'sc_N6_NRW0_ND5_CNN14_FNN16x2_vis8_rep{x}' for x in range(n)]))
    # groups.append(('vis12',[f'sc_N6_NRW0_ND5_CNN14_FNN16_vis12_rep{x}' for x in range(n)]))
    # # groups.append(('CNN18/FNN16x2',[f'sc_N6_NRW0_ND5_CNN18_FNN16x2_vis8_rep{x}' for x in range(n)]))
    # # groups.append(('CNN18/FNN64x2',[f'sc_N6_NRW0_ND5_CNN18_FNN64x2_vis8_rep{x}' for x in range(n)]))


    # groups.append(('0x0', [f'sc_N1_NRW0_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]))
    # groups.append(('1x0', [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]))
    # groups.append(('2x0', [f'sc_N3_NRW0_ND2_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]))
    # groups.append(('3x0', [f'sc_N4_NRW0_ND3_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]))
    # groups.append(('4x0', [f'sc_N5_NRW0_ND4_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]))
    # groups.append(('5x0', [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)])) #
    # groups.append(('0x1', [f'sc_N2_NRW1_ND0_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]))
    # groups.append(('1x1', [f'sc_N3_NRW1_ND1_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]))
    # groups.append(('2x1', [f'sc_N4_NRW1_ND2_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]))
    # groups.append(('3x1', [f'sc_N5_NRW1_ND3_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]))
    # groups.append(('4x1', [f'sc_N6_NRW1_ND4_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)])) #
    # groups.append(('0x2', [f'sc_N3_NRW2_ND0_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]))
    # groups.append(('1x2', [f'sc_N4_NRW2_ND1_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]))
    # groups.append(('2x2', [f'sc_N5_NRW2_ND2_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]))
    # groups.append(('3x2', [f'sc_N6_NRW2_ND3_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)])) #
    # groups.append(('0x3', [f'sc_N4_NRW3_ND0_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]))
    # groups.append(('1x3', [f'sc_N5_NRW3_ND1_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]))
    # groups.append(('2x3', [f'sc_N6_NRW3_ND2_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)])) #
    # groups.append(('0x4', [f'sc_N5_NRW4_ND0_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]))
    # groups.append(('1x4', [f'sc_N6_NRW4_ND1_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)])) #
    # groups.append(('0x5', [f'sc_N6_NRW5_ND0_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)])) #

    # groups.append(('0x5',[f'sc_N6_NRW5_ND0_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]))
    # groups.append(('1x4',[f'sc_N6_NRW4_ND1_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]))
    # groups.append(('2x3',[f'sc_N6_NRW3_ND2_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]))
    # groups.append(('3x2',[f'sc_N6_NRW2_ND3_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]))
    # groups.append(('4x1',[f'sc_N6_NRW1_ND4_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]))
    # groups.append(('5x0',[f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]))




    # groups.append(('5x0',[f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]))
    # groups.append(('0x5',[f'sc_N6_NRW5_ND0_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]))
    # groups.append(('5x0-Ag100',[f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_SinitAg100_collinput_rep{x}' for x in range(n)]))


    groups.append(('0x5', [f'sc_N6_NRW5_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)])) #
    groups.append(('0x5-CA',[f'sc_N6_NRW5_ND0_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]))
    groups.append(('5x0', [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_rep{x+40}' for x in range(n)])) #
    groups.append(('5x0-CA',[f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]))
    groups.append(('5x0-Ag100',[f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]))
    groups.append(('5x0-Ag100-CA',[f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_SinitAg100_collinput_rep{x}' for x in range(n)]))



    tests = [
        'dist_og',
        # 'dist_shift_NSET',
        # 'dist_shift_NSER',
        # 'dist_shift_ETER',
        'dist_JSspatial_NSET',
        # 'dist_learning_time'
        'dist_dirent_NS',
        # 'dist_dirent_ET',
        # 'dist_shift_Nd+2',
        # 'dist_shift_SinitAg100-1',
        # 'dist_norm_NSET',
        ]
    for test in tests:
        # for cmap in ['berlin']:
        for cmap in ['fit_og']:
        # for cmap in ['fit_ET']:
        # for cmap in ['dist_JSspatial_NSET']:
        # for cmap in ['dist_shift_NSET']:
    #         # plot_mult_EA_trends_groups_endonly_means(groups,metric_type=test,cmap=cmap,save_name='groups_sc_Nall_CNN14_FNN16_vis8')
    #         # plot_mult_EA_trends_groups_endonly_means(groups,metric_type=test,cmap=cmap,save_name='groups_sc_ratio_CNN14_FNN16_vis8')
    #         # plot_mult_EA_trends_groups_endonly_means(groups,metric_type=test,cmap=cmap,save_name='groups_sc_ag_CNN14_FNN16_vis8')
    #         # plot_mult_EA_trends_groups_endonly_means(groups,metric_type=test,cmap=cmap,save_name='groups_sc_agNd5_CNN14_FNN16_vis8')
    #         # plot_mult_EA_trends_groups_endonly_means(groups,metric_type=test,cmap=cmap,save_name='groups_sc_res_CNN14_FNN16_vis8')
    #         # plot_mult_EA_trends_groups_endonly_means(groups,metric_type=test,cmap=cmap,save_name='groups_sc_resNd2_CNN14_FNN16_vis8')
    #         # plot_mult_EA_trends_groups_endonly_means(groups,metric_type=test,cmap=cmap,save_name='groups_sc_resNd5_CNN14_FNN16_vis8')
    #         # plot_mult_EA_trends_groups_endonly_means(groups,metric_type=test,cmap=cmap,save_name='groups_sc_Nd0_CNN14_FNN16_vis8')
    #         # plot_mult_EA_trends_groups_endonly_means(groups,metric_type=test,cmap=cmap,save_name='groups_sc_Nall_nocoll_CNN14_FNN16_vis8')
    #         # plot_mult_EA_trends_groups_endonly_means(groups,metric_type=test,cmap=cmap,save_name='groups_sc_ratio_nocoll_CNN14_FNN16_vis8')
    #         # plot_mult_EA_trends_groups_endonly_means(groups,metric_type=test,cmap=cmap,save_name='groups_sc_Nd0_nocoll_CNN14_FNN16_vis8')
    #         # plot_mult_EA_trends_groups_endonly_means(groups,metric_type=test,cmap=cmap,save_name='groups_sc_Nd5_varycoll_CNN14_FNN16_vis8')
    #         # plot_mult_EA_trends_groups_endonly_means(groups,metric_type=test,cmap=cmap,save_name='groups_sc_Nd5_varycog_CNN14_FNN16_vis8')
            # plot_mult_EA_trends_groups_endonly_means(groups,metric_type=test,cmap=cmap,save_name='groups_sc_agratio_CNN14_FNN16_vis8')
            plot_mult_EA_trends_groups_endonly_means(groups,metric_type=test,cmap=cmap,save_name='groups_sc_collinput_CNN14_FNN16_vis8')
            # learning time
    #         # plot_mult_EA_trends_groups_endonly_means(groups,metric_type=test,metric_thresh=500,cmap=cmap,save_name='groups_sc_ratio_CNN14_FNN16_vis8')
    #         # plot_mult_EA_trends_groups_endonly_means(groups,metric_type=test,metric_thresh=500,cmap=cmap,save_name='groups_sc_ratio_nocoll_CNN14_FNN16_vis8')
            # plot_mult_EA_trends_groups_endonly_means(groups,metric_type=test,metric_thresh=500,cmap=cmap,save_name='groups_sc_agratio_CNN14_FNN16_vis8')
    

    tests = [
        # ('dist_shift_OGNS', 'JS_mean_OGNS'),
        ('dist_shift_NSET', 'JS_mean_NSET'),
        # ('dist_shift_NSER', 'JS_mean_NSER'),
        # ('dist_shift_ETER', 'JS_mean_ETER'),
        # ('dist_nosoc', 'JS_mean_NSET'),
    ]
    color_types = [
        # '',
        # 'COM',
        'num_direct',
        # 'num_rand',
        # 'num_total',
        # 'num_direct_split',
        # 'num_direct_COM',
        # 'num_rand_COM',
        # 'heatmap',
        # 'learning_time',
        # 'dirent_OG',
        # 'dirent_NS',
        # 'dirent_ET',
        # 'dirent_ER',
        # 'fit_OG',
        # 'fit_NS',
        # 'fit_ET',
        # 'fit_ER',
        # 'Sinit_dist',
        # 'dist_shift_OGNS',
        # 'dist_shift_NSET',
        # 'dist_shift_NSER',
        # 'dist_shift_ETER',
        # 'dist_JSspatial_ETER',
        # 'dist_shift_Nd+2',
        # 'dist_shift_Nr+2',
        # 'distance',
    ]
    # for test1,test2 in tests:
    #     for color in color_types:
    #         # plot_mult_EA_trends_groups_2D(groups, metric_type1=test1, metric_type2=test2, color_type=color, save_name='groups_sc_Nall_CNN14_FNN16_vis8_2D')
    #         # plot_mult_EA_trends_groups_2D(groups, metric_type1=test1, metric_type2=test2, color_type=color, save_name='groups_sc_ratio_CNN14_FNN16_vis8_2D')
    #         plot_mult_EA_trends_groups_2D(groups, metric_type1=test1, metric_type2=test2, color_type=color, save_name='groups_sc_ag_CNN14_FNN16_vis8_2D')
    #         # plot_mult_EA_trends_groups_2D(groups, metric_type1=test1, metric_type2=test2, color_type=color, save_name='groups_sc_ag_Nd5_CNN14_FNN16_vis8_2D')
    #         # plot_mult_EA_trends_groups_2D(groups, metric_type1=test1, metric_type2=test2, color_type=color, save_name='groups_sc_res_CNN14_FNN16_vis8_2D')
    #         # plot_mult_EA_trends_groups_2D(groups, metric_type1=test1, metric_type2=test2, color_type=color, save_name='groups_sc_resNd2_CNN14_FNN16_vis8_2D')
    #         # plot_mult_EA_trends_groups_2D(groups, metric_type1=test1, metric_type2=test2, color_type=color, save_name='groups_sc_res_Nd5_CNN14_FNN16_vis8_2D')
    #         # plot_mult_EA_trends_groups_2D(groups, metric_type1=test1, metric_type2=test2, color_type=color, save_name='groups_sc_Nd0_CNN14_FNN16_vis8_2D')
    #         # plot_mult_EA_trends_groups_2D(groups, metric_type1=test1, metric_type2=test2, color_type=color, save_name='groups_sc_Nd2_CNN14_FNN16_vis8_2D')
    #         # plot_mult_EA_trends_groups_2D(groups, metric_type1=test1, metric_type2=test2, color_type=color, save_name='groups_sc_Nall_nocoll_CNN14_FNN16_vis8_2D')
    #         # plot_mult_EA_trends_groups_2D(groups, metric_type1=test1, metric_type2=test2, color_type=color, save_name='groups_sc_ratio_nocoll_CNN14_FNN16_vis8_2D')
    #         # plot_mult_EA_trends_groups_2D(groups, metric_type1=test1, metric_type2=test2, color_type=color, save_name='groups_sc_Nd2_nocoll_CNN14_FNN16_vis8_2D')
    #         # plot_mult_EA_trends_groups_2D(groups, metric_type1=test1, metric_type2=test2, color_type=color, save_name='groups_sc_Nd0_nocoll_CNN14_FNN16_vis8_2D')
    #         # plot_mult_EA_trends_groups_2D(groups, metric_type1=test1, metric_type2=test2, color_type=color, save_name='groups_sc_Nd5_varycoll_CNN14_FNN16_vis8_2D')
    #         # plot_mult_EA_trends_groups_2D(groups, metric_type1=test1, metric_type2=test2, color_type=color, save_name='groups_sc_Nd5_varycog_CNN14_FNN16_vis8_2D')

    # for test1,test2 in tests:
    #     for i in range(21):
    #         group = groups[i]
    #         # plot_mult_EA_trends_groups_2D([group], metric_type1=test1, metric_type2=test2, color_type='', save_name=f'groups_sc_Nall_CNN14_FNN16_vis8_2D_{group[0]}')
    #         # plot_mult_EA_trends_groups_2D([group], metric_type1=test1, metric_type2=test2, color_type='fit_OG', save_name=f'groups_sc_Nall_CNN14_FNN16_vis8_2D_{group[0]}')
    #         # plot_mult_EA_trends_groups_2D([group], metric_type1=test1, metric_type2=test2, color_type='heatmap', save_name=f'groups_sc_Nall_CNN14_FNN16_vis8_2D_{group[0]}')
    #         plot_mult_EA_trends_groups_2D([group], metric_type1=test1, metric_type2=test2, color_type='fit_OG', cbar=False, save_name=f'groups_sc_ag_CNN14_FNN16_vis8_2D_{group[0]}')
    #         # plot_mult_EA_trends_groups_2D([group], metric_type1=test1, metric_type2=test2, color_type='fit_OG', cbar=False, save_name=f'groups_sc_res_Nd5_CNN14_FNN16_vis8_2D_{group[0]}')
    #         # plot_mult_EA_trends_groups_2D([group], metric_type1=test1, metric_type2=test2, color_type='fit_OG', cbar=False, save_name=f'groups_sc_Nd0_CNN14_FNN16_vis8_2D_{group[0]}')
    #         # plot_mult_EA_trends_groups_2D([group], metric_type1=test1, metric_type2=test2, color_type='fit_OG', cbar=False, save_name=f'groups_sc_Nall_nocoll_CNN14_FNN16_vis8_2D_{group[0]}')
    #         # plot_mult_EA_trends_groups_2D([group], metric_type1=test1, metric_type2=test2, color_type='fit_OG', cbar=False, save_name=f'groups_sc_Nd0_nocoll_CNN14_FNN16_vis8_2D_{group[0]}')
    #         # plot_mult_EA_trends_groups_2D([group], metric_type1=test1, metric_type2=test2, color_type='fit_OG', cbar=False, save_name=f'groups_sc_Nd5_varycoll_CNN14_FNN16_vis8_2D_{group[0]}')
    #         # plot_mult_EA_trends_groups_2D([group], metric_type1=test1, metric_type2=test2, color_type='fit_OG', cbar=False, save_name=f'groups_sc_agratio_Nd5_CNN14_FNN16_vis8_2D_{group[0]}')

    # n = 0
    # plot_mult_EA_trends_multievo(names=[f'sc_N2_multi_CNN14_FNN16_vis8_rep{x}' for x in range(n,n+1)], val='cen', save_name=f'sc_N2_multi_ag{n}')

    # for n in range(20):
    #     plot_mult_EA_trends_multievo(names=[f'sc_N2_multi_CNN14_FNN16_vis8_rep{x}' for x in range(n,n+1)], val='cen', save_name=f'sc_N2_multi_ag{n}')
    # for n in range(20):
    #     plot_mult_EA_trends_multievo(names=[f'sc_N2_multi_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n,n+1)], val='cen', save_name=f'sc_N2_SinitAg100_multi_ag{n}')
    # for n in range(20):
    #     plot_mult_EA_trends_multievo(names=[f'sc_N6_multi_CNN14_FNN16_vis8_rep{x}' for x in range(n,n+1)], val=None, save_name=f'sc_N6_multi_ag{n}')

    # n = 20
    # plot_mult_EA_trends_multievo(names=[f'sc_N2_multi_CNN14_FNN16_vis8_rep{x}' for x in range(n)], val='cen', save_name=f'sc_N2_multi')
    # plot_mult_EA_trends_multievo(names=[f'sc_N2_multi_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)], val='cen', save_name=f'sc_N2_SinitAg100_multi')
    # plot_mult_EA_trends_multievo(names=[f'sc_N6_multi_CNN14_FNN16_vis8_rep{x}' for x in range(n)], val='cen', save_name=f'sc_N6_multi')


    # plot_mult_EA_trends_multievo(names=[f'sc_N6_multi_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n,n+1)], val=None, save_name='sc_N6_SinitAg100_multi')

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
    # names = []
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep{x}')
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_seed10k_rep{x}')
    # groups.append(('vis 6', names))
    # names = []
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}')
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep{x}')
    # groups.append(('vis 8', names))
    # names = []
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep{x}')
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_seed10k_rep{x}')
    # groups.append(('vis 10', names))
    # names = []
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep{x}')
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed10k_rep{x}')
    # groups.append(('vis 12', names))
    # names = []
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep{x}')
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_seed10k_rep{x}')
    # groups.append(('vis 14', names))
    # names = []
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep{x}')
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_seed10k_rep{x}')
    # groups.append(('vis 16', names))
    # names = []
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_rep{x}')
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis18_PGPE_ss20_mom8_seed10k_rep{x}')
    # groups.append(('vis 18', names))
    # names = []
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_rep{x}')
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_seed10k_rep{x}')
    # groups.append(('vis 20', names))
    # names = []
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep{x}')
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed10k_rep{x}')
    # groups.append(('vis 24', names))
    # names = []
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep{x}')
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_seed10k_rep{x}')
    # groups.append(('vis 32', names))
    # # # groups.append(('vis 64', [f'sc_CNN14_FNN2_p50e20_vis64_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # # # groups.append(('vis 128', [f'sc_CNN14_FNN2_p50e20_vis128_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # # plot_mult_EA_trends_groups(groups, val='cen', group_est='median', save_name='groups_singlecorner_vis')
    # # plot_mult_EA_trends_groups_endonly(groups, val='cen', save_name='groups_endonly_singlecorner_vis')
    # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,save_name='groups_sc_N1_vis_CNN14_FNN16_vis8_endonly_numfound_bees')
    # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,save_name='groups_sc_N1_vis_CNN14_FNN16_vis8_endonly_mean_bees')


    # ### FNN SIZE ###
    # groups = []
    # groups.append(('FNN 1', [f'sc_CNN14_FNN1_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('FNN 2', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # # groups.append(('FNN 3', [f'sc_CNN14_FNN3_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('FNN 4', [f'sc_CNN14_FNN4_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # # groups.append(('FNN 8', [f'sc_CNN14_FNN8_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('FNN 16', [f'sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('FNN 2x2', [f'sc_CNN14_FNN2x2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # # groups.append(('FNN 2x3', [f'sc_CNN14_FNN2x3_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('FNN 2x4', [f'sc_CNN14_FNN2x4_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # # groups.append(('FNN 2x8', [f'sc_CNN14_FNN2x8_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('FNN 2x16', [f'sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
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
    # # # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_singlecorner_FOV')
    # # plot_mult_EA_trends_groups_endonly(groups, val='cen', save_name='groups_endonly_singlecorner_FOV')
    # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,save_name='groups_sc_N1_fov_CNN14_FNN16_vis8_endonly_numfound_bees')
    # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,save_name='groups_sc_N1_fov_CNN14_FNN16_vis8_endonly_mean_bees')

    # # +ext
    # groups.append(('fov94, vis16', [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_fov94_rep{x}' for x in range(20)]))
    # groups.append(('fov97, vis32', [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_fov97_rep{x}' for x in range(20)]))
    # groups.append(('fov94, vis16, FNN16', [f'sc_CNN14_FNN16_p50e20_vis16_PGPE_ss20_mom8_fov94_rep{x}' for x in range(20)]))
    # groups.append(('fov97, vis32, FNN16', [f'sc_CNN14_FNN16_p50e20_vis32_PGPE_ss20_mom8_fov97_rep{x}' for x in range(20)]))
    # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_singlecorner_FOV_ext')
    # plot_mult_EA_trends_groups_endonly(groups, val='cen', save_name='groups_endonly_singlecorner_FOV_ext')


    ### DIST ###
    # groups = []
    # names = []
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep{x}')
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_seed10k_rep{x}')
    # groups.append((r'$\sigma = 1$', names))
    # names = []
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep{x}')
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed10k_rep{x}')
    # groups.append((r'$\sigma = 0.5$', names))
    # names = []
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep{x}')
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_seed10k_rep{x}')
    # groups.append((r'$\sigma = 0.4$', names))
    # names = []
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep{x}')
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_seed10k_rep{x}')
    # groups.append((r'$\sigma = 0.3$', names))
    # names = []
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep{x}')
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed10k_rep{x}')
    # groups.append((r'$\sigma = 0.2$', names))
    # names = []
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep{x}')
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_seed10k_rep{x}')
    # groups.append((r'$\sigma = 0.1$', names))
    # names = []
    # for x in range(20):
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}')
    #     names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep{x}')
    # groups.append((r'$\sigma = 0$', names))
    # plot_mult_EA_trends_groups(groups, val='cen', group_est='median', save_name='groups_singlecorner_WF_scaling_s10')
    # # plot_mult_EA_trends_groups_endonly(groups, val='cen', save_name='groups_endonly_singlecorner_WF_scaling_s10')
    # plot_mult_EA_trends_groups_endonly_perfect(groups, val='cen', save_name='groups_endonly_singlecorner_WF_scaling_s10')
    # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,save_name='groups_sc_N1_dist_CNN14_FNN16_vis8_endonly_numfound_bees')
    # plot_mult_EA_trends_groups_endonly_sums(groups,bees=True,save_name='groups_sc_N1_dist_CNN14_FNN16_vis8_endonly_mean_bees')

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
    # groups = []
    # groups.append(('BS 0', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('BS 500', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_bound500_rep{x}' for x in range(20)]))
    # groups.append(('BS 1000', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)]))
    # groups.append(('maxWF, BS 0', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep{x}' for x in range(20)]))
    # groups.append(('maxWF, BS 500', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_bound500_rep{x}' for x in range(20)]))
    # groups.append(('maxWF, BS 1000', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_bound1000_rep{x}' for x in range(20)]))
    # groups.append(('mlWF, BS 0', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep{x}' for x in range(20)]))
    # groups.append(('mlWF, BS 500', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_bound500_rep{x}' for x in range(20)]))
    # groups.append(('mlWF, BS 1000', [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_bound1000_rep{x}' for x in range(20)]))
    # groups.append(('VIS 32', [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep{x}' for x in range(20)]))
    # groups.append(('VIS 32, BS 500', [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_bound500_rep{x}' for x in range(20)]))
    # groups.append(('VIS 32, BS 1000', [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)]))
    # plot_mult_EA_trends_groups(groups, val='cen', save_name='groups_singlecorner_boundary_scale')
    # plot_mult_EA_trends_groups_endonly(groups, val='cen', save_name='groups_endonly_singlecorner_boundary_scale')



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


### ----------nonNNs----------- ###

    # names = [
    #     # 'rotdiff_0p0001_randbouncy',
    #     'rotdiff_0p0005_randbouncy',
    #     'rotdiff_0p001_randbouncy',
    #     'rotdiff_0p005_randbouncy',
    #     'rotdiff_0p01_randbouncy',
    #     'rotdiff_0p05_randbouncy',
    #     'rotdiff_0p10_randbouncy',
    #     'rotdiff_0p50_randbouncy',
    # ]
    # # plot_mult_EA_trends_randomwalk(names,save_name='_randbouncy')
    # # plot_mult_EA_trends_randomwalk(names,clean_outside=True,save_name='_randbouncy_clean')
    # plot_mult_EA_trends_randomwalk(names,bees=True,save_name='_randbouncy_bees')
    # # plot_mult_EA_trends_randomwalk(names,bees=True,norm_outside=True,save_name='_randbouncy_beesnorm')
    # # # plot_mult_EA_trends_randomwalk(names,clean_outside=True,bees=True,save_name='_randbouncy_beesclean')
    # names = [
    #     'rotdiff_0p01_randbouncy',
    #     # 'rotdiff_0p01_randbouncy_N2',
    #     'rotdiff_0p01_randbouncy_N5',
    #     'rotdiff_0p01_randbouncy_N10',
    #     'rotdiff_0p01_randbouncy_N15',
    #     'rotdiff_0p01_randbouncy_N20',
    # ]
    # # plot_mult_EA_trends_randomwalk(names,social=True,save_name='_randbouncy_social')
    # # plot_mult_EA_trends_randomwalk(names,social=True,clean_outside=True,save_name='_randbouncy_social_clean')
    # plot_mult_EA_trends_randomwalk(names,social=True,bees=True,save_name='_randbouncy_social_bees')
    # # plot_mult_EA_trends_randomwalk(names,social=True,bees=True,norm_outside=True,save_name='_randbouncy_social_beesnorm')
    # # plot_mult_EA_trends_randomwalk(names,social=True,clean_outside=True,bees=True,save_name='_randbouncy_social_beesclean')
