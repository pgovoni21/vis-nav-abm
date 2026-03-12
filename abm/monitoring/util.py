
import numpy as np
from pathlib import Path
import pickle


from collections import deque
from itertools import islice
from matplotlib import collections as mc
from matplotlib.colors import LinearSegmentedColormap as lsc
import colorcet as cc

# -------------------------- retrieval -------------------------- #

def find_top_val_gen(exp_name, rank='cen', archive=False):

    if archive:
        data_dir = Path(__file__).parent.parent / r'data/simulation_data/archive - ISBDDP/'
    else:
        data_dir = Path(__file__).parent.parent / r'data/simulation_data/'

    # parse val results text file
    if rank == 'top': 
        with open(fr'{data_dir}/{exp_name}/val_results.txt') as f:
            lines = f.readlines()

            val_data = np.zeros((len(lines)-1, 3))
            for i, line in enumerate(lines[1:]):
                data = [item.strip() for item in line.split(' ')]
                val_data[i,0] = data[1] # generation
                val_data[i,1] = data[4] # train fitness
                val_data[i,2] = data[7] # val fitness

            # sort according to val fitness
            top_ind = np.argsort(val_data[:,2])[0] 
            top_gen = int(val_data[top_ind,0])
            top_valfit = int(val_data[top_ind,2])
            # print(f'gen {top_gen}: fit {top_valfit}')

    elif rank == 'cen': 
        with open(fr'{data_dir}/{exp_name}/val_results_cen.txt') as f:
            lines = f.readlines()
        # with open(fr'{data_dir}/{exp_name}/val_matrix_cen.bin') as f:
        #     val_data = pickle.load(f)

            val_data = np.zeros((len(lines)-1, 3))
            for i, line in enumerate(lines[1:]):
                data = [item.strip() for item in line.split(' ')]
                val_data[i,0] = data[1] # generation
                val_data[i,1] = data[4] # train fitness
                val_data[i,2] = data[7] # val fitness

            # sort according to val fitness
            top_ind = np.argsort(val_data[:,2])[0] 
            top_gen = int(val_data[top_ind,0])
            top_valfit = int(val_data[top_ind,2])
            # print(f'gen {top_gen}: fit {top_valfit}')

    return f'gen{top_gen}', top_valfit

# ------------------------------- calcs ---------------------------------------- #

def calc_entropy(h):
    h_norm = h / np.sum(h)
    e = np.sum( h_norm*np.log(1/h_norm) )
    return e

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

# ------------------------------- tools ---------------------------------------- #

def sliding_window(iterable, n):
  """
  sliding_window('ABCDEFG', 4) -> ABCD BCDE CDEF DEFG
  https://docs.python.org/3/library/itertools.html
  """
  it = iter(iterable)
  window = deque(islice(it, n-1), maxlen=n)
  for x in it:
      window.append(x)
      yield tuple(window)

def sliding_window_ori(iterable, n):
  """
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
    
    # cm = plt.get_cmap('plasma')
    # cm = cmr.chroma
    cm = lsc.from_list('bgy', cc.bgy)
    cm = cm.reversed()

    cm_disc = cm(np.linspace(0, 1, n-1, endpoint=False))
    cm_disc = cm(np.linspace(1, 0, n-1, endpoint=False)) # flipped for some cmaps
    cm_disc[:,-1] = np.linspace(0.2, 1, n-1, endpoint=False) # start with lower alpha

    return mc.LineCollection(sliding_window(zip(x, y), 2),
                            colors=cm_disc,
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

# ------------------------------- social metrics ---------------------------------------- #

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
    with open(fr'{data_dir}/{name}/val_matrix_best_ghostexplorer_perturb.bin','rb') as f:
        data_explorer = pickle.load(f)

    if 'og' in metric_type:
        metric = np.mean(data_og)
    elif 'nosoc' in metric_type:
        metric = np.mean(data_nosoc)
    elif 'exploiter' in metric_type:
        metric = np.mean(data_exploiter)
    elif 'explorer' in metric_type:
        metric = np.mean(data_explorer)
    elif 'Nd2' in metric_type:
        with open(fr'{data_dir}/{name}/val_matrix_best_Nd+2_perturb.bin','rb') as f:
            data_Nd2 = pickle.load(f)
        metric = np.mean(data_Nd2)

    elif 'shift_OGNS' in metric_type:
        metric = np.mean(data_nosoc) - np.mean(data_og)
    elif 'shift_NSETs' in metric_type: # check if multiple first
        with open(fr'{data_dir}/{name}/val_matrix_best_ghostexploiters_perturb.bin','rb') as f:
            data_exploiters = pickle.load(f)
        metric = np.mean(data_nosoc) - np.mean(data_exploiters)
    elif 'shift_NSET' in metric_type:
        metric = np.mean(data_nosoc) - np.mean(data_exploiter)
    elif 'shift_NSER' in metric_type:
        metric = np.mean(data_nosoc) - np.mean(data_explorer)
    elif 'shift_ETERs' in metric_type: # check if multiple first
        with open(fr'{data_dir}/{name}/val_matrix_best_ghostexploiters_perturb.bin','rb') as f:
            data_exploiters = pickle.load(f)
        with open(fr'{data_dir}/{name}/val_matrix_best_ghostexplorers_perturb.bin','rb') as f:
            data_explorers = pickle.load(f)
        metric = np.mean(data_explorers) - np.mean(data_exploiters)
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

    elif 'JSspatial' in metric_type or 'dirent' in metric_type:
        # spatial_metric_list = [
        #         'de_mean_OG', 'de_mean_NS', 'de_mean_ET', 'de_mean_ER',
        #         'JS_mean_OGNS', 'JS_mean_NSET', 'JS_mean_NSER', 'JS_mean_ETER'
        #         ]
        if 'OGNS' in metric_type:
            index = 4
        elif 'NSET' in metric_type:
            index = 5
            # index = 10 # for social_extra + patch_only + JS
        elif 'NSER' in metric_type:
            index = 6
        elif 'ETER' in metric_type:
            index = 7
        elif 'OG' in metric_type:
            index = 0
        elif 'NS' in metric_type:
            index = 1
            # index = 8 # for social_extra + patch_only + dirent_NS
        elif 'ET' in metric_type:
            index = 2
        elif 'ER' in metric_type:
            index = 3

        with open(fr'{data_dir}/traj_matrices/gamut_social.bin', 'rb') as f:
            data_dict = pickle.load(f)
        # with open(fr'{data_dir}/traj_matrices/gamut_social_extra.bin', 'rb') as f:
        #     data_dict = pickle.load(f)
        # # with open(fr'{data_dir}/traj_matrices/gamut_social_mults.bin', 'rb') as f:
        # #     data_dict = pickle.load(f)
        # # if 'NSETs' in metric_type:
        # #     index = 8
        # # elif 'ETERs' in metric_type:
        # #     index = 9
        # # elif 'Nd+2' in metric_type:
        # #     index = 10

        metric = data_dict[name][index]

    elif 'learning_time' in metric_type:
        with open(fr'{data_dir}/{name}/fitness_spread_per_generation.bin','rb') as f:
            data = pickle.load(f)

        data_genxpop = np.mean(data, axis=2)
        top_data = np.min(data_genxpop, axis=1)

        thresh = 500
        if np.min(top_data) <= thresh:
            metric = int(np.argwhere(top_data <= thresh)[0][0])
        else:
            metric = 1000

    elif 'distance-scaled' in metric_type:
        with open(fr'{data_dir}/{name}/val_matrix_best_nosocial-dist-scaled_perturb.bin','rb') as f:
            data = pickle.load(f)
        metric = np.mean(data)

    elif 'distance' in metric_type:
        with open(fr'{data_dir}/{name}/val_matrix_best_nosocial-dist_perturb.bin','rb') as f:
            data = pickle.load(f)
        # data = data[data <= np.quantile(data,.9)] # use only best 90%
        metric = np.mean(data)
        # metric = np.std(data)
        # metric = data # by_run

    elif 'time' in metric_type: # by_run
        metric = data_og
        # metric = data_nosoc
        # metric = data_exploiter
        # metric = data_explorer

    else:
        print(f'{metric_type} not valid metric type - type1')
    
    return metric


def names_to_metric(names, metric_type):

    row = []
    for name in names:
        metric = name_to_metric(name, metric_type)
        row.append(metric)

    row = np.asarray(row)
    if 'fit' in metric_type:
        return np.median(row)
    elif 'meanshift' in metric_type: # perf_diff in tables
        return np.median(row)
    elif 'dist' in metric_type: # endonly_means
        return row
    elif 'JSspatial' in metric_type:
        return np.median(row)
    elif 'learning_time' in metric_type:
        return np.median(row)
    elif 'dirent' in metric_type:
        return np.median(row)
    else:
        print(f'{metric_type} not valid metric type -type2')
