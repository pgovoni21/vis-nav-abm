import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pickle
import scipy


def build_data_lib(names, max_score):

    root_dir = Path(__file__).parent.parent
    data_dir = Path(root_dir, r'data/simulation_data')

    num_runs = len(names)
    # print(f'Calculating CI of {num_runs} runs')

    top_indiv_final_scores = np.zeros(num_runs)

    for r_num, name in enumerate(names):

        with open(fr'{data_dir}/{name}/fitness_spread_per_generation.bin','rb') as f:
            trend_data = pickle.load(f)
        trend_data = np.array(trend_data)

        # top_trend_data = np.min(trend_data, axis=1) # min : top
        top_trend_data = np.max(trend_data, axis=1) # max : top

        top_indiv_final_scores[r_num] = top_trend_data[-1] / max_score
    
    # fig, ax = plt.subplots()
    # ax.hist(top_indiv_final_scores, bins=20)
    # ax.set_xlabel('scores')
    # ax.set_ylabel('freq')
    # plt.show()

    return top_indiv_final_scores


def compile_bs_stats(data, statistic, true_value, num_sets=10000, n_resamples=10000, method='percentile'):

    data = (data,)
    true_coverage = np.zeros(num_sets)
    CI_width = np.empty(num_sets)

    rng = np.random.default_rng()
    for i in range(num_sets):

        bs = scipy.stats.bootstrap(data, 
                                statistic, 
                                n_resamples=n_resamples,
                                confidence_level=.95,
                                method=method,
                                random_state=rng
                                )
        min, max = bs.confidence_interval

        if min <= true_value <= max:
            true_coverage[i] = 1
        CI_width[i] = max - min
    
    return true_coverage, CI_width


def iqm(data):
    return scipy.stats.trim_mean(data, proportiontocut=0.25, axis=None)


def plot_bs_data(data, save_name=None):

    # compute true agg estimates
    # true_mean = np.mean(data)
    true_median = np.median(data)
    true_iqm = scipy.stats.trim_mean(data, proportiontocut=0.25, axis=None)
    # print(f'True agg estimates: {true_mean}, {true_median}, {true_iqm}')

    # prepare input ranges + output matrices
    test_range = [3,5,10,20,50]
    method_range = ['percentile','basic','bca']

    num_tests = len(test_range)
    num_methods = len(method_range)

    tc_array_mean = np.empty((num_tests, num_methods))
    CIw_array_mean= np.empty((num_tests, num_methods))
    tc_array_iqm = np.empty((num_tests, num_methods))
    CIw_array_iqm= np.empty((num_tests, num_methods))

    # compile bootstrap statistics
    for m, method in enumerate(method_range):
        for t, test_samp_size in enumerate(test_range):

            subsample = np.random.choice(data, size=test_samp_size, replace=True)

            tc, CIw = compile_bs_stats(subsample, 
                                    np.median, 
                                    true_median, 
                                    method=method)
            tc_array_mean[t,m] = np.mean(tc)
            CIw_array_mean[t,m] = np.mean(CIw)
            
            tc, CIw = compile_bs_stats(subsample, 
                                       iqm, 
                                       true_iqm,
                                       method=method)
            tc_array_iqm[t,m] = np.mean(tc)
            CIw_array_iqm[t,m] = np.mean(CIw)
            

    fig, axs = plt.subplots(2,2, figsize=(15,10))
    cmap = plt.get_cmap('hsv')
    cmap_range = len(method_range)
    lns = []
    for m, method in enumerate(method_range):

        l = axs[0,0].plot(test_range, tc_array_mean[:,m], '-o', color=cmap(m/cmap_range), label=method)
        axs[1,0].plot(test_range, CIw_array_mean[:,m], '-o', color=cmap(m/cmap_range))

        axs[0,1].plot(test_range, tc_array_iqm[:,m], '-o', color=cmap(m/cmap_range))
        axs[1,1].plot(test_range, CIw_array_iqm[:,m], '-o', color=cmap(m/cmap_range))

        lns.append(l[0])

    axs[1,0].set_ylim([-.05,.65])
    axs[1,1].set_ylim([-.05,.65])

    fig.supxlabel('# resampled runs')   
    axs[0,0].set_ylabel('true coverage %')
    axs[1,0].set_ylabel('average CI width')

    labs = [l.get_label() for l in lns]
    fig.legend(lns, labs, loc='upper right')

    if save_name: 
        root_dir = Path(__file__).parent.parent
        data_dir = Path(root_dir, r'data/simulation_data')
        plt.savefig(fr'{data_dir}/{save_name}.png')
    else:
        plt.show()


if __name__ == '__main__':

    names = [f'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}' for x in range(100)]
    data = build_data_lib(names, max_score=10)
    plot_bs_data(data, save_name=f'CIstats_doublepoint_CNN1128_GRU2_incl')

    names = [f'symdoublepoint_CNN1128_FNN2_rep{x}' for x in range(78)]
    data = build_data_lib(names, max_score=10)
    plot_bs_data(data, save_name=f'CIstats_symdoublepoint_CNN1128_FNN2_incl')

    names = [f'symdoublepoint_CNN1128_GRU2_rep{x}' for x in range(100)]
    data = build_data_lib(names, max_score=10)
    plot_bs_data(data, save_name=f'CIstats_symdoublepoint_CNN1128_GRU2_incl')