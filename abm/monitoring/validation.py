from abm.start_sim import start
from abm.start_sim_multi import start as start_multi
from abm.monitoring.trajs import find_top_val_gen

from pathlib import Path
import pickle
import numpy as np
import multiprocessing as mp
import os
import dotenv as de


def rerun_NNs(name, num_NNs=20, num_seeds=100, noise_type=None, perturb_type=None, time=False):

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    exp_path = fr'{data_dir}/{name}'
    env_path = fr'{exp_path}/.env'
    envconf = de.dotenv_values(env_path)

    if noise_type is None and perturb_type is None:
        print(f'running: {name}')
        if Path(fr'{exp_path}/val_results_cen.txt').is_file():
            return print(f'val_results_cen already exists')
    elif noise_type is not None:
        print(f'running: {name} + {noise_type} noise')
        if Path(fr'{exp_path}/val_matrix_cen_{noise_type}_noise.bin').is_file():
            return print(f'val_matrix_cen already exists')
    elif perturb_type is not None:
        print(f'running: {name} + {perturb_type} perturb')
        if Path(fr'{exp_path}/val_matrix_cen_{perturb_type}_perturb.bin').is_file():
            return print(f'val_matrix_cen already exists')

    with open(fr'{data_dir}/{name}/fitness_spread_per_generation.bin','rb') as f:
        data = pickle.load(f)
    data_genxpop = np.mean(data, axis=2) # average across episodes
    avg_data = np.mean(data_genxpop, axis=1) # average across instances in population
    # avg_data = avg_data[:num_gens]

    if envconf['SIM_TYPE'].startswith('walls'):
        top_ind = np.argsort(avg_data)[:num_NNs] # min : top
    elif envconf['SIM_TYPE'].startswith('nowalls'):
        top_ind = np.argsort(avg_data)[-1:-num_NNs-1:-1] # max : top
    else:
        raise ValueError('SIM_TYPE not recognized')
    top_fit = [avg_data[i] for i in top_ind]

    val_matrix = np.zeros((num_NNs,
                           num_seeds))
    val_matrix_time = np.zeros((num_NNs,
                           num_seeds))
    print(f'Validation matrix shape (num_NNs, num_seeds): {val_matrix.shape}')

    # pack inputs for multiprocessing map
    mp_inputs = []
    for g in top_ind:

        # with open(fr'{data_dir}/{name}/gen{g}_NN0_pickle.bin','rb') as f: # --> NN0 / top
        with open(fr'{data_dir}/{name}/gen{g}_NNcen_pickle.bin','rb') as f: # --> NNcen / dist center
            pv = pickle.load(f)

        for s in range(num_seeds):
            mp_inputs.append( (None, pv, None, s, env_path) ) # model_tuple=None, load_dir=None

    # run agent NNs in parallel
    with mp.Pool() as pool:
        results = pool.starmap_async(start, mp_inputs)
        pool.close()
        pool.join()
    results_list = results.get()

    # skip to start of each seed series/chunk + allocate fitness to save matrix
    if envconf['SIM_TYPE'].startswith('walls'):
        for i,c in enumerate(range(0, len(results_list), num_seeds)):
            for s,(time_taken, dist_from_patch, data) in enumerate(results_list[c : c + num_seeds]):
                val_matrix[i,s] = int(time_taken)
    elif envconf['SIM_TYPE'].startswith('nowalls'):
        for i,c in enumerate(range(0, len(results_list), num_seeds)):
            for s,(first_time_consume, total_res_collected) in enumerate(results_list[c : c + num_seeds]):
                val_matrix[i,s] = int(total_res_collected)
                val_matrix_time[i,s] = int(first_time_consume)

    # saving protocol for noise/perturb/regular

    if noise_type is not None:

        # dump raw data + calc avg val perf
        with open(fr'{exp_path}/val_matrix_cen_{noise_type}_noise.bin', 'wb') as f:
            pickle.dump(val_matrix, f)
        avg_per_NN_new = np.average(val_matrix, axis=1).round(1)

        # previous val perf
        with open(fr'{exp_path}/val_matrix_cen.bin','rb') as f:
            val_matrix = pickle.load(f)
        avg_per_NN_prev = np.average(val_matrix, axis=1).round(1)

        # print both
        for i, ef, vfn, vfp in zip(top_ind, top_fit, avg_per_NN_new, avg_per_NN_prev):
            print(f'gen: {i} | EA_fit: {int(ef)} | val_fit: {int(vfp)} | val_fit + noise: {int(vfn)}')

    elif perturb_type is not None:

        with open(fr'{exp_path}/val_matrix_cen_{perturb_type}_perturb.bin', 'wb') as f:
            pickle.dump(val_matrix, f)
        avg_per_NN_new = np.average(val_matrix, axis=1).round(1)

        with open(fr'{exp_path}/val_matrix_cen.bin','rb') as f:
            val_matrix = pickle.load(f)
        avg_per_NN_prev = np.average(val_matrix, axis=1).round(1)

        if time:
            with open(fr'{exp_path}/val_matrix_cen_{perturb_type}_time_perturb.bin', 'wb') as f:
                pickle.dump(val_matrix_time, f)
            time_avg_per_NN_new = np.average(val_matrix_time, axis=1).round(1)

            with open(fr'{exp_path}/val_matrix_cen_time.bin','rb') as f:
                val_matrix = pickle.load(f)
            time_avg_per_NN_prev = np.average(val_matrix, axis=1).round(1)

            for i, ef, vfp, tp, vfn, tn in zip(top_ind, top_fit, avg_per_NN_prev, time_avg_per_NN_prev, avg_per_NN_new, time_avg_per_NN_new):
                print(f'gen: {i} | EA_fit: {int(ef)} | val_fit: {int(vfp)} @ {int(tp)} | val_fit + {perturb_type}: {int(vfn)} @ {int(tn)}')

        else:
            for i, ef, vfp, vfn in zip(top_ind, top_fit, avg_per_NN_prev, avg_per_NN_new):
                print(f'gen: {i} | EA_fit: {int(ef)} | val_fit: {int(vfp)} | val_fit + {perturb_type}: {int(vfn)}')

    else:

        with open(fr'{exp_path}/val_matrix_cen.bin', 'wb') as f:
            pickle.dump(val_matrix, f)
        avg_per_NN = np.average(val_matrix, axis=1).round(1)

        if time:
            with open(fr'{exp_path}/val_matrix_cen_time.bin', 'wb') as f:
                pickle.dump(val_matrix_time, f)
            time_avg_per_NN = np.average(val_matrix_time, axis=1).round(1)

        with open(fr'{exp_path}/val_results_cen.txt', 'w') as f:
            f.write(f'Validation matrix shape (num_NNs, num_seeds): {val_matrix.shape}\n')
            for i, ef, vf in zip(top_ind, top_fit, avg_per_NN):
                f.write(str(f'gen: {i} | EA_fit: {int(ef)} | val_fit: {int(vf)}\n'))

        if time:
            for i, ef, vf, t in zip(top_ind, top_fit, avg_per_NN, time_avg_per_NN):
                print(f'gen: {i} | EA_fit: {int(ef)} | val_fit: {int(vf)} @ {int(t)}')
        else:
            for i, ef, vf in zip(top_ind, top_fit, avg_per_NN):
                print(f'gen: {i} | EA_fit: {int(ef)} | val_fit: {int(vf)}')



def rerun_best_val_NN(name, num_seeds=1000, perturb_type=None):

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    exp_path = fr'{data_dir}/{name}'
    env_path = fr'{exp_path}/.env'

    if perturb_type is not None:
        print(f'running: {name} + {perturb_type} perturb')
        if Path(fr'{exp_path}/val_matrix_best_{perturb_type}_perturb.bin').is_file():
            return print(f'val_matrix_best_{perturb_type}_perturb already exists')
    else:
        print(f'running: {name}')
        if Path(fr'{exp_path}/val_matrix_best.bin').is_file():
            return print(f'val_results_best already exists')

    with open(fr'{data_dir}/{name}/fitness_spread_per_generation.bin','rb') as f:
        data = pickle.load(f)
    data_genxpop = np.mean(data, axis=2) # average across episodes
    avg_data = np.mean(data_genxpop, axis=1)

    avg_data = avg_data[:1000]
    # print(avg_data.shape)

    with open(fr'{exp_path}/val_matrix_cen.bin','rb') as f: # 20 top NNs
        val_matrix_prev = pickle.load(f)
    avg_per_NN_prev = np.average(val_matrix_prev, axis=1)
    best_NN_ind = np.argmin(avg_per_NN_prev)

    valmatrix_inds = np.argsort(avg_data)[:20] # min : top
    best_NN_gen = valmatrix_inds[best_NN_ind]

    with open(fr'{data_dir}/{name}/gen{best_NN_gen}_NNcen_pickle.bin','rb') as f:
        pv = pickle.load(f)

    # pack inputs for multiprocessing map
    mp_inputs = []
    for s in range(num_seeds):
        mp_inputs.append( (None, pv, None, s, env_path) ) # model_tuple=None, load_dir=None

    # run agent NNs in parallel
    with mp.Pool() as pool:
        results = pool.starmap_async(start, mp_inputs)
        pool.close()
        pool.join()
    results_list = results.get()

    # allocate fitnesses
    val_matrix = np.zeros((num_seeds))
    for s,(time_taken, dist_from_patch, data) in enumerate(results_list):
        val_matrix[s] = int(time_taken)
        # val_matrix[s] = int(dist_from_patch)

    # saving protocol for perturb/regular
    if perturb_type == 'ghostexplorer':

        with open(fr'{exp_path}/val_matrix_best_{perturb_type}_perturb.bin', 'wb') as f:
            pickle.dump(val_matrix, f)
        avg_per_NN_new = np.average(val_matrix)

        with open(fr'{exp_path}/val_matrix_best.bin', 'rb') as f:
            val_matrix = pickle.load(f)
        avg_per_NN_prevbest = np.average(val_matrix)

        with open(fr'{exp_path}/val_matrix_best_ghostexploiter_perturb.bin', 'rb') as f:
            val_matrix = pickle.load(f)
        avg_per_NN_exploiter = np.average(val_matrix)

        print(f'gen: {best_NN_gen} | val_fit_prevbest: {int(avg_per_NN_prevbest)} | val_fit_exploiter: {int(avg_per_NN_exploiter)} | val_fit_new: {int(avg_per_NN_new)}')

    elif perturb_type is not None:

        with open(fr'{exp_path}/val_matrix_best_{perturb_type}_perturb.bin', 'wb') as f:
            pickle.dump(val_matrix, f)
        avg_per_NN_new = np.average(val_matrix)

        with open(fr'{exp_path}/val_matrix_best.bin', 'rb') as f:
            val_matrix = pickle.load(f)
        avg_per_NN_prevbest = np.average(val_matrix)

        print(f'gen: {best_NN_gen} | val_fit_prevbest: {int(avg_per_NN_prevbest)} | val_fit_new: {int(avg_per_NN_new)}')

    else:

        with open(fr'{exp_path}/val_matrix_best.bin', 'wb') as f:
            pickle.dump(val_matrix, f)
        avg_per_NN_new = np.average(val_matrix)

        print(f'gen: {best_NN_gen} | EA_fit: {int(avg_data[best_NN_gen])} | val_fit_prev: {int(avg_per_NN_prev[best_NN_ind])} | val_fit_new: {int(avg_per_NN_new)}')




def rerun_NNs_multievo(name, num_NNs=20, num_seeds=100):

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    exp_path = fr'{data_dir}/{name}'
    env_path = fr'{exp_path}/.env'
    envconf = de.dotenv_values(env_path)

    print(f'running: {name}')
    if Path(fr'{exp_path}/val_results_cen.txt').is_file():
        return print(f'val_results_cen already exists')

    with open(fr'{data_dir}/{name}/fitness_spread_per_generation.bin','rb') as f:
        data = pickle.load(f)
    num_steps,num_gen,num_eps,num_indivs = data.shape
    data_genxpop = np.mean(data, axis=2) # average across episodes
    avg_data = np.mean(data_genxpop, axis=1) # average across instances in population
    avg_data_summed_across_indivs = np.sum(avg_data, axis=1) # sum bw each agent

    # print(avg_data, avg_data.shape)

    # --> NNcen / dist center
    if envconf['SIM_TYPE'].startswith('walls'):
        top_ind = np.argsort(avg_data_summed_across_indivs)[:num_NNs] # min : top
    elif envconf['SIM_TYPE'].startswith('nowalls'):
        top_ind = np.argsort(avg_data_summed_across_indivs)[-1:-num_NNs-1:-1] # max : top
    else:
        raise ValueError('SIM_TYPE not recognized')
    top_fit = [avg_data[i,:].round(0) for i in top_ind]

    val_matrix = np.zeros((num_NNs,
                           num_seeds,
                           num_indivs))
    print(f'Validation matrix shape (num_NNs, num_seeds): {val_matrix.shape}')

    # pack inputs for multiprocessing map
    mp_inputs = []
    for g in top_ind:
        pvs = []
        for indiv in range(num_indivs):
            with open(fr'{data_dir}/{name}/gen{g}_NNcen_pickle_ag{indiv}.bin','rb') as f: 
                pv = pickle.load(f)
            pvs.append(pv)

        for s in range(num_seeds):
            mp_inputs.append( (None, pvs, None, s, env_path) ) # model_tuple=None, load_dir=None

    # run agent NNs in parallel
    with mp.Pool() as pool:
        results = pool.starmap_async(start_multi, mp_inputs)
        pool.close()
        pool.join()
    results_list = results.get()

    # skip to start of each seed series/chunk + allocate fitness to save matrix
    if envconf['SIM_TYPE'].startswith('walls'):
        for i,c in enumerate(range(0, len(results_list), num_seeds)):
            for s,(time_taken, dist_from_patch, data) in enumerate(results_list[c : c + num_seeds]):
                for indiv in range(num_indivs):
                    val_matrix[i,s,indiv] = int(time_taken[indiv])
    elif envconf['SIM_TYPE'].startswith('nowalls'):
        for i,c in enumerate(range(0, len(results_list), num_seeds)):
            for s,(first_time_consume, total_res_collected) in enumerate(results_list[c : c + num_seeds]):
                for indiv in range(num_indivs):
                    val_matrix[i,s,indiv] = int(total_res_collected)
                    # val_matrix_time[i,s,indiv] = int(first_time_consume)

    # saving protocol for noise/perturb/regular
    with open(fr'{exp_path}/val_matrix_cen.bin', 'wb') as f:
        pickle.dump(val_matrix, f)
    avg_per_NN = np.average(val_matrix, axis=1).round(0)

    with open(fr'{exp_path}/val_results_cen.txt', 'w') as f:
        f.write(f'Validation matrix shape (num_NNs, num_seeds): {val_matrix.shape}\n')
        for i, ef, vf in zip(top_ind, top_fit, avg_per_NN):
            f.write(str(f'gen: {i} | EA_fit: {*ef,} | val_fit: {*vf,}\n'))

    for i, ef, vf in zip(top_ind, top_fit, avg_per_NN):
        print(f'gen: {i} | EA_fit: {*ef,} | val_fit: {*vf,}')


def rerun_best_val_NN_multievo(name, num_seeds=1000, perturb_type=None):

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    exp_path = fr'{data_dir}/{name}'
    env_path = fr'{exp_path}/.env'

    if perturb_type is not None:
        print(f'running: {name} + {perturb_type} perturb')
        if Path(fr'{exp_path}/val_matrix_best_{perturb_type}_perturb.bin').is_file():
            return print(f'val_matrix_best_{perturb_type}_perturb already exists')
    else:
        print(f'running: {name}')
        if Path(fr'{exp_path}/val_matrix_best.bin').is_file():
            return print(f'val_results_best already exists')

    with open(fr'{data_dir}/{name}/fitness_spread_per_generation.bin','rb') as f:
        data = pickle.load(f)
    num_steps,num_gen,num_eps,num_indivs = data.shape
    data_genxpop = np.mean(data, axis=2) # average across episodes
    avg_data = np.mean(data_genxpop, axis=1) # average across individuals in population
    avg_data_summed_across_indivs = np.sum(avg_data, axis=1) # sum bw each agent

    with open(fr'{exp_path}/val_matrix_cen.bin','rb') as f: # 20 top NNs
        val_matrix_prev = pickle.load(f)
    avg_per_NN_prev = np.average(val_matrix_prev, axis=1)
    avg_per_NN_prev_summed_across_indivs = np.sum(avg_per_NN_prev, axis=1)
    best_NN_ind = np.argmin(avg_per_NN_prev_summed_across_indivs)

    valmatrix_inds = np.argsort(avg_data_summed_across_indivs)[:20] # min : top
    best_NN_gen = valmatrix_inds[best_NN_ind]

    pvs = []
    for indiv in range(num_indivs):
        with open(fr'{data_dir}/{name}/gen{best_NN_gen}_NNcen_pickle_ag{indiv}.bin','rb') as f:
            pv = pickle.load(f)
        pvs.append(pv)

    # pack inputs for multiprocessing map
    mp_inputs = []
    for s in range(num_seeds):
        mp_inputs.append( (None, pvs, None, s, env_path) ) # model_tuple=None, load_dir=None

    # run agent NNs in parallel
    with mp.Pool() as pool:
        results = pool.starmap_async(start_multi, mp_inputs)
        pool.close()
        pool.join()
    results_list = results.get()

    # allocate fitnesses
    val_matrix = np.zeros((num_seeds, num_indivs))
    for s,(time_taken, dist_from_patch, data) in enumerate(results_list):
        for indiv in range(num_indivs):
            val_matrix[s,indiv] = int(time_taken[indiv])

    # saving protocol for perturb/regular
    if perturb_type == 'ghostexplorer':

        with open(fr'{exp_path}/val_matrix_best_{perturb_type}_perturb.bin', 'wb') as f:
            pickle.dump(val_matrix, f)
        avg_per_NN_new = np.average(val_matrix, axis=0).round(0)

        with open(fr'{exp_path}/val_matrix_best.bin', 'rb') as f:
            val_matrix = pickle.load(f)
        avg_per_NN_prevbest = np.average(val_matrix, axis=0).round(0)

        with open(fr'{exp_path}/val_matrix_best_ghostexploiter_perturb.bin', 'rb') as f:
            val_matrix = pickle.load(f)
        avg_per_NN_exploiter = np.average(val_matrix, axis=0).round(0)

        print(f'gen: {best_NN_gen} | val_fit_prevbest: {avg_per_NN_prevbest} | val_fit_exploiter: {avg_per_NN_exploiter} | val_fit_new: {avg_per_NN_new}')

    elif perturb_type is not None:

        with open(fr'{exp_path}/val_matrix_best_{perturb_type}_perturb.bin', 'wb') as f:
            pickle.dump(val_matrix, f)
        avg_per_NN_new = np.average(val_matrix, axis=0).round(0)

        with open(fr'{exp_path}/val_matrix_best.bin', 'rb') as f:
            val_matrix = pickle.load(f)
        avg_per_NN_prevbest = np.average(val_matrix, axis=0).round(0)

        print(f'gen: {best_NN_gen} | val_fit_prevbest: {avg_per_NN_prevbest} | val_fit_new: {avg_per_NN_new}')

    else:

        with open(fr'{exp_path}/val_matrix_best.bin', 'wb') as f:
            pickle.dump(val_matrix, f)
        avg_per_NN_new = np.average(val_matrix, axis=0).round(0)

        print(f'gen: {best_NN_gen} | EA_fit: {avg_data[best_NN_gen]} | val_fit_prev: {avg_per_NN_prev[best_NN_ind]} | val_fit_new: {avg_per_NN_new}')


def rerun_best_val_NN_multievo_asindiv(name, num_seeds=1000, perturb_type=None):

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    exp_path = fr'{data_dir}/{name}'
    env_path = fr'{exp_path}/.env'

    if perturb_type is not None:
        print(f'running: {name} + {perturb_type} perturb')
        if Path(fr'{exp_path}/val_matrix_best_{perturb_type}_ag0_perturb.bin').is_file():
            return print(f'val_matrix_best_{perturb_type}_perturb already exists')
    else:
        print(f'running: {name}')
        if Path(fr'{exp_path}/val_matrix_best_ag0.bin').is_file():
            return print(f'val_results_best already exists')

    with open(fr'{data_dir}/{name}/fitness_spread_per_generation.bin','rb') as f:
        data = pickle.load(f)
    num_steps,num_gen,num_eps,num_indivs = data.shape
    data_genxpop = np.mean(data, axis=2) # average across episodes
    avg_data = np.mean(data_genxpop, axis=1) # average across individuals in population
    avg_data_summed_across_indivs = np.sum(avg_data, axis=1) # sum bw each agent

    with open(fr'{exp_path}/val_matrix_cen.bin','rb') as f: # 20 top NNs
        val_matrix_prev = pickle.load(f)
    avg_per_NN_prev = np.average(val_matrix_prev, axis=1)
    avg_per_NN_prev_summed_across_indivs = np.sum(avg_per_NN_prev, axis=1)
    best_NN_ind = np.argmin(avg_per_NN_prev_summed_across_indivs)

    valmatrix_inds = np.argsort(avg_data_summed_across_indivs)[:20] # min : top
    best_NN_gen = valmatrix_inds[best_NN_ind]

    fits = []
    for indiv in range(num_indivs):
        with open(fr'{data_dir}/{name}/gen{best_NN_gen}_NNcen_pickle_ag{indiv}.bin','rb') as f:
            pv = pickle.load(f)

        # pack inputs for multiprocessing map
        mp_inputs = []
        for s in range(num_seeds):
            mp_inputs.append( (None, pv, None, s, env_path) ) # model_tuple=None, load_dir=None

        # run agent NNs in parallel
        with mp.Pool() as pool:
            results = pool.starmap_async(start, mp_inputs)
            pool.close()
            pool.join()
        results_list = results.get()

        # allocate fitnesses
        val_matrix = np.zeros((num_seeds))
        for s,(time_taken, dist_from_patch, data) in enumerate(results_list):
            val_matrix[s] = int(time_taken)
            # val_matrix[s] = int(dist_from_patch)

        # saving protocol
        with open(fr'{exp_path}/val_matrix_best_{perturb_type}_ag{indiv}_perturb.bin', 'wb') as f:
            pickle.dump(val_matrix, f)
        avg_per_NN_new = np.average(val_matrix)
        fits.append(int(avg_per_NN_new))

    print(f'gen: {best_NN_gen} | val_fit_prevbest: {avg_per_NN_prev[best_NN_ind]} | val_fit_new: {fits}')



def run_randomwalk(name, num_RWs=1, num_seeds=100):

    print(f'running: random walk {name}')

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    exp_path = fr'{data_dir}/nonNN/{name}'

    val_matrix = np.zeros((num_RWs,
                           num_seeds))
    print(f'Validation matrix shape (num_NNs, num_seeds): {val_matrix.shape}')

    # pack inputs for multiprocessing map
    mp_inputs = []
    for n in range(num_RWs):
        for s in range(num_seeds):
            mp_inputs.append( (None, None, None, s, None) ) # model_tuple=None, pv=None, load_dir=None, env_path=None

    # run agents in parallel
    with mp.Pool() as pool:
        results = pool.starmap_async(start, mp_inputs)
        pool.close()
        pool.join()
    results_list = results.get()

    # skip to start of each seed series/chunk + allocate fitness to save matrix
    for i,c in enumerate(range(0, len(results_list), num_seeds)):
        for s,fitnesses in enumerate(results_list[c : c + num_seeds]):
            val_matrix[i,s] = round(fitnesses[0],0)

    # saving protocol 
    with open(fr'{exp_path}.bin', 'wb') as f:
        pickle.dump(val_matrix, f)
    avg_per_NN = np.average(val_matrix).round(1)

    print(f'val_fit: {int(avg_per_NN)}')


def run_perfect(name='perfect', num_RWs=1, num_seeds=100):

    print(f'running: perfect')

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    exp_path = fr'{data_dir}/nonNN/{name}'

    val_matrix = np.zeros((num_RWs,
                           num_seeds))
    print(f'Validation matrix shape (num_NNs, num_seeds): {val_matrix.shape}')

    x_min, x_max = 20,980
    y_min, y_max = 20,980
    target = np.array([400,400])
    res_radius = 50
    agent_max_vel = 2

    # numerically compute perfect trajectories
    for s in range(num_seeds):
        
        # fire two blanks before location call
        _ = np.random.randint(x_min, x_max)
        _ = np.random.randint(x_min, x_max)

        x = np.random.randint(x_min, x_max)
        y = np.random.randint(y_min, y_max)
        # orient = np.random.uniform(0, 2*np.pi)

        start = np.array([x,y])

        dist = np.linalg.norm(start - target)
        dist_to_edge = dist - res_radius

        time_to_edge = dist_to_edge/agent_max_vel

        val_matrix[0,s] = time_to_edge

    # saving protocol 
    with open(fr'{exp_path}.bin', 'wb') as f:
        pickle.dump(val_matrix, f)

    print(f'val_fit: {int(np.average(val_matrix))}')



if __name__ == '__main__':

    names = []

    # run_randomwalk('rotdiff_0p0005')
    # run_randomwalk('rotdiff_0p001')
    # run_randomwalk('rotdiff_0p005')
    # run_randomwalk('rotdiff_0p01')
    # run_randomwalk('rotdiff_0p05')
    # run_randomwalk('rotdiff_0p10')
    # run_randomwalk('rotdiff_0p50')
    # run_perfect()

    # run_randomwalk('rotdiff_0p0005_randbouncy')
    # run_randomwalk('rotdiff_0p001_randbouncy')
    # run_randomwalk('rotdiff_0p005_randbouncy')
    # run_randomwalk('rotdiff_0p01_randbouncy')
    # run_randomwalk('rotdiff_0p05_randbouncy')
    # run_randomwalk('rotdiff_0p10_randbouncy')
    # run_randomwalk('rotdiff_0p50_randbouncy')

    # run_randomwalk('rotdiff_0p01_randbouncy_N2')
    # run_randomwalk('rotdiff_0p01_randbouncy_N5')
    # run_randomwalk('rotdiff_0p01_randbouncy_N10')
    # run_randomwalk('rotdiff_0p01_randbouncy_N15')
    # run_randomwalk('rotdiff_0p01_randbouncy_N20')

    # # vis
    # for name in [f'sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)

    # # cnn
    # for name in [f'sc_CNN12_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN15_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN16_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)

    # # fnn
    # for name in [f'sc_CNN14_FNN1_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # # for name in [f'sc_CNN14_FNN3_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    # #     names.append(name)
    # for name in [f'sc_CNN14_FNN4_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN8_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2x2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # # for name in [f'sc_CNN14_FNN2x3_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    # #     names.append(name)
    # for name in [f'sc_CNN14_FNN2x4_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # # for name in [f'sc_CNN14_FNN2x8_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    # #     names.append(name)
    # for name in [f'sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    
    # # fov
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov2_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov3_rep{x}' for x in range(20)]:
    #     names.append(name)
    # # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep{x}' for x in range(20)]:
    # #     names.append(name)
    # # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov45_rep{x}' for x in range(20)]:
    # #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov5_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov6_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov7_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov875_rep{x}' for x in range(20)]:
    #     names.append(name)

    # # dist
    # # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_minmax_rep{x}' for x in range(20)]:
    # #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep{x}' for x in range(20)]:
    # #     names.append(name)
    # # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep{x}' for x in range(20)]:
    # #     names.append(name)
    # # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_rep{x}' for x in range(20)]:
    # #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)

    # for s in [10000,20000,30000,40000]:
    #     for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed{str(int(s/1000))}k_rep{x}' for x in range(20)]:
    #         names.append(name)
    # for s in [10000,20000,30000,40000]:
    #     for name in [f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed{str(int(s/1000))}k_rep{x}' for x in range(20)]:
    #         names.append(name)
    # for s in [10000,20000,30000,40000]:
    #     for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_seed{str(int(s/1000))}k_rep{x}' for x in range(20)]:
    #         names.append(name)
    # for s in [10000,20000,30000,40000]:
    #     for name in [f'sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed{str(int(s/1000))}k_rep{x}' for x in range(20)]:
    #         names.append(name)
    # for s in [10000,20000,30000,40000]:
    #     for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed{str(int(s/1000))}k_rep{x}' for x in range(20)]:
    #         names.append(name)
    # for s in [10000,20000,30000,40000]:
    #     for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed{str(int(s/1000))}k_rep{x}' for x in range(20)]:
    #         names.append(name)

    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_n1_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_n2_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_n3_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_n4_rep{x}' for x in range(20)]:
    #     names.append(name)

    # for name in [f'sc_lm_CNN14_FNN2_p50e20_vis8_lm100_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_lm_CNN14_FNN2_p50e20_vis10_lm100_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_lm_CNN14_FNN2_p50e20_vis16_lm100_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_lm_CNN14_FNN2_p50e20_vis24_lm100_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_lm_CNN14_FNN2_p50e20_vis32_lm100_rep{x}' for x in range(20)]:
    #     names.append(name)



    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
        
    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)


    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_2xpinball_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_2xpinball_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_maxWF_2xpinball_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN17_FNN16_p50e20_vis16_2xpinball_rep{x}' for x in range(20)]:
    #     names.append(name)

    # n = 20
    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_fov94_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_fov97_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN16_p50e20_vis16_PGPE_ss20_mom8_fov94_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN16_p50e20_vis32_PGPE_ss20_mom8_fov97_rep{x}' for x in range(n)]:
    #     names.append(name)


    n = 40
    # for name in [f'sc_N1_NRW0_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N3_NRW0_ND2_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # # for name in [f'sc_N4_NRW0_ND3_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    # #     names.append(name)
    # for name in [f'sc_N4_NRW0_ND3_CNN14_FNN16_vis8_rep{x+40}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N5_NRW0_ND4_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N5_NRW0_ND4_CNN14_FNN16_vis8_rep{x+40}' for x in range(n)]:
    #     names.append(name)
    # # for name in [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    # #     names.append(name)
    for name in [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_rep{x+40}' for x in range(n)]:
        names.append(name)

    # for name in [f'sc_N2_NRW1_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N3_NRW1_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N4_NRW1_ND2_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # # for name in [f'sc_N4_NRW1_ND2_CNN14_FNN16_vis8_rep{x+40}' for x in range(n)]: ## not complete yet
    # #     names.append(name)
    # for name in [f'sc_N5_NRW1_ND3_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    for name in [f'sc_N6_NRW1_ND4_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
        names.append(name)

    # for name in [f'sc_N3_NRW2_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N4_NRW2_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N5_NRW2_ND2_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # # for name in [f'sc_N5_NRW2_ND2_CNN14_FNN16_vis8_rep{x+40}' for x in range(n)]: ## not complete yet
    # #     names.append(name)
    for name in [f'sc_N6_NRW2_ND3_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
        names.append(name)
    # for name in [f'sc_N6_NRW2_ND3_CNN14_FNN16_vis8_rep{x+40}' for x in range(n)]:
    #     names.append(name)

    # for name in [f'sc_N4_NRW3_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N5_NRW3_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    for name in [f'sc_N6_NRW3_ND2_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
        names.append(name)
    # for name in [f'sc_N6_NRW3_ND2_CNN14_FNN16_vis8_rep{x+40}' for x in range(n)]: ## not complete yet
    #     names.append(name)

    # for name in [f'sc_N5_NRW4_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    for name in [f'sc_N6_NRW4_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
        names.append(name)

    for name in [f'sc_N6_NRW5_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
        names.append(name)

    # # for name in [f'sc_N11_NRW0_ND10_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    # #     names.append(name)
    # # for name in [f'sc_N11_NRW5_ND5_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    # #     names.append(name)
    # for name in [f'sc_N11_NRW10_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N16_NRW15_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # # for name in [f'sc_N21_NRW0_ND20_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    # #     names.append(name)
    # # for name in [f'sc_N21_NRW10_ND10_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    # #     names.append(name)
    # for name in [f'sc_N21_NRW20_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)

    # n = 20
    # for name in [f'sc_N1_NRW0_ND0_CNN14_FNN16_vis8_ghost_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_ghost_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_ghost_rep{x}' for x in range(n)]:
    #     names.append(name)

    # for name in [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_SinitAg200_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_SinitAg300_rep{x}' for x in range(n)]:
    #     names.append(name)

    # for name in [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_SinitRes100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_SinitRes200_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_SinitRes300_rep{x}' for x in range(n)]:
    #     names.append(name)

    # for name in [f'sc_N3_NRW0_ND2_CNN14_FNN16_vis8_SinitRes100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N3_NRW0_ND2_CNN14_FNN16_vis8_SinitRes200_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N3_NRW0_ND2_CNN14_FNN16_vis8_SinitRes300_rep{x}' for x in range(n)]:
    #     names.append(name)

    # for name in [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_SinitAg200_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_SinitAg300_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_SinitRes100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_SinitRes200_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_SinitRes300_rep{x}' for x in range(n)]:
    #     names.append(name)

    # for name in [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW1_ND4_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW2_ND3_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW3_ND2_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW4_ND1_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW5_ND0_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N5_NRW0_ND4_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N5_NRW1_ND3_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N5_NRW2_ND2_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N5_NRW3_ND1_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N5_NRW4_ND0_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N4_NRW0_ND3_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N4_NRW1_ND2_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N4_NRW2_ND1_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N4_NRW3_ND0_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N3_NRW0_ND2_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N3_NRW1_ND1_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N3_NRW2_ND0_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N2_NRW1_ND0_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)



    # n = 40
    # for name in [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW1_ND4_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW2_ND3_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW3_ND2_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW4_ND1_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW5_ND0_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)

    # for name in [f'sc_N5_NRW0_ND4_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N5_NRW1_ND3_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N5_NRW2_ND2_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N5_NRW3_ND1_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N5_NRW4_ND0_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)

    # for name in [f'sc_N4_NRW0_ND3_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N4_NRW1_ND2_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N4_NRW2_ND1_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N4_NRW3_ND0_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)

    # for name in [f'sc_N3_NRW0_ND2_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N3_NRW1_ND1_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N3_NRW2_ND0_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)

    # for name in [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N2_NRW1_ND0_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)

    # for name in [f'sc_N11_NRW10_ND0_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N16_NRW15_ND0_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N21_NRW20_ND0_CNN14_FNN16_vis8_nocoll_rep{x}' for x in range(n)]:
    #     names.append(name)

    # for name in [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_nocollpatch_rep{x}' for x in range(n)]:
    #     names.append(name)


    # for name in [f'sc_N6_NRW0_ND5_CNN18_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # # for name in [f'sc_N6_NRW0_ND5_CNN14_FNN64_vis8_rep{x}' for x in range(n)]:
    # #     names.append(name)
    # # for name in [f'sc_N6_NRW0_ND5_CNN14_FNN16x2_vis8_rep{x}' for x in range(n)]:
    # #     names.append(name)
    # for name in [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis12_rep{x}' for x in range(n)]:
    #     names.append(name)
    # # for name in [f'sc_N6_NRW0_ND5_CNN18_FNN16x2_vis8_rep{x}' for x in range(n)]:
    # #     names.append(name)
    # # for name in [f'sc_N6_NRW0_ND5_CNN18_FNN64x2_vis8_rep{x}' for x in range(n)]:
    # #     names.append(name)



    # n = 20
    # for name in [f'sc_N2_multi_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N2_multi_CNN14_FNN16_vis8_SinitAg100_rep{x}' for x in range(n)]:
    #     names.append(name)

    # for name in [f'sc_N3_multi_CNN14_FNN16_vis8_rep{x}' for x in range(n)]: ###
    #     names.append(name)
    # for name in [f'sc_N4_multi_CNN14_FNN16_vis8_rep{x}' for x in range(n)]: ###
    #     names.append(name)
    # for name in [f'sc_N5_multi_CNN14_FNN16_vis8_rep{x}' for x in range(n)]: ###
    #     names.append(name)

    # for name in [f'sc_N6_multi_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)

    # names = [
    #     'sc_CNN18_FNN2x64_p50e20_vis32_fov97_rep0',
    #     'sc_CNN18_FNN2x64_p50e20_vis32_fov97_rep1',
    #     'sc_CNN18_FNN2x64_p50e20_vis32_fov97_rep2',
    # ]


    # for name in [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_SinitAg100_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)

    # n = 40
    # for name in [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW1_ND4_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW2_ND3_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW3_ND2_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW4_ND1_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N6_NRW5_ND0_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)

    # for name in [f'sc_N5_NRW0_ND4_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N5_NRW1_ND3_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N5_NRW2_ND2_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N5_NRW3_ND1_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N5_NRW4_ND0_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)

    # for name in [f'sc_N4_NRW0_ND3_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N4_NRW1_ND2_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N4_NRW2_ND1_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N4_NRW3_ND0_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)

    # for name in [f'sc_N3_NRW0_ND2_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N3_NRW1_ND1_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N3_NRW2_ND0_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)

    # for name in [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'sc_N2_NRW1_ND0_CNN14_FNN16_vis8_collinput_rep{x}' for x in range(n)]:
    #     names.append(name)


    for name in names:  
        rerun_NNs(name)
        # rerun_NNs(name, perturb_type='nosocial')
        # rerun_NNs(name, perturb_type='allRW')
        # rerun_NNs(name, perturb_type='allD')
        # rerun_NNs(name, perturb_type='selfsocial')
        # rerun_NNs(name, perturb_type='ghostexploiter')
        # rerun_NNs(name, perturb_type='ghostexplorer')
        # rerun_NNs(name, perturb_type='N2-ghostexploiter')
        # rerun_NNs(name, perturb_type='N2-ghostexplorer')
        # rerun_NNs(name, perturb_type='nowalls-ghostexploiter')

        rerun_best_val_NN(name, num_seeds=5000)
        # rerun_best_val_NN(name, perturb_type='nosocial', num_seeds=5000)
        # rerun_best_val_NN(name, perturb_type='ghostexploiter', num_seeds=5000)
        # rerun_best_val_NN(name, perturb_type='ghostexplorer', num_seeds=5000)
        # rerun_best_val_NN(name, perturb_type='Nd+2', num_seeds=5000)
        # rerun_best_val_NN(name, perturb_type='Nr+2', num_seeds=5000)
        # rerun_best_val_NN(name, perturb_type='nosocial-dist', num_seeds=5000)
        # rerun_best_val_NN(name, perturb_type='SinitAg100-1', num_seeds=5000)
        # rerun_best_val_NN(name, perturb_type='ghostexploiters', num_seeds=5000)
        # rerun_best_val_NN(name, perturb_type='ghostexplorers', num_seeds=5000)
    
        # rerun_NNs(name, time=True)
        # rerun_NNs(name, perturb_type='ghostexploiter', time=True)
        # rerun_NNs(name, perturb_type='ghostexplorer', time=True)
        # rerun_NNs(name, noise_type='angle_n10')
        # rerun_NNs(name, noise_type='dist_n025')

        # rerun_NNs_multievo(name)
        # rerun_best_val_NN_multievo(name, num_seeds=5000)
        # rerun_best_val_NN_multievo_asindiv(name, perturb_type='nosocial', num_seeds=5000)
        # rerun_best_val_NN_multievo_asindiv(name, perturb_type='ghostexploiter', num_seeds=5000)
        # rerun_best_val_NN_multievo_asindiv(name, perturb_type='ghostexplorer', num_seeds=5000)