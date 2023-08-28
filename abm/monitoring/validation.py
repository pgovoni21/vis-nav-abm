from abm.start_sim import start

from pathlib import Path
import pickle
import numpy as np
import multiprocessing as mp


def rerun_topNNs(name, num_NNs=5, num_seeds=3):

    print(f'running: {name}')

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    exp_path = fr'{data_dir}/{name}'
    env_path = fr'{exp_path}/.env'

    with open(fr'{data_dir}/{name}/fitness_spread_per_generation.bin','rb') as f:
        trend_data = pickle.load(f)
    trend_data = np.array(trend_data)

    # top_trend_data = np.min(trend_data, axis=1) # min : top
    top_trend_data = np.max(trend_data, axis=1) # max : top
    # top_ind = np.argsort(top_trend_data)[:num_NNs] # min : top
    top_ind = np.argsort(top_trend_data)[-1:-num_NNs-1:-1] # max : top
    top_fit = [top_trend_data[i] for i in top_ind]

    val_matrix = np.zeros((num_NNs,
                            num_seeds))
    print(f'Validation matrix shape (num_NNs, num_seeds): {val_matrix.shape}')

    # pack inputs for multiprocessing map
    mp_inputs = []
    for g,f in zip(top_ind, top_fit):

        NN_pv_path = fr'{data_dir}/{name}/gen{g}/NN0_af{int(f)}/NN_pickle.bin'
        with open(NN_pv_path,'rb') as f:
            pv = pickle.load(f)

        for s in range(num_seeds):
            mp_inputs.append( (None, pv, None, s, env_path) ) # model_tuple=None, save_ext=None

    # run agent NNs in parallel
    with mp.Pool() as pool:
        results = pool.starmap_async( start, mp_inputs)
        pool.close()
        pool.join()

    # unpack results into matrix + save
    results_list = results.get()

    # skip to start of each seed series/chunk + allocate fitness to save matrix
    for i,c in enumerate(range(0, len(results_list), num_seeds)):
        for s,(_, fitnesses, _) in enumerate(results_list[c : c + num_seeds]):

            val_matrix[i,s] = round(fitnesses[0],0)

    # dump raw data
    with open(fr'{exp_path}/val_matrix.bin', 'wb') as f:
        pickle.dump(val_matrix, f)

    # take avg + print results
    avg_per_NN = np.average(val_matrix, axis=1).round(1)
    for i, ef, vf in zip(top_ind, top_fit, avg_per_NN):
        print(f'gen: {i} | EA_fit: {ef} | val_fit: {vf}')

    # writing txt file
    with open(fr'{exp_path}/val_results.txt', 'w') as f:
        f.write(f'Validation matrix shape (num_NNs, num_seeds): {val_matrix.shape}\n')
        for i, ef, vf in zip(top_ind, top_fit, avg_per_NN):
            f.write(str(f'gen: {i} | EA_fit: {ef} | val_fit: {vf}\n'))


if __name__ == '__main__':

    names = []

    # for x in range(4):
    #     names.append(f'doublepoint_CNN1128_FNN2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}')
    # for x in range(4):
    #     names.append(f'doublepoint_CNN1128_CTRNN2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}') 
    # for x in range(4):
    #     names.append(f'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep{x}')

    # for x in range(6):
    #     names.append(f'doublepoint_CNN1128_FNN2_p25e5g1000_sig0p1_vis8_dirfit_simp_rep{x}')
    # for x in range(6):
    #     names.append(f'doublepoint_CNN1128_CTRNN2_p25e5g1000_sig0p1_vis8_dirfit_simp_rep{x}')
    # for x in range(6):
    #     names.append(f'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_simp_rep{x}')

    # for x in range(4):
    #     names.append(f'doublepoint_CNN1128_FNN2_p25e5g1000_sig0p1_vis8_dirfit_other0_rep{x}')
    # for x in range(4):
    #     names.append(f'doublepoint_CNN1128_CTRNN2_p25e5g1000_sig0p1_vis8_dirfit_other0_rep{x}')
    # for x in range(4):
    #     names.append(f'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_other0_rep{x}')

    for name in names:
        rerun_topNNs(name, num_NNs=25, num_seeds=100)