from abm.start_sim import start

from pathlib import Path
import pickle
import numpy as np
import multiprocessing as mp


def rerun_topNNs(name, num_NNs=5, num_seeds=3, best=True):

    print(f'running: {name}')

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    exp_path = fr'{data_dir}/{name}'
    env_path = fr'{exp_path}/.env'

    with open(fr'{data_dir}/{name}/fitness_spread_per_generation.bin','rb') as f:
        data = pickle.load(f)
    data_genxpop = np.mean(data, axis=2)

    top_data = np.min(data_genxpop, axis=1) # min : top
    # top_data = np.max(data_genxpop, axis=1) # max : top
    top_ind = np.argsort(top_data)[:num_NNs] # min : top
    # top_ind = np.argsort(top_data)[-1:-num_NNs-1:-1] # max : top
    top_fit = [top_data[i] for i in top_ind]

    val_matrix = np.zeros((num_NNs,
                            num_seeds))
    print(f'Validation matrix shape (num_NNs, num_seeds): {val_matrix.shape}')

    # pack inputs for multiprocessing map
    mp_inputs = []
    for g,f in zip(top_ind, top_fit):

        if best: NN_pv_path = fr'{data_dir}/{name}/gen{g}_NN0_pickle.bin'
        else:    NN_pv_path = fr'{data_dir}/{name}/gen{g}_NNcen_pickle.bin'

        with open(NN_pv_path,'rb') as f:
            pv = pickle.load(f)

        for s in range(num_seeds):
            mp_inputs.append( (None, pv, None, s, env_path) ) # model_tuple=None, save_ext=None

    # run agent NNs in parallel
    with mp.Pool() as pool:
        results = pool.starmap_async( start, mp_inputs)
        pool.close()
        pool.join()
    results_list = results.get()

    # skip to start of each seed series/chunk + allocate fitness to save matrix
    for i,c in enumerate(range(0, len(results_list), num_seeds)):
        for s,(_, fitnesses, _) in enumerate(results_list[c : c + num_seeds]):
            val_matrix[i,s] = round(fitnesses[0],0)

    # take avg + print results
    avg_per_NN = np.average(val_matrix, axis=1).round(1)
    for i, ef, vf in zip(top_ind, top_fit, avg_per_NN):
        print(f'gen: {i} | EA_fit: {int(ef)} | val_fit: {int(vf)}')

    # dump raw data + txt file
    if best:
        with open(fr'{exp_path}/val_matrix.bin', 'wb') as f:
            pickle.dump(val_matrix, f)
        with open(fr'{exp_path}/val_results.txt', 'w') as f:
            f.write(f'Validation matrix shape (num_NNs, num_seeds): {val_matrix.shape}\n')
            for i, ef, vf in zip(top_ind, top_fit, avg_per_NN):
                f.write(str(f'gen: {i} | EA_fit: {int(ef)} | val_fit: {int(vf)}\n'))
    else:
        with open(fr'{exp_path}/val_matrix_cen.bin', 'wb') as f:
            pickle.dump(val_matrix, f)
        with open(fr'{exp_path}/val_results_cen.txt', 'w') as f:
            f.write(f'Validation matrix shape (num_NNs, num_seeds): {val_matrix.shape}\n')
            for i, ef, vf in zip(top_ind, top_fit, avg_per_NN):
                f.write(str(f'gen: {i} | EA_fit: {int(ef)} | val_fit: {int(vf)}\n'))


if __name__ == '__main__':

    names = []

    # names = [f'singlecorner_exp_CNN1124_FNN2_p200e20_vis8_rep{x}' for x in range(3)]
    # for name in names:
    #     rerun_topNNs(name, num_NNs=20, num_seeds=100)
    
    # names = [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_rep{x}' for x in range(2)]
    # for name in names:
    #     rerun_topNNs(name, num_NNs=20, num_seeds=100)

    # rerun_topNNs('singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_rep0', num_NNs=20, num_seeds=100)
    rerun_topNNs('singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_rep4', num_NNs=20, num_seeds=100, best=False)