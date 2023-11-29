from abm.start_sim import start

from pathlib import Path
import pickle
import numpy as np
import multiprocessing as mp


def rerun_topNNs(name, num_NNs=20, num_seeds=100):

    print(f'running: {name}')

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    exp_path = fr'{data_dir}/{name}'
    env_path = fr'{exp_path}/.env'

    if Path(fr'{exp_path}/val_matrix.bin').is_file():
        return print(f'val_matrix already exists')
    
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

        NN_pv_path = fr'{data_dir}/{name}/gen{g}_NN0_pickle.bin'

        with open(NN_pv_path,'rb') as f:
            pv = pickle.load(f)

        for s in range(num_seeds):
            mp_inputs.append( (None, pv, None, s, env_path) ) # model_tuple=None, load_dir=None

    # run agent NNs in parallel
    with mp.Pool() as pool:
        results = pool.starmap_async( start, mp_inputs)
        pool.close()
        pool.join()
    results_list = results.get()

    # skip to start of each seed series/chunk + allocate fitness to save matrix
    for i,c in enumerate(range(0, len(results_list), num_seeds)):
        for s,fitnesses in enumerate(results_list[c : c + num_seeds]):
            val_matrix[i,s] = round(fitnesses[0],0)

    # take avg + print results
    avg_per_NN = np.average(val_matrix, axis=1).round(1)
    for i, ef, vf in zip(top_ind, top_fit, avg_per_NN):
        print(f'gen: {i} | EA_fit: {int(ef)} | val_fit: {int(vf)}')

    # dump raw data + txt file
    with open(fr'{exp_path}/val_matrix.bin', 'wb') as f:
        pickle.dump(val_matrix, f)
    with open(fr'{exp_path}/val_results.txt', 'w') as f:
        f.write(f'Validation matrix shape (num_NNs, num_seeds): {val_matrix.shape}\n')
        for i, ef, vf in zip(top_ind, top_fit, avg_per_NN):
            f.write(str(f'gen: {i} | EA_fit: {int(ef)} | val_fit: {int(vf)}\n'))


def rerun_cenNNs(name, num_NNs=20, num_seeds=100):

    print(f'running: {name}')

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    exp_path = fr'{data_dir}/{name}'
    env_path = fr'{exp_path}/.env'

    # if Path(fr'{exp_path}/val_matrix_cen.bin').is_file():
    #     return print(f'val_matrix_cen already exists')
    
    with open(fr'{data_dir}/{name}/fitness_spread_per_generation.bin','rb') as f:
        data = pickle.load(f)
    data_genxpop = np.mean(data, axis=2) # average across episodes
    avg_data = np.mean(data_genxpop, axis=1) # average across individuals in population --> different than rerun_topNNs()

    top_ind = np.argsort(avg_data)[:num_NNs] # min : top
    # top_ind = np.argsort(top_data)[-1:-num_NNs-1:-1] # max : top
    top_fit = [avg_data[i] for i in top_ind]

    val_matrix = np.zeros((num_NNs,
                            num_seeds))
    print(f'Validation matrix shape (num_NNs, num_seeds): {val_matrix.shape}')

    # pack inputs for multiprocessing map
    mp_inputs = []
    for g,f in zip(top_ind, top_fit):

        with open(fr'{data_dir}/{name}/gen{g}_NNcen_pickle.bin','rb') as f:
            pv = pickle.load(f)

        for s in range(num_seeds):
            mp_inputs.append( (None, pv, None, s, env_path) ) # model_tuple=None, load_dir=None

    # run agent NNs in parallel
    with mp.Pool() as pool:
        results = pool.starmap_async( start, mp_inputs)
        pool.close()
        pool.join()
    results_list = results.get()

    # skip to start of each seed series/chunk + allocate fitness to save matrix
    for i,c in enumerate(range(0, len(results_list), num_seeds)):
        for s,fitnesses in enumerate(results_list[c : c + num_seeds]):
            val_matrix[i,s] = round(fitnesses[0],0)

    # take avg + print results
    avg_per_NN = np.average(val_matrix, axis=1).round(1)
    for i, ef, vf in zip(top_ind, top_fit, avg_per_NN):
        print(f'gen: {i} | EA_fit: {int(ef)} | val_fit: {int(vf)}')

    # dump raw data + txt file
    with open(fr'{exp_path}/val_matrix_cen.bin', 'wb') as f:
        pickle.dump(val_matrix, f)
    with open(fr'{exp_path}/val_results_cen.txt', 'w') as f:
        f.write(f'Validation matrix shape (num_NNs, num_seeds): {val_matrix.shape}\n')
        for i, ef, vf in zip(top_ind, top_fit, avg_per_NN):
            f.write(str(f'gen: {i} | EA_fit: {int(ef)} | val_fit: {int(vf)}\n'))


if __name__ == '__main__':

    names = []
    
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_block52_rep{x+11}' for x in range(9)]:
    #     names.append(name) ## --> need to convert model params
    # for name in [f'sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_block52_rep{x}' for x in range(20)]:
    #     names.append(name) ## --> need to convert model params
    

    for name in [f'sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep{x}' for x in range(9)]:
        names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep{x}' for x in range(9)]:
        names.append(name)
    for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov45_rep{x}' for x in range(11)]:
        names.append(name)



    for name in names:
        rerun_cenNNs(name)