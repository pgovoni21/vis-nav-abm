from abm.start_sim import start as start_sim
from abm.monitoring import plot_funcs

from abm.NN.model import WorldModel as Model

from pathlib import Path
import shutil, os, warnings, time
import numpy as np
import cma
import multiprocessing
import pickle
import zarr

class EvolAlgo():
    
    def __init__(self, arch=(50,128,2), activ='relu', RNN_type='fnn',
                 population_size=96, init_sigma=1, generations=500, episodes=5,
                 num_top_nn_saved=3, num_top_nn_plots=5, EA_save_name=None, start_seed=1000):
        
        # init_time = time.time()
        self.overall_time = time.time()

        # Pack model parameters 
        self.model_tuple = (arch, activ, RNN_type)

        # Calculate parameter vector size using an example NN (easy generalizable)
        param_vec_size = sum(p.numel() for p in Model(arch,activ,RNN_type).parameters())

        print(f'Model Architecture: {arch}')
        print(f'Total #Params: {param_vec_size}')

        # Evolution + Simulation parameters
        self.population_size = population_size
        self.init_sigma = init_sigma
        self.generations = generations
        self.episodes = episodes
        self.start_seed = start_seed

        # Initialize CMA-ES
        self.es = cma.CMAEvolutionStrategy(
            param_vec_size * [0], 
            self.init_sigma, 
            {'popsize': self.population_size,}
            )
        
        # Generate initial RNN parameters
        self.NN_param_vectors = self.es.ask()
        
        # # Initialize RNN instances for 0th generation
        # self.NNs = [Model(arch, activ, pv) for pv in self.NN_param_vectors]

        # Saving parameters
        self.fitness_evol = []
        self.num_top_nn_saved = num_top_nn_saved
        self.num_top_nn_plots = num_top_nn_plots
        self.EA_save_name = EA_save_name
        self.root_dir = Path(__file__).parent.parent.parent                                  # /home/govoni/gh-repos/DavidMezey-PyGame-ABM
        self.EA_save_dir = Path(self.root_dir, 'abm/data/simulation_data', EA_save_name)     # /home/govoni/gh-repos/DavidMezey-PyGame-ABM/abm/data/simulation_data/stationarypatch_CMAES
        
        self.mean_param_vec = np.zeros([generations, param_vec_size])
        self.std_param_vec = np.zeros([generations, param_vec_size])
        self.mean_param_vec[0,:] = self.es.mean
        self.std_param_vec[0,:] = self.es.stds

        # Create save directory + copy .env file over
        if os.path.isdir(self.EA_save_dir):
            warnings.warn("Temporary directory for env files is not empty and will be overwritten")
            shutil.rmtree(self.EA_save_dir)
        Path(self.EA_save_dir).mkdir()
        shutil.copy(
            Path(self.root_dir, '.env'), 
            Path(self.EA_save_dir, '.env')
            )
        
        # end_init_time = time.time() - init_time
        # print(f'Init Time: {round( end_init_time, 2)} sec')


    def fit_parallel(self):

        for i in range(self.generations):

            #### ---- Run sim + Save in running/nn/ep folder ---- ####

            print(f'---------------- Generation {i} ----------------')

            # determine PRNG seeds + reset for next generation
            # (circumventing multiprocessing bug where multiple children can have overlapping seeds)
            seeds_per_gen = range(self.start_seed, self.start_seed + self.episodes)
            self.start_seed += self.episodes

            # load inputs for each simulation instance
            sim_inputs_per_gen = []
            for n, pv in enumerate(self.NN_param_vectors):
                for e in range(self.episodes):
                    save_ext = fr'{self.EA_save_name}/running/NN{n}/ep{e}'
                    sim_inputs_per_gen.append( (self.model_tuple, pv, save_ext, seeds_per_gen[e]) )

            sim_time = time.time()

            # using process pool executor/manager
            with multiprocessing.Pool(processes=8) as pool:

                # issue all tasks to pool at once (non-blocking + ordered)
                results = pool.starmap_async( start_sim, sim_inputs_per_gen)

                # wait for all tasks to finish before continuing
                pool.close()
                pool.join()

            end_sim_time = time.time() - sim_time
            print(f'Generational Sim Run Time: {round( end_sim_time, 2)} sec')

            # convert results iterator to list
            results_list = results.get()

            # print('Sim Results:')
            # for result in results_list:
            #     print(result)


            #### ---- Find fitness averages across episodes ---- ####


            # set non-sim timer
            eval_time = time.time()

            # skip to start of each episode series/chunk
            fitness_gen = []
            for n, NN_index in enumerate(range(0, len(results_list), self.episodes)):

                # pull sim data for each episode
                fitness_ep = []
                for _, fitnesses, _, _ in results_list[NN_index : NN_index + self.episodes]:
                    fitness_ep.append(round(fitnesses[0],0))

                avg_fitness = np.mean(fitness_ep)
                fitness_gen.append(avg_fitness)

            # # list all averaged fitnesses
            # print(f'Fitnesses: {fitness_gen}')
            
            # Track top fitness per generation
            # top_fg = int(np.max(fitness_gen)) # max : first
            top_fg = int(np.min(fitness_gen)) # min : first
            avg_fg = int(np.mean(fitness_gen))
            print(f'Highest Across Gen: {top_fg} | Avg Across Gen: {avg_fg} ---')
            
            # print('Running Dir Contents:')
            # run_dir = Path(self.EA_save_dir, 'running')
            # contents = list(run_dir.iterdir())
            # for c in contents:
            #     print(c)


            #### ---- Save/plot performance + NN from top performers  ---- ####


            # cycle through the top X performers
            # top_indices = np.argsort(fitness_gen)[ : -1-self.num_top_nn_saved : -1] # max : first
            top_indices = np.argsort(fitness_gen)[ : self.num_top_nn_saved] # min : first

            # top_fitnesses = [int(fitness_gen[n_gen]) for n_gen in top_indices]
            # print(f'Saving performance for NNs with avg fitnesses: {top_fitnesses}')

            for n_top, n_gen in enumerate(top_indices):

                # pull saved sim runs from 'running' directory + archive in parent directory
                # ('running' directory is rewritten each generation)
                NN_load_name = fr'running/NN{n_gen}'
                NN_save_name = fr'gen{i}/NN{n_top}_af{int(fitness_gen[n_gen])}'

                # print(f'From: {NN_load_name}')
                # print(f'To:   {NN_save_name}')

                NN_load_dir = Path(self.EA_save_dir, NN_load_name)
                NN_save_dir = Path(self.EA_save_dir, NN_save_name)

                shutil.move(NN_load_dir, NN_save_dir)

                # print('Save Dir Contents:')
                # contents = list(NN_save_dir.iterdir())
                # for c in contents:
                #     print(c)

                # plot saved runs + output in parent directory
                for e in range(self.num_top_nn_plots):

                    ag_zarr = zarr.open(fr'{NN_save_dir}/ep{e}/ag.zarr', mode='r')
                    res_zarr = zarr.open(fr'{NN_save_dir}/ep{e}/res.zarr', mode='r')
                    plot_data = ag_zarr, res_zarr

                    plot_funcs.plot_map(plot_data, x_max=400, y_max=400, 
                                        save_name=f'{NN_save_dir}_ep{e}')

                # # pickle NN
                # NN = self.NNs[n_gen]
                # with open(fr'{NN_save_dir}/NN_pickle.bin','wb') as f:
                #     pickle.dump(NN, f)
            
            # update/pickle generational fitness data in parent directory
            self.fitness_evol.append(fitness_gen)
            with open(fr'{self.EA_save_dir}/fitness_spread_per_generation.bin', 'wb') as f:
                pickle.dump(self.fitness_evol, f)


            #### ---- Update optimizer + RNN instances ---- ####

            # Pass parameters + resulting fitness list to *minimizing* optimizer class
            self.es.tell(self.NN_param_vectors, fitness_gen)

            # Save param_vec distribution
            self.mean_param_vec[i,:] = self.es.mean
            self.std_param_vec[i,:] = self.es.stds
            
            # Generate new RNN parameters + instances for next generation
            self.NN_param_vectors = self.es.ask()
            # self.NNs = [Model(self.arch, self.activ, pv) for pv in self.NN_param_vectors]

            end_eval_time = time.time() - eval_time
            print(f'Performance Evaluation Time: {round( end_eval_time, 2)} sec')
            

        #### ---- Post-evolution tasks ---- ####

        end_overall_time = round(time.time() - self.overall_time, 2)
        print(f'Overall EA run time: {end_overall_time} sec')

        # save run data
        run_data = (
            self.mean_param_vec,
            self.std_param_vec,
            end_overall_time
        )
        with open(fr'{self.EA_save_dir}/run_data.bin', 'wb') as f:
            pickle.dump(run_data, f)

        # delete running folder
        shutil.rmtree(Path(self.EA_save_dir, 'running'))

        # plot violin plot for the EA trend
        plot_funcs.plot_EA_trend_violin(self.fitness_evol, self.EA_save_dir)
