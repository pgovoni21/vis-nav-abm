from abm.NN.model import WorldModel as Model
# from abm.NN.model_simp import WorldModel as Model
from abm import start_sim
from abm.monitoring import plot_funcs

from pathlib import Path
import shutil, os, warnings, time
import numpy as np
import cma
import multiprocessing
import pickle
# import zarr

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

        print(f'EA Save Name: {EA_save_name}')
        print(f'Model Architecture: {arch}, {RNN_type}')
        print(f'Total #Params: {param_vec_size}')
        print(f'# vCPUs: {os.cpu_count()}')

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

        # Saving parameters
        self.fitness_evol = []
        self.num_top_nn_saved = num_top_nn_saved
        self.num_top_nn_plots = num_top_nn_plots
        self.EA_save_name = EA_save_name
        self.root_dir = Path(__file__).parent.parent.parent
        self.EA_save_dir = Path(self.root_dir, 'abm/data/simulation_data', EA_save_name)
        
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

            # load inputs for each simulation instance into list (for starmap_async)
            sim_inputs_per_gen = []
            for n, pv in enumerate(self.NN_param_vectors):
                for e in range(self.episodes):
                    save_ext = fr'{self.EA_save_name}/running/NN{n}/ep{e}'
                    sim_inputs_per_gen.append( (self.model_tuple, pv, save_ext, seeds_per_gen[e]) )

                # save NN param_vec in the parent folder
                NN_dir = Path(self.root_dir,fr'abm/data/simulation_data/{self.EA_save_name}/running')
                Path(NN_dir).mkdir(parents=True, exist_ok=True)
                with open(fr'{NN_dir}/NN{n}_pickle.bin','wb') as f:
                    pickle.dump(pv, f)

            sim_time = time.time()

            # using process pool executor/manager
            with multiprocessing.Pool() as pool:

                # issue all tasks to pool at once (non-blocking + ordered)
                results = pool.starmap_async( start_sim.start, sim_inputs_per_gen)

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
            for NN_index in range(0, len(results_list), self.episodes):

                # pull sim data for each episode
                fitness_ep = []
                for _, fitnesses, _ in results_list[NN_index : NN_index + self.episodes]:
                    fitness_ep.append(round(fitnesses[0],0))

                avg_fitness = np.mean(fitness_ep)
                fitness_gen.append(avg_fitness)

            # # list all averaged fitnesses
            # print(f'Fitnesses: {fitness_gen}')
            
            # Track top fitness per generation
            top_fg = round(np.max(fitness_gen),1) # max : top
            # top_fg = int(np.min(fitness_gen)) # min : top
            avg_fg = round(np.mean(fitness_gen),2)
            print(f'Highest Across Gen: {top_fg} | Avg Across Gen: {avg_fg} ---')


            #### ---- Save/plot performance + NN from top performers  ---- ####


            # cycle through the top X performers
            top_indices = np.argsort(fitness_gen)[ : -1-self.num_top_nn_saved : -1] # max : top
            # top_indices = np.argsort(fitness_gen)[ : self.num_top_nn_saved] # min : top

            # # top_fitnesses = [int(fitness_gen[n_gen]) for n_gen in top_indices]
            # top_fitnesses = [round(fitness_gen[n_gen],2) for n_gen in top_indices]
            # print(f'Saving performance for NNs with avg fitnesses: {top_fitnesses}')

            for n_top, n_gen in enumerate(top_indices):

                # pull saved sim runs from 'running' directory + archive in parent directory
                # ('running' directory is rewritten each generation)
                NN_load_dir = Path(self.EA_save_dir, fr'running/NN{n_gen}_pickle.bin')
                NN_save_dir = Path(self.EA_save_dir, fr'gen{i}_NN{n_top}_pickle.bin')
                shutil.move(NN_load_dir, NN_save_dir)

                # # plot saved runs + output in parent directory
                # for e in range(self.num_top_nn_plots):

                #     ag_zarr = zarr.open(fr'{NN_save_dir}/ep{e}/ag.zarr', mode='r')
                #     res_zarr = zarr.open(fr'{NN_save_dir}/ep{e}/res.zarr', mode='r')
                #     plot_data = ag_zarr, res_zarr

                #     plot_funcs.plot_map(plot_data, x_max=500, y_max=500, 
                #                         save_name=f'{NN_save_dir}_ep{e}')
            
            # update/pickle generational fitness data in parent directory
            self.fitness_evol.append(fitness_gen)
            with open(fr'{self.EA_save_dir}/fitness_spread_per_generation.bin', 'wb') as f:
                pickle.dump(self.fitness_evol, f)


            #### ---- Update optimizer + RNN instances ---- ####

            # Pass parameters + resulting fitness list to *minimizing* optimizer class
            fitness_gen = [-f for f in fitness_gen] # flips sign (only applicable if max : top)
            self.es.tell(self.NN_param_vectors, fitness_gen)

            # Save param_vec distribution
            self.mean_param_vec[i,:] = self.es.mean
            self.std_param_vec[i,:] = self.es.stds
            
            # Generate new RNN parameters for next generation
            self.NN_param_vectors = self.es.ask()

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
