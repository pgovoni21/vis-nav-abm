from abm.NN.model import WorldModel as Model
from abm import start_sim, start_sim_env, start_sim_iter
from abm.monitoring import plot_funcs

from pathlib import Path
import shutil, os, warnings, time
import numpy as np
from pgpelib import PGPE
import multiprocessing
import pickle

class EvolAlgo():
    
    def __init__(self, arch, activ, RNN_type,
                 generations, population_size, episodes, 
                 init_sigma, step_sigma, step_mu, momentum,
                 num_top_nn_saved, num_top_nn_plots, EA_save_name, start_seed,
                 est_method):
        
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
        self.generations = generations
        self.population_size = population_size
        self.episodes = episodes
        self.init_sigma = init_sigma
        self.start_seed = start_seed
        self.est_method = est_method

        # Initialize ES optimizer
        self.es = PGPE(
            solution_length = param_vec_size,
            popsize = population_size,

            stdev_init = init_sigma, # clipup paper suggests init_sigma = sqrt(radius^2 / n) ; where radius ~ 15*max_speed = 15*0.15 = 2.25 ; tf init_sigma = sqrt(2.25^2 / 300) = 0.13
            center_learning_rate = step_mu,
            stdev_learning_rate = step_sigma,
            stdev_max_change = step_sigma*2,
            solution_ranking=True,

            optimizer = 'clipup',
            optimizer_config = {
                'momentum' : momentum,
                'max_speed': step_mu*2, # clipup paper suggests pinning max_speed to twice stepsize
                },

            # --- for adaptive pop sizes --- 
            # popsize_max = population_size * 5,
            # num_interactions = generations * population_size * episodes * 1000 # final parameter = sim_time
        )
        
        # Generate initial RNN parameters
        self.NN_param_vectors = self.es.ask()

        # Saving parameters
        self.fitness_evol = np.zeros([generations, population_size, episodes])
        self.num_top_nn_saved = num_top_nn_saved
        self.num_top_nn_plots = num_top_nn_plots
        self.EA_save_name = EA_save_name
        self.root_dir = Path(__file__).parent.parent.parent
        self.EA_save_dir = Path(self.root_dir, 'abm/data/simulation_data', EA_save_name)
        
        self.mean_param_vec = np.zeros([generations, param_vec_size])
        self.std_param_vec = np.zeros([generations, param_vec_size])
        self.mean_param_vec[0,:] = self.es.center
        self.std_param_vec[0,:] = self.es.stdev

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
        # print(f'init time: {round( end_init_time, 2)} s')


    def fit_parallel(self):

        # init process pool executor/manager
        # pool = multiprocessing.Pool()
        pool = multiprocessing.Pool(initializer=start_sim_iter.start_env, initargs=(self.model_tuple,))

        for i in range(self.generations):

            #### ---- Run sim + Save in running/nn/ep folder ---- ####

            print(f'---------------- gen {i} ----------------')

            # determine PRNG seeds + reset for next generation
            # (circumventing multiprocessing bug where multiple children can have overlapping seeds)
            seeds_per_gen = range(self.start_seed, self.start_seed + self.episodes)
            self.start_seed += self.episodes

            # load inputs for each simulation instance into list (for starmap_async)
            sim_inputs_per_gen = []
            for pv in self.NN_param_vectors:
                for e in range(self.episodes):
                    sim_inputs_per_gen.append( (pv, seeds_per_gen[e]) )

            # issue all tasks to pool at once (non-blocking + ordered)
            sim_time = time.time()
            results = pool.starmap_async( start_sim_iter.start_iter, sim_inputs_per_gen)

            # convert results iterator to list
            results_list = results.get()

            end_sim_time = time.time() - sim_time
            print(f'sim time: {round( end_sim_time, 2)} s')

            # print('Sim Results:')
            # for result in results_list:
            #     print(result)


            #### ---- Find fitness averages across episodes ---- ####

            # # set non-sim timer
            # eval_time = time.time()

            # skip to start of each episode series/chunk
            for p, NN_index in enumerate(range(0, len(results_list), self.episodes)):
                # pull sim data for each episode
                for e, fitnesses in enumerate(results_list[NN_index : NN_index + self.episodes]):
                    self.fitness_evol[i,p,e] = int(fitnesses[0])

            # estimate episodal fitnesses by mean or median
            if self.est_method == 'mean':
                fitness_rank = np.mean(self.fitness_evol[i,:,:], axis=1)
            else:
                fitness_rank = np.median(self.fitness_evol[i,:,:], axis=1)

            # # list all averaged fitnesses
            # print(f'Fitnesses: {fitness_rank}')
            
            # Track top fitness per generation
            # top_fg = round(np.max(fitness_rank),1) # max : top
            top_fg = int(np.min(fitness_rank)) # min : top
            avg_fg = round(np.mean(fitness_rank),2)
            med_fg = round(np.median(fitness_rank),2)
            print(f'--- top: {top_fg} | avg: {avg_fg} | med: {med_fg} ---')

            # update/pickle generational fitness data in parent directory
            with open(fr'{self.EA_save_dir}/fitness_spread_per_generation.bin', 'wb') as f:
                pickle.dump(self.fitness_evol, f)


            #### ---- Save/plot performance + NN from top performers  ---- ####


            # cycle through the top X performers
            # top_indices = np.argsort(fitness_rank)[ : -1-self.num_top_nn_saved : -1] # max : top
            top_indices = np.argsort(fitness_rank)[ : self.num_top_nn_saved] # min : top

            # # top_fitnesses = [int(fitness_rank[n_gen]) for n_gen in top_indices]
            # top_fitnesses = [round(fitness_rank[n_gen],2) for n_gen in top_indices]
            # print(f'Saving performance for NNs with avg fitnesses: {top_fitnesses}')

            # save top sim params
            for n_top, n_gen in enumerate(top_indices):
                with open(fr'{self.EA_save_dir}/NN{n_top}_pickle.bin','wb') as f:
                    pickle.dump(self.NN_param_vectors[n_gen], f)
            
            # save center sim params
            with open(fr'{self.EA_save_dir}/gen{i}_NNcen_pickle.bin', 'wb') as f:
                pickle.dump(self.es.center.copy(), f)


            #### ---- Update optimizer + RNN instances ---- ####

            # Pass parameters + resulting fitness list to *maximizing* optimizer class
            fitness_rank = [-f for f in fitness_rank] # flips sign (only applicable if min : top)
            self.es.tell(fitness_rank)

            # Save param_vec distribution
            self.mean_param_vec[i,:] = self.es.center
            self.std_param_vec[i,:] = self.es.stdev
            
            # Generate new RNN parameters for next generation
            self.NN_param_vectors = self.es.ask()

            # end_eval_time = time.time() - eval_time
            # print(f'eval time: {round( end_eval_time, 2)} s')
            

        #### ---- Post-evolution tasks ---- ####

        pool.close()
        pool.join()
            
        end_overall_time = round(time.time() - self.overall_time, 2)
        print(f'overall time: {end_overall_time} s')

        # save run data
        run_data = (
            self.mean_param_vec,
            self.std_param_vec,
            end_overall_time
        )
        with open(fr'{self.EA_save_dir}/run_data.bin', 'wb') as f:
            pickle.dump(run_data, f)

        # plot violin plot for the EA trend
        plot_funcs.plot_EA_trend_violin(self.fitness_evol, self.est_method, self.EA_save_dir)
