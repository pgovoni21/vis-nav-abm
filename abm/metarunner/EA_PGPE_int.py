from abm.NN.model import WorldModel as Model
from abm import start_sim
# from abm.monitoring import plot_funcs

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
                 EA_save_name, start_seed, est_method):
        
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

        self.population_size_max = population_size * 10
        self.num_interactions = int(population_size * episodes / 10)

        # Initialize ES optimizer
        self.es = PGPE(
            solution_length = param_vec_size,
            popsize = population_size,

            num_interactions = self.num_interactions,
            popsize_max = self.population_size_max,

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
        )

        # Saving parameters
        self.fitness_evol = np.zeros([generations, self.population_size_max, episodes])
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


    def fit_parallel(self):

        # init process pool executor/manager
        pool = multiprocessing.Pool()

        for i in range(self.generations):

            #### ---- Run sim + Save in running/nn/ep folder ---- ####

            gen_time = time.time()

            # increments each time ES is re-ran (until reaching num_interactions or popsize_max)
            pop_clicker = 0
            gen_patches_found = 0

            while True:
                current_popsize = pop_clicker + self.population_size
                # print(f'Current popsize: {current_popsize} / {self.population_size_max}')

                # gather model params from ES optimizer
                self.NN_param_vectors = self.es.ask()

                # determine PRNG seeds + reset for next generation
                # (circumventing multiprocessing bug where multiple children can have overlapping seeds)
                seeds_per_gen = range(self.start_seed, self.start_seed + self.episodes)
                self.start_seed += self.episodes

                # load inputs for each simulation instance into list (for starmap_async)
                sim_inputs_per_gen = []
                for pv in self.NN_param_vectors:
                    for e in range(self.episodes):
                        sim_inputs_per_gen.append( (self.model_tuple, pv, self.EA_save_dir, seeds_per_gen[e]) )

                # issue all tasks to pool at once (non-blocking + ordered)
                results = pool.starmap_async( start_sim.start, sim_inputs_per_gen )
                results_list = results.get()

                # print('Sim Results:')
                # for result in results_list:
                #     print(result)

                #### ---- Find fitness averages across episodes ---- ####

                # pull sim data, skipping to start of each episode series/chunk
                current_patches_found = np.zeros(self.population_size)
                for p, NN_index in enumerate(range(0, len(results_list), self.episodes)):
                    for e, (time_taken, dist_from_patch) in enumerate(results_list[NN_index : NN_index + self.episodes]):
                        self.fitness_evol[i, p + pop_clicker, e] = int(time_taken + dist_from_patch)
                        if dist_from_patch == 0:
                            current_patches_found[p] += 1
                # print(f'All fitnesses so far: {self.fitness_evol[i,:,:]}')

                current_batch = self.fitness_evol[i, pop_clicker:current_popsize, :]
                # print(f'Current batch: {current_batch}')
                # print(f'Current patches found: {current_patches_found}')

                # estimate episodal fitnesses by mean or median
                if self.est_method == 'mean':
                    fitness_rank = np.mean(current_batch, axis=1)
                else:
                    fitness_rank = np.median(current_batch, axis=1)
                # print(f'Est fitnesses: {fitness_rank}')

                # # pass mask over sims for number of patches found
                # current_patches_found = ((0 < current_batch) & (current_batch < 1000))*1
                # current_patches_found = np.sum(current_patches_found, axis=1)
                # # print(f'Current patches found: {current_patches_found}')

                gen_patches_found += int(current_patches_found.sum())
                # print(f'Gen patches found: {gen_patches_found} / {self.num_interactions}')

                #### ---- Update optimizer ---- ####

                # pass parameters + resulting fitnesses to *maximizing* optimizer class
                # increase popsize if num_patches found < self.num_interactions, break if threshold met

                fitness_rank = [-f for f in fitness_rank] # flips sign (only applicable if min : top)
                iter_done = self.es.tell(fitness_rank, current_patches_found)

                pop_clicker += self.population_size

                if iter_done:
                    break

            # print run info
            gen_time = round(time.time() - gen_time,2)
            pop = f'{current_popsize}/{self.population_size_max}'
            patches = f'{gen_patches_found}/{self.num_interactions}'

            gen_batch = self.fitness_evol[i,:current_popsize,:]
            if self.est_method == 'mean':
                gen_fitness_rank = np.mean(gen_batch, axis=1)
            else:
                gen_fitness_rank = np.median(gen_batch, axis=1)

            # top_fg = int(np.max(fitness_rank)) # max : top
            top_fg = int(np.min(gen_fitness_rank)) # min : top
            avg_fg = int(np.mean(gen_fitness_rank))
            med_fg = int(np.median(gen_fitness_rank))
            print(f'--- gen {i} | t: {gen_time}s | pop: {pop} | patches: {patches} || top: {top_fg} | avg: {avg_fg} | med: {med_fg} ---')

            # update/pickle generational fitness data in parent directory
            with open(fr'{self.EA_save_dir}/fitness_spread_per_generation.bin', 'wb') as f:
                pickle.dump(self.fitness_evol, f)


            #### ---- Save current ES/model params  ---- ####
            
            # save center sim params
            with open(fr'{self.EA_save_dir}/gen{i}_NNcen_pickle.bin', 'wb') as f:
                pickle.dump(self.es.center.copy(), f)

            # Save param_vec distribution
            self.mean_param_vec[i,:] = self.es.center
            self.std_param_vec[i,:] = self.es.stdev
            

        #### ---- Post-evolution tasks ---- ####

        pool.close()
        pool.join()

        end_overall_time = round(time.time() - self.overall_time, 2)
        print(f'overall t: {end_overall_time} s')

        # save run data
        run_data = (
            self.mean_param_vec,
            self.std_param_vec,
            end_overall_time
        )
        with open(fr'{self.EA_save_dir}/run_data.bin', 'wb') as f:
            pickle.dump(run_data, f)

        # # plot violin plot for the EA trend
        # plot_funcs.plot_EA_trend_violin(self.fitness_evol, self.est_method, self.EA_save_dir)
