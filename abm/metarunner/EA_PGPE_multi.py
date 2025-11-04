from abm.NN.model import WorldModel as Model
from abm import start_sim_multi
# from abm.monitoring import plot_funcs

from pathlib import Path
import shutil, os, warnings, time
import numpy as np
from pgpelib import PGPE
import multiprocessing
import pickle

class EvolAlgo():
    
    def __init__(self, arch, activ, RNN_type, N,
                 generations, population_size, episodes, 
                 init_sigma, step_sigma, step_mu, momentum,
                 EA_save_name, start_seed, est_method, sim_type):

        # init_time = time.time()
        self.overall_time = time.time()

        # Pack model parameters 
        self.model_tuple = (arch, activ, RNN_type)
        self.N = N # number of NN models to evolve

        # Calculate parameter vector size using an example NN (easy generalizable)
        param_vec_size = sum(p.numel() for p in Model(arch,activ,RNN_type).parameters())

        print(f'EA Save Name: {EA_save_name}')
        print(f'Model Architecture: {arch}, {RNN_type}')
        print(f'Total #Params: {param_vec_size}')

        # Evolution + Simulation parameters
        self.generations = generations
        self.population_size = population_size
        self.episodes = episodes
        self.init_sigma = init_sigma
        self.start_seed = start_seed
        self.est_method = est_method
        self.sim_type = sim_type

        # Initialize ES optimizers
        self.es = []
        for n in range(N):
            es = PGPE(
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
            )
            self.es.append(es)

        # Saving parameters
        self.fitness_evol = np.zeros([generations, population_size, episodes, N])
        self.EA_save_name = EA_save_name
        self.root_dir = Path(__file__).parent.parent.parent
        self.EA_save_dir = Path(self.root_dir, 'abm/data/simulation_data', EA_save_name)

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

            # gather model params from ES optimizers
            self.NN_param_vectors = []
            for n in range(self.N):
                self.NN_param_vectors.append(self.es[n].ask())

            # determine PRNG seeds + reset for next generation
            # (circumventing multiprocessing bug where multiple children can have overlapping seeds)
            seeds_per_gen = range(self.start_seed, self.start_seed + self.episodes)
            self.start_seed += self.episodes

            # load inputs for each simulation instance into list (for starmap_async)
            sim_inputs_per_gen = []
            for p in range(self.population_size):
                pv_list = [self.NN_param_vectors[n][p] for n in range(self.N)]
                for e in range(self.episodes):
                    sim_inputs_per_gen.append( (self.model_tuple, pv_list, self.EA_save_dir, seeds_per_gen[e]) ) # env_path=None

            # issue all tasks to pool at once (non-blocking + ordered)
            results = pool.starmap_async( start_sim_multi.start, sim_inputs_per_gen )
            results_list = results.get()

            # print('Sim Results:')
            # for result in results_list:
            #     print(result)

            #### ---- Find fitness averages across episodes ---- ####

            # pull sim data, skipping to start of each episode series/chunk
            for p, NN_index in enumerate(range(0, len(results_list), self.episodes)):
                for e, (time_taken, dist_from_patch, data) in enumerate(results_list[NN_index : NN_index + self.episodes]):
                    for n in range(self.N):

                        if dist_from_patch[n] == 0:
                            self.fitness_evol[i,p,e,n] = int(time_taken[n])
                        else:
                            self.fitness_evol[i,p,e,n] = int(time_taken[n] + dist_from_patch[n])

            tops = []
            avgs = []
            for n in range(self.N):

                # estimate episodal fitnesses by mean or median
                if self.est_method == 'mean':
                    fitness_rank = np.mean(self.fitness_evol[i,:,:,n], axis=1)
                else:
                    fitness_rank = np.median(self.fitness_evol[i,:,:,n], axis=1)
                # print(f'Fitnesses: {fitness_rank}')

                # Pass parameters + resulting fitness list to *maximizing* optimizer class
                if self.sim_type.startswith('walls'):
                    fitness_rank = [-f for f in fitness_rank] # flips sign (only applicable if min : top)
                elif self.sim_type.startswith('nowalls'):
                    pass # no sign flip needed
                self.es[n].tell(fitness_rank)
                tops.append(int(np.max(fitness_rank)))
                avgs.append(int(np.mean(fitness_rank)))

                # save center sim params
                with open(fr'{self.EA_save_dir}/gen{i}_NNcen_pickle_ag{n}.bin', 'wb') as f:
                    pickle.dump(self.es[n].center.copy(), f)

            # update/pickle generational fitness data in parent directory
            with open(fr'{self.EA_save_dir}/fitness_spread_per_generation.bin', 'wb') as f:
                pickle.dump(self.fitness_evol, f)

            # print run info
            gen_time = round(time.time() - gen_time,2)
            print(f'--- gen {i} | t: {gen_time}s | tops: {tops} | avgs: {avgs} ---')

        #### ---- Post-evolution tasks ---- ####

        pool.close()
        pool.join()

        end_overall_time = round(time.time() - self.overall_time, 2)
        print(f'overall time: {end_overall_time} s')
