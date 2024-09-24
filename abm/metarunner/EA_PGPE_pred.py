from abm.NN.model_pred import Model
from abm.start_sim_pred import start
# from abm.monitoring import plot_funcs

from pathlib import Path
import shutil, os, warnings, time
import numpy as np
from pgpelib import PGPE
import multiprocessing
import pickle

class EvolAlgo():

    def __init__(self, arch, activ, sharpness,
                 generations, population_size, episodes, 
                 init_sigma, step_sigma, step_mu, momentum,
                 EA_save_name, start_seed, est_method,
                 mode, exp=None, gen=None):

        # init_time = time.time()
        self.overall_time = time.time()

        self.model_tuple = (arch, activ, sharpness)
        self.generations = generations
        self.population_size = population_size
        self.episodes = episodes
        self.init_sigma = init_sigma
        self.start_seed = start_seed
        self.est_method = est_method
        self.mode = mode

        if self.mode == 'train_pred':
            self.T = 100
            param_vec_size = sum(p.numel() for p in Model(arch,activ).parameters())
        elif self.mode == 'train_act':
            self.T = 1000
            with open(fr'abm/data/simulation_data/{exp}/gen{gen}_NN0_pickle.bin', 'rb') as f:
                self.pv = pickle.load(f)
            # param_vec_size = sum(p.numel() for p in Model(arch,activ).parameters())
            # print(param_vec_size, self.pv.shape)
            m = Model(arch,activ, param_vector=self.pv)
            param_vec_size = sum(p.numel() for p in m.h2o_act.parameters())

        print(f'EA Save Name: {EA_save_name}')
        print(f'Model Architecture: {arch}, {activ}, {sharpness}')
        print(f'Total #Params: {param_vec_size}')
        print(f'# vCPUs: {os.cpu_count()}')

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
        )

        # Saving parameters
        self.fitness_evol = np.zeros([generations, population_size, episodes])
        self.EA_save_name = EA_save_name
        self.root_dir = Path(__file__).parent.parent.parent
        if self.mode == 'train_pred':
            self.EA_save_dir = Path(self.root_dir, 'abm/data/simulation_data', EA_save_name)
        elif self.mode == 'train_act':
            self.EA_save_dir = Path(self.root_dir, 'abm/data/simulation_data', EA_save_name, 'train_act_from_gen'+str(gen))
        
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

        last_top = -500 # min : top
        for i in range(self.generations):

            #### ---- Run sim + Save in running/nn/ep folder ---- ####

            gen_time = time.time()

            # gather model params from ES optimizer
            self.NN_param_vectors = self.es.ask()

            # determine PRNG seeds + reset for next generation
            # (circumventing multiprocessing bug where multiple children can have overlapping seeds)
            seeds_per_gen = range(self.start_seed, self.start_seed + self.episodes)
            self.start_seed += self.episodes

            # load inputs for each simulation instance into list (for starmap_async)
            sim_inputs_per_gen = []
            if self.mode == 'train_pred':
                for pv in self.NN_param_vectors:
                    for e in range(self.episodes):
                        sim_inputs_per_gen.append( (self.model_tuple, pv, self.EA_save_dir, seeds_per_gen[e], self.mode, None, self.T) )
            elif self.mode == 'train_act':
                for pv_h2o_act in self.NN_param_vectors:
                    for e in range(self.episodes):
                        sim_inputs_per_gen.append( (self.model_tuple, self.pv, self.EA_save_dir, seeds_per_gen[e], self.mode, pv_h2o_act, self.T) )

            # issue all tasks to pool at once (non-blocking + ordered)
            results = pool.starmap_async( start, sim_inputs_per_gen )
            results_list = results.get()

            # print('Sim Results:')
            # for result in results_list:
            #     print(result)

            #### ---- Find fitness averages across episodes ---- ####

            # pull sim data, skipping to start of each episode series/chunk
            for p, NN_index in enumerate(range(0, len(results_list), self.episodes)):
                for e, (o_pred, o_actual, time_taken, dist_from_patch) in enumerate(results_list[NN_index : NN_index + self.episodes]):

                    if self.mode == 'train_pred':
                        # L2 pred norm
                        # print(f'Pre: {o_pre}')
                        # print(f'Post: {o_next}')
                        # self.fitness_evol[i,p,e] = ((o_pred - o_actual)**2).mean(axis=None)
                        self.fitness_evol[i,p,e] = ((o_pred - o_actual)**2).sum(axis=None)

                    elif self.mode == 'train_act':
                        if dist_from_patch == 0:
                            self.fitness_evol[i,p,e] = int(time_taken)
                        else:
                            self.fitness_evol[i,p,e] = int(time_taken + dist_from_patch)


            # estimate episodal fitnesses by mean or median
            if self.est_method == 'mean':
                fitness_rank = np.mean(self.fitness_evol[i,:,:], axis=1)
            else:
                fitness_rank = np.median(self.fitness_evol[i,:,:], axis=1)
            # print(f'Fitnesses: {fitness_rank}')

            # Pass parameters + resulting fitness list to *maximizing* optimizer class
            fitness_rank = [-f for f in fitness_rank] # flips sign (only applicable if min : top)
            self.es.tell(fitness_rank)

            # update/pickle generational fitness data in parent directory
            with open(fr'{self.EA_save_dir}/fitness_spread_per_generation.bin', 'wb') as f:
                pickle.dump(self.fitness_evol, f)

            #### ---- Save current ES/model params + print info ---- ####

            # top_fg = int(np.min(fitness_rank)) # max : top
            top_fg = round(np.max(fitness_rank),2) # min : top
            avg_fg = round(np.mean(fitness_rank),2)
            med_fg = round(np.median(fitness_rank),2)

            # cycle through the top X performers
            # top_indices = np.argsort(fitness_rank)[ : -1-self.num_top_nn_saved : -1] # max : top
            # top_indices = np.argsort(fitness_rank)[ : self.num_top_nn_saved] # min : top
            # top_indices = np.argsort(fitness_rank)[-1] # max : top
            top_indices = np.argsort(fitness_rank)[:1] # min : top

            if top_fg > last_top: # min : top
                # save top sim params
                for n_top, n_gen in enumerate(top_indices):
                    with open(fr'{self.EA_save_dir}/gen{i}_NN{n_top}_pickle.bin','wb') as f:
                        pickle.dump(self.NN_param_vectors[n_gen], f)

                # # save center sim params
                # with open(fr'{self.EA_save_dir}/gen{i}_NNcen_pickle.bin', 'wb') as f:
                #     pickle.dump(self.es.center.copy(), f)

                last_top = top_fg

            # Save param_vec distribution
            self.mean_param_vec[i,:] = self.es.center
            self.std_param_vec[i,:] = self.es.stdev

            # print run info
            gen_time = round(time.time() - gen_time,2)
            print(f'--- gen {i} | t: {gen_time}s | top: {top_fg} | avg: {avg_fg} | med: {med_fg} ---')


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

        # # plot violin plot for the EA trend
        # plot_funcs.plot_EA_trend_violin(self.fitness_evol, self.est_method, self.EA_save_dir)
