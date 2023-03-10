from abm.NN.RNNs import RNN
import abm.app as sim
from abm.app import start as start_sim

import numpy as np
import pickle
import random
from pathlib import Path
import shutil, os, warnings
import zarr
from abm.monitoring import plot_funcs

class EvolAlgo():
    
    def __init__(self, arch=(50,128,2), RNN_type='static-Yang', rule='hebb', activ='relu', dt=100, init=None, 
                 population_size=96, generations=500, episodes=5, mutation_variance=0.02, repop_method='ES', 
                 hybrid_scaling_factor=0.001, hybrid_new_intro_num=5, num_top_saved=5, EA_save_name=None):
        
        # Initialize NN population + fitness lists
        self.arch = arch
        self.RNN_type = RNN_type
        self.rule = rule
        self.activ = activ
        self.dt = dt
        self.init = init
        self.networks = [RNN(arch, RNN_type, rule, activ, dt, init) for _ in range(population_size)]
        self.fitness_evol = []

        # Evolution + Simulation parameters
        self.population_size = population_size
        self.generations = generations
        self.episodes = episodes
        self.mutation_variance = mutation_variance
        self.repop_method = repop_method
        self.hybrid_scaling_factor = hybrid_scaling_factor
        self.hybrid_new_intro_num = hybrid_new_intro_num

        # Saving parameters
        self.num_top_saved = num_top_saved
        self.EA_save_name = EA_save_name
        self.root_dir = Path(__file__).parent.parent.parent
        self.EA_save_dir = Path(self.root_dir, 'abm\data\simulation_data', EA_save_name)
        
        # Create save directory + copy .env file over
        if os.path.isdir(self.EA_save_dir):
            warnings.warn("Temporary directory for env files is not empty and will be overwritten")
            shutil.rmtree(self.EA_save_dir)
        Path(self.EA_save_dir).mkdir()
        shutil.copy(
            Path(self.root_dir, '.env'), 
            Path(self.EA_save_dir, '.env')
            )
        
    def fit(self):

        for i in range(self.generations):

            # Simulate performance of each NN + store results as array
            fitness_gen = []
            for n, NN in enumerate(self.networks):

                fitness_ep = []
                for x in range(self.episodes):

                    # construct save name for current simulation, to be called later if needed (e.g. to plot top performers)
                    save_ext = fr'{self.EA_save_name}\running\NN{n}\ep{x}'

                    # run sim + record fitness/time
                    fitness, elapsed_time, crash = sim.start(NN=NN, save_ext=save_ext)
                    fitness_ep.append(fitness)
                    print(f'Episode Fitness: {fitness} \t| Elapsed Time: {elapsed_time}')

                    if crash: # save crashed NN in binary mode + continue
                        print('Crashed agent - pickled NN')
                        with open("crashed_NN.bin", "wb") as f:
                            pickle.dump(NN, f)
                
                avg_fitness = np.mean(fitness_ep)
                fitness_gen.append(avg_fitness)
                print(f'--- NN {n+1} of {self.population_size} \t| Avg Across Episodes: {avg_fitness} ---')

            # Track top fitness per generation
            max_fitness_gen = int(np.max(fitness_gen))
            avg_fitness_gen = int(np.mean(fitness_gen))
            print(f'---+--- Generation: {i+1} | Highest Across Gen: {max_fitness_gen} | Avg Across Gen: {avg_fitness_gen} ---+---')


            # cycle through the top X performers
            top_indices = np.argsort(fitness_gen)[ : -1-self.num_top_saved : -1] # best in generation : first (n_top = 1)
            for n_top, n_gen in enumerate(top_indices):

                # pull saved sim runs from 'running' directory + archive in parent directory
                # ('running' directory is rewritten each generation)
                NN_load_name = fr'running\NN{n_gen}'
                NN_save_name = fr'gen{i}\NN{n_top}_fitness{int(fitness_gen[n_gen])}'

                NN_load_dir = Path(self.EA_save_dir, NN_load_name)
                NN_save_dir = Path(self.EA_save_dir, NN_save_name)

                shutil.move(NN_load_dir, NN_save_dir)

                # plot saved runs + output in parent directory
                for x in range(self.episodes):

                    ag_zarr = zarr.open(fr'{NN_save_dir}\ep{x}\ag.zarr', mode='r')
                    res_zarr = zarr.open(fr'{NN_save_dir}\ep{x}\res.zarr', mode='r')
                    plot_data = ag_zarr, res_zarr

                    plot_funcs.plot_map(plot_data, x_max=400, y_max=400, save_dir=NN_save_dir, save_name=f'ep{x}')

                # pickle NN
                NN = self.networks[n_gen]
                with open(rf'{NN_save_dir}\NN_pickle.bin','wb') as f:
                    pickle.dump(NN, f)
            
            # update/pickle generational fitness data in parent directory
            self.fitness_evol.append(fitness_gen)

            with open(rf'{self.EA_save_dir}\fitness_spread_per_generation.bin', 'wb') as f:
                pickle.dump(self.fitness_evol, f)

            # Select/Mutate to generate next generation NNs according to method specified
            if self.repop_method == 'ES':
                best_network = self.networks[np.argmax(fitness_gen)]
                self.repop_ES(best_network)
            elif self.repop_method == 'ROULETTE':
                self.repop_roulette(fitness_gen)
            elif self.repop_method == 'HYBRID':
                self.repop_hybrid(fitness_gen, top_indices, self.hybrid_scaling_factor, self.hybrid_new_intro_num)
            else: 
                return f'Invalid repopulation method specified: {self.repop_method}'
            
    def fit_parallel(self):

        import multiprocessing
        # from collections import defaultdict
        import time

        for i in range(self.generations):

            sim_inputs_per_gen = []
            for n, NN in enumerate(self.networks):
                for e in range(self.episodes):
                    # construct temporary save extension for sim data files
                    save_ext = fr'{self.EA_save_name}\running\NN{n}\ep{e}'
                    # pack inputs for current generation sims as tuple
                    sim_inputs_per_gen.append( (NN, save_ext) )

            start = time.time()

            # # non parallel (comment out)
            # results = map(start_sim, sim_inputs_per_gen)

            # # without executor (comment out)
            # processes = []
            # for inputs in sim_inputs_per_gen:
            #     proc = multiprocessing.Process(target=start_sim, kwargs={'inputs': inputs})
            #     proc.start()
            #     processes.append(proc)
            # for proc in processes:    
            #     proc.join()

            # using process pool executor/manager
            with multiprocessing.Pool() as pool:

                # # individually send tasks to pool
                # results = []
                # for inputs in sim_inputs_per_gen:
                #     print(inputs)
                #     result = pool.apply_async(start_sim, args=(inputs,))
                #     results.append(result)

                # iterate through tasks, send individually to pool (non-blocking + non-ordered)
                # results = pool.imap_unordered( start_sim, sim_inputs_per_gen)

                # issue all tasks at once (non-blocking + ordered)
                results = pool.starmap_async( start_sim, sim_inputs_per_gen)

                pool.close()
                pool.join()

            print('run_time: ', time.time() - start)

            # convert results iterator to list
            results_list = results.get()

            # skip to start of each episodes series/chunk
            fitness_gen = []
            for n, NN_index in enumerate(range(0, len(results_list), self.episodes)):

                # pull sim data for each episode
                fitness_ep = []
                simtime_ep = []
                for save_ext, fitnesses, elapsed_time, crash in results_list[NN_index : NN_index + self.episodes]:

                    fitness_ep.append(fitnesses[0])
                    simtime_ep.append(elapsed_time)

                avg_fitness = np.mean(fitness_ep)
                fitness_gen.append(avg_fitness)
                print(f'--- NN {n+1} of {self.population_size} \t| Avg Across Episodes: {avg_fitness} ---')

            print(fitness_gen)
            
            # Track top fitness per generation
            max_fitness_gen = int(np.max(fitness_gen))
            avg_fitness_gen = int(np.mean(fitness_gen))
            print(f'---+--- Generation: {i+1} | Highest Across Gen: {max_fitness_gen} | Avg Across Gen: {avg_fitness_gen} ---+---')


            # # map NN to list of fitnesses per episode (comment out - use for imap_unordered)
            # d = defaultdict(list)
            # for num, NN, save_ext, fitnesses, elapsed_time, crash_bool in results:
            #     d[num].append(fitnesses[0])
            # NN_to_avg_fitness = []
            # for NN, fitness_list in d.items():
            #     avg_fitness = np.mean(fitness_list)
            #     NN_to_avg_fitness.append( (NN, avg_fitness) )


            # cycle through the top X performers
            top_indices = np.argsort(fitness_gen)[ : -1-self.num_top_saved : -1] # best in gen : first (n_top = 1)
            for n_top, n_gen in enumerate(top_indices):

                # pull saved sim runs from 'running' directory + archive in parent directory
                # ('running' directory is rewritten each generation)
                NN_load_name = fr'running\NN{n_gen}'
                NN_save_name = fr'gen{i}\NN{n_top}_fitness{int(fitness_gen[n_gen])}'

                NN_load_dir = Path(self.EA_save_dir, NN_load_name)
                NN_save_dir = Path(self.EA_save_dir, NN_save_name)

                shutil.move(NN_load_dir, NN_save_dir)

                # plot saved runs + output in parent directory
                for x in range(self.episodes):

                    ag_zarr = zarr.open(fr'{NN_save_dir}\ep{x}\ag.zarr', mode='r')
                    res_zarr = zarr.open(fr'{NN_save_dir}\ep{x}\res.zarr', mode='r')
                    plot_data = ag_zarr, res_zarr

                    plot_funcs.plot_map(plot_data, x_max=400, y_max=400, save_dir=NN_save_dir, save_name=f'ep{x}')

                # pickle NN
                NN = self.networks[n_gen]
                with open(rf'{NN_save_dir}\NN_pickle.bin','wb') as f:
                    pickle.dump(NN, f)
            
            # update/pickle generational fitness data in parent directory
            self.fitness_evol.append(fitness_gen)

            with open(rf'{self.EA_save_dir}\fitness_spread_per_generation.bin', 'wb') as f:
                pickle.dump(self.fitness_evol, f)

            # Select/Mutate to generate next generation NNs according to method specified
            if self.repop_method == 'ES':
                best_network = self.networks[np.argmax(fitness_gen)]
                self.repop_ES(best_network)
            elif self.repop_method == 'ROULETTE':
                self.repop_roulette(fitness_gen)
            elif self.repop_method == 'HYBRID':
                self.repop_hybrid(fitness_gen, top_indices, self.hybrid_scaling_factor, self.hybrid_new_intro_num)
            else: 
                return f'Invalid repopulation method specified: {self.repop_method}'


    def repop_ES(self, best_network):

        # Fully elitist method - risk of over-convergence/exploitation via local optima traps

        # Create child NNs as mutations of the top NN
        new_networks = []
        for _ in range(self.population_size - 1):
            new_network = RNN(self.arch, self.RNN_type, self.rule, self.activ, self.dt, 
                              copy_network=best_network, var=self.mutation_variance)
            new_networks.append(new_network)
        
        # Set NNs for next generation
        self.networks = [best_network] + new_networks


    def repop_roulette(self, fitness_gen):

        # Random choice proportionate to fitness - risk of over-divergence/exploration via gradient loss

        # Calculate ratio of fitness:total fitness for each colony, add to previous, append
        sum_fitness_gen = np.sum(fitness_gen)
        breeding_probs = [0]
        for i in range(self.population_size):
            breeding_prob = fitness_gen[i] / sum_fitness_gen + breeding_probs[i]
            breeding_probs.append(breeding_prob)

        # Pick parents randomly according to breeding probability line
        parent_fitnesses = []
        parents = []
        for _ in range(self.population_size):
            
            rand_num = random.random()
            for i in range(self.population_size + 1):
                if rand_num < breeding_probs[i]:
                    parent_fitnesses.append(fitness_gen[i-1])
                    parents.append(self.networks[i-1])
                    break

        # Set child NNs as mutations of parents
        self.networks = [RNN(self.arch, self.RNN_type, self.rule, self.activ, self.dt, 
                              copy_network=parent, var=self.mutation_variance) for parent in parents]


    def repop_hybrid(self, fitness_gen, top_indices, scaling_factor, new_intro_num):

        # Combines Elitist convergence + Roulette divergence + adding new networks each generation

        ## elitism : best X NNs passed on
        top_networks = []
        for i in top_indices:
            top_networks.append(self.networks[i])

        ## scaled fitness proportionate : biased roulette wheel selection to favor top performers
        
            # Calculate ratio of fitness:total fitness for each colony, add to previous, append
        fitness_gen_array = np.array(fitness_gen)
        scaled_fitness_gen = fitness_gen_array * np.exp (scaling_factor * fitness_gen_array)

        print(f'Nonscaled fitnesses: {np.sort(fitness_gen_array)[::-1]}')
        print(f'Scaled fitnesses: {np.sort(scaled_fitness_gen)[::-1]}')

        sum_scaled_fitness_gen = np.sum(scaled_fitness_gen)
        breeding_probs = [0]
        for i in range(self.population_size):
            breeding_prob = scaled_fitness_gen[i] / sum_scaled_fitness_gen + breeding_probs[i]
            breeding_probs.append(breeding_prob)

            # Pick parents randomly according to breeding probability line
        parent_fitnesses = []
        parents = []
        for _ in range(self.population_size - len(top_indices) - new_intro_num):
            
            rand_num = random.random()
            for i in range(self.population_size + 1):
                if rand_num < breeding_probs[i]:
                    parent_fitnesses.append(fitness_gen[i-1])
                    parents.append(self.networks[i-1])
                    break
        
            # Set child NNs as mutations of parents
        roulette_networks = [RNN(self.arch, self.RNN_type, self.rule, self.activ, self.dt, 
                              copy_network=parent, var=self.mutation_variance) for parent in parents]

        ## new network introduction : random NNs added to enhance exploration
        new_networks = [RNN(self.arch, self.RNN_type, self.rule, self.activ, self.dt, self.init) for _ in range(new_intro_num)]

        ## compile list of NNs for the next generation
        self.networks = top_networks + roulette_networks + new_networks
