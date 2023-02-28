import numpy as np
from abm.NN.RNNs import RNN
import abm.app as sim
import pickle
# import json
import random
import os
import shutil
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
        self.networks = [RNN(arch, type, rule, activ, dt, init) for _ in range(population_size)]
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
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        self.sim_data_dir = os.path.join(root_dir, 'abm\data\simulation_data')
        
    def fit(self):

        for i in range(self.generations):

            # Simulate performance of each NN + store results as array
            fitness_gen = []
            for n, NN in enumerate(self.networks):

                fitness_ep = []
                for x in range(self.episodes):

                    # construct save name for current simulation, to be called later if needed (e.g. to plot top performers)
                    sim_save_name = fr'{self.EA_save_name}\running\NN{n}\ep{x}'

                    # run sim + record fitness/time
                    fitness, elapsed_time, crash = sim.start_headless(NN=NN, sim_save_name=sim_save_name)
                    fitness_ep.append(fitness)
                    print(f'Episode Fitness: {fitness} \t| Elapsed Time: {elapsed_time}')

                    if crash: # save crashed NN in binary mode + continue
                        print('Crashed agent - pickled NN')
                        with open("crashed_NN.bin", "wb") as f:
                            pickle.dump(NN, f)
                            # json.dump(NN, f)
                
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
                NN_load_name = fr'{self.EA_save_name}\running\NN{n_gen}'
                NN_save_name = fr'{self.EA_save_name}\gen{i}\NN{n_top}_fitness{int(fitness_gen[n_gen])}'

                NN_load_dir = os.path.join(self.sim_data_dir, NN_load_name)
                NN_save_dir = os.path.join(self.sim_data_dir, NN_save_name)

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

                # # pull/pickle weights/biases from network
                # param_list = NN.pull_parameters()

                # print(type(param_list))
                # print(type(param_list[-1]))
                # print(type(param_list[0][-1]))
                # print(param_list[0][0][-1], type(param_list[0][0][-1]))

                # with open(rf'{NN_save_dir}\NN_params.json','wb') as f:
                #     json.dump(param_list, f)
            
            # update/pickle generational fitness data in parent directory
            self.fitness_evol.append(fitness_gen)

            gen_save_dir = os.path.join(self.sim_data_dir, self.EA_save_name)
            with open(rf'{gen_save_dir}\fitness_spread_per_generation.bin', 'wb') as f:
                pickle.dump(self.fitness_evol, f)
                # json.dump(self.fitness_evol, f)

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
