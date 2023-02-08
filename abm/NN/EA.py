import numpy as np
from abm.NN.CTRNN import CTRNN
import abm.app as sim
import pickle
import random

class EvolAlgo():
    
    def __init__(self, architecture=(99,128,2), dt=100, init=None, population_size=50, generations=500, episodes=10, 
                 mutation_variance=0.02, repop_method='ES'):
        
        # Initialize NN population + fitness lists
        self.architecture = architecture
        self.dt = dt
        self.networks = [CTRNN(architecture, dt, init) for _ in range(population_size)]
        self.fitness_evol = []

        # Evolution + Simulation parameters
        self.population_size = population_size
        self.generations = generations
        self.episodes = episodes
        self.mutation_variance = mutation_variance
        self.repop_method = repop_method
        
    def fit(self):

        for i in range(self.generations):

            # Simulate performance of each NN + store results as array
            fitness_gen = []
            for n, NN in enumerate(self.networks):

                fitness_ep = []
                for _ in range(self.episodes):

                    fitness, elapsed_time, crash = sim.start_headless(NN=NN)
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

            # pickle top performing NN
            best_network = self.networks[np.argmax(fitness_gen)]
            input, hidden, output = self.architecture
            with open(f"_best_NN_gen{i+1}_{input}_{hidden}_{output}_avg{max_fitness_gen}.bin", "wb") as f:
                pickle.dump(best_network, f)

            # pickle generational fitness data
            self.fitness_evol.append(fitness_gen)
            with open("fitness_spread_per_generation.bin", "wb") as f:
                pickle.dump(self.fitness_evol, f)

            # checks stopping condition --> reached 95% of max for X episodes
            # if max_fitness_gen >= self.max_episode_length * 0.95:

            #     # Return best NN from evolutionary run
            #     self.best_network = best_network
            #     return i+1

            # Select/Mutate to generate next generation NNs according to method specified
            if self.repop_method == 'ES':
                self.repop_ES(best_network)
            elif self.repop_method == 'ROULETTE':
                self.repop_roulette(fitness_gen)
            else: 
                return f'Invalid repopulation method specified: {self.repop_method}'


    def repop_ES(self, best_network):

        # Create child NNs as mutations of the top NN
        new_networks = []
        for _ in range(self.population_size - 1):
            new_network = CTRNN(self.architecture, self.dt, copy_network=best_network, var=self.mutation_variance)
            new_networks.append(new_network)
        
        # Set NNs for next generation
        self.networks = [best_network] + new_networks


    def repop_roulette(self, fitness_gen):

        # Select parent NNs via roulette wheel (randomly chosen proportionate to fitness)

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
        self.networks = [CTRNN(self.architecture, self.dt, copy_network=parent, var=self.mutation_variance) for parent in parents]
