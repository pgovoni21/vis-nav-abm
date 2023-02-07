import numpy as np
from abm.NN.CTRNN import CTRNN
import abm.app as sim
import pickle

class EvolAlgo():
    
    def __init__(self, architecture=(99,128,2), dt=100, init=None, population_size=50, generations=500, episodes=10, 
                 mutation_variance=0.02):
        
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
        
    def fit(self):

        for i in range(self.generations):

            # Simulate performance of each NN + store results as array
            fitness_gen = []
            for n, NN in enumerate(self.networks):

                fitness_ep = []
                for _ in range(self.episodes):

                    fitness, elapsed_time, crash = sim.start_headless(NN=NN)
                    fitness_ep.append(fitness)
                    print(f'---+--- Episode Fitness: {fitness} | Elapsed Time: {elapsed_time} ---+---')

                    if crash: # save crashed NN in binary mode + exit EA loop
                        with open("crashed_NN.bin", "wb") as f: 
                            pickle.dump(NN, f)
                        return print('Crashed agent - pickled NN')

                avg_fitness = sum(fitness_ep)/len(fitness_ep)
                fitness_gen.append(avg_fitness)
                print(f'--- NN {n+1} of {self.population_size} | Avg Across Episodes: {avg_fitness} ---')

            # Track top fitness per generation
            max_fitness_gen = round(np.max(fitness_gen),1)
            avg_fitness_gen = round(np.mean(fitness_gen),1)
            self.fitness_evol.append(max_fitness_gen)
            print(f'Generation: {i+1} | Highest Across Gen: {max_fitness_gen} | Avg Across Gen: {avg_fitness_gen}')

            # Select NN with top fitness
            best_network = self.networks[np.argmax(fitness_gen)]

            # pickle it
            input, hidden, output = self.architecture
            with open(f"_best_NN_gen{i+1}_{input}_{hidden}_{output}_avg{int(max_fitness_gen)}.bin", "wb") as f:
                pickle.dump(best_network, f)

            # # pickle the generational trend
            # with open("best_NN_avg_fitness_per_generation.bin", "wb") as f:
            #     pickle.dump(self.fitness_evol, f)

            # checks stopping condition --> reached 95% of max for X episodes
            # if max_fitness_gen >= self.max_episode_length * 0.95:

            #     # Return best NN from evolutionary run
            #     self.best_network = best_network
            #     return i+1

            # Create child NNs as mutations of the top NN
            new_networks = []
            for _ in range(self.population_size - 1):
                new_network = CTRNN(self.architecture, self.dt, copy_network=best_network, var=self.mutation_variance)
                new_networks.append(new_network)
            
            # Set NNs for next generation
            self.networks = [best_network] + new_networks