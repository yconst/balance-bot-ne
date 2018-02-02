import random
import time
import pickle

import numpy as np

from deap import base, creator
from deap import tools

class EvolAlgorithm():
    '''
    Simple elitist genetic algorithm for optimizing the balancing bot network
    '''
    def __init__(self, objective_function, param_size = 10, pop_size=200):
        self.checkpoint_frequency = 5
        self.p_xo = 0.95
        self.p_mt = 0.2
        self.objective_function = objective_function
        self.param_size = param_size
        self.pop_size = pop_size

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attribute", scale_random)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                         self.toolbox.attribute, n=self.param_size)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", objective_function)

        self.halloffame = tools.HallOfFame(maxsize=1)
        self.logbook = tools.Logbook()
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("mean", np.mean)
        self.stats.register("max", np.max)

        self.reset()
    
    def reset(self):
        self.pop = self.toolbox.population(n=self.pop_size)
        self.best = None

    def run(self, ngen):
        # Evaluate the entire population
        fitnesses = map(self.toolbox.evaluate, self.pop)
        for ind, fit in zip(self.pop, fitnesses):
            ind.fitness.values = fit

        for g in range(ngen):
            start = time.time()
            # Select the next generation individuals
            offspring = self.toolbox.select(self.pop, len(self.pop))
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.p_xo:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.p_mt:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # The population is entirely replaced by the offspring
            self.pop[:] = tools.selBest(offspring+self.pop, len(self.pop))

            dt = time.time() - start

            self.halloffame.update(self.pop)
            record = self.stats.compile(self.pop)
            self.logbook.record(gen=g, evals=len(invalid_ind), **record)

            if g % self.checkpoint_frequency == 0 and g > 0:
                print("Saving checkpoint...")
                cp = dict(population=self.pop, generation=g, halloffame=self.halloffame,
                          logbook=self.logbook, rndstate=random.getstate())

                with open("checkpoint.pkl", "wb") as cp_file:
                    pickle.dump(cp, cp_file)

            mu = record["mean"]
            mx = record["max"]

            print("Generation {0}, mean fitness is {1:.5f}, max fitness is \
{2:.5f}. Finished in {3:.2f} sec".format(g, mu, mx, dt))

        self.best = self.halloffame[0]


def scale_random(factor=0.1, zero_center=True):
    return random.random() * factor - factor * 0.5