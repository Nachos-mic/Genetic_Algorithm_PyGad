import random
import numpy as np
from algorithm.chromosome_binary import BinaryChromosome


def uniform_crossover(parent1: BinaryChromosome, parent2: BinaryChromosome):
    if parent1 is None or parent2 is None:
        print("Ostrzeżenie: Jeden z rodziców jest None")
        return None, None

    if not isinstance(parent1, BinaryChromosome) or not isinstance(parent2, BinaryChromosome):
        print("Ostrzeżenie: Rodzice muszą być obiektami klasy Chromosome")
        return None, None

    if parent1.get_chromosome_len() != parent2.get_chromosome_len():
        print("Ostrzeżenie: Chromosomy rodziców mają różne długości")
        return None, None

    child1 = BinaryChromosome(parent1.get_chromosome_len(), random_init=False)
    child2 = BinaryChromosome(parent2.get_chromosome_len(), random_init=False)

    new_chromosome1 = []
    new_chromosome2 = []

    for i in range(parent1.get_chromosome_len()):
        if random.random() < 0.5:
            new_chromosome1.append(parent1.chromosome[i])
            new_chromosome2.append(parent2.chromosome[i])
        else:
            new_chromosome1.append(parent2.chromosome[i])
            new_chromosome2.append(parent1.chromosome[i])

    child1.set_chromosome(new_chromosome1)
    child2.set_chromosome(new_chromosome2)

    return child1, child2

def arithmetic_crossover(parent1, parent2):
    alpha = random.random()
    child1 = [alpha * p1 + (1 - alpha) * p2 for p1, p2 in zip(parent1, parent2)]
    child2 = [(1 - alpha) * p1 + alpha * p2 for p1, p2 in zip(parent1, parent2)]
    return child1, child2

def linear_crossover(parent1, parent2):
    c1 = [0.5 * (p1 + p2) for p1, p2 in zip(parent1, parent2)]
    c2 = [1.5 * p1 - 0.5 * p2 for p1, p2 in zip(parent1, parent2)]
    c3 = [-0.5 * p1 + 1.5 * p2 for p1, p2 in zip(parent1, parent2)]

    return c1, c2

def blend_alpha_crossover(parent1, parent2, alpha=0.5):
    child1 = []
    child2 = []
    for p1, p2 in zip(parent1, parent2):
        d = abs(p1 - p2)
        lower = min(p1, p2) - alpha * d
        upper = max(p1, p2) + alpha * d
        c1 = random.uniform(lower, upper)
        c2 = random.uniform(lower, upper)
        child1.append(c1)
        child2.append(c2)
    return child1, child2

def blend_alpha_beta_crossover(parent1, parent2, alpha=0.75, beta=0.25):
    child1 = []
    child2 = []
    for p1, p2 in zip(parent1, parent2):
        d = abs(p1 - p2)
        min_val = min(p1, p2)
        max_val = max(p1, p2)
        lower = min_val - alpha * d
        upper = max_val + beta * d
        c1 = random.uniform(lower, upper)
        c2 = random.uniform(lower, upper)
        child1.append(c1)
        child2.append(c2)
    return child1, child2

def averaging_crossover(parent1, parent2):
    child1 = [(p1 + p2) / 2 for p1, p2 in zip(parent1, parent2)]
    child2 = [(p1 + p2) / 2 for p1, p2 in zip(parent1, parent2)]
    return child1, child2

def custom_arithmetic_crossover(parents, offspring_size, ga_instance):
    offspring = []
    for k in range(0, offspring_size[0], 2):
        if k + 1 < offspring_size[0]:
            parent1 = parents[k % parents.shape[0], :].copy()
            parent2 = parents[(k + 1) % parents.shape[0], :].copy()
            child1, child2 = arithmetic_crossover(parent1, parent2)
            offspring.append(child1)
            offspring.append(child2)
        else:
            offspring.append(parents[k % parents.shape[0], :].copy())
    return np.array(offspring)


def custom_linear_crossover(parents, offspring_size, ga_instance):
    offspring = []
    for k in range(0, offspring_size[0], 2):
        if k + 1 < offspring_size[0]:
            parent1 = parents[k % parents.shape[0], :].copy()
            parent2 = parents[(k + 1) % parents.shape[0], :].copy()
            child1, child2 = linear_crossover(parent1, parent2)
            offspring.append(child1)
            if len(offspring) < offspring_size[0]:
                offspring.append(child2)
        else:
            offspring.append(parents[k % parents.shape[0], :].copy())
    return np.array(offspring)


def custom_blend_alpha_crossover(parents, offspring_size, ga_instance):
    offspring = []
    for k in range(0, offspring_size[0], 2):
        if k + 1 < offspring_size[0]:
            parent1 = parents[k % parents.shape[0], :].copy()
            parent2 = parents[(k + 1) % parents.shape[0], :].copy()
            child1, child2 = blend_alpha_crossover(parent1, parent2)
            offspring.append(child1)
            offspring.append(child2)
        else:
            offspring.append(parents[k % parents.shape[0], :].copy())
    return np.array(offspring)


def custom_blend_alpha_beta_crossover(parents, offspring_size, ga_instance):
    offspring = []
    for k in range(0, offspring_size[0], 2):
        if k + 1 < offspring_size[0]:
            parent1 = parents[k % parents.shape[0], :].copy()
            parent2 = parents[(k + 1) % parents.shape[0], :].copy()
            child1, child2 = blend_alpha_beta_crossover(parent1, parent2)
            offspring.append(child1)
            offspring.append(child2)
        else:
            offspring.append(parents[k % parents.shape[0], :].copy())
    return np.array(offspring)


def custom_averaging_crossover(parents, offspring_size, ga_instance):
    offspring = []
    for k in range(0, offspring_size[0], 2):
        if k + 1 < offspring_size[0]:
            parent1 = parents[k % parents.shape[0], :].copy()
            parent2 = parents[(k + 1) % parents.shape[0], :].copy()
            child1, child2 = averaging_crossover(parent1, parent2)
            offspring.append(child1)
            offspring.append(child2)
        else:
            offspring.append(parents[k % parents.shape[0], :].copy())
    return np.array(offspring)


def pygad_uniform_crossover(parents, offspring_size, ga_instance):

    offspring = []
    num_parents = parents.shape[0]
    num_genes = parents.shape[1]


    for k in range(offspring_size[0] // 2):

        parent1 = BinaryChromosome(num_genes, random_init=False)
        parent2 = BinaryChromosome(num_genes, random_init=False)
        parent1.set_chromosome(list(parents[k % num_parents]))
        parent2.set_chromosome(list(parents[(k + 1) % num_parents]))

        child1, child2 = uniform_crossover(parent1, parent2)

        offspring.append(np.array(child1.chromosome))
        offspring.append(np.array(child2.chromosome))

    if offspring_size[0] % 2 != 0:
        parent1 = BinaryChromosome(num_genes, random_init=False)
        parent2 = BinaryChromosome(num_genes, random_init=False)
        parent1.set_chromosome(list(parents[-1]))
        parent2.set_chromosome(list(parents[0]))
        child1, _ = uniform_crossover(parent1, parent2)
        offspring.append(np.array(child1.chromosome))

    return np.array(offspring)