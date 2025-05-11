import random
import numpy
from algorithm.chromosome_binary import BinaryChromosome

def edge_mutation_binary(chromosome, probability=1):
    if random.random() <= probability:
        chromosome.change_chromosome_bit(0)
        chromosome.change_chromosome_bit(chromosome.get_chromosome_len() - 1)
    return chromosome


def single_point_mutation_binary(chromosome, probability=1):
    if random.random() <= probability:
        mutation_point = random.randint(0, chromosome.get_chromosome_len() - 1)
        chromosome.change_chromosome_bit(mutation_point)
    return chromosome


def two_point_mutation_binary(chromosome, probability=1):
    if random.random() <= probability:
        points = random.sample(range(chromosome.get_chromosome_len()), 2)
        for point in points:
            chromosome.change_chromosome_bit(point)
    return chromosome


def uniform_mutation_real(chromosome, probability=1, a=-20, b=20):
    if random.random() <= probability:
        position = random.randint(0, chromosome.get_chromosome_len() - 1)
        chromosome.chromosome[position] = random.uniform(a, b)
    return chromosome


def gaussian_mutation_real(chromosome, probability=1, sigma=1.0, a=-20, b=20):
    if random.random() <= probability:
        position = random.randint(0, chromosome.get_chromosome_len() - 1)
        mutation = random.gauss(0, sigma)
        chromosome.chromosome[position] += mutation
        chromosome.chromosome[position] = max(a, min(b, chromosome.chromosome[position]))
    return chromosome


def custom_uniform_mutation(offspring, ga_instance):

    a = ga_instance.random_mutation_min_val
    b = ga_instance.random_mutation_max_val

    mutation_num = numpy.uint32((ga_instance.mutation_percent_genes * offspring.shape[1]) / 100)
    mutation_indices = numpy.array(random.sample(range(0, offspring.shape[1]), mutation_num))

    for idx in range(offspring.shape[0]):
        for gene_idx in mutation_indices:
            offspring[idx, gene_idx] = random.uniform(a, b)

    return offspring

def custom_single_point_mutation_binary(offspring, ga_instance):
    for idx in range(offspring.shape[0]):
        chrom = BinaryChromosome(list(offspring[idx]))
        mutated = single_point_mutation_binary(chrom)
        offspring[idx] = mutated.chromosome
    return offspring

def custom_edge_mutation_binary(offspring, ga_instance):
    for idx in range(offspring.shape[0]):
        chrom = BinaryChromosome(list(offspring[idx]))
        mutated = edge_mutation_binary(chrom)
        offspring[idx] = mutated.chromosome
    return offspring

def custom_two_point_mutation_binary(offspring, ga_instance):
    for idx in range(offspring.shape[0]):
        chrom = BinaryChromosome(list(offspring[idx]))
        mutated = two_point_mutation_binary(chrom)
        offspring[idx] = mutated.chromosome
    return offspring

def custom_gaussian_mutation(offspring, ga_instance):

    a = ga_instance.random_mutation_min_val
    b = ga_instance.random_mutation_max_val
    sigma = (b - a) / 10


    mutation_num = numpy.uint32((ga_instance.mutation_percent_genes * offspring.shape[1]) / 100)
    mutation_indices = numpy.array(random.sample(range(0, offspring.shape[1]), mutation_num))

    for idx in range(offspring.shape[0]):
        for gene_idx in mutation_indices:
            mutation = random.gauss(0, sigma)
            offspring[idx, gene_idx] += mutation
            offspring[idx, gene_idx] = max(a, min(b, offspring[idx, gene_idx]))

    return offspring