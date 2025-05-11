import logging
import pygad
import numpy
import benchmark_functions as bf
import time
from app.config import config
from app.plotter import Plotter
from algorithm.crossover import (custom_arithmetic_crossover, custom_blend_alpha_crossover,
                                 custom_blend_alpha_beta_crossover, custom_averaging_crossover, custom_linear_crossover)
from algorithm.mutation import (custom_uniform_mutation, custom_gaussian_mutation, custom_edge_mutation_binary,
                                custom_single_point_mutation_binary, custom_two_point_mutation_binary)


class GeneticSolution:
    def __init__(self, chromosome_values, fitness):
        self.chromosome_values = chromosome_values
        self.fitness = fitness


func = bf.MartinGaddy()


def fitness_func(ga_instance, solution, solution_idx):
    # For binary representation, decode the solution first
    if config.chromosome_representation == "binary":
        num_bits_per_var = solution.shape[0] // config.num_variables
        decoded_solution = []

        for i in range(config.num_variables):
            start_idx = i * num_bits_per_var
            end_idx = (i + 1) * num_bits_per_var
            bin_var = solution[start_idx:end_idx]

            decimal_value = 0
            for j in range(num_bits_per_var):
                decimal_value += bin_var[j] * (2 ** (num_bits_per_var - 1 - j))

            # Map to range
            decoded_var = config.range_start + decimal_value * (config.range_end - config.range_start) / (
                    2 ** num_bits_per_var - 1)
            decoded_solution.append(decoded_var)

        solution = numpy.array(decoded_solution)

    fitness = func(solution)
    if config.optimization_type == "min":
        return 1. / fitness
    else:
        return fitness


crossover_functions = {
    "arithmetic": custom_arithmetic_crossover,
    "linear": custom_linear_crossover,
    "blend_alpha": custom_blend_alpha_crossover,
    "blend_alpha_beta": custom_blend_alpha_beta_crossover,
    "averaging": custom_averaging_crossover
}

mutation_functions = {
    "single_point": custom_single_point_mutation_binary,
    "two_point": custom_two_point_mutation_binary,
    "edge": custom_edge_mutation_binary,
    "uniform": custom_uniform_mutation,
    "gaussian": custom_gaussian_mutation
}

level = logging.DEBUG
name = 'logfile.txt'
logger = logging.getLogger(name)
logger.setLevel(level)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_format = logging.Formatter('%(message)s')
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)


def run_genetic_algorithm():
    start_time = time.time()

    plotter = Plotter()

    num_generations = config.epochs
    sol_per_pop = config.population_size
    num_parents_mating = int(sol_per_pop * 0.5)

    init_range_low = config.range_start
    init_range_high = config.range_end
    mutation_num_genes = 1

    parent_selection_type = config.selection_method
    crossover_type = config.crossover_method
    mutation_type = config.mutation_method

    if config.chromosome_representation == "binary":
        bits_per_variable = 16
        num_genes = config.num_variables * bits_per_variable
        gene_type = int
        gene_space = [0, 1]
        if config.crossover_method == "single_point":
            crossover_type = "single_point"
        elif config.crossover_method == "two_point":
            crossover_type = "two_points"
        else:
            crossover_type = "uniform"
        mutation_type = "random"
    else:
        num_genes = config.num_variables
        gene_type = float
        gene_space = None

        if config.crossover_method in crossover_functions:
            crossover_type = crossover_functions[config.crossover_method]

        if config.mutation_method in mutation_functions:
            mutation_type = mutation_functions[config.mutation_method]

    def on_generation(ga_instance):
        ga_instance.logger.info("Generation = {generation}".format(generation=ga_instance.generations_completed))
        solution, solution_fitness, solution_idx = ga_instance.best_solution(
            pop_fitness=ga_instance.last_generation_fitness)

        if config.optimization_type == "min":
            fitness_value = 1. / solution_fitness
        else:
            fitness_value = solution_fitness

        if config.chromosome_representation == "binary":
            decoded_solution = []
            bits_per_variable = len(solution) // config.num_variables

            for i in range(config.num_variables):
                start_idx = i * bits_per_variable
                end_idx = (i + 1) * bits_per_variable
                bin_var = solution[start_idx:end_idx]

                decimal_value = 0
                for j in range(bits_per_variable):
                    decimal_value += bin_var[j] * (2 ** (bits_per_variable - 1 - j))

                decoded_var = config.range_start + decimal_value * (config.range_end - config.range_start) / (
                        2 ** bits_per_variable - 1)
                decoded_solution.append(decoded_var)

            display_solution = decoded_solution
        else:
            display_solution = solution

        ga_instance.logger.info("Best = {fitness}".format(fitness=fitness_value))
        ga_instance.logger.info("Individual = {solution}".format(solution=repr(display_solution)))

        if config.optimization_type == "min":
            tmp = [1. / x for x in ga_instance.last_generation_fitness]
        else:
            tmp = ga_instance.last_generation_fitness

        avg_fitness = numpy.average(tmp)

        plotter.update_history(fitness_value, avg_fitness)

        ga_instance.logger.info("Min = {min}".format(min=numpy.min(tmp)))
        ga_instance.logger.info("Max = {max}".format(max=numpy.max(tmp)))
        ga_instance.logger.info("Average = {average}".format(average=avg_fitness))
        ga_instance.logger.info("Std = {std}".format(std=numpy.std(tmp)))
        ga_instance.logger.info("\r\n")

    ga_instance = pygad.GA(
        num_generations=num_generations,
        sol_per_pop=sol_per_pop,
        num_parents_mating=num_parents_mating,
        num_genes=num_genes,
        fitness_func=fitness_func,
        init_range_low=init_range_low if config.chromosome_representation != "binary" else 0,
        init_range_high=init_range_high if config.chromosome_representation != "binary" else 1,
        mutation_num_genes=mutation_num_genes,
        parent_selection_type=parent_selection_type,
        crossover_type=crossover_type,
        mutation_type=mutation_type,
        keep_elitism=1,
        K_tournament=config.tournament_size,
        random_mutation_max_val=init_range_high if config.chromosome_representation != "binary" else 1,
        random_mutation_min_val=init_range_low if config.chromosome_representation != "binary" else 0,
        gene_type=gene_type,
        gene_space=gene_space,
        crossover_probability=config.crossover_probability,
        mutation_percent_genes=config.mutation_probability * 100,
        logger=logger,
        on_generation=on_generation,
        parallel_processing=['thread', 4]
    )

    ga_instance.run()

    execution_time = time.time() - start_time

    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    if config.optimization_type == "min":
        fitness_value = 1. / solution_fitness
    else:
        fitness_value = solution_fitness

    if config.chromosome_representation == "binary":
        decoded_solution = []
        bits_per_variable = len(solution) // config.num_variables

        for i in range(config.num_variables):
            start_idx = i * bits_per_variable
            end_idx = (i + 1) * bits_per_variable
            bin_var = solution[start_idx:end_idx]

            decimal_value = 0
            for j in range(bits_per_variable):
                decimal_value += bin_var[j] * (2 ** (bits_per_variable - 1 - j))
            decoded_var = config.range_start + decimal_value * (config.range_end - config.range_start) / (
                    2 ** bits_per_variable - 1)
            decoded_solution.append(decoded_var)

        solution = numpy.array(decoded_solution)

    best_solution = GeneticSolution(solution, fitness_value)

    return best_solution, execution_time, plotter
