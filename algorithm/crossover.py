import random

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