import random


class RealChromosome:
    def __init__(self, num_genes=1, a=-20, b=20, random_init=True):
        self.num_genes = num_genes
        self.a = a
        self.b = b

        if random_init:
            self.chromosome = [random.uniform(a, b) for _ in range(self.num_genes)]
        else:

            self.chromosome = [(a + b) / 2 for _ in range(self.num_genes)]

    def set_chromosome(self, chromosome):
        if len(chromosome) != self.num_genes:
            raise ValueError(f"Długość chromosomu musi być równa {self.num_genes}")
        self.chromosome = chromosome

    def set_from_values(self, values):
        if len(values) != self.num_genes:
            raise ValueError(f"Liczba wartości musi być równa {self.num_genes}")

        for val in values:
            if val < self.a or val > self.b:
                raise ValueError(f"Wartość {val} poza zakresem [{self.a}, {self.b}]")

        self.chromosome = values
        return self

    def get_chromosome_len(self):
        return self.num_genes

    def __getitem__(self, index):
        if 0 <= index < self.num_genes:
            return self.chromosome[index]
        else:
            raise IndexError("Index out of range")