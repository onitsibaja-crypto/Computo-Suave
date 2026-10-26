import random


class CromosomaDecimal(object):
    def __init__(self, genes: int):
        self.n_genes = genes
        self.score = -1
        self.genes = [1] * genes      # un número por tarea

    def llenar_random_data(self):
        for i in range(self.n_genes):
            self.genes[i] = random.randint(1, 3)

    def get_bits(self, gen):          # gen va de 1 a 6
        return self.genes[gen - 1]

    def set_bits(self, valor, gen):
        self.genes[gen - 1] = valor
