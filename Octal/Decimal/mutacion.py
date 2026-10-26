from abc import ABC, abstractmethod
from typing import List
from CromosomaDecimal import CromosomaDecimal
import random


class Mutacion(ABC):
    @abstractmethod
    def mutacion_poblacion(self, poblacion: List[CromosomaOctal]):
        pass


class MutacionFlip(Mutacion):
    def __init__(self, probabilidad: float):
        super().__init__()
        self.prob = probabilidad

    def mutacion_poblacion(self, poblacion: List[CromosomaOctal]):
        max_valor = 2 ** poblacion[0].gen_width - 1
        for crom in poblacion:
            for i in range(1, crom.n_genes + 1):
                prob_gen = random.random()  # de 0 a 1
                if prob_gen < self.prob:
                    valor = crom.get_bits(i)
                    # mutación flip
                    valor = valor ^ max_valor
                    crom.set_bits(valor, i)


class MutacionDecimal(Mutacion):
    def __init__(self, probabilidad):
        super().__init__()
        self.prob = probabilidad

    def mutacion_poblacion(self, poblacion):
        for crom in poblacion:
            for i in range(1, crom.n_genes + 1):
                if random.random() < self.prob:
                    crom.set_bits(random.randint(1, 3), i)
