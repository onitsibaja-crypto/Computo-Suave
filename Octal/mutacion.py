from abc import ABC, abstractmethod
from typing import List
from CromosomaOctal import CromosomaOctal
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


class MutacionOctal(Mutacion):
    def __init__(self, probabilidad: float):
        super().__init__()
        self.prob = probabilidad

    def mutacion_poblacion(self, poblacion: List[CromosomaOctal]):
        for crom in poblacion:
            for i in range(1, crom.n_genes + 1):
                if random.random() < self.prob:
                    # poner a la reina en una fila al azar de 0 a 7
                    nueva_fila = random.randint(0, 7)
                    crom.set_bits(nueva_fila, i)
