from abc import ABC, abstractmethod
from typing import List
from CromosomaDecimal import CromosomaDecimal
import random


class Reproduccion(ABC):
    @abstractmethod
    def reproduccion_poblacion(self, poblacion: List[CromosomaDecimal], potencial: List[CromosomaDecimal], max_poblacion: int) -> None:
        pass


class ReproduccionUnCruce(Reproduccion):
    def __init__(self, num_genes):
        super().__init__()
        self.num_genes = num_genes

    def reproduccion_poblacion(self, poblacion: List[CromosomaDecimal], potencial: List[CromosomaDecimal], max_poblacion: int) -> None:
        curr_po = len(poblacion)

        ord_padres = sorted(potencial, key=lambda c: c.score, reverse=True)
        id_padres: int = 0
        while curr_po < max_poblacion:
            hijoA = CromosomaDecimal(self.num_genes)
            hijoB = CromosomaDecimal(self.num_genes)
            punto_cruce = random.randint(1, self.num_genes - 1)
            padreA = ord_padres[id_padres % max_poblacion]
            padreB = ord_padres[(id_padres + 1) % max_poblacion]
            # cruzar
            for i in range(1, self.num_genes + 1):  # +1 para que no se salte la reina 8
                if i <= punto_cruce:
                    hijoA.set_bits(padreA.get_bits(i), i)
                    hijoB.set_bits(padreB.get_bits(i), i)
                else:
                    hijoA.set_bits(padreB.get_bits(i), i)
                    hijoB.set_bits(padreA.get_bits(i), i)
            if curr_po < max_poblacion:
                poblacion.append(hijoA)
                curr_po += 1
            if curr_po < max_poblacion:
                poblacion.append(hijoB)
                curr_po += 1
            id_padres += 1
