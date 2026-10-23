from abc import ABC, abstractmethod
from typing import Dict, List
from CromosomaOctal import CromosomaOctal


class Aptitud(ABC):
    @abstractmethod
    def evaluar_poblacion(self, poblacion: List[CromosomaOctal], var: Dict):
        """interfaz encargada para darle un score a la población"""
        pass


class AptitudReinas(Aptitud):
    def evaluar_poblacion(self, poblacion: List[CromosomaOctal], var: Dict):
        for cromosoma in poblacion:
            filas = []
            for i in range(1, cromosoma.n_genes + 1):
                filas.append(cromosoma.get_bits(i))
            conflictos = 0
            for i in range(cromosoma.n_genes):
                for j in range(i + 1, cromosoma.n_genes):
                    if filas[i] == filas[j]:                      # misma fila
                        conflictos += 1
                    elif abs(filas[i] - filas[j]) == abs(i - j):  # misma diagonal
                        conflictos += 1
            cromosoma.score = 28 - conflictos
