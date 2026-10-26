from abc import ABC, abstractmethod
from typing import Dict, List
from CromosomaDecimal import CromosomaDecimal


class Aptitud(ABC):
    @abstractmethod
    def evaluar_poblacion(self, poblacion: List[CromosomaOctal], var: Dict):
        """interfaz encargada para darle un score a la población"""
        pass


class AptitudProcesadores(Aptitud):
    def evaluar_poblacion(self, poblacion, var):
        tiempos = var["tiempos"]               # [8, 5, 7, 3, 6, 4]
        for cromosoma in poblacion:
            carga = [0, 0, 0]                  # P1, P2, P3
            for i in range(1, cromosoma.n_genes + 1):
                procesador = cromosoma.get_bits(i)
                carga[procesador - 1] += tiempos[i - 1]
            M = max(carga)
            cromosoma.score = 1 / (1 + M)