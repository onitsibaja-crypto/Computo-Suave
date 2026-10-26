import random
from abc import ABC, abstractmethod
from typing import List
from CromosomaDecimal import CromosomaDecimal


class Seleccion(ABC):
    @abstractmethod
    def seleccion_poblacion(self, poblacion: List[CromosomaOctal], participantes: int = 3) -> List[CromosomaOctal]:
        pass


class SeleccionTorneo(Seleccion):
    def __init__(self, participantes=3) -> None:
        super().__init__()
        self.participantes = participantes

    def seleccion_poblacion(self, poblacion: List[CromosomaOctal]) -> List[CromosomaOctal]:
        ganadores = []
        for i in range(len(poblacion)):
            competidores = random.sample(poblacion, self.participantes)
            ganador = max(competidores, key=lambda cromosoma: cromosoma.score)
            ganadores.append(ganador)
        return ganadores


# Estas dos no venian en las diapositivas pero GAbin.py las importa


class SeleccionRuleta(Seleccion):
    def seleccion_poblacion(self, poblacion):
        # sumar todos los scores
        total = 0
        for c in poblacion:
            total = total + c.score

        ganadores = []
        for i in range(len(poblacion)):
            # si todos tienen 0 se escoge uno al azar
            if total == 0:
                ganadores.append(random.choice(poblacion))
                continue
            # girar la ruleta
            r = random.uniform(0, total)
            acumulado = 0
            for c in poblacion:
                acumulado = acumulado + c.score
                if acumulado >= r:
                    ganadores.append(c)
                    break
        return ganadores


class SeleccionRank(Seleccion):
    def seleccion_poblacion(self, poblacion):
        # ordenar de menor a mayor, el peor tiene rango 1 y el mejor rango n
        ordenada = sorted(poblacion, key=lambda c: c.score)
        n = len(ordenada)
        total = n * (n + 1) / 2  # suma de 1 + 2 + ... + n

        ganadores = []
        for i in range(n):
            r = random.uniform(0, total)
            acumulado = 0
            for rango in range(n):
                acumulado = acumulado + (rango + 1)
                if acumulado >= r:
                    ganadores.append(ordenada[rango])
                    break
        return ganadores
