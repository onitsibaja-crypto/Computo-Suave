import random
import copy
from typing import Dict, List
from CromosomaDecimal import CromosomaDecimal


class GA(object):
    def __init__(self, num_inv: int, num_genes: int, hyperparams: Dict):
        self.poblacion: List[CromosomaDecimal] = [CromosomaDecimal(num_genes) for i in range(num_inv)]
        self.num_inv: int = num_inv
        self.hyperparams: Dict = hyperparams
        self.evaluar = hyperparams["Aptitud"]
        self.seleccion = hyperparams["Seleccion"]
        self.reproducion = hyperparams["Reproduccion"]
        self.mutacion = hyperparams["Mutacion"]
        self.num_remplazo = int(hyperparams["por_remplazo"] * num_inv)
        # inicializar
        for i in range(num_inv):
            self.poblacion[i].llenar_random_data()

    def buscar_solucion(self, tolerancia: float, var: Dict):
        avg_tol: float = 99999999.99
        sum_score_prev: float = 0
        sum_score: float = 1
        generacion: int = 1
        mating_pool = None
        generacion_inicial: List = copy.deepcopy(self.poblacion)
        generacion_nueva: List = None
        fail: bool = False
        fail_count: int = 0
        fail_score: float = 0.0
        while not fail:
            self.evaluar.evaluar_poblacion(self.poblacion, var)
            sum_scores: float = sum([cromosoma.score for cromosoma in self.poblacion])
            sum_scores /= self.num_inv
            mejor = max(c.score for c in self.poblacion)
            if mejor == 28:
                print(f"Solución encontrada en la generación {generacion}")
                break
            avg_tol: float = abs(sum_scores - sum_score_prev) / sum_score
            sum_score_prev = sum_scores
            if (generacion + 1) % 100 == 0:
                print(f"[{generacion:04d}] avg err: {avg_tol:.5f} avg score: {sum_scores:.5f}")
          
            if generacion >= 1000:
                break
            fail_score = avg_tol
            self.poblacion = sorted(self.poblacion, key=lambda c: c.score, reverse=True)
            mating_pool = self.seleccion.seleccion_poblacion(self.poblacion)
            generacion_nueva = copy.deepcopy(self.poblacion[:self.num_remplazo])
            self.reproducion.reproduccion_poblacion(generacion_nueva, mating_pool, self.num_inv)
            self.mutacion.mutacion_poblacion(generacion_nueva)
            generacion += 1
            self.poblacion = copy.deepcopy(generacion_nueva)


def imprimir_procesadores(cromosoma, tiempos):
    for p in range(1, 4):
        tareas = []
        total = 0
        for i in range(cromosoma.n_genes):
            if cromosoma.genes[i] == p:
                tareas.append("T" + str(i + 1))
                total += tiempos[i]
        print(f"P{p}: {', '.join(tareas):<16} tiempo = {total}")


if __name__ == "__main__":
    from interfaz import AptitudProcesadores, SeleccionRuleta, ReproduccionUnCruce, MutacionDecimal
    variables = {"tiempos": [8, 5, 7, 3, 6, 4]}
    num_genes = 6
    cant_ind = 100
    gaparams = {}
    gaparams["Aptitud"] = AptitudProcesadores()
    gaparams["Seleccion"] = SeleccionRuleta()
    gaparams["Reproduccion"] = ReproduccionUnCruce(num_genes)
    gaparams["Mutacion"] = MutacionDecimal(0.1)
    gaparams["por_remplazo"] = 0.5  # %
    algoritmoGA = GA(cant_ind, num_genes, gaparams)
    algoritmoGA.buscar_solucion(1e-4, variables)

    solucion = sorted(algoritmoGA.poblacion, key=lambda c: c.score, reverse=True)
    print(f"\n{'id':^6} {'Cromosoma':^20} Fitness    M(x)")
    for i, cromosoma in enumerate(solucion[:10]):
        M = round(1 / cromosoma.score - 1)
        print(f"[{i + 1:^4d}] {str(cromosoma.genes):^20} {cromosoma.score:.4f}   {M}")

    mejor = solucion[0]
    print("\nMejor solución:", "".join(str(g) for g in mejor.genes),
          " Fitness:", round(mejor.score, 4), " M(x):", round(1 / mejor.score - 1))
    imprimir_procesadores(mejor, variables["tiempos"])