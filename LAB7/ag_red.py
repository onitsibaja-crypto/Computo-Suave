import copy
import random
import time
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier


# ---------------------------------------------------------------------------
# Espacio de búsqueda: cada gen elige un valor de su lista
# ---------------------------------------------------------------------------
ESPACIO: Dict[str, list] = {
    "n_capas":       [1, 2, 3],
    "neuronas":      [8, 16, 32, 64, 128],
    "forma":         ["constante", "decreciente"],
    "learning_rate": [0.0001, 0.0005, 0.001, 0.005, 0.01],
    "activacion":    ["relu", "tanh", "logistic"],
    "alpha":         [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    "batch_size":    [32, 64, 128, 256],
    "paciencia":     [10, 20, 40],     
}


HIPER_BASE: Dict = {
    "capas": (64, 32),
    "activacion": "relu",
    "learning_rate": 0.001,
    "alpha": 1e-4,
    "batch_size": 64,
    "paciencia": 10,
}


GENES_BASE = [1, 3, 1, 2, 0, 1, 1, 0]

MAX_ITER = 300  



class CromosomaRed(object):
    def __init__(self, espacio: Dict = ESPACIO):
        self.espacio = espacio
        self.nombres = list(espacio)
        self.n_genes = len(self.nombres)
        self.score = -1
        self.genes = [0] * self.n_genes   

    def llenar_random_data(self):
        for i, nombre in enumerate(self.nombres):
            self.genes[i] = random.randrange(len(self.espacio[nombre]))

    def get_bits(self, gen):             
        return self.genes[gen - 1]

    def set_bits(self, valor, gen):
        self.genes[gen - 1] = valor

    def rango(self, gen):               
        return len(self.espacio[self.nombres[gen - 1]])

    def clave(self):
        return tuple(self.genes)

    def decodificar(self) -> Dict:
        """Traduce los genes a hiperparámetros de la red."""
        v = {n: self.espacio[n][g] for n, g in zip(self.nombres, self.genes)}
        if v["forma"] == "decreciente":
            capas = tuple(max(4, v["neuronas"] // (2 ** i)) for i in range(v["n_capas"]))
        else:
            capas = tuple([v["neuronas"]] * v["n_capas"])
        return {
            "capas": capas,
            "activacion": v["activacion"],
            "learning_rate": v["learning_rate"],
            "alpha": v["alpha"],
            "batch_size": v["batch_size"],
            "paciencia": v["paciencia"],
        }



def entrenar(hiper, X, y, semilla=42, max_iter=MAX_ITER) -> MLPClassifier:
    modelo = MLPClassifier(
        hidden_layer_sizes=hiper["capas"],
        activation=hiper["activacion"],
        solver="adam",
        learning_rate_init=hiper["learning_rate"],
        alpha=hiper["alpha"],
        batch_size=hiper["batch_size"],
        max_iter=max_iter,
        early_stopping=True,              
        validation_fraction=0.1,
        n_iter_no_change=hiper["paciencia"],
        random_state=semilla,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        modelo.fit(X, y)
    return modelo


def evaluar(modelo, X, y, promedio_f1="macro") -> Dict:
    """Métrica de rendimiento (F1 y accuracy) y pérdida (log-loss)."""
    pred = modelo.predict(X)
    return {
        "f1": f1_score(y, pred, average=promedio_f1),
        "accuracy": accuracy_score(y, pred),
        "loss": log_loss(y, modelo.predict_proba(X), labels=modelo.classes_),
    }


def balancear(indices, y, semilla=42):
    """Submuestreo: deja tantos ejemplos de cada clase como tenga la clase más pequeña."""
    rng = np.random.default_rng(semilla)
    clases = np.unique(y[indices])
    n_min = min((y[indices] == c).sum() for c in clases)
    elegidos = [rng.choice(indices[y[indices] == c], n_min, replace=False) for c in clases]
    return rng.permutation(np.concatenate(elegidos))


def _entrenar_fold(hiper, X, y, idx_tr, idx_va, promedio_f1, semilla):
    modelo = entrenar(hiper, X[idx_tr], y[idx_tr], semilla)
    return evaluar(modelo, X[idx_va], y[idx_va], promedio_f1)


class Aptitud(ABC):
    @abstractmethod
    def evaluar_poblacion(self, poblacion: List[CromosomaRed]):
        pass


class AptitudRedNeuronal(Aptitud):


    def __init__(self, X, y, promedio_f1="macro", folds=3, balancear_clases=False,
                 semilla=42, n_jobs=-1):
        self.X, self.y = X, y
        self.promedio_f1 = promedio_f1
        self.n_jobs = n_jobs
        self.cache: Dict = {}
        self.n_entrenamientos = 0
        kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=semilla)
        self.folds = []
        for i, (idx_tr, idx_va) in enumerate(kf.split(X, y)):
            if balancear_clases:
                idx_tr = balancear(idx_tr, y, semilla + i)
            self.folds.append((idx_tr, idx_va))

    def evaluar_hiperparametros(self, lista_hiper: List[Dict]) -> List[Dict]:

        tareas = [(i, h, f) for i, h in enumerate(lista_hiper) for f in range(len(self.folds))]
        resultados = Parallel(n_jobs=self.n_jobs)(
            delayed(_entrenar_fold)(h, self.X, self.y, *self.folds[f], self.promedio_f1, f)
            for _, h, f in tareas)
        self.n_entrenamientos += len(tareas)
        salida = []
        for i in range(len(lista_hiper)):
            res = [r for (j, _, _), r in zip(tareas, resultados) if j == i]
            prom = {k: float(np.mean([r[k] for r in res])) for k in res[0]}
            prom["f1_std"] = float(np.std([r["f1"] for r in res]))
            salida.append(prom)
        return salida

    def evaluar_poblacion(self, poblacion):
        nuevos = {}
        for c in poblacion:
            if c.clave() not in self.cache:
                nuevos[c.clave()] = c.decodificar()
        if nuevos:
            for clave, res in zip(nuevos, self.evaluar_hiperparametros(list(nuevos.values()))):
                self.cache[clave] = res
        for c in poblacion:
            c.score = self.cache[c.clave()]["f1"]


class Seleccion(ABC):
    @abstractmethod
    def seleccion_poblacion(self, poblacion: List[CromosomaRed]) -> List[CromosomaRed]:
        pass


class SeleccionTorneo(Seleccion):
    def __init__(self, participantes=3):
        self.participantes = participantes

    def seleccion_poblacion(self, poblacion):
        ganadores = []
        for _ in range(len(poblacion)):
            competidores = random.sample(poblacion, self.participantes)
            ganadores.append(max(competidores, key=lambda c: c.score))
        return ganadores


class Reproduccion(ABC):
    @abstractmethod
    def reproduccion_poblacion(self, poblacion, potencial, max_poblacion) -> None:
        pass


class ReproduccionUnCruce(Reproduccion):
    def __init__(self, espacio=ESPACIO):
        self.espacio = espacio
        self.num_genes = len(espacio)

    def reproduccion_poblacion(self, poblacion, potencial, max_poblacion):
        curr_po = len(poblacion)
        ord_padres = sorted(potencial, key=lambda c: c.score, reverse=True)
        id_padres = 0
        while curr_po < max_poblacion:
            hijoA = CromosomaRed(self.espacio)
            hijoB = CromosomaRed(self.espacio)
            punto_cruce = random.randint(1, self.num_genes - 1)
            padreA = ord_padres[id_padres % len(ord_padres)]
            padreB = ord_padres[(id_padres + 1) % len(ord_padres)]
            for i in range(1, self.num_genes + 1):
                if i <= punto_cruce:
                    hijoA.set_bits(padreA.get_bits(i), i)
                    hijoB.set_bits(padreB.get_bits(i), i)
                else:
                    hijoA.set_bits(padreB.get_bits(i), i)
                    hijoB.set_bits(padreA.get_bits(i), i)
            for hijo in (hijoA, hijoB):
                if curr_po < max_poblacion:
                    poblacion.append(hijo)
                    curr_po += 1
            id_padres += 1


class Mutacion(ABC):
    @abstractmethod
    def mutacion_poblacion(self, poblacion: List[CromosomaRed]):
        pass


class MutacionPorGen(Mutacion):

    def __init__(self, probabilidad: float):
        self.prob = probabilidad

    def mutacion_poblacion(self, poblacion):
        for crom in poblacion:
            for i in range(1, crom.n_genes + 1):
                if random.random() < self.prob:
                    opciones = [v for v in range(crom.rango(i)) if v != crom.get_bits(i)]
                    crom.set_bits(random.choice(opciones), i)



class GA(object):
    def __init__(self, num_inv: int, hyperparams: Dict, espacio: Dict = ESPACIO, incluir=None):
        self.poblacion: List[CromosomaRed] = [CromosomaRed(espacio) for _ in range(num_inv)]
        self.num_inv = num_inv
        self.evaluar = hyperparams["Aptitud"]
        self.seleccion = hyperparams["Seleccion"]
        self.reproducion = hyperparams["Reproduccion"]
        self.mutacion = hyperparams["Mutacion"]
        self.num_remplazo = max(1, int(hyperparams["por_remplazo"] * num_inv))
        self.num_elite = hyperparams.get("elite", 1)
        self.historial: List[Dict] = []
        self.mejor: CromosomaRed = None
        for c in self.poblacion:
            c.llenar_random_data()
        # individuos que se meten a mano en la primera generación (p. ej. la red base)
        for c, genes in zip(self.poblacion, incluir or []):
            c.genes = list(genes)

    def buscar_solucion(self, max_generaciones=12, paciencia=5, verbose=True):
        sin_mejora = 0
        for generacion in range(1, max_generaciones + 1):
            t0 = time.time()
            self.evaluar.evaluar_poblacion(self.poblacion)
            self.poblacion = sorted(self.poblacion, key=lambda c: c.score, reverse=True)
            scores = [c.score for c in self.poblacion]

            if self.mejor is None or self.poblacion[0].score > self.mejor.score + 1e-6:
                self.mejor = copy.deepcopy(self.poblacion[0])
                sin_mejora = 0
            else:
                sin_mejora += 1

            self.historial.append({
                "generacion": generacion,
                "mejor": max(scores),
                "promedio": float(np.mean(scores)),
                "peor": min(scores),
                "mejor_global": self.mejor.score,
                "entrenamientos": self.evaluar.n_entrenamientos,
            })
            if verbose:
                h = self.historial[-1]
                print(f"Generación {generacion:2d}:  mejor F1 = {h['mejor']:.3f} | "
                      f"promedio = {h['promedio']:.3f}   ({time.time() - t0:.0f} s)")

            if generacion == max_generaciones:
                break
            if sin_mejora >= paciencia:
                if verbose:
                    print(f"El AG se detiene: {paciencia} generaciones sin mejorar.")
                break

            # nueva generación: élite intacta + mejores + hijos (mutados)
            mating_pool = self.seleccion.seleccion_poblacion(self.poblacion)
            elite = copy.deepcopy(self.poblacion[:self.num_elite])
            generacion_nueva = copy.deepcopy(self.poblacion[self.num_elite:self.num_remplazo])
            self.reproducion.reproduccion_poblacion(generacion_nueva, mating_pool,
                                                    self.num_inv - self.num_elite)
            self.mutacion.mutacion_poblacion(generacion_nueva)
            self.poblacion = elite + generacion_nueva
        return self.mejor

    def top(self, n=5) -> List[Dict]:
        """Las n mejores configuraciones evaluadas en todo el proceso."""
        filas = []
        for clave, res in self.evaluar.cache.items():
            c = CromosomaRed(self.poblacion[0].espacio)
            c.genes = list(clave)
            filas.append({"genes": clave, **c.decodificar(), **res})
        return sorted(filas, key=lambda f: f["f1"], reverse=True)[:n]
