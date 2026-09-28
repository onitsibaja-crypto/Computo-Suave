
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
 
np.random.seed(42)
BASE = os.path.dirname(os.path.abspath(__file__))
 

def optimizador(W, gradW, alpha):
    return W - alpha * gradW
 
 
class Capa:
    def __init__(self, neuronas, entradas, activacion=None):
        self.activacion = activacion
        self.pesos = np.random.randn(entradas, neuronas) * np.sqrt(2 / entradas)  # He
        self.bias = np.zeros((1, neuronas))
 
    def feed_forward(self, X):
        self.entrada = X
        self.z = np.dot(X, self.pesos) + self.bias
        if self.activacion == "relu":
            return np.maximum(0, self.z)
        return self.z
 
    def derivada_activacion(self):
        if self.activacion == "relu":
            return (self.z > 0).astype(float)
        return np.ones_like(self.z)
 
    def backward(self, es_ultima, error_derivada=None, delta_siguiente=None, W_siguiente=None):
        if es_ultima:
            delta = error_derivada * self.derivada_activacion()
        else:
            delta = np.dot(delta_siguiente, W_siguiente.T) * self.derivada_activacion()
        self.grad_pesos = np.dot(self.entrada.T, delta)
        self.grad_bias = np.sum(delta, axis=0, keepdims=True)
        return delta
 
 
class Red:
    def __init__(self, capas):
        self.capas = capas
        self.historial = []
 
    def forward(self, X):
        for capa in self.capas:
            X = capa.feed_forward(X)
        return X
 
    def fit(self, X, y, alpha, epocas, tam_lote):
        y = y.reshape(-1, 1)
        n = len(X)
        for ep in range(epocas):
            indices = np.random.permutation(n)
            for inicio in range(0, n, tam_lote):
                lote = indices[inicio:inicio + tam_lote]
                y_hat = self.forward(X[lote])
                error = (y_hat - y[lote]) / len(lote)   # derivada del MSE
 
                delta = self.capas[-1].backward(es_ultima=True, error_derivada=error)
                for i in reversed(range(len(self.capas) - 1)):
                    delta = self.capas[i].backward(False, delta_siguiente=delta,
                                                   W_siguiente=self.capas[i + 1].pesos)
                for capa in self.capas:
                    capa.pesos = optimizador(capa.pesos, capa.grad_pesos, alpha)
                    capa.bias = optimizador(capa.bias, capa.grad_bias, alpha)
 
            self.historial.append(np.mean((self.forward(X) - y) ** 2))
            if ep % (epocas // 5) == 0:
                print(f"  Época {ep}: MSE = {self.historial[-1]:.4f}")
 
    def predict(self, X):
        return self.forward(X).ravel()
 
 

 
def resolver(nombre, df, capas_ocultas, alpha, epocas, tam_lote):
    print(f"\n===== {nombre} =====")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
 
    # 60% entrenamiento / 40% prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
 
    # Escalado (se ajusta solo con entrenamiento)
    esc_X = StandardScaler().fit(X_train)
    esc_y = StandardScaler().fit(y_train.reshape(-1, 1))
    X_train = esc_X.transform(X_train)
    X_test = esc_X.transform(X_test)
    y_train_e = esc_y.transform(y_train.reshape(-1, 1)).ravel()
 
    # Arquitectura: ocultas con ReLU, salida lineal de 1 neurona
    capas, entradas = [], X.shape[1]
    for neuronas in capas_ocultas:
        capas.append(Capa(neuronas, entradas, "relu"))
        entradas = neuronas
    capas.append(Capa(1, entradas))
    red = Red(capas)
 
    red.fit(X_train, y_train_e, alpha, epocas, tam_lote)
 
    # Predicción en unidades originales y métricas
    y_pred = esc_y.inverse_transform(red.predict(X_test).reshape(-1, 1)).ravel()
    r2 = r2_score(y_test, y_pred)
    r = np.corrcoef(y_test, y_pred)[0, 1]
    print(f"  R² = {r2:.4f}   R = {r:.4f}")
 
    # Gráficas
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(red.historial)
    ax[0].set(title="Curva de aprendizaje", xlabel="Época", ylabel="MSE")
    ax[1].scatter(y_test, y_pred, s=6, alpha=0.4)
    ax[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    ax[1].set(title=f"Real vs. predicho (R² = {r2:.3f})", xlabel="Real", ylabel="Predicho")
    fig.suptitle(nombre)
    fig.tight_layout()
    fig.savefig(os.path.join(BASE, nombre + ".png"), dpi=150)
 
    return r2, r
 
 
# ======================= PROGRAMA PRINCIPAL =======================
 
if __name__ == "__main__":
    airfoil = pd.read_csv(os.path.join(BASE, "airfoil+self+noise", "airfoil_self_noise.dat"),
                          sep=r"\s+", header=None)
    ccpp = pd.read_excel(os.path.join(BASE, "combined+cycle+power+plant", "CCPP", "Folds5x2_pp.xlsx"))
    superc = pd.read_csv(os.path.join(BASE, "superconductivty+data", "train.csv"))
 
    resolver("Airfoil", airfoil, [32, 16], alpha=0.01, epocas=600, tam_lote=32)
    resolver("Power Plant", ccpp, [16, 8], alpha=0.01, epocas=150, tam_lote=64)
    resolver("Superconductividad", superc, [64, 32], alpha=0.01, epocas=100, tam_lote=64)
 
    plt.show()
 