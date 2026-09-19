import numpy as np


def optimizador(W, gradW, alpha):

    return W - alpha * gradW


class Capa:
    def __init__(self, neuronas: int, entradas: int, Usarbias: bool = True, activacion: str = None):
        self.neuronas = neuronas
        self.entradas = entradas
        self.Usarbias = Usarbias
        self.activacion = activacion
        self.pesos = np.random.randn(self.entradas, self.neuronas)
        if Usarbias:
            self.bias = np.random.randn(1, self.neuronas)
        else:
            self.bias = None

    def feed_forward(self, X):
        self.entrada = X  
        self.z = np.dot(X, self.pesos)

        if self.bias is not None:  
            self.z = self.z + self.bias

        if self.activacion == "relu":
            self.salida = np.maximum(0, self.z)
        else:
            self.salida = self.z

        return self.salida

    def derivada_activacion(self):
        if self.activacion == "relu":
            return (self.z > 0).astype(float)
        else:
            return np.ones_like(self.z)  

    def backward(self, es_ultima, error_derivada=None, delta_siguiente=None, W_siguiente=None):
        if es_ultima:
            delta = error_derivada * self.derivada_activacion()
        else:
            delta = np.dot(delta_siguiente, W_siguiente.T)

        self.delta = delta
        self.grad_pesos = np.dot(delta.T, self.entrada).T
        if self.bias is not None:
            self.grad_bias = np.sum(delta, axis=0, keepdims=True)

        return delta

if __name__ == "__main__":
    X = np.ones((3, 2)) * 5
    y = np.zeros((3, 2))

    capa1 = Capa(neuronas=4, entradas=2, Usarbias=True, activacion="relu")
    capa2 = Capa(neuronas=2, entradas=4, Usarbias=True, activacion=None)

    a1 = capa1.feed_forward(X)
    y_pred = capa2.feed_forward(a1)
    print("y_pred =\n", y_pred)

    de_dyhat = y_pred - y
    delta2 = capa2.backward(es_ultima=True, error_derivada=de_dyhat)
    delta1 = capa1.backward(es_ultima=False, delta_siguiente=delta2, W_siguiente=capa2.pesos)

    print("delta2 =\n", delta2)
    print("gradiente W capa2 =\n", capa2.grad_pesos)
    print("delta1 =\n", delta1)
    print("gradiente W capa1 =\n", capa1.grad_pesos)