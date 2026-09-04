import numpy as np

class Capa:
    def __init__(self, neuronas: int, entradas: int, Usarbias: bool=True,activacion: str=None):
        self.neuronas = neuronas
        self.entradas = entradas
        self.Usarbias = Usarbias
        self.activacion = activacion
        self.pesos = np.random.randn(self.entradas,self.neuronas)
        if Usarbias:
            self.bias=np.random.randn(1,self.neuronas)
        else:
            self.bias=None
    def feed_forward(self, X):
        self.salida = np.dot(X, self.pesos)
        
        if self.bias != None:
            self.salida += self.bias
        
        if self.activacion == "relu":
            self.salida = np.maximum(0, self.salida)            
        return self.salida

if __name__ == "__main__":
    X = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])
    W = np.array([
        [0,   0,   0.1, 0.2],
        [0.3, -1,  0.4, 0.5],
        [1,   0.6, 0.7, 0.8]
    ])
    capa = Capa(neuronas=4, entradas=3, Usarbias=False, activacion="relu")
    capa.pesos = W
    R = capa.feed_forward(X)

    
    print(R)