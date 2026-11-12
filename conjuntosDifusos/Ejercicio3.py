
import numpy as np
import skfuzzy as fuzz
from composicion import composicion, t_norma_minimo, imprimir_matriz

R = [[1, 0],
     [0, 1]]

S = [[1, 0, 0],
     [1, 0, 1]]

imprimir_matriz("R", R)
imprimir_matriz("S", S)

R_compuesta_S = composicion(R, S, t_norma_minimo)
imprimir_matriz("R o S (max-min)", R_compuesta_S)

# Comprobacion: R es la matriz identidad, asi que R o S debe ser igual a S
if R_compuesta_S == S:
    print("Comprobado: R o S es igual a S (porque R es la identidad).")
else:
    print("Algo salio mal: el resultado no es igual a S.")

# Paso a paso del primer elemento, para entenderlo mejor
print()
print("Ejemplo del calculo de (R o S)(u1, w1):")
print("  max( min(R[u1][v1], S[v1][w1]), min(R[u1][v2], S[v2][w1]) )")
print("  max( min(1, 1), min(0, 1) ) = max(1, 0) = 1")

# Comparamos con la funcion de scikit-fuzzy (necesita arreglos de numpy)
resultado_skfuzzy = fuzz.maxmin_composition(np.array(R), np.array(S))
print()
imprimir_matriz("R o S con fuzz.maxmin_composition", resultado_skfuzzy)
print("Nuestro resultado es igual al de skfuzzy?", np.allclose(R_compuesta_S, resultado_skfuzzy))
