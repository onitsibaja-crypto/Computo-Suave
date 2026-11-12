
import numpy as np
import skfuzzy as fuzz
from composicion import (composicion, t_norma_producto,
                         t_norma_minimo, imprimir_matriz)

# Filas: u1, u2   Columnas: v1, v2, v3
mu_C = [[0.9, 0.4, 0.1],
        [0.1, 0.4, 0.9]]

# Filas: v1, v2, v3   Columnas: w1, w2
mu_MB = [[0.0, 0.0],
         [0.6, 0.0],
         [1.0, 0.7]]

imprimir_matriz("uC(u, v)", mu_C)
imprimir_matriz("uMB(v, w)", mu_MB)

# Composicion con T-norma = producto
resultado_producto = composicion(mu_C, mu_MB, t_norma_producto)
imprimir_matriz("uC o MB (u, w)  con T-norma PRODUCTO", resultado_producto)

# Paso a paso de cada elemento (para el reporte)
nombres_u = ["u1", "u2"]
nombres_w = ["w1", "w2"]
print("Desarrollo paso a paso:")
for u in range(2):
    for w in range(2):
        productos = []
        texto = "  (" + nombres_u[u] + "," + nombres_w[w] + ") = max("
        for v in range(3):
            producto = mu_C[u][v] * mu_MB[v][w]
            productos.append(producto)
            texto = texto + str(mu_C[u][v]) + "*" + str(mu_MB[v][w])
            if v < 2:
                texto = texto + ", "
        texto = texto + ") = max("
        texto = texto + str(round(productos[0], 2)) + ", "
        texto = texto + str(round(productos[1], 2)) + ", "
        texto = texto + str(round(productos[2], 2)) + ")"
        texto = texto + " = " + str(round(max(productos), 2))
        print(texto)
print()

# Comprobamos con la funcion de scikit-fuzzy
resultado_skfuzzy = fuzz.maxprod_composition(np.array(mu_C), np.array(mu_MB))
imprimir_matriz("Con fuzz.maxprod_composition", resultado_skfuzzy)
print("Nuestro resultado es igual al de skfuzzy?", np.allclose(resultado_producto, resultado_skfuzzy))
print()

# Para comparar: la misma composicion con T-norma = minimo (la de la clase)
resultado_minimo = composicion(mu_C, mu_MB, t_norma_minimo)
imprimir_matriz("uC o MB (u, w)  con T-norma MINIMO (para comparar)", resultado_minimo)

