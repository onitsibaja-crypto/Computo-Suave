
def t_norma_minimo(a, b):
    return min(a, b)


def t_norma_producto(a, b):
    return a * b


def composicion(R, S, t_norma):
    filas_R = len(R)        # numero de elementos u
    columnas_R = len(R[0])  # numero de elementos v
    filas_S = len(S)        # tambien debe ser el numero de elementos v
    columnas_S = len(S[0])  # numero de elementos w


    if columnas_R != filas_S:
        print("Error: las columnas de R deben ser iguales a las filas de S")
        return None
    resultado = []
    for i in range(filas_R):
        fila = []
        for j in range(columnas_S):
            fila.append(0)
        resultado.append(fila)

    for u in range(filas_R):
        for w in range(columnas_S):
            valores = []
            for v in range(columnas_R):
                valores.append(t_norma(R[u][v], S[v][w]))
            resultado[u][w] = max(valores)

    return resultado


def imprimir_matriz(nombre, matriz):
    print(nombre + " =")
    for fila in matriz:
        texto = "   [ "
        for valor in fila:
            texto = texto + str(round(valor, 2)).rjust(5) + " "
        texto = texto + "]"
        print(texto)
    print()
