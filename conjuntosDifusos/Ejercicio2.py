
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz


def union(conjunto_a, conjunto_b):
    resultado = []
    for i in range(len(conjunto_a)):
        resultado.append(max(conjunto_a[i], conjunto_b[i]))
    return resultado


def interseccion(conjunto_a, conjunto_b):
    resultado = []
    for i in range(len(conjunto_a)):
        resultado.append(min(conjunto_a[i], conjunto_b[i]))
    return resultado


def complemento(conjunto_a):

    resultado = []
    for i in range(len(conjunto_a)):
        resultado.append(1 - conjunto_a[i])
    return resultado

temperatura = np.linspace(0, 40, 401)

frio = fuzz.trapmf(temperatura, [0, 0, 10, 18])
templado = fuzz.trimf(temperatura, [12, 20, 28])

frio_union_templado = union(frio, templado)
frio_inter_templado = interseccion(frio, templado)
no_frio = complemento(frio)


# ------------------------------------------------------------
# Graficas
# ------------------------------------------------------------
fig, ejes = plt.subplots(3, 1, figsize=(9, 10))

# Union
ejes[0].plot(temperatura, frio, "--", label="A = Frio")
ejes[0].plot(temperatura, templado, "--", label="B = Templado")
ejes[0].fill_between(temperatura, frio_union_templado, alpha=0.4, label="A union B (max)")
ejes[0].set_title("Union")

# Interseccion
ejes[1].plot(temperatura, frio, "--", label="A = Frio")
ejes[1].plot(temperatura, templado, "--", label="B = Templado")
ejes[1].fill_between(temperatura, frio_inter_templado, alpha=0.4, color="green", label="A interseccion B (min)")
ejes[1].set_title("Interseccion")

# Complemento
ejes[2].plot(temperatura, frio, "--", label="A = Frio")
ejes[2].fill_between(temperatura, no_frio, alpha=0.4, color="red", label="Complemento de A (1 - A)")
ejes[2].set_title("Complemento")

for eje in ejes:
    eje.set_xlabel("Grados C")
    eje.set_ylabel("Membresia")
    eje.set_ylim(-0.05, 1.05)
    eje.legend(loc="center right")
    eje.grid(True)

plt.tight_layout()
plt.savefig("ejercicio2_operadores.png")
plt.show()


posicion = 140   # temperatura[140] = 14.0
print("En x =", temperatura[posicion], "grados:")
print("  Frio        =", round(frio[posicion], 3))
print("  Templado    =", round(templado[posicion], 3))
print("  Union       =", round(frio_union_templado[posicion], 3))
print("  Interseccion=", round(frio_inter_templado[posicion], 3))
print("  No frio     =", round(no_frio[posicion], 3))
print()


x_union, union_skfuzzy = fuzz.fuzzy_or(temperatura, frio, temperatura, templado)
x_inter, inter_skfuzzy = fuzz.fuzzy_and(temperatura, frio, temperatura, templado)
complemento_skfuzzy = fuzz.fuzzy_not(frio)

print("Union igual a skfuzzy?        ", np.allclose(frio_union_templado, union_skfuzzy))
print("Interseccion igual a skfuzzy? ", np.allclose(frio_inter_templado, inter_skfuzzy))
print("Complemento igual a skfuzzy?  ", np.allclose(no_frio, complemento_skfuzzy))
