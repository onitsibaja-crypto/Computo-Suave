
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
#temperatura
temperatura = np.linspace(0, 40, 401)

frio = fuzz.trapmf(temperatura, [0, 0, 10, 18])
templado = fuzz.trimf(temperatura, [12, 20, 28])
caliente = fuzz.trapmf(temperatura, [22, 30, 40, 40])


#Velocidad 

velocidad = np.linspace(0, 120, 601)

lenta = fuzz.gaussmf(velocidad, 0, 12)          # centro 0, sigma 12
media = fuzz.gbellmf(velocidad, 15, 3, 45)      # ancho 15, pendiente 3, centro 45
rapida = fuzz.gaussmf(velocidad, 80, 10)        # centro 80, sigma 10
muy_rapida = fuzz.trapmf(velocidad, [90, 105, 120, 120])

#humedad
humedad = np.linspace(0, 100, 501)

seca = fuzz.trimf(humedad, [0, 0, 40])
normal = fuzz.gbellmf(humedad, 15, 2, 50)
humeda = fuzz.gaussmf(humedad, 100, 12)


fig, ejes = plt.subplots(3, 1, figsize=(9, 10))

ejes[0].plot(temperatura, frio, label="Frio (trapezoidal)")
ejes[0].plot(temperatura, templado, label="Templado (triangular)")
ejes[0].plot(temperatura, caliente, label="Caliente (trapezoidal)")
ejes[0].set_title("Variable 1: Temperatura")
ejes[0].set_xlabel("Grados C")

ejes[1].plot(velocidad, lenta, label="Lenta (gaussiana)")
ejes[1].plot(velocidad, media, label="Media (campana)")
ejes[1].plot(velocidad, rapida, label="Rapida (gaussiana)")
ejes[1].plot(velocidad, muy_rapida, label="Muy rapida (trapezoidal)")
ejes[1].set_title("Variable 2: Velocidad")
ejes[1].set_xlabel("km/h")

ejes[2].plot(humedad, seca, label="Seca (triangular)")
ejes[2].plot(humedad, normal, label="Normal (campana)")
ejes[2].plot(humedad, humeda, label="Humeda (gaussiana)")
ejes[2].set_title("Variable 3: Humedad")
ejes[2].set_xlabel("%")

for eje in ejes:
    eje.set_ylabel("Membresia")
    eje.set_ylim(-0.05, 1.05)
    eje.legend(loc="center right")
    eje.grid(True)

plt.tight_layout()
plt.savefig("ejercicio1_variables.png")
plt.show()



valor_temp = 15
print("Temperatura =", valor_temp, "grados")
print("  Frio     :", round(fuzz.interp_membership(temperatura, frio, valor_temp), 3))
print("  Templado :", round(fuzz.interp_membership(temperatura, templado, valor_temp), 3))
print("  Caliente :", round(fuzz.interp_membership(temperatura, caliente, valor_temp), 3))
