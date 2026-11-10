

import json
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CARPETA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CARPETA)

import ag_red
import utilidades as ut

SEMILLA = 42
random.seed(SEMILLA)
np.random.seed(SEMILLA)
SALIDA = os.path.join(CARPETA, "resultados", "wine")
os.makedirs(SALIDA, exist_ok=True)
sns.set_theme(style="whitegrid")

tinto = pd.read_csv(os.path.join(CARPETA, "datos", "winequality-red.csv"), sep=";")
blanco = pd.read_csv(os.path.join(CARPETA, "datos", "winequality-white.csv"), sep=";")
tinto["tipo"] = "tinto"
blanco["tipo"] = "blanco"
df = pd.concat([tinto, blanco], ignore_index=True)
df.columns = [c.replace(" ", "_") for c in df.columns]

print(f"Datos cargados: {df.shape[0]} vinos y {df.shape[1]} columnas")

variables = list(df.columns.drop(["quality", "tipo"]))   
descripcion = df.describe().T                          
correlaciones = df[variables + ["quality"]].corr()

print("Valores faltantes:", df.isna().sum().sum())
print("Filas duplicadas:", df.duplicated().sum())
print("Vinos por calificación:", df.quality.value_counts().sort_index().to_dict())
print("Variable más relacionada con la calidad: alcohol",
      f"(correlación = {correlaciones.loc['alcohol', 'quality']:.2f})")

fig, ax = plt.subplots(figsize=(7, 4))
sns.countplot(data=df, x="quality", hue="tipo", palette=["#8e1b3a", "#e6c86e"], ax=ax)
ax.set_title("Calificación de los vinos (0 a 10)")
ut.guardar(fig, SALIDA, "eda_objetivo.png")

fig, axes = plt.subplots(3, 4, figsize=(15, 9))
for ax, col in zip(axes.flat, variables):
    sns.histplot(data=df, x=col, hue="tipo", palette=["#8e1b3a", "#e6c86e"],
                 element="step", legend=False, ax=ax)
axes.flat[-1].axis("off")
fig.suptitle("Distribución de las variables (rojo = tinto, amarillo = blanco)")
ut.guardar(fig, SALIDA, "eda_histogramas.png")

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlaciones, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            annot_kws={"size": 7}, ax=ax)
ax.set_title("Matriz de correlación")
ut.guardar(fig, SALIDA, "eda_correlacion.png")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["alcohol", "volatile_acidity", "density"]):
    sns.boxplot(data=df, x="quality", y=col, color=ut.AZUL, ax=ax)
    ax.set_title(f"{col} vs calidad")
ut.guardar(fig, SALIDA, "eda_boxplots.png")


datos = df.drop_duplicates().reset_index(drop=True)

datos["clase"] = pd.cut(datos.quality, bins=[0, 5, 6, 10], labels=[0, 1, 2]).astype(int)
ETIQUETAS = ["baja (≤5)", "media (6)", "alta (≥7)"]

datos["es_tinto"] = (datos.tipo == "tinto").astype(int)

X = datos[variables + ["es_tinto"]].values
y = datos["clase"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEMILLA)

escala = StandardScaler().fit(X_train)
X_train = escala.transform(X_train)
X_test = escala.transform(X_test)

print(f"Vinos sin duplicados: {len(datos)}  ->  train: {len(X_train)} | test: {len(X_test)}")

PROMEDIO_F1 = "macro"

aptitud = ag_red.AptitudRedNeuronal(X_train, y_train, promedio_f1=PROMEDIO_F1, folds=3)

print("Red base:", ut.describir_hiper(ag_red.HIPER_BASE))
cv_base = aptitud.evaluar_hiperparametros([ag_red.HIPER_BASE])[0]["f1"]
modelo_base = ag_red.entrenar(ag_red.HIPER_BASE, X_train, y_train, semilla=SEMILLA)
res_base = ut.resultados_red(modelo_base, X_test, y_test, PROMEDIO_F1, cv_base)
print(f"F1 validación cruzada (red base): {cv_base:.3f}")
ut.mostrar("Red base", res_base)

parametros_ag = {
    "Aptitud": aptitud,
    "Seleccion": ag_red.SeleccionTorneo(participantes=3),
    "Reproduccion": ag_red.ReproduccionUnCruce(),
    "Mutacion": ag_red.MutacionPorGen(0.2),
    "por_remplazo": 0.5,   # la mitad mejor pasa a la siguiente generación
    "elite": 2,            # los 2 mejores pasan sin cambios
}
algoritmoGA = ag_red.GA(num_inv=16, hyperparams=parametros_ag, incluir=[ag_red.GENES_BASE])
mejor = algoritmoGA.buscar_solucion(max_generaciones=12, paciencia=5)
HIPER_AG = mejor.decodificar()

print("Mejor red encontrada:", ut.describir_hiper(HIPER_AG))

top = pd.DataFrame(algoritmoGA.top(10))      # ver en el Explorador de variables
top.to_csv(os.path.join(SALIDA, "top_configuraciones.csv"), index=False)
ut.convergencia(algoritmoGA.historial, cv_base, SALIDA,
                titulo="Wine Quality - convergencia del AG")

modelo_ag = ag_red.entrenar(HIPER_AG, X_train, y_train, semilla=SEMILLA)
res_ag = ut.resultados_red(modelo_ag, X_test, y_test, PROMEDIO_F1, mejor.score)

print("\nCOMPARACIÓN EN PRUEBA")
print(f"F1 validación cruzada ->  Base: {cv_base:.3f} | AG: {mejor.score:.3f}")
ut.mostrar("Red base", res_base)
ut.mostrar("Red AG", res_ag)

tabla = pd.DataFrame({"Base": res_base["test"], "AG": res_ag["test"]}).T
tabla["parametros"] = [res_base["parametros"], res_ag["parametros"]]
tabla.to_csv(os.path.join(SALIDA, "comparacion.csv"))

modelos = {"Base": modelo_base, "AG": modelo_ag}
ut.barras_comparacion(tabla, SALIDA, titulo="Wine Quality - red base vs red optimizada")
ut.curva_perdida(modelos, SALIDA, titulo="Wine Quality")
ut.matrices_confusion(modelos, X_test, y_test, ETIQUETAS, SALIDA)

semillas_df, robustez = ut.robustez({"Base": ag_red.HIPER_BASE, "AG": HIPER_AG},
                                    X_train, y_train, X_test, y_test, PROMEDIO_F1)
ut.boxplot_robustez(semillas_df, SALIDA, titulo="Wine Quality - F1 con 5 semillas")
for nombre in ["Base", "AG"]:
    print(f"F1 promedio con 5 semillas ({nombre}): "
          f"{robustez.loc[nombre, ('f1', 'mean')]:.3f} ± {robustez.loc[nombre, ('f1', 'std')]:.3f}")

resultados = {
    "dataset": "Wine Quality",
    "base": {"hiper": ut.describir_hiper(ag_red.HIPER_BASE), **res_base},
    "ag": {"hiper": ut.describir_hiper(HIPER_AG), "genes": mejor.genes, **res_ag},
    "redes_entrenadas_ag": aptitud.n_entrenamientos,
    "historial_ag": algoritmoGA.historial,
    "robustez_5_semillas": {n: {"f1_media": float(robustez.loc[n, ("f1", "mean")]),
                                "f1_std": float(robustez.loc[n, ("f1", "std")])}
                            for n in ["Base", "AG"]},
}
with open(os.path.join(SALIDA, "resultados.json"), "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)
print("\nListo. Resultados guardados en la carpeta resultados/wine")
