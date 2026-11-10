
#%% 0. Librerías y configuración
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
SALIDA = os.path.join(CARPETA, "resultados", "bank")
os.makedirs(SALIDA, exist_ok=True)
sns.set_theme(style="whitegrid")

#%% 1. Cargar los datos
df = pd.read_csv(os.path.join(CARPETA, "datos", "bank-full.csv"), sep=";")
print(f"Datos cargados: {df.shape[0]} clientes y {df.shape[1]} columnas")

#%% 2. Análisis exploratorio (EDA)
CATEGORICAS = ["job", "marital", "education", "default", "housing", "loan",
               "contact", "month", "poutcome"]
NUMERICAS = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]

descripcion = df[NUMERICAS].describe().T                  # ver en el Explorador de variables
desconocidos = (df[CATEGORICAS] == "unknown").sum()       # "unknown" = dato faltante
correlaciones = df[NUMERICAS].assign(y=(df.y == "yes").astype(int)).corr()

print("Valores faltantes:", df.isna().sum().sum(), "| pero hay valores 'unknown' en:",
      list(desconocidos[desconocidos > 0].index))
print(f"Clientes que contrataron (y = yes): {(df.y == 'yes').mean():.1%}  ->  datos desbalanceados")

fig, ax = plt.subplots(figsize=(5, 4))
conteo = df.y.value_counts()
ax.bar(conteo.index, conteo.values, color=[ut.GRIS, ut.NARANJA])
ax.set_title("¿El cliente contrató el depósito? (y)")
ut.guardar(fig, SALIDA, "eda_objetivo.png")

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, col in zip(axes.flat, NUMERICAS):
    sns.boxplot(data=df, x="y", y=col, hue="y", palette=[ut.GRIS, ut.NARANJA],
                showfliers=False, legend=False, ax=ax)
    ax.set_title(col)
axes.flat[-1].axis("off")
ut.guardar(fig, SALIDA, "eda_numericas.png")

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
for ax, col in zip(axes.flat, CATEGORICAS):
    tasa = (df.y == "yes").groupby(df[col]).mean().sort_values()
    tasa.plot.barh(color=ut.AZUL, ax=ax)
    ax.axvline((df.y == "yes").mean(), ls="--", color=ut.NARANJA)
    ax.set_title(f"% que contrata según {col}")
fig.suptitle("Tasa de contratación por categoría (línea = promedio general)")
ut.guardar(fig, SALIDA, "eda_categoricas.png")


fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(correlaciones, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
ax.set_title("Correlación entre variables numéricas y la respuesta")
ut.guardar(fig, SALIDA, "eda_correlacion.png")

datos = df.copy()


datos = datos.drop(columns=["duration"])

datos["contactado_antes"] = (datos.pdays != -1).astype(int)
datos.loc[datos.pdays == -1, "pdays"] = 0

y = (datos.pop("y") == "yes").astype(int).values

X = pd.get_dummies(datos, columns=CATEGORICAS, dtype=float).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEMILLA)


escala = StandardScaler().fit(X_train)
X_train = escala.transform(X_train)
X_test = escala.transform(X_test)


idx_bal = ag_red.balancear(np.arange(len(y_train)), y_train, SEMILLA)
X_bal, y_bal = X_train[idx_bal], y_train[idx_bal]

print(f"Variables tras one-hot: {X.shape[1]}")
print(f"train: {len(X_train)} (balanceado: {len(X_bal)}) | test: {len(X_test)}")


PROMEDIO_F1 = "binary"
ETIQUETAS = ["no", "sí"]


aptitud = ag_red.AptitudRedNeuronal(X_train, y_train, promedio_f1=PROMEDIO_F1, folds=3,
                                    balancear_clases=True)


print("Red base:", ut.describir_hiper(ag_red.HIPER_BASE))
cv_base = aptitud.evaluar_hiperparametros([ag_red.HIPER_BASE])[0]["f1"]
modelo_base = ag_red.entrenar(ag_red.HIPER_BASE, X_bal, y_bal, semilla=SEMILLA)
res_base = ut.resultados_red(modelo_base, X_test, y_test, PROMEDIO_F1, cv_base)
print(f"F1 validación cruzada (red base): {cv_base:.3f}")
ut.mostrar("Red base", res_base)

#%% 5. Algoritmo genético
parametros_ag = {
    "Aptitud": aptitud,
    "Seleccion": ag_red.SeleccionTorneo(participantes=3),
    "Reproduccion": ag_red.ReproduccionUnCruce(),
    "Mutacion": ag_red.MutacionPorGen(0.2),
    "por_remplazo": 0.5,   
    "elite": 2,            
}
algoritmoGA = ag_red.GA(num_inv=16, hyperparams=parametros_ag, incluir=[ag_red.GENES_BASE])
mejor = algoritmoGA.buscar_solucion(max_generaciones=12, paciencia=5)
HIPER_AG = mejor.decodificar()

print("Mejor red encontrada:", ut.describir_hiper(HIPER_AG))

top = pd.DataFrame(algoritmoGA.top(10))      # ver en el Explorador de variables
top.to_csv(os.path.join(SALIDA, "top_configuraciones.csv"), index=False)
ut.convergencia(algoritmoGA.historial, cv_base, SALIDA,
                titulo="Bank Marketing - convergencia del AG")

modelo_ag = ag_red.entrenar(HIPER_AG, X_bal, y_bal, semilla=SEMILLA)
res_ag = ut.resultados_red(modelo_ag, X_test, y_test, PROMEDIO_F1, mejor.score)

print("\nCOMPARACIÓN EN PRUEBA")
print(f"F1 validación cruzada ->  Base: {cv_base:.3f} | AG: {mejor.score:.3f}")
ut.mostrar("Red base", res_base)
ut.mostrar("Red AG", res_ag)

tabla = pd.DataFrame({"Base": res_base["test"], "AG": res_ag["test"]}).T
tabla["parametros"] = [res_base["parametros"], res_ag["parametros"]]
tabla.to_csv(os.path.join(SALIDA, "comparacion.csv"))

modelos = {"Base": modelo_base, "AG": modelo_ag}
ut.barras_comparacion(tabla, SALIDA, titulo="Bank Marketing - red base vs red optimizada")
ut.curva_perdida(modelos, SALIDA, titulo="Bank Marketing")
ut.matrices_confusion(modelos, X_test, y_test, ETIQUETAS, SALIDA)

semillas_df, robustez = ut.robustez({"Base": ag_red.HIPER_BASE, "AG": HIPER_AG},
                                    X_bal, y_bal, X_test, y_test, PROMEDIO_F1)
ut.boxplot_robustez(semillas_df, SALIDA, titulo="Bank Marketing - F1 con 5 semillas")
for nombre in ["Base", "AG"]:
    print(f"F1 promedio con 5 semillas ({nombre}): "
          f"{robustez.loc[nombre, ('f1', 'mean')]:.3f} ± {robustez.loc[nombre, ('f1', 'std')]:.3f}")


resultados = {
    "dataset": "Bank Marketing",
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
print("\nListo. Resultados guardados en la carpeta resultados/bank")
