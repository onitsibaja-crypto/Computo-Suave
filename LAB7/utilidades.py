
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from ag_red import MAX_ITER, entrenar, evaluar

AZUL, NARANJA, GRIS = "#2a6fdb", "#e8743b", "#8a8f98"


def guardar(fig, carpeta, nombre):
    os.makedirs(carpeta, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(carpeta, nombre), dpi=150, bbox_inches="tight")
    plt.show()


def describir_hiper(h):
    return (f"capas={h['capas']}, act={h['activacion']}, lr={h['learning_rate']}, "
            f"alpha={h['alpha']}, batch={h['batch_size']}, paciencia={h.get('paciencia', 10)}")


def curva_perdida(modelos: dict, carpeta, nombre="curvas_perdida.png", titulo=""):
    """Pérdida de entrenamiento y accuracy de validación interna por época."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    colores = [GRIS, AZUL, NARANJA]
    for (etiqueta, m), col in zip(modelos.items(), colores[-len(modelos):]):
        ax[0].plot(m.loss_curve_, label=etiqueta, color=col, lw=2)
        if getattr(m, "validation_scores_", None) is not None:
            ax[1].plot(m.validation_scores_, label=etiqueta, color=col, lw=2)
    ax[0].set(title="Pérdida de entrenamiento (log-loss)", xlabel="Época", ylabel="Pérdida")
    ax[1].set(title="Accuracy en validación interna", xlabel="Época", ylabel="Accuracy")
    for a in ax:
        a.legend()
        a.grid(alpha=0.3)
    fig.suptitle(titulo)
    guardar(fig, carpeta, nombre)


def convergencia(historial, base_f1, carpeta, nombre="convergencia_ag.png", titulo=""):
    df = pd.DataFrame(historial)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(df.generacion, df.peor, df.mejor, color=AZUL, alpha=0.12, label="Rango de la población")
    ax.plot(df.generacion, df.promedio, "o-", color=AZUL, alpha=0.6, label="F1 promedio")
    ax.plot(df.generacion, df.mejor_global, "o-", color=NARANJA, lw=2.5, label="Mejor F1 (global)")
    ax.axhline(base_f1, ls="--", color=GRIS, label=f"Red base ({base_f1:.3f})")
    ax.set(xlabel="Generación", ylabel="F1 (validación cruzada)", title=titulo or "Convergencia del AG")
    ax.set_xticks(df.generacion)
    ax.legend()
    ax.grid(alpha=0.3)
    guardar(fig, carpeta, nombre)


def matrices_confusion(modelos: dict, X, y, etiquetas, carpeta, nombre="matrices_confusion.png"):
    fig, ax = plt.subplots(1, len(modelos), figsize=(5.5 * len(modelos), 4.5))
    for a, (etiqueta, m) in zip(np.atleast_1d(ax), modelos.items()):
        ConfusionMatrixDisplay.from_estimator(m, X, y, display_labels=etiquetas,
                                              cmap="Blues", ax=a, colorbar=False)
        a.set_title(etiqueta)
    guardar(fig, carpeta, nombre)


def barras_comparacion(tabla: pd.DataFrame, carpeta, nombre="comparacion_metricas.png", titulo=""):
    metricas = ["f1", "accuracy", "loss"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, met in zip(ax, metricas):
        vals = tabla[met]
        barras = a.bar(tabla.index, vals, color=[GRIS, NARANJA][:len(vals)])
        a.bar_label(barras, fmt="%.4f")
        a.set_title(met.upper() + (" (menor es mejor)" if met == "loss" else ""))
        a.set_ylim(0, max(vals) * 1.2)
        a.grid(axis="y", alpha=0.3)
    fig.suptitle(titulo)
    guardar(fig, carpeta, nombre)


def robustez(hipers: dict, X_tr, y_tr, X_te, y_te, promedio_f1, semillas=(0, 1, 2, 3, 4), max_iter=MAX_ITER):
    """Entrena cada configuración con varias semillas para ver si la mejora es real o suerte."""
    from joblib import Parallel, delayed

    def una_red(etiqueta, h, s):
        m = entrenar(h, X_tr, y_tr, semilla=s, max_iter=max_iter)
        return {"modelo": etiqueta, "semilla": s, **evaluar(m, X_te, y_te, promedio_f1)}

    # se entrenan en paralelo (una red por núcleo del CPU)
    filas = Parallel(n_jobs=-1)(delayed(una_red)(etiqueta, h, s)
                                for etiqueta, h in hipers.items() for s in semillas)
    df = pd.DataFrame(filas)
    resumen = df.groupby("modelo")[["f1", "accuracy", "loss"]].agg(["mean", "std"])
    return df, resumen


def boxplot_robustez(df, carpeta, nombre="robustez_semillas.png", titulo=""):
    fig, ax = plt.subplots(figsize=(6, 4))
    grupos = list(df.modelo.unique())
    data = [df[df.modelo == g].f1 for g in grupos]
    bp = ax.boxplot(data, patch_artist=True)
    ax.set_xticks(range(1, len(grupos) + 1), grupos)
    for patch, col in zip(bp["boxes"], [GRIS, NARANJA]):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    ax.set(ylabel="F1 en test", title=titulo or "F1 con 5 semillas distintas")
    ax.grid(axis="y", alpha=0.3)
    guardar(fig, carpeta, nombre)


def resultados_red(modelo, X_te, y_te, promedio_f1, f1_cv=None):
    """Junta en un diccionario todo lo que se guarda de una red entrenada."""
    return {
        "f1_cv": f1_cv,
        "test": evaluar(modelo, X_te, y_te, promedio_f1),
        "loss_entrenamiento": float(modelo.loss_curve_[-1]),
        "epocas": int(modelo.n_iter_),
        "parametros": int(sum(w.size for w in modelo.coefs_) + sum(b.size for b in modelo.intercepts_)),
    }


def mostrar(nombre, res):
    """Imprime una sola línea con los resultados en el conjunto de prueba."""
    t = res["test"]
    print(f"{nombre:<10} ->  F1: {t['f1']:.3f} | Accuracy: {t['accuracy']:.3f} | "
          f"Pérdida: {t['loss']:.3f} | Parámetros: {res['parametros']}")
