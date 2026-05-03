import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =====================================================================
# CARGAR MODELO
# =====================================================================
def cargar_modelo(path='../models/final_model.pkl'):
    """
    Carga un modelo guardado en formato pickle.

    Parameters
    ----------
    path : str
        Ruta del modelo a cargar.

    Returns
    -------
    Pipeline
        Modelo cargado.
    """
    with open(path, 'rb') as f:
        modelo = pickle.load(f)
    print(f"Modelo cargado desde: {path}")
    return modelo

# =====================================================================
# MÉTRICAS
# =====================================================================
def calcular_metricas(y_test, y_pred, nombre_modelo='Modelo'):
    """
    Calcula y muestra las métricas de evaluación del modelo.

    Parameters
    ----------
    y_test : pd.Series
        Valores reales del target.
    y_pred : np.array
        Valores predichos por el modelo.
    nombre_modelo : str
        Nombre del modelo para mostrar en el output.

    Returns
    -------
    dict
        Diccionario con las métricas calculadas.
    """
    metricas = {
        'Modelo'             : nombre_modelo,
        'Balanced Accuracy'  : balanced_accuracy_score(y_test, y_pred),
        'F1 Macro'           : f1_score(y_test, y_pred, average='macro'),
        'Recall Macro'       : recall_score(y_test, y_pred, average='macro'),
        'Precision Macro'    : precision_score(y_test, y_pred, average='macro')
    }

    for k, v in metricas.items():
        if k != 'Modelo':
            print(f"{k}: {v}")

    return metricas

# =====================================================================
# COMPARAR MODELOS
# =====================================================================
def comparar_modelos(modelos_dict, X_test, y_test):
    """
    Compara múltiples modelos sobre X_test y devuelve una tabla con 
    las métricas de cada uno ordenada por Balanced Accuracy.

    Parameters
    ----------
    modelos_dict : dict
        Diccionario con nombre y modelo entrenado.
    X_test : pd.DataFrame
        Features de test.
    y_test : pd.Series
        Target de test.

    Returns
    -------
    pd.DataFrame
        Tabla comparativa de métricas ordenada por Balanced Accuracy.
    """
    resultados = []

    for nombre, modelo in modelos_dict.items():
        y_pred = modelo.predict(X_test)
        metricas = calcular_metricas(y_test, y_pred, nombre)
        resultados.append(metricas)

    return pd.DataFrame(resultados).sort_values('Balanced Accuracy', ascending=False)

# =====================================================================
# MATRIZ DE CONFUSIÓN
# =====================================================================
def plot_confusion_matrix(y_test, y_pred, titulo='Matriz de Confusión'):
    """
    Genera y muestra la matriz de confusión.

    Parameters
    ----------
    y_test : pd.Series
        Valores reales del target.
    y_pred : np.array
        Valores predichos por el modelo.
    titulo : str
        Título del gráfico.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=[1, 2, 3, 4, 5],
        cmap='Blues',
        ax=ax
    )
    ax.set_title(titulo)
    plt.tight_layout()
    plt.show()

# =====================================================================

