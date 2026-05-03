import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.cluster import KMeans

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    ConfusionMatrixDisplay,
    silhouette_score
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
# EVALUACIÓN KMEANS — NO SUPERVISADO
# =====================================================================
def entrenar_kmeans(X_prep, n_clusters=5):
    """
    Entrena un modelo KMeans con el número de clusters especificado.

    Parameters
    ----------
    X_prep : np.array
        Datos preprocesados y escalados.
    n_clusters : int, optional
        Número de clusters. Por defecto 5.

    Returns
    -------
    KMeans
        Modelo KMeans entrenado.
    """   
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_prep)
    print(f"KMeans entrenado con {n_clusters} clusters")
    print(f"Inercia: {kmeans.inertia_:.0f}")
    return kmeans

# =====================================================================
# SCATTERPLO KMEANS — NO SUPERVISADO
# =====================================================================

def scatterplot_kmeas(X_pca,df_full,pca):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Gráfico 1 — coloreado por cluster KMeans
    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=df_full['cluster'],
        palette='viridis',
        alpha=0.5,
        s=15,
        ax=axes[0]
    )
    axes[0].set_title('Clusters KMeans')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} varianza)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} varianza)')

    # Gráfico 2 — coloreado por satisfacción real
    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=df_full['Patient Satisfaction'],
        palette='viridis',
        alpha=0.5,
        s=15,
        ax=axes[1]
    )
    axes[1].set_title('Niveles de Satisfacción reales')
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} varianza)')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} varianza)')

    plt.suptitle('KMeans vs Satisfacción Real — Visualización PCA', fontsize=13)
    plt.tight_layout()
    plt.show()