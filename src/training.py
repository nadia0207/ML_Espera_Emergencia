import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Preprocesamiento
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

# Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier
from sklearn.cluster import KMeans

# Evaluación
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import balanced_accuracy_score

# =====================================================================
# PREPROCESADOR
# =====================================================================
def get_preprocesador(numericas, ordinales, nominales):
    """
    Crea y devuelve el ColumnTransformer con las transformaciones
    necesarias para cada tipo de variable.

    Returns
    -------
    ColumnTransformer
        Preprocesador con StandardScaler para numéricas,
        OrdinalEncoder para ordinales y OneHotEncoder para nominales.
    """
    # Orden para OrdinalEncoder
    urgency_order = [['Low', 'Medium', 'High', 'Critical']]
    time_order = [['Early Morning', 'Late Morning', 'Afternoon', 'Evening', 'Night']]
    espera_order = [[1, 2, 3, 4]]

    preprocesador = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numericas),
        ('ord', OrdinalEncoder(categories=urgency_order + time_order + espera_order), ordinales),
        ('nom', OneHotEncoder(drop='first', sparse_output=False), nominales)
    ])
    return preprocesador

# =====================================================================
# PIPELINES
# =====================================================================
def get_pipelines(preprocesador):
    """
    Crea y devuelve los pipelines de todos los modelos.

    Parameters
    ----------
    preprocesador : ColumnTransformer
        Preprocesador creado con get_preprocesador()

    Returns
    -------
    dict
        Diccionario con los pipelines de cada modelo.
    """
    pipelines = {
        'Logistic Regression': Pipeline(steps=[
            ('preprocesador', preprocesador),
            ('modelo', LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                solver='saga',
                l1_ratio=1,
                random_state=42))
        ]),
        'SVM': Pipeline(steps=[
            ('preprocesador', preprocesador),
            ('modelo', SVC(
                class_weight='balanced',
                probability=True,
                random_state=42))
        ]),
        'Decision Tree': Pipeline(steps=[
            ('preprocesador', preprocesador),
            ('modelo', DecisionTreeClassifier(
                class_weight='balanced',
                random_state=42))
        ]),
        'Random Forest': Pipeline(steps=[
            ('preprocesador', preprocesador),
            ('modelo', RandomForestClassifier(
                class_weight='balanced',
                n_jobs=-1,
                random_state=42))
        ]),
        'GradientBoosting': Pipeline(steps=[
            ('preprocesador', preprocesador),
            ('modelo', GradientBoostingClassifier(
                random_state=42))
        ]),
        'CatBoost': Pipeline(steps=[
            ('preprocesador', preprocesador),
            ('modelo', CatBoostClassifier(
                auto_class_weights='Balanced',
                verbose=0,
                random_state=42))
        ])
    }
    return pipelines

# =====================================================================
# ENTRENAMIENTO CON GRIDSEARCH
# =====================================================================
def entrenar_top3(X_train, y_train, preprocesador):
    """
    Entrena los 3 mejores modelos con GridSearchCV.

    Parameters
    ----------
    X_train : pd.DataFrame
        Features de entrenamiento.
    y_train : pd.Series
        Target de entrenamiento.
        preprocesador : ColumnTransformer
        Preprocesador creado con get_preprocesador()

    Returns
    -------
    dict
        Diccionario con los 3 mejores modelos entrenados.
    """
    pipelines = get_pipelines(preprocesador)

    lr_params = {
        'modelo__C': [0.01, 0.1, 1, 10, 100],
        'modelo__l1_ratio': [0, 0.5, 1],
        'modelo__solver': ['saga']
    }

    svm_params = {
        'modelo__kernel': ['rbf', 'linear'],
        'modelo__C': [0.1, 1, 10, 100]
    }

    cat_params = {
        'modelo__iterations': [200, 300, 400],
        'modelo__learning_rate': [0.01, 0.05, 0.1],
        'modelo__depth': [3, 4, 5],
        'modelo__l2_leaf_reg': [1, 3, 5]
    }

    gs_lr = GridSearchCV(pipelines['Logistic Regression'], lr_params,
                         cv=5, scoring='balanced_accuracy', n_jobs=-1)
    gs_svm = GridSearchCV(pipelines['SVM'], svm_params,
                          cv=5, scoring='balanced_accuracy', n_jobs=-1)
    gs_cat = GridSearchCV(pipelines['CatBoost'], cat_params,
                          cv=5, scoring='balanced_accuracy', n_jobs=-1)

    gs_lr.fit(X_train, y_train)
    gs_svm.fit(X_train, y_train)
    gs_cat.fit(X_train, y_train)

    # Mostrar resultados
    print(f"Mejor score LR:       {gs_lr.best_score_:.4f} | {gs_lr.best_params_}")
    print(f"Mejor score SVM:      {gs_svm.best_score_:.4f}| {gs_svm.best_params_}")
    print(f"Mejor score CatBoost: {gs_cat.best_score_:.4f}| {gs_cat.best_params_}")

    return {
        'Logistic Regression': gs_lr.best_estimator_,
        'SVM': gs_svm.best_estimator_,
        'CatBoost': gs_cat.best_estimator_
    }, {
        'Logistic Regression': gs_lr.best_score_,
        'SVM': gs_svm.best_score_,
        'CatBoost': gs_cat.best_score_
    }

# =====================================================================
# GUARDAR MODELO FINAL
# =====================================================================
def guardar_modelo_final(modelo, path='../models/final_model.pkl'):
    """
    Guarda el modelo final en formato pickle.

    Parameters
    ----------
    modelo : Pipeline
        Modelo final entrenado.
    path : str
        Ruta donde se guardará el modelo final.
    """
    with open(path, 'wb') as f:
        pickle.dump(modelo, f)
    print(f"Modelo final guardado: {path}")

# =====================================================================
# KMEANS — NO SUPERVISADO
# =====================================================================
def grafico_codo_inercia(K, inercias, silhouettes):  
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Método del codo
    axes[0].plot(K, inercias, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Número de Clusters (K)')
    axes[0].set_ylabel('Inercia')
    axes[0].set_title('Método del Codo')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].annotate(
        'Codo',
        xy=(4, inercias[2]),      # ← apunta a K=4
        xytext=(6, 20000),        # posición del texto
        fontsize=13,
        color='red',
        arrowprops=dict(facecolor='red', shrink=0.1)
    )

    # Silhouette score
    axes[1].plot(K, silhouettes, 'ro-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Número de Clusters (K)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Score por K')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.suptitle('Selección del número óptimo de clusters', fontsize=13)
    plt.tight_layout()
    plt.show()