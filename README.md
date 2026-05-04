![alt text](img/ml_hospital_satisfaction.png)
# Predicción de Satisfacción del Paciente en Urgencias Hospitalarias

> Modelo de Machine Learning para predecir el nivel de satisfacción del paciente (escala 1-5) a partir de variables operativas de la visita a urgencias.

---

## <img src="img/hospital_data_icon.png" width="40" height="40"> Descripción del proyecto

Las urgencias hospitalarias son uno de los puntos de mayor presión del sistema sanitario. La experiencia del paciente depende de múltiples factores: tiempo de espera, nivel de urgencia, disponibilidad de personal y recursos del centro.

Este proyecto desarrolla un modelo de clasificación multiclase supervisada que predice el nivel de satisfacción del paciente (1 a 5) usando datos operativos de la visita, sin necesidad de encuestas post-atención.

**Tipo de problema:** Clasificación multiclase supervisada (5 clases)  
**Modelo final:** Logistic Regression  
**Balanced Accuracy:** 81%  

---

## <img src="img/estructura_proyecto.png" width="40" height="40"> Estructura del proyecto

```
Espera_Emergencia_ML/
│
├── data/
│   ├── raw/                          # Dataset original de Kaggle
│   │   └── ER_Wait_Time_Dataset.csv
│   ├── processed/                    # Datos tras limpieza y feature engineering
│   │   ├── train_processed.csv
│   │   └── test_processed.csv
│   ├── train/                        # Split de entrenamiento
│   │   └── train.csv
│   └── test/                         # Split de evaluación
│       └── test.csv
│
├── notebooks/
│   ├── 01_Fuentes.ipynb              # Adquisición y descripción del dataset
│   ├── 02_LimpiezaEDA.ipynb          # Limpieza, EDA y feature engineering
│   ├── 03_Entrenamiento_Evaluacion_Superv.ipynb    # 6 modelos supervisados
│   └── 04_Entrenamiento_Evaluacion_No_Superv.ipynb # KMeans no supervisado
│
├── src/
│   ├── data_processing.py            # Funciones de limpieza y transformación
│   ├── training.py                   # Funciones de entrenamiento y guardado
│   └── evaluation.py                 # Funciones de evaluación y métricas
│
├── models/
│   ├── trained_model_1_lr.pkl        # Logistic Regression (GridSearch)
│   ├── trained_model_2_svm.pkl       # SVM (GridSearch)
│   ├── trained_model_3_dt.pkl        # Decision Tree
│   ├── trained_model_4_rf.pkl        # Random Forest
│   ├── trained_model_5_gb.pkl        # Gradient Boosting
│   ├── trained_model_6_cat.pkl       # CatBoost (GridSearch)
│   ├── trained_model_kmeans.pkl      # KMeans (No supervisado)
│   └── final_model.pkl               # Modelo final (Logistic Regression)
│
├── app_streamlit/
│   ├── img/                          # Imágenes de la aplicación
│   ├── app.py                        # Aplicación web Streamlit
│   └── requirements.txt              # Dependencias del proyecto
│
├── docs/
│   ├── Presentacion_negocio.pptx     # Presentación de negocio
│   └── Presentacion_tecnica.pptx     # Presentación técnica
│
├── img/                              # Imágenes del README
│
└── README.md
```

---

## <img src="img/dataset.png" width="40" height="40"> Dataset

- **Fuente:** [Kaggle — ER Wait Time Dataset](https://www.kaggle.com/datasets/rivalytics/er-wait-time)
- **Registros:** 5.000 visitas hospitalarias
- **Período:** 2024 (datos simulados)
- **Target:** `Patient Satisfaction` (escala 1-5)

### Variables finales para el modelo

| Variable | Tipo | Descripción |
|---|---|---|
| `Nurse-to-Patient Ratio` | Numérica | Ratio enfermeras por paciente (1-5) |
| `Time to Registration (min)` | Numérica | Minutos hasta registro |
| `Time to Triage (min)` | Numérica | Minutos hasta triaje |
| `Time to Medical Professional (min)` | Numérica | Minutos hasta atención médica |
| `Urgency Level` | Ordinal | Low < Medium < High < Critical |
| `Time of Day` | Ordinal | Early Morning → Night |
| `categoria_espera` | Ordinal | Tramos de espera 1-4 (feature engineering) |
| `Season` | Nominal | Estación del año |
| `Day of Week` | Nominal | Día de la semana |

---

## <img src="img/metodologia.png" width="40" height="40"> Metodología

### 1. Limpieza y Feature Engineering
- Eliminación de identificadores, variables con data leakage y variables con correlación casi nula con el target
- Creación de `categoria_espera`: variable ordinal que agrupa el tiempo de espera hasta médico en 4 tramos
- Split train/test estratificado (80/20)

### 2. Preprocesamiento (Pipeline)
- **StandardScaler** → variables numéricas
- **OrdinalEncoder** → variables con orden natural
- **OneHotEncoder** → variables nominales sin orden

### 3. Modelos supervisados evaluados
1. Logistic Regression ⭐ (modelo final)
2. SVM
3. Decision Tree
4. Random Forest
5. Gradient Boosting
6. CatBoost

### 4. Estrategia de selección
```
CV inicial (6 modelos) → Top 3 → GridSearchCV → Evaluación en X_test → Modelo final
```

### 5. Modelo no supervisado
- **KMeans** con K=5 clusters
- Validación con Silhouette Score y visualización PCA

---

## <img src="img/modelo_ml.png" width="40" height="40"> Resultados

### Comparativa Top 3 modelos (evaluación sobre X_test)

| Modelo | Balanced Accuracy | F1 Macro | Recall Macro | Precision Macro |
|---|---|---|---|---|
| **Logistic Regression** ⭐ | **0.8097** | **0.7659** | **0.8097** | **0.7638** |
| SVM | 0.7979 | 0.7565 | 0.7979 | 0.7535 |
| CatBoost | 0.7962 | 0.7504 | 0.7962 | 0.7512 |

### Classification Report — Modelo final

| Clase | Precision | Recall | F1-Score |
|---|---|---|---|
| 1 (Muy insatisfecho) | 0.99 | 0.94 | 0.96 ✅ |
| 2 | 0.71 | 0.88 | 0.79 ✅ |
| 3 | 0.74 | 0.79 | 0.77 ✅ |
| 4 | 0.87 | 0.53 | 0.66 ⚠️ |
| 5 (Muy satisfecho) | 0.51 | 0.90 | 0.65 ✅ |

---

## <img src="img/instalacion_uso.png" width="40" height="40"> Instalación y uso

### Requisitos
```bash
uv add pandas numpy scikit-learn catboost matplotlib seaborn streamlit pillow
```

### Ejecutar el proyecto
```bash
# 1. Clonar el repositorio
git clone https://github.com/nadia0207/ML_Espera_Emergencia.git
cd ML_Espera_Emergencia

# 2. Ejecutar notebooks en orden
jupyter notebook notebooks/01_Fuentes.ipynb
jupyter notebook notebooks/02_LimpiezaEDA.ipynb
jupyter notebook notebooks/03_Entrenamiento_Evaluacion_Superv.ipynb
jupyter notebook notebooks/04_Entrenamiento_Evaluacion_No_Superv.ipynb
```

### Usar el modelo final
```python
import pickle
import pandas as pd

# Cargar modelo
with open('models/final_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predecir
y_pred = model.predict(X_test)
```

---

## <img src="img/streamlit_icon.png" width="40" height="40"> Demo Streamlit

### Ejecutar la aplicación web
```bash
cd app_streamlit
streamlit run app.py
```

La aplicación permite:
- **Predicción individual** — introduce los datos de un paciente y obtén su nivel de satisfacción predicho (1-5)
- **Predicción masiva por CSV** — sube un archivo CSV con múltiples pacientes y descarga los resultados

### Estructura del CSV de entrada

| Columna | Ejemplo |
|---|---|
| `Urgency Level` | Low / Medium / High / Critical |
| `Day of Week` | Monday / Tuesday / Wednesday... |
| `Time of Day` | Early Morning / Afternoon / Evening... |
| `Season` | Winter / Spring / Summer / Fall |
| `Nurse-to-Patient Ratio` | 1 - 5 |
| `Time to Registration (min)` | 10 |
| `Time to Triage (min)` | 20 |
| `Time to Medical Professional (min)` | 30 |

---

## <img src="img/hallazgos_claves.png" width="40" height="40">  Hallazgos principales

- **A mayor tiempo de espera → menor satisfacción** (correlación -0.80 a -0.86)
- **Los pacientes críticos están más satisfechos** — son atendidos primero por el triaje
- **El 89% de pacientes con urgencia baja tienen satisfacción nivel 1** — esperan más tiempo
- **KMeans encontró 5 grupos naturales** que coinciden con los niveles de satisfacción reales

---

## <img src="img/limitaciones_actuales.png" width="40" height="40"> Limitaciones

- Dataset sintético — no son datos reales de un hospital
- 5.000 registros es un volumen limitado
- No incluye variables clínicas del paciente (diagnóstico, edad, motivo de consulta)
- La clase 4 tiene rendimiento menor (recall 53%)

---

## <img src="img/proximos_pasos.png" width="40" height="40"> Próximos pasos

- Integrar datos reales de un hospital
- Añadir variables clínicas
- Ampliar a predicción de tiempos de espera

---


## <img src="img/autora.png" width="40" height="40"> Autor

**Nadia Llamoca Cordova** — Bootcamp Data Science · TheBridge · Mayo 2026
