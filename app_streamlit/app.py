import streamlit as st
import pickle
import pandas as pd
import base64
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# ── Configuración ────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
icon = Image.open(os.path.join(BASE, "img", "modelo_ml.png"))
st.set_page_config(
    page_title="Satisfacción en Urgencias",
    page_icon=icon,
    layout="wide"
)


# ── Función para cargar imágenes en base64 ───────────────────────────
@st.cache_resource
def img_to_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

# ── Rutas de imágenes (relativas al app.py) ──────────────────────────

IMGS = {
    1: img_to_base64(os.path.join(BASE, "img", "muy_insatisfecho.png")),
    2: img_to_base64(os.path.join(BASE, "img", "insatisfecho.png")),
    3: img_to_base64(os.path.join(BASE, "img", "neutra.png")),
    4: img_to_base64(os.path.join(BASE, "img", "satisfecho.png")),
    5: img_to_base64(os.path.join(BASE, "img", "muy_satisfecho.png")),
}
IMG_HOSPITAL  = img_to_base64(os.path.join(BASE, "img", "datos_hospital_v2.png"))
IMG_TIEMPOS   = img_to_base64(os.path.join(BASE, "img", "tiempos_espera.png"))
IMG_SATISF    = img_to_base64(os.path.join(BASE, "img", "satisfaccion_paciente_v2.png"))
IMG_BG        = img_to_base64(os.path.join(BASE, "img", "ml_hospital_satisfaction.png"))


# ── CSS global ───────────────────────────────────────────────────────
st.markdown(f"""
<style>
    /* Fondo principal con imagen */
    .stApp {{
        background-image: linear-gradient(rgba(5, 30, 60, 0.82), rgba(5, 30, 60, 0.82)),
                          url("data:image/png;base64,{IMG_BG}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    /* Texto general blanco */
    html, body, [class*="css"] {{
        color: #E8F4FD;
        font-family: 'Segoe UI', sans-serif;
    }}

    /* Cards de sección */
    .card {{
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
    }}

    .card-title {{
        font-size: 30px;
        font-weight: 700;
        color: #02C39A;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 10px;
    }}

    /* Sliders y selects */
    .stSlider > div > div > div > div {{
        background: #02C39A !important;
    }}

    /* Botón principal */
    .stButton > button {{
        background: linear-gradient(135deg, #02C39A, #065A82);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 14px 32px;
        font-size: 18px;
        font-weight: 700;
        width: 100%;
        cursor: pointer;
        transition: all 0.3s;
        box-shadow: 0 4px 20px rgba(2, 195, 154, 0.4);
    }}
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(2, 195, 154, 0.6);
    }}

    /* Resultado */
    .result-box {{
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
        backdrop-filter: blur(15px);
        border: 2px solid;
    }}

    /* Métricas CSV */
    .metric-card {{
        background: rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.2);
    }}
    .metric-val {{
        font-size: 36px;
        font-weight: 800;
        color: #02C39A;
    }}
    .metric-label {{
        font-size: 13px;
        color: #A8D8EA;
        margin-top: 4px;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: #A8D8EA;
        font-weight: 600;
    }}
    .stTabs [aria-selected="true"] {{
        background: rgba(2,195,154,0.2) !important;
        color: #02C39A !important;
    }}

    /* Labels de inputs */
    label {{
        color: #A8D8EA !important;
        font-weight: 500 !important;
    }}

    /* Ocultar footer Streamlit */
    footer {{ visibility: hidden; }}
    #MainMenu {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

# ── Cargar modelo ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = os.path.join(BASE, "..", "models", "final_model.pkl")
    with open(path, 'rb') as f:
        return pickle.load(f)

model = load_model()

# ── Header ───────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 30px 0 10px 0'>
    <h1 style='color:white; font-size:42px; font-weight:800; margin:0; 
               text-shadow: 0 2px 20px rgba(2,195,154,0.5)'>
         Predicción de Satisfacción del Paciente
    </h1>
    <p style='color:#A8D8EA; font-size:18px; margin:8px 0 0 0'>
        Urgencias Hospitalarias · Modelo de Machine Learning · Balanced Accuracy 81%
    </p>
</div>
<hr style='border-color:rgba(255,255,255,0.1); margin: 20px 0'>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔮 Predicción individual", "📂 Predicción por CSV"])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICCIÓN INDIVIDUAL
# ════════════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns(2, gap="large")

    with col1:
        # Card datos hospital
        icon_h = f'<img src="data:image/png;base64,{IMG_HOSPITAL}" width="65" style="vertical-align:middle">' if IMG_HOSPITAL else "🏨"
        st.markdown(f'<div class="card"><div class="card-title">{icon_h} &nbsp; Datos del Hospital</div>', unsafe_allow_html=True)
        
        urgency    = st.selectbox("Nivel de urgencia", ['Low', 'Medium', 'High', 'Critical'])
        season     = st.selectbox("Estación del año", ['Winter', 'Spring', 'Summer', 'Fall'])
        day        = st.selectbox("Día de la semana", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        time_day   = st.selectbox("Franja horaria", ['Early Morning', 'Late Morning', 'Afternoon', 'Evening', 'Night'])
        nurse_ratio = st.slider("👩‍⚕️ Ratio enfermeras por paciente", 1, 5, 3)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Card tiempos espera
        icon_t = f'<img src="data:image/png;base64,{IMG_TIEMPOS}" width="65" style="vertical-align:middle">' if IMG_TIEMPOS else "⏱️"
        st.markdown(f'<div class="card"><div class="card-title">{icon_t} &nbsp; Tiempos de Espera</div>', unsafe_allow_html=True)
        reg_time     = st.slider("Tiempo hasta registro (min)", 0, 65, 10)
        triage_time  = st.slider("Tiempo hasta triaje (min)", 0, 160, 20)
        medical_time = st.slider("Tiempo hasta médico (min)", 0, 230, 30)

        # Categoría espera con color
        if medical_time <= 15:
            cat_espera, cat_label, cat_color = 1, "⚡ Corta (0-15 min)", "#02C39A"
        elif medical_time <= 45:
            cat_espera, cat_label, cat_color = 2, "✅ Media (15-45 min)", "#F39C12"
        elif medical_time <= 90:
            cat_espera, cat_label, cat_color = 3, "⚠️ Larga (45-90 min)", "#E67E22"
        else:
            cat_espera, cat_label, cat_color = 4, "🚨 Muy larga (>90 min)", "#C0392B"

        st.markdown(f"""
        <div style='background:{cat_color}22; border-left:4px solid {cat_color}; 
                    padding:12px 16px; border-radius:8px; margin-top:12px'>
            <b style='color:{cat_color}'>Categoría de espera:</b> 
            <span style='color:white'>{cat_label}</span>
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Botón predicción
    st.markdown("<br>", unsafe_allow_html=True)
    col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
    with col_b2:
        predecir = st.button("🔮 Predecir Satisfacción del Paciente")

    # ── Resultado ────────────────────────────────────────────────────
    if predecir:
        datos = pd.DataFrame([{
            'Nurse-to-Patient Ratio'             : nurse_ratio,
            'Time to Registration (min)'          : reg_time,
            'Time to Triage (min)'                : triage_time,
            'Time to Medical Professional (min)'  : medical_time,
            'Urgency Level'                       : urgency,
            'Time of Day'                         : time_day,
            'categoria_espera'                    : cat_espera,
            'Season'                              : season,
            'Day of Week'                         : day
        }])

        prediccion = model.predict(datos)[0]

        labels  = {1: "Muy insatisfecho", 2: "Insatisfecho", 3: "Neutral",
                   4: "Satisfecho",       5: "Muy satisfecho"}
        colores = {1: "#C0392B", 2: "#E67E22", 3: "#F39C12",
                   4: "#27AE60", 5: "#02C39A"}
        color   = colores[prediccion]
        label   = labels[prediccion]
        img_b64 = IMGS.get(prediccion, "")

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 3, 1])
        with c2:
            img_tag = f'<img src="data:image/png;base64,{img_b64}" width="120" style="margin-bottom:10px">' if img_b64 else ""
            st.markdown(f"""
            <div class="result-box" style='border-color:{color}; 
                        background:linear-gradient(135deg, {color}22, {color}11)'>
                {img_tag}
                <h2 style='color:{color}; font-size:36px; margin:10px 0'>
                    Nivel {prediccion} — {label}
                </h2>
                <p style='color:#A8D8EA; font-size:14px; margin:0'>
                    Predicción basada en Logistic Regression · Balanced Accuracy: 81%
                </p>
            </div>""", unsafe_allow_html=True)

        # Barra de 5 niveles
        st.markdown("<br>", unsafe_allow_html=True)
        cols = st.columns(5)
        for i, col in enumerate(cols):
            nivel = i + 1
            img_b64_n = IMGS.get(nivel, "")
            bg  = colores[nivel] if nivel == prediccion else "rgba(255,255,255,0.08)"
            brd = colores[nivel] if nivel == prediccion else "rgba(255,255,255,0.1)"
            img_html = f'<img src="data:image/png;base64,{img_b64_n}" width="50"><br>' if img_b64_n else ""
            col.markdown(f"""
            <div style='background:{bg}; border:2px solid {brd}; text-align:center; 
                        padding:12px 8px; border-radius:12px; transition:all 0.3s'>
                {img_html}
                <b style='color:white'>Nivel {nivel}</b><br>
                <span style='font-size:11px; color:#A8D8EA'>{labels[nivel]}</span>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# TAB 2 — PREDICCIÓN POR CSV
# ════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="card">
        <div class="card-title">📂 Carga tu CSV para predicción masiva</div>
        <p style='color:#A8D8EA; font-size:14px'>
            El CSV debe tener las siguientes columnas:<br>
            <code style='color:#02C39A'>Nurse-to-Patient Ratio, Time to Registration (min), 
            Time to Triage (min), Time to Medical Professional (min), 
            Urgency Level, Time of Day, Season, Day of Week</code>
        </p>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Sube tu archivo CSV", type=["csv"])

    if uploaded is not None:
        # Detectar separador automáticamente
        contenido = uploaded.read()
        uploaded.seek(0)  # resetear para poder leerlo de nuevo
        primera_linea = contenido.decode('utf-8').split('\n')[0]
        sep = ';' if ';' in primera_linea else ','
        df_csv = pd.read_csv(uploaded, sep=sep)
        st.markdown(f"**{len(df_csv)} registros cargados**")

        # Calcular categoria_espera automáticamente
        def calc_cat(t):
            if t <= 15: return 1
            elif t <= 45: return 2
            elif t <= 90: return 3
            else: return 4

        df_csv['categoria_espera'] = df_csv['Time to Medical Professional (min)'].apply(calc_cat)

        # Columnas necesarias
        cols_modelo = ['Nurse-to-Patient Ratio', 'Time to Registration (min)',
                       'Time to Triage (min)', 'Time to Medical Professional (min)',
                       'Urgency Level', 'Time of Day', 'categoria_espera',
                       'Season', 'Day of Week']

        try:
            X_csv = df_csv[cols_modelo]
            predicciones = model.predict(X_csv)

            labels  = {1: "Muy insatisfecho", 2: "Insatisfecho", 3: "Neutral",
                       4: "Satisfecho",       5: "Muy satisfecho"}
            colores_hex = {1: "#C0392B", 2: "#E67E22", 3: "#F39C12",
                           4: "#27AE60", 5: "#02C39A"}

            df_csv['Satisfacción Predicha'] = predicciones
            df_csv['Nivel'] = df_csv['Satisfacción Predicha'].map(labels)

            # ── Resumen estadístico ──────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="card-title">📊 Resumen de predicciones</div>', unsafe_allow_html=True)

            conteo = df_csv['Satisfacción Predicha'].value_counts().sort_index()
            total  = len(df_csv)

            cols_m = st.columns(5)
            for i, col in enumerate(cols_m):
                nivel = i + 1
                n     = conteo.get(nivel, 0)
                pct   = round(n / total * 100, 1)
                color = colores_hex[nivel]
                img_b64_n = IMGS.get(nivel, "")
                img_html  = f'<img src="data:image/png;base64,{img_b64_n}" width="45"><br>' if img_b64_n else ""
                col.markdown(f"""
                <div class="metric-card" style='border-top: 4px solid {color}'>
                    {img_html}
                    <div class="metric-val" style='color:{color}'>{n}</div>
                    <div class="metric-label">{labels[nivel]}</div>
                    <div style='font-size:12px; color:#A8D8EA'>{pct}%</div>
                </div>""", unsafe_allow_html=True)

            # ── Gráfico de distribución ──────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8, 3.5))
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')

            niveles = [1, 2, 3, 4, 5]
            valores = [conteo.get(n, 0) for n in niveles]
            colores_list = [colores_hex[n] for n in niveles]

            bars = ax.bar([labels[n] for n in niveles], valores, color=colores_list,
                          edgecolor='white', linewidth=0.5, width=0.6)
            for bar, val in zip(bars, valores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        str(val), ha='center', va='bottom', color='white', fontweight='bold')

            ax.set_ylabel('Nº Pacientes', color='white')
            ax.tick_params(colors='white')

            ax.spines['bottom'].set_color((1, 1, 1, 0.2))
            ax.spines['left'].set_color((1, 1, 1, 0.2))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_title('Distribución de Satisfacción Predicha', color='white', fontweight='bold')
            plt.xticks(rotation=15, ha='right')
            plt.tight_layout()
            st.pyplot(fig, transparent=True)

            # ── Tabla de resultados ──────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="card-title">📋 Tabla de resultados por paciente</div>', unsafe_allow_html=True)
            st.dataframe(
                
                df_csv[['Urgency Level', 'Day of Week','Season','Time of Day','Time to Triage (min)',
                        'Nurse-to-Patient Ratio','Time to Registration (min)','Time to Medical Professional (min)',
                        'Satisfacción Predicha', 'Nivel']].reset_index(drop=True),

                use_container_width=True,
                height=350
            )

            # ── Botón descarga ───────────────────────────────────────
            csv_out = df_csv.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Descargar resultados completos",
                data=csv_out,
                file_name="predicciones_satisfaccion.csv",
                mime="text/csv"
            )

        except KeyError as e:
            st.error(f"Columna no encontrada en el CSV: {e}")

# ── Footer ────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:20px 0; margin-top:20px; 
            border-top:1px solid rgba(255,255,255,0.1)'>
    <p style='color:#4A6FA5; font-size:12px; margin:0'>
        Bootcamp Data Science · TheBridge · Mayo 2026 · 
        Modelo: Logistic Regression · Balanced Accuracy: 81%
    </p>
</div>""", unsafe_allow_html=True)