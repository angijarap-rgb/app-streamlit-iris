import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Iris AI · Clasificador de flores", page_icon="🌸", layout="wide")

COLORES = {"setosa": "#FF6FB5", "versicolor": "#9B7BFF", "virginica": "#22D3EE"}
EMOJIS = {"setosa": "🌸", "versicolor": "🪻", "virginica": "🌺"}
DESCRIPCIONES = {
    "setosa": "Pétalos pequeños y delicados. Es la más fácil de distinguir de las tres.",
    "versicolor": "Tamaño intermedio en todo. Se parece un poco a la virginica.",
    "virginica": "La más grande de la familia, con pétalos largos y anchos.",
}
EJEMPLOS = {
    "setosa": (5.0, 3.4, 1.5, 0.2),
    "versicolor": (5.9, 2.8, 4.3, 1.3),
    "virginica": (6.6, 3.0, 5.5, 2.0),
}

# ---------- ESTILO ----------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');

html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

.stApp {
    background: radial-gradient(circle at 15% 10%, #3b1d6e 0%, transparent 40%),
                radial-gradient(circle at 85% 20%, #0e4a6b 0%, transparent 40%),
                radial-gradient(circle at 50% 100%, #5a1a58 0%, transparent 45%),
                #0b0620;
    color: #EDEBFF;
}
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1200px; }

.hero { text-align: center; padding: 1.5rem 0 0.5rem 0; }
.hero h1 {
    font-size: 3.4rem; font-weight: 800; margin-bottom: 0.2rem;
    background: linear-gradient(90deg, #FF6FB5, #9B7BFF, #22D3EE);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { font-size: 1.1rem; color: #B9B3E6; font-weight: 300; }

.card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 22px; padding: 1.4rem 1.6rem;
    backdrop-filter: blur(14px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}
.card h3 { margin-top: 0; font-weight: 600; }

.resultado {
    border-radius: 22px; padding: 1.6rem; text-align: center;
    box-shadow: 0 0 40px var(--glow);
    border: 1px solid var(--color);
    background: linear-gradient(145deg, rgba(255,255,255,0.08), rgba(255,255,255,0.02));
}
.resultado .emoji { font-size: 4.5rem; line-height: 1; }
.resultado .nombre {
    font-size: 2.4rem; font-weight: 800; color: var(--color);
    text-transform: capitalize; margin: 0.3rem 0;
}
.resultado .desc { color: #CFCAF5; font-weight: 300; }
.confianza { font-size: 0.95rem; color: #B9B3E6; margin-top: 0.6rem; }
.confianza b { color: var(--color); font-size: 1.3rem; }

.vacio { text-align: center; padding: 3rem 1rem; color: #B9B3E6; }
.vacio .grande { font-size: 4rem; }

div.stButton > button {
    width: 100%; border-radius: 14px; border: 1px solid rgba(255,255,255,0.2);
    background: rgba(255,255,255,0.08); color: #EDEBFF; font-weight: 600;
    padding: 0.6rem 0; transition: all .25s;
}
div.stButton > button:hover {
    border-color: #FF6FB5; color: #fff; transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(255,111,181,0.35);
}
div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #FF6FB5, #9B7BFF 55%, #22D3EE);
    border: none; color: white; font-size: 1.1rem; padding: 0.8rem 0;
}
div.stButton > button[kind="primary"]:hover {
    box-shadow: 0 8px 28px rgba(155,123,255,0.6); transform: translateY(-2px) scale(1.01);
}

button[data-baseweb="tab"] { font-weight: 600; }
.pie { text-align: center; color: #8F89BF; font-size: 0.85rem; margin-top: 2.5rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------- MODELO ----------
@st.cache_resource
def cargar():
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df["especie"] = df["target"].map(dict(enumerate(iris.target_names)))
    modelo = RandomForestClassifier(n_estimators=200, random_state=42)
    modelo.fit(iris.data, iris.target)
    return modelo, list(iris.target_names), df, list(iris.data.columns)


modelo, clases, df, columnas = cargar()

# ---------- ESTADO ----------
valores_iniciales = {"sl": 5.1, "sw": 3.5, "pl": 1.4, "pw": 0.2}
for k, v in valores_iniciales.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("predicho", False)


def cargar_ejemplo(especie):
    sl, sw, pl, pw = EJEMPLOS[especie]
    st.session_state.update({"sl": sl, "sw": sw, "pl": pl, "pw": pw, "predicho": True})


def predecir():
    st.session_state["predicho"] = True


# ---------- ENCABEZADO ----------
st.markdown(
    """
<div class="hero">
<h1>🌸 Iris AI</h1>
<p>Clasificador inteligente de flores · Machine Learning con Random Forest</p>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")

izq, der = st.columns([1, 1], gap="large")

# ---------- COLUMNA IZQUIERDA: ENTRADAS ----------
with izq:
    st.markdown('<div class="card"><h3>🎛️ Medidas de tu flor</h3></div>', unsafe_allow_html=True)
    st.write("")
    st.caption("Prueba con una flor de ejemplo:")
    b1, b2, b3 = st.columns(3)
    b1.button("🌸 Setosa", on_click=cargar_ejemplo, args=("setosa",))
    b2.button("🪻 Versicolor", on_click=cargar_ejemplo, args=("versicolor",))
    b3.button("🌺 Virginica", on_click=cargar_ejemplo, args=("virginica",))

    st.write("")
    sepal_length = st.slider("Longitud del sépalo (cm)", 4.0, 8.0, step=0.1, key="sl")
    sepal_width = st.slider("Ancho del sépalo (cm)", 2.0, 4.5, step=0.1, key="sw")
    petal_length = st.slider("Longitud del pétalo (cm)", 1.0, 7.0, step=0.1, key="pl")
    petal_width = st.slider("Ancho del pétalo (cm)", 0.1, 2.5, step=0.1, key="pw")

    st.write("")
    st.button("✨ Predecir especie", type="primary", on_click=predecir)

# ---------- CÁLCULO ----------
datos = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]], columns=columnas
)
probs = modelo.predict_proba(datos)[0]
idx = int(probs.argmax())
especie = clases[idx]
color = COLORES[especie]

# ---------- COLUMNA DERECHA: RESULTADO ----------
with der:
    if st.session_state["predicho"]:
        st.markdown(
            f"""
<div class="resultado" style="--color:{color}; --glow:{color}55;">
<div class="emoji">{EMOJIS[especie]}</div>
<div class="nombre">Iris {especie}</div>
<div class="desc">{DESCRIPCIONES[especie]}</div>
<div class="confianza">Confianza del modelo: <b>{probs[idx]*100:.0f}%</b></div>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
<div class="card vacio">
<div class="grande">🔮</div>
<h3>Aquí aparecerá tu resultado</h3>
<p>Ajusta las medidas o elige una flor de ejemplo<br>y presiona <b>Predecir especie</b>.</p>
</div>
""",
            unsafe_allow_html=True,
        )

# ---------- GRÁFICOS ----------
def estilo(fig, alto=400):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#EDEBFF", family="Poppins"),
        height=alto,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


if st.session_state["predicho"]:
    st.write("")
    st.markdown("### 📊 Análisis del resultado")
    tab1, tab2, tab3 = st.tabs(["Probabilidades", "Mapa de flores", "Perfil comparado"])

    with tab1:
        fig = go.Figure(
            go.Bar(
                x=[p * 100 for p in probs],
                y=[f"{EMOJIS[c]} {c}" for c in clases],
                orientation="h",
                marker=dict(color=[COLORES[c] for c in clases]),
                text=[f"{p*100:.0f}%" for p in probs],
                textposition="outside",
            )
        )
        fig.update_xaxes(range=[0, 115], title="Probabilidad (%)", gridcolor="rgba(255,255,255,0.08)")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(estilo(fig, 320), use_container_width=True)

    with tab2:
        fig = px.scatter(
            df,
            x="petal length (cm)",
            y="petal width (cm)",
            color="especie",
            color_discrete_map=COLORES,
            opacity=0.75,
        )
        fig.update_traces(marker=dict(size=10))
        fig.add_trace(
            go.Scatter(
                x=[petal_length],
                y=[petal_width],
                mode="markers",
                name="Tu flor",
                marker=dict(size=26, symbol="star", color="#FFFFFF",
                            line=dict(width=2, color=color)),
            )
        )
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
        st.plotly_chart(estilo(fig, 430), use_container_width=True)
        st.caption("Cada punto es una flor del dataset. La estrella blanca es la tuya.")

    with tab3:
        etiquetas = ["Long. sépalo", "Ancho sépalo", "Long. pétalo", "Ancho pétalo"]
        promedios = df.groupby("especie")[columnas].mean()
        fig = go.Figure()
        for c in clases:
            vals = promedios.loc[c].tolist()
            fig.add_trace(
                go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=etiquetas + [etiquetas[0]],
                    name=f"Promedio {c}",
                    line=dict(color=COLORES[c], width=2, dash="dot"),
                )
            )
        mia = [sepal_length, sepal_width, petal_length, petal_width]
        fig.add_trace(
            go.Scatterpolar(
                r=mia + [mia[0]],
                theta=etiquetas + [etiquetas[0]],
                name="Tu flor",
                fill="toself",
                fillcolor="rgba(255,255,255,0.18)",
                line=dict(color="#FFFFFF", width=3),
            )
        )
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(gridcolor="rgba(255,255,255,0.15)"),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.15)"),
            )
        )
        st.plotly_chart(estilo(fig, 450), use_container_width=True)

st.markdown(
    '<div class="pie">Hecho con ❤️ usando Streamlit · scikit-learn · Plotly</div>',
    unsafe_allow_html=True,
)
