import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Iris AI · Plataforma de flores", page_icon="🌸", layout="wide")

# =====================================================
# CONSTANTES
# =====================================================
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
SIMPLES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
ETIQ = ["Long. sépalo", "Ancho sépalo", "Long. pétalo", "Ancho pétalo"]

# nombre: (etiqueta del parámetro, mínimo, máximo, valor por defecto)
MODELOS = {
    "Random Forest": ("Número de árboles", 10, 300, 200),
    "K-Vecinos (KNN)": ("Vecinos (k)", 1, 25, 5),
    "SVM": ("Parámetro C", 1, 20, 3),
    "Regresión logística": ("Parámetro C", 1, 20, 5),
    "Árbol de decisión": ("Profundidad máxima", 1, 10, 4),
}
PAGINAS = [
    "🔮 Predictor",
    "📊 Explorador de datos",
    "🧪 Laboratorio de IA",
    "📁 Predicción por lotes",
    "🎮 Humano vs IA",
]

# =====================================================
# ESTILO
# =====================================================
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
.block-container { padding-top: 2rem; max-width: 1250px; }

section[data-testid="stSidebar"] {
    background: rgba(11, 6, 32, 0.92);
    border-right: 1px solid rgba(255,255,255,0.1);
}

.hero { text-align: center; padding: 1rem 0 0.5rem 0; }
.hero h1 {
    font-size: 3.2rem; font-weight: 800; margin-bottom: 0.2rem;
    background: linear-gradient(90deg, #FF6FB5, #9B7BFF, #22D3EE);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { font-size: 1.1rem; color: #B9B3E6; font-weight: 300; }

.card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 22px; padding: 1.2rem 1.6rem;
    backdrop-filter: blur(14px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}
.card h3 { margin: 0; font-weight: 600; }

.resultado {
    border-radius: 22px; padding: 1.6rem; text-align: center;
    box-shadow: 0 0 40px var(--glow);
    border: 1px solid var(--color);
    background: linear-gradient(145deg, rgba(255,255,255,0.08), rgba(255,255,255,0.02));
}
.resultado .emoji { font-size: 4.5rem; line-height: 1; }
.resultado .nombre {
    font-size: 2.3rem; font-weight: 800; color: var(--color);
    text-transform: capitalize; margin: 0.3rem 0;
}
.resultado .desc { color: #CFCAF5; font-weight: 300; }
.confianza { font-size: 0.95rem; color: #B9B3E6; margin-top: 0.6rem; }
.confianza b { color: var(--color); font-size: 1.3rem; }

.vacio { text-align: center; padding: 3rem 1rem; color: #B9B3E6; }
.vacio .grande { font-size: 4rem; }

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px; padding: 0.8rem 1rem;
}

div.stButton > button, div.stDownloadButton > button {
    width: 100%; border-radius: 14px; border: 1px solid rgba(255,255,255,0.2);
    background: rgba(255,255,255,0.08); color: #EDEBFF; font-weight: 600;
    padding: 0.6rem 0; transition: all .25s;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
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

# =====================================================
# DATOS Y MODELOS
# =====================================================
@st.cache_data
def cargar_datos():
    iris = load_iris(as_frame=True)
    datos = iris.frame.copy()
    datos["especie"] = datos["target"].map(dict(enumerate(iris.target_names)))
    return datos, list(iris.target_names), list(iris.data.columns)


df, clases, columnas = cargar_datos()


def construir(nombre, p):
    p = int(p)
    if nombre == "Random Forest":
        return RandomForestClassifier(n_estimators=p, random_state=42)
    if nombre == "K-Vecinos (KNN)":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=p))
    if nombre == "SVM":
        return make_pipeline(StandardScaler(), SVC(C=p, probability=True, random_state=42))
    if nombre == "Regresión logística":
        return make_pipeline(StandardScaler(), LogisticRegression(C=p, max_iter=500))
    return DecisionTreeClassifier(max_depth=p, random_state=42)


@st.cache_resource
def entrenar(nombre, p):
    modelo_ = construir(nombre, p)
    modelo_.fit(df[columnas], df["target"])
    return modelo_


@st.cache_data
def evaluar(nombre, p, test, semilla):
    X, y = df[columnas], df["target"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test, stratify=y, random_state=semilla)
    m = construir(nombre, p).fit(Xtr, ytr)
    pred = m.predict(Xte)
    acc = accuracy_score(yte, pred)
    cm = confusion_matrix(yte, pred)
    imp = permutation_importance(m, Xte, yte, n_repeats=10, random_state=semilla).importances_mean
    cv = cross_val_score(construir(nombre, p), X, y, cv=5)
    return acc, cm, np.clip(imp, 0, None), cv.mean(), cv.std(), len(Xte)


@st.cache_data
def comparar_todos():
    filas = []
    for nombre, (_, _, _, defecto) in MODELOS.items():
        sc = cross_val_score(construir(nombre, defecto), df[columnas], df["target"], cv=5)
        filas.append(
            {"Algoritmo": nombre, "Precisión media (%)": round(sc.mean() * 100, 2),
             "Desviación (%)": round(sc.std() * 100, 2)}
        )
    return pd.DataFrame(filas).sort_values("Precisión media (%)", ascending=False)


# =====================================================
# BARRA LATERAL
# =====================================================
with st.sidebar:
    st.markdown("## 🌸 Iris AI")
    st.caption("Plataforma de Machine Learning")
    pagina = st.radio("Navegación", PAGINAS, label_visibility="collapsed")
    st.divider()
    st.markdown("#### 🤖 Motor de IA")
    modelo_nombre = st.selectbox("Algoritmo", list(MODELOS.keys()), key="modelo_sel")
    etiqueta_p, p_min, p_max, p_def = MODELOS[modelo_nombre]
    p_val = st.slider(etiqueta_p, p_min, p_max, p_def, key=f"p_{modelo_nombre}")
    st.caption("Este motor se usa en todas las secciones.")

modelo = entrenar(modelo_nombre, p_val)

# =====================================================
# UTILIDADES
# =====================================================
def estilo(fig, alto=400):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#EDEBFF", family="Poppins"),
        height=alto,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    return fig


def hero(titulo, subtitulo):
    st.markdown(
        f'<div class="hero"><h1>{titulo}</h1><p>{subtitulo}</p></div>',
        unsafe_allow_html=True,
    )


def csv_bytes(tabla):
    return tabla.to_csv(index=False).encode("utf-8")


# =====================================================
# PÁGINA 1: PREDICTOR
# =====================================================
def cargar_ejemplo(especie):
    sl, sw, pl, pw = EJEMPLOS[especie]
    st.session_state.update({"sl": sl, "sw": sw, "pl": pl, "pw": pw, "predicho": True})


def borrar_historial():
    st.session_state["historial"] = []


def pagina_predictor():
    hero("🔮 Predictor", f"Clasifica una flor con {modelo_nombre}")
    s = st.session_state
    for k, v in {"sl": 5.1, "sw": 3.5, "pl": 1.4, "pw": 0.2}.items():
        s.setdefault(k, v)
    s.setdefault("predicho", False)
    s.setdefault("historial", [])

    izq, der = st.columns(2, gap="large")
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
        clic = st.button("✨ Predecir especie", type="primary")

    datos = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=columnas)
    probs = modelo.predict_proba(datos)[0]
    idx = int(probs.argmax())
    especie = clases[idx]
    color = COLORES[especie]

    if clic:
        s["predicho"] = True
        s["historial"].append(
            {
                "sepal_length": sepal_length,
                "sepal_width": sepal_width,
                "petal_length": petal_length,
                "petal_width": petal_width,
                "especie": especie,
                "confianza_%": round(float(probs[idx]) * 100, 1),
                "modelo": modelo_nombre,
            }
        )

    with der:
        if s["predicho"]:
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

    if s["predicho"]:
        st.write("")
        st.markdown("### 📊 Análisis del resultado")
        t1, t2, t3, t4, t5 = st.tabs(
            ["Probabilidades", "Mapa 2D", "Mapa 3D", "Perfil comparado", "Flores parecidas"]
        )

        with t1:
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
            fig.update_xaxes(range=[0, 115], title="Probabilidad (%)")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(estilo(fig, 320))

        with t2:
            c1, c2 = st.columns(2)
            ex = c1.selectbox("Eje X", columnas, index=2, key="p2x")
            ey = c2.selectbox("Eje Y", columnas, index=3, key="p2y")
            fig = px.scatter(df, x=ex, y=ey, color="especie", color_discrete_map=COLORES, opacity=0.75)
            fig.update_traces(marker=dict(size=10))
            fig.add_trace(
                go.Scatter(
                    x=[datos[ex].iloc[0]], y=[datos[ey].iloc[0]], mode="markers", name="Tu flor",
                    marker=dict(size=26, symbol="star", color="#FFFFFF", line=dict(width=2, color=color)),
                )
            )
            st.plotly_chart(estilo(fig, 430))

        with t3:
            fig = px.scatter_3d(
                df, x=columnas[2], y=columnas[3], z=columnas[0],
                color="especie", color_discrete_map=COLORES, opacity=0.8,
            )
            fig.update_traces(marker=dict(size=4))
            fig.add_trace(
                go.Scatter3d(
                    x=[petal_length], y=[petal_width], z=[sepal_length], mode="markers", name="Tu flor",
                    marker=dict(size=10, color="#FFFFFF", symbol="diamond"),
                )
            )
            st.plotly_chart(estilo(fig, 520))
            st.caption("Arrastra para girar el gráfico. El diamante blanco es tu flor.")

        with t4:
            promedios = df.groupby("especie")[columnas].mean()
            fig = go.Figure()
            for c in clases:
                vals = promedios.loc[c].tolist()
                fig.add_trace(
                    go.Scatterpolar(
                        r=vals + [vals[0]], theta=ETIQ + [ETIQ[0]], name=f"Promedio {c}",
                        line=dict(color=COLORES[c], width=2, dash="dot"),
                    )
                )
            mia = [sepal_length, sepal_width, petal_length, petal_width]
            fig.add_trace(
                go.Scatterpolar(
                    r=mia + [mia[0]], theta=ETIQ + [ETIQ[0]], name="Tu flor", fill="toself",
                    fillcolor="rgba(255,255,255,0.18)", line=dict(color="#FFFFFF", width=3),
                )
            )
            fig.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(gridcolor="rgba(255,255,255,0.15)"),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.15)"),
                )
            )
            st.plotly_chart(estilo(fig, 450))

        with t5:
            dist = np.sqrt(((df[columnas].values - datos.values) ** 2).sum(axis=1))
            cerca = df.assign(distancia=dist).nsmallest(5, "distancia")
            tabla = cerca[columnas + ["especie", "distancia"]].round(2)
            tabla.columns = ETIQ + ["Especie", "Distancia"]
            st.write("Las 5 flores reales del dataset más parecidas a la tuya:")
            st.dataframe(tabla, hide_index=True)

    if s["historial"]:
        st.write("")
        with st.expander(f"🕘 Historial de predicciones ({len(s['historial'])})"):
            hist = pd.DataFrame(s["historial"])
            st.dataframe(hist, hide_index=True)
            h1, h2 = st.columns(2)
            h1.download_button("⬇️ Descargar historial (CSV)", csv_bytes(hist), "historial_iris.csv", "text/csv")
            h2.button("🗑️ Borrar historial", on_click=borrar_historial)


# =====================================================
# PÁGINA 2: EXPLORADOR
# =====================================================
def pagina_explorador():
    hero("📊 Explorador de datos", "Descubre los patrones escondidos en el dataset Iris")
    sel = st.multiselect("Especies a mostrar", clases, default=clases)
    if not sel:
        st.warning("Elige al menos una especie para ver los datos.")
        return
    d = df[df["especie"].isin(sel)]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Flores", len(d))
    m2.metric("Especies", len(sel))
    m3.metric("Long. pétalo promedio", f"{d[columnas[2]].mean():.2f} cm")
    m4.metric("Ancho pétalo promedio", f"{d[columnas[3]].mean():.2f} cm")
    st.write("")

    t1, t2, t3, t4, t5 = st.tabs(["Tabla", "Distribuciones", "Relaciones", "Correlación", "Proyección PCA"])

    with t1:
        tabla = d[columnas + ["especie"]]
        st.dataframe(tabla, hide_index=True)
        st.download_button("⬇️ Descargar datos filtrados (CSV)", csv_bytes(tabla), "iris_filtrado.csv", "text/csv")
        st.markdown("**Resumen estadístico**")
        st.dataframe(d[columnas].describe().round(2))

    with t2:
        var = st.selectbox("Variable", columnas, index=2, key="var_dist")
        c1, c2 = st.columns(2)
        fig = px.histogram(d, x=var, color="especie", barmode="overlay", opacity=0.7,
                           nbins=20, color_discrete_map=COLORES)
        c1.plotly_chart(estilo(fig, 380))
        fig = px.box(d, x="especie", y=var, color="especie", points="all", color_discrete_map=COLORES)
        c2.plotly_chart(estilo(fig, 380))

    with t3:
        fig = px.scatter_matrix(d, dimensions=columnas, color="especie", color_discrete_map=COLORES, opacity=0.7)
        fig.update_traces(diagonal_visible=False, marker=dict(size=4))
        st.plotly_chart(estilo(fig, 650))
        st.markdown("**Vista 3D personalizada**")
        a, b, c = st.columns(3)
        x3 = a.selectbox("Eje X", columnas, index=0, key="x3")
        y3 = b.selectbox("Eje Y", columnas, index=2, key="y3")
        z3 = c.selectbox("Eje Z", columnas, index=3, key="z3")
        fig = px.scatter_3d(d, x=x3, y=y3, z=z3, color="especie", color_discrete_map=COLORES, opacity=0.85)
        fig.update_traces(marker=dict(size=4))
        st.plotly_chart(estilo(fig, 520))

    with t4:
        corr = d[columnas].corr()
        fig = px.imshow(
            corr.values, x=ETIQ, y=ETIQ, text_auto=".2f", zmin=-1, zmax=1,
            color_continuous_scale=["#22D3EE", "#1b1140", "#FF6FB5"],
        )
        st.plotly_chart(estilo(fig, 480))
        st.caption("Cerca de 1 significa que las dos medidas crecen juntas. Cerca de -1, que van en sentido contrario.")

    with t5:
        pca = PCA(n_components=2).fit(df[columnas])
        proy = pca.transform(d[columnas])
        pdf = pd.DataFrame({"PC1": proy[:, 0], "PC2": proy[:, 1], "especie": d["especie"].values})
        fig = px.scatter(pdf, x="PC1", y="PC2", color="especie", color_discrete_map=COLORES, opacity=0.8)
        fig.update_traces(marker=dict(size=10))
        st.plotly_chart(estilo(fig, 450))
        var_exp = pca.explained_variance_ratio_.sum() * 100
        st.caption(f"PCA reduce las 4 medidas a 2 ejes conservando el {var_exp:.1f}% de la información.")


# =====================================================
# PÁGINA 3: LABORATORIO
# =====================================================
def pedir_competencia():
    st.session_state["comp"] = True


def pagina_laboratorio():
    hero("🧪 Laboratorio de IA", f"Evalúa {modelo_nombre} ({etiqueta_p}: {p_val})")
    c1, c2 = st.columns(2)
    test_pct = c1.slider("% de datos para prueba", 10, 50, 25, 5, key="test_pct")
    semilla = c2.number_input("Semilla aleatoria", 0, 999, 42, key="semilla")

    with st.spinner("Entrenando y evaluando..."):
        acc, cm, imp, cv_media, cv_std, n_test = evaluar(modelo_nombre, p_val, test_pct / 100, int(semilla))

    m1, m2, m3 = st.columns(3)
    m1.metric("Precisión en prueba", f"{acc*100:.1f}%", help=f"Evaluado con {n_test} flores que el modelo no vio.")
    m2.metric("Validación cruzada (5 folds)", f"{cv_media*100:.1f}%")
    m3.metric("Variación entre folds", f"± {cv_std*100:.1f}%")
    st.write("")

    t1, t2, t3, t4 = st.tabs(
        ["Matriz de confusión", "Importancia de variables", "Frontera de decisión", "Competencia de algoritmos"]
    )

    with t1:
        fig = px.imshow(
            cm, x=clases, y=clases, text_auto=True,
            labels=dict(x="Predicho", y="Real", color="Flores"),
            color_continuous_scale=["#1b1140", "#9B7BFF", "#FF6FB5"],
        )
        st.plotly_chart(estilo(fig, 430))
        st.caption("La diagonal son los aciertos. Todo lo que está fuera de la diagonal son errores del modelo.")

    with t2:
        orden = np.argsort(imp)
        fig = go.Figure(
            go.Bar(
                x=imp[orden], y=[ETIQ[i] for i in orden], orientation="h",
                marker=dict(color=imp[orden], colorscale=["#22D3EE", "#9B7BFF", "#FF6FB5"]),
            )
        )
        fig.update_xaxes(title="Importancia (caída de precisión al mezclar la variable)")
        st.plotly_chart(estilo(fig, 380))

    with t3:
        a, b = st.columns(2)
        fx = a.selectbox("Variable X", columnas, index=2, key="fx")
        fy = b.selectbox("Variable Y", columnas, index=3, key="fy")
        if fx == fy:
            st.warning("Elige dos variables distintas.")
        else:
            X2 = df[[fx, fy]]
            m2 = construir(modelo_nombre, p_val).fit(X2, df["target"])
            xs = np.linspace(X2[fx].min() - 0.3, X2[fx].max() + 0.3, 120)
            ys = np.linspace(X2[fy].min() - 0.3, X2[fy].max() + 0.3, 120)
            xx, yy = np.meshgrid(xs, ys)
            grid = pd.DataFrame({fx: xx.ravel(), fy: yy.ravel()})
            zz = m2.predict(grid).reshape(xx.shape)
            fig = go.Figure(
                go.Heatmap(
                    x=xs, y=ys, z=zz, zmin=0, zmax=2, showscale=False, opacity=0.35, hoverinfo="skip",
                    colorscale=[[0, COLORES[clases[0]]], [0.5, COLORES[clases[1]]], [1, COLORES[clases[2]]]],
                )
            )
            for c in clases:
                sub = df[df["especie"] == c]
                fig.add_trace(
                    go.Scatter(
                        x=sub[fx], y=sub[fy], mode="markers", name=c,
                        marker=dict(size=9, color=COLORES[c], line=dict(width=1, color="#FFFFFF")),
                    )
                )
            fig.update_xaxes(title=fx)
            fig.update_yaxes(title=fy)
            st.plotly_chart(estilo(fig, 500))
            st.caption("Los colores de fondo muestran qué especie predeciría el modelo en cada zona.")

    with t4:
        st.write("Compara los 5 algoritmos con sus parámetros por defecto usando validación cruzada.")
        st.button("🏁 Iniciar competencia", on_click=pedir_competencia)
        if st.session_state.get("comp"):
            with st.spinner("Compitiendo..."):
                tabla = comparar_todos()
            ganador = tabla.iloc[0]
            st.success(f"🏆 Ganador: **{ganador['Algoritmo']}** con {ganador['Precisión media (%)']}%")
            fig = px.bar(tabla, x="Algoritmo", y="Precisión media (%)", error_y="Desviación (%)",
                         color="Precisión media (%)",
                         color_continuous_scale=["#22D3EE", "#9B7BFF", "#FF6FB5"])
            fig.update_yaxes(range=[80, 100])
            st.plotly_chart(estilo(fig, 400))
            st.dataframe(tabla, hide_index=True)


# =====================================================
# PÁGINA 4: PREDICCIÓN POR LOTES
# =====================================================
def pagina_lotes():
    hero("📁 Predicción por lotes", "Sube un CSV con muchas flores y clasifícalas todas de una vez")
    plantilla = df.sample(8, random_state=1)[columnas].copy()
    plantilla.columns = SIMPLES

    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        st.markdown('<div class="card"><h3>1️⃣ Descarga la plantilla</h3></div>', unsafe_allow_html=True)
        st.write("")
        st.download_button("⬇️ Plantilla CSV", csv_bytes(plantilla), "plantilla_iris.csv", "text/csv")
        st.caption("Columnas: " + ", ".join(SIMPLES))
    with c2:
        st.markdown('<div class="card"><h3>2️⃣ Sube tu archivo</h3></div>', unsafe_allow_html=True)
        archivo = st.file_uploader("Archivo CSV", type=["csv"], label_visibility="collapsed")

    if archivo is None:
        st.info("Sube un CSV para ver los resultados. Puedes usar la plantilla como ejemplo.")
        return

    try:
        datos = pd.read_csv(archivo)
    except Exception:
        st.error("No pude leer el archivo. Verifica que sea un CSV válido.")
        return

    faltan = [c for c in SIMPLES if c not in datos.columns]
    if faltan:
        st.error("Faltan estas columnas: " + ", ".join(faltan))
        return

    datos = datos[SIMPLES].apply(pd.to_numeric, errors="coerce").dropna()
    if datos.empty:
        st.error("No hay filas con datos numéricos válidos.")
        return

    X = datos.copy()
    X.columns = columnas
    probs = modelo.predict_proba(X)
    idx = probs.argmax(axis=1)
    resultado = datos.copy()
    resultado["especie_predicha"] = [clases[i] for i in idx]
    resultado["confianza_%"] = (probs.max(axis=1) * 100).round(1)

    st.write("")
    st.markdown("### ✅ Resultados")
    m1, m2, m3 = st.columns(3)
    m1.metric("Flores clasificadas", len(resultado))
    m2.metric("Confianza promedio", f"{resultado['confianza_%'].mean():.1f}%")
    m3.metric("Especie más común", resultado["especie_predicha"].mode()[0])

    g1, g2 = st.columns([1, 1])
    conteo = resultado["especie_predicha"].value_counts().reset_index()
    conteo.columns = ["especie", "cantidad"]
    fig = px.pie(conteo, names="especie", values="cantidad", hole=0.55,
                 color="especie", color_discrete_map=COLORES)
    g1.plotly_chart(estilo(fig, 360))
    g2.dataframe(resultado, hide_index=True)
    st.download_button("⬇️ Descargar resultados (CSV)", csv_bytes(resultado), "resultados_iris.csv", "text/csv")


# =====================================================
# PÁGINA 5: HUMANO VS IA
# =====================================================
def nueva_flor():
    st.session_state["g_idx"] = int(np.random.randint(0, len(df)))
    st.session_state["g_ultimo"] = None


def reiniciar_marcador():
    st.session_state.update({"g_jug": 0, "g_ia": 0, "g_rondas": 0})
    nueva_flor()


def responder(eleccion, nombre, p):
    s = st.session_state
    idx = s["g_idx"]
    real = df.loc[idx, "especie"]
    resto = df.drop(index=idx)
    m = construir(nombre, p).fit(resto[columnas], resto["target"])
    ia = clases[int(m.predict(df.loc[[idx], columnas])[0])]
    s["g_rondas"] += 1
    s["g_jug"] += int(eleccion == real)
    s["g_ia"] += int(ia == real)
    s["g_ultimo"] = {"real": real, "jug": eleccion, "ia": ia}


def pagina_juego():
    hero("🎮 Humano vs IA", "¿Puedes adivinar la especie mejor que el modelo?")
    s = st.session_state
    for k, v in {"g_jug": 0, "g_ia": 0, "g_rondas": 0, "g_ultimo": None}.items():
        s.setdefault(k, v)
    if "g_idx" not in s:
        nueva_flor()

    m1, m2, m3 = st.columns(3)
    m1.metric("🧑 Tú", s["g_jug"])
    m2.metric("🤖 IA", s["g_ia"])
    m3.metric("Rondas", s["g_rondas"])
    st.write("")

    fila = df.loc[s["g_idx"]]
    st.markdown('<div class="card"><h3>🔍 Esta es la flor misteriosa</h3></div>', unsafe_allow_html=True)
    st.write("")
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Long. sépalo", f"{fila[columnas[0]]:.1f} cm")
    f2.metric("Ancho sépalo", f"{fila[columnas[1]]:.1f} cm")
    f3.metric("Long. pétalo", f"{fila[columnas[2]]:.1f} cm")
    f4.metric("Ancho pétalo", f"{fila[columnas[3]]:.1f} cm")
    st.write("")

    respondido = s["g_ultimo"] is not None
    st.caption("¿Qué especie crees que es?")
    b1, b2, b3 = st.columns(3)
    for col, c in zip((b1, b2, b3), clases):
        col.button(f"{EMOJIS[c]} {c.capitalize()}", key=f"op_{c}", on_click=responder,
                   args=(c, modelo_nombre, p_val), disabled=respondido)

    if respondido:
        u = s["g_ultimo"]
        color = COLORES[u["real"]]
        st.write("")
        st.markdown(
            f"""
<div class="resultado" style="--color:{color}; --glow:{color}55;">
<div class="emoji">{EMOJIS[u['real']]}</div>
<div class="nombre">Era Iris {u['real']}</div>
<div class="desc">
Tú dijiste <b>{u['jug']}</b> {'✅' if u['jug'] == u['real'] else '❌'} ·
La IA dijo <b>{u['ia']}</b> {'✅' if u['ia'] == u['real'] else '❌'}
</div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.write("")
        st.button("➡️ Siguiente flor", type="primary", on_click=nueva_flor)

    st.write("")
    st.button("🔄 Reiniciar marcador", on_click=reiniciar_marcador)
    st.caption("En cada ronda la IA se entrena sin esa flor, así que no hace trampa.")


# =====================================================
# ENRUTADOR
# =====================================================
if pagina == PAGINAS[0]:
    pagina_predictor()
elif pagina == PAGINAS[1]:
    pagina_explorador()
elif pagina == PAGINAS[2]:
    pagina_laboratorio()
elif pagina == PAGINAS[3]:
    pagina_lotes()
else:
    pagina_juego()

st.markdown(
    '<div class="pie">Hecho con ❤️ usando Streamlit · scikit-learn · Plotly</div>',
    unsafe_allow_html=True,
)
