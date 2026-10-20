import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Clasificador de Iris", page_icon="🌸")

st.title("🌸 Clasificador de flores Iris")
st.write("Ingresa las medidas de la flor y presiona **Predecir** para saber su especie.")


@st.cache_resource
def entrenar_modelo():
    iris = load_iris()
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(iris.data, iris.target)
    return modelo, iris.target_names


modelo, clases = entrenar_modelo()

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.number_input("Longitud del sépalo (cm)", 0.0, 10.0, 5.1, 0.1)
    petal_length = st.number_input("Longitud del pétalo (cm)", 0.0, 10.0, 1.4, 0.1)
with col2:
    sepal_width = st.number_input("Ancho del sépalo (cm)", 0.0, 10.0, 3.5, 0.1)
    petal_width = st.number_input("Ancho del pétalo (cm)", 0.0, 10.0, 0.2, 0.1)

if st.button("Predecir"):
    datos = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=["sepal length (cm)", "sepal width (cm)",
                 "petal length (cm)", "petal width (cm)"],
    )
    prediccion = modelo.predict(datos)[0]
    probabilidades = modelo.predict_proba(datos)[0]

    st.success(f"La flor pertenece a la clase: **{clases[prediccion]}**")
    st.bar_chart(pd.DataFrame({"Probabilidad": probabilidades}, index=clases))
