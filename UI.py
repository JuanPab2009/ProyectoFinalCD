#UI

import streamlit as st
import requests

st.title("Predicción de Resultados de La Liga")

# Formularios para ingresar los datos
dia = st.number_input("Día", min_value=1, max_value=38, value=1)
sedes = st.selectbox("Sedes", options=[0, 1])
edad_opp = st.number_input("Edad del equipo oponente", min_value=15.0, max_value=40.0, value=28.0)
pos_opp = st.number_input("Posición del equipo oponente", min_value=1.0, max_value=20.0, value=10.0)
# Añade más campos según sea necesario

# Botón para predecir
if st.button("Predecir"):
    data = {
        "Dia": dia,
        "Sedes": sedes,
        "Edad_opp": edad_opp,
        "Pos_opp": pos_opp,
        # Añade más campos
    }
    response = requests.post("http://localhost:8000/predict", json=data)
    if response.status_code == 200:
        prediction = response.json()["prediction"]
        st.write(f"Predicción del resultado: {prediction}")
    else:
        st.write("Error en la predicción")


#streamlit run app.py
