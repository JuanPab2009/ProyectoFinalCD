import streamlit as st
import requests
import json

st.write("""
# LaLiga Match Result Predictor
""")

st.sidebar.header('Input Parameters')

def user_input_features():
    # Collect input from the user
    Día = st.sidebar.number_input("Día", min_value=1, max_value=38, value=1)
    Sedes = st.sidebar.selectbox("Sedes", options=[0,1], help="0 for away, 1 for home")

    st.sidebar.subheader("Datos del Adversario (Opp)")
    Edad_opp = st.sidebar.number_input("Edad_opp", value=28.0)
    Pos_opp = st.sidebar.number_input("Posición_opp", value=10.0)
    Ass_opp = st.sidebar.number_input("Asistencias_opp", min_value=0, value=5)
    TPint_opp = st.sidebar.number_input("Total Pases interceptados_opp", min_value=0, value=10)
    PrgC_opp = st.sidebar.number_input("Progresión con balón_opp", min_value=0, value=50)
    PrgP_opp = st.sidebar.number_input("Progresión de pases_opp", min_value=0, value=70)
    pct_de_TT_opp = st.sidebar.number_input("pct_de_TT_opp", min_value=0.0, max_value=100.0, value=50.0)
    Dist_opp = st.sidebar.number_input("Distancia recorrida_opp", min_value=0.0, value=10000.0)
    pct_Cmp_opp = st.sidebar.number_input("pct_Cmp_opp", min_value=0.0, max_value=100.0, value=80.0)
    Dist_tot_opp = st.sidebar.number_input("Distancia total_opp", min_value=0.0, value=20000.0)
    TklG_opp = st.sidebar.number_input("Tackles ganados_opp", min_value=0, value=20)
    Int_opp = st.sidebar.number_input("Intercepciones_opp", min_value=0, value=15)
    Err_opp = st.sidebar.number_input("Errores_opp", min_value=0, value=5)
    RL_opp = st.sidebar.number_input("Ranking Liga_opp", min_value=1, max_value=20, value=10)
    PG_opp = st.sidebar.number_input("Partidos ganados_opp", min_value=0, max_value=38, value=10)
    PE_opp = st.sidebar.number_input("Partidos empatados_opp", min_value=0, max_value=38, value=10)
    PP_opp = st.sidebar.number_input("Partidos perdidos_opp", min_value=0, max_value=38, value=10)
    GF_opp = st.sidebar.number_input("Goles a favor_opp", min_value=0, value=30)
    GC_opp = st.sidebar.number_input("Goles en contra_opp", min_value=0, value=30)
    xG_opp = st.sidebar.number_input("xG_opp", min_value=0.0, value=40.0)
    xGA_opp = st.sidebar.number_input("xGA_opp", min_value=0.0, value=30.0)
    Ultimos5_opp = st.sidebar.number_input("Ultimos5_opp", min_value=0, max_value=15, value=2, help="Desempeño en los últimos 5 partidos")
    Max_Goleador_opp = st.sidebar.number_input("Max_Goleador_opp", min_value=0, value=10)

    st.sidebar.subheader("Datos del Equipo Local (tm)")
    Edad_tm = st.sidebar.number_input("Edad_tm", value=28.0)
    Pos_tm = st.sidebar.number_input("Posición_tm", value=10.0)
    Ass_tm = st.sidebar.number_input("Asistencias_tm", min_value=0, value=5)
    TPint_tm = st.sidebar.number_input("Total Pases interceptados_tm", min_value=0, value=10)
    PrgC_tm = st.sidebar.number_input("Progresión con balón_tm", min_value=0, value=50)
    PrgP_tm = st.sidebar.number_input("Progresión de pases_tm", min_value=0, value=70)
    pct_de_TT_tm = st.sidebar.number_input("pct_de_TT_tm", min_value=0.0, max_value=100.0, value=50.0)
    Dist_tm = st.sidebar.number_input("Distancia recorrida_tm", min_value=0.0, value=10000.0)
    pct_Cmp_tm = st.sidebar.number_input("pct_Cmp_tm", min_value=0.0, max_value=100.0, value=80.0)
    Dist_tot_tm = st.sidebar.number_input("Distancia total_tm", min_value=0.0, value=20000.0)
    TklG_tm = st.sidebar.number_input("Tackles ganados_tm", min_value=0, value=20)
    Int_tm = st.sidebar.number_input("Intercepciones_tm", min_value=0, value=15)
    Err_tm = st.sidebar.number_input("Errores_tm", min_value=0, value=5)
    RL_tm = st.sidebar.number_input("Ranking Liga_tm", min_value=1, max_value=20, value=10)
    PG_tm = st.sidebar.number_input("Partidos ganados_tm", min_value=0, max_value=38, value=10)
    PE_tm = st.sidebar.number_input("Partidos empatados_tm", min_value=0, max_value=38, value=10)
    PP_tm = st.sidebar.number_input("Partidos perdidos_tm", min_value=0, max_value=38, value=10)
    GF_tm = st.sidebar.number_input("Goles a favor_tm", min_value=0, value=30)
    GC_tm = st.sidebar.number_input("Goles en contra_tm", min_value=0, value=30)
    xG_tm = st.sidebar.number_input("xG_tm", min_value=0.0, value=40.0)
    xGA_tm = st.sidebar.number_input("xGA_tm", min_value=0.0, value=30.0)
    Ultimos5_tm = st.sidebar.number_input("Ultimos5_tm", min_value=0, max_value=15, value=2, help="Desempeño en los últimos 5 partidos")
    Max_Goleador_tm = st.sidebar.number_input("Max_Goleador_tm", min_value=0, value=10)

    input_dict = {
        "Día": Día,
        "Sedes": Sedes,
        "Edad_opp": Edad_opp,
        "Pos_opp": Pos_opp,
        "Ass_opp": Ass_opp,
        "TPint_opp": TPint_opp,
        "PrgC_opp": PrgC_opp,
        "PrgP_opp": PrgP_opp,
        "pct_de_TT_opp": pct_de_TT_opp,
        "Dist_opp": Dist_opp,
        "pct_Cmp_opp": pct_Cmp_opp,
        "Dist_tot_opp": Dist_tot_opp,
        "TklG_opp": TklG_opp,
        "Int_opp": Int_opp,
        "Err_opp": Err_opp,
        "RL_opp": RL_opp,
        "PG_opp": PG_opp,
        "PE_opp": PE_opp,
        "PP_opp": PP_opp,
        "GF_opp": GF_opp,
        "GC_opp": GC_opp,
        "xG_opp": xG_opp,
        "xGA_opp": xGA_opp,
        "Ultimos5_opp": Ultimos5_opp,
        "Max_Goleador_opp": Max_Goleador_opp,
        "Edad_tm": Edad_tm,
        "Pos_tm": Pos_tm,
        "Ass_tm": Ass_tm,
        "TPint_tm": TPint_tm,
        "PrgC_tm": PrgC_tm,
        "PrgP_tm": PrgP_tm,
        "pct_de_TT_tm": pct_de_TT_tm,
        "Dist_tm": Dist_tm,
        "pct_Cmp_tm": pct_Cmp_tm,
        "Dist_tot_tm": Dist_tot_tm,
        "TklG_tm": TklG_tm,
        "Int_tm": Int_tm,
        "Err_tm": Err_tm,
        "RL_tm": RL_tm,
        "PG_tm": PG_tm,
        "PE_tm": PE_tm,
        "PP_tm": PP_tm,
        "GF_tm": GF_tm,
        "GC_tm": GC_tm,
        "xG_tm": xG_tm,
        "xGA_tm": xGA_tm,
        "Ultimos5_tm": Ultimos5_tm,
        "Max_Goleador_tm": Max_Goleador_tm
    }

    return input_dict

input_dict = user_input_features()

if st.button('Predict'):
    # In frontend/main.py
    response = requests.post(
    url="http://laliga-model-container:8001/predict",  # Change 'laliga-model-container' to 'model' http://model:8001/predict
    data=json.dumps(input_dict)
)

    if response.status_code == 200:
        prediction = response.json()['prediction']
        if prediction == 1:
            result = "Empate"
        elif prediction == 2:
            result = "Gana el equipo visitante"
        elif prediction == 3:
            result = "Gana el equipo local"
        else:
            result = "Resultado desconocido"
        st.write(f"El resultado predicho es: {result}")
    else:
        st.write("Error en la predicción")
