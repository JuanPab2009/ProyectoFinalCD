import streamlit as st
import requests

st.title("Predicción de Resultados de La Liga")

# Formularios para ingresar los datos

st.header("Información del Partido")
dia = st.number_input("Día", min_value=1, max_value=38, value=1)
sedes = st.selectbox("Sedes (0: Visitante, 1: Local)", options=[0, 1])

st.header("Características del Equipo Oponente")
edad_opp = st.number_input("Edad promedio del equipo oponente", min_value=15.0, max_value=40.0, value=28.0)
pos_opp = st.number_input("Posición del equipo oponente", min_value=1.0, max_value=20.0, value=10.0)
ass_opp = st.number_input("Asistencias del equipo oponente", min_value=0.0, value=5.0)
tpint_opp = st.number_input("Intercepciones totales (opp)", min_value=0.0, value=10.0)
prgC_opp = st.number_input("Progresión de carreras (opp)", min_value=0.0, value=20.0)
prgP_opp = st.number_input("Progresión de pases (opp)", min_value=0.0, value=30.0)
pct_de_TT_opp = st.number_input("% de TT (opp)", min_value=0.0, max_value=100.0, value=50.0)
dist_opp = st.number_input("Distancia (opp)", min_value=0.0, value=100.0)
pct_cmp_opp = st.number_input("% de pases completados (opp)", min_value=0.0, max_value=100.0, value=80.0)
dist_tot_opp = st.number_input("Distancia total (opp)", min_value=0.0, value=500.0)
tklG_opp = st.number_input("Tackles ganados (opp)", min_value=0.0, value=5.0)
int_opp = st.number_input("Intercepciones (opp)", min_value=0.0, value=5.0)
err_opp = st.number_input("Errores (opp)", min_value=0.0, value=1.0)
rl_opp = st.number_input("Ranking en la liga (opp)", min_value=1.0, max_value=20.0, value=10.0)
pg_opp = st.number_input("Partidos ganados (opp)", min_value=0.0, value=10.0)
pe_opp = st.number_input("Partidos empatados (opp)", min_value=0.0, value=10.0)
pp_opp = st.number_input("Partidos perdidos (opp)", min_value=0.0, value=10.0)
gf_opp = st.number_input("Goles a favor (opp)", min_value=0.0, value=30.0)
gc_opp = st.number_input("Goles en contra (opp)", min_value=0.0, value=30.0)
xg_opp = st.number_input("xG (opp)", min_value=0.0, value=25.0)
xga_opp = st.number_input("xGA (opp)", min_value=0.0, value=25.0)
ultimos5_opp = st.number_input("Últimos 5 resultados (opp)", min_value=0.0, max_value=15.0, value=8.0)
maxGoleadorEquipo_opp = st.number_input("Goles del máximo goleador (opp)", min_value=0.0, value=10.0)

st.header("Características del Equipo Propio")
edad_tm = st.number_input("Edad promedio del equipo", min_value=15.0, max_value=40.0, value=28.0)
pos_tm = st.number_input("Posición del equipo", min_value=1.0, max_value=20.0, value=10.0)
ass_tm = st.number_input("Asistencias del equipo", min_value=0.0, value=5.0)
tpint_tm = st.number_input("Intercepciones totales (tm)", min_value=0.0, value=10.0)
prgC_tm = st.number_input("Progresión de carreras (tm)", min_value=0.0, value=20.0)
prgP_tm = st.number_input("Progresión de pases (tm)", min_value=0.0, value=30.0)
pct_de_TT_tm = st.number_input("% de TT (tm)", min_value=0.0, max_value=100.0, value=50.0)
dist_tm = st.number_input("Distancia (tm)", min_value=0.0, value=100.0)
pct_cmp_tm = st.number_input("% de pases completados (tm)", min_value=0.0, max_value=100.0, value=80.0)
dist_tot_tm = st.number_input("Distancia total (tm)", min_value=0.0, value=500.0)
tklG_tm = st.number_input("Tackles ganados (tm)", min_value=0.0, value=5.0)
int_tm = st.number_input("Intercepciones (tm)", min_value=0.0, value=5.0)
err_tm = st.number_input("Errores (tm)", min_value=0.0, value=1.0)
rl_tm = st.number_input("Ranking en la liga (tm)", min_value=1.0, max_value=20.0, value=10.0)
pg_tm = st.number_input("Partidos ganados (tm)", min_value=0.0, value=10.0)
pe_tm = st.number_input("Partidos empatados (tm)", min_value=0.0, value=10.0)
pp_tm = st.number_input("Partidos perdidos (tm)", min_value=0.0, value=10.0)
gf_tm = st.number_input("Goles a favor (tm)", min_value=0.0, value=30.0)
gc_tm = st.number_input("Goles en contra (tm)", min_value=0.0, value=30.0)
xg_tm = st.number_input("xG (tm)", min_value=0.0, value=25.0)
xga_tm = st.number_input("xGA (tm)", min_value=0.0, value=25.0)
ultimos5_tm = st.number_input("Últimos 5 resultados (tm)", min_value=0.0, max_value=15.0, value=8.0)
maxGoleadorEquipo_tm = st.number_input("Goles del máximo goleador (tm)", min_value=0.0, value=10.0)

# Botón para predecir
if st.button("Predecir"):
    data = {
        "Dia": dia,
        "Sedes": sedes,
        "Edad_opp": edad_opp,
        "Pos_opp": pos_opp,
        "Ass_opp": ass_opp,
        "TPint_opp": tpint_opp,
        "PrgC_opp": prgC_opp,
        "PrgP_opp": prgP_opp,
        "Pct_de_TT_opp": pct_de_TT_opp,
        "Dist_opp": dist_opp,
        "Pct_Cmp_opp": pct_cmp_opp,
        "Dist_tot_opp": dist_tot_opp,
        "TklG_opp": tklG_opp,
        "Int_opp": int_opp,
        "Err_opp": err_opp,
        "RL_opp": rl_opp,
        "PG_opp": pg_opp,
        "PE_opp": pe_opp,
        "PP_opp": pp_opp,
        "GF_opp": gf_opp,
        "GC_opp": gc_opp,
        "xG_opp": xg_opp,
        "xGA_opp": xga_opp,
        "Ultimos5_opp": ultimos5_opp,
        "MaxGoleadorEquipo_opp": maxGoleadorEquipo_opp,
        "Edad_tm": edad_tm,
        "Pos_tm": pos_tm,
        "Ass_tm": ass_tm,
        "TPint_tm": tpint_tm,
        "PrgC_tm": prgC_tm,
        "PrgP_tm": prgP_tm,
        "Pct_de_TT_tm": pct_de_TT_tm,
        "Dist_tm": dist_tm,
        "Pct_Cmp_tm": pct_cmp_tm,
        "Dist_tot_tm": dist_tot_tm,
        "TklG_tm": tklG_tm,
        "Int_tm": int_tm,
        "Err_tm": err_tm,
        "RL_tm": rl_tm,
        "PG_tm": pg_tm,
        "PE_tm": pe_tm,
        "PP_tm": pp_tm,
        "GF_tm": gf_tm,
        "GC_tm": gc_tm,
        "xG_tm": xg_tm,
        "xGA_tm": xga_tm,
        "Ultimos5_tm": ultimos5_tm,
        "MaxGoleadorEquipo_tm": maxGoleadorEquipo_tm
    }
    response = requests.post("http://localhost:8000/predict", json=data)
    if response.status_code == 200:
        prediction = response.json()["prediction"]
        st.write(f"Predicción del resultado: {prediction}")
    else:
        st.write("Error en la predicción")
