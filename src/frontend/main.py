import streamlit as st
import pandas as pd
import requests
import json
import re
from unidecode import unidecode

st.write("""
# LaLiga Match Result Predictor
""")

st.sidebar.header('Input Parameters')

def user_input_features():
    # Collect input from the user
    jornada = st.sidebar.number_input("Número de la jornada", min_value=1, max_value=38, value=1)

    url = "https://fbref.com/es/comps/12/horario/Resultados-y-partidos-en-La-Liga"
    tables = pd.read_html(url)
    df = tables[0]
    # seleccionamos las variables
    df = df[['Sem.', 'Día', 'Fecha', 'Local', 'Visitante']]
    # Hacemos la columna fecha del formato correspondiente
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    # Hacemos el encoding de los días
    dias_map = {
        'Lun': 1,
        'Mar': 2,
        'Mié': 3,
        'Jue': 4,
        'Vie': 5,
        'Sáb': 6,
        'Dom': 7
    }
    df["Día"] = df["Día"].map(dias_map)
    # Filtramos por jornada
    df = df[df["Sem."] == jornada]
    # Agregamos la columna de sede
    df["Sedes"] = 1
    # Renombramos las columnas
    df = df[["Día", "Sedes", "Visitante", "Local"]]
    df = df.rename(columns={"Local": "Anfitrion", "Visitante": "Adversario"})
    # Duplicamos el dataframe e invertimos las columnas para hacer la concatenación
    df_2 = df.copy()
    df_2 = df_2.rename(columns={"Adversario": "Anfitrion", "Anfitrion": "Adversario"})
    df_2["Sedes"] = 0
    df = pd.concat([df, df_2], ignore_index=True)
    df["Día"] = df["Día"].astype(int)
    ### Estadísticas básicas
    url = "https://fbref.com/es/comps/12/Estadisticas-de-La-Liga"
    tables = pd.read_html(url)
    df_basic = tables[0]
    df_basic = df_basic[['RL', 'Equipo', 'PG', 'PE', 'PP', 'GF', 'GC', 'xG', 'xGA', 'Últimos 5', 'Máximo Goleador del Equipo']]
    df_basic['Máximo Goleador del Equipo'] = df_basic['Máximo Goleador del Equipo'].apply(
        lambda x: int(re.search(r'\b(\d+)\b', x).group(1)) if re.search(r'\b(\d+)\b', x) else None)
    df_basic['Últimos 5'] = df_basic['Últimos 5'].apply(lambda resultados: sum(
        [3 if resultado == 'PG' else (1 if resultado == 'PE' else 0) for resultado in resultados.split()]))

    ### Estadísticas de ofensiva
    df_ataque = tables[2].drop(["Tiempo Jugado", "Expectativa", 'Por 90 Minutos'], axis=1)
    df_ataque.columns = df_ataque.columns.droplevel(level=0)
    df_ataque = df_ataque[['Equipo', 'Edad', 'Pos.', 'Ass', 'TPint', 'PrgC', 'PrgP']]

    # Disparos
    df_disparos = tables[8]
    df_disparos.columns = df_disparos.columns.droplevel(level=0)
    df_disparos = df_disparos[['Equipo', '% de TT', 'Dist']]
    df_ataque = pd.merge(df_ataque, df_disparos, left_on='Equipo', right_on='Equipo', how='left')

    # Pases
    df_pases = tables[10].drop(["Cortos", "Medios", 'Largos', 'Expectativa'], axis=1)
    df_pases.columns = df_pases.columns.droplevel(level=0)
    df_pases = df_pases[['Equipo', '% Cmp', 'Dist. tot.']]
    df_ataque = pd.merge(df_ataque, df_pases, left_on='Equipo', right_on='Equipo', how='left')

    ### Estadísticas de defensa
    df_porteria = tables[4].drop(["Tiempo Jugado", "Tiros penales"], axis=1)
    df_porteria.columns = df_porteria.columns.droplevel(level=0)
    df_porteria = df_porteria[['Equipo', 'GC', 'DaPC', 'Salvadas', 'PaC']]
    df_defensa = tables[16].drop(['Desafíos'], axis=1)
    df_defensa.columns = df_defensa.columns.droplevel(level=0)
    df_defensa = df_defensa[['Equipo', 'TklG', 'Int', 'Err']]
    df_final = pd.merge(df_ataque, df_defensa, left_on='Equipo', right_on='Equipo', how='left')
    df_final = pd.merge(df_final, df_basic, left_on='Equipo', right_on='Equipo', how='left')
    df_opp = df_final.copy()
    df_tm = df_final.copy()

    # Renombramos las columnas
    columns_to_rename = ['Edad', 'Pos.', 'Ass', 'TPint', 'PrgC', 'PrgP', '% de TT',
                         'Dist', '% Cmp', 'Dist. tot.', 'TklG', 'Int', 'Err', 'RL', 'PG', 'PE',
                         'PP', 'GF', 'GC', 'xG', 'xGA', 'Últimos 5', 'Máximo Goleador del Equipo']
    new_column_names_tm = [f"{col}(tm)" for col in columns_to_rename]
    df_tm.rename(columns=dict(zip(columns_to_rename, new_column_names_tm)), inplace=True)
    new_column_names_opp = [f"{col}(opp)" for col in columns_to_rename]
    df_opp.rename(columns=dict(zip(columns_to_rename, new_column_names_opp)), inplace=True)

    df = pd.merge(df, df_opp, left_on='Adversario', right_on='Equipo', how='left')
    df = pd.merge(df, df_tm, left_on='Anfitrion', right_on='Equipo', how='left')
    df = df.drop(['Equipo_x', 'Equipo_y'], axis=1)

    X = df[['Día','Sedes','Edad(opp)','Pos.(opp)', 'Ass(opp)', 'TPint(opp)',
      'PrgC(opp)', 'PrgP(opp)','% de TT(opp)', 'Dist(opp)', '% Cmp(opp)', 'Dist. tot.(opp)','TklG(opp)', 'Int(opp)',
      'Err(opp)', 'RL(opp)', 'PG(opp)', 'PE(opp)','PP(opp)', 'GF(opp)', 'GC(opp)', 'xG(opp)', 'xGA(opp)','Últimos 5(opp)',
      'Máximo Goleador del Equipo(opp)', 'Edad(tm)', 'Pos.(tm)', 'Ass(tm)', 'TPint(tm)', 'PrgC(tm)', 'PrgP(tm)',
      '% de TT(tm)', 'Dist(tm)', '% Cmp(tm)', 'Dist. tot.(tm)', 'TklG(tm)','Int(tm)', 'Err(tm)', 'RL(tm)', 'PG(tm)',
      'PE(tm)', 'PP(tm)', 'GF(tm)','GC(tm)', 'xG(tm)', 'xGA(tm)', 'Últimos 5(tm)','Máximo Goleador del Equipo(tm)']]

    # Convertimos X a un diccionario
    jornada_dict = X.iloc[0].to_dict()

    # Ajustamos los nombres de las claves para que coincidan con el backend
    input_dict = {}
    for key, value in jornada_dict.items():
        # Reemplazar caracteres especiales y espacios
        new_key = key.replace('(', '_').replace(')', '').replace('.', '').replace('%', 'pct').replace(' ', '_')
        # Eliminar acentos y caracteres especiales
        new_key = unidecode(new_key)
        # Casos especiales
        if new_key == 'Ultimos_5_opp':
            new_key = 'Ultimos5_opp'
        elif new_key == 'Ultimos_5_tm':
            new_key = 'Ultimos5_tm'
        elif new_key == 'Maximo_Goleador_del_Equipo_opp':
            new_key = 'Max_Goleador_opp'
        elif new_key == 'Maximo_Goleador_del_Equipo_tm':
            new_key = 'Max_Goleador_tm'
        input_dict[new_key] = value

    # Opcional: imprimir el diccionario para verificar
    print("Input dict:", input_dict)

    return input_dict, df

input_dict,df = user_input_features()

if st.button('Predict'):
    response = requests.post(
        url="http://model:8001/predict",
        data=json.dumps(input_dict)
    )
    if response.status_code == 200:
        # Assuming the response contains probabilities
        probabilities = response.json()
        final_probabilites = pd.DataFrame(probabilities)
        final_probabilites['Anfitrion'] = df['Anfitrion']
        final_probabilites['Rival']=df['Adversario']
        st.write("### Predicted Probabilities")
        st.write(final_probabilites)
    else:
        st.error(f"Error in the prediction request: {response.status_code} - {response.text}")
