# from fastapi import FastAPI
# from pydantic import BaseModel
# import mlflow.pyfunc
# import pandas as pd
# from mlflow import MlflowClient
# import dagshub
# import os
# import mlflow
# import os
# from prefect.settings import PREFECT_API_URL, PREFECT_API_KEY, PREFECT_API_ENABLE_HTTP2

# # Set PREFECT_API_URL to None
# os.environ["PREFECT_API_URL"] = ""


# # DAGSHUB_USER = os.getenv("DAGSHUB_USER")
# # DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
# # REPO_OWNER = "JuanPab2009"
# # REPO_NAME = "ProyectoFinalCD"

# # # # Set the tracking URI to DAGsHub
# # # mlflow.set_tracking_uri(f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow")

# # # Set the MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD environment variables
# # os.environ['MLFLOW_TRACKING_USERNAME'] = 'diegomercadoc'
# # os.environ['MLFLOW_TRACKING_PASSWORD'] = '87ebd63fd77e2ef94b83fc2c172f083bff205461'

# #cmd ro run mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root file:/C:/Users/juanp/OneDrive/Documentos/ProyectoFinalCD/ProyectoFinalCD.mlflow


# # Set the tracking URI to where your MLflow server is running
# # MLFLOW_TRACKING_URI = "https://dagshub.com/JuanPab2009/ProyectoFinalCD.mlflow"  # Replace with your actual tracking URI
# # mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)  # Replace with your actual tracking URI


# # MLFLOW_TRACKING_URI="http://localhost:5000"

# # Configurar MLflow para que use el backend de DagsHub
# # client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# app = FastAPI()

# # Set the MLflow tracking URI
# mlflow.set_tracking_uri("http://localhost:5000")

# # Cargar el modelo registrado
# model_name = "LaLigaBestModel"
# model_version = 1  # Cambia al número de versión correspondiente
# model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")

# # Definir el esquema de entrada
# class MatchData(BaseModel):
#     # Características del partido
#     Dia: int
#     Sedes: int

#     # Características del equipo oponente
#     Edad_opp: float
#     Pos_opp: float
#     Ass_opp: float
#     TPint_opp: float
#     PrgC_opp: float
#     PrgP_opp: float
#     Pct_de_TT_opp: float
#     Dist_opp: float
#     Pct_Cmp_opp: float
#     Dist_tot_opp: float
#     TklG_opp: float
#     Int_opp: float
#     Err_opp: float
#     RL_opp: float
#     PG_opp: float
#     PE_opp: float
#     PP_opp: float
#     GF_opp: float
#     GC_opp: float
#     xG_opp: float
#     xGA_opp: float
#     Ultimos5_opp: float
#     MaxGoleadorEquipo_opp: float

#     # Características del equipo propio
#     Edad_tm: float
#     Pos_tm: float
#     Ass_tm: float
#     TPint_tm: float
#     PrgC_tm: float
#     PrgP_tm: float
#     Pct_de_TT_tm: float
#     Dist_tm: float
#     Pct_Cmp_tm: float
#     Dist_tot_tm: float
#     TklG_tm: float
#     Int_tm: float
#     Err_tm: float
#     RL_tm: float
#     PG_tm: float
#     PE_tm: float
#     PP_tm: float
#     GF_tm: float
#     GC_tm: float
#     xG_tm: float
#     xGA_tm: float
#     Ultimos5_tm: float
#     MaxGoleadorEquipo_tm: float

# # Mapping de nombres de variables a nombres de columnas originales
# variable_name_to_feature_name = {
#     'Dia': 'Día',
#     'Sedes': 'Sedes',
#     'Edad_opp': 'Edad(opp)',
#     'Pos_opp': 'Pos.(opp)',
#     'Ass_opp': 'Ass(opp)',
#     'TPint_opp': 'TPint(opp)',
#     'PrgC_opp': 'PrgC(opp)',
#     'PrgP_opp': 'PrgP(opp)',
#     'Pct_de_TT_opp': '% de TT(opp)',
#     'Dist_opp': 'Dist(opp)',
#     'Pct_Cmp_opp': '% Cmp(opp)',
#     'Dist_tot_opp': 'Dist. tot.(opp)',
#     'TklG_opp': 'TklG(opp)',
#     'Int_opp': 'Int(opp)',
#     'Err_opp': 'Err(opp)',
#     'RL_opp': 'RL(opp)',
#     'PG_opp': 'PG(opp)',
#     'PE_opp': 'PE(opp)',
#     'PP_opp': 'PP(opp)',
#     'GF_opp': 'GF(opp)',
#     'GC_opp': 'GC(opp)',
#     'xG_opp': 'xG(opp)',
#     'xGA_opp': 'xGA(opp)',
#     'Ultimos5_opp': 'Últimos 5(opp)',
#     'MaxGoleadorEquipo_opp': 'Máximo Goleador del Equipo(opp)',
#     'Edad_tm': 'Edad(tm)',
#     'Pos_tm': 'Pos(tm)',
#     'Ass_tm': 'Ass(tm)',
#     'TPint_tm': 'TPint(tm)',
#     'PrgC_tm': 'PrgC(tm)',
#     'PrgP_tm': 'PrgP(tm)',
#     'Pct_de_TT_tm': '% de TT(tm)',
#     'Dist_tm': 'Dist(tm)',
#     'Pct_Cmp_tm': '% Cmp(tm)',
#     'Dist_tot_tm': 'Dist. tot(tm)',
#     'TklG_tm': 'TklG(tm)',
#     'Int_tm': 'Int(tm)',
#     'Err_tm': 'Err(tm)',
#     'RL_tm': 'RL(tm)',
#     'PG_tm': 'PG(tm)',
#     'PE_tm': 'PE(tm)',
#     'PP_tm': 'PP(tm)',
#     'GF_tm': 'GF(tm)',
#     'GC_tm': 'GC(tm)',
#     'xG_tm': 'xG(tm)',
#     'xGA_tm': 'xGA(tm)',
#     'Ultimos5_tm': 'Últimos 5(tm)',
#     'MaxGoleadorEquipo_tm': 'Máximo Goleador del Equipo(tm)'
# }

# @app.post("/predict")
# def predict(data: MatchData):
#     # Convertir los datos a DataFrame
#     input_data = pd.DataFrame([data.dict()])
#     # Renombrar las columnas para que coincidan con las características del modelo
#     input_data.rename(columns=variable_name_to_feature_name, inplace=True)
#     # Realizar la predicción
#     prediction = model.predict(input_data)
#     return {"prediction": int(prediction[0])}

# # Para ejecutar la API:
# # uvicorn API:app --host 0.0.0.0 --port 8000


from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
import pandas as pd 
import os
from dotenv import load_dotenv
import os

app = FastAPI()

# Cargar las variables del archivo .env
load_dotenv()

# Set your DAGsHub username and access token as environment variables
os.environ["MLFLOW_TRACKING_USERNAME"] = "JuanPab2009"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "87ebd63fd77e2ef94b83fc2c172f083bff205461"

# Set the tracking URI to your DAGsHub repository's MLflow server
mlflow.set_tracking_uri("https://dagshub.com/JuanPab2009/ProyectoFinalCD.mlflow")



# # Set the MLflow tracking URI
# mlflow.set_tracking_uri("http://localhost:5000")

# Load the model registered in MLflow
model_name = "LaLigaBestModel"
model_version = 2  # Update if necessary
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.pyfunc.load_model(model_uri=model_uri)

# Definir el esquema de entrada
class MatchData(BaseModel):
    # Características del partido
    Dia: int
    Sedes: int

    # Características del equipo oponente
    Edad_opp: float
    Pos_opp: float
    Ass_opp: float
    TPint_opp: float
    PrgC_opp: float
    PrgP_opp: float
    Pct_de_TT_opp: float
    Dist_opp: float
    Pct_Cmp_opp: float
    Dist_tot_opp: float
    TklG_opp: float
    Int_opp: float
    Err_opp: float
    RL_opp: float
    PG_opp: float
    PE_opp: float
    PP_opp: float
    GF_opp: float
    GC_opp: float
    xG_opp: float
    xGA_opp: float
    Ultimos5_opp: float
    MaxGoleadorEquipo_opp: float

    # Características del equipo propio
    Edad_tm: float
    Pos_tm: float
    Ass_tm: float
    TPint_tm: float
    PrgC_tm: float
    PrgP_tm: float
    Pct_de_TT_tm: float
    Dist_tm: float
    Pct_Cmp_tm: float
    Dist_tot_tm: float
    TklG_tm: float
    Int_tm: float
    Err_tm: float
    RL_tm: float
    PG_tm: float
    PE_tm: float
    PP_tm: float
    GF_tm: float
    GC_tm: float
    xG_tm: float
    xGA_tm: float
    Ultimos5_tm: float
    MaxGoleadorEquipo_tm: float

# Mapping de nombres de variables a nombres de columnas originales
variable_name_to_feature_name = {
    'Dia': 'Día',
    'Sedes': 'Sedes',
    'Edad_opp': 'Edad(opp)',
    'Pos_opp': 'Pos.(opp)',
    'Ass_opp': 'Ass(opp)',
    'TPint_opp': 'TPint(opp)',
    'PrgC_opp': 'PrgC(opp)',
    'PrgP_opp': 'PrgP(opp)',
    'Pct_de_TT_opp': '% de TT(opp)',
    'Dist_opp': 'Dist(opp)',
    'Pct_Cmp_opp': '% Cmp(opp)',
    'Dist_tot_opp': 'Dist. tot.(opp)',
    'TklG_opp': 'TklG(opp)',
    'Int_opp': 'Int(opp)',
    'Err_opp': 'Err(opp)',
    'RL_opp': 'RL(opp)',
    'PG_opp': 'PG(opp)',
    'PE_opp': 'PE(opp)',
    'PP_opp': 'PP(opp)',
    'GF_opp': 'GF(opp)',
    'GC_opp': 'GC(opp)',
    'xG_opp': 'xG(opp)',
    'xGA_opp': 'xGA(opp)',
    'Ultimos5_opp': 'Últimos 5(opp)',
    'MaxGoleadorEquipo_opp': 'Máximo Goleador del Equipo(opp)',
    'Edad_tm': 'Edad(tm)',
    'Pos_tm': 'Pos(tm)',
    'Ass_tm': 'Ass(tm)',
    'TPint_tm': 'TPint(tm)',
    'PrgC_tm': 'PrgC(tm)',
    'PrgP_tm': 'PrgP(tm)',
    'Pct_de_TT_tm': '% de TT(tm)',
    'Dist_tm': 'Dist(tm)',
    'Pct_Cmp_tm': '% Cmp(tm)',
    'Dist_tot_tm': 'Dist. tot(tm)',
    'TklG_tm': 'TklG(tm)',
    'Int_tm': 'Int(tm)',
    'Err_tm': 'Err(tm)',
    'RL_tm': 'RL(tm)',
    'PG_tm': 'PG(tm)',
    'PE_tm': 'PE(tm)',
    'PP_tm': 'PP(tm)',
    'GF_tm': 'GF(tm)',
    'GC_tm': 'GC(tm)',
    'xG_tm': 'xG(tm)',
    'xGA_tm': 'xGA(tm)',
    'Ultimos5_tm': 'Últimos 5(tm)',
    'MaxGoleadorEquipo_tm': 'Máximo Goleador del Equipo(tm)'
}

@app.post("/predict")
def predict(data: MatchData):
    # Convertir los datos a DataFrame
    input_data = pd.DataFrame([data.dict()])
    # Renombrar las columnas para que coincidan con las características del modelo
    input_data.rename(columns=variable_name_to_feature_name, inplace=True)
    # Realizar la predicción
    prediction = model.predict(input_data)
    return {"prediction": int(prediction[0])}

# Para ejecutar la API:
# uvicorn API:app --host 0.0.0.0 --port 8000
