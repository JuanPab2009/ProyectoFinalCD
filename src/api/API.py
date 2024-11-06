from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow.pyfunc
import pandas as pd 
import os
from dotenv import load_dotenv

app = FastAPI()

# Cargar las variables del archivo .env
load_dotenv()

# Recuperar las credenciales de MLflow desde variables de entorno
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

if not all([MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD, MLFLOW_TRACKING_URI]):
    raise EnvironmentError("Faltan variables de entorno para MLflow. Verifica tu archivo .env.")

# Configurar las credenciales y el URI de tracking para MLflow
os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Cargar el modelo registrado en MLflow
model_name = "LaLigaBestModel"
model_version = 2  # Actualizar si es necesario
model_uri = f"models:/{model_name}/{model_version}"

try:
    model = mlflow.pyfunc.load_model(model_uri=model_uri)
except Exception as e:
    raise RuntimeError(f"No se pudo cargar el modelo desde MLflow: {e}")

# Definir el esquema de entrada utilizando Pydantic
class MatchData(BaseModel):
    # Características del partido
    Dia: int = Field(..., description="Día del partido")
    Sedes: int = Field(..., description="Número de sedes")
    
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
    try:
        # Convertir los datos a DataFrame
        input_data = pd.DataFrame([data.dict()])
        
        # Renombrar las columnas para que coincidan con las características del modelo
        input_data.rename(columns=variable_name_to_feature_name, inplace=True)
        
        # Verificar que todas las columnas requeridas están presentes
        missing_features = set(variable_name_to_feature_name.values()) - set(input_data.columns)
        if missing_features:
            raise ValueError(f"Faltan las siguientes características: {missing_features}")
        
        # Realizar la predicción
        prediction = model.predict(input_data)
        
        # Asegurarse de que la predicción tenga el formato esperado
        if not isinstance(prediction, (list, pd.Series, np.ndarray)):
            raise ValueError("La predicción no tiene el formato esperado.")
        
        return {"prediction": int(prediction[0])}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Para ejecutar la API, usa el siguiente comando en tu terminal:
# uvicorn nombre_del_archivo:app --host 0.0.0.0 --port 8000
