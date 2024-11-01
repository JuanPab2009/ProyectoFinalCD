from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

# Cargar el modelo registrado
model_name = "LaLigaBestModel"
model_version = 1  # Cambia al número de versión correspondiente
model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")

# Definir el esquema de entrada
class MatchData(BaseModel):
    # Añade todas las características necesarias
    Dia: int
    Sedes: int
    Edad_opp: float
    Pos_opp: float
    # ...

@app.post("/predict")
def predict(data: MatchData):
    # Convertir los datos a DataFrame
    input_data = pd.DataFrame([data.dict()])
    # Realizar la predicción
    prediction = model.predict(input_data)
    return {"prediction": int(prediction[0])}

# uvicorn main:app --host 0.0.0.0 --port 8000