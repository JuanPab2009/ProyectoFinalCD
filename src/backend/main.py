import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd


# Load the model from MLflow Model Registry
model_name = "LaLigaBestModel"
model_version = 2  # Update to the desired version

model_uri = f"models:/{model_name}/{model_version}"

# Set the MLflow tracking URI (update with your DAGsHub credentials if necessary)
mlflow.set_tracking_uri("https://dagshub.com/JuanPab2009/ProyectoFinalCD.mlflow")

# Load the model as a PyFuncModel
model = mlflow.pyfunc.load_model(model_uri)

# Define the preprocessor function
def preprocess(input_data):
    # Convert input data to DataFrame
    df = pd.DataFrame([input_data])

    # If scaling was used during training, apply the same scaler here
    # For this example, we'll assume no scaling is needed

    return df

# Set up FastAPI app
app = FastAPI()

# Define the input data model using Pydantic
class InputData(BaseModel):
    Día: int
    Sedes: int
    Edad_opp: float
    Pos_opp: float
    Ass_opp: int
    TPint_opp: int
    PrgC_opp: int
    PrgP_opp: int
    pct_de_TT_opp: float
    Dist_opp: float
    pct_Cmp_opp: float
    Dist_tot_opp: float
    TklG_opp: int
    Int_opp: int
    Err_opp: int
    RL_opp: int
    PG_opp: int
    PE_opp: int
    PP_opp: int
    GF_opp: int
    GC_opp: int
    xG_opp: float
    xGA_opp: float
    Ultimos5_opp: int
    Max_Goleador_opp: int
    Edad_tm: float
    Pos_tm: float
    Ass_tm: int
    TPint_tm: int
    PrgC_tm: int
    PrgP_tm: int
    pct_de_TT_tm: float
    Dist_tm: float
    pct_Cmp_tm: float
    Dist_tot_tm: float
    TklG_tm: int
    Int_tm: int
    Err_tm: int
    RL_tm: int
    PG_tm: int
    PE_tm: int
    PP_tm: int
    GF_tm: int
    GC_tm: int
    xG_tm: float
    xGA_tm: float
    Ultimos5_tm: int
    Max_Goleador_tm: int

@app.post("/predict")
def predict_endpoint(input_data: InputData):
    # Convert input data to dictionary
    input_dict = input_data.dict()
    # Preprocess the input data
    features = preprocess(input_dict)
    # Make predictions
    predictions = model.predict(features)
    # Return the predictions
    return {"prediction": int(predictions[0])}


dependencies = mlflow.pyfunc.get_model_dependencies(model_uri)
print(dependencies)
