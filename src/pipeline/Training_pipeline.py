import os

# Set PREFECT_API_URL to an empty string to run in local mode
os.environ["PREFECT_API_URL"] = ""

from prefect import flow, task
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import pandas as pd

# Set the MLflow tracking URI to point to the local server running on port 5000
mlflow.set_tracking_uri("http://localhost:5000")

# Define the experiment name
experiment_name = "LaLiga_Predictions"

# Create or set the experiment
mlflow.set_experiment(experiment_name)

@task
def load_data():
    df = pd.read_excel('LaLiga Dataset 2023-2024.xlsx')
    return df

@task
def preprocess_data(df):
    X = df[[  # Your feature columns here
        'Día',
        'Sedes',
        'Edad(opp)',
        'Pos.(opp)',
        'Ass(opp)',
        'TPint(opp)',
        'PrgC(opp)',
        'PrgP(opp)',
        '% de TT(opp)',
        'Dist(opp)',
        '% Cmp(opp)',
        'Dist. tot.(opp)',
        'TklG(opp)',
        'Int(opp)',
        'Err(opp)',
        'RL(opp)',
        'PG(opp)',
        'PE(opp)',
        'PP(opp)',
        'GF(opp)',
        'GC(opp)',
        'xG(opp)',
        'xGA(opp)',
        'Últimos 5(opp)',
        'Máximo Goleador del Equipo(opp)',
        'Edad(tm)',
        'Pos(tm)',
        'Ass(tm)',
        'TPint(tm)',
        'PrgC(tm)',
        'PrgP(tm)',
        '% de TT(tm)',
        'Dist(tm)',
        '% Cmp(tm)',
        'Dist. tot(tm)',
        'TklG(tm)',
        'Int(tm)',
        'Err(tm)',
        'RL(tm)',
        'PG(tm)',
        'PE(tm)',
        'PP(tm)',
        'GF(tm)',
        'GC(tm)',
        'xG(tm)',
        'xGA(tm)',
        'Últimos 5(tm)',
        'Máximo Goleador del Equipo(tm)'
    ]]
    y = df['Resultado']
    return X, y

@task
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

@task
def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

@task
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc

@task
def log_experiment(model, acc):
    # Start an MLflow run
    with mlflow.start_run() as run:
        # Log parameters, metrics, and the model
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_params(model.get_params())
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        # Register the model in the MLflow Model Registry
        model_uri = f"runs:/{run.info.run_id}/model"
        registered_model_name = "LaLigaBestModel"

        # Register the model
        result = mlflow.register_model(model_uri, registered_model_name)
        print(f"Model registered with name: {result.name}, version: {result.version}")

@flow(name="Training Pipeline")
def main_flow():
    df = load_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    acc = evaluate_model(model, X_test, y_test)
    log_experiment(model, acc)

# Execute the flow
if __name__ == '__main__':
    main_flow()
