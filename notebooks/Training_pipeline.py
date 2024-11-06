from prefect import task, Flow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import pandas as pd

@task
def load_data():
    df = pd.read_excel('LaLiga Dataset 2023-2024.xlsx')
    return df

@task
def preprocess_data(df):
    X = df[['Día', 'Sedes', 'Edad(opp)', 'Pos.(opp)', ...]]  # Añade todas las características
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
    with mlflow.start_run():
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_params(model.get_params())
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

with Flow("Training Pipeline") as flow:
    df = load_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    acc = evaluate_model(model, X_test, y_test)
    log_experiment(model, acc)

# Ejecutar el flujo
if __name__ == '__main__':
    flow.run()