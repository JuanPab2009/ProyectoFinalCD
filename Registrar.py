import mlflow

# Assuming you have the best_run_id from your experiments
best_run_id = '228d47b3b1a14a498e27aa07c391f7a9'  # Replace with the actual run ID
model_name = "LaLigaBestModel"

# Register the model
result = mlflow.register_model(f"runs:/{best_run_id}/model", model_name)
print(f"Model registered with name: {result.name}, version: {result.version}")
