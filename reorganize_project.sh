#!/bin/bash

# Create directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p notebooks
mkdir -p src/api
mkdir -p src/pipeline
mkdir -p src/ui
mkdir -p models
mkdir -p artifacts
mkdir -p docker
mkdir -p scripts
mkdir -p config

# Move data files
mv "Laliga Dataset 2023-2024.xlsx" data/raw/Laliga_Dataset_2023-2024.xlsx

# Move notebooks
mv *.ipynb notebooks/

# Move source code files
mv API.py src/api/
mv Training_pipeline.py src/pipeline/
mv UI.py src/ui/

# If you have Preprocesamiento.py
if [ -f "Preprocesamiento.py" ]; then
  mv Preprocesamiento.py src/pipeline/
fi

# Move Docker files
mv Dockerfile docker/
mv Dockerfile_mlflow docker/
mv docker-compose.yaml docker/

# Move configuration files
mv .env config/

# Ensure .gitignore, README.md, requirements.txt remain at root
# Optionally, move 'notas_correcciones.txt' to a docs/ directory
mkdir -p docs
mv notas_correcciones.txt docs/

echo "Project reorganization complete."

chmod +x reorganize_project.sh

