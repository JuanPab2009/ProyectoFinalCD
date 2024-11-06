#!/bin/bash

# Create the recommended directory structure
mkdir -p data/external data/interim data/processed data/raw
mkdir -p notebooks
mkdir -p "Informe Escrito"
mkdir -p src/frontend src/backend

# Make the script executable
chmod +x reorganize_project.sh