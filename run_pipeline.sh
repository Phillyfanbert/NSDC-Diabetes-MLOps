#!/bin/bash

set -e

echo "Starting Full MLOps Pipeline.."

echo "1. Fetching Data..."
python src/fetch_data.py

echo "2. Validating Data..."
python src/validate_data.py

echo "3. Cleaning Data..."
python src/cleaning.py

echo "4. Engineering Features..."
python src/features.py

echo "5. Training Model & Logging to MLflow..."
python src/train_model.py

echo "Pipeline Complete! Run 'mlflow ui' to see results."