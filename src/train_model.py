import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# 1. Setup MLflow Experiment
mlflow.set_experiment("NSDC_Diabetes_Project")

def train_baseline():
    with mlflow.start_run(run_name="Linear_Regression_Baseline") as run:
        # --- 2. Load the Final Processed Data ---
        # (This is the file produced by Member 5: The Architect)
        data_path = 'data/processed/features.parquet'
        df = pd.read_parquet(data_path)
        df = df.dropna()

        # Define Features and Target
        X = df[['feature_obesity_scaled', 'obesity_lag_1y_scaled', 'obesity_lag_2y_scaled', 'obesity_lag_3y_scaled']]
        y = df['target_diabetes']

        # --- 3. Train the Model ---
        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)

        # --- 4. Calculate Metrics ---
        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)

        # --- 5. Log Parameters, Metrics, and Model ---
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("data_version", "v1.0_lagged_5yr")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Log the signature (input/output schema) for the registry
        signature = infer_signature(X, y)

        # Register the model in the MLflow Model Registry
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name="Diabetes_Prevalence_Model"
        )

        # --- 6. Log the "Chain of Custody" (Scripts) ---
        # This logs the code used to create the data so it's reproducible
        mlflow.log_artifact("src/cleaning.py")
        mlflow.log_artifact("src/validate_data.py")
        mlflow.log_artifact("src/features.py")

        print(f"✅ Baseline Model Logged Successfully!")
        print(f"RMSE: {rmse:.4f} | R2: {r2:.4f}")
        print(f"Run ID: {run.info.run_id}")

if __name__ == "__main__":
    train_baseline()