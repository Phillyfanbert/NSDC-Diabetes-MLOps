from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from config import FEATURES_PATH, TARGET_COL, MLFLOW_TRACKING, EXPERIMENT_NAME


# ----------------------------
# 1. Paths
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "top_misses"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# 2. Load processed features
# ----------------------------
if not FEATURES_PATH.exists():
    raise FileNotFoundError(
        f"Features file not found at: {FEATURES_PATH}\n"
        "Run python src/features.py first."
    )

print("Loading processed data from:", FEATURES_PATH)
df = pd.read_parquet(FEATURES_PATH)

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in processed features file.")


# ----------------------------
# 3. Connect to MLflow tracking server
# ----------------------------
mlflow.set_tracking_uri(MLFLOW_TRACKING)
print("Using MLflow tracking URI:", MLFLOW_TRACKING)

client = MlflowClient(tracking_uri=MLFLOW_TRACKING)

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise FileNotFoundError(
        f"MLflow experiment '{EXPERIMENT_NAME}' not found.\n"
        "Run python src/train_model.py first."
    )

print("Using experiment:", experiment.name)
print("Experiment ID:", experiment.experiment_id)


# ----------------------------
# 4. Get latest finished training run
# ----------------------------
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["start_time DESC"],
)

if runs.empty:
    raise FileNotFoundError(
        "No finished runs found in this MLflow experiment.\n"
        "Run python src/train_model.py first."
    )

# Prefer Ridge run if present
ridge_runs = runs[runs["tags.model_type"] == "Ridge"]
if not ridge_runs.empty:
    run_id = ridge_runs.iloc[0]["run_id"]
else:
    run_id = runs.iloc[0]["run_id"]

print("Using MLflow run ID:", run_id)


# ----------------------------
# 5. Load model from MLflow
# ----------------------------
model_uri = f"runs:/{run_id}/model"
model = mlflow.sklearn.load_model(model_uri)
print("Loaded model from MLflow:", model_uri)


# ----------------------------
# 6. Build X using exact model feature names
# ----------------------------
if not hasattr(model, "feature_names_in_"):
    raise ValueError(
        "Loaded model does not expose feature_names_in_. "
        "Re-train the model using a pandas DataFrame."
    )

expected_cols = list(model.feature_names_in_)
print("Model expects these feature columns:")
print(expected_cols)

missing = [col for col in expected_cols if col not in df.columns]
if missing:
    raise ValueError(
        f"Processed features file is missing expected columns: {missing}\n"
        "Run python src/features.py and python src/train_model.py so they match."
    )

id_columns = [c for c in ["country_code", "year", "ParentLocation"] if c in df.columns]

X = df[expected_cols].copy()
y = df[TARGET_COL].copy()
meta = df[id_columns].copy() if id_columns else pd.DataFrame(index=df.index)

valid_rows = ~(X.isna().any(axis=1) | y.isna())
X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()
meta = meta.loc[valid_rows].copy()

print("Columns in X used for prediction:")
print(X.columns.tolist())


# ----------------------------
# 7. Predict
# ----------------------------
predictions = model.predict(X)


# ----------------------------
# 8. Residual analysis
# ----------------------------
results = meta.copy()
results["actual_diabetes"] = y.values
results["predicted_diabetes"] = predictions
results["residual"] = results["actual_diabetes"] - results["predicted_diabetes"]
results["abs_residual"] = results["residual"].abs()

results["error_direction"] = np.where(
    results["residual"] > 0,
    "Underpredicted",
    np.where(results["residual"] < 0, "Overpredicted", "Exact"),
)

top10_misses = results.sort_values("abs_residual", ascending=False).head(10).copy()
for col in ["actual_diabetes", "predicted_diabetes", "residual", "abs_residual"]:
    top10_misses[col] = top10_misses[col].round(3)

print("\n" + "=" * 90)
print("TOP 10 MISSES (BIGGEST ABSOLUTE RESIDUALS)")
print("=" * 90)
print(top10_misses)

top10_misses.to_csv(OUTPUT_DIR / "top10_misses.csv", index=False)

top10_underpredicted = results.sort_values("residual", ascending=False).head(10).copy()
for col in ["actual_diabetes", "predicted_diabetes", "residual", "abs_residual"]:
    top10_underpredicted[col] = top10_underpredicted[col].round(3)
top10_underpredicted.to_csv(OUTPUT_DIR / "top10_underpredicted.csv", index=False)

top10_overpredicted = results.sort_values("residual", ascending=True).head(10).copy()
for col in ["actual_diabetes", "predicted_diabetes", "residual", "abs_residual"]:
    top10_overpredicted[col] = top10_overpredicted[col].round(3)
top10_overpredicted.to_csv(OUTPUT_DIR / "top10_overpredicted.csv", index=False)

if "ParentLocation" in results.columns:
    region_summary = (
        results.groupby("ParentLocation")["abs_residual"]
        .agg(["count", "mean", "median", "max"])
        .reset_index()
        .rename(
            columns={
                "ParentLocation": "region",
                "count": "num_rows",
                "mean": "mean_abs_residual",
                "median": "median_abs_residual",
                "max": "max_abs_residual",
            }
        )
        .round(3)
    )
    region_summary.to_csv(OUTPUT_DIR / "region_error_summary.csv", index=False)

summary_text = """Prisha Week 6 Summary - Top 10 Misses

Task:
Identify the country-year rows where the model is most wrong using residuals.

Residual definition:
Residual = Actual - Predicted

Interpretation:
- Positive residual = the model underpredicted diabetes
- Negative residual = the model overpredicted diabetes
- Large absolute residual = the model was very wrong

Why this matters:
- Big misses may suggest hidden confounders beyond obesity alone
- Misses clustered in one region may suggest geography-specific drivers
- Underpredicted rows mean diabetes is higher than the model expected
- Overpredicted rows mean diabetes is lower than the model expected
"""

with open(OUTPUT_DIR / "prisha_top_misses_summary.txt", "w", encoding="utf-8") as f:
    f.write(summary_text)

print("\nOutputs saved to:", OUTPUT_DIR)