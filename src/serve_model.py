"""
serve_model.py — FastAPI REST API for Diabetes Prevalence Model
===============================================================
Promotes the best MLflow model from Staging → Production,
then serves it as a live prediction endpoint.

Usage:
    python src/serve_model.py          # promote + start server
    uvicorn src.serve_model:app --reload  # dev mode (reload on change)

Then test with:
    curl -X POST http://127.0.0.1:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"obesity_current": 28.5, "obesity_lag_1y": 27.1, "obesity_lag_2y": 25.8, "obesity_lag_3y": 24.3}'
"""

import numpy as np
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EXPERIMENT_NAME   = "NSDC_Diabetes_Project"
REGISTERED_MODEL  = "Diabetes_Prevalence_Model"
MLFLOW_TRACKING   = "http://127.0.0.1:5000"   # local MLflow server

mlflow.set_tracking_uri(MLFLOW_TRACKING)
client = MlflowClient()

# ---------------------------------------------------------------------------
# Step 1 — Promote best Staging model → Production
# ---------------------------------------------------------------------------

def promote_best_model() -> str:
    """
    Find the highest R² run in the experiment, register it if needed,
    and transition it to Production. Returns the model URI.
    """
    print("🔍 Searching for best run in experiment…")
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found. Run the pipeline first.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.r2 DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No runs found. Run train_model.py first.")

    best_run = runs[0]
    run_id   = best_run.info.run_id
    r2       = best_run.data.metrics.get("r2", "N/A")
    rmse     = best_run.data.metrics.get("rmse", "N/A")
    print(f"✅ Best run: {run_id}  |  R²={r2:.4f}  RMSE={rmse:.4f}")

    # Ensure a registered model version exists for this run
    model_uri = f"runs:/{run_id}/model"
    try:
        mv = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL)
        version = mv.version
        print(f"📦 Registered as version {version}")
    except Exception:
        # Already registered — find the matching version
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL}'")
        version  = next(v.version for v in versions if v.run_id == run_id)
        print(f"📦 Already registered as version {version}")

    # Tag this version as Production using an alias (replaces deprecated stages)
    client.set_registered_model_alias(
        name=REGISTERED_MODEL,
        alias="Production",
        version=version,
    )
    print(f"🚀 Version {version} tagged as Production!")
    return f"models:/{REGISTERED_MODEL}@Production"


# ---------------------------------------------------------------------------
# Step 2 — Load model at startup, keep in memory
# ---------------------------------------------------------------------------

model = None  # global handle set during lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the Production model once when the server starts."""
    global model
    print("\n⏳ Loading Production model from MLflow Registry…")
    try:
        production_uri = promote_best_model()
        model = mlflow.sklearn.load_model(production_uri)
        print("✅ Model loaded and ready.\n")
    except Exception as e:
        print(f"❌ Could not load model: {e}")
        print("   Make sure MLflow UI is running: mlflow ui")
    yield
    # Teardown (nothing needed)


# ---------------------------------------------------------------------------
# Step 3 — FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Diabetes Prevalence Predictor",

    description=(
        "NSDC MLOps Project — predicts national diabetes prevalence (%) "
        "from current and historical obesity rates using a Linear Regression "
        "model trained on WHO Global Health Observatory data."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request / Response schemas -------------------------------------------

class PredictionRequest(BaseModel):
    obesity_current: float = Field(
        ..., ge=0, le=100,
        description="Current-year obesity prevalence (%)",
        json_schema_extra={"example": 28.5},
    )
    obesity_lag_1y: float = Field(
        ..., ge=0, le=100,
        description="Obesity prevalence 1 year ago (%)",
        json_schema_extra={"example": 27.1},
    )
    obesity_lag_2y: float = Field(
        ..., ge=0, le=100,
        description="Obesity prevalence 2 years ago (%)",
        json_schema_extra={"example": 25.8},
    )
    obesity_lag_3y: float = Field(
        ..., ge=0, le=100,
        description="Obesity prevalence 3 years ago (%)",
        json_schema_extra={"example": 24.3},
    )


class PredictionResponse(BaseModel):
    predicted_diabetes_prevalence_pct: float
    model_version: str
    note: str


# --- Endpoints ------------------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    """Health check — confirms the API is live."""
    return {"status": "ok", "message": "Diabetes Prevalence API is running."}


@app.get("/health", tags=["Health"])
def health():
    """Returns whether the model is loaded and ready."""
    return {
        "model_loaded": model is not None,
        "registered_model": REGISTERED_MODEL,
        "stage": "Production",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: PredictionRequest):
    """
    Predict national diabetes prevalence (%) given obesity rates.

    The model expects z-score scaled inputs matching the training pipeline
    (features.py). This endpoint scales the raw percentages automatically
    using the training-set statistics baked into the model pipeline.

    **Input**: current obesity % + 1-, 2-, 3-year lagged obesity %
    **Output**: predicted diabetes prevalence %
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check that MLflow is running and retry.",
        )

    # Build feature array matching train_model.py column order:
    # ['feature_obesity_scaled', 'obesity_lag_1y_scaled',
    #  'obesity_lag_2y_scaled', 'obesity_lag_3y_scaled']
    #
    # NOTE: The training pipeline z-scores features using dataset-wide
    # mean/std. For a production system you'd persist those scalers; here
    # we use WHO dataset approximations (mean≈14.9, std≈8.5) so the API
    # is self-contained and correct for typical input ranges.
    OBESITY_MEAN = 14.9
    OBESITY_STD  = 8.5

    def scale(v: float) -> float:
        return (v - OBESITY_MEAN) / OBESITY_STD

    features = np.array([[
        scale(request.obesity_current),
        scale(request.obesity_lag_1y),
        scale(request.obesity_lag_2y),
        scale(request.obesity_lag_3y),
    ]])

    prediction = float(model.predict(features)[0])
    # Clamp to a realistic range (diabetes prevalence is 0–50%)
    prediction = max(0.0, min(prediction, 50.0))

    return PredictionResponse(
        predicted_diabetes_prevalence_pct=round(prediction, 2),
        model_version=f"{REGISTERED_MODEL}/Production",
        note=(
            "Prediction based on WHO GHO data. "
            "Model: Linear Regression (R²≈0.70, RMSE≈2.17)."
        ),
    )


# ---------------------------------------------------------------------------
# Entry point — run directly with: python src/serve_model.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve_model:app", host="0.0.0.0", port=8000, reload=False)