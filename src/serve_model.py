"""
serve_model.py — FastAPI REST API for Diabetes Prevalence Model
===============================================================
Promotes the best MLflow model to Production using a Ridge-preference
strategy: if Ridge's test_r2 is within RIDGE_PREFERENCE_TOLERANCE of
the best overall model, Ridge is promoted — because equal predictive
performance with stable, interpretable coefficients is the better
production decision.

Usage:
    python src/serve_model.py              # promote + start server
    uvicorn serve_model:app --reload       # dev mode

Test with:
    curl -X POST http://127.0.0.1:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"obesity_current": 28.5, "obesity_lag_1y": 27.1,
              "obesity_lag_2y": 25.8, "obesity_lag_3y": 24.3}'
"""
from __future__ import annotations

import json
import logging
import numpy as np
from pathlib import Path
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
MLFLOW_TRACKING   = "http://127.0.0.1:5000"
SCALE_PARAMS_PATH = Path("data/processed/scale_params.json")

# Column order must match FEATURE_COLS in train_model.py exactly
FEATURE_COLS = [
    "feature_obesity_scaled",
    "obesity_lag_1y_scaled",
    "obesity_lag_2y_scaled",
    "obesity_lag_3y_scaled",
]

# If Ridge test_r2 is within this gap of the best model, Ridge wins.
# Rationale: equal predictive power + stable/interpretable coefficients
# is the better production choice over marginally higher raw R².
RIDGE_PREFERENCE_TOLERANCE = 0.01

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(MLFLOW_TRACKING)
client = MlflowClient()


# ---------------------------------------------------------------------------
# Scale params
# ---------------------------------------------------------------------------
def load_scale_params(path: Path) -> dict:
    """Load exact mean/std values from feature engineering. No approximations."""
    if not path.exists():
        raise FileNotFoundError(
            f"Scale params not found at '{path}'. "
            "Run features.py first to generate scale_params.json."
        )
    with open(path) as f:
        params = json.load(f)
    logger.info(f"✅ Scale params loaded from {path}")
    for col, p in params.items():
        logger.info(f"   {col}: mean={p['mean']}, std={p['std']}")
    return params


def scale_value(value: float, col: str, params: dict) -> float:
    """Apply the exact z-score transform used during training."""
    if col not in params:
        raise KeyError(
            f"Column '{col}' not found in scale_params.json. "
            "Re-run features.py to regenerate the params file."
        )
    return (value - params[col]["mean"]) / params[col]["std"]


# ---------------------------------------------------------------------------
# Model promotion — Ridge preference strategy
# ---------------------------------------------------------------------------
def promote_best_model() -> tuple[str, dict]:
    """
    Promotion strategy:

    1. Find the best overall run by test_r2 (any model type).
    2. Find the best Ridge run by test_r2.
    3. If Ridge exists and its test_r2 is within RIDGE_PREFERENCE_TOLERANCE
       of the best overall, promote Ridge.
    4. Otherwise promote the best overall model.

    Why prefer Ridge when performance is equal:
    - OLS coefficients are wildly inflated by multicollinearity (VIF > 1M),
      producing values like -192 and +122 that flip signs between retraining
      runs. This makes the model unreliable and uninterpretable.
    - Ridge's L2 penalty shrinks all coefficients toward zero, producing
      stable, biologically meaningful values that won't flip on retrain.
    - A 0.0002 R² difference is noise. Coefficient stability is not.
    """
    logger.info("🔍 Searching for best training runs…")

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(
            f"Experiment '{EXPERIMENT_NAME}' not found. Run the pipeline first."
        )

    base_filter = "tags.stage = 'training' and tags.status = 'SUCCESS'"

    # Best run overall
    all_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=base_filter,
        order_by=["metrics.test_r2 DESC"],
        max_results=1,
    )
    if not all_runs:
        raise RuntimeError("No successful training runs found. Run train_model.py first.")

    best_overall = all_runs[0]
    best_test_r2 = best_overall.data.metrics.get("test_r2", float("nan"))

    # Best Ridge run
    ridge_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=base_filter + " and tags.model_type = 'Ridge'",
        order_by=["metrics.test_r2 DESC"],
        max_results=1,
    )

    # Decision
    if ridge_runs:
        ridge_run     = ridge_runs[0]
        ridge_test_r2 = ridge_run.data.metrics.get("test_r2", float("nan"))
        gap           = best_test_r2 - ridge_test_r2

        if gap <= RIDGE_PREFERENCE_TOLERANCE:
            selected_run = ridge_run
            reason = (
                f"Ridge promoted — test_r2 gap = {gap:.4f} "
                f"(within tolerance {RIDGE_PREFERENCE_TOLERANCE}). "
                "Equal performance, more stable and interpretable coefficients."
            )
        else:
            selected_run = best_overall
            reason = (
                f"Ridge underperforms by {gap:.4f} R² "
                f"(exceeds tolerance {RIDGE_PREFERENCE_TOLERANCE}). "
                "Promoting best overall model."
            )
    else:
        selected_run = best_overall
        reason = "No Ridge run found — promoting best overall model."

    logger.info(f"📋 {reason}")

    run_id     = selected_run.info.run_id
    metrics    = selected_run.data.metrics
    model_type = selected_run.data.tags.get("model_type", "unknown")

    test_r2   = metrics.get("test_r2",    float("nan"))
    test_rmse = metrics.get("test_rmse",  float("nan"))
    train_r2  = metrics.get("train_r2",   float("nan"))
    cv_r2     = metrics.get("cv_r2_mean", float("nan"))

    logger.info(f"✅ Promoting {model_type} — run {run_id}")
    logger.info(f"   train_r2={train_r2:.4f} | test_r2={test_r2:.4f} "
                f"| cv_r2={cv_r2:.4f} | test_rmse={test_rmse:.4f}")

    # Register if needed
    model_uri = f"runs:/{run_id}/model"
    try:
        mv      = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL)
        version = mv.version
        logger.info(f"📦 Registered as version {version}")
    except Exception:
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL}'")
        version  = next(v.version for v in versions if v.run_id == run_id)
        logger.info(f"📦 Already registered as version {version}")

    client.set_registered_model_alias(
        name=REGISTERED_MODEL,
        alias="Production",
        version=version,
    )
    logger.info(f"🚀 {model_type} v{version} tagged as Production")

    return f"models:/{REGISTERED_MODEL}@Production", {
        "run_id":      run_id,
        "version":     version,
        "model_type":  model_type,
        "train_r2":    round(train_r2,  4),
        "test_r2":     round(test_r2,   4),
        "cv_r2":       round(cv_r2,     4),
        "test_rmse":   round(test_rmse, 4),
        "promotion_reason": reason,
    }


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------
app_state: dict = {
    "model":        None,
    "scale_params": None,
    "model_meta":   None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and scale params once at startup."""
    logger.info("\n⏳ Starting up — loading model and scale params…")
    try:
        app_state["scale_params"] = load_scale_params(SCALE_PARAMS_PATH)
        production_uri, meta      = promote_best_model()
        app_state["model"]        = mlflow.sklearn.load_model(production_uri)
        app_state["model_meta"]   = meta
        logger.info(f"✅ API ready — serving {meta['model_type']} v{meta['version']}\n")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        logger.error("   Ensure MLflow is running and features.py has been executed.")
    yield
    logger.info("🛑 Shutting down API.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Diabetes Prevalence Predictor",
    description=(
        "NSDC MLOps Project — predicts national diabetes prevalence (%) "
        "from current and historical obesity rates. Trained on WHO Global "
        "Health Observatory data. Ridge Regression preferred when performance "
        "is within tolerance of the best model."
    ),
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
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
    model_type:                        str
    model_version:                     str
    train_r2:                          float
    test_r2:                           float
    note:                              str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Diabetes Prevalence API is running."}


@app.get("/health", tags=["Health"])
def health():
    """Model readiness + live metrics for the dashboard."""
    meta = app_state.get("model_meta") or {}
    return {
        "model_loaded":        app_state["model"] is not None,
        "scale_params_loaded": app_state["scale_params"] is not None,
        "registered_model":    REGISTERED_MODEL,
        "alias":               "Production",
        **meta,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: PredictionRequest):
    """
    Predict diabetes prevalence (%) from obesity rates.
    Inputs are z-score scaled using exact training-time parameters.
    """
    model        = app_state["model"]
    scale_params = app_state["scale_params"]
    meta         = app_state["model_meta"] or {}

    if model is None or scale_params is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model or scale params not loaded. "
                "Check that MLflow is running and features.py has been executed."
            ),
        )

    try:
        scaled = np.array([[
            scale_value(request.obesity_current, "feature_obesity", scale_params),
            scale_value(request.obesity_lag_1y,  "obesity_lag_1y",  scale_params),
            scale_value(request.obesity_lag_2y,  "obesity_lag_2y",  scale_params),
            scale_value(request.obesity_lag_3y,  "obesity_lag_3y",  scale_params),
        ]])
    except KeyError as e:
        raise HTTPException(status_code=500, detail=str(e))

    prediction = float(model.predict(scaled)[0])
    prediction = max(0.0, min(prediction, 50.0))

    model_type = meta.get("model_type", "unknown")
    version    = meta.get("version",    "?")

    return PredictionResponse(
        predicted_diabetes_prevalence_pct=round(prediction, 2),
        model_type=model_type,
        model_version=f"{REGISTERED_MODEL}/Production (v{version})",
        train_r2=meta.get("train_r2", float("nan")),
        test_r2=meta.get("test_r2",  float("nan")),
        note=(
            f"Served by {model_type}. Scaled with exact training-time params. "
            f"WHO GHO data. test_rmse={meta.get('test_rmse', '?')}."
        ),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve_model:app", host="0.0.0.0", port=8000, reload=False)