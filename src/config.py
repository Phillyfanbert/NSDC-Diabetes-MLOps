"""
config.py — Shared constants for the NSDC Diabetes MLOps pipeline
=================================================================
Single source of truth for values that were previously copy-pasted
across every script.  Import from here instead of redefining locally:

    from config import (
        MLFLOW_TRACKING, EXPERIMENT_NAME, REGISTERED_MODEL,
        FEATURE_COLS, TARGET_COL, API_URL,
        get_production_run_id,
    )

Changing a value here propagates to every script automatically.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
MLFLOW_TRACKING = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "NSDC_Diabetes_Project"
REGISTERED_MODEL = "Diabetes_Prevalence_Model"

# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------
API_URL = "http://127.0.0.1:8000"

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
RAW_DATA_PATH      = Path("data/raw/diabetes_obesity_raw.parquet")
CLEAN_DATA_PATH    = Path("data/clean/diabetes_obesity_clean.parquet")
FEATURES_PATH      = Path("data/processed/features.parquet")
SCALE_PARAMS_PATH  = Path("data/processed/scale_params.json")

# ---------------------------------------------------------------------------
# Model features
# ---------------------------------------------------------------------------
# Order matters — must match the column order used in features.py and
# the order the model was trained on.  Change LAG_YEARS in features.py
# first; then update this list to match.
FEATURE_COLS = [
    "obesity_level_scaled",   # current obesity level
    "obesity_trend_scaled",   # how fast obesity is rising (pp/year over last 3 yrs)
]

TARGET_COL = "target_diabetes"

# ---------------------------------------------------------------------------
# Shared utility — production run lookup
# ---------------------------------------------------------------------------
def get_production_run_id() -> str:
    """
    Return the MLflow run_id for the currently promoted Production model.

    Strategy:
      1. Ask the live FastAPI /health endpoint — fastest and most current.
      2. If the API is unreachable, fall back to the MLflow Model Registry
         alias lookup — works even when the server is down.

    Raises RuntimeError if neither source can supply a run_id.
    This single implementation replaces the three slightly-different versions
    that previously existed in analyze_coefficients.py, outlier_detective.py,
    and visualize_errors.py.
    """
    import requests
    from mlflow import MlflowClient

    # ── 1. Live API ──────────────────────────────────────────────────────────
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        resp.raise_for_status()
        run_id = resp.json().get("run_id", "")
        if run_id:
            return run_id
        raise ValueError("run_id missing from /health response")
    except Exception as e:
        print(f"  ⚠️  API unreachable ({e}). Trying MLflow registry fallback...")

    # ── 2. MLflow registry alias ─────────────────────────────────────────────
    try:
        mv = MlflowClient(tracking_uri=MLFLOW_TRACKING).get_model_version_by_alias(
            REGISTERED_MODEL, "Production"
        )
        if mv.run_id:
            return mv.run_id
        raise ValueError("run_id missing from registry alias")
    except Exception as e:
        raise RuntimeError(
            f"Could not determine Production run_id from API or MLflow registry: {e}\n"
            "Ensure the pipeline has been run at least once (bash run_pipeline.sh)."
        ) from e