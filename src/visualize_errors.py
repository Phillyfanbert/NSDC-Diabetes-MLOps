"""
visualize_errors.py — Predicted vs Actual error visualisation
=============================================================
Loads the Production model from MLflow, reconstructs the same
chronological train/test split used during training, and produces
a two-panel scatter plot so train and test errors are visually distinct.

The plot is saved locally AND logged as an MLflow artifact so it is
permanently linked to the training run it describes.

Usage:
    python src/visualize_errors.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import json

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

# ---------------------------------------------------------------------------
# Configuration — must match train_model.py exactly
# ---------------------------------------------------------------------------
# FIX: FEATURES_PATH and SCALE_PARAMS_PATH removed — imported from config.py
from config import (
    MLFLOW_TRACKING, EXPERIMENT_NAME, REGISTERED_MODEL,
    FEATURE_COLS, TARGET_COL, API_URL,
    FEATURES_PATH, SCALE_PARAMS_PATH,
    get_production_run_id,
)

OUTPUT_PATH = Path("predicted_vs_actual.png")

mlflow.set_tracking_uri(MLFLOW_TRACKING)
client = MlflowClient()   # module-level so get_production_run() can use it


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_production_run() -> tuple[str, dict]:
    """
    Resolve the Production run_id via config.get_production_run_id()
    (API first, MLflow registry fallback), then pull the full training
    metrics from MLflow. The health endpoint doesn't expose train_rmse
    or cv_r2_std, so we always go to MLflow for the complete picture.
    """
    run_id = get_production_run_id()   # API → registry fallback in config.py

    run = client.get_run(run_id)
    m   = run.data.metrics
    metrics = {
        "model_type": run.data.tags.get("model_type",  "unknown"),
        "train_r2":   m.get("train_r2",                float("nan")),
        "train_rmse": m.get("train_rmse",               float("nan")),
        "test_r2":    m.get("test_r2",                  float("nan")),
        "test_rmse":  m.get("test_rmse",                float("nan")),
        "cv_r2_mean": m.get("cv_r2_mean",               float("nan")),
        "cv_r2_std":  m.get("cv_r2_std",                float("nan")),
    }
    return run_id, metrics


def chronological_split(
    df: pd.DataFrame,
    split_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Reproduce the exact same chronological split used in train_model.py
    so the visualisation reflects true train vs held-out test performance.
    """
    df_sorted  = df.sort_values("year").reset_index(drop=True)
    train_mask = df_sorted["year"] < split_year

    X = df_sorted[FEATURE_COLS]
    y = df_sorted[TARGET_COL]

    return (
        X[train_mask],  X[~train_mask],
        y[train_mask],  y[~train_mask],
    )


def scatter_panel(
    ax: plt.Axes,
    y_true: pd.Series,
    y_pred: np.ndarray,
    r2: float,
    rmse: float,
    color: str,
    label: str,
) -> None:
    """Draw a single predicted-vs-actual scatter panel with reference line."""
    ax.scatter(
        y_true, y_pred,
        alpha=0.55, color=color,
        edgecolors="white", linewidth=0.4, s=60,
        label=label, zorder=3,
    )

    bound = (
        min(float(y_true.min()), float(y_pred.min())),
        max(float(y_true.max()), float(y_pred.max())),
    )
    ax.plot(bound, bound, "r--", linewidth=1.8, label="Perfect prediction", zorder=4)

    ax.set_xlabel("Actual diabetes prevalence (%)", fontsize=12)
    ax.set_ylabel("Predicted diabetes prevalence (%)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)

    ax.text(
        0.05, 0.95,
        f"R²:   {r2:.4f}\nRMSE: {rmse:.4f}",
        transform=ax.transAxes,
        fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.6),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def visualize_predictions() -> None:

    # ── 1. Get Production run metadata & metrics from MLflow ─────────────────
    run_id, metrics = get_production_run()
    print(f"📊 Visualising Production run: {run_id}")
    print(f"   train  — R²: {metrics['train_r2']:.4f}  RMSE: {metrics['train_rmse']:.4f}")
    print(f"   test   — R²: {metrics['test_r2']:.4f}   RMSE: {metrics['test_rmse']:.4f}")
    print(f"   CV R²  — {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")

    # ── 2. Load features & reproduce the chronological split ─────────────────
    df = pd.read_parquet(FEATURES_PATH, engine="pyarrow").dropna(
        subset=FEATURE_COLS + [TARGET_COL]
    )
    split_year = json.load(open(SCALE_PARAMS_PATH))["_meta"]["split_year"]
    X_train, X_test, y_train, y_test = chronological_split(df, split_year)
    print(f"\nSplit reproduced at year {split_year} "
          f"({len(X_train):,} train / {len(X_test):,} test rows)")

    # ── 3. Load the Production model ─────────────────────────────────────────
    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL}@Production")

    train_preds = model.predict(X_train)
    test_preds  = model.predict(X_test)

    # ── 4. Two-panel figure: train (left) and test (right) ───────────────────
    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(
        "Predicted vs Actual Diabetes Prevalence\n"
        f"Production model · run {run_id[:8]}… · split year {split_year}",
        fontsize=14, fontweight="bold", y=1.01,
    )

    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.30)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    scatter_panel(
        ax1, y_train, train_preds,
        r2=metrics["train_r2"], rmse=metrics["train_rmse"],
        color="steelblue",
        label=f"Train set (n={len(y_train):,})",
    )
    ax1.set_title("Train set (in-sample)", fontsize=13)

    scatter_panel(
        ax2, y_test, test_preds,
        r2=metrics["test_r2"], rmse=metrics["test_rmse"],
        color="darkorange",
        label=f"Test set (n={len(y_test):,})",
    )
    ax2.set_title(f"Test set — held-out (years ≥ {split_year})", fontsize=13)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n✅ Plot saved to {OUTPUT_PATH}")

    # ── 5. Log the plot as an artifact on the Production training run ─────────
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(str(OUTPUT_PATH))
    print(f"   Artifact logged to MLflow run {run_id}")


if __name__ == "__main__":
    visualize_predictions()