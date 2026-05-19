"""
analyze_coefficients.py — Linear regression coefficient analysis
================================================================
Loads the Production model from the MLflow Registry (no hardcoded
run ID), extracts feature coefficients, prints a ranked table, and
saves a horizontal bar chart. The plot is logged back to the
originating training run as an MLflow artifact.

Usage:
    python src/analyze_coefficients.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_PATH      = Path("coefficient_plot.png")

from config import (
    MLFLOW_TRACKING, REGISTERED_MODEL, EXPERIMENT_NAME,
    FEATURE_COLS, get_production_run_id,
)

# Human-readable labels for the plot
FEATURE_LABELS = {
    "obesity_level_scaled": "Obesity level (current %)",
    "obesity_trend_scaled": "Obesity trend (pp/year, 3yr slope)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_coef_table(model, run_metrics: dict) -> pd.DataFrame:
    """
    Assemble a tidy coefficient table sorted by absolute magnitude.
    Includes the logged test_r2 so the table is self-contained.
    """
    assert len(FEATURE_COLS) == len(model.coef_), (
        f"Feature count mismatch: {len(FEATURE_COLS)} names "
        f"vs {len(model.coef_)} coefficients"
    )

    df = pd.DataFrame({
        "feature":     FEATURE_COLS,
        "label":       [FEATURE_LABELS[c] for c in FEATURE_COLS],
        "coefficient": model.coef_,
        "abs_coef":    np.abs(model.coef_),
    }).sort_values("abs_coef", ascending=True).reset_index(drop=True)

    return df


def print_coef_table(df: pd.DataFrame, intercept: float, metrics: dict) -> None:
    print("\n── Coefficient analysis ────────────────────────────────────")
    print(f"   Production model metrics — "
          f"train R²: {metrics.get('train_r2', '?')}  |  "
          f"test R²:  {metrics.get('test_r2',  '?')}  |  "
          f"test RMSE: {metrics.get('test_rmse', '?')}")
    print(f"   Intercept: {intercept:+.4f}\n")
    print(f"   {'Feature':<30} {'Coefficient':>12} {'|Coef|':>10}  Direction")
    print("   " + "─" * 60)
    for _, row in df.sort_values("abs_coef", ascending=False).iterrows():
        direction = "↑ positive" if row["coefficient"] > 0 else "↓ negative"
        print(f"   {row['label']:<30} {row['coefficient']:>+12.4f} "
              f"{row['abs_coef']:>10.4f}  {direction}")
    print("─" * 64)


def plot_coefficients(df: pd.DataFrame, metrics: dict, run_id: str) -> None:
    colors = ["steelblue" if c > 0 else "salmon" for c in df["coefficient"]]

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.barh(df["label"], df["coefficient"], color=colors, edgecolor="white",
                   linewidth=0.5, height=0.55)

    for bar, val in zip(bars, df["coefficient"]):
        x_pos = val + (0.5 if val >= 0 else -0.5)
        ha    = "left" if val >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.2f}", va="center", ha=ha, fontsize=10)

    model_type = metrics.get("model_type", "Unknown")          # ← DYNAMIC
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient (weight on z-scored feature)", fontsize=11)
    ax.set_title(
        f"{model_type} — Feature Coefficients\n"               # ← FIXED LINE
        f"test R²={metrics.get('test_r2', '?')}  "
        f"test RMSE={metrics.get('test_rmse', '?')}  "
        f"· run {run_id[:8]}…",
        fontsize=12,
    )

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="steelblue", label="Positive — higher value → higher diabetes risk"),
        Patch(facecolor="salmon",    label="Negative — higher value → lower diabetes risk"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n✅ Coefficient plot saved to {OUTPUT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def analyze_coefficients() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING)
    client = MlflowClient()

    # ── 1. Find the Production run dynamically ────────────────────────────────
    run_id = get_production_run_id()
    print(f"📊 Analysing Production run: {run_id}")

    # ── 2. Pull logged metrics from MLflow (no recomputation) ─────────────────
    run     = client.get_run(run_id)
    metrics = {
        "model_type": run.data.tags.get("model_type", "Unknown"),
        "train_r2":   round(run.data.metrics.get("train_r2",  float("nan")), 4),
        "test_r2":    round(run.data.metrics.get("test_r2",   float("nan")), 4),
        "test_rmse":  round(run.data.metrics.get("test_rmse", float("nan")), 4),
    }

    # ── 3. Load Production model ───────────────────────────────────────────────
    model = mlflow.sklearn.load_model(
        f"models:/{REGISTERED_MODEL}@Production"
    )

    # ── 4. Build table, print, plot ───────────────────────────────────────────
    coef_df = build_coef_table(model, metrics)
    print_coef_table(coef_df, float(model.intercept_), metrics)
    plot_coefficients(coef_df, metrics, run_id)

    # ── 5. Log the plot back to the originating training run ──────────────────
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(str(OUTPUT_PATH))
    print(f"   Artifact logged to MLflow run {run_id}")


if __name__ == "__main__":
    analyze_coefficients()