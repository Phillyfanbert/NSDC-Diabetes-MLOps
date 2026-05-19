"""
outlier_detective.py — Outlier Detective (Prisha)
=================================================
Identifies the "Top 10 Misses" — country-year observations where the
model was most wrong. These are the hidden confounders: countries where
diabetes is very high despite low obesity, or very low despite high obesity,
suggesting factors the model cannot see (diet, genetics, healthcare access,
sugar consumption, etc.).

Two outputs:
  1. outlier_detective_plot.png — two-panel scatter:
       Left  → top 10 under-predictions (model too low)
       Right → top 10 over-predictions  (model too high)
  2. top10_residuals.csv — ranked table of the worst misses

Both are logged to the Production MLflow training run as artifacts.

Usage:
    python src/outlier_detective.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# FIX: FEATURES_PATH and CLEAN_PATH removed — imported from config.py.
# CLEAN_PATH was a local alias for CLEAN_DATA_PATH; now using the canonical name.
from config import (
    MLFLOW_TRACKING, EXPERIMENT_NAME, REGISTERED_MODEL,
    FEATURE_COLS, TARGET_COL, get_production_run_id,
    FEATURES_PATH, CLEAN_DATA_PATH,
)

PLOT_PATH = Path("outlier_detective_plot.png")
CSV_PATH  = Path("top10_residuals.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_data(features_path: Path, clean_path: Path) -> pd.DataFrame:
    """
    Load features and join back the raw obesity value and region from
    the clean file so the residual table is human-readable.
    """
    feat = pd.read_parquet(features_path, engine="pyarrow")
    feat = feat.dropna(subset=FEATURE_COLS + [TARGET_COL])

    # Bring in ParentLocation (WHO region) for richer context
    if clean_path.exists():
        clean = pd.read_parquet(clean_path, engine="pyarrow")[
            ["country_code", "year", "ParentLocation", "feature_obesity"]
        ]
        feat = feat.merge(clean, on=["country_code", "year"], how="left",
                          suffixes=("", "_raw"))

    return feat


def compute_residuals(df: pd.DataFrame, model) -> pd.DataFrame:
    """Add predicted value and residual columns."""
    X = df[FEATURE_COLS]
    df = df.copy()
    df["predicted"]  = model.predict(X)
    df["residual"]   = df[TARGET_COL] - df["predicted"]   # positive = under-prediction
    df["abs_residual"] = df["residual"].abs()
    return df


def top10_by_country(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate residuals by country and return the 10 with the
    highest mean absolute residual — the chronic worst misses.
    """
    agg = (
        df.groupby("country_code")
          .agg(
              mean_residual    =("residual",     "mean"),
              mean_abs_residual=("abs_residual", "mean"),
              n_years          =("residual",     "count"),
              region           =("ParentLocation","first"),
          )
          .reset_index()
          .sort_values("mean_abs_residual", ascending=False)
          .head(10)
    )
    return agg


def print_residual_table(top10: pd.DataFrame) -> None:
    print("\n── Top 10 worst-miss countries ─────────────────────────────")
    print(f"  {'Country':<10} {'Mean residual':>14} {'Mean |resid|':>13} {'N yrs':>6}  Region")
    print("  " + "─" * 60)
    for _, row in top10.iterrows():
        direction = "↑ under" if row["mean_residual"] > 0 else "↓ over"
        print(f"  {row['country_code']:<10} {row['mean_residual']:>+14.4f} "
              f"{row['mean_abs_residual']:>13.4f} {row['n_years']:>6}  "
              f"{row.get('region', '?')}  {direction}")
    print()


def plot_outliers(df: pd.DataFrame, top10: pd.DataFrame, run_id: str) -> None:
    under = top10[top10["mean_residual"] > 0]["country_code"].tolist()
    over  = top10[top10["mean_residual"] < 0]["country_code"].tolist()

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(
        f"Top 10 Worst-Miss Countries — Production run {run_id[:8]}…\n"
        "Countries where obesity alone can't explain diabetes rates",
        fontsize=13, fontweight="bold",
    )

    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.30)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    def scatter(ax, countries, color, title):
        mask = df["country_code"].isin(countries)
        sub  = df[mask]
        ax.scatter(sub["feature_obesity"], sub[TARGET_COL],
                   alpha=0.6, color=color, edgecolors="white",
                   linewidth=0.4, s=55, label="Actual", zorder=3)
        ax.scatter(sub["feature_obesity"], sub["predicted"],
                   alpha=0.4, color="grey", marker="x", s=40,
                   label="Predicted", zorder=2)
        for country in countries:
            pts = sub[sub["country_code"] == country]
            if not pts.empty:
                ax.annotate(country,
                            (pts["feature_obesity"].mean(), pts[TARGET_COL].mean()),
                            fontsize=7, ha="center")
        ax.set_xlabel("Obesity prevalence (%)", fontsize=11)
        ax.set_ylabel("Diabetes prevalence (%)", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)

    scatter(ax1, under, "steelblue",  "Under-predictions (model too low)")
    scatter(ax2, over,  "darkorange", "Over-predictions (model too high)")

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n✅ Plot saved to {PLOT_PATH}")


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------
def run_outlier_detective() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING)

    # ── 1. Identify Production run ────────────────────────────────────────────
    run_id = get_production_run_id()
    print(f"🔍 Outlier Detective — Production run: {run_id}")

    # ── 2. Load data & model ──────────────────────────────────────────────────
    print("📂 Loading features and clean data...")
    df = load_data(FEATURES_PATH, CLEAN_DATA_PATH)  # FIX: CLEAN_PATH → CLEAN_DATA_PATH
    print(f"   {len(df):,} complete rows loaded")

    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL}@Production")

    # ── 3. Compute residuals on the FULL dataset ──────────────────────────────
    df = compute_residuals(df, model)

    # ── 4. Top 10 worst-miss countries ───────────────────────────────────────
    top10 = top10_by_country(df)
    print_residual_table(top10)

    # ── 5. Save CSV ───────────────────────────────────────────────────────────
    top10.to_csv(CSV_PATH, index=False)
    print(f"\n✅ Top 10 residuals table saved to {CSV_PATH}")

    # ── 6. Plot ───────────────────────────────────────────────────────────────
    plot_outliers(df, top10, run_id)

    # ── 7. Log to MLflow ──────────────────────────────────────────────────────
    print(f"\n📎 Logging artifacts to Production run {run_id}...")
    with mlflow.start_run(run_id=run_id):
        worst_under = top10[top10["mean_residual"] > 0]["mean_abs_residual"].max()
        worst_over  = top10[top10["mean_residual"] < 0]["mean_abs_residual"].max()
        mlflow.log_metric("worst_under_prediction", round(float(worst_under), 4))
        mlflow.log_metric("worst_over_prediction",  round(float(worst_over),  4))
        mlflow.log_metric("top10_mean_abs_residual",
                          round(float(top10["mean_abs_residual"].mean()), 4))

        mlflow.set_tag("top10_miss_countries",
                       ", ".join(top10["country_code"].tolist()))

        mlflow.log_artifact(str(PLOT_PATH))
        mlflow.log_artifact(str(CSV_PATH))

    print(f"   Artifacts logged to MLflow run {run_id}")
    print("\n🏁 Outlier detective analysis complete.")


if __name__ == "__main__":
    run_outlier_detective()