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
FEATURES_PATH   = Path("data/processed/features.parquet")
CLEAN_PATH      = Path("data/clean/diabetes_obesity_clean.parquet")
PLOT_PATH       = Path("outlier_detective_plot.png")
CSV_PATH        = Path("top10_residuals.csv")
MLFLOW_TRACKING = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "NSDC_Diabetes_Project"
REGISTERED_MODEL = "Diabetes_Prevalence_Model"

FEATURE_COLS = [
    "feature_obesity_scaled",
    "obesity_lag_1y_scaled",
    "obesity_lag_2y_scaled",
    "obesity_lag_3y_scaled",
]
TARGET_COL = "target_diabetes"
TEST_SIZE  = 0.20   # must match train_model.py


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_production_run_id() -> str:
    """Ask the live API which run is in Production — single source of truth."""
    import requests
    data = requests.get("http://127.0.0.1:8000/health", timeout=5).json()
    return data["run_id"]


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


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce the same split used in train_model.py."""
    df_sorted = df.sort_values("year").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - TEST_SIZE))
    return df_sorted.iloc[:split_idx], df_sorted.iloc[split_idx:]


def compute_residuals(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Add predicted value and residual columns.
    Residual = Actual − Predicted
      > 0 → model under-predicted (missed high diabetes)
      < 0 → model over-predicted  (missed low diabetes)
    """
    X           = df[FEATURE_COLS].values
    predicted   = model.predict(X)
    df          = df.copy()
    df["predicted_diabetes"] = predicted
    df["residual"]           = df[TARGET_COL] - predicted
    df["abs_residual"]       = np.abs(df["residual"])
    return df


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def top10_by_country(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate residuals by country (mean absolute residual across all years)
    and return the 10 worst-performing countries. Aggregating by country
    is more useful for the showcase than individual year observations,
    since a systematic miss on one country across many years tells a
    clearer story about hidden confounders.
    """
    group_cols = ["country_code"]
    if "ParentLocation" in df.columns:
        group_cols.append("ParentLocation")

    agg = (
        df.groupby(group_cols)
        .agg(
            mean_actual    = (TARGET_COL,            "mean"),
            mean_predicted = ("predicted_diabetes",  "mean"),
            mean_residual  = ("residual",             "mean"),
            mean_abs_residual = ("abs_residual",      "mean"),
            mean_obesity   = ("feature_obesity",      "mean") if "feature_obesity" in df.columns
                             else ("feature_obesity_scaled", "mean"),
            n_years        = (TARGET_COL,             "count"),
        )
        .reset_index()
        .sort_values("mean_abs_residual", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    agg["rank"] = agg.index + 1
    agg["direction"] = agg["mean_residual"].apply(
        lambda r: "Under-predicted (diabetes higher than model expected)"
                  if r > 0 else
                  "Over-predicted  (diabetes lower than model expected)"
    )

    # Round for readability
    for col in ["mean_actual", "mean_predicted", "mean_residual",
                "mean_abs_residual", "mean_obesity"]:
        if col in agg.columns:
            agg[col] = agg[col].round(2)

    return agg


def print_residual_table(top10: pd.DataFrame) -> None:
    print("\n── Top 10 country misses (by mean absolute residual) ───────")
    print(f"  {'Rank':<5} {'Country':<8} {'Region':<28} "
          f"{'Actual':>8} {'Predicted':>10} {'Residual':>10} {'|Residual|':>11}")
    print("  " + "─" * 82)
    for _, row in top10.iterrows():
        region = row.get("ParentLocation", "—")
        print(
            f"  {int(row['rank']):<5} {row['country_code']:<8} {str(region):<28} "
            f"{row['mean_actual']:>8.2f} {row['mean_predicted']:>10.2f} "
            f"{row['mean_residual']:>+10.2f} {row['mean_abs_residual']:>11.2f}"
        )

    print("\n  Interpretation:")
    print("  • Positive residual → model under-predicted diabetes")
    print("    These countries have MORE diabetes than obesity alone predicts.")
    print("    Hidden factors: diet (high sugar/refined carbs), genetics,")
    print("    sedentary lifestyle, limited healthcare access.")
    print("  • Negative residual → model over-predicted diabetes")
    print("    These countries have LESS diabetes than obesity alone predicts.")
    print("    Hidden factors: active lifestyle, diet quality, strong")
    print("    preventive healthcare, younger demographics.")
    print("─" * 60)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def plot_outliers(df_full: pd.DataFrame, top10: pd.DataFrame, run_id: str) -> None:
    """
    Two-panel scatter plot:
      Left  — under-predictions: countries where diabetes was higher than predicted
              (model missed hidden drivers of diabetes)
      Right — over-predictions: countries where diabetes was lower than predicted
              (model overestimated based on obesity alone)
    All other country-years are shown as a faint grey backdrop for context.
    """
    under = top10[top10["mean_residual"] > 0].copy()
    over  = top10[top10["mean_residual"] < 0].copy()

    top10_codes = set(top10["country_code"])

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(
        "Outlier Detective — Top 10 country misses\n"
        f"Countries where obesity alone fails to explain diabetes prevalence"
        f"  ·  run {run_id[:8]}…",
        fontsize=13, fontweight="bold", y=1.01,
    )
    gs   = gridspec.GridSpec(1, 2, figure=fig, wspace=0.32)
    ax_u = fig.add_subplot(gs[0])
    ax_o = fig.add_subplot(gs[1])

    def _draw_panel(ax, highlight_df, color, title, direction_label):
        # Backdrop: all country-year points, faint
        backdrop = df_full[~df_full["country_code"].isin(top10_codes)]
        ax.scatter(
            backdrop["feature_obesity"], backdrop[TARGET_COL],
            alpha=0.10, color="gray", s=15, zorder=1, label="All other countries",
        )

        # Highlighted outlier countries
        for _, row in highlight_df.iterrows():
            mask = df_full["country_code"] == row["country_code"]
            ax.scatter(
                df_full.loc[mask, "feature_obesity"],
                df_full.loc[mask, TARGET_COL],
                alpha=0.75, s=55, color=color, zorder=3,
            )
            # Label with country code at latest year
            latest = df_full.loc[mask].sort_values("year").iloc[-1]
            ax.annotate(
                row["country_code"],
                xy=(latest["feature_obesity"], latest[TARGET_COL]),
                fontsize=8, color=color, fontweight="bold",
                xytext=(4, 2), textcoords="offset points",
            )

        ax.set_xlabel("Obesity prevalence (%)", fontsize=11)
        ax.set_ylabel("Diabetes prevalence (%)", fontsize=11)
        ax.set_title(f"{title}\n({direction_label})", fontsize=11)
        ax.grid(True, alpha=0.2)

    _draw_panel(
        ax_u, under, "#E24B4A",
        "Under-predicted countries",
        "diabetes HIGHER than model expected",
    )
    _draw_panel(
        ax_o, over, "#378ADD",
        "Over-predicted countries",
        "diabetes LOWER than model expected",
    )

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n✅ Outlier plot saved to {PLOT_PATH}")


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------
def run_outlier_detective() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING)
    client = MlflowClient()

    # ── 1. Identify Production run ────────────────────────────────────────────
    run_id = get_production_run_id()
    print(f"🔍 Outlier Detective — Production run: {run_id}")

    # ── 2. Load data & model ──────────────────────────────────────────────────
    print("📂 Loading features and clean data...")
    df = load_data(FEATURES_PATH, CLEAN_PATH)
    print(f"   {len(df):,} complete rows loaded")

    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL}@Production")

    # ── 3. Compute residuals on the FULL dataset ──────────────────────────────
    # We analyse all data (not just test) so we can identify systematic country-
    # level patterns — a country may appear in training and still be a chronic miss.
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
        # Log the worst miss per direction as summary metrics
        worst_under = top10[top10["mean_residual"] > 0]["mean_abs_residual"].max()
        worst_over  = top10[top10["mean_residual"] < 0]["mean_abs_residual"].max()
        mlflow.log_metric("worst_under_prediction", round(float(worst_under), 4))
        mlflow.log_metric("worst_over_prediction",  round(float(worst_over),  4))
        mlflow.log_metric("top10_mean_abs_residual",
                          round(float(top10["mean_abs_residual"].mean()), 4))

        # Log top 10 country codes as a tag for quick reference in the MLflow UI
        mlflow.set_tag("top10_miss_countries",
                       ", ".join(top10["country_code"].tolist()))

        # Log artifacts
        mlflow.log_artifact(str(PLOT_PATH))
        mlflow.log_artifact(str(CSV_PATH))

    print(f"   Artifacts logged to MLflow run {run_id}")
    print("\n🏁 Outlier detective analysis complete.")


if __name__ == "__main__":
    run_outlier_detective()