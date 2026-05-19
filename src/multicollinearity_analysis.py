"""
multicollinearity_analysis.py — Redundancy Auditor (Ava)
=========================================================
Checks whether the 1y, 2y, and 3y obesity lag features are too
similar to each other (multicollinearity), which can destabilise
linear regression coefficients even when the overall R² looks fine.

Two complementary diagnostics are produced:
  1. VIF (Variance Inflation Factor) — per-feature redundancy score
  2. Correlation heatmap — pairwise similarity between all model features

Both outputs are saved as PNGs and logged to the Production MLflow
training run as artifacts so they live alongside the model they describe.

Usage:
    python src/multicollinearity_analysis.py

Interpretation guide (printed to console and logged to MLflow):
    VIF = 1          → no multicollinearity
    VIF 1–5          → moderate, generally acceptable
    VIF 5–10         → high, worth investigating
    VIF > 10         → severe, consider dropping the feature
"""
from __future__ import annotations
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from statsmodels.stats.outliers_influence import variance_inflation_factor

import mlflow
from mlflow import MlflowClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FEATURES_PATH    = Path("data/processed/features.parquet")
HEATMAP_PATH     = Path("multicollinearity_heatmap.png")
VIF_PATH         = Path("vif_scores.png")
MLFLOW_TRACKING  = "http://127.0.0.1:5000"
EXPERIMENT_NAME  = "NSDC_Diabetes_Project"

# The four model features — must match train_model.py FEATURE_COLS
MODEL_FEATURE_COLS = [
    "feature_obesity_scaled",
    "obesity_lag_1y_scaled",
    "obesity_lag_2y_scaled",
    "obesity_lag_3y_scaled",
]

# Human-readable labels for plots
LABELS = {
    "feature_obesity_scaled":  "Current obesity",
    "obesity_lag_1y_scaled":   "Lag 1y",
    "obesity_lag_2y_scaled":   "Lag 2y",
    "obesity_lag_3y_scaled":   "Lag 3y",
}

# VIF severity thresholds
VIF_MODERATE = 5
VIF_HIGH     = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_production_run_id() -> str:
    """Ask the live API which run is in Production — single source of truth."""
    import requests
    data = requests.get("http://127.0.0.1:8000/health", timeout=5).json()
    return data["run_id"]


def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Features not found at '{path}'. Run features.py first."
        )
    df = pd.read_parquet(path, engine="pyarrow")
    missing = [c for c in MODEL_FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Features file is missing columns: {missing}")
    return df.dropna(subset=MODEL_FEATURE_COLS)


# ---------------------------------------------------------------------------
# VIF
# ---------------------------------------------------------------------------
def compute_vif(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor for each model feature.
    VIF_i = 1 / (1 - R²_i), where R²_i is from regressing feature i
    on all other features. A high VIF means that feature is nearly
    redundant — other features already explain most of its variance.
    """
    X = df[MODEL_FEATURE_COLS].values
    vif_scores = [
        variance_inflation_factor(X, i)
        for i in range(X.shape[1])
    ]
    vif_df = pd.DataFrame({
        "feature": MODEL_FEATURE_COLS,
        "label":   [LABELS[c] for c in MODEL_FEATURE_COLS],
        "VIF":     vif_scores,
    }).sort_values("VIF", ascending=False).reset_index(drop=True)

    return vif_df


def interpret_vif(vif_df: pd.DataFrame) -> dict:
    """Classify each feature and return a summary for MLflow logging."""
    summary = {}
    print("\n── VIF scores ───────────────────────────────────────────────")
    print(f"  {'Feature':<30} {'VIF':>8}  Severity")
    print("  " + "─" * 52)
    for _, row in vif_df.iterrows():
        vif = row["VIF"]
        if vif > VIF_HIGH:
            severity = "❌ SEVERE  (consider dropping)"
        elif vif > VIF_MODERATE:
            severity = "⚠️  HIGH    (investigate)"
        else:
            severity = "✅ OK"
        print(f"  {row['label']:<30} {vif:>8.2f}  {severity}")
        summary[row["feature"]] = round(vif, 4)
    print()

    max_vif = vif_df["VIF"].max()
    if max_vif > VIF_HIGH:
        print("  ⚠️  At least one feature has severe multicollinearity.")
        print("     The model's coefficients may be unstable.")
        print("     Consider dropping the highest-VIF lag feature.")
    elif max_vif > VIF_MODERATE:
        print("  ⚠️  Moderate multicollinearity detected.")
        print("     Coefficients may be inflated but the model is likely still valid.")
        print("     Monitor coefficient stability across retraining runs.")
    else:
        print("  ✅ All VIF scores are acceptable (< 5).")
        print("     No redundancy concerns for the current feature set.")
    print("─" * 60)

    return summary


def plot_vif(vif_df: pd.DataFrame) -> None:
    """Horizontal bar chart of VIF scores with severity colour coding."""
    colors = [
        "#E24B4A" if v > VIF_HIGH else
        "#EF9F27" if v > VIF_MODERATE else
        "#1D9E75"
        for v in vif_df["VIF"]
    ]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(vif_df["label"], vif_df["VIF"], color=colors,
                   edgecolor="white", linewidth=0.5, height=0.5)

    # Value labels
    for bar, val in zip(bars, vif_df["VIF"]):
        ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=10)

    # Threshold lines
    ax.axvline(VIF_MODERATE, color="#EF9F27", linewidth=1.2,
               linestyle="--", label=f"Moderate threshold (VIF={VIF_MODERATE})")
    ax.axvline(VIF_HIGH,     color="#E24B4A", linewidth=1.2,
               linestyle="--", label=f"Severe threshold (VIF={VIF_HIGH})")

    ax.set_xlabel("Variance Inflation Factor (VIF)", fontsize=11)
    ax.set_title("Multicollinearity — VIF per feature\n"
                 "VIF < 5: OK   |   5–10: High   |   > 10: Severe", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(VIF_PATH, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ VIF chart saved to {VIF_PATH}")


# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------
def compute_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Pearson correlation between all model features."""
    corr = df[MODEL_FEATURE_COLS].corr()
    # Rename index/columns to human labels for readability
    label_map = {c: LABELS[c] for c in MODEL_FEATURE_COLS}
    return corr.rename(index=label_map, columns=label_map)


def interpret_correlation(corr: pd.DataFrame) -> None:
    """Print and interpret the strongest pairwise correlations."""
    print("── Pairwise correlations ────────────────────────────────────")
    # Extract upper triangle (avoid duplicate pairs)
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], corr.iloc[i, j]))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    for a, b, r in pairs:
        flag = ""
        if abs(r) > 0.95:
            flag = " ← ⚠️  very high — may destabilise coefficients"
        elif abs(r) > 0.85:
            flag = " ← ⚠️  high"
        print(f"  {a} ↔ {b}: r = {r:+.4f}{flag}")
    print()

    max_r = max(abs(r) for _, _, r in pairs)
    if max_r > 0.95:
        print("  ⚠️  Very high inter-lag correlation detected (r > 0.95).")
        print("     This is expected for closely-spaced time lags on the same")
        print("     underlying trend. The model still works but coefficient")
        print("     signs may flip between retraining runs — use VIF to decide")
        print("     whether to drop a lag.")
    else:
        print("  ✅ No extreme pairwise correlations (all r < 0.95).")
    print("─" * 60)


def plot_heatmap(corr: pd.DataFrame) -> None:
    """Annotated correlation heatmap for the four model features."""
    fig, ax = plt.subplots(figsize=(7, 5))
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True      # show lower triangle + diagonal

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 11},
        square=True,
    )
    ax.set_title(
        "Feature correlation heatmap\n"
        "Obesity lag features: current, 1y, 2y, 3y",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(HEATMAP_PATH, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Correlation heatmap saved to {HEATMAP_PATH}")


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------
def analyze_multicollinearity() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING)
    client = MlflowClient()

    # ── 1. Load features ──────────────────────────────────────────────────────
    print("📂 Loading features...")
    df = load_features(FEATURES_PATH)
    print(f"   {len(df):,} complete rows loaded\n")

    # ── 2. VIF ────────────────────────────────────────────────────────────────
    vif_df      = compute_vif(df)
    vif_summary = interpret_vif(vif_df)
    plot_vif(vif_df)

    # ── 3. Correlation heatmap ────────────────────────────────────────────────
    corr = compute_correlation(df)
    interpret_correlation(corr)
    plot_heatmap(corr)

    # ── 4. Log everything to the Production training run ──────────────────────
    run_id = get_production_run_id()
    print(f"\n📎 Logging artifacts to Production run {run_id}...")

    with mlflow.start_run(run_id=run_id):
        # Log VIF scores as metrics for trend monitoring
        for col, vif_val in vif_summary.items():
            mlflow.log_metric(f"vif_{col}", vif_val)

        # Log max pairwise correlation as a single summary metric
        corr_vals = corr.values
        np.fill_diagonal(corr_vals, np.nan)
        max_pairwise_r = float(np.nanmax(np.abs(corr_vals)))
        mlflow.log_metric("max_pairwise_correlation", round(max_pairwise_r, 4))

        # Log plot artifacts
        mlflow.log_artifact(str(HEATMAP_PATH))
        mlflow.log_artifact(str(VIF_PATH))

    print(f"   VIF scores and plots logged to MLflow run {run_id}")
    print("\n🏁 Multicollinearity analysis complete.")


if __name__ == "__main__":
    analyze_multicollinearity()