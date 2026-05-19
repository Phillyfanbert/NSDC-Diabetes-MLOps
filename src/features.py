from __future__ import annotations
import json
import uuid
import numpy as np
import pandas as pd
from pathlib import Path

import mlflow

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CLEAN_DATA_PATH    = Path("data/clean/diabetes_obesity_clean.parquet")
FEATURES_DATA_PATH = Path("data/processed/features.parquet")
SCALE_PARAMS_PATH  = Path("data/processed/scale_params.json")
MLFLOW_TRACKING    = "http://127.0.0.1:5000"
EXPERIMENT_NAME    = "NSDC_Diabetes_Project"

# Lag windows — changing this list is the only thing needed to try new windows
LAG_YEARS = [1, 2, 3]

# Must match TEST_SIZE in train_model.py so the split year is identical
TEST_SIZE = 0.20

CORE_COLS = ["country_code", "year", "target_diabetes", "feature_obesity"]


# ---------------------------------------------------------------------------
# Step 1 — Load
# ---------------------------------------------------------------------------
def load_clean(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Clean data not found at '{path}'. "
            "Run fetch_data.py → cleaning.py first."
        )
    df = pd.read_parquet(path, engine="pyarrow")
    print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols from {path}")
    return df


# ---------------------------------------------------------------------------
# Step 2 — Select core columns
# ---------------------------------------------------------------------------
def select_core(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the four columns the feature pipeline needs."""
    missing = [c for c in CORE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Clean data is missing expected columns: {missing}")
    core = df[CORE_COLS].copy()
    core["year"] = core["year"].astype(int)
    core = core.sort_values(["country_code", "year"]).reset_index(drop=True)
    return core


# ---------------------------------------------------------------------------
# Step 3 — Temporal lags
# ---------------------------------------------------------------------------
def add_temporal_lags(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """
    For each lag N, pull each country's obesity value from N years prior.
    Rows without enough history get NaN — they are dropped before training.
    """
    df = df.copy()
    for lag in lags:
        col = f"obesity_lag_{lag}y"
        df[col] = df.groupby("country_code")["feature_obesity"].shift(lag)
        n_valid = int(df[col].notna().sum())
        n_null  = int(df[col].isna().sum())
        print(f"  lag {lag}y → {n_valid:,} valid, {n_null:,} NaN (insufficient history)")
    return df


# ---------------------------------------------------------------------------
# Step 4 — Chronological train mask
# ---------------------------------------------------------------------------
def get_train_mask(df: pd.DataFrame, test_size: float = TEST_SIZE) -> tuple[pd.Series, int]:
    """
    Return a boolean Series that is True for the training rows (oldest 80%)
    and False for the test rows (most recent 20%), using the same chronological
    split logic as train_model.py.

    This mask is used to compute scale params on training data only —
    preventing test-set statistics from leaking into the scaler.

    Returns (train_mask, split_year). split_year is written into
    scale_params.json so train_model.py reads the exact same boundary
    instead of recomputing it independently — fixing the potential 1-year
    drift caused by differing post-dropna() row counts between the two scripts.
    """
    df_sorted  = df.sort_values("year").reset_index(drop=True)
    split_idx  = int(len(df_sorted) * (1 - test_size))
    split_year = int(df_sorted.iloc[split_idx]["year"])

    train_mask = df["year"] < split_year
    n_train = int(train_mask.sum())
    n_test  = int((~train_mask).sum())
    print(f"\n  Scale params will be fit on training rows only (year < {split_year})")
    print(f"  Train rows: {n_train:,}  |  Test rows (held out of scaler): {n_test:,}")
    return train_mask, split_year


# ---------------------------------------------------------------------------
# Step 5 — Standard scaling  (train-only params, applied to full dataset)
# ---------------------------------------------------------------------------
def standard_scale(
    df: pd.DataFrame,
    train_mask: pd.Series,
) -> tuple[pd.DataFrame, dict]:
    """
    Z-score scale every feature column using statistics computed on training
    rows only, then apply those same params to the entire dataset.

    Scale params are fit on train rows only (year < split_year), then applied
    uniformly, matching how a real production scaler would work.

    The saved scale_params.json contains these train-only stats, which
    serve_model.py loads verbatim for inference — preserving training-serving
    parity without any approximation.

    Note: target_diabetes is NOT scaled here. train_model.py predicts the
    raw percentage directly, so scaling the target would require inverse-
    transforming every prediction. Keeping it unscaled keeps inference simple.
    """
    scale_cols = (
        ["feature_obesity"]
        + [c for c in df.columns if c.startswith("obesity_lag_") and "_scaled" not in c]
    )

    params = {}
    df = df.copy()

    for col in scale_cols:
        train_vals = df.loc[train_mask, col].dropna()

        mu  = float(train_vals.mean())
        std = float(train_vals.std(ddof=0))

        if std == 0 or np.isnan(std):
            print(f"  ⚠️  Skipping '{col}' — zero variance on training rows, cannot scale")
            continue

        df[f"{col}_scaled"] = (df[col] - mu) / std
        params[col] = {"mean": round(mu, 6), "std": round(std, 6)}
        print(f"  {col}: mean={params[col]['mean']}, std={params[col]['std']}  (train rows only)")

    return df, params


def save_scale_params(params: dict, path: Path, split_year: int, pipeline_run_id: str) -> None:
    """
    Persist scale params as JSON so serve_model.py can load the exact
    training-time values instead of using hardcoded approximations.

    FIX 1 — split_year is now written here so train_model.py reads the
    authoritative boundary instead of recomputing it independently. This
    closes the potential 1-year drift caused by differing post-dropna()
    row counts between features.py and train_model.py.

    FIX 2 — pipeline_run_id is a UUID generated once per pipeline execution
    and stamped into every MLflow training run tag. serve_model.py reads it
    here and filters promotion to only the current run's models, preventing
    stale runs from previous pipeline executions from being promoted.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_meta": {
            "split_year":      split_year,
            "pipeline_run_id": pipeline_run_id,
        },
        **params,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n💾 Scale params saved to {path}")
    print(f"   split_year      = {split_year}  (authoritative boundary for train_model.py)")
    print(f"   pipeline_run_id = {pipeline_run_id}  (used by serve_model.py to filter promotion)")


# ---------------------------------------------------------------------------
# Step 6 — EDA metrics (logged to MLflow, not just printed)
# ---------------------------------------------------------------------------
def compute_eda_metrics(df: pd.DataFrame) -> dict:
    """
    Compute key EDA statistics and return them as a flat dict so they can
    be logged to MLflow as metrics — making every pipeline run comparable.
    """
    lag_cols = [c for c in df.columns if c.startswith("obesity_lag_") and "_scaled" not in c]

    metrics = {
        "n_countries":        int(df["country_code"].nunique()),
        "year_min":           int(df["year"].min()),
        "year_max":           int(df["year"].max()),
        "n_rows_total":       len(df),
        "n_rows_complete":    int(df.dropna().shape[0]),
        "diabetes_mean":      round(float(df["target_diabetes"].mean()), 4),
        "diabetes_std":       round(float(df["target_diabetes"].std()), 4),
        "diabetes_min":       round(float(df["target_diabetes"].min()), 4),
        "diabetes_max":       round(float(df["target_diabetes"].max()), 4),
        "obesity_mean":       round(float(df["feature_obesity"].mean()), 4),
        "obesity_std":        round(float(df["feature_obesity"].std()), 4),
    }

    # Lag correlations with diabetes — a useful feature-quality signal over time
    for col in lag_cols:
        r = df[["target_diabetes", col]].dropna().corr().iloc[0, 1]
        metrics[f"corr_{col}_vs_diabetes"] = round(float(r), 4)

    return metrics


def print_eda_summary(df: pd.DataFrame, metrics: dict) -> None:
    print("\n── EDA summary ─────────────────────────────────────────────")
    print(f"   Countries : {metrics['n_countries']}")
    print(f"   Years     : {metrics['year_min']} – {metrics['year_max']}")
    print(f"   Rows total: {metrics['n_rows_total']:,}  |  complete: {metrics['n_rows_complete']:,}")
    print(f"\n   Diabetes  — mean: {metrics['diabetes_mean']}  std: {metrics['diabetes_std']}"
          f"  range: [{metrics['diabetes_min']}, {metrics['diabetes_max']}]")
    print(f"   Obesity   — mean: {metrics['obesity_mean']}  std: {metrics['obesity_std']}")

    print("\n   Lag correlations with diabetes (higher = stronger signal):")
    for key, val in metrics.items():
        if key.startswith("corr_"):
            col_name = key.replace("corr_", "").replace("_vs_diabetes", "")
            print(f"     {col_name}: r = {val:+.4f}")

    # Global trend every 5 years
    print("\n   Global averages (every 5 years):")
    trend = df.groupby("year")[["target_diabetes", "feature_obesity"]].mean().round(2)
    trend_5 = trend[trend.index % 5 == 0]
    print(f"   {'Year':<8} {'Diabetes':>10} {'Obesity':>10}")
    for yr, row in trend_5.iterrows():
        print(f"   {yr:<8} {row['target_diabetes']:>10} {row['feature_obesity']:>10}")

    # Fastest rising countries
    print("\n   Fastest rising diabetes (pp/year):")
    slopes = {}
    for country, grp in df.groupby("country_code"):
        grp = grp.dropna(subset=["target_diabetes"]).sort_values("year")
        if len(grp) >= 5:
            slopes[country] = np.polyfit(grp["year"], grp["target_diabetes"], 1)[0]
    for country in sorted(slopes, key=slopes.get, reverse=True)[:5]:
        print(f"     {country}: +{slopes[country]:.4f} pp/year")
    print("─" * 60)


# ---------------------------------------------------------------------------
# Step 7 — Save feature parquet
# ---------------------------------------------------------------------------
def save_features(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    print(f"\n✅ Features saved to {path}  ({df.shape[0]:,} rows × {df.shape[1]} cols)")


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------
def run_feature_pipeline() -> tuple[pd.DataFrame, dict]:
    print("🚀 Starting feature engineering pipeline...")

    # FIX 2: generate a UUID once here — the same ID will be stamped into
    # scale_params.json AND into every MLflow training run tag so
    # serve_model.py can filter promotion to only this pipeline's models.
    pipeline_run_id = str(uuid.uuid4())
    print(f"   pipeline_run_id = {pipeline_run_id}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="feature_engineering") as run:

        # ── Steps ───────────────────────────────────────────────────────────
        df = load_clean(CLEAN_DATA_PATH)
        df = select_core(df)

        print(f"\nAdding temporal lags: {LAG_YEARS}")
        df = add_temporal_lags(df, LAG_YEARS)

        # Determine train/test boundary before scaling so scale params
        # are fit on training rows only — no test-set leakage into the scaler.
        print("\nDetermining chronological split for leakage-free scaling...")
        train_mask, split_year = get_train_mask(df, TEST_SIZE)

        print("\nScaling features to N(0,1) using training statistics only...")
        df, scale_params = standard_scale(df, train_mask)

        # FIX 1 + 2: persist split_year and pipeline_run_id into the JSON so
        # downstream scripts read the authoritative values instead of
        # recomputing them independently.
        save_scale_params(scale_params, SCALE_PARAMS_PATH, split_year, pipeline_run_id)

        # ── EDA ─────────────────────────────────────────────────────────────
        eda_metrics = compute_eda_metrics(df)
        print_eda_summary(df, eda_metrics)

        # ── Save features ────────────────────────────────────────────────────
        save_features(df, FEATURES_DATA_PATH)

        # ── Log to MLflow ────────────────────────────────────────────────────
        mlflow.log_param("lag_years",          LAG_YEARS)
        mlflow.log_param("clean_data_path",    str(CLEAN_DATA_PATH))
        mlflow.log_param("features_path",      str(FEATURES_DATA_PATH))
        mlflow.log_param("scale_params_path",  str(SCALE_PARAMS_PATH))
        mlflow.log_param("scaled_columns",     list(scale_params.keys()))
        mlflow.log_param("scale_fit_on",       "train_rows_only")
        mlflow.log_param("scale_split_year",   split_year)
        mlflow.log_param("test_size",          TEST_SIZE)
        mlflow.log_param("pipeline_run_id",    pipeline_run_id)

        # Log scale params so every run's exact scalers are in MLflow
        for col, p in scale_params.items():
            mlflow.log_param(f"scale_mean_{col}", p["mean"])
            mlflow.log_param(f"scale_std_{col}",  p["std"])

        # Log EDA metrics for trend monitoring across pipeline runs
        for metric_name, value in eda_metrics.items():
            mlflow.log_metric(metric_name, value)

        # Log the scale params JSON as an artifact for full reproducibility
        mlflow.log_artifact(str(SCALE_PARAMS_PATH))

        mlflow.set_tag("stage",           "feature_engineering")
        mlflow.set_tag("status",          "SUCCESS")
        mlflow.set_tag("pipeline_run_id", pipeline_run_id)

        print(f"\n   MLflow run: {run.info.run_id}")
        print("🏁 Feature pipeline complete.\n")

    return df, scale_params


if __name__ == "__main__":
    run_feature_pipeline()