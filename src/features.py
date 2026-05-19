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
from config import (
    MLFLOW_TRACKING, EXPERIMENT_NAME,
    CLEAN_DATA_PATH, FEATURES_PATH, SCALE_PARAMS_PATH,
)

TREND_WINDOW = 3

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
            "Run fetch_data.py -> cleaning.py first."
        )
    df = pd.read_parquet(path, engine="pyarrow")
    print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols from {path}")
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
# Step 3 — Trend features  (replaces raw lag columns)
# ---------------------------------------------------------------------------
def add_trend_features(df: pd.DataFrame, window: int = TREND_WINDOW) -> pd.DataFrame:
    """
    Replace raw lags with two interpretable, low-collinearity features:

      obesity_level : the current obesity value (same scale as the raw data)
      obesity_trend : linear slope of obesity over the past `window` years
                      units = percentage points per year
                      positive = rising, negative = falling

    Why this fixes multicollinearity:
      Raw lags (current, 1y, 2y, 3y) all track the same slow-moving trend,
      so r ~= 0.9999 and VIF > 1,000,000.  Level + slope are mathematically
      orthogonal summaries of the same data, giving VIF ~ 1-3.

    Rows without enough prior history to compute the slope get NaN and are
    dropped after this call.
    """
    df = df.copy().sort_values(["country_code", "year"])

    def rolling_slope(series: pd.Series) -> pd.Series:
        """Fit a 1-degree polynomial to the last `window` values, return slope."""
        result = series.copy() * np.nan
        arr = series.values
        for i in range(window - 1, len(arr)):
            y_window = arr[i - window + 1 : i + 1]
            if not np.isnan(y_window).any():
                x_window = np.arange(window, dtype=float)
                result.iloc[i] = np.polyfit(x_window, y_window, 1)[0]
        return result

    df["obesity_level"] = df["feature_obesity"]
    df["obesity_trend"] = (
        df.groupby("country_code")["feature_obesity"]
        .transform(rolling_slope)
    )

    n_valid = int(df["obesity_trend"].notna().sum())
    n_null  = int(df["obesity_trend"].isna().sum())
    print(f"  obesity_level -> {n_valid:,} valid")
    print(f"  obesity_trend -> {n_valid:,} valid, {n_null:,} NaN "
          f"(insufficient history for {window}-yr window)")

    return df


# ---------------------------------------------------------------------------
# Step 4 — Chronological train mask
# ---------------------------------------------------------------------------
def get_train_mask(df: pd.DataFrame, test_size: float = TEST_SIZE) -> tuple[pd.Series, int]:
    """
    Return a boolean Series that is True for the training rows (oldest 80%)
    and False for the test rows (most recent 20%), using the same chronological
    split logic as train_model.py.

    This mask is used to compute scale params on training data only,
    preventing test-set statistics from leaking into the scaler.

    Returns (train_mask, split_year). split_year is written into
    scale_params.json so train_model.py reads the exact same boundary
    instead of recomputing it independently.
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
def standard_scale(df: pd.DataFrame, train_mask: pd.Series) -> tuple[pd.DataFrame, dict]:
    """
    Z-score scale obesity_level and obesity_trend using training-row
    statistics only. Scaled columns are suffixed with _scaled.

    target_diabetes is NOT scaled — train_model.py predicts the raw
    percentage directly.
    """
    scale_cols = ["obesity_level", "obesity_trend"]

    params = {}
    df = df.copy()
    for col in scale_cols:
        train_vals = df.loc[train_mask, col].dropna()
        mu  = float(train_vals.mean())
        std = float(train_vals.std(ddof=0))
        if std == 0 or np.isnan(std):
            print(f"  WARNING: Skipping '{col}' -- zero variance on training rows")
            continue
        df[f"{col}_scaled"] = (df[col] - mu) / std
        params[col] = {"mean": round(mu, 6), "std": round(std, 6)}
        print(f"  {col}: mean={params[col]['mean']}, std={params[col]['std']}  (train rows only)")
    return df, params


def save_scale_params(params: dict, path: Path, split_year: int, pipeline_run_id: str) -> None:
    """
    Persist scale params as JSON so serve_model.py can load the exact
    training-time values instead of using hardcoded approximations.

    split_year written here so train_model.py reads the authoritative
    boundary instead of recomputing it independently.

    pipeline_run_id is stamped into every MLflow training run tag.
    serve_model.py reads it here and filters promotion to only the current
    run's models, preventing stale runs from previous executions from
    being promoted.
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
    print(f"\nScale params saved to {path}")
    print(f"   split_year      = {split_year}  (authoritative boundary for train_model.py)")
    print(f"   pipeline_run_id = {pipeline_run_id}  (used by serve_model.py to filter promotion)")


# ---------------------------------------------------------------------------
# Step 6 — EDA metrics (logged to MLflow, not just printed)
# ---------------------------------------------------------------------------
def compute_eda_metrics(df: pd.DataFrame) -> dict:
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

    for col in ["obesity_level", "obesity_trend"]:
        if col in df.columns:
            r = df[["target_diabetes", col]].dropna().corr().iloc[0, 1]
            metrics[f"corr_{col}_vs_diabetes"] = round(float(r), 4)

    return metrics


def print_eda_summary(df: pd.DataFrame, metrics: dict) -> None:
    print("\n-- EDA summary ---------------------------------------------")
    print(f"   Countries : {metrics['n_countries']}")
    print(f"   Years     : {metrics['year_min']} - {metrics['year_max']}")
    print(f"   Rows total: {metrics['n_rows_total']:,}  |  complete: {metrics['n_rows_complete']:,}")
    print(f"\n   Diabetes  -- mean: {metrics['diabetes_mean']}  std: {metrics['diabetes_std']}"
          f"  range: [{metrics['diabetes_min']}, {metrics['diabetes_max']}]")
    print(f"   Obesity   -- mean: {metrics['obesity_mean']}  std: {metrics['obesity_std']}")

    print("\n   Feature correlations with diabetes (higher = stronger signal):")
    for key, val in metrics.items():
        if key.startswith("corr_"):
            col_name = key.replace("corr_", "").replace("_vs_diabetes", "")
            print(f"     {col_name}: r = {val:+.4f}")

    print("\n   Global averages (every 5 years):")
    trend = df.groupby("year")[["target_diabetes", "feature_obesity"]].mean().round(2)
    trend_5 = trend[trend.index % 5 == 0]
    print(f"   {'Year':<8} {'Diabetes':>10} {'Obesity':>10}")
    for yr, row in trend_5.iterrows():
        print(f"   {yr:<8} {row['target_diabetes']:>10} {row['feature_obesity']:>10}")

    print("\n   Fastest rising diabetes (pp/year):")
    slopes = {}
    for country, grp in df.groupby("country_code"):
        grp = grp.dropna(subset=["target_diabetes"]).sort_values("year")
        if len(grp) >= 5:
            slopes[country] = np.polyfit(grp["year"], grp["target_diabetes"], 1)[0]
    for country in sorted(slopes, key=slopes.get, reverse=True)[:5]:
        print(f"     {country}: +{slopes[country]:.4f} pp/year")
    print("-" * 60)


# ---------------------------------------------------------------------------
# Step 7 — Save feature parquet
# ---------------------------------------------------------------------------
def save_features(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    print(f"\nFeatures saved to {path}  ({df.shape[0]:,} rows x {df.shape[1]} cols)")


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------
def run_feature_pipeline() -> tuple[pd.DataFrame, dict]:
    print("Starting feature engineering pipeline...")

    pipeline_run_id = str(uuid.uuid4())
    print(f"   pipeline_run_id = {pipeline_run_id}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="feature_engineering") as run:

        df = load_clean(CLEAN_DATA_PATH)
        df = select_core(df)

        print("\nAdding trend features (obesity level + 3yr slope)...")
        df = add_trend_features(df, window=TREND_WINDOW)

        # Drop rows where slope couldn't be computed (insufficient history)
        before = len(df)
        df = df.dropna(subset=["obesity_level", "obesity_trend", "target_diabetes"])
        print(f"   Dropped {before - len(df):,} rows (insufficient history for trend window)")

        print("\nDetermining chronological split for leakage-free scaling...")
        train_mask, split_year = get_train_mask(df, TEST_SIZE)

        print("\nScaling features to N(0,1) using training statistics only...")
        df, scale_params = standard_scale(df, train_mask)

        save_scale_params(scale_params, SCALE_PARAMS_PATH, split_year, pipeline_run_id)

        eda_metrics = compute_eda_metrics(df)
        print_eda_summary(df, eda_metrics)

        save_features(df, FEATURES_PATH)

        mlflow.log_param("trend_window",       TREND_WINDOW)
        mlflow.log_param("clean_data_path",    str(CLEAN_DATA_PATH))
        mlflow.log_param("features_path",      str(FEATURES_PATH))
        mlflow.log_param("scale_params_path",  str(SCALE_PARAMS_PATH))
        mlflow.log_param("scaled_columns",     list(scale_params.keys()))
        mlflow.log_param("scale_fit_on",       "train_rows_only")
        mlflow.log_param("scale_split_year",   split_year)
        mlflow.log_param("test_size",          TEST_SIZE)
        mlflow.log_param("pipeline_run_id",    pipeline_run_id)

        for col, p in scale_params.items():
            mlflow.log_param(f"scale_mean_{col}", p["mean"])
            mlflow.log_param(f"scale_std_{col}",  p["std"])

        for metric_name, value in eda_metrics.items():
            mlflow.log_metric(metric_name, value)

        mlflow.log_artifact(str(SCALE_PARAMS_PATH))

        mlflow.set_tag("stage",           "feature_engineering")
        mlflow.set_tag("status",          "SUCCESS")
        mlflow.set_tag("pipeline_run_id", pipeline_run_id)

        print(f"\n   MLflow run: {run.info.run_id}")
        print("Feature pipeline complete.\n")

    return df, scale_params


if __name__ == "__main__":
    run_feature_pipeline()