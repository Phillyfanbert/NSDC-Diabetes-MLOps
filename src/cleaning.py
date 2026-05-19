"""
cleaning.py — Data Preprocessing
=================================
Reads the merged raw Parquet produced by fetch_data.py, strips ~36
columns of technical metadata and duplicate join-remnant identifiers,
converts remaining numeric columns from object/string dtype to float64,
and saves a compressed Parquet file to data/clean/.

The output satisfies the Data Contract enforced by validate_data.py:
    country_code      str   — ISO 3-letter country code
    year              int   — observation year
    target_diabetes   float — diabetes prevalence (%)
    feature_obesity   float — obesity prevalence (%)
    ParentLocation    str   — WHO region label (kept for EDA grouping)

Usage:
    python src/cleaning.py

Pipeline order:
    fetch_data.py  →  cleaning.py  →  validate_data.py  →  features.py  →  train_model.py
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path

import mlflow

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# FIX: RAW_DATA_PATH and CLEAN_DATA_PATH removed — imported from config.py
from config import MLFLOW_TRACKING, EXPERIMENT_NAME, RAW_DATA_PATH, CLEAN_DATA_PATH

# Columns produced by the inner-join in fetch_data.py that carry no
# predictive value — confidence intervals, string-formatted display values,
# API bookkeeping fields, and duplicate identifiers left over from the merge.
#
# The list is explicit so that any new column the WHO API adds in the future
# will NOT be silently kept — it will appear in the "unexpected columns"
# warning and can be reviewed before being added here or to KEEP_COLS.
DROP_COLS = [
    # --- Confidence-interval bounds (not used in modelling) ---
    "Low",
    "High",
    # --- String-formatted display values (e.g. "30.5 [28.0-33.1]") ---
    "Value",
    # --- Dimension metadata (sex filter already applied in fetch_data.py) ---
    "Dim1",                    # always "SEX_BTSX" after filtering
    "Dim1Type",
    "Dim2",
    "Dim2Type",
    "Dim3",
    "Dim3Type",
    # --- Spatial dimension type labels (redundant with country_code) ---
    "SpatialDimType",
    # --- API timestamp / bookkeeping columns ---
    "Date",
    "TimeDimensionBegin",
    "TimeDimensionEnd",
    "TimeDimensionValue",
    "TimeDimType",
    # --- Indicator metadata (same value in every row after filtering) ---
    "IndicatorCode",
    "Indicator",
    # --- Data-source / data-type labels ---
    "DataSourceDimType",
    "DataSourceDim",
    "Comments",
    # --- Duplicate join-remnant identifiers added by pd.merge suffixes ---
    # fetch_data.py merges with suffixes=("_diabetes", "_obesity"),
    # producing *_diabetes / *_obesity pairs for shared columns.
    # We keep the _diabetes variant (arbitrarily) and drop the _obesity copy.
    "Id_diabetes",
    "Id_obesity",
    "IndicatorCode_diabetes",
    "IndicatorCode_obesity",
    "Dim1_diabetes",
    "Dim1_obesity",
    "SpatialDimType_diabetes",
    "SpatialDimType_obesity",
    "TimeDimType_diabetes",
    "TimeDimType_obesity",
    "DataSourceDimType_diabetes",
    "DataSourceDimType_obesity",
    "DataSourceDim_diabetes",
    "DataSourceDim_obesity",
    "Comments_diabetes",
    "Comments_obesity",
    "Date_diabetes",
    "Date_obesity",
    "TimeDimensionBegin_diabetes",
    "TimeDimensionBegin_obesity",
    "TimeDimensionEnd_diabetes",
    "TimeDimensionEnd_obesity",
    "TimeDimensionValue_diabetes",
    "TimeDimensionValue_obesity",
    "Value_diabetes",
    "Value_obesity",
    "Low_diabetes",
    "Low_obesity",
    "High_diabetes",
    "High_obesity",
]

# Columns that must survive the cleaning step.
# validate_data.py checks for exactly these four; ParentLocation is kept
# so downstream EDA scripts can group by WHO region.
KEEP_COLS = [
    "country_code",
    "year",
    "target_diabetes",
    "feature_obesity",
    "ParentLocation",
]


# ---------------------------------------------------------------------------
# Step 1 — Load
# ---------------------------------------------------------------------------
def load_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at '{path}'. Run fetch_data.py first."
        )
    df = pd.read_parquet(path, engine="pyarrow")
    print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols from {path}")
    print(f"  Columns: {df.columns.tolist()}")
    return df


# ---------------------------------------------------------------------------
# Step 2 — Drop metadata columns
# ---------------------------------------------------------------------------
def drop_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove every column listed in DROP_COLS that actually exists in the
    DataFrame. Columns that are absent (e.g. the API changed its schema)
    are skipped gracefully with a warning rather than crashing.

    Also warns about any columns not in either DROP_COLS or KEEP_COLS so
    that schema changes from the WHO API are surfaced immediately.
    """
    cols_before = set(df.columns)

    to_drop = [c for c in DROP_COLS if c in df.columns]
    not_found = [c for c in DROP_COLS if c not in df.columns]
    if not_found:
        print(f"  ℹ️  {len(not_found)} DROP_COLS entries not present in data "
              f"(schema may have changed): {not_found}")

    df = df.drop(columns=to_drop)

    known = set(DROP_COLS) | set(KEEP_COLS)
    unexpected = [c for c in df.columns if c not in known]
    if unexpected:
        print(f"  ⚠️  Unexpected columns (not in DROP_COLS or KEEP_COLS) — "
              f"review and add to the appropriate list: {unexpected}")

    cols_after = set(df.columns)
    print(f"\n  Columns dropped : {len(cols_before) - len(cols_after)}")
    print(f"  Columns remaining: {len(cols_after)}  → {sorted(cols_after)}")
    return df


# ---------------------------------------------------------------------------
# Step 3 — Convert dtypes
# ---------------------------------------------------------------------------
def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert object/string numeric columns to float64 and year to int.
    """
    numeric_cols = ["target_diabetes", "feature_obesity"]
    for col in numeric_cols:
        if col in df.columns:
            before_dtype = df[col].dtype
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if before_dtype != df[col].dtype:
                print(f"  {col}: {before_dtype} → {df[col].dtype}")

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    return df


# ---------------------------------------------------------------------------
# Step 4 — Report missing values (informational only — do not drop here)
# ---------------------------------------------------------------------------
def report_missing(df: pd.DataFrame) -> None:
    """
    We deliberately do NOT drop rows here — that decision belongs
    to validate_data.py (soft warning) and features.py (dropna on lag NaNs).
    """
    counts = df[KEEP_COLS].isnull().sum()
    missing = counts[counts > 0]
    if missing.empty:
        print("\n  ✅ No missing values in core columns.")
    else:
        print("\n  ⚠️  Missing values in core columns (rows kept — "
              "validate_data.py will flag these):")
        for col, n in missing.items():
            print(f"     {col}: {n} missing ({n / len(df) * 100:.1f}%)")


# ---------------------------------------------------------------------------
# Step 5 — Save
# ---------------------------------------------------------------------------
def save_clean(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    print(f"\n✅ Clean data saved to {path}  "
          f"({df.shape[0]:,} rows × {df.shape[1]} cols)")


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------
def run_cleaning_pipeline() -> None:
    """
    Full cleaning layer:
      1. Load raw Parquet from fetch_data.py
      2. Drop ~36 metadata / duplicate-identifier columns
      3. Convert object columns to correct dtypes
      4. Report missing values (informational)
      5. Save compressed Parquet to data/clean/
      6. Log a lightweight MLflow run for audit trail
    """
    print("🚀 Starting Data Cleaning Pipeline...")

    mlflow.set_tracking_uri(MLFLOW_TRACKING)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="data_cleaning") as run:

        # ── Steps 1-4 ───────────────────────────────────────────────────────
        df = load_raw(RAW_DATA_PATH)
        cols_before = df.shape[1]

        print("\nDropping metadata columns...")
        df = drop_metadata_columns(df)

        print("\nConverting dtypes...")
        df = convert_dtypes(df)

        report_missing(df)

        # ── Step 5: Save ─────────────────────────────────────────────────────
        save_clean(df, CLEAN_DATA_PATH)

        # ── Step 6: Log to MLflow ─────────────────────────────────────────────
        mlflow.log_param("raw_data_path",    str(RAW_DATA_PATH))
        mlflow.log_param("clean_data_path",  str(CLEAN_DATA_PATH))
        mlflow.log_param("cols_dropped",     cols_before - df.shape[1])
        mlflow.log_param("cols_kept",        df.shape[1])
        mlflow.log_param("keep_cols",        KEEP_COLS)

        mlflow.log_metric("rows_input",      df.shape[0])
        mlflow.log_metric("cols_before",     cols_before)
        mlflow.log_metric("cols_after",      df.shape[1])

        for col in ["target_diabetes", "feature_obesity"]:
            if col in df.columns:
                mlflow.log_metric(
                    f"missing_{col}", int(df[col].isnull().sum())
                )

        mlflow.set_tag("stage",  "data_cleaning")
        mlflow.set_tag("status", "SUCCESS")

        print("\n" + "=" * 40)
        print("✅ DATA CLEANING SUCCESSFUL")
        print(f"   Raw columns  : {cols_before}")
        print(f"   Clean columns: {df.shape[1]}")
        print(f"   Rows         : {df.shape[0]:,}")
        print(f"   Saved to     : {CLEAN_DATA_PATH}")
        print(f"   MLflow run   : {run.info.run_id}")
        print("=" * 40)


if __name__ == "__main__":
    run_cleaning_pipeline()