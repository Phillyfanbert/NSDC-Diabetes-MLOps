from __future__ import annotations
import time
import pandas as pd
import requests
from pathlib import Path

import mlflow

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INDICATORS = {
    "target_diabetes": "NCD_GLUC_04",
    "feature_obesity":  "NCD_BMI_30A",
}
RAW_DATA_PATH    = Path("data/raw/diabetes_obesity_raw.parquet")
WHO_BASE_URL     = "https://ghoapi.azureedge.net/api"
MLFLOW_TRACKING  = "http://127.0.0.1:5000"
EXPERIMENT_NAME  = "NSDC_Diabetes_Project"

# Retry configuration
MAX_RETRIES      = 3      # number of attempts per indicator
BACKOFF_SECONDS  = 5      # wait between retries (doubles each attempt)
REQUEST_TIMEOUT  = 30     # seconds before a single request times out


# ---------------------------------------------------------------------------
# Fetch with retry
# ---------------------------------------------------------------------------
def fetch_who_data(code: str) -> pd.DataFrame | None:
    """
    Fetch a WHO GHO indicator and return it as a DataFrame.

    Retries up to MAX_RETRIES times with exponential back-off so that
    a transient network blip does not abort the whole pipeline.
    """
    url = f"{WHO_BASE_URL}/{code}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"📡 Requesting {code} (attempt {attempt}/{MAX_RETRIES})...")
            r = requests.get(url, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            df = pd.DataFrame(r.json()["value"])
            print(f"   ✅ {code}: {len(df):,} rows received")
            return df

        except requests.exceptions.Timeout:
            print(f"   ⏱  Timeout on attempt {attempt}.")
        except requests.exceptions.HTTPError as e:
            print(f"   ❌ HTTP error on attempt {attempt}: {e}")
        except Exception as e:
            print(f"   ❌ Unexpected error on attempt {attempt}: {e}")

        if attempt < MAX_RETRIES:
            wait = BACKOFF_SECONDS * attempt   # 5 s, 10 s, 15 s …
            print(f"   ⏳ Retrying in {wait}s...")
            time.sleep(wait)

    print(f"❌ All {MAX_RETRIES} attempts failed for {code}. Aborting.")
    return None


# ---------------------------------------------------------------------------
# Filter & rename
# ---------------------------------------------------------------------------
def clean_and_prepare(df: pd.DataFrame, value_name: str) -> pd.DataFrame | None:
    """
    Filter for both-sexes rows and standardise join-key column names.
    This enforces the Data Contract expected by validate_data.py.
    """
    if df is None or df.empty:
        return None

    df = df[df["Dim1"] == "SEX_BTSX"].copy()
    df = df.rename(columns={
        "SpatialDim":   "country_code",
        "TimeDim":      "year",
        "NumericValue": value_name,
    })
    return df


# ---------------------------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------------------------
def run_ingestion_pipeline() -> None:
    """
    Full data-ingestion layer:
      1. Fetch each WHO indicator (with retries)
      2. Filter & rename columns
      3. Inner-join on shared keys
      4. Save as Parquet
      5. Log a lightweight MLflow run so the fetch is versioned & auditable
    """
    print("🚀 Starting Data Ingestion Pipeline...")

    mlflow.set_tracking_uri(MLFLOW_TRACKING)
    mlflow.set_experiment(EXPERIMENT_NAME)

    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="data_ingestion") as run:

        # ── 1 & 2: Fetch + prepare ─────────────────────────────────────────
        data_frames: dict[str, pd.DataFrame] = {}
        row_counts:  dict[str, int]           = {}

        for label, code in INDICATORS.items():
            raw_df = fetch_who_data(code)
            prepared = clean_and_prepare(raw_df, label)
            if prepared is not None:
                data_frames[label]  = prepared
                row_counts[label]   = len(prepared)

        if len(data_frames) < 2:
            mlflow.set_tag("status", "FAILED")
            print("❌ Pipeline failed: one or more indicators could not be fetched.")
            return

        # ── 3: Merge ────────────────────────────────────────────────────────
        join_keys = ["country_code", "year", "Dim1", "ParentLocation"]
        print("🔗 Merging datasets...")
        master_df = pd.merge(
            data_frames["target_diabetes"],
            data_frames["feature_obesity"],
            on=join_keys,
            suffixes=("_diabetes", "_obesity"),
        )

        # ── 4: Save ─────────────────────────────────────────────────────────
        master_df.to_parquet(RAW_DATA_PATH, index=False, engine="pyarrow")

        # ── 5: Log to MLflow ────────────────────────────────────────────────
        # Parameters: what we fetched and how
        mlflow.log_param("indicators",       list(INDICATORS.values()))
        mlflow.log_param("sex_filter",       "SEX_BTSX")
        mlflow.log_param("output_path",      str(RAW_DATA_PATH))
        mlflow.log_param("max_retries",      MAX_RETRIES)

        # Metrics: row counts for reproducibility checks
        mlflow.log_metric("rows_diabetes",   row_counts["target_diabetes"])
        mlflow.log_metric("rows_obesity",    row_counts["feature_obesity"])
        mlflow.log_metric("rows_merged",     len(master_df))
        mlflow.log_metric("columns_merged",  master_df.shape[1])

        mlflow.set_tag("stage",  "data_ingestion")
        mlflow.set_tag("status", "SUCCESS")

        print("\n" + "=" * 40)
        print("✅ DATA INGESTION SUCCESSFUL")
        print(f"   Saved to   : {RAW_DATA_PATH}")
        print(f"   Rows       : {len(master_df):,}")
        print(f"   Columns    : {master_df.shape[1]}")
        print(f"   MLflow run : {run.info.run_id}")
        print("=" * 40)


if __name__ == "__main__":
    run_ingestion_pipeline()