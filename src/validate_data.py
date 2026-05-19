from __future__ import annotations
import sys
import pandas as pd
from pathlib import Path

import mlflow

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# FIX: DATA_PATH (hardcoded) removed — CLEAN_DATA_PATH imported from config.py
from config import MLFLOW_TRACKING, EXPERIMENT_NAME, CLEAN_DATA_PATH

MIN_YEAR        = 1985
MAX_YEAR        = 2024
REQUIRED_COLS   = ["country_code", "year", "target_diabetes", "feature_obesity"]
PERCENT_COLS    = ["target_diabetes", "feature_obesity"]


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Run fetch_data.py then cleaning.py first."
        )
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix}. Use .parquet or .csv.")


# ---------------------------------------------------------------------------
# Individual checks — each returns (passed: bool, detail: str)
# ---------------------------------------------------------------------------
def check_required_columns(df: pd.DataFrame) -> tuple[bool, str]:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {missing}"
    return True, f"All required columns present: {REQUIRED_COLS}"


def check_missing_values(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Reports missing values per required column.
    Does NOT drop rows — missing data is flagged for human review,
    which is the correct MLOps behaviour at the validation stage.
    """
    counts = df[REQUIRED_COLS].isnull().sum()
    missing = counts[counts > 0]
    if missing.empty:
        return True, "No missing values in required columns"
    lines = [f"  {col}: {n} missing ({n / len(df) * 100:.1f}%)" for col, n in missing.items()]
    return False, "Missing values found (rows kept for human review):\n" + "\n".join(lines)


def check_percentage_ranges(df: pd.DataFrame) -> tuple[bool, str]:
    violations = {}
    for col in PERCENT_COLS:
        bad = df[(df[col] < 0) | (df[col] > 100)]
        if not bad.empty:
            violations[col] = len(bad)
    if violations:
        lines = [f"  {col}: {n} out-of-range values" for col, n in violations.items()]
        return False, "Out-of-range percentages:\n" + "\n".join(lines)
    return True, f"All percentage columns within [0, 100]: {PERCENT_COLS}"


def check_year_range(df: pd.DataFrame) -> tuple[bool, str]:
    bad = df[(df["year"] < MIN_YEAR) | (df["year"] > MAX_YEAR)]
    if not bad.empty:
        bad_vals = sorted(bad["year"].unique().tolist())
        return False, f"{len(bad)} rows have years outside [{MIN_YEAR}, {MAX_YEAR}]: {bad_vals[:10]}"
    year_min, year_max = int(df["year"].min()), int(df["year"].max())
    return True, f"Year range valid: {year_min} – {year_max}"


def check_duplicates(df: pd.DataFrame) -> tuple[bool, str]:
    dupes = df[df.duplicated(subset=["country_code", "year"], keep=False)]
    if not dupes.empty:
        examples = dupes[["country_code", "year"]].drop_duplicates().head(5).values.tolist()
        return False, f"{len(dupes)} duplicate (country_code, year) rows found. Examples: {examples}"
    return True, f"All (country_code, year) pairs are unique: {df['country_code'].nunique()} countries"


# ---------------------------------------------------------------------------
# Main validation runner
# ---------------------------------------------------------------------------
def validate_data(df: pd.DataFrame) -> tuple[bool, dict]:
    """
    Run all checks and return (all_pass, results).

    all_pass — True only if every *hard* check passes; False otherwise.
    results  — dict keyed by check name, each value is:
                 {"passed": bool, "detail": str, "hard": bool}

    Missing-value check is a soft warning — it sets passed=False in results
    but does NOT flip all_pass, because WHO data commonly has sparse coverage
    for some countries/years.  Only hard checks gate the pipeline.
    """
    checks = [
        ("required_columns",   check_required_columns,   True),   # hard
        ("missing_values",     check_missing_values,      False),  # soft (warn only)
        ("percentage_ranges",  check_percentage_ranges,   True),   # hard
        ("year_range",         check_year_range,          True),   # hard
        ("no_duplicates",      check_duplicates,          True),   # hard
    ]

    results  = {}
    all_pass = True

    print(f"\nValidating: {CLEAN_DATA_PATH}")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

    for name, fn, is_hard in checks:
        passed, detail = fn(df)
        results[name] = {"passed": passed, "detail": detail, "hard": is_hard}

        icon = "✅" if passed else ("❌" if is_hard else "⚠️ ")
        print(f"{icon} [{name}]")
        print(f"   {detail}\n")

        if not passed and is_hard:
            all_pass = False

    return all_pass, results


# ---------------------------------------------------------------------------
# Pipeline entry point — logs everything to MLflow
# ---------------------------------------------------------------------------
def run_validation_pipeline() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_data(CLEAN_DATA_PATH)

    with mlflow.start_run(run_name="data_validation") as run:

        mlflow.log_param("data_path",  str(CLEAN_DATA_PATH))
        mlflow.log_param("min_year",   MIN_YEAR)
        mlflow.log_param("max_year",   MAX_YEAR)
        mlflow.log_metric("rows",      df.shape[0])
        mlflow.log_metric("columns",   df.shape[1])

        all_pass, results = validate_data(df)

        for name, r in results.items():
            mlflow.log_metric(f"check_{name}", int(r["passed"]))

        for col in REQUIRED_COLS:
            n_missing = int(df[col].isnull().sum())
            mlflow.log_metric(f"missing_{col}", n_missing)

        mlflow.set_tag("stage",  "data_validation")
        mlflow.set_tag("status", "PASSED" if all_pass else "FAILED")

        print("=" * 40)
        if all_pass:
            print("✅ VALIDATION PASSED — pipeline may continue")
            print(f"   MLflow run: {run.info.run_id}")
        else:
            print("❌ VALIDATION FAILED — pipeline halted")
            print(f"   MLflow run: {run.info.run_id}")
            print("   Fix the issues above before re-running.")

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    run_validation_pipeline()