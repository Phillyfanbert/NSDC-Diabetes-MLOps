import pandas as pd
from pathlib import Path

# using file from fetch_data.py
# change path to cleaning.py file later
DATA_PATH = Path("data/raw/diabetes_obesity_raw.parquet")

MIN_YEAR = 1975
MAX_YEAR = 2024

def load_data(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}, run fetch_data.py first or change DATA_PATH"
        )

    if path.suffix == ".csv":
        return pd.read_csv(path)

    if path.suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError("DATA_PATH must be a .csv or .parquet file")


def validate_data(df):
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # check columns needed
    needed_cols = ["country_code", "year", "target_diabetes", "feature_obesity"]

    for col in needed_cols:
        assert col in df.columns, f"Missing column: {col}"

    print("Required columns exist")

    # check for missing values
    missing = df[needed_cols].isnull().sum()
    missing = missing[missing > 0]

    if len(missing) == 0:
        print("No missing values in important columns")
    else:
        print("Missing values found:")
        print(missing)
        print("Report missing values for now, do not automatically drop")

    # check all percentages are between 0 and 100
    percent_cols = ["target_diabetes", "feature_obesity"]

    for col in percent_cols:
        bad_values = df[(df[col] < 0) | (df[col] > 100)]
        assert len(bad_values) == 0, f"{col} has values outside 0-100"

    print("Percentage columns are between 0 and 100")

    # check year column within expected range (1975 - 2024)
    bad_years = df[(df["year"] < MIN_YEAR) | (df["year"] > MAX_YEAR)]
    assert len(bad_years) == 0, f"Some years are outside {MIN_YEAR}-{MAX_YEAR}"

    print(f"Years are between {MIN_YEAR} and {MAX_YEAR}")

    # check combination of country_code + year is unique
    duplicates = df[df.duplicated(subset=["country_code", "year"], keep=False)]
    assert len(duplicates) == 0, "Duplicate country_code + year rows found"

    print("All combinations of country_code + year are unique")

    print("Validation passed")


if __name__ == "__main__":
    df = load_data(DATA_PATH)
    validate_data(df)
