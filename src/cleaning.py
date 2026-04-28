import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/diabetes_obesity_raw.parquet")
OUT_PATH = Path("data/clean/diabetes_obesity_clean.parquet")

def clean(df: pd.DataFrame) -> pd.DataFrame:
    metadata_cols = [
        "Id_diabetes", "Id_obesity",
        "IndicatorCode_diabetes", "IndicatorCode_obesity",
        "SpatialDimType_diabetes", "SpatialDimType_obesity",
        "TimeDimType_diabetes", "TimeDimType_obesity",
        "Dim1Type_diabetes", "Dim1Type_obesity",
        "Dim2Type_diabetes", "Dim2Type_obesity",
        "Dim3Type_diabetes", "Dim3Type_obesity",
        "Dim2_diabetes", "Dim2_obesity",
        "Dim3_diabetes", "Dim3_obesity",
        "DataSourceDim_diabetes", "DataSourceDim_obesity",
        "DataSourceDimType_diabetes", "DataSourceDimType_obesity",
        "Comments_diabetes", "Comments_obesity",
        "Date_diabetes", "Date_obesity",
        "TimeDimensionValue_diabetes", "TimeDimensionValue_obesity",
        "TimeDimensionBegin_diabetes", "TimeDimensionBegin_obesity",
        "TimeDimensionEnd_diabetes", "TimeDimensionEnd_obesity",
        "Value_diabetes", "Value_obesity",
        "Dim1",
    ]
    dup_cols = [
        "SpatialDim_diabetes", "SpatialDim_obesity",
        "TimeDim_diabetes", "TimeDim_obesity",
    ]
    drop = [c for c in metadata_cols + dup_cols if c in df.columns]
    df = df.drop(columns=drop)

    if "ParentLocationCode_diabetes" in df.columns and "ParentLocationCode_obesity" in df.columns:
        mismatches = (df["ParentLocationCode_diabetes"] != df["ParentLocationCode_obesity"]).sum()
        if mismatches > 0:
            print(f"WARNING: {mismatches} ParentLocationCode mismatches")
        df = df.rename(columns={"ParentLocationCode_diabetes": "ParentLocationCode"})
        df = df.drop(columns=["ParentLocationCode_obesity"])

    for col in ["target_diabetes", "feature_obesity", "Low_diabetes", "High_diabetes", "Low_obesity", "High_obesity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    return df

def main():
    df = pd.read_parquet(RAW_PATH)
    print(f"Loaded {len(df)} rows x {df.shape[1]} cols from {RAW_PATH}")
    cleaned = clean(df)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(OUT_PATH, index=False, engine="pyarrow")
    print(f"Saved {len(cleaned)} rows x {cleaned.shape[1]} cols to {OUT_PATH}")
    print(f"Remaining columns: {list(cleaned.columns)}")

if __name__ == "__main__":
    main()
