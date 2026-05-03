import pandas as pd
import numpy as np
from pathlib import Path

RAW_DATA_PATH      = Path("data/clean/diabetes_obesity_clean.parquet")
FEATURES_DATA_PATH = Path("data/processed/features.parquet")

# years to lag obesity by — diabetes develops over time so we want past values
LAG_YEARS = [1, 2, 3]


def load_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at '{path}'. Run fetch_data.py first.")
    df = pd.read_parquet(path, engine="fastparquet")
    print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


def select_core(df: pd.DataFrame) -> pd.DataFrame:
    # drop everything except the 4 columns we actually need
    core = df[["country_code", "year", "target_diabetes", "feature_obesity"]].copy()
    core["year"] = core["year"].astype(int)
    core = core.sort_values(["country_code", "year"]).reset_index(drop=True)
    return core


def add_temporal_lags(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    # for each lag, pull obesity value from N years ago for that country
    # this lets the model learn that obesity now -> diabetes later
    df = df.copy()
    for lag in lags:
        col = f"obesity_lag_{lag}y"
        df[col] = df.groupby("country_code")["feature_obesity"].shift(lag)
        print(f"  lag {lag}y: {df[col].notna().sum():,} non-null values")
    return df


def standard_scale(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # z-score each feature: subtract mean, divide by std -> roughly N(0,1)
    # save the params so we can apply the same transform at inference time
    feature_cols = (
        ["feature_obesity", "target_diabetes"]
        + [c for c in df.columns if c.startswith("obesity_lag_")]
    )

    params = {}
    df = df.copy()

    for col in feature_cols:
        mu  = df[col].mean()
        std = df[col].std(ddof=0)
        if std == 0 or np.isnan(std):
            print(f"  skipping '{col}' (zero variance)")
            continue
        df[f"{col}_scaled"] = (df[col] - mu) / std
        params[col] = {"mean": round(mu, 6), "std": round(std, 6)}

    return df, params


def run_eda(df: pd.DataFrame) -> None:
    print("\n--- EDA ---")

    n_countries = df["country_code"].nunique()
    year_min, year_max = df["year"].min(), df["year"].max()
    print(f"Countries: {n_countries}  |  Years: {year_min} - {year_max}")

    # check for missing values
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    if miss.empty:
        print("No missing values")
    else:
        print("\nMissing values:")
        for col, n in miss.items():
            print(f"  {col}: {n} ({n / len(df) * 100:.1f}%)")

    # basic stats
    print("\nDescriptive stats:")
    print(df[["target_diabetes", "feature_obesity"]].describe().round(2).to_string())

    # how correlated are the lags with diabetes — higher is better
    lag_cols = [c for c in df.columns if c.startswith("obesity_lag_")]
    if lag_cols:
        print("\nCorrelation: obesity lags vs diabetes")
        for col in lag_cols:
            r = df[["target_diabetes", col]].dropna().corr().iloc[0, 1]
            print(f"  {col}: r = {r:+.4f}")

    # which countries have the highest average diabetes rates
    print("\nTop 10 countries by mean diabetes prevalence:")
    top10 = (
        df.groupby("country_code")["target_diabetes"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .round(2)
    )
    for rank, (country, val) in enumerate(top10.items(), 1):
        print(f"  {rank}. {country}: {val}%")

    # global trend every 5 years
    print("\nGlobal averages (every 5 years):")
    trend = df.groupby("year")[["target_diabetes", "feature_obesity"]].mean().round(2)
    trend_sample = trend[trend.index % 5 == 0]
    print(f"  {'Year':<8} {'Diabetes':>10} {'Obesity':>10}")
    for yr, row in trend_sample.iterrows():
        print(f"  {yr:<8} {row['target_diabetes']:>10} {row['feature_obesity']:>10}")

    # fit a linear slope per country to find who's rising fastest
    print("\nFastest rising diabetes by country (pp/year):")
    slopes = {}
    for country, grp in df.groupby("country_code"):
        grp = grp.dropna(subset=["target_diabetes"]).sort_values("year")
        if len(grp) >= 5:
            slope = np.polyfit(grp["year"], grp["target_diabetes"], 1)[0]
            slopes[country] = slope
    for country in sorted(slopes, key=slopes.get, reverse=True)[:10]:
        print(f"  {country}: +{slopes[country]:.4f} pp/year")


def save_features(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="fastparquet")
    print(f"\nSaved to {path}  ({df.shape[0]:,} rows x {df.shape[1]} cols)")


def run_feature_pipeline():
    print("Starting feature pipeline...")

    df = load_raw(RAW_DATA_PATH)
    df = select_core(df)

    print(f"\nAdding lags: {LAG_YEARS}")
    df = add_temporal_lags(df, LAG_YEARS)

    print("\nScaling to N(0,1)...")
    df, scale_params = standard_scale(df)

    # print scale params — you'll need these to scale new data at inference time
    print("\nScale params:")
    for col, p in scale_params.items():
        print(f"  {col}: mean={p['mean']}  std={p['std']}")

    run_eda(df)
    save_features(df, FEATURES_DATA_PATH)

    print("\nDone.")
    return df, scale_params


if __name__ == "__main__":
    run_feature_pipeline()