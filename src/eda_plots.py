from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_raw_data(project_root: Path) -> pd.DataFrame:
    raw_dir = project_root / "data" / "raw"

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_dir}\n"
            "Run `python src/fetch_data.py` first."
        )

    parquet_files = list(raw_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found in {raw_dir}\n"
            "Run `python src/fetch_data.py` first."
        )

    raw_path = parquet_files[0]
    print(f"Loading raw data from: {raw_path}")
    return pd.read_parquet(raw_path)


def save_histograms(eda_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(
        eda_df["target_diabetes"].dropna(),
        bins=30,
        kde=True,
        ax=axes[0],
        color="#4C78A8"
    )
    axes[0].set_title("Distribution of Diabetes Prevalence", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Diabetes prevalence (%)")
    axes[0].set_ylabel("Count")

    sns.histplot(
        eda_df["feature_obesity"].dropna(),
        bins=30,
        kde=True,
        ax=axes[1],
        color="#F58518"
    )
    axes[1].set_title("Distribution of Obesity Prevalence", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Obesity prevalence (%)")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(output_dir / "histograms_diabetes_obesity.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_boxplots(eda_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.boxplot(y=eda_df["target_diabetes"], ax=axes[0], color="#4C78A8")
    axes[0].set_title("Boxplot of Diabetes Prevalence", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Diabetes prevalence (%)")

    sns.boxplot(y=eda_df["feature_obesity"], ax=axes[1], color="#F58518")
    axes[1].set_title("Boxplot of Obesity Prevalence", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Obesity prevalence (%)")

    plt.tight_layout()
    plt.savefig(output_dir / "boxplots_diabetes_obesity.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_global_trends(eda_df: pd.DataFrame, output_dir: Path) -> None:
    yearly_avg = eda_df.groupby("year", as_index=False)[["target_diabetes", "feature_obesity"]].mean()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=yearly_avg, x="year", y="target_diabetes", label="Average Diabetes", color="#4C78A8", marker="o")
    sns.lineplot(data=yearly_avg, x="year", y="feature_obesity", label="Average Obesity", color="#F58518", marker="o")
    plt.title("Global Average Obesity and Diabetes Trends Over Time", fontsize=14, fontweight="bold")
    plt.xlabel("Year")
    plt.ylabel("Average prevalence (%)")
    plt.tight_layout()
    plt.savefig(output_dir / "global_trends_combined.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    sns.lineplot(data=yearly_avg, x="year", y="target_diabetes", ax=axes[0], color="#4C78A8", marker="o")
    axes[0].set_title("Global Average Diabetes Prevalence Over Time", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Diabetes prevalence (%)")

    sns.lineplot(data=yearly_avg, x="year", y="feature_obesity", ax=axes[1], color="#F58518", marker="o")
    axes[1].set_title("Global Average Obesity Prevalence Over Time", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Obesity prevalence (%)")

    plt.tight_layout()
    plt.savefig(output_dir / "global_trends_separate.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_regional_trends(eda_df: pd.DataFrame, output_dir: Path) -> None:
    regional_diabetes = (
        eda_df.dropna(subset=["ParentLocation"])
        .groupby(["year", "ParentLocation"], as_index=False)["target_diabetes"]
        .mean()
    )

    plt.figure(figsize=(14, 7))
    sns.lineplot(data=regional_diabetes, x="year", y="target_diabetes", hue="ParentLocation", marker="o")
    plt.title("Regional Diabetes Trends Over Time", fontsize=14, fontweight="bold")
    plt.xlabel("Year")
    plt.ylabel("Average diabetes prevalence (%)")
    plt.legend(title="Region", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "regional_diabetes_trends.png", dpi=300, bbox_inches="tight")
    plt.close()

    regional_obesity = (
        eda_df.dropna(subset=["ParentLocation"])
        .groupby(["year", "ParentLocation"], as_index=False)["feature_obesity"]
        .mean()
    )

    plt.figure(figsize=(14, 7))
    sns.lineplot(data=regional_obesity, x="year", y="feature_obesity", hue="ParentLocation", marker="o")
    plt.title("Regional Obesity Trends Over Time", fontsize=14, fontweight="bold")
    plt.xlabel("Year")
    plt.ylabel("Average obesity prevalence (%)")
    plt.legend(title="Region", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "regional_obesity_trends.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_scatterplots(eda_df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=eda_df,
        x="feature_obesity",
        y="target_diabetes",
        hue="ParentLocation",
        alpha=0.7
    )
    plt.title("Obesity vs Diabetes Prevalence", fontsize=14, fontweight="bold")
    plt.xlabel("Obesity prevalence (%)")
    plt.ylabel("Diabetes prevalence (%)")
    plt.legend(title="Region", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "obesity_vs_diabetes_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()

    latest_year = eda_df["year"].max()
    latest_df = eda_df[eda_df["year"] == latest_year].copy()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=latest_df,
        x="feature_obesity",
        y="target_diabetes",
        hue="ParentLocation",
        alpha=0.8
    )
    plt.title(f"Latest Year ({latest_year}) Obesity vs Diabetes", fontsize=14, fontweight="bold")
    plt.xlabel("Obesity prevalence (%)")
    plt.ylabel("Diabetes prevalence (%)")
    plt.legend(title="Region", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "latest_year_obesity_vs_diabetes.png", dpi=300, bbox_inches="tight")
    plt.close()


def make_country_notes() -> str:
    return (
        "Abbreviation notes (ISO-3 country codes):\n"
        "- NRU = Nauru\n"
        "- COK = Cook Islands\n"
        "- NIU = Niue\n"
        "- WSM = Samoa\n"
        "- TON = Tonga\n"
        "- TUV = Tuvalu\n"
        "- PLW = Palau\n"
        "- KIR = Kiribati\n"
        "- FSM = Federated States of Micronesia\n"
        "- MHL = Marshall Islands\n"
        "- BHS = Bahamas\n"
    )


def save_top10_tables(eda_df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    latest_year = int(eda_df["year"].max())
    latest_df = eda_df[eda_df["year"] == latest_year].copy()

    top10_diabetes = latest_df.sort_values("target_diabetes", ascending=False).head(10).copy()
    top10_obesity = latest_df.sort_values("feature_obesity", ascending=False).head(10).copy()

    rename_map = {
        "country_code": "country_code",
        "year": "year",
        "ParentLocation": "region",
        "target_diabetes": "diabetes_prevalence",
        "feature_obesity": "obesity_prevalence"
    }

    top10_diabetes = top10_diabetes.rename(columns=rename_map)
    top10_obesity = top10_obesity.rename(columns=rename_map)

    top10_diabetes = top10_diabetes[[
        "country_code", "year", "region", "diabetes_prevalence", "obesity_prevalence"
    ]].round({
        "diabetes_prevalence": 3,
        "obesity_prevalence": 3
    })

    top10_obesity = top10_obesity[[
        "country_code", "year", "region", "diabetes_prevalence", "obesity_prevalence"
    ]].round({
        "diabetes_prevalence": 3,
        "obesity_prevalence": 3
    })

    top10_diabetes.to_csv(output_dir / "top10_diabetes_latest_year.csv", index=False)
    top10_obesity.to_csv(output_dir / "top10_obesity_latest_year.csv", index=False)

    print("\n" + "=" * 80)
    print(f"TOP 10 COUNTRIES BY DIABETES PREVALENCE IN {latest_year}")
    print("=" * 80)
    print(top10_diabetes)

    print("\n" + "=" * 80)
    print(f"TOP 10 COUNTRIES BY OBESITY PREVALENCE IN {latest_year}")
    print("=" * 80)
    print(top10_obesity)

    return top10_diabetes, top10_obesity, latest_year


def save_summary_text(
    output_dir: Path,
    df: pd.DataFrame,
    summary_stats: pd.DataFrame,
    top10_diabetes: pd.DataFrame,
    top10_obesity: pd.DataFrame,
    latest_year: int
) -> None:
    diabetes_mean = summary_stats.loc["target_diabetes", "mean"]
    diabetes_median = summary_stats.loc["target_diabetes", "median"]
    obesity_mean = summary_stats.loc["feature_obesity", "mean"]
    obesity_median = summary_stats.loc["feature_obesity", "median"]

    text = f"""Prisha Week 4 EDA Summary
Dataset: WHO diabetes-obesity merged dataset
Observations: {df.shape[0]}
Columns: {df.shape[1]}

Main variables used:
- target_diabetes: diabetes prevalence (%)
- feature_obesity: obesity prevalence (%)
- year: observation year
- ParentLocation: WHO region
- country_code: ISO 3-letter country code

{make_country_notes()}
Summary statistics interpretation:
- Mean = arithmetic average
- Median = middle value after sorting
- Std = standard deviation, showing spread/variability
- Min = smallest value
- Max = largest value
- q1 = 25th percentile
- q3 = 75th percentile

Key findings:
1. Diabetes prevalence has a mean of about {diabetes_mean:.3f} and a median of {diabetes_median:.3f}, suggesting a right-skewed distribution with some higher-prevalence country-year observations.
2. Obesity prevalence has a mean of about {obesity_mean:.3f} and a median of {obesity_median:.3f}, also showing meaningful spread across countries and years.
3. Global average obesity and diabetes both increase over time, which supports the project’s modeling idea of using historical obesity trends to predict diabetes prevalence.
4. Regional trend differences suggest that geography and hidden confounding factors may matter in addition to obesity alone.
5. The obesity vs diabetes scatterplot shows a positive relationship overall, but with visible dispersion, meaning obesity is important but not the only driver of diabetes prevalence.

Latest-year interpretation:
- The top 10 diabetes and obesity tables show the highest-prevalence country-year observations in the latest available year in the dataset ({latest_year}).
- Many of the highest-obesity and highest-diabetes entries come from Pacific Island countries, which suggests a strong regional pattern worth mentioning in the final presentation.

Files produced:
- summary_stats.csv
- histograms_diabetes_obesity.png
- boxplots_diabetes_obesity.png
- global_trends_combined.png
- global_trends_separate.png
- regional_diabetes_trends.png
- regional_obesity_trends.png
- obesity_vs_diabetes_scatter.png
- latest_year_obesity_vs_diabetes.png
- top10_diabetes_latest_year.csv
- top10_obesity_latest_year.csv
- prisha_eda_summary.txt
"""

    with open(output_dir / "prisha_eda_summary.txt", "w", encoding="utf-8") as f:
        f.write(text)


def main() -> None:
    sns.set_theme(style="whitegrid", palette="deep")

    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "outputs" / "eda"
    ensure_dir(output_dir)

    df = load_raw_data(project_root)

    print("=" * 80)
    print("DATASET OVERVIEW")
    print("=" * 80)
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values per column:")
    print(df.isna().sum().sort_values(ascending=False))

    if "country_code" in df.columns:
        print(f"\nUnique country codes: {df['country_code'].nunique()}")

    if "year" in df.columns:
        print(f"Year range: {df['year'].min()} to {df['year'].max()}")

    eda_cols = ["country_code", "year", "ParentLocation", "target_diabetes", "feature_obesity"]
    eda_df = df[eda_cols].copy()

    print("\n" + "=" * 80)
    print("CENTRAL TENDENCY AND SUMMARY STATISTICS")
    print("=" * 80)
    summary_stats = eda_df[["target_diabetes", "feature_obesity"]].agg(
        ["count", "mean", "median", "std", "min", "max"]
    ).T
    summary_stats["q1"] = eda_df[["target_diabetes", "feature_obesity"]].quantile(0.25)
    summary_stats["q3"] = eda_df[["target_diabetes", "feature_obesity"]].quantile(0.75)
    summary_stats = summary_stats.round(3)

    print(summary_stats)
    summary_stats.to_csv(output_dir / "summary_stats.csv")

    print("\nSaving histograms...")
    save_histograms(eda_df, output_dir)

    print("Saving boxplots...")
    save_boxplots(eda_df, output_dir)

    print("Saving global trends...")
    save_global_trends(eda_df, output_dir)

    print("Saving regional trends...")
    save_regional_trends(eda_df, output_dir)

    print("Saving scatterplots...")
    save_scatterplots(eda_df, output_dir)

    print("Saving top-10 tables...")
    top10_diabetes, top10_obesity, latest_year = save_top10_tables(eda_df, output_dir)

    print("Saving summary text...")
    save_summary_text(output_dir, df, summary_stats, top10_diabetes, top10_obesity, latest_year)

    print("\n" + "=" * 80)
    print("KEY EDA INSIGHTS")
    print("=" * 80)
    print("- Mean > median for both diabetes and obesity suggests some right-skew.")
    print("- Global obesity and diabetes increase over time.")
    print("- Regional differences suggest possible hidden confounders.")
    print("- Obesity and diabetes are positively related, but not perfectly.")
    print("- Pacific Island countries appear frequently in the latest-year top-10 tables.")

    print(f"\nEDA outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()