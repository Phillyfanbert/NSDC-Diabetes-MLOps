# NSDC Diabetes MLOps Pipeline

An end-to-end MLOps pipeline that predicts global diabetes prevalence using historical obesity trends from the WHO Global Health Observatory (GHO) API. Built by the NSDC Spring Quarter team as a demonstration of production machine learning practices: live data ingestion, automated validation, experiment tracking, model registry, and REST API serving.

---

## What This Project Does

The pipeline pulls live obesity and diabetes data from the WHO API across 200+ countries and multiple decades, engineers obesity **level** and **trend** features to capture both the absolute burden and rate of change of obesity (replacing raw temporal lags that produced VIF > 1,000,000), trains and compares two regression models, automatically promotes the best model to production, and serves predictions through a REST API and interactive dashboard.

A single command runs the entire system from raw data to live predictions:

```bash
bash run_pipeline.sh
```

---

## Prerequisites

Before you start, make sure you have the following installed:

- [Git](https://git-scm.com/)
- [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda)
- Python 3.9

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/Phillyfanbert/NSDC-Diabetes-MLOps.git
cd NSDC-Diabetes-MLOps
```

If you already have the repo, pull the latest changes instead:

```bash
git pull origin main
```

### 2. Create and activate the environment

```bash
conda create -n diabetes_mlops python=3.9 -y
conda activate diabetes_mlops
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the full pipeline

```bash
bash run_pipeline.sh
```

This single command runs all steps in order and opens the dashboard automatically when finished. Press `Ctrl+C` once to shut everything down cleanly.

---

## What the Pipeline Does Step by Step

When you run `bash run_pipeline.sh`, here is what happens:

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `run_pipeline.sh` | Creates the logs directory and starts the MLflow UI in the background |
| 2 | `src/fetch_data.py` | Fetches live diabetes and obesity data from the WHO GHO API and saves a joined Parquet file |
| 3 | `src/cleaning.py` | Drops ~36 metadata columns, converts strings to floats, saves a clean Parquet file |
| 4 | `src/validate_data.py` | Enforces the data contract, halts the pipeline if any check fails |
| 5 | `src/features.py` | Replaces collinear lag features with obesity level (current %) + 3yr trend slope (pp/yr); Z-score scales both; saves `scale_params.json`. This reduced VIF from > 1,000,000 to 1.41 |
| 6 | `src/train_model.py` | Trains Linear Regression and Ridge Regression; logs all metrics and artifacts to MLflow |
| 7 | `src/serve_model.py` | Promotes the best model to Production; starts the FastAPI prediction server |
| 8–11 | Analysis scripts | Runs coefficient analysis, error visualization, VIF/multicollinearity analysis, and outlier detection against the Production model |
| 12 | `src/generate_dashboard.py` | Generates and opens an interactive HTML dashboard connected to the live API |

---

## Viewing Results

### MLflow UI: experiment tracking

Once the pipeline is running, open the MLflow dashboard to inspect every experiment run, compare model metrics, and view logged artifacts (charts, source scripts, scale parameters):

```
http://127.0.0.1:5000
```

Expected results:
- Linear Regression: test R² ≈ 0.603, RMSE ≈ 2.796
- Ridge Regression: test R² ≈ 0.603, RMSE ≈ 2.796 (L2 regularisation; alpha tuned via GridSearchCV with TimeSeriesSplit)
- Ridge is automatically promoted to Production (preferred when within tolerance of Linear Regression, since it handles multicollinearity more robustly)

> **Note:** Earlier pipeline iterations using raw 1/2/3yr lag features produced R² ≈ 0.699, RMSE ≈ 2.17. The switch to level + trend features traded a small accuracy drop for a dramatic reduction in multicollinearity (VIF from > 1,000,000 → 1.41) and stable, interpretable coefficients.

### Prediction API

Send a POST request to get a diabetes prevalence prediction:

```
http://127.0.0.1:8000/predict
```

Example request body:

```json
{
  "obesity_current": 28.5,
  "obesity_trend": 0.45
}
```

Check which model is live and its current metrics:

```
http://127.0.0.1:8000/health
```

### Interactive dashboard

The dashboard opens automatically in your browser at the end of the pipeline run. You can also open `dashboard.html` directly from the project folder. It shows live model metrics and lets you adjust obesity sliders to get real-time predictions.

---

## Running Individual Scripts

If you want to run a single step without the full pipeline, activate the environment first and then run the script directly:

```bash
conda activate diabetes_mlops

python src/fetch_data.py
python src/cleaning.py
python src/validate_data.py
python src/features.py
python src/train_model.py
```

Note: each script depends on the output of the previous one. MLflow must be running before any training scripts, and `features.py` must run before `train_model.py` and `serve_model.py`.

---

## Project Structure

```
NSDC-Diabetes-MLOps/
├── src/
│   ├── fetch_data.py               # WHO API ingestion
│   ├── cleaning.py                 # Data preprocessing
│   ├── validate_data.py            # Data contract enforcement
│   ├── features.py                 # Obesity level + trend features + Z-score scaling
│   ├── train_model.py              # Model training + MLflow logging
│   ├── serve_model.py              # Model promotion + FastAPI server
│   ├── analyze_coefficients.py     # Feature coefficient bar chart
│   ├── visualize_errors.py         # Predicted vs. Actual scatter plot
│   ├── multicollinearity_analysis.py  # VIF + correlation heatmap
│   ├── outlier_detective.py        # Top-10 country residuals
│   ├── eda_plots.py                # EDA visualizations
│   └── generate_dashboard.py       # Interactive HTML dashboard
├── data/
│   ├── raw/                        # WHO API output (auto-generated)
│   ├── clean/                      # Cleaned Parquet (auto-generated)
│   └── processed/                  # Feature-engineered Parquet + scale_params.json
├── logs/                           # MLflow and API server logs
├── src/config.py                   # Shared constants (paths, model name, feature cols)
├── run_pipeline.sh                 # Master orchestrator script
├── requirements.txt
└── .gitignore
```

---

## Key Results & Findings

### Dataset
4,623 complete observations across 201 countries, 1992–2014. Diabetes prevalence ranges from 2.0% to 29.8% (mean 8.0%, std 3.9%, right-skewed). Obesity level correlation with diabetes: r = +0.83 (strong). Obesity trend correlation: r = +0.38 (moderate, independent signal).

### Model performance
| Model | Test R² | Test RMSE |
|-------|---------|-----------|
| Linear Regression | 0.6031 | 2.7959 |
| Ridge Regression (promoted) | 0.6031 | 2.7959 |

Ridge coefficients: **+3.41** for obesity level (higher absolute obesity → higher diabetes risk), **−0.52** for obesity trend. The negative trend coefficient reflects a real pattern: countries with rapidly rising obesity tend to be earlier in their epidemic with lower diagnostic rates, while high-but-stable obesity nations have mature healthcare systems that detect more cases.

### Where the model is most wrong (and why that's informative)
The outlier detective analysis (`src/outlier_detective.py`) surfaces systematic gaps, countries where obesity alone cannot explain diabetes rates:

| Country | Residual | Interpretation |
|---------|----------|----------------|
| Palau | +5.88 pp | Pacific Island nations: thrifty genotype hypothesis + dietary shift to refined carbohydrates independent of measured obesity |
| Nauru | +5.24 pp | Same pattern |
| Bhutan | +4.62 pp | Unique dietary and genetic factors |
| USA | −4.87 pp | High-income country: better metabolic health management, earlier intervention, different obesity subtypes |
| Germany | −4.30 pp | Same pattern |

These are not model failures, they are signals that obesity is a strong but incomplete predictor and that region-specific confounders (genetics, diet composition, healthcare access) matter.

### Lag optimizer experiment
We tested whether longer lag windows (5, 7, 10 years) would improve accuracy over the default 1/2/3yr lags. Result: R² decreased from 0.6989 → 0.6932 and RMSE increased from 2.1737 → 2.2855. Shorter lags were retained.

---

## Key Engineering Decisions

| Decision | Why |
|----------|-----|
| **Live WHO API instead of static CSV** | Static CSVs go stale. A live API means the pipeline always uses current WHO data and demonstrates real MLOps, the system can retrain on new data automatically |
| **Level + trend features instead of raw lags** | Raw 1/2/3yr lag features had VIF > 1,000,000 (r ≈ 0.9999 pairwise), making coefficients unstable. Replacing them with obesity level (current %) and 3yr trend slope reduced VIF to 1.41 and correlation to r = 0.54 |
| **Ridge over Lasso** | Lasso (L1) pushes coefficients to exactly zero, which would silently drop lag-like features. Ridge (L2) shrinks all coefficients together without elimination, preserving interpretability while stabilising them |
| **Inner join on country + year** | An outer join would introduce NaN rows requiring imputation assumptions. Inner join keeps only rows with paired, verified measurements |
| **Parquet instead of CSV** | Parquet preserves column dtypes across scripts, compresses 3–5× smaller, and reads 10–50× faster for columnar queries |
| **TimeSeriesSplit for cross-validation** | Standard k-fold randomly shuffles rows, causing look-ahead bias in time-series data. TimeSeriesSplit always trains on past data and validates on future data |
| **Pipeline-run-id scoping for model promotion** | Promotion logic queries only models from the current pipeline run, preventing stale models from a previous run from being accidentally promoted |

---

## Troubleshooting

**Pipeline fails at validation:**
The WHO API data failed one or more quality checks. Check the console output for which check failed (percentage out of range, duplicate country-year entries, or year out of bounds). Re-running `fetch_data.py` usually resolves transient API issues.

**MLflow UI not loading:**
The MLflow server takes a few seconds to start. Wait 5–10 seconds and refresh. If it still doesn't load, check `logs/mlflow.log` for errors.

**API server not responding:**
Check `logs/api.log`. The most common cause is that `features.py` hasn't been run yet and `scale_params.json` is missing from `data/processed/`.

**Port already in use:**
If you ran the pipeline previously and didn't shut it down cleanly, old processes may still be holding ports 5000 or 8000. Kill them with:

```bash
lsof -ti:5000 | xargs kill -9
lsof -ti:8000 | xargs kill -9
```

---

## Team

| Member | Role |
|--------|------|
| Haruto | Data Cleaning, Coefficient Analysis |
| Aliya | Data Validation, Lag Optimization |
| Ava | EDA, VIF / Multicollinearity Analysis |
| Prisha | EDA Visualizations, Outlier Detection |
| Krish | Feature Engineering, Error Visualization |
| Philbert (Project Lead) | MLflow, Model Serving, Pipeline Orchestration |

---

## Presentation

[NSDC Spring Quarter MLOps Presentation (Google Slides)](https://docs.google.com/presentation/d/1u1C_aP37xGg-HK-rk37yvjgU2-WwfOo9lpElvjNk0gM/edit#slide=id.p)

Covers: what MLOps is, why it matters, how we built the training pipeline, critical decisions we made, and interpretation of results.

---

## Sample Outputs

The pipeline generates these artifacts automatically (excluded from the repo by `.gitignore` since they are reproducible from source):

| Output | Script | Description |
|--------|--------|-------------|
| `coefficient_plot.png` | `analyze_coefficients.py` | Ridge feature coefficients (+3.41 level, −0.52 trend) |
| `multicollinearity_heatmap.png` | `multicollinearity_analysis.py` | Pairwise correlation between features (r = 0.54 after fix) |
| `vif_scores.png` | `multicollinearity_analysis.py` | VIF per feature (1.41 for both, well below threshold) |
| `predicted_vs_actual.png` | `visualize_errors.py` | Scatter plot of model predictions vs. ground truth |
| `outlier_detective_plot.png` | `outlier_detective.py` | Top-10 countries with highest residuals |
| `top10_residuals.csv` | `outlier_detective.py` | Ranked residual table with country, region, mean residual |
| `dashboard.html` | `generate_dashboard.py` | Interactive dashboard with live API sliders |

---

## Data Source

WHO Global Health Observatory (GHO) API
- Diabetes: indicator `NCD_GLUC_04`
- Obesity: indicator `NCD_BMI_30A`
- Coverage: 200+ countries, 1992–2014
```