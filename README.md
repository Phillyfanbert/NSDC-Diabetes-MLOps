# NSDC Diabetes MLOps Pipeline

An end-to-end MLOps pipeline that predicts global diabetes prevalence using historical obesity trends from the WHO Global Health Observatory (GHO) API. Built by the NSDC Spring Quarter team as a demonstration of production machine learning practices: live data ingestion, automated validation, experiment tracking, model registry, and REST API serving.

---

## What This Project Does

The pipeline pulls live obesity and diabetes data from the WHO API across 200+ countries and multiple decades, engineers obesity level and trend features to capture both the absolute burden and rate of change of obesity, trains and compares two regression models, automatically promotes the best model to production, and serves predictions through a REST API and interactive dashboard.

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
| 4 | `src/validate_data.py` | Enforces the data contract — halts the pipeline if any check fails |
| 5 | `src/features.py` | Computes obesity level and 3yr trend slope; Z-score scales both; saves `scale_params.json` |
| 6 | `src/train_model.py` | Trains Linear Regression and Ridge Regression; logs all metrics and artifacts to MLflow |
| 7 | `src/serve_model.py` | Promotes the best model to Production; starts the FastAPI prediction server |
| 8–11 | Analysis scripts | Runs coefficient analysis, error visualization, VIF/multicollinearity analysis, and outlier detection against the Production model |
| 12 | `src/generate_dashboard.py` | Generates and opens an interactive HTML dashboard connected to the live API |

---

## Viewing Results

### MLflow UI — experiment tracking

Once the pipeline is running, open the MLflow dashboard to inspect every experiment run, compare model metrics, and view logged artifacts (charts, source scripts, scale parameters):

```
http://127.0.0.1:5000
```

Expected results:
- Linear Regression: test R² ≈ 0.603, RMSE ≈ 2.796
- Ridge Regression: test R² ≈ 0.603, RMSE ≈ 2.796
- Ridge is automatically promoted to Production

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

## Data Source

WHO Global Health Observatory (GHO) API
- Diabetes: indicator `NCD_GLUC_04`
- Obesity: indicator `NCD_BMI_30A`
- Coverage: 200+ countries, 1992–2014
```