from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FEATURES_PATH     = Path("data/processed/features.parquet")
SCALE_PARAMS_PATH = Path("data/processed/scale_params.json")
MLFLOW_TRACKING   = "http://127.0.0.1:5000"
EXPERIMENT_NAME   = "NSDC_Diabetes_Project"

FEATURE_COLS = [
    "feature_obesity_scaled",
    "obesity_lag_1y_scaled",
    "obesity_lag_2y_scaled",
    "obesity_lag_3y_scaled",
]
TARGET_COL   = "target_diabetes"
TEST_SIZE    = 0.20
RANDOM_STATE = 42
CV_FOLDS     = 5

# Alpha values to search over for Ridge — spans several orders of magnitude
# so we catch both lightly and heavily regularised solutions
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]


# ---------------------------------------------------------------------------
# Load scale params meta (split_year + pipeline_run_id)
# ---------------------------------------------------------------------------
def load_scale_params_meta(path: Path) -> tuple[int, str]:
    """
    Read split_year and pipeline_run_id written by features.py.

    FIX 1: Using the split_year from scale_params.json guarantees that
    train_model.py uses the exact same train/test boundary as features.py,
    even if post-dropna() row counts differ between the two scripts.

    FIX 2: pipeline_run_id is stamped onto every MLflow training run tag
    so serve_model.py can filter promotion to only this pipeline's models,
    preventing stale runs from previous executions from being promoted.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"scale_params.json not found at '{path}'. Run features.py first."
        )
    with open(path) as f:
        data = json.load(f)
    meta = data.get("_meta", {})
    split_year      = int(meta["split_year"])
    pipeline_run_id = str(meta["pipeline_run_id"])
    print(f"  Read from scale_params.json:")
    print(f"    split_year      = {split_year}  (authoritative boundary from features.py)")
    print(f"    pipeline_run_id = {pipeline_run_id}")
    return split_year, pipeline_run_id


# ---------------------------------------------------------------------------
# Load & split
# ---------------------------------------------------------------------------
def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Features not found at '{path}'. Run features.py first."
        )
    df = pd.read_parquet(path, engine="pyarrow")
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Features file is missing expected columns: {missing}")

    before  = len(df)
    df      = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    dropped = before - len(df)
    print(f"Loaded: {len(df):,} complete rows ({dropped:,} dropped for NaN — lag warm-up)")
    return df


def split_data(df: pd.DataFrame, split_year: int) -> tuple[pd.DataFrame, pd.DataFrame,
                                                            pd.Series,    pd.Series,    int]:
    """
    Chronological train/test split using the authoritative split_year read
    from scale_params.json (written by features.py).

    FIX 1: Previously this function recomputed split_year independently from
    features.py using the same 80/20 formula. After dropna() removes NaN
    lag-warmup rows the total row count is smaller here than in features.py,
    so int(len * 0.8) could land on a different year — causing the scaler to
    have been fit on rows that train_model.py placed in the test set.
    Reading the pre-computed year eliminates that drift entirely.
    """
    df_sorted = df.sort_values("year").reset_index(drop=True)

    X = df_sorted[FEATURE_COLS]
    y = df_sorted[TARGET_COL]

    train_mask = df_sorted["year"] < split_year
    X_train, X_test = X[train_mask],  X[~train_mask]
    y_train, y_test = y[train_mask],  y[~train_mask]

    print(f"\nTrain/test split (chronological, split at year {split_year}):")
    print(f"  Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
    return X_train, X_test, y_train, y_test, split_year


# ---------------------------------------------------------------------------
# Shared evaluation helper
# ---------------------------------------------------------------------------
def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, label: str) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    print(f"  {label:<10} — R²: {r2:.4f}  |  RMSE: {rmse:.4f}  |  MAE: {mae:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}


# ---------------------------------------------------------------------------
# Shared MLflow logging helper — used by both models
# ---------------------------------------------------------------------------
def train_and_log(
    model,
    model_type:      str,
    run_name:        str,
    X_train:         pd.DataFrame,
    X_test:          pd.DataFrame,
    y_train:         pd.Series,
    y_test:          pd.Series,
    pipeline_run_id: str,
    extra_params:    dict | None = None,
) -> str:
    """
    Fit a model, evaluate it on train / test / CV, log everything to MLflow,
    and register it in the Model Registry. Returns the MLflow run ID.

    Because serve_model.py promotes by test_r2 DESC, whichever model scores
    higher on the held-out test set automatically becomes Production — no
    manual intervention needed.

    CV uses TimeSeriesSplit to respect temporal ordering within the training
    set — consistent with the chronological train/test split and prevents
    look-ahead bias inside the cross-validation folds.

    FIX 2: pipeline_run_id is stamped as a tag on every run so
    serve_model.py can filter to only this pipeline's models during promotion.
    """
    with mlflow.start_run(run_name=run_name) as run:

        # ── Fit ──────────────────────────────────────────────────────────────
        model.fit(X_train, y_train)

        # ── Metrics ──────────────────────────────────────────────────────────
        print(f"\nMetrics — {run_name}:")
        train_metrics = compute_metrics(y_train, model.predict(X_train), "Train")
        test_metrics  = compute_metrics(y_test,  model.predict(X_test),  "Test")

        # TimeSeriesSplit for CV — respects temporal ordering, prevents
        # look-ahead bias within cross-validation folds.
        tscv = TimeSeriesSplit(n_splits=CV_FOLDS)

        cv_r2   = cross_val_score(
            model, X_train, y_train, cv=tscv, scoring="r2"
        )
        cv_rmse = np.sqrt(-cross_val_score(
            model, X_train, y_train, cv=tscv,
            scoring="neg_mean_squared_error",
        ))
        cv_r2_mean   = float(cv_r2.mean())
        cv_r2_std    = float(cv_r2.std())
        cv_rmse_mean = float(cv_rmse.mean())
        print(f"  CV ({CV_FOLDS}-fold, temporal)  — R²: {cv_r2_mean:.4f} ± {cv_r2_std:.4f}"
              f"  |  RMSE: {cv_rmse_mean:.4f}")

        # ── Coefficients ─────────────────────────────────────────────────────
        print("\n  Coefficients:")
        for feat, coef in zip(FEATURE_COLS, model.coef_):
            print(f"    {feat}: {coef:+.4f}")
        print(f"    intercept: {model.intercept_:+.4f}")

        # ── Log params ───────────────────────────────────────────────────────
        mlflow.log_param("model_type",       model_type)
        mlflow.log_param("feature_cols",     FEATURE_COLS)
        mlflow.log_param("target_col",       TARGET_COL)
        mlflow.log_param("test_size",        TEST_SIZE)
        mlflow.log_param("cv_folds",         CV_FOLDS)
        mlflow.log_param("cv_strategy",      "TimeSeriesSplit")
        mlflow.log_param("n_train",          len(X_train))
        mlflow.log_param("n_test",           len(X_test))
        mlflow.log_param("pipeline_run_id",  pipeline_run_id)
        if extra_params:
            for k, v in extra_params.items():
                mlflow.log_param(k, v)

        # ── Log metrics ──────────────────────────────────────────────────────
        mlflow.log_metric("train_r2",    train_metrics["r2"])
        mlflow.log_metric("train_rmse",  train_metrics["rmse"])
        mlflow.log_metric("train_mae",   train_metrics["mae"])
        mlflow.log_metric("test_r2",     test_metrics["r2"])
        mlflow.log_metric("test_rmse",   test_metrics["rmse"])
        mlflow.log_metric("test_mae",    test_metrics["mae"])
        mlflow.log_metric("cv_r2_mean",  cv_r2_mean)
        mlflow.log_metric("cv_r2_std",   cv_r2_std)
        mlflow.log_metric("cv_rmse_mean", cv_rmse_mean)

        # ── Log coefficients as metrics ───────────────────────────────────────
        for feat, coef in zip(FEATURE_COLS, model.coef_):
            mlflow.log_metric(f"coef_{feat}", coef)
        mlflow.log_metric("intercept", model.intercept_)

        # ── Log source scripts as artifacts ───────────────────────────────────
        for script in ["src/cleaning.py", "src/validate_data.py",
                       "src/features.py", "src/train_model.py"]:
            if Path(script).exists():
                mlflow.log_artifact(script)

        # ── Log model ─────────────────────────────────────────────────────────
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            registered_model_name=f"Diabetes_Prevalence_Model",
        )

        # ── Tags ──────────────────────────────────────────────────────────────
        mlflow.set_tag("stage",           "training")
        mlflow.set_tag("status",          "SUCCESS")
        mlflow.set_tag("model_type",      model_type)
        # FIX 2: stamp pipeline_run_id so serve_model.py can scope promotion
        # to only this pipeline's runs — stale runs from previous executions
        # (potentially trained on different WHO data) are never promoted.
        mlflow.set_tag("pipeline_run_id", pipeline_run_id)

        print(f"\n  ✅ Logged — Run ID: {run.info.run_id}")
        print(f"     Train R²: {train_metrics['r2']:.4f}  |  "
              f"Test R²: {test_metrics['r2']:.4f}  |  "
              f"CV R²: {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")

        return run.info.run_id


# ---------------------------------------------------------------------------
# Model 1 — Linear Regression (OLS baseline)
# ---------------------------------------------------------------------------
def train_linear(X_train, X_test, y_train, y_test, pipeline_run_id: str) -> str:
    print("\n" + "═" * 55)
    print("  Model 1: Linear Regression (OLS baseline)")
    print("═" * 55)
    return train_and_log(
        model            = LinearRegression(),
        model_type       = "LinearRegression",
        run_name         = "Linear_Regression_Baseline",
        X_train          = X_train,
        X_test           = X_test,
        y_train          = y_train,
        y_test           = y_test,
        pipeline_run_id  = pipeline_run_id,
    )


# ---------------------------------------------------------------------------
# Model 2 — Ridge Regression (L2 regularisation)
# ---------------------------------------------------------------------------
def train_ridge(X_train, X_test, y_train, y_test, pipeline_run_id: str) -> str:
    """
    Ridge adds an L2 penalty (alpha * ||w||²) to the OLS loss, which
    directly addresses the multicollinearity diagnosed by the VIF analysis
    (VIF > 1,000,000 across all four lag features).

    Why Ridge and not Lasso:
      Lasso (L1) pushes coefficients to exactly zero, which would silently
      drop lag features that are near-identical (r = 0.9999). That would
      destroy the temporal structure — the whole point of the project.
      Ridge shrinks all four lags together toward zero without eliminating
      any of them, preserving interpretability while stabilising coefficients.

    Alpha tuning:
      GridSearchCV uses TimeSeriesSplit (consistent with train_and_log) to
      find the alpha that maximises CV R² on the training set without
      introducing look-ahead bias. The search range spans four orders of
      magnitude to avoid missing the optimal value. The best alpha is logged
      to MLflow as a parameter.
    """
    print("\n" + "═" * 55)
    print("  Model 2: Ridge Regression (L2 regularisation)")
    print("  Directly addresses multicollinearity from VIF analysis")
    print("═" * 55)

    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)

    print(f"\n  Searching alpha values: {RIDGE_ALPHAS}")
    gs = GridSearchCV(
        Ridge(),
        param_grid = {"alpha": RIDGE_ALPHAS},
        cv         = tscv,
        scoring    = "r2",
        refit      = True,
    )
    gs.fit(X_train, y_train)

    best_alpha    = gs.best_params_["alpha"]
    best_cv_score = gs.best_score_

    print(f"\n  Alpha search results (TimeSeriesSplit CV):")
    for alpha, mean_s, std_s in zip(
        RIDGE_ALPHAS,
        gs.cv_results_["mean_test_score"],
        gs.cv_results_["std_test_score"],
    ):
        marker = "  ← best" if alpha == best_alpha else ""
        print(f"    alpha={str(alpha):<8}  CV R²={mean_s:.4f} ± {std_s:.4f}{marker}")

    print(f"\n  Best alpha: {best_alpha}  (CV R² = {best_cv_score:.4f})")

    return train_and_log(
        model            = gs.best_estimator_,
        model_type       = "Ridge",
        run_name         = "Ridge_Regression_L2",
        X_train          = X_train,
        X_test           = X_test,
        y_train          = y_train,
        y_test           = y_test,
        pipeline_run_id  = pipeline_run_id,
        extra_params     = {
            "alpha":              best_alpha,
            "alpha_search_space": str(RIDGE_ALPHAS),
            "alpha_selection":    f"GridSearchCV_{CV_FOLDS}fold_TimeSeriesSplit_r2",
        },
    )


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------
def run_training_pipeline() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("\nReading authoritative split boundary from features.py output...")
    split_year, pipeline_run_id = load_scale_params_meta(SCALE_PARAMS_PATH)

    df = load_features(FEATURES_PATH)
    X_train, X_test, y_train, y_test, split_year = split_data(df, split_year)

    lr_run_id    = train_linear(X_train, X_test, y_train, y_test, pipeline_run_id)
    ridge_run_id = train_ridge(X_train,  X_test, y_train, y_test, pipeline_run_id)

    print("\n" + "═" * 55)
    print("  Both models logged — MLflow will promote the best")
    print(f"  Linear Regression : {lr_run_id}")
    print(f"  Ridge Regression  : {ridge_run_id}")
    print(f"  pipeline_run_id   : {pipeline_run_id}")
    print("  Run 'bash run_pipeline.sh' to deploy the winner.")
    print("═" * 55)


if __name__ == "__main__":
    run_training_pipeline()