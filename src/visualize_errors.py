import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

# 1. Setup MLflow
mlflow.set_experiment("NSDC_Diabetes_Project")

def visualize_predictions():
    # 2. Load the processed data
    data_path = 'data/processed/features.parquet'
    df = pd.read_parquet(data_path)
    df = df.dropna()

    # 3. Define features and target (same as train_model.py)
    X = df[['feature_obesity_scaled', 'obesity_lag_1y_scaled', 
            'obesity_lag_2y_scaled', 'obesity_lag_3y_scaled']]
    y = df['target_diabetes']

    # 4. Load the latest model from MLflow registry
    model = mlflow.sklearn.load_model("models:/Diabetes_Prevalence_Model/latest")

    # 5. Generate predictions, model.predict(x)
    predictions = model.predict(X)

    # 6. Create Predicted vs Actual scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    # scatterplot with individual subplots

    # Scatter plot
    ax.scatter(y, predictions, alpha=0.6, color='steelblue', 
               edgecolors='white', linewidth=0.5, s=80, label='Predictions')

    # 45-degree reference line (perfect predictions)
    min_val = min(y.min(), predictions.min())
    max_val = max(y.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2, label='Perfect Prediction Line')

    # Labels and formatting
    ax.set_xlabel('Actual Diabetes Prevalence', fontsize=13)
    ax.set_ylabel('Predicted Diabetes Prevalence', fontsize=13)
    ax.set_title('Predicted vs Actual Diabetes Prevalence\n(Points far from line = model drift)', 
                 fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add R2 and RMSE to plot
    rmse = np.sqrt(np.mean((y - predictions) ** 2))
    r2 = 1 - (np.sum((y - predictions) ** 2) / np.sum((y - y.mean()) ** 2))
    ax.text(0.05, 0.95, f'RMSE: {rmse:.4f}\nR²: {r2:.4f}', 
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('predicted_vs_actual.png', dpi=150)
    plt.show()
    print("✅ Plot saved as predicted_vs_actual.png!")

if __name__ == "__main__":
    visualize_predictions()