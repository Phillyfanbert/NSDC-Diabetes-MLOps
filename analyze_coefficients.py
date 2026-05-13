import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

RUN_ID = "95039fe096f243fd85eeaca89c8cd1a8"
model = mlflow.sklearn.load_model(f"runs:/{RUN_ID}/model")

# The model was trained on the scaled features only
feature_names = [
    "feature_obesity_scaled",
    "obesity_lag_1y_scaled",
    "obesity_lag_2y_scaled",
    "obesity_lag_3y_scaled",
]

assert len(feature_names) == len(model.coef_), \
    f"Mismatch: {len(feature_names)} names vs {len(model.coef_)} coefs"

coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": model.coef_
}).sort_values("coefficient", key=abs, ascending=True)

print(coef_df)

plt.figure(figsize=(8, 6))
colors = ["steelblue" if c > 0 else "salmon" for c in coef_df["coefficient"]]
plt.barh(coef_df["feature"], coef_df["coefficient"], color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Coefficient (weight)")
plt.title("Linear Regression Feature Importance")
plt.tight_layout()
plt.savefig("coefficient_plot.png", dpi=150)
plt.show()
