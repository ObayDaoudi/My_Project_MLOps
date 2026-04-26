import os
import matplotlib.pyplot as plt
import lightgbm as lgb
from zenml.client import Client

os.makedirs("_assets", exist_ok=True)

client = Client()

# Find the LightGBM run specifically
pipeline = client.get_pipeline("train_pipeline")
lgbm_model = None

for run in pipeline.runs:
    try:
        model = run.steps["train_model"].output.load()
        if isinstance(model, lgb.LGBMRegressor):
            lgbm_model = model
            print(f"Found LightGBM model in run: {run.name}")
            break
    except Exception:
        continue

if lgbm_model is None:
    print("No LightGBM model found in any run.")
else:
    fig, ax = plt.subplots(figsize=(10, 6))
    lgb.plot_importance(lgbm_model, importance_type="gain", ax=ax,
                        title="Feature Importance (Gain) - LightGBM")
    plt.tight_layout()
    plt.savefig("_assets/feature_importance_gain.png", dpi=150)
    print("Saved to _assets/feature_importance_gain.png")
