import mlflow
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from zenml.client import Client

MLFLOW_TRACKING_URI = (
    "/home/obay/.config/zenml/local_stores/"
    "7a28ef06-49bc-4199-b8e6-a0f78079f476/mlruns"
)


def main():
    st.title("Customer Satisfaction Prediction Pipeline")
    st.markdown(
        """
        This application is part of an MSc thesis project investigating MLOps practices
        for continuous deployment of machine learning models. The pipeline predicts
        customer satisfaction scores for Olist e-commerce orders using order features
        such as price, payment method, freight value, and product dimensions.
        """
    )

    try:
        high_level_image = Image.open("_assets/high_level_overview.png")
        st.image(high_level_image, caption="High Level Pipeline Overview")

        whole_pipeline_image = Image.open(
            "_assets/training_and_deployment_pipeline_updated.png"
        )
        st.image(whole_pipeline_image, caption="Training and Deployment Pipeline")
    except FileNotFoundError:
        st.warning(
            "Pipeline diagram images not found in '_assets' folder, "
            "but the app will still work."
        )

    st.markdown(
        """
        #### Problem Statement
        The objective is to predict the customer satisfaction score (1–5) for a given
        order based on observable features at the time of purchase. The pipeline is
        built with [ZenML](https://zenml.io/) for orchestration and
        [MLflow](https://mlflow.org/) for experiment tracking and model deployment.
        """
    )

    st.markdown(
        """
        #### Pipeline Overview
        The system ingests raw order data, applies feature engineering, trains a
        regression model with Optuna hyperparameter tuning, evaluates it against
        held-out test data, and conditionally deploys it if it meets the minimum
        accuracy threshold. The deployed model is served here for live predictions.
        """
    )

    st.markdown("#### Input Features")
    st.markdown(
        "Adjust the sliders and input fields below to describe an order, "
        "then click **Predict** to get the estimated satisfaction score."
    )

    # --- Sidebar inputs ---
    payment_sequential = st.sidebar.slider("Payment Sequential", 1, 10, 1)
    payment_installments = st.sidebar.slider("Payment Installments", 1, 24, 1)

    # --- Main inputs ---
    col1, col2 = st.columns(2)

    with col1:
        payment_value = st.number_input(
            "Payment Value (R$)", min_value=0.0, value=100.0
        )
        price = st.number_input("Product Price (R$)", min_value=0.0, value=50.0)
        freight_value = st.number_input(
            "Freight Value (R$)", min_value=0.0, value=15.0
        )
        product_name_length = st.number_input(
            "Product Name Length (chars)", min_value=0, value=50
        )
        product_description_length = st.number_input(
            "Product Description Length (chars)", min_value=0, value=200
        )
        product_photos_qty = st.number_input(
            "Number of Product Photos", min_value=0, value=1
        )

    with col2:
        product_weight_g = st.number_input(
            "Product Weight (g)", min_value=0.0, value=500.0
        )
        product_length_cm = st.number_input(
            "Product Length (cm)", min_value=0.0, value=20.0
        )
        product_height_cm = st.number_input(
            "Product Height (cm)", min_value=0.0, value=10.0
        )
        product_width_cm = st.number_input(
            "Product Width (cm)", min_value=0.0, value=15.0
        )

    # --- Predict button ---
    if st.button("Predict"):
        try:
            client = Client()
            run = client.get_pipeline(
                "continuous_deployment_pipeline"
            ).last_successful_run
            model = run.steps["train_model"].output.load()

            df = pd.DataFrame(
                {
                    "payment_sequential": [payment_sequential],
                    "payment_installments": [payment_installments],
                    "payment_value": [payment_value],
                    "price": [price],
                    "freight_value": [freight_value],
                    "product_name_lenght": [product_name_length],
                    "product_description_lenght": [product_description_length],
                    "product_photos_qty": [product_photos_qty],
                    "product_weight_g": [product_weight_g],
                    "product_length_cm": [product_length_cm],
                    "product_height_cm": [product_height_cm],
                    "product_width_cm": [product_width_cm],
                }
            )

            pred = model.predict(df)
            score = float(pred[0]) if isinstance(pred, np.ndarray) else float(pred)
            score_clipped = max(1.0, min(5.0, score))

            st.success(
                f"Predicted Customer Satisfaction Score: **{score_clipped:.2f} / 5.00**"
            )
            st.progress(score_clipped / 5.0)
            st.balloons()

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info(
                "Make sure the deployment pipeline has been run at least once "
                "before using the prediction feature."
            )

    # --- Results button ---
    if st.button("Results"):
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow_client = mlflow.tracking.MlflowClient()

            experiments = mlflow_client.search_experiments()
            all_ids = [
                e.experiment_id for e in experiments
                if e.name in ("train_pipeline", "continuous_deployment_pipeline")
            ]

            if all_ids:
                runs = mlflow.search_runs(
                    experiment_ids=all_ids,
                    order_by=["start_time DESC"],
                )

                if not runs.empty:
                    st.write(
                        "The table below shows results from all pipeline runs, "
                        "pulled directly from MLflow. Each row corresponds to "
                        "one training run."
                    )

                    available_cols = {
                        "tags.mlflow.runName": "Run",
                        "tags.model_name": "Model",
                        "tags.fine_tuning": "Tuned",
                        "metrics.mse": "MSE",
                        "metrics.rmse": "RMSE",
                        "metrics.r2_score": "R² Score",
                    }

                    cols = [c for c in available_cols.keys() if c in runs.columns]
                    results = runs[cols].copy()
                    results.columns = [available_cols[c] for c in cols]
                    results = results.dropna(subset=["MSE", "RMSE", "R² Score"])
                    results[["MSE", "RMSE", "R² Score"]] = results[
                        ["MSE", "RMSE", "R² Score"]
                    ].round(4)
                    st.dataframe(results, use_container_width=True)

                else:
                    st.warning(
                        "No runs found yet. Run the training pipeline first."
                    )
            else:
                st.warning(
                    "No MLflow experiments found. "
                    "Run the training or deployment pipeline first."
                )

        except Exception as e:
            st.error(f"Could not load results from MLflow: {e}")

        try:
            image = Image.open("_assets/feature_importance_gain.png")
            st.image(image, caption="Feature Importance Gain")
        except FileNotFoundError:
            st.warning("Feature importance image not found.")


if __name__ == "__main__":
    main()