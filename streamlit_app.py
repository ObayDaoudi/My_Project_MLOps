import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from zenml.client import Client

def main():
    st.title("End to End Customer Satisfaction Pipeline with ZenML")

    try:
        high_level_image = Image.open("_assets/high_level_overview.png")
        st.image(high_level_image, caption="High Level Pipeline")

        whole_pipeline_image = Image.open("_assets/training_and_deployment_pipeline_updated.png")
        st.image(whole_pipeline_image, caption="Whole Pipeline")
    except FileNotFoundError:
        st.warning("Diagram images not found in the '_assets' folder, but the app will still work")

    st.markdown(
        """ 
    #### Problem Statement 
     The objective here is to predict the customer satisfaction score for a given order based on features like order status, price, payment, etc. I will be using [ZenML](https://zenml.io/) to build a production-ready pipeline to predict the customer satisfaction score for the next order or purchase.    """
    )
    
    st.markdown(
        """ 
    Above is a figure of the whole pipeline, we first ingest the data, clean it, train the model, and evaluate the model, and if data source changes or any hyperparameter values changes, deployment will be triggered, and (re) trains the model and if the model meets minimum accuracy requirement, the model will be deployed.
    """
    )

    st.markdown(
        """ 
    #### Description of Features 
    This app is designed to predict the customer satisfaction score for a given customer. You can input the features of the product listed below and get the customer satisfaction score. 
    """
    )
    
    payment_sequential = st.sidebar.slider("Payment Sequential", 1, 10, 1)
    payment_installments = st.sidebar.slider("Payment Installments", 1, 24, 1)
    payment_value = st.number_input("Payment Value", min_value=0.0, value=100.0)
    price = st.number_input("Price", min_value=0.0, value=50.0)
    freight_value = st.number_input("Freight Value", min_value=0.0, value=15.0)
    product_name_length = st.number_input("Product name length", min_value=0, value=50)
    product_description_length = st.number_input("Product Description length", min_value=0, value=200)
    product_photos_qty = st.number_input("Product photos Quantity", min_value=0, value=1)
    product_weight_g = st.number_input("Product weight measured in grams", min_value=0.0, value=500.0)
    product_length_cm = st.number_input("Product length (CMs)", min_value=0.0, value=20.0)
    product_height_cm = st.number_input("Product height (CMs)", min_value=0.0, value=10.0)
    product_width_cm = st.number_input("Product width (CMs)", min_value=0.0, value=15.0)

    if st.button("Predict"):
        try:
            # 1. Connect to ZenML directly
            client = Client()
            
            # 2. Fetch the latest successful run
            run = client.get_pipeline("continuous_deployment_pipeline").last_successful_run
            
            # 3. Load the actual model artifact into memory
            model = run.steps["train_model"].output.load()

            # 4. Format the data from the user interface
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
            
            # 5. Make the prediction
            pred = model.predict(df)
            
            # Formats the prediction nicely
            score = pred[0] if isinstance(pred, np.ndarray) else pred
            
            st.success(
                f"Your Customer Satisfaction rate (range between 0 - 5) with given product details is: {score:.2f}"
            )
            st.balloons()

        except Exception as e:
            st.error(f"An error occurred while making the prediction: {e}")

    if st.button("Results"):
        st.write(
            "We have experimented with two ensemble and tree based models and compared the performance of each model. The results are as follows:"
        )

        df = pd.DataFrame(
            {
                "Models": ["Linear Regression", "LightGBM", "XGBoost"],
                "MSE": [1.864, 1.665, 1.665],
                "RMSE": [1.365, 1.290, 1.290],
                "R2 Score": [0.017, 0.124, 0.122]
            }
        )
        st.dataframe(df)

        st.write(
            "Following figure shows how important each feature is in the model that contributes to the target variable or contributes in predicting customer satisfaction rate."
        )
        try:
            image = Image.open("_assets/feature_importance_gain.png")
            st.image(image, caption="Feature Importance Gain")
        except FileNotFoundError:
            st.warning("Feature importance image not found.")

if __name__ == "__main__":
    main()