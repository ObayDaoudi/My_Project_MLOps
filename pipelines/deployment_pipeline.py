import json
import numpy as np
import pandas as pd
from steps.clean_data import clean_data
from steps.evaluation import evaluation
from steps.ingest_data import ingest_data
from steps.model_train import train_model
from zenml import pipeline, step
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from pydantic import BaseModel
from .utils import get_data_for_test

class DeploymentTriggerConfig(BaseModel):
    min_accuracy: float = 0.0 

@step(enable_cache=False)
def deployment_trigger(accuracy: float) -> bool:
    config = DeploymentTriggerConfig()
    return accuracy > config.min_accuracy

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )
    if not existing_services:
        raise RuntimeError("No MLflow prediction service deployed.")
    return existing_services[0]

@step(enable_cache=False)
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:
    service.start(timeout=10)
    data_dict = json.loads(data)
    data_dict.pop("columns", None)
    data_dict.pop("index", None)
    columns_for_df = [
        "payment_sequential", "payment_installments", "payment_value", "price",
        "freight_value", "product_name_lenght", "product_description_lenght",
        "product_photos_qty", "product_weight_g", "product_length_cm",
        "product_height_cm", "product_width_cm",
    ]
    df = pd.DataFrame(data_dict["data"], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    array_data = np.array(json_list)
    return service.predict(array_data)

@pipeline(enable_cache=False)
def continuous_deployment_pipeline():
    # 1. Ingest, Clean, Train (using fast Linear Regression)
    df = ingest_data()
    x_train, x_test, y_train, y_test = clean_data(df)
    
    # 2. Train the model and pass it EXACTLY how the deployer wants it
    model = train_model(x_train, x_test, y_train, y_test)
    
    r2_score, rmse = evaluation(model, x_test, y_test)
    deployment_decision = deployment_trigger(accuracy=r2_score)
    
    # 3. Deploy the exact model we just trained
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=3,
        timeout=60,
    )

@pipeline(enable_cache=False)
def inference_pipeline():
    batch_data = get_data_for_test()
    model_deployment_service = prediction_service_loader(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        running=False,
    )
    predictor(service=model_deployment_service, data=batch_data)