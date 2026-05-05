import json
import numpy as np
import pandas as pd
from steps.clean_data import clean_data
from steps.evaluation import evaluation
from steps.ingest_data import ingest_data
from steps.model_train import train_model
from steps.drift_detection import detect_data_drift
from zenml import pipeline, step
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from pydantic import BaseModel
from .utils import get_data_for_test


class DeploymentTriggerConfig(BaseModel):
    """Configuration for the drift-aware deployment trigger."""
    # Minimum R² score the model must achieve to be considered for deployment
    min_accuracy: float = 0.1
    # If drift is detected AND the model meets accuracy, always redeploy
    # If no drift and model meets accuracy, also deploy (standard behaviour)
    # Deployment is only blocked when accuracy threshold is not met
    redeploy_on_drift: bool = True


@step(enable_cache=False)
def deployment_trigger(
    accuracy: float,
    drift_detected: bool,
) -> bool:
    """
    Drift-aware deployment trigger.

    Deployment logic:
    - If model does NOT meet the minimum accuracy threshold → never deploy.
    - If model meets accuracy AND drift was detected → deploy (retraining was
      triggered by drift, so we always want to push the fresh model).
    - If model meets accuracy AND no drift → deploy (standard performance-gate).

    In practice this means: accuracy is the hard gate, drift accelerates
    the decision to push the retrained model to production.

    Args:
        accuracy:      R² score of the newly trained model on the test set.
        drift_detected: Whether data drift was detected in the incoming data.

    Returns:
        bool: True if the model should be deployed.
    """
    config = DeploymentTriggerConfig()

    meets_accuracy = accuracy > config.min_accuracy

    if not meets_accuracy:
        print(
            f"[Deployment Trigger] Model R²={accuracy:.4f} is below the minimum "
            f"threshold of {config.min_accuracy}. Deployment BLOCKED."
        )
        return False

    if drift_detected:
        print(
            f"[Deployment Trigger] Data drift detected. Model R²={accuracy:.4f} meets "
            f"threshold. Deploying retrained model to replace drifted production model."
        )
    else:
        print(
            f"[Deployment Trigger] No drift detected. Model R²={accuracy:.4f} meets "
            f"threshold. Deploying as standard performance-gated update."
        )

    return True


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
    """
    Drift-aware continuous deployment pipeline.

    Flow:
    1.  Ingest full dataset (represents latest available data).
    2.  Clean and split into train / test sets.
    3.  Run Evidently drift detection: compare a recent sample of the data
        (current_data = test split) against the training baseline (reference_data
        = train split). In a production setting, current_data would come from a
        live feature store or a windowed batch of recent predictions.
    4.  Train the model with Optuna hyperparameter optimisation.
    5.  Evaluate on held-out test set.
    6.  Drift-aware deployment trigger: considers BOTH R² score AND drift signal.
    7.  Conditionally deploy via MLflow model server.
    """
    # Step 1 & 2: Ingest and clean
    df = ingest_data()
    x_train, x_test, y_train, y_test = clean_data(df)

    # Step 3: Drift detection
    # reference_data = training distribution baseline
    # current_data   = test split (simulates incoming production data)
    # In a live system, swap current_data for a real-time batch from a feature store.
    drift_detected, drift_share = detect_data_drift(
        reference_data=x_train,
        current_data=x_test,
        drift_threshold=0.5,
    )

    # Step 4 & 5: Train and evaluate
    model = train_model(x_train, x_test, y_train, y_test)
    r2_score, rmse = evaluation(model, x_test, y_test)

    # Step 6: Drift-aware deployment decision
    deploy = deployment_trigger(accuracy=r2_score, drift_detected=drift_detected)

    # Step 7: Conditional deployment
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deploy,
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
