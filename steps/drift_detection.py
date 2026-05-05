import logging
import json
from typing import Tuple

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from typing_extensions import Annotated
from zenml import step
import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name, enable_cache=False)
def detect_data_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    drift_threshold: float = 0.5,
) -> Tuple[
    Annotated[bool, "drift_detected"],
    Annotated[float, "drift_share"],
]:
    try:
        # Drop target column if present
        ref = reference_data.drop(columns=["review_score"], errors="ignore")
        cur = current_data.drop(columns=["review_score"], errors="ignore")

        # Keep only shared numeric columns
        shared_cols = list(set(ref.columns) & set(cur.columns))
        ref = ref[shared_cols].select_dtypes(include="number")
        cur = cur[shared_cols].select_dtypes(include="number")

        # Run Evidently drift report (v0.7.x API)
        report = Report([DataDriftPreset()])
        result = report.run(reference_data=ref, current_data=cur)
        result_dict = result.dict()

        # Extract drift share from first metric (DriftedColumnsCount)
        drift_share: float = float(result_dict["metrics"][0]["value"]["share"])
        drift_detected: bool = drift_share > drift_threshold

        # Log to MLflow
        mlflow.log_metric("drift_share", drift_share)
        mlflow.log_metric("drift_threshold", drift_threshold)
        mlflow.log_metric("drift_detected", int(drift_detected))

        if drift_detected:
            logging.warning(
                f"Data drift detected! {drift_share:.1%} of features drifted "
                f"(threshold: {drift_threshold:.1%}). Retraining will be triggered."
            )
        else:
            logging.info(
                f"No significant data drift detected. "
                f"Drift share: {drift_share:.1%} (threshold: {drift_threshold:.1%})."
            )

        return drift_detected, drift_share

    except Exception as e:
        logging.error(f"Error in drift detection step: {e}")
        raise e
