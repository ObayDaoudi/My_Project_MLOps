import logging
import mlflow
import pandas as pd
from sklearn.base import RegressorMixin
from zenml import step
from zenml.client import Client
from .config import ModelNameConfig
from model.model_dev import (
    HyperparameterTuner,
    LightGBMModel,
    LinearRegressionModel,
    RandomForestModel,
    XGBoostModel,
)
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name, enable_cache=False)
def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> RegressorMixin:
    """
    Trains the model on the ingested data.
    """
    config = ModelNameConfig()

    try:
        model = None
        if config.model_name == "lightgbm":
            if config.fine_tuning:
                tuner = HyperparameterTuner(LightGBMModel(), x_train, y_train, x_test, y_test)
                best_params = tuner.optimize()
                model = lgb.LGBMRegressor(**best_params)
                model.fit(x_train, y_train)
            else:
                model = LightGBMModel().train(x_train, y_train)

        elif config.model_name == "randomforest":
            if config.fine_tuning:
                tuner = HyperparameterTuner(RandomForestModel(), x_train, y_train, x_test, y_test)
                best_params = tuner.optimize()
                model = RandomForestRegressor(**best_params)
                model.fit(x_train, y_train)
            else:
                model = RandomForestModel().train(x_train, y_train)

        elif config.model_name == "xgboost":
            if config.fine_tuning:
                tuner = HyperparameterTuner(XGBoostModel(), x_train, y_train, x_test, y_test)
                best_params = tuner.optimize()
                model = xgb.XGBRegressor(**best_params)
                model.fit(x_train, y_train)
            else:
                model = XGBoostModel().train(x_train, y_train)

        elif config.model_name == "linear_regression":
            model = LinearRegressionModel().train(x_train, y_train)
        else:
            raise ValueError("Model {} not supported".format(config.model_name))

        # Log the model to MLflow so the deployer can find it
        mlflow.sklearn.log_model(model, "model")

        # Tag the run with the model name so it appears in the Results table
        mlflow.set_tag("model_name", config.model_name)
        mlflow.set_tag("fine_tuning", str(config.fine_tuning))

        return model
    except Exception as e:
        logging.error("Error in training model: {}".format(e))
        raise e

