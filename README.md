# Customer Satisfaction Prediction Pipeline: An MLOps Study in Drift-Aware Continuous Deployment

## Problem Statement

E-commerce platforms process millions of orders across diverse product categories,
sellers, and logistics networks. Predicting whether a customer will be satisfied with
their experience — before they even leave a review has direct business value: it
enables proactive intervention, seller quality control, and smarter logistics decisions.

This project investigates how production-grade MLOps practices can be applied to
customer satisfaction scoring using the Brazilian Olist e-commerce dataset. The system
is designed to remain reliable and monitorable as real-world data distributions shift
over time — for example, as new product categories emerge, seasonal demand changes, or
logistics performance degrades.

The central question this project addresses is: how should a machine learning deployment
pipeline respond to data drift, and what is the right trigger for model retraining?

## What This Project Does

This pipeline trains, evaluates, and continuously deploys a customer satisfaction
scoring model using ZenML for pipeline orchestration and MLflow for experiment tracking.
Unlike standard ML pipelines that treat deployment as a one-time event, this system
monitors incoming data for statistical drift and incorporates that signal into the
redeployment decision.

The pipeline covers:
- Automated ingestion and versioning of Olist order and review data
- Feature engineering with full lineage tracking
- Multi-model training with Optuna hyperparameter optimisation
- Stacking ensemble with conformal prediction intervals
- Data drift detection via Evidently AI, integrated into the deployment trigger
- Continuous deployment via MLflow model serving

## Architecture

The system is composed of two pipelines:

**Training + Deployment Pipeline**
Raw data → Feature engineering → Model training (LightGBM / XGBoost / RandomForest) →
Stacking ensemble → Evaluation (RMSE, R², MAE) → Drift check → Conditional deployment

**Inference Pipeline**
Batch input → Feature retrieval → Prediction with confidence intervals → Drift logging

## Setup
```bash
git clone <your-repo>
cd <your-repo>
pip install -r requirements.txt
zenml integration install mlflow evidently -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set
```

## Running the Pipeline

Training with hyperparameter tuning:
```bash
python run_pipeline.py --model ensemble
```

Deployment with drift-aware trigger:
```bash
python run_deployment.py --config deploy_and_predict
```

## Research Findings

[To be completed as experiments run]

Preliminary results suggest that drift-based retraining triggers maintain higher
predictive accuracy over a simulated temporal window where order volume, product
mix, and logistics performance shift compared to accuracy-only deployment triggers.

## Technical Stack

- Orchestration: ZenML
- Experiment tracking: MLflow
- Drift detection: Evidently AI
- Models: LightGBM, XGBoost, RandomForest, stacking meta-learner
- Uncertainty quantification: MAPIE (conformal prediction)
- Data: Brazilian E-Commerce Public Dataset by Olist (Kaggle)