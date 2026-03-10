import argparse
from pipelines.deployment_pipeline import (
    continuous_deployment_pipeline,
    inference_pipeline,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        choices=["deploy", "predict", "deploy_and_predict"],
        default="deploy_and_predict",
        help="Choose what you want to run.",
    )
    args = parser.parse_args()

    if args.config == "deploy" or args.config == "deploy_and_predict":
        print("🚀 Running deployment pipeline...")
        # Run the deployment pipeline (no arguments needed now!)
        continuous_deployment_pipeline()
    
    if args.config == "predict" or args.config == "deploy_and_predict":
        print("🔮 Running inference pipeline...")
        # Run the inference pipeline
        inference_pipeline()