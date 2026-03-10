import argparse
import os
from pipelines.training_pipeline import train_pipeline

if __name__ == "__main__":
    # Create the terminal menu
    parser = argparse.ArgumentParser(description="Run the ZenML training pipeline.")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["linear_regression", "lightgbm", "xgboost", "randomforest"], 
        default="lightgbm",
        help="Choose the model you want to train."
    )
    parser.add_argument(
        "--no-tune", 
        action="store_true",
        help="Type this to turn OFF Optuna fine-tuning."
    )
    
    args = parser.parse_args()
    
    # Send the terminal choices to the config file
    os.environ["ZENML_MODEL_NAME"] = args.model
    os.environ["ZENML_FINE_TUNING"] = "False" if args.no_tune else "True"
    
    print(f"\n🚀 Starting Pipeline...")
    print(f"🧠 Model: {args.model}")
    print(f"⚙️  Fine Tuning (Optuna): {not args.no_tune}\n")
    
    # Run the pipeline
    train_pipeline()