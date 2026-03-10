import os
from pydantic import BaseModel

class ModelNameConfig(BaseModel):
    """Model Configurations"""
    # Listen to the terminal, but default to lightgbm if nothing is typed
    model_name: str = os.getenv("ZENML_MODEL_NAME", "lightgbm")
    
    # Listen to the terminal, but default to True for tuning
    fine_tuning: bool = os.getenv("ZENML_FINE_TUNING", "True") == "True"