import joblib
import os
from model import train_models

def pre_train():
    DATA_PATH = "data/BankChurners.csv"
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    print("Training models locally...")
    results = train_models(DATA_PATH)
    
    # We only need the pipeline for prediction
    # Storing Logistic Regression as the primary model
    model_path = "backend/models/pipeline.joblib"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    joblib.dump(results["Logistic Regression"]["pipeline"], model_path)
    print(f"Successfully saved model to {model_path}")

if __name__ == "__main__":
    pre_train()
