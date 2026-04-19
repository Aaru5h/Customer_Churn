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
    
    os.makedirs("backend/models", exist_ok=True)
    
    # Save both pipelines
    lr_path = "backend/models/pipeline_lr.joblib"
    dt_path = "backend/models/pipeline_dt.joblib"
    metrics_path = "backend/models/metrics.joblib"
    
    joblib.dump(results["Logistic Regression"]["pipeline"], lr_path)
    print(f"Saved Logistic Regression pipeline to {lr_path}")
    
    joblib.dump(results["Decision Tree"]["pipeline"], dt_path)
    print(f"Saved Decision Tree pipeline to {dt_path}")
    
    # Save metrics for both models (strip non-serializable objects like numpy arrays)
    saved_metrics = {}
    for model_name in ["Logistic Regression", "Decision Tree"]:
        m = results[model_name]["metrics"]
        saved_metrics[model_name] = {
            "accuracy": float(m["accuracy"]),
            "precision": float(m["precision"]),
            "recall": float(m["recall"]),
            "f1": float(m["f1"]),
        }
    
    joblib.dump(saved_metrics, metrics_path)
    print(f"Saved metrics to {metrics_path}")
    
    # Also keep the legacy single pipeline for backwards compatibility
    joblib.dump(results["Logistic Regression"]["pipeline"], "backend/models/pipeline.joblib")
    print("Done! All models and metrics saved.")

if __name__ == "__main__":
    pre_train()
