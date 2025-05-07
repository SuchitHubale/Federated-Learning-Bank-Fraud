import json
import os
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'dashboard', 'logs')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

def load_model(model_path):
    """Load a model from disk."""
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return model
    else:
        print(f"No model found at {model_path}")
        return None

def save_model(model, filename):
    """Save a model to disk."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def save_metrics(metrics, file_path=None):
    """Save metrics to a JSON file."""
    if file_path is None:
        file_path = os.path.join(LOGS_DIR, 'global_metrics.json')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {file_path}")

def save_aggregation_log(log_entry, file_path=None):
    """Save aggregation log to a JSON file."""
    if file_path is None:
        file_path = os.path.join(LOGS_DIR, 'aggregation_log.json')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Load existing log if available
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            log = json.load(f)
    else:
        log = []
    
    # Add new entry
    log.append(log_entry)
    
    # Save updated log
    with open(file_path, 'w') as f:
        json.dump(log, f, indent=4)
    print(f"Aggregation log saved to {file_path}")

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics."""
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=['NonFraud', 'Fraud'], output_dict=True)
    
    # Simulate loss (negative log likelihood)
    y_pred_proba = np.clip(model.predict_proba(X_test)[:, 1], 1e-10, 1-1e-10)
    loss = -np.mean(y_test * np.log(y_pred_proba) + (1 - y_test) * np.log(1 - y_pred_proba))
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'loss': float(loss),
        'classification_report': report
    }
    
    return metrics 