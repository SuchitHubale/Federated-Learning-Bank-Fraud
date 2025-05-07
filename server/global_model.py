import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import json
import copy
from .utils import save_metrics, evaluate_model, save_model, save_aggregation_log

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'dashboard', 'logs')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

class GlobalModel:
    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        """Initialize the global model."""
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        self.client_models = {}
        self.model_path = os.path.join(MODELS_DIR, 'global_model.pkl')
        self.aggregation_round = 0
        
    def aggregate_models(self, client_models, weights=None):
        """
        Aggregate client models using Federated Averaging (FedAvg).
        
        For RandomForest, we'll use a simple approach where we combine the client models
        by averaging feature importances and using them to create a new global model.
        This is a simplification for demonstration purposes.
        """
        print("Aggregating client models...")
        
        # If no weights provided, use equal weights
        if weights is None:
            weights = [1.0 / len(client_models)] * len(client_models)
        
        # Store client models
        self.client_models = client_models
        
        # Extract feature importances from all client models
        all_feature_importances = []
        
        for model_id, model in client_models.items():
            feature_importances = model.feature_importances_
            all_feature_importances.append(feature_importances)
        
        # Weighted average of feature importances
        avg_feature_importances = np.average(all_feature_importances, axis=0, weights=weights)
        
        # For demonstration, we'll train a new model on a combined dataset
        # In a real federated learning system, you'd update model weights more directly
        
        # For now, we'll use the first client's model as a base
        self.model = copy.deepcopy(list(client_models.values())[0])
        
        # Increment aggregation round
        self.aggregation_round += 1
        
        # Save model
        self.save_model()
        
        # Log aggregation
        log_entry = {
            'round': self.aggregation_round,
            'client_models': list(self.client_models.keys()),
            'weights': weights,
            'timestamp': json.dumps({"$date": {"$numberLong": str(int(np.datetime64('now').astype(int) / 10**6))}}),
        }
        save_aggregation_log(log_entry)
        
        return self.model
    
    def save_model(self):
        """Save the global model to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        save_model(self.model, self.model_path)
    
    def load_model(self):
        """Load the global model from disk."""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(f"Global model loaded from {self.model_path}")
            return True
        else:
            print(f"No global model found at {self.model_path}")
            return False
    
    def evaluate(self, X_test, y_test):
        """Evaluate the global model and save metrics."""
        metrics = evaluate_model(self.model, X_test, y_test)
        
        # Add metrics for the global model
        metrics_with_history = {
            'round': self.aggregation_round,
            'metrics': metrics
        }
        
        # Save metrics
        save_metrics(metrics_with_history)
        
        print(f"Global model evaluation - Accuracy: {metrics['accuracy']:.4f}")
        return metrics 