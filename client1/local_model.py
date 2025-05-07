import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import os
import json
from .utils import save_metrics

class LocalModel:
    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        """Initialize the local model with RandomForestClassifier."""
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        self.model_path = './models/local_model1.pkl'
        
    def train(self, X_train, y_train, X_test, y_test, n_epochs=5):
        """Train the local model and save metrics."""
        print("Training local model...")
        
        # Training metrics history
        metrics_history = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'loss': []
        }
        
        # Simulate multiple epochs for federated learning
        for epoch in range(n_epochs):
            print(f"Epoch {epoch+1}/{n_epochs}")
            
            # Train the model (for RandomForest, we're not doing incremental training,
            # so this is the same in each epoch, but included for demonstration)
            self.model.fit(X_train, y_train)
            
            # Evaluate on test set
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Simulate loss (negative log likelihood)
            y_pred_proba = np.clip(self.model.predict_proba(X_test)[:, 1], 1e-10, 1-1e-10)
            loss = -np.mean(y_test * np.log(y_pred_proba) + (1 - y_test) * np.log(1 - y_pred_proba))
            
            # Save metrics for this epoch
            metrics_history['accuracy'].append(float(accuracy))
            metrics_history['precision'].append(float(precision))
            metrics_history['recall'].append(float(recall))
            metrics_history['f1'].append(float(f1))
            metrics_history['loss'].append(float(loss))
            
            print(f"Epoch {epoch+1} - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
        
        # Final evaluation
        y_pred = self.model.predict(X_test)
        
        # Generate classification report
        report = classification_report(y_test, y_pred, target_names=['NonFraud', 'Fraud'], output_dict=True)
        
        # Save final metrics
        final_metrics = {
            'history': metrics_history,
            'final_accuracy': float(accuracy_score(y_test, y_pred)),
            'final_precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'final_recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'final_f1': float(f1_score(y_test, y_pred, zero_division=0)),
            'classification_report': report
        }
        
        save_metrics(final_metrics)
        
        # Save the model
        self.save_model()
        return final_metrics
    
    def save_model(self):
        """Save the trained model to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load a previously trained model from disk."""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(f"Model loaded from {self.model_path}")
            return True
        else:
            print(f"No model found at {self.model_path}")
            return False
    
    def predict(self, X):
        """Make predictions using the trained model."""
        return self.model.predict(X)
    
    def get_model_weights(self):
        """Get model weights (for RandomForest, return the feature importances)."""
        return self.model.feature_importances_
    
    def set_model_weights(self, weights):
        """Set model weights (for demonstration purposes only, not applicable for RandomForest)."""
        print("Setting model weights not applicable for RandomForestClassifier.")
        return False 