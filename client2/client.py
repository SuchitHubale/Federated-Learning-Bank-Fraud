import os
import numpy as np
from sklearn.model_selection import train_test_split
from .utils import load_data, preprocess_data, send_model_to_server, receive_global_model
from .local_model import LocalModel

class Client:
    def __init__(self, data_path="./datasets/bank2.csv"):
        """Initialize the client with the dataset path."""
        self.data_path = data_path
        self.model = LocalModel()
        self.preprocessor = None
    
    def load_and_preprocess_data(self):
        """Load and preprocess the client's data."""
        # Load data
        print("[CLIENT2] Loading data...", flush=True)
        df = load_data(self.data_path)
        
        # Preprocess data
        print("[CLIENT2] Preprocessing data...", flush=True)
        X_processed, y, self.preprocessor = preprocess_data(df)
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"[CLIENT2] Data loaded and preprocessed. Training set size: {X_train.shape}, Test set size: {X_test.shape}", flush=True)
        return X_train, X_test, y_train, y_test
    
    def train_model(self, n_epochs=5):
        """Train the local model on client data."""
        # Load and preprocess data
        X_train, X_test, y_train, y_test = self.load_and_preprocess_data()
        
        # Train the model
        print("[CLIENT2] Training local model...", flush=True)
        metrics = self.model.train(X_train, y_train, X_test, y_test, n_epochs=n_epochs)
        
        print(f"[CLIENT2] Local model training completed. Final accuracy: {metrics['final_accuracy']:.4f}", flush=True)
        return metrics
    
    def send_model(self, server_url='http://localhost:5000/upload_model'):
        """Send the trained model to the server."""
        model_path = self.model.model_path
        print("[CLIENT2] Sending model to server...", flush=True)
        return send_model_to_server(model_path, server_url)
    
    def update_model(self, server_url='http://localhost:5000/get_global_model'):
        """Update local model with the global model from the server."""
        global_model_path = './models/global_model.pkl'
        print("[CLIENT2] Receiving global model from server...", flush=True)
        success = receive_global_model(server_url, global_model_path)
        
        if success:
            # Load the global model
            print("[CLIENT2] Loading global model...", flush=True)
            self.model.load_model()
            return True
        return False
    
    def evaluate_model(self):
        """Evaluate the current model on test data."""
        # Load and preprocess data for evaluation
        _, X_test, _, y_test = self.load_and_preprocess_data()
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y_test)
        print(f"[CLIENT2] Model evaluation - Accuracy: {accuracy:.4f}", flush=True)
        return accuracy

def run_client(n_epochs=5):
    print("[CLIENT2] Client 2 started!", flush=True)
    client = Client()
    # Train local model
    metrics = client.train_model(n_epochs=n_epochs)
    # Send model to server
    client.send_model()
    # Wait for global model (in a real system, this would be event-driven)
    print("[CLIENT2] Waiting for global model...", flush=True)
    # For demonstration, we'll simulate this by directly updating now
    client.update_model()
    # Evaluate the updated model
    client.evaluate_model()
    
if __name__ == "__main__":
    run_client() 