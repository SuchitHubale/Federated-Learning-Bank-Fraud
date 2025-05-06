import os
import sys
import torch
import requests
import json
import time
import pickle
from .train_local import prepare_data, train_model
from .model import FraudDetectionModel  # Import from local file instead of server

class FederatedClient:
    def __init__(self, client_id, server_url, data_path='dataset/bank2_data.csv', model_save_path='models/local_model2.pth', epochs=10):
        self.client_id = client_id
        self.server_url = server_url
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.epochs = epochs
        
        # Ensure model directory exists
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
    def train(self):
        """Train the local model using local data"""
        print(f"Client {self.client_id}: Starting local training...")
        
        # Prepare data
        X_train, y_train, X_test, y_test, input_dim = prepare_data(self.data_path)
        
        # Initialize model
        model = FraudDetectionModel(input_dim=input_dim)
        
        # Check if there's a global model to start from
        global_model_path = 'models/global_model.pth'
        if os.path.exists(global_model_path):
            print(f"Client {self.client_id}: Loading global model weights...")
            model.load_state_dict(torch.load(global_model_path))
        
        # Train model
        trained_model = train_model(
            model, X_train, y_train, X_test, y_test, epochs=self.epochs, client_id=self.client_id, server_url=self.server_url
        )
        
        # Save local model
        torch.save(trained_model.state_dict(), self.model_save_path)
        print(f"Client {self.client_id}: Model saved at {self.model_save_path}")
        
        return trained_model
    
    def send_weights(self, model):
        """Send model weights to the server"""
        print(f"Client {self.client_id}: Sending weights to server...")
        
        # Get model weights
        weights = model.get_weights()
        
        # Convert weights to serializable format (list of numpy arrays)
        serialized_weights = [w.cpu().numpy().tolist() for w in weights]
        
        # Prepare payload
        payload = {
            'client_id': self.client_id,
            'weights': serialized_weights
        }
        
        try:
            # Send weights to server
            response = requests.post(
                f"{self.server_url}/update",
                json=payload
            )
            
            if response.status_code == 200:
                print(f"Client {self.client_id}: Weights sent successfully")
                return True
            else:
                print(f"Client {self.client_id}: Failed to send weights, status: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Client {self.client_id}: Error sending weights - {str(e)}")
            return False
    
    def get_global_model(self):
        """Retrieve global model from server"""
        print(f"Client {self.client_id}: Requesting global model...")
        max_retries = 10
        for attempt in range(max_retries):
            try:
                response = requests.get(f"{self.server_url}/model")
                if response.status_code == 200:
                    # Save global model weights
                    global_model_path = 'models/global_model.pth'
                    with open(global_model_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Client {self.client_id}: Global model retrieved and saved")
                    return True
                else:
                    print(f"Client {self.client_id}: Global model not ready (status: {response.status_code}), retrying...")
                    time.sleep(3)
            except Exception as e:
                print(f"Client {self.client_id}: Error getting global model - {str(e)}. Retrying...")
                time.sleep(3)
        print(f"Client {self.client_id}: Failed to get global model after {max_retries} attempts.")
        return False
    
    def run_federated_round(self):
        """Run one round of federated learning"""
        # Train local model
        model = self.train()
        
        # Send weights to server
        self.send_weights(model)
        
        # Wait for server to aggregate
        time.sleep(2)
        
        # Get global model
        self.get_global_model() 