import argparse
import sys
import os

# Ensure the parent directory is in sys.path so 'client' can be imported from any subdirectory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from client.client import FederatedClient

def run_client(client_id, server_url, data_path, model_save_path, epochs):
    """Run a federated learning client"""
    client = FederatedClient(
        client_id=client_id,
        server_url=server_url,
        data_path=data_path,
        model_save_path=model_save_path,
        epochs=epochs
    )
    
    # Run federated learning round
    client.run_federated_round()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated Learning Client')
    parser.add_argument('--client_id', type=str, default='client2', 
                        help='Client ID')
    parser.add_argument('--server_url', type=str, default='http://192.168.1.42:5000', 
                        help='192.168.1.42')
    parser.add_argument('--data_path', type=str, default='dataset/bank2_data.csv', 
                        help='Path to client data')
    parser.add_argument('--model_path', type=str, default='models/local_model2.pth', 
                        help='Path to save local model')
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Training epochs per round')
    
    args = parser.parse_args()
    
    # Run client
    run_client(
        client_id=args.client_id,
        server_url=args.server_url,
        data_path=args.data_path,
        model_save_path=args.model_path,
        epochs=args.epochs
    )