import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, send_file, jsonify
from sklearn.model_selection import train_test_split
from .global_model import GlobalModel
from .utils import load_model, save_model, evaluate_model, save_aggregation_log

app = Flask(__name__)

# Initialize global model
global_model = GlobalModel()

# Store uploaded client models
client_models = {}

# Path to save uploaded models
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'dashboard', 'logs')
DATASETS_DIR = os.path.join(PROJECT_ROOT, 'datasets')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Global dataset for evaluation (combined from clients)
global_dataset = None

@app.route('/upload_model/<client_id>', methods=['POST'])
def upload_model(client_id):
    """Endpoint for clients to upload their trained models."""
    try:
        if 'model' not in request.files:
            return jsonify({'error': 'No model file provided'}), 400
        
        # Save the uploaded model
        model_file = request.files['model']
        client_model_path = os.path.join(MODELS_DIR, f'local_model_{client_id}.pkl')
        model_file.save(client_model_path)
        
        # Load the saved model
        client_model = load_model(client_model_path)
        if client_model is None:
            return jsonify({'error': 'Failed to load client model'}), 500
        
        # Store the client model
        client_models[client_id] = client_model
        print(f"Received model from {client_id}. Total models: {len(client_models)}")
        
        # If we have all expected client models, perform model aggregation
        if len(client_models) >= 2:  # Change this if you have more clients
            print("All client models received. Performing model aggregation...")
            perform_aggregation()
        
        return jsonify({'status': 'success', 'message': f'Model from {client_id} received'}), 200
    
    except Exception as e:
        print(f"Error processing model upload: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_global_model', methods=['GET'])
def get_global_model():
    """Endpoint for clients to download the global model."""
    try:
        global_model_path = os.path.join(MODELS_DIR, 'global_model.pkl')
        if not os.path.exists(global_model_path):
            return jsonify({'error': 'Global model not available yet'}), 404
        
        return send_file(global_model_path, as_attachment=True)
    
    except Exception as e:
        print(f"Error sending global model: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_metrics', methods=['GET'])
def get_metrics():
    """Endpoint to get the current metrics of all models."""
    try:
        # Load metrics files
        metrics_dir = LOGS_DIR
        metrics = {}
        
        # Global metrics
        global_metrics_path = os.path.join(metrics_dir, 'global_metrics.json')
        if os.path.exists(global_metrics_path):
            with open(global_metrics_path, 'r') as f:
                metrics['global'] = eval(f.read())
        
        # Client 1 metrics
        client1_metrics_path = os.path.join(metrics_dir, 'client1_metrics.json')
        if os.path.exists(client1_metrics_path):
            with open(client1_metrics_path, 'r') as f:
                metrics['client1'] = eval(f.read())
        
        # Client 2 metrics
        client2_metrics_path = os.path.join(metrics_dir, 'client2_metrics.json')
        if os.path.exists(client2_metrics_path):
            with open(client2_metrics_path, 'r') as f:
                metrics['client2'] = eval(f.read())
        
        return jsonify(metrics), 200
    
    except Exception as e:
        print(f"Error getting metrics: {e}")
        return jsonify({'error': str(e)}), 500

def perform_aggregation():
    """Aggregate client models and evaluate the global model."""
    try:
        # Aggregate models
        global_model.aggregate_models(client_models)
        
        # For demonstration, we'll evaluate the global model on a combined test set
        # In a real federated learning system, the server might not have access to any data
        load_evaluation_data()
        
        if global_dataset is not None:
            X, y = global_dataset

            # Load the preprocessor from client1 (or any reference client)
            preprocessor_path = os.path.join(MODELS_DIR, 'client1_preprocessor.pkl')
            if not os.path.exists(preprocessor_path):
                preprocessor_path = os.path.join(MODELS_DIR, 'client2_preprocessor.pkl')
            if os.path.exists(preprocessor_path):
                preprocessor = joblib.load(preprocessor_path)
                X_processed = preprocessor.transform(X)
            else:
                raise FileNotFoundError("No client preprocessor found for evaluation.")

            X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

            # Evaluate the global model
            global_model.evaluate(X_test, y_test)
            print("Model aggregation and evaluation completed")
        else:
            print("No evaluation data available")
        return True
    except Exception as e:
        print(f"Error during model aggregation: {e}")
        # Save error log
        log_entry = {
            'round': getattr(global_model, 'aggregation_round', None),
            'error': str(e)
        }
        save_aggregation_log(log_entry, file_path=os.path.join(LOGS_DIR, 'aggregation_log.json'))
        return False

def load_evaluation_data():
    """
    Load a small subset of data for global model evaluation.
    
    In a real federated learning system, the server might have a separate validation dataset
    or might rely on clients to report metrics. For simplicity, we're loading data here.
    """
    global global_dataset
    
    try:
        # Load samples from both banks for demonstration
        df1 = pd.read_csv(os.path.join(DATASETS_DIR, 'bank1.csv'))
        df2 = pd.read_csv(os.path.join(DATASETS_DIR, 'bank2.csv'))
        
        # Combine datasets
        df = pd.concat([df1, df2], ignore_index=True)
        
        # Extract features and target
        X = df[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]
        y = df['isFraud']
        
        global_dataset = (X, y)
        print(f"Evaluation data loaded. Shape: {X.shape}")
        
    except Exception as e:
        print(f"Error loading evaluation data: {e}")
        global_dataset = None

def run_server(host='0.0.0.0', port=5000):
    """Run the Flask server."""
    app.run(host=host, port=port, debug=True, use_reloader=False)

if __name__ == "__main__":
    run_server() 