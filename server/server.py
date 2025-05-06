from flask import Flask, request, send_file, jsonify
import os
import torch
import numpy as np
import pickle
from aggregation import fed_avg
from datetime import datetime
import torch.nn as nn
import json
import pandas as pd

app = Flask(__name__)

# Store received weights and client IDs
received_weights = []
client_ids = []
log_file = 'server/logs/server_log.txt'
global_model_path = 'server/models/global_model.pth'

# Metrics storage
metrics_file = 'server/logs/metrics.json'
if not os.path.exists(metrics_file):
    with open(metrics_file, 'w') as f:
        json.dump({}, f)

# Ensure directories exist
os.makedirs('server/models', exist_ok=True)
os.makedirs('server/logs', exist_ok=True)

# Helper: log messages
def log(message):
    print(message)
    with open(log_file, 'a') as f:
        f.write(f"{datetime.now()} - {message}\n")

# Model architecture (same as client)
class FraudDetectionModel(nn.Module):
    def __init__(self, input_dim):
        super(FraudDetectionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

# Initialize global model if not exists
init_input_dim = 6  # amount, type, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest
if not os.path.exists(global_model_path):
    model = FraudDetectionModel(init_input_dim)
    torch.save(model.state_dict(), global_model_path)
    log('Initialized global model at server startup')

current_round = 1
last_aggregated_round = 0
EXPECTED_CLIENTS = 2

@app.route('/update', methods=['POST'])
def update():
    global last_aggregated_round, current_round
    data = request.get_json()
    client_id = data['client_id']
    weights = data['weights']
    received_weights.append(weights)
    client_ids.append(client_id)
    log(f"Received weights from {client_id} (Round {current_round})")
    # Automatic aggregation
    if len(received_weights) == EXPECTED_CLIENTS:
        avg_weights = fed_avg(received_weights)
        torch.save({'weights': avg_weights, 'round': current_round}, global_model_path)
        log(f"Aggregated weights from clients: {client_ids} (Round {current_round}) [AUTO]")
        last_aggregated_round = current_round
        current_round += 1
        received_weights.clear()
        client_ids.clear()
    return jsonify({'status': 'received', 'round': current_round})

@app.route('/upload_scaler', methods=['POST'])
def upload_scaler():
    if 'scaler' not in request.files:
        return jsonify({'error': 'No scaler file uploaded'}), 400
    scaler_file = request.files['scaler']
    scaler_path = 'server/models/scaler.pkl'
    scaler_file.save(scaler_path)
    log('Scaler uploaded and saved to server/models/scaler.pkl')
    return jsonify({'status': 'scaler uploaded'})

@app.route('/model', methods=['GET'])
def get_model():
    # Only serve model if aggregation has occurred for the current round
    if not os.path.exists(global_model_path):
        return jsonify({'error': 'No global model available'}), 404
    try:
        model_data = torch.load(global_model_path)
        model_round = model_data.get('round', 0)
    except Exception:
        model_round = 0
    if model_round < last_aggregated_round:
        return jsonify({'error': 'Global model not updated for this round'}), 404
    log(f"Global model sent to client (Round {last_aggregated_round})")
    return send_file(global_model_path, as_attachment=True)

@app.route('/logs', methods=['GET'])
def get_logs():
    if not os.path.exists(log_file):
        open(log_file, 'w').close()  # Create empty log file if it doesn't exist
    with open(log_file, 'r') as f:
        logs = f.read()
    return jsonify({'logs': logs})

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({
        'current_round': current_round,
        'last_aggregated_round': last_aggregated_round,
        'clients_waiting': len(received_weights),
    })

@app.route('/metrics', methods=['GET'])
def get_metrics():
    if not os.path.exists(metrics_file):
        return jsonify({})
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    return jsonify(metrics)

@app.route('/metrics', methods=['POST'])
def post_metrics():
    data = request.get_json()
    client_id = data['client_id']
    epoch = data['epoch']
    metrics = data['metrics']
    # Accept additional metrics
    allowed = {'loss', 'val_accuracy', 'precision', 'recall', 'f1'}
    filtered_metrics = {k: v for k, v in metrics.items() if k in allowed}
    # Load existing metrics
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}
    if client_id not in all_metrics:
        all_metrics[client_id] = []
    all_metrics[client_id].append({'epoch': epoch, **filtered_metrics, 'round': last_aggregated_round})
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f)
    log(f"Received metrics from {client_id} for epoch {epoch} (Round {last_aggregated_round})")
    return jsonify({'status': 'metrics received'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Expected: data is a dict with feature names as in the CSV (minus target columns)
    # Example: {'step': 1, 'type': 'PAYMENT', ...}
    # 1. Load scaler and label encoder
    scaler_path = 'models/scaler.pkl'
    if not os.path.exists(scaler_path):
        scaler_path = 'server/models/scaler.pkl'
    if not os.path.exists(scaler_path):
        return jsonify({'error': 'Scaler not found on server.'}), 500
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    # For label encoding 'type', use the same mapping as in training
    type_map = {'PAYMENT': 1, 'TRANSFER': 4, 'CASH_OUT': 0, 'DEBIT': 2, 'CASH_IN': 3}  # Example, adjust as needed
    # 2. Prepare input features (order: amount, type, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest)
    feature_order = ['amount', 'type', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    try:
        input_row = []
        for feat in feature_order:
            if feat == 'type':
                val = data[feat]
                val = type_map.get(val, 0)
            else:
                val = float(data[feat])
            input_row.append(val)
        X = np.array(input_row).reshape(1, -1)
        X_scaled = scaler.transform(X)
    except Exception as e:
        return jsonify({'error': f'Input preprocessing failed: {e}'}), 400
    # 3. Load model
    global_model_path = 'server/models/global_model.pth'
    if not os.path.exists(global_model_path):
        return jsonify({'error': 'No global model available'}), 404
    model_data = torch.load(global_model_path, map_location=torch.device('cpu'))
    input_dim = X_scaled.shape[1]
    model = FraudDetectionModel(input_dim)
    if isinstance(model_data, dict) and 'weights' in model_data:
        weights = model_data['weights']
        # Convert weights to tensors if needed
        for param, w in zip(model.parameters(), weights):
            param.data = torch.tensor(w, dtype=param.data.dtype)
    else:
        model.load_state_dict(model_data)
    model.eval()
    # 4. Predict
    with torch.no_grad():
        x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        pred = model(x_tensor).item()
    return jsonify({'fraud_probability': float(pred), 'fraud_prediction': int(pred > 0.5)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 