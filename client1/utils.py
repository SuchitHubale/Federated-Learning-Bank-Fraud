import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
import json
import requests

def load_data(data_path):
    """Load and preprocess the bank1 dataset."""
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    return df

def preprocess_data(df):
    """Preprocess the data for model training."""
    # Select relevant features
    X = df[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]
    y = df['isFraud']
    
    # Define preprocessing for categorical and numerical features
    categorical_features = ['type']
    numerical_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    
    # Create transformers
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Combine transformers in a preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numerical_transformer, numerical_features)
        ])
    
    # Preprocess the data
    X_processed = preprocessor.fit_transform(X)
    
    # Save the preprocessor for later use
    os.makedirs('../models', exist_ok=True)
    joblib.dump(preprocessor, './models/client1_preprocessor.pkl')
    
    return X_processed, y, preprocessor

def save_metrics(metrics, file_path='../dashboard/logs/client1_metrics.json'):
    """Save training metrics to a JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {file_path}")

def send_model_to_server(model_path, server_url='http://localhost:5000'):
    """Send the trained model to the server."""
    try:
        files = {'model': open(model_path, 'rb')}
        response = requests.post(f"{server_url}/client1", files=files)
        if response.status_code == 200:
            print("Model successfully sent to server")
            return True
        else:
            print(f"Failed to send model to server. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"Error sending model to server: {e}")
        return False

def receive_global_model(server_url='http://localhost:5000/get_global_model', save_path='../models/global_model.pkl'):
    """Receive the global model from the server."""
    try:
        response = requests.get(server_url)
        if response.status_code == 200:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"Global model received and saved to {save_path}")
            return True
        else:
            print(f"Failed to receive global model. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"Error receiving global model: {e}")
        return False 