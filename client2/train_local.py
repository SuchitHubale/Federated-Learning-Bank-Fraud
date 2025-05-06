import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import pickle
import requests

def prepare_data(data_path):
    """Prepare the data for training"""
    # Load data
    df = pd.read_csv(data_path)
    
    # Drop unnecessary columns
    df = df.drop(['step', 'nameOrig', 'nameDest'], axis=1)
    
    # Encode categorical features
    le = LabelEncoder()
    df['type'] = le.fit_transform(df['type'])
    
    # Split features and target
    X = df.drop(['isFraud', 'isFlaggedFraud'], axis=1)
    y = df['isFraud']
    
    # Save feature names and scaler for future inference
    feature_names = X.columns.tolist()
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler for future use
    os.makedirs('models', exist_ok=True)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
    
    return X_train, y_train, X_test, y_test, X.shape[1]

def train_model(model, X_train, y_train, X_test, y_test, epochs=20, batch_size=64, learning_rate=0.001, client_id=None, server_url=None):
    """Train the model and return the trained model"""
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train.size()[0])
        running_loss = 0.0
        
        for i in range(0, X_train.size()[0], batch_size):
            optimizer.zero_grad()
            
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validation at each epoch
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_preds = (val_outputs > 0.5).float()
            val_accuracy = accuracy_score(y_test, val_preds)
            val_precision = precision_score(y_test, val_preds)
            val_recall = recall_score(y_test, val_preds)
            val_f1 = f1_score(y_test, val_preds)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        # Send metrics to server
        if client_id and server_url:
            try:
                requests.post(f"{server_url}/metrics", json={
                    'client_id': client_id,
                    'epoch': epoch+1,
                    'metrics': {
                        'loss': running_loss,
                        'val_accuracy': val_accuracy,
                        'precision': val_precision,
                        'recall': val_recall,
                        'f1': val_f1
                    }
                })
            except Exception as e:
                print(f"Warning: Could not send metrics to server: {e}")
    # Final evaluation
    model.eval()
    with torch.no_grad():
        y_pred_probs = model(X_test)
        y_pred = (y_pred_probs > 0.5).float()
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print("\n🔵 Final Evaluation Metrics:")
        print(f"Accuracy  : {acc:.4f}")
        print(f"Precision : {prec:.4f}")
        print(f"Recall    : {rec:.4f}")
        print(f"F1 Score  : {f1:.4f}")
    # Upload scaler to server
    if server_url:
        try:
            with open('models/scaler.pkl', 'rb') as f:
                requests.post(f"{server_url}/upload_scaler", files={'scaler': f})
        except Exception as e:
            print(f"Warning: Could not upload scaler to server: {e}")
    return model 