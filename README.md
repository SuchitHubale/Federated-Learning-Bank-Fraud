# 🏛️ Federated Learning Framework for Banking Fraud Detection

A collaborative Machine Learning system that allows banks to train fraud detection models without sharing sensitive transaction data, while a centralized crime branch aggregates and monitors the models.

## 📋 Project Overview

This project demonstrates a federated learning system for fraud detection in banking transactions. It consists of:

- **Client 1 (Bank 1)**: Trains a local fraud detection model on its own transaction data
- **Client 2 (Bank 2)**: Trains a local fraud detection model on its own transaction data
- **Server (Crime Branch)**: Aggregates client models using Federated Averaging (FedAvg) algorithm
- **Dashboard**: Visualizes training progress, metrics, and performance of all models

The system allows banks to collaboratively train a global fraud detection model while keeping their transaction data private and secure on their own premises.

## 🔍 Key Features

- **Privacy Preservation**: Raw transaction data stays local to each bank
- **Collaborative Learning**: Knowledge from all participants contributes to a better global model
- **Centralized Monitoring**: Crime branch coordinates and aggregates models
- **Real-time Dashboard**: Monitor training progress and performance metrics
- **One-click Deployment**: Run the entire pipeline with a single command

## 🗂️ Project Structure

```
federated_fraud_detection/
├── client1/                 # Bank 1 implementation
│   ├── data/                # Local data directory
│   ├── client.py            # Client implementation
│   ├── local_model.py       # Local model training
│   └── utils.py             # Utility functions
├── client2/                 # Bank 2 implementation (similar structure)
├── server/                  # Crime Branch implementation
│   ├── logs/                # Server logs
│   ├── global_model.py      # Global model implementation
│   ├── server.py            # Flask server for model exchange
│   └── utils.py             # Server utilities
├── dashboard/               # Streamlit dashboard
│   ├── logs/                # Metrics and visualization data
│   └── dashboard.py         # Streamlit app
├── models/                  # Saved model files
├── datasets/                # Dataset files
│   ├── bank1.csv            # Bank 1 transactions
│   └── bank2.csv            # Bank 2 transactions
├── main.py                  # Main script to run the system
└── requirements.txt         # Dependencies
```

## 📊 Dataset

The dataset consists of banking transaction records with the following features:
- `step`: Time step of transaction
- `type`: Transaction type (TRANSFER, CASH_OUT, etc.)
- `amount`: Transaction amount
- `nameOrig`: Origin account
- `oldbalanceOrg`: Origin balance before
- `newbalanceOrig`: Origin balance after
- `nameDest`: Destination account
- `oldbalanceDest`: Destination balance before
- `newbalanceDest`: Destination balance after
- `isFraud`: Target label (1 = Fraud, 0 = Legitimate)
- `isFlaggedFraud`: Flagged fraud (optional)

## 🧠 Machine Learning Model

The fraud detection model is a **Random Forest Classifier** with the following features:
- Transaction type (encoded)
- Transaction amount
- Account balances (before and after)

## 🚀 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/federated-fraud-detection.git
   cd federated-fraud-detection
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the system**:
   ```bash
   python main.py
   ```

## 🔄 Workflow

1. **Clients (Banks)**:
   - Load and preprocess their local transaction data
   - Train local fraud detection models
   - Send model weights to the server
   - Receive the global model from the server

2. **Server (Crime Branch)**:
   - Receive local models from banks
   - Aggregate models using Federated Averaging (FedAvg)
   - Evaluate the global model performance
   - Send the global model back to clients

3. **Dashboard**:
   - Display training progress and metrics
   - Compare performance of local and global models
   - Show classification reports and evaluation metrics

## 📈 Dashboard

The Streamlit dashboard provides:
- Real-time monitoring of training metrics (accuracy, precision, recall, F1-score)
- Comparison of Client 1, Client 2, and Global model performances
- Classification reports and confusion matrices
- Aggregation logs and training history

## 💻 Technical Implementation

- **Backend**: Python, Flask API for server-client communication
- **Machine Learning**: Scikit-learn for model training and evaluation
- **Federated Learning**: Custom FedAvg implementation for model aggregation
- **Frontend**: Streamlit for dashboard visualization

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- The dataset is inspired by real-world banking transaction data
- The federated learning approach is based on research in privacy-preserving machine learning

## 📞 Contact

For any questions or feedback, please contact:
- Email: example@email.com
- GitHub: [https://github.com/Tanmaygangurde20](https://github.com/Tanmaygangurde20) 
