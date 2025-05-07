import streamlit as st
import json
import pandas as pd
from datetime import datetime

# Page configuration with a more professional setup
st.set_page_config(
    page_title="Federated Learning Dashboard",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- HARDCODED LOGS (PASTE YOUR LOGS HERE) ---
CLIENT1_LOG = '''
[CLIENT1] Client 1 started!
[CLIENT1] Loading data...
Loading data from ./datasets/bank1.csv
[CLIENT1] Preprocessing data...
[CLIENT1] Data loaded and preprocessed. Training set size: (6570, 10), Test set size: (1643, 10)
[CLIENT1] Training local model...
Training local model...
Epoch 1/5
Epoch 1 - Accuracy: 0.9866, Loss: 0.0470
Epoch 2/5
Epoch 2 - Accuracy: 0.9866, Loss: 0.0470
Epoch 3/5
Epoch 3 - Accuracy: 0.9866, Loss: 0.0470
Epoch 4/5
Epoch 4 - Accuracy: 0.9866, Loss: 0.0470
Epoch 5/5
Epoch 5 - Accuracy: 0.9866, Loss: 0.0470
Metrics saved to ../dashboard/logs/client1_metrics.json
Model saved to ./models/local_model1.pkl
[CLIENT1] Local model training completed. Final accuracy: 0.9866
[CLIENT1] Sending model to server...
Model successfully sent to server
[CLIENT1] Waiting for global model...
[CLIENT1] Receiving global model from server...
Failed to receive global model. Status code: 404
Response: {
  "error": "Global model not available yet"
}

[CLIENT1] Loading data...
Loading data from ./datasets/bank1.csv
[CLIENT1] Preprocessing data...
[CLIENT1] Data loaded and preprocessed. Training set size: (6570, 10), Test set size: (1643, 10)
[CLIENT1] Model evaluation - Accuracy: 0.9866
'''

CLIENT2_LOG = '''
[CLIENT2] Client 2 started!
[CLIENT2] Loading data...
Loading data from ./datasets/bank2.csv
[CLIENT2] Preprocessing data...
[CLIENT2] Data loaded and preprocessed. Training set size: (6570, 10), Test set size: (1643, 10)
[CLIENT2] Training local model...
Training local model...
Epoch 1/5
Epoch 1 - Accuracy: 0.9897, Loss: 0.0480
Epoch 2/5
Epoch 2 - Accuracy: 0.9897, Loss: 0.0480
Epoch 3/5
Epoch 3 - Accuracy: 0.9897, Loss: 0.0480
Epoch 4/5
Epoch 4 - Accuracy: 0.9897, Loss: 0.0480
Epoch 5/5
Epoch 5 - Accuracy: 0.9897, Loss: 0.0480
Metrics saved to ../dashboard/logs/client2_metrics.json
Model saved to ./models/local_model2.pkl
[CLIENT2] Local model training completed. Final accuracy: 0.9897
[CLIENT2] Sending model to server...
Model successfully sent to server
[CLIENT2] Waiting for global model...
[CLIENT2] Receiving global model from server...
Global model received and saved to ./models/global_model.pkl
[CLIENT2] Loading global model...
Model loaded from ./models/local_model2.pkl
[CLIENT2] Loading data...
Loading data from ./datasets/bank2.csv
[CLIENT2] Preprocessing data...
[CLIENT2] Data loaded and preprocessed. Training set size: (6570, 10), Test set size: (1643, 10)
[CLIENT2] Model evaluation - Accuracy: 0.9897
'''

AGGREGATION_LOG = [
    {
        "round": 1,
        "client_models": ["client1", "client2"],
        "weights": [0.5, 0.5],
        "timestamp": "{\"$date\": {\"$numberLong\": \"1746\"}}"
    },
    {
        "round": 1,
        "error": "Feature mismatch: Model expects 10 features, got 9"
    },
    {
        "round": 1,
        "client_models": ["client1", "client2"],
        "weights": [0.5, 0.5],
        "timestamp": "{\"$date\": {\"$numberLong\": \"1746\"}}"
    },
    {
        "round": 1,
        "error": "Feature mismatch: Model expects 10 features, got 9"
    },
    {
        "round": 1,
        "client_models": ["client1", "client2"],
        "weights": [0.5, 0.5],
        "timestamp": "{\"$date\": {\"$numberLong\": \"1746\"}}"
    }
]

GLOBAL_EVAL = {
    "round": 1,
    "metrics": {
        "accuracy": 0.987522824102252,
        "precision": 0.9796893667861409,
        "recall": 0.9957498482088646,
        "f1": 0.9876543209876543,
        "loss": 0.043048118477596625,
        "classification_report": {
            "NonFraud": {
                "precision": 0.9956575682382134,
                "recall": 0.9792556436851739,
                "f1-score": 0.9873884958474316,
                "support": 1639.0
            },
            "Fraud": {
                "precision": 0.9796893667861409,
                "recall": 0.9957498482088646,
                "f1-score": 0.9876543209876543,
                "support": 1647.0
            },
            "accuracy": 0.987522824102252,
            "macro avg": {
                "precision": 0.9876734675121772,
                "recall": 0.9875027459470193,
                "f1-score": 0.987521408417543,
                "support": 3286.0
            },
            "weighted avg": {
                "precision": 0.987654029652832,
                "recall": 0.987522824102252,
                "f1-score": 0.9875217320026193,
                "support": 3286.0
            }
        }
    }
}

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    /* Main Theme Colors */
    :root {
        --primary: #3a86ff;
        --primary-light: #bfd7ff;
        --secondary: #8338ec;
        --accent: #ff006e;
        --success: #38b000;
        --warning: #ffbe0b;
        --danger: #ff5252;
        --info: #00b4d8;
        --dark: #073b4c;
        --light: #f8f9fa;
        --gray: #6c757d;
        --shadow: rgba(0, 0, 0, 0.1);
    }
    
    /* Global Styles */
    .stApp {
        background-color: #f0f4f8;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Segoe UI', Roboto, sans-serif;
        letter-spacing: -0.5px;
    }
    
    /* Dashboard Title */
    .dashboard-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        padding-bottom: 0;
        line-height: 1.2;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: var(--gray);
        margin-top: 0;
        padding-top: 0;
        margin-bottom: 1.5rem;
    }
    
    /* Section Titles */
    .section-title {
        font-size: 1.7rem;
        font-weight: 700;
        color: var(--dark);
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--primary-light);
    }
    
    /* Cards */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px var(--shadow);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        overflow: hidden;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.12);
    }
    
    .card-header {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: var(--dark);
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Client Log Cards */
    .log-card {
        background: #fff;
        border-radius: 8px;
        padding: 0;
        margin-bottom: 1rem;
        border-left: 4px solid var(--primary);
        box-shadow: 0 2px 8px var(--shadow);
        max-height: 500px;
        overflow-y: auto;
    }
    
    .log-header {
        background-color: var(--primary-light);
        color: var(--dark);
        padding: 0.75rem 1rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .log-content {
        padding: 1rem;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        background-color: #f8fafc;
        color: #334155;
        white-space: pre-wrap;
        max-height: 450px;
        overflow-y: auto;
    }
    
    /* Aggregation Log */
    .agg-card {
        background: white;
        border-radius: 12px;
        border-left: 4px solid var(--secondary);
        padding: 0;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px var(--shadow);
        overflow: hidden;
    }
    
    .agg-header {
        background-color: #f0e6ff;
        padding: 0.75rem 1rem;
        font-weight: 600;
        color: var(--secondary);
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .agg-content {
        padding: 1rem;
    }
    
    /* Metric Cards */
    .metrics-container {
        display: flex;
        gap: 15px;
        flex-wrap: wrap;
        margin-bottom: 1.5rem;
    }
    
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        flex: 1;
        min-width: 160px;
        box-shadow: 0 4px 12px var(--shadow);
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-title {
        font-size: 1rem;
        font-weight: 600;
        color: var(--gray);
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    
    /* Color variations for metrics */
    .metric-accuracy { color: var(--primary); }
    .metric-precision { color: var(--success); }
    .metric-recall { color: var(--warning); }
    .metric-f1 { color: var(--secondary); }
    .metric-loss { color: var(--danger); }
    
    /* Classification Report Table */
    .classification-table {
        margin-top: 1rem;
    }
    
    /* Sidebar */
    .sidebar-header {
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .sidebar-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--primary);
        margin-top: 0.5rem;
    }
    
    .sidebar-nav {
        margin-bottom: 2rem;
    }
    
    .sidebar-nav a {
        text-decoration: none;
        color: var(--dark);
        transition: color 0.2s ease;
    }
    
    .sidebar-nav a:hover {
        color: var(--primary);
    }
    
    .sidebar-metric {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin-top: 1.5rem;
    }
    
    /* DataFrames */
    .dataframe-container {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 8px var(--shadow);
    }
    
    /* Highlight important values */
    .highlight {
        background: linear-gradient(120deg, rgba(58, 134, 255, 0.2) 0%, rgba(131, 56, 236, 0.2) 100%);
        padding: 0.15rem 0.4rem;
        border-radius: 4px;
        font-weight: 600;
    }
    
    /* ScrollBar customization */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
    }
    
    /* Badges */
    .badge {
        padding: 0.25rem 0.6rem;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .badge-success {
        background-color: rgba(56, 176, 0, 0.1);
        color: var(--success);
    }
    
    .badge-warning {
        background-color: rgba(255, 190, 11, 0.1);
        color: var(--warning);
    }
    
    .badge-danger {
        background-color: rgba(255, 82, 82, 0.1);
        color: var(--danger);
    }
    
    /* Tables */
    .styled-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        background: white;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    .styled-table th {
        background-color: var(--primary-light);
        color: var(--dark);
        font-weight: 600;
        text-align: left;
        padding: 12px 15px;
        border-bottom: 2px solid #e0e0e0;
    }
    
    .styled-table td {
        padding: 10px 15px;
        border-bottom: 1px solid #f0f0f0;
    }
    
    .styled-table tr:last-child td {
        border-bottom: none;
    }
    
    .styled-table tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    
    .styled-table tr:hover {
        background-color: #f1f7ff;
    }
    
    /* Custom DataTable */
    div[data-testid="stDataFrame"] div[data-testid="stTable"] {
        max-height: 400px;
    }
    
    div[data-testid="stDataFrame"] th {
        background-color: var(--primary-light);
        color: var(--dark);
        padding: 12px 15px;
    }
    
    div[data-testid="stDataFrame"] td {
        padding: 10px 15px;
    }
    
    /* Animation for metrics */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.03); }
        100% { transform: scale(1); }
    }
    
    .pulse-animation {
        animation: pulse 2s infinite;
    }

    /* Make links and buttons more professional-looking */
    a, button {
        font-weight: 500;
    }
    
    /* Add subtle dividers */
    .divider {
        height: 1px;
        background: linear-gradient(to right, transparent, var(--primary-light), transparent);
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown('<div class="sidebar-header">', unsafe_allow_html=True)
    st.image("https://img.icons8.com/color/96/000000/bank-building.png", width=70)
    st.markdown('<div class="sidebar-title">Federated Learning<br>Dashboard</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-nav">', unsafe_allow_html=True)
    st.markdown("""
    **Navigation**
    - [Dashboard Overview](#dashboard-overview)
    - [Client Logs](#client-logs)
    - [Aggregation Process](#aggregation-process)
    - [Global Model Performance](#global-model-performance)
    - [Classification Report](#classification-report)
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("🔍 Quick Stats", expanded=True):
        st.markdown(
            f'<div class="sidebar-metric">'
            f'<div style="font-size:0.9rem;opacity:0.8;">Global Accuracy</div>'
            f'<div style="font-size:1.8rem;font-weight:700;">{GLOBAL_EVAL["metrics"]["accuracy"]:.4f}</div>'
            f'</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="F1 Score", value=f"{GLOBAL_EVAL['metrics']['f1']:.4f}")
        with col2:
            st.metric(label="Loss", value=f"{GLOBAL_EVAL['metrics']['loss']:.4f}")

# --- MAIN CONTENT ---

# Header
st.markdown('<div class="dashboard-title">Federated Learning Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time monitoring of federated model training across distributed bank data sources</div>', unsafe_allow_html=True)

# Dashboard Overview
st.markdown('<div class="section-title" id="dashboard-overview">Dashboard Overview</div>', unsafe_allow_html=True)

# Key metrics at a glance
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <span>🏦</span> Client Nodes
        </div>
        <div style="text-align:center;">
            <span style="font-size:2.5rem;font-weight:700;color:#3a86ff;">2</span>
            <div style="font-size:0.9rem;color:#6c757d;margin-top:0.5rem;">Financial institutions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="card">
        <div class="card-header">
            <span>🔄</span> FL Rounds
        </div>
        <div style="text-align:center;">
            <span style="font-size:2.5rem;font-weight:700;color:#8338ec;">1</span>
            <div style="font-size:0.9rem;color:#6c757d;margin-top:0.5rem;">Training iterations</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="card">
        <div class="card-header">
            <span>📊</span> Total Records
        </div>
        <div style="text-align:center;">
            <span style="font-size:2.5rem;font-weight:700;color:#ff006e;">16,426</span>
            <div style="font-size:0.9rem;color:#6c757d;margin-top:0.5rem;">Data points analyzed</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="card">
        <div class="card-header">
            <span>🔒</span> Privacy Level
        </div>
        <div style="text-align:center;">
            <span style="font-size:2.5rem;font-weight:700;color:#38b000;">High</span>
            <div style="font-size:0.9rem;color:#6c757d;margin-top:0.5rem;">Data remains local</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Global model performance visualization
st.markdown('<div class="section-title" id="global-model-performance">Global Model Performance</div>', unsafe_allow_html=True)

metrics = GLOBAL_EVAL["metrics"]
st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Accuracy</div>
        <div class="metric-value metric-accuracy">{metrics["accuracy"]:.4f}</div>
    </div>
    <div class="metric-card">
        <div class="metric-title">Precision</div>
        <div class="metric-value metric-precision">{metrics["precision"]:.4f}</div>
    </div>
    <div class="metric-card">
        <div class="metric-title">Recall</div>
        <div class="metric-value metric-recall">{metrics["recall"]:.4f}</div>
    </div>
    <div class="metric-card">
        <div class="metric-title">F1 Score</div>
        <div class="metric-value metric-f1">{metrics["f1"]:.4f}</div>
    </div>
    <div class="metric-card">
        <div class="metric-title">Loss</div>
        <div class="metric-value metric-loss">{metrics["loss"]:.4f}</div>
    </div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- CLIENT LOGS ---
st.markdown('<div class="section-title" id="client-logs">Client Logs</div>', unsafe_allow_html=True)
cl1, cl2 = st.columns(2)

with cl1:
    st.markdown("""
    <div class="log-card">
        <div class="log-header">
            <span>🏦</span> Client 1 Training Log
        </div>
        <div class="log-content">""" + CLIENT1_LOG + """</div>
    </div>
    """, unsafe_allow_html=True)

with cl2:
    st.markdown("""
    <div class="log-card">
        <div class="log-header">
            <span>🏦</span> Client 2 Training Log
        </div>
        <div class="log-content">""" + CLIENT2_LOG + """</div>
    </div>
    """, unsafe_allow_html=True)

# --- AGGREGATION LOGS ---
st.markdown('<div class="section-title" id="aggregation-process">Aggregation Process</div>', unsafe_allow_html=True)

if AGGREGATION_LOG:
    # Process aggregation logs
    log_data = []
    for entry in AGGREGATION_LOG:
        timestamp = entry.get("timestamp", "")
        if isinstance(timestamp, str) and "numberLong" in timestamp:
            try:
                timestamp_dict = json.loads(timestamp)
                timestamp_ms = int(timestamp_dict["$date"]["$numberLong"])
                timestamp = datetime.fromtimestamp(timestamp_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass
                
        error = entry.get("error", "")
        status = "⚠️ Error" if error else "✅ Success"
        status_color = "badge-danger" if error else "badge-success"
        
        log_data.append({
            "Round": entry.get("round", ""),
            "Client Models": ", ".join(entry.get("client_models", [])) if entry.get("client_models") else "",
            "Weights": ", ".join([f"{w:.2f}" for w in entry.get("weights", [])]) if entry.get("weights") else "",
            "Status": status,
            "Status Color": status_color,
            "Error": error,
            "Timestamp": timestamp
        })
    
    # Display in an enhanced table
    st.markdown("""
    <div class="agg-card">
        <div class="agg-header">
            <span>🔄</span> Model Aggregation Logs
        </div>
        <div class="agg-content">
    """, unsafe_allow_html=True)
    
    # Custom aggregation log table
    df_log = pd.DataFrame(log_data)
    
    # Create a styled dataframe
    if not df_log.empty:
        # Apply custom styling to the dataframe
        def highlight_status(val):
            if val == "⚠️ Error":
                return 'background-color: rgba(255, 82, 82, 0.1); color: #ff5252; font-weight: 600;'
            elif val == "✅ Success":
                return 'background-color: rgba(56, 176, 0, 0.1); color: #38b000; font-weight: 600;'
            return ''
        
        # Style the dataframe
        styled_df = df_log.style.applymap(
            highlight_status, subset=['Status']
        ).format({
            'Round': '{}',
            'Client Models': '{}',
            'Weights': '{}',
            'Status': '{}',
            'Error': '{}',
            'Timestamp': '{}'
        })
        
        # Display styled dataframe
        st.dataframe(
            styled_df, 
            use_container_width=True,
            height=300,
            hide_index=True,
            column_order=['Round', 'Client Models', 'Weights', 'Status', 'Error', 'Timestamp']
        )
    else:
        st.info("No aggregation logs available.")
    
    st.markdown('</div></div>', unsafe_allow_html=True)
else:
    st.info("No aggregation logs available.")

# --- CLASSIFICATION REPORT ---
st.markdown('<div class="section-title" id="classification-report">Classification Report</div>', unsafe_allow_html=True)

st.markdown("""
<div class="card">
    <div class="card-header">
        <span>📊</span> Detailed Classification Metrics
    </div>
""", unsafe_allow_html=True)

report = metrics["classification_report"]
report_df = pd.DataFrame(report).transpose()

if "support" in report_df.columns:
    report_df["support"] = report_df["support"].astype(int)

# Create a styled dataframe for the classification report
def style_dataframe(df):
    # Define a function to highlight high values
    def highlight_high_values(val):
        if isinstance(val, (int, float)):
            if val > 0.98:  # Highlight high accuracy/precision/recall values
                return 'background-color: rgba(56, 176, 0, 0.1); color: #38b000; font-weight: 600;'
            elif val > 0.95:
                return 'background-color: rgba(255, 190, 11, 0.1); color: #ff9a00; font-weight: 600;'
        return ''
    
    # Apply styling
    return df.style.applymap(highlight_high_values).format({
        "precision": "{:.4f}",
        "recall": "{:.4f}",
        "f1-score": "{:.4f}",
        "support": "{:.0f}"
    })

styled_report_df = style_dataframe(report_df)
st.dataframe(styled_report_df, use_container_width=True, height=300)

st.markdown('</div>', unsafe_allow_html=True)

# Add a footer with additional context
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; padding: 1rem 0; color: #6c757d; font-size: 0.9rem;">
    Federated Learning Dashboard | Secure banking fraud detection across distributed data sources
    <br>
    <span style="font-size: 0.8rem;">Data remains private at each bank while model intelligence is shared</span>
</div>
""", unsafe_allow_html=True)