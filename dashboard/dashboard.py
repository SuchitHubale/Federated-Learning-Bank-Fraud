import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import time
from datetime import datetime
from typing import Optional

# Set page config
st.set_page_config(
    page_title="Federated Learning Dashboard",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
REFRESH_INTERVAL = 10  # Refresh every 10 seconds
LOGS_DIR = 'logs'
CLIENT1_METRICS_PATH = os.path.join(LOGS_DIR, 'client1_metrics.json')
CLIENT2_METRICS_PATH = os.path.join(LOGS_DIR, 'client2_metrics.json')
GLOBAL_METRICS_PATH = os.path.join(LOGS_DIR, 'global_metrics.json')
AGGREGATION_LOG_PATH = os.path.join(LOGS_DIR, 'aggregation_log.json')

# Sidebar
st.sidebar.title("🏛️ Federated Learning Dashboard")
st.sidebar.subheader("Banking Fraud Detection")

# Auto-refresh option
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
if auto_refresh:
    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 5, 60, REFRESH_INTERVAL)

# Helper functions
def load_metrics(file_path: str) -> Optional[dict]:
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading metrics from {file_path}: {e}")
    return None

def load_aggregation_log() -> list:
    if os.path.exists(AGGREGATION_LOG_PATH):
        try:
            with open(AGGREGATION_LOG_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading aggregation log: {e}")
    return []

def safe_fmt(val):
    return f"{val:.4f}" if isinstance(val, (float, int)) and val is not None else (val if val is not None else "N/A")

def status_card(title, status, value=None, color=None, icon=None, help_text=None):
    color = color or ("#e0f7fa" if status == "waiting" else ("#e8f5e9" if status == "done" else "#fffde7"))
    icon = icon or ("⏳" if status == "waiting" else ("✅" if status == "done" else "⚠️"))
    st.markdown(f'''
        <div style="background:{color};border-radius:12px;padding:1.2em 1em 1em 1em;margin-bottom:1em;box-shadow:0 2px 8px rgba(0,0,0,0.04);">
            <span style="font-size:1.5em;">{icon}</span> <span style="font-size:1.2em;font-weight:600;">{title}</span><br>
            <span style="font-size:2em;font-weight:bold;color:#1a7ee6;">{value if value is not None else ''}</span>
            <div style="font-size:1em;color:#888;">{help_text if help_text else ''}</div>
        </div>
    ''', unsafe_allow_html=True)

def metric_card(label, value, color=None):
    color = color or ("#f7fafd" if value != "N/A" else "#fffde7")
    st.markdown(f'''
        <div style="background:{color};border-radius:8px;padding:0.7em 1em 0.7em 1em;display:inline-block;margin:0.2em 0.5em 0.2em 0;">
            <span style="font-size:1.1em;font-weight:600;">{label}:</span> <span style="font-size:1.1em;">{value}</span>
        </div>
    ''', unsafe_allow_html=True)

def format_classification_report(report, title):
    if not report:
        st.info(f"No {title} classification report available yet.")
        return
    try:
        df = pd.DataFrame(report).transpose()
        if "support" in df.columns:
            df["support"] = df["support"].astype(int)
        st.dataframe(df.style.format({
            "precision": lambda x: f"{x:.4f}" if isinstance(x, (float, int)) else x,
            "recall": lambda x: f"{x:.4f}" if isinstance(x, (float, int)) else x,
            "f1-score": lambda x: f"{x:.4f}" if isinstance(x, (float, int)) else x
        }), use_container_width=True)
    except Exception as e:
        st.warning(f"Could not display {title} classification report: {e}")

def plot_metrics_history(client1_metrics, client2_metrics, global_metrics, metric_name):
    fig, ax = plt.subplots(figsize=(7, 4))
    if client1_metrics and "history" in client1_metrics and metric_name in client1_metrics["history"]:
        epochs = range(1, len(client1_metrics["history"][metric_name]) + 1)
        ax.plot(epochs, client1_metrics["history"][metric_name], marker='o', linestyle='-', label="🏦 Bank 1")
    if client2_metrics and "history" in client2_metrics and metric_name in client2_metrics["history"]:
        epochs = range(1, len(client2_metrics["history"][metric_name]) + 1)
        ax.plot(epochs, client2_metrics["history"][metric_name], marker='s', linestyle='-', label="🏦 Bank 2")
    if global_metrics and "metrics" in global_metrics and metric_name in global_metrics["metrics"]:
        ax.axhline(y=global_metrics["metrics"][metric_name], color='r', linestyle='--', label="🏛️ Crime Branch (Global)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name.capitalize())
    ax.set_title(f"{metric_name.capitalize()} During Training")
    ax.legend()
    ax.grid(True)
    return fig

def progress_status(client1_metrics, client2_metrics, global_metrics):
    steps = [
        ("Bank 1 Training", client1_metrics is not None),
        ("Bank 2 Training", client2_metrics is not None),
        ("Model Aggregation", global_metrics is not None)
    ]
    done = sum(1 for _, ok in steps if ok)
    st.progress(done / len(steps))
    for label, ok in steps:
        status_card(label, "done" if ok else "waiting", icon="✅" if ok else "⏳", help_text="Done" if ok else "Waiting...")

# Main dashboard
def main():
    st.title("🏛️ Federated Learning for Banking Fraud Detection")
    st.markdown("""
        <style>
        .section-title {
            font-size: 1.3em;
            font-weight: 600;
            margin-bottom: 0.5em;
        }
        </style>
    """, unsafe_allow_html=True)

    # Load metrics
    client1_metrics = load_metrics(CLIENT1_METRICS_PATH)
    client2_metrics = load_metrics(CLIENT2_METRICS_PATH)
    global_metrics = load_metrics(GLOBAL_METRICS_PATH)
    aggregation_log = load_aggregation_log()

    # --- PROGRESS BAR ---
    st.markdown('<div class="section-title">🚦 Progress Overview</div>', unsafe_allow_html=True)
    progress_status(client1_metrics, client2_metrics, global_metrics)

    # --- STATUS CARDS ---
    st.markdown('<div class="section-title">🔎 Model Status</div>', unsafe_allow_html=True)
    status_cols = st.columns(3)
    with status_cols[0]:
        if client1_metrics:
            status_card("🏦 Bank 1", "done", value=safe_fmt(client1_metrics.get("final_accuracy", client1_metrics.get("metrics", {}).get("accuracy", None))), help_text="Model Trained")
        else:
            status_card("🏦 Bank 1", "waiting", help_text="Waiting for training data...")
    with status_cols[1]:
        if client2_metrics:
            status_card("🏦 Bank 2", "done", value=safe_fmt(client2_metrics.get("final_accuracy", client2_metrics.get("metrics", {}).get("accuracy", None))), help_text="Model Trained")
        else:
            status_card("🏦 Bank 2", "waiting", help_text="Waiting for training data...")
    with status_cols[2]:
        if global_metrics:
            status_card("🏛️ Crime Branch (Global)", "done", value=safe_fmt(global_metrics.get("metrics", {}).get("accuracy", global_metrics.get("final_accuracy", None))), help_text="Model Aggregated")
        else:
            status_card("🏛️ Crime Branch (Global)", "waiting", help_text="Waiting for aggregation...")

    # --- METRICS ---
    st.markdown('<div class="section-title">📊 Metrics Overview</div>', unsafe_allow_html=True)
    metric_options = ["accuracy", "precision", "recall", "f1", "loss"]
    selected_metric = st.selectbox("Select Metric to Plot", metric_options, index=0, key="metric_select")
    if client1_metrics or client2_metrics or global_metrics:
        fig = plot_metrics_history(client1_metrics, client2_metrics, global_metrics, selected_metric)
        st.pyplot(fig)
    else:
        st.info("Waiting for training data...")
    # Final metrics comparison
    st.markdown("**Final Metrics Comparison**")
    metrics_data = {
        "Model": ["Bank 1", "Bank 2", "Crime Branch (Global)"],
        "Accuracy": [
            safe_fmt(client1_metrics.get("final_accuracy", client1_metrics.get("metrics", {}).get("accuracy", None))) if client1_metrics else "N/A",
            safe_fmt(client2_metrics.get("final_accuracy", client2_metrics.get("metrics", {}).get("accuracy", None))) if client2_metrics else "N/A",
            safe_fmt(global_metrics.get("metrics", {}).get("accuracy", global_metrics.get("final_accuracy", None))) if global_metrics else "N/A"
        ],
        "Precision": [
            safe_fmt(client1_metrics.get("final_precision", client1_metrics.get("metrics", {}).get("precision", None))) if client1_metrics else "N/A",
            safe_fmt(client2_metrics.get("final_precision", client2_metrics.get("metrics", {}).get("precision", None))) if client2_metrics else "N/A",
            safe_fmt(global_metrics.get("metrics", {}).get("precision", global_metrics.get("final_precision", None))) if global_metrics else "N/A"
        ],
        "Recall": [
            safe_fmt(client1_metrics.get("final_recall", client1_metrics.get("metrics", {}).get("recall", None))) if client1_metrics else "N/A",
            safe_fmt(client2_metrics.get("final_recall", client2_metrics.get("metrics", {}).get("recall", None))) if client2_metrics else "N/A",
            safe_fmt(global_metrics.get("metrics", {}).get("recall", global_metrics.get("final_recall", None))) if global_metrics else "N/A"
        ],
        "F1-Score": [
            safe_fmt(client1_metrics.get("final_f1", client1_metrics.get("metrics", {}).get("f1", None))) if client1_metrics else "N/A",
            safe_fmt(client2_metrics.get("final_f1", client2_metrics.get("metrics", {}).get("f1", None))) if client2_metrics else "N/A",
            safe_fmt(global_metrics.get("metrics", {}).get("f1", global_metrics.get("final_f1", None))) if global_metrics else "N/A"
        ]
    }
    df_metrics = pd.DataFrame(metrics_data)
    st.table(df_metrics)

    # --- CLASSIFICATION REPORTS ---
    st.markdown('<div class="section-title">📋 Classification Reports</div>', unsafe_allow_html=True)
    with st.expander("Show Classification Reports", expanded=True):
        rep_cols = st.columns(3)
        with rep_cols[0]:
            st.markdown("##### 🏦 Bank 1 Report")
            if client1_metrics and ("classification_report" in client1_metrics or ("metrics" in client1_metrics and "classification_report" in client1_metrics["metrics"])):
                report = client1_metrics.get("classification_report", client1_metrics.get("metrics", {}).get("classification_report", {}))
                format_classification_report(report, "Bank 1")
            else:
                st.info("No Bank 1 classification report available yet. Train Bank 1 model to see this report.")
        with rep_cols[1]:
            st.markdown("##### 🏦 Bank 2 Report")
            if client2_metrics and ("classification_report" in client2_metrics or ("metrics" in client2_metrics and "classification_report" in client2_metrics["metrics"])):
                report = client2_metrics.get("classification_report", client2_metrics.get("metrics", {}).get("classification_report", {}))
                format_classification_report(report, "Bank 2")
            else:
                st.info("No Bank 2 classification report available yet. Train Bank 2 model to see this report.")
        with rep_cols[2]:
            st.markdown("##### 🏛️ Crime Branch Report")
            if global_metrics and ("classification_report" in global_metrics or ("metrics" in global_metrics and "classification_report" in global_metrics["metrics"])):
                report = global_metrics.get("classification_report", global_metrics.get("metrics", {}).get("classification_report", {}))
                format_classification_report(report, "Crime Branch")
            else:
                st.info("No Crime Branch classification report available yet. Wait for aggregation.")

    # --- AGGREGATION LOGS ---
    st.markdown('<div class="section-title">🛠️ Aggregation Logs</div>', unsafe_allow_html=True)
    if aggregation_log:
        log_data = []
        for entry in aggregation_log:
            timestamp = entry.get("timestamp", "")
            if isinstance(timestamp, str) and "numberLong" in timestamp:
                try:
                    timestamp_dict = json.loads(timestamp)
                    timestamp_ms = int(timestamp_dict["$date"]["$numberLong"])
                    timestamp = datetime.fromtimestamp(timestamp_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    pass
            log_data.append({
                "Round": entry.get("round", ""),
                "Client Models": ", ".join(entry.get("client_models", [])) if entry.get("client_models") else "",
                "Weights": ", ".join([f"{w:.2f}" for w in entry.get("weights", [])]) if entry.get("weights") else "",
                "Error": entry.get("error", ""),
                "Timestamp": timestamp
            })
        df_log = pd.DataFrame(log_data)
        st.dataframe(df_log, use_container_width=True)
    else:
        st.info("No aggregation logs available yet. Aggregation will appear here after both clients upload their models.")

    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        if hasattr(st, 'rerun'):
            st.rerun()
        elif hasattr(st, 'experimental_rerun'):
            st.experimental_rerun()
        else:
            st.warning("Streamlit rerun is not available in your version. Please upgrade Streamlit.")

if __name__ == "__main__":
    main() 