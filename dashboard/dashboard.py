import streamlit as st
import requests
import time
import pandas as pd
import re
import io

st.set_page_config(page_title="Federated Learning Dashboard", layout="wide")

# Color palettes
PALETTES = {
    "Classic": {"bg": "#f5f7fa", "primary": "#1976d2", "secondary": "#43a047", "accent": "#fbc02d"},
    "Ocean": {"bg": "#e0f7fa", "primary": "#0288d1", "secondary": "#26c6da", "accent": "#00bfae"},
    "Sunset": {"bg": "#fff3e0", "primary": "#ff7043", "secondary": "#ffa726", "accent": "#ffd600"},
    "Forest": {"bg": "#e8f5e9", "primary": "#388e3c", "secondary": "#8bc34a", "accent": "#cddc39"},
    "Dark": {"bg": "#222", "primary": "#90caf9", "secondary": "#a5d6a7", "accent": "#fff59d"}
}
palette_name = st.selectbox("🌈 Select Color Palette", list(PALETTES.keys()), index=0)
palette = PALETTES[palette_name]

# Apply palette
st.markdown(f"""
    <style>
    .stApp {{ background-color: {palette['bg']} !important; }}
    .css-1v0mbdj, .css-1d391kg, .css-1cpxqw2 {{ color: {palette['primary']} !important; }}
    .css-1cpxqw2 {{ background-color: {palette['secondary']} !important; }}
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Federated Learning Server Dashboard")

SERVER_URL = st.text_input("🌐 Server URL", "http://localhost:5000")

log_placeholder = st.empty()
refresh_rate = st.slider("🔄 Refresh rate (seconds)", 1, 30, 5)

st.markdown("---")
status_placeholder = st.empty()

# Fetch available clients for selection
try:
    metrics_resp = requests.get(f"{SERVER_URL}/metrics")
    if metrics_resp.status_code == 200:
        all_metrics = metrics_resp.json()
        all_clients = list(all_metrics.keys())
    else:
        all_clients = []
except Exception:
    all_clients = []

selected_clients = st.multiselect("👤 Select Clients to Display", all_clients, default=all_clients[:2])

# Main metrics area
client_cols = st.columns(2)
agg_placeholder = st.empty()

metrics_placeholder = st.empty()

# Custom prediction form
st.markdown("---")
st.header("🔍 Custom Fraud Prediction (Server Model)")
with st.form("predict_form"):
    amount = st.number_input("💰 Amount", min_value=0.0, value=1000.0)
    type_option = st.selectbox("📝 Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"])
    oldbalanceOrg = st.number_input("🏦 Old Balance Origin", min_value=0.0, value=0.0)
    newbalanceOrig = st.number_input("🏦 New Balance Origin", min_value=0.0, value=0.0)
    oldbalanceDest = st.number_input("🏦 Old Balance Dest", min_value=0.0, value=0.0)
    newbalanceDest = st.number_input("🏦 New Balance Dest", min_value=0.0, value=0.0)
    submit = st.form_submit_button("🚦 Predict Fraud")
if submit:
    predict_payload = {
        "amount": amount,
        "type": type_option,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }
    try:
        response = requests.post(f"{SERVER_URL}/predict", json=predict_payload)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Fraud Probability: {result['fraud_probability']:.4f}")
            st.info(f"Fraud Prediction: {'🛑 FRAUD' if result['fraud_prediction'] else '✅ NOT FRAUD'}")
        else:
            st.error(f"Prediction failed: {response.json().get('error', 'Unknown error')}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

st.markdown("---")
st.header("📜 Server Logs & Client Updates")

# For advanced analytics
summary_stats = {}
latest_metrics = []

while True:
    # Status
    try:
        response = requests.get(f"{SERVER_URL}/status")
        if response.status_code == 200:
            status = response.json()
            status_placeholder.info(f"🕒 **Current Round:** `{status['current_round']}` | 🏁 **Last Aggregated Round:** `{status['last_aggregated_round']}` | 👥 **Clients Waiting:** `{status['clients_waiting']}`")
        else:
            status_placeholder.error("Failed to fetch status from server.")
    except Exception as e:
        status_placeholder.error(f"Error: {e}")
    # Metrics
    try:
        response = requests.get(f"{SERVER_URL}/metrics")
        if response.status_code == 200:
            metrics = response.json()
            if metrics:
                st.subheader("📊 Client Training Metrics (Per Epoch)")
                for idx, client_id in enumerate(selected_clients[:2]):
                    if client_id in metrics:
                        with client_cols[idx]:
                            df = pd.DataFrame(metrics[client_id])
                            st.markdown(f"### 🖥️ Client: `{client_id}`")
                            st.line_chart(df.set_index('epoch')[['val_accuracy', 'loss']])
                            st.line_chart(df.set_index('epoch')[['precision', 'recall', 'f1']])
                            st.success(f"Latest Accuracy: {df['val_accuracy'].iloc[-1]:.4f}")
                            st.info(f"Latest F1: {df['f1'].iloc[-1]:.4f}")
                            # Advanced analytics: summary stats
                            stats = df[['val_accuracy', 'loss', 'precision', 'recall', 'f1']].describe().T
                            summary_stats[client_id] = stats
                            latest_metrics.append({
                                'Client': client_id,
                                'Accuracy': df['val_accuracy'].iloc[-1],
                                'F1': df['f1'].iloc[-1]
                            })
                # Show summary stats for all selected clients
                if summary_stats:
                    st.markdown("---")
                    st.subheader("📈 Summary Statistics (Selected Clients)")
                    for client_id, stats in summary_stats.items():
                        st.markdown(f"#### `{client_id}`")
                        st.dataframe(stats)
                # Comparison bar chart for latest metrics
                if latest_metrics:
                    st.markdown("---")
                    st.subheader("🏆 Latest Metrics Comparison")
                    comp_df = pd.DataFrame(latest_metrics).set_index('Client')
                    st.bar_chart(comp_df[['Accuracy', 'F1']])
            else:
                metrics_placeholder.info("No metrics reported yet.")
        else:
            metrics_placeholder.error("Failed to fetch metrics from server.")
    except Exception as e:
        metrics_placeholder.error(f"Error: {e}")
    # Aggregation animation block
    try:
        response = requests.get(f"{SERVER_URL}/status")
        if response.status_code == 200:
            status = response.json()
            if status['clients_waiting'] == 0 and status['last_aggregated_round'] > 0:
                with agg_placeholder:
                    st.markdown(f"""
                    <div style='background-color:{palette['accent']};padding:20px;border-radius:10px;text-align:center;'>
                        <h2>🔄 Aggregation in Progress!</h2>
                        <p style='color:{palette['primary']};font-size:20px;'>Averaging client models for round <b>{status['last_aggregated_round']+1}</b>...</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.spinner('Averaging...')
                    st.balloons()
            else:
                agg_placeholder.empty()
    except Exception:
        agg_placeholder.empty()
    # Aggregation history (parse from logs)
    try:
        response = requests.get(f"{SERVER_URL}/logs")
        if response.status_code == 200:
            logs = response.json().get('logs', '')
            log_placeholder.code(logs, language='text')
            # Parse aggregation events
            agg_events = []
            for line in logs.splitlines():
                m = re.search(r'Aggregated weights from clients: (.*?) \(Round (\d+)\)', line)
                if m:
                    clients = m.group(1)
                    round_num = int(m.group(2))
                    time_match = re.match(r'^(.*?) - ', line)
                    agg_time = time_match.group(1) if time_match else ''
                    agg_events.append({"Round": round_num, "Clients": clients, "Time": agg_time})
            if agg_events:
                st.markdown("---")
                st.subheader("🧾 Aggregation History")
                agg_df = pd.DataFrame(agg_events).sort_values("Round", ascending=False)
                st.dataframe(agg_df)
                # Export as CSV
                csv = agg_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Export Aggregation History as CSV",
                    data=csv,
                    file_name='aggregation_history.csv',
                    mime='text/csv',
                    key=f'agg_history_csv_{time.time()}'
                )
        else:
            log_placeholder.error("Failed to fetch logs from server.")
    except Exception as e:
        log_placeholder.error(f"Error: {e}")
    time.sleep(refresh_rate) 