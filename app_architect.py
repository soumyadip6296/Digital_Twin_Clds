import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import asyncio
import websockets
import json
import time

# FIX: nest_asyncio allows asyncio.run() to work inside Streamlit's already-running
# event loop. Without this, clicking Start Simulation raises:
#   RuntimeError: This event loop is already running
import nest_asyncio
nest_asyncio.apply()

# --- Page Configuration ---
st.set_page_config(page_title="Network Architect", layout="wide")
st.title("🛠️ SDN Network Architect")

# Initialize session state for simulation control
if 'stop_simulation' not in st.session_state:
    st.session_state.stop_simulation = False

# --- 1. Router Configuration ---
st.header("1. Router Configuration")
col1, col2, col3 = st.columns(3)
with col1:
    primary_bw = st.text_input("Primary Route Bandwidth", "1 Gbps")
    primary_lat = st.text_input("Primary Base Latency", "10ms")
with col2:
    backup_bw = st.text_input("Backup Route Bandwidth", "100 Mbps")
    backup_lat = st.text_input("Backup Base Latency", "50ms")
with col3:
    sim_speed = st.slider("Injection Speed (Packets/Sec)", 1, 50, 2)
    delay_per_packet = 1.0 / sim_speed

# --- 2. Dataset Injection ---
st.header("2. Dataset Injection")
if not os.path.exists("data"):
    os.makedirs("data")
datasets = [f for f in os.listdir("data") if f.endswith('.csv')]
selected_dataset = st.selectbox("Select Traffic Scenario", datasets)

# Feature list for the AI Model
FEATURES = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max',
    'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
    'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Bwd IAT Total', 'Fwd Header Length', 'Bwd Header Length',
    'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length', 'Packet Length Mean', 'Packet Length Std',
    'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
    'URG Flag Count', 'Down/Up Ratio', 'Average Packet Size', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward'
]


def parse_latency(val):
    nums = re.findall(r"\d+", str(val))
    return float(nums[0]) / 1000 if nums else 0.01


# --- 3. Simulation Engine ---
async def start_injection(df, p_lat, b_lat):
    uri = "ws://localhost:8000/ws/network"
    total = len(df)

    progress_bar = st.progress(0)
    status_text = st.empty()
    time_text = st.empty()

    try:
        async with websockets.connect(uri) as websocket:
            for i in range(total):
                if st.session_state.stop_simulation:
                    st.warning("🛑 Simulation manually terminated.")
                    break

                row = df.iloc[i]
                features = pd.to_numeric(row[FEATURES], errors='coerce').fillna(0).tolist()
                vol = float(
                    row.get('Total Length of Fwd Packets', 0) +
                    row.get('Total Length of Bwd Packets', 0)
                )

                label = str(row.get('Label', 'BENIGN')).strip().upper()
                is_attack = "BENIGN" not in label
                lat_a = 0.95 if is_attack else np.random.uniform(p_lat, p_lat + 0.02)

                payload = {"features": features, "volume": vol, "lat_a": lat_a, "lat_b": b_lat}
                await websocket.send(json.dumps(payload))
                await websocket.recv()

                prog = (i + 1) / total
                progress_bar.progress(prog)

                rem_packets = total - (i + 1)
                rem_seconds = rem_packets * delay_per_packet
                hrs, rem = divmod(int(rem_seconds), 3600)
                mins, secs = divmod(rem, 60)

                status_text.text(f"Injecting: {i+1}/{total} | Type: {label}")
                time_text.markdown(f"**⏱️ Time Left:** {hrs}h {mins}m {secs}s")

                await asyncio.sleep(delay_per_packet)

    except Exception as e:
        st.error(f"Core API connection failed: {e}")


# Controls
c1, c2 = st.columns(2)
if c1.button("▶️ Start Simulation", type="primary", use_container_width=True):
    st.session_state.stop_simulation = False
    if selected_dataset:
        df_sim = pd.read_csv(f"data/{selected_dataset}")
        df_sim.columns = [c.strip() for c in df_sim.columns]
        # FIX: nest_asyncio (applied above) makes this safe inside Streamlit's loop.
        asyncio.run(start_injection(df_sim, parse_latency(primary_lat), parse_latency(backup_lat)))
    else:
        st.warning("Please select a dataset first.")

if c2.button("🛑 Stop Injection", use_container_width=True):
    st.session_state.stop_simulation = True