import asyncio
import websockets
import json
import pandas as pd
import numpy as np
import os
import glob
import re

print("🚀 Booting up Network Bridge...")


def parse_latency(val, default_ms=10):
    try:
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(nums[0]) / 1000 if nums else (default_ms / 1000)
    except Exception:
        return default_ms / 1000


primary_latency_base = parse_latency(os.getenv("PRIMARY_LAT", "10"), default_ms=10)
backup_latency_base = parse_latency(os.getenv("BACKUP_LAT", "50"), default_ms=50)
selected_dataset = os.getenv("SELECTED_DATASET", "")
# FIX: Respect the injection-speed slider set in the Architect UI.
# The Architect writes SIM_SPEED (packets/sec) to the environment before
# launching this bridge. Default = 2 pkt/s (matches old hard-coded 0.5 s).
_sim_speed   = max(1, int(os.getenv("SIM_SPEED", "2")))
_delay_per_packet = 1.0 / _sim_speed

FEATURES = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max',
    'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
    'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Bwd IAT Total', 'Fwd Header Length', 'Bwd Header Length',
    'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length', 'Packet Length Mean', 'Packet Length Std',
    'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
    'URG Flag Count', 'Down/Up Ratio', 'Average Packet Size', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward'
]


async def inject_traffic():
    uri = "ws://localhost:8000/ws/network"
    target_csv = (
        f"data/{selected_dataset}" if selected_dataset
        else (glob.glob("data/*.csv")[0] if glob.glob("data/*.csv") else None)
    )

    if not target_csv or not os.path.exists(target_csv):
        print(f"❌ Dataset not found: {target_csv}")
        return

    print(f"📡 Loading: {target_csv}")
    df = pd.read_csv(target_csv)
    df.columns = [c.strip() for c in df.columns]
    df = df.replace(['Infinity', 'inf', 'NaN'], np.nan).fillna(0)
    print("✅ Headers cleaned. Connecting...")

    async with websockets.connect(uri) as websocket:
        for i in range(len(df)):
            row = df.iloc[i]
            features = pd.to_numeric(row[FEATURES], errors='coerce').fillna(0).astype(float).values.tolist()
            vol = float(row.get('Total Length of Fwd Packets', 0) + row.get('Total Length of Bwd Packets', 0))

            raw_label = str(row.get('Label', 'BENIGN')).strip().upper()
            is_attack = raw_label != 'BENIGN'

            lat_a = 0.95 if is_attack else np.random.uniform(primary_latency_base, primary_latency_base + 0.02)
            lat_b = backup_latency_base

            if is_attack:
                print(f"🔥 row {i}: ATTACK ({raw_label}) -> lat_a: {lat_a}")

            payload = {"features": features, "volume": vol, "lat_a": lat_a, "lat_b": lat_b}
            await websocket.send(json.dumps(payload))
            await websocket.recv()
            await asyncio.sleep(_delay_per_packet)


if __name__ == "__main__":
    asyncio.run(inject_traffic())