# ns3_bridge.py
import asyncio
import websockets
import json
import pandas as pd
import numpy as np
import glob
import os

FEATURES = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max',
    'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
    'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Bwd IAT Total', 'Fwd Header Length', 'Bwd Header Length',
    'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length', 'Packet Length Mean', 'Packet Length Std',
    'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
    'URG Flag Count', 'Down/Up Ratio', 'Average Packet Size', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward'
]


async def simulate_network():
    # FIX: Corrected endpoint from /ws/ns3 (which doesn't exist in core_api.py)
    # to /ws/network — the only packet-ingestion endpoint the API exposes.
    uri = "ws://localhost:8000/ws/network"

    csv_files = glob.glob("data/*.csv")
    if not csv_files:
        print("❌ No CSV files found in data/ folder! Please add a dataset.")
        return

    df = pd.read_csv(csv_files[0])
    # FIX: Strip whitespace from column headers (matches network_bridge.py behaviour)
    df.columns = df.columns.str.strip()
    df = df.replace(['Infinity', 'inf', 'NaN'], np.nan).fillna(0)
    print(f"📡 NS-3 Bridge Started. Injecting dataset: {csv_files[0]}")

    async with websockets.connect(uri) as websocket:
        for i in range(len(df)):
            row = df.iloc[i]

            # FIX: Convert to numeric and sanitize (handles "Infinity" strings not
            # caught by fillna, and prevents json.dumps() from failing on np.nan/inf).
            features = pd.to_numeric(row[FEATURES], errors='coerce').fillna(0).astype(float).values.tolist()

            vol = float(
                row.get('Total Length of Fwd Packets', 0) +
                row.get('Total Length of Bwd Packets', 0)
            )

            # FIX: Strip whitespace and normalise case before comparing label,
            # otherwise " BENIGN" (with a leading space) is treated as an attack.
            raw_label = str(row.get('Label', 'BENIGN')).strip().upper()
            is_attack = raw_label != 'BENIGN'

            lat_a = 0.95 if is_attack else np.random.uniform(0.01, 0.05)
            lat_b = 0.05  # Backup is stable

            payload = {
                "features": features,
                "volume": vol,
                "lat_a": lat_a,
                "lat_b": lat_b,
            }

            await websocket.send(json.dumps(payload))
            response = await websocket.recv()
            decision = json.loads(response)

            route_str = "Primary (Fiber)" if decision["route"] == 0 else "Backup (Sat)"
            print(f"Packet {i} | Vol: {vol:.0f} | Attack: {is_attack} | Route: {route_str}")

            # FIX: MUST be await asyncio.sleep(), not time.sleep().
            # time.sleep() is a blocking call — it freezes the entire async event loop,
            # preventing any WebSocket sends/receives while it's sleeping.
            await asyncio.sleep(0.1)


if __name__ == "__main__":
    asyncio.run(simulate_network())