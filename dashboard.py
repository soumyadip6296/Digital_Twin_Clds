# dashboard.py
import streamlit as st
import asyncio
import websockets
import json
import plotly.graph_objects as go
from collections import deque  # FIX: O(1) history trimming
from streamlit_agraph import agraph, Node, Edge, Config

# FIX: nest_asyncio allows asyncio.run() inside Streamlit's running loop.
# Without this, the "Connect to Live Stream" button crashes with RuntimeError.
import nest_asyncio
nest_asyncio.apply()

st.set_page_config(layout="wide", page_title="Digital Twin Command Center")
st.title("🌐 Network Architect & AI Visualizer")

col1, col2 = st.columns([1, 1])

# FIX: Use deque(maxlen=50) so we never need manual pop(0) calls.
if 'history_error' not in st.session_state:
    st.session_state.history_error = deque(maxlen=50)
if 'history_route' not in st.session_state:
    st.session_state.history_route = deque(maxlen=50)

# --- 1. NETWORK TOPOLOGY GRAPH ---
with col1:
    st.subheader("Live SDN Topology")
    nodes = [
        Node(id="Client", label="User Endpoint", size=25, color="#3b82f6"),
        Node(id="AI_Router", label="AI Controller", size=35, color="#8b5cf6", symbolType="diamond"),
        Node(id="Server", label="Data Center", size=25, color="#10b981")
    ]

    current_route = st.session_state.history_route[-1] if st.session_state.history_route else 0
    route_color = "#10b981" if current_route == 0 else "#ef4444"
    route_label = "Primary Link" if current_route == 0 else "Backup Link"

    edges = [
        Edge(source="Client", target="AI_Router", label="Ingress"),
        Edge(source="AI_Router", target="Server", label=route_label, color=route_color, width=4)
    ]

    config = Config(width=500, height=400, directed=True, nodeHighlightBehavior=True, linkHighlightBehavior=True)
    agraph(nodes=nodes, edges=edges, config=config)

# --- 2. LIVE METRICS ---
with col2:
    st.subheader("Telemetry & AI Decisions")
    metric_placeholder = st.empty()
    graph_placeholder = st.empty()


# --- 3. ASYNC WEBSOCKET LISTENER ---
async def listen_to_twin():
    # FIX: Corrected endpoint from /ws/ui (which doesn't exist) to /ws/dashboard.
    uri = "ws://localhost:8000/ws/dashboard"
    try:
        async with websockets.connect(uri) as websocket:
            while True:
                data = await websocket.recv()
                parsed = json.loads(data)

                # Update State — deque trims automatically at maxlen=50
                st.session_state.history_error.append(parsed["error"])
                st.session_state.history_route.append(parsed["route"])

                # Update Metrics
                with metric_placeholder.container():
                    c1, c2 = st.columns(2)
                    c1.metric("Anomaly Score", f"{parsed['error']:.4f}")
                    route_name = "Primary" if parsed["route"] == 0 else "Backup"
                    c2.metric("Active Route", route_name)

                    if parsed.get("is_attack"):
                        st.error("⚠️ CYBER ATTACK DETECTED!")
                    else:
                        st.success("✅ Network Stable.")

                # Update Anomaly Graph
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=list(st.session_state.history_error),  # deque → list for Plotly
                    mode='lines',
                    fill='tozeroy',
                    line=dict(color='#ef4444' if parsed.get("is_attack") else '#10b981')
                ))
                fig.add_hline(
                    y=0.08, line_dash="dot", line_color="orange",
                    annotation_text="⚠️ Threat Threshold", annotation_position="top left"
                )
                fig.update_layout(
                    title="Real-Time Anomaly Detection",
                    height=300,
                    margin=dict(l=0, r=0, t=30, b=0),
                    yaxis=dict(range=[0, 0.1])
                )
                graph_placeholder.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Waiting for Core API... {e}")


if st.button("Connect to Live Stream"):
    # FIX: nest_asyncio (applied at top) makes asyncio.run() safe here.
    asyncio.run(listen_to_twin())