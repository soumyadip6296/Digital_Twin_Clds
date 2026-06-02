import streamlit as st
import asyncio
import websockets
import json
import plotly.graph_objects as go
import threading
import time
from collections import deque  # FIX: O(1) append/pop vs O(n) list.pop(0)
from streamlit.runtime.scriptrunner import add_script_run_ctx

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="AI Digital Twin")
st.title("🌍 Digital Twin: Live Telemetry")

# --- Session State Initialization ---
# FIX: Use deque(maxlen=50) so trimming the history is O(1) and automatic.
if 'history_error' not in st.session_state:
    st.session_state.history_error = deque(maxlen=50)
if 'history_route' not in st.session_state:
    st.session_state.history_route = deque(maxlen=50)
if 'latest_data' not in st.session_state:
    st.session_state.latest_data = None
if 'ws_running' not in st.session_state:
    st.session_state.ws_running = False
# FIX: Track a render counter to detect new data and avoid a redundant rerun.
if 'render_id' not in st.session_state:
    st.session_state.render_id = 0
if 'last_render_id' not in st.session_state:
    st.session_state.last_render_id = -1


# --- Background Worker ---
async def fetch_telemetry():
    """Listens to the WebSocket and updates session_state in the background."""
    uri = "ws://localhost:8000/ws/dashboard"
    try:
        async with websockets.connect(uri) as websocket:
            while st.session_state.ws_running:
                data = await websocket.recv()
                st.session_state.latest_data = json.loads(data)
                # Increment so the main thread knows there is genuinely new data
                st.session_state.render_id += 1
    except Exception as e:
        st.session_state.latest_data = {"error_msg": str(e)}


def start_background_loop():
    """Sets up a new event loop for the background thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(fetch_telemetry())


# --- UI Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Live Topology Routing")
    graph_container = st.empty()

with col2:
    st.subheader("Observer Anomaly Score")
    metric_alert = st.empty()
    chart_container = st.empty()

# --- Connection Button ---
button_container = st.empty()

if not st.session_state.ws_running:
    if button_container.button("🟢 Connect to Digital Twin"):
        st.session_state.ws_running = True
        thread = threading.Thread(target=start_background_loop, daemon=True)
        add_script_run_ctx(thread)
        thread.start()
        st.rerun()
else:
    if button_container.button("🛑 Stop Monitoring"):
        st.session_state.ws_running = False
        st.session_state.latest_data = None
        st.rerun()

# --- Main Render Loop ---
if st.session_state.ws_running:
    if st.session_state.latest_data is None:
        graph_container.info("📡 WebSocket Connected. Listening for first telemetry packet...")
        chart_container.warning("Awaiting data stream from backend API...")
        time.sleep(1)
        st.rerun()

    else:
        parsed = st.session_state.latest_data

        if "error_msg" in parsed:
            st.error(f"Disconnected from Core API: {parsed['error_msg']}")
            st.session_state.ws_running = False
            st.stop()

        # FIX: Only update history and rerender if genuinely new data has arrived.
        # This prevents the infinite CPU-hammering rerun loop.
        if st.session_state.render_id != st.session_state.last_render_id:
            st.session_state.last_render_id = st.session_state.render_id

            # 1. Update History — deque handles maxlen trimming automatically
            st.session_state.history_error.append(parsed["error"])
            st.session_state.history_route.append(parsed["route"])

            # 2. Status Alert
            if parsed["is_attack"]:
                metric_alert.error("⚠️ CYBER ATTACK DETECTED! Rerouting traffic...")
            else:
                metric_alert.success("✅ Network Stable.")

            # 3. Plotly Network Topology
            fig_topo = go.Figure()

            node_x, node_y = [0, 1, 2], [0.5, 0.5, 0.5]
            node_text = ["Users", "AI Router", "Server"]
            node_colors = ["#3b82f6", "#8b5cf6", "#10b981"]

            fig_topo.add_trace(go.Scatter(
                x=[0, 1], y=[0.5, 0.5], mode='lines',
                line=dict(color='gray', width=4, dash='dot')
            ))

            route_color = "#10b981" if parsed["route"] == 0 else "#ef4444"
            route_name = "Primary Link (Fiber)" if parsed["route"] == 0 else "Backup Link (Sat)"

            fig_topo.add_trace(go.Scatter(
                x=[1, 2], y=[0.5, 0.5], mode='lines+text',
                line=dict(color=route_color, width=8),
                text=[None, route_name], textposition="top center",
                textfont=dict(size=14, color=route_color, family="Arial Black")
            ))

            fig_topo.add_trace(go.Scatter(
                x=node_x, y=node_y, mode='markers+text',
                marker=dict(size=40, color=node_colors, line=dict(width=2, color='white')),
                text=node_text, textposition="bottom center",
                textfont=dict(size=14, color="white")
            ))

            fig_topo.update_layout(
                height=350, showlegend=False,
                xaxis=dict(visible=False, range=[-0.2, 2.2]),
                yaxis=dict(visible=False, range=[0.2, 0.8]),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=10, b=0)
            )

            # 4. Anomaly Line Chart
            fig_chart = go.Figure()

            line_color = '#ef4444' if parsed["is_attack"] else '#10b981'
            fill_color = 'rgba(239, 68, 68, 0.2)' if parsed["is_attack"] else 'rgba(16, 185, 129, 0.2)'

            fig_chart.add_trace(go.Scatter(
                y=list(st.session_state.history_error),  # deque → list for Plotly
                mode='lines',
                fill='tozeroy',
                line=dict(color=line_color, width=2),
                fillcolor=fill_color
            ))

            fig_chart.add_hline(
                y=0.08,
                line_dash="dot",
                line_color="orange",
                annotation_text="⚠️ Critical Threat Threshold",
                annotation_position="top left",
                annotation_font_color="orange"
            )

            fig_chart.update_layout(
                title="Network Threat Analysis",
                xaxis_title="Time (Last 50 Packets)",
                yaxis_title="Anomaly Score",
                height=300,
                margin=dict(l=50, r=20, t=40, b=40),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(range=[0, 0.1])
            )

            # 5. Render
            graph_container.plotly_chart(fig_topo, use_container_width=True)
            chart_container.plotly_chart(fig_chart, use_container_width=True)

        # 6. Throttle — poll for new data every 0.5s
        time.sleep(0.5)
        st.rerun()