"""
app_brains.py — AI Brains Inspector Dashboard

Shows the internal state of each of the 4 AI brains in real-time:
  Observer  — anomaly score vs threshold, attack detection history
  Prophet   — traffic volume forecast, load trend
  Analyst   — traffic classification, cluster distribution
  Manager   — route decisions, decision context, switch events
"""

import streamlit as st
import asyncio
import websockets
import json
import time
import plotly.graph_objects as go
import urllib.request
import uuid

st.set_page_config(layout="wide", page_title="AI Brains Inspector")

# ── Title ──────────────────────────────────────────────────────────────
st.markdown("## 🧠 AI Brains Inspector")
st.caption("Real-time internal state of each AI brain. Connect to a live simulation to see them think.")

MAX_HISTORY   = 60
DEFAULT_THRESHOLD = 0.05

# ── Session state init ─────────────────────────────────────────────────
_defaults = {
    "obs_h":       [],   # [{score, is_attack, threshold}]
    "proph_h":     [],   # [{forecast}]
    "analyst_h":   [],   # [{traffic_type, cluster_id}]
    "manager_h":   [],   # [{route, lat_a, lat_b}]
    "brains_conn": False,
    "brains_stop": False,
    "brains_auto": False,
    "brains_frame": 0,
    "brains_cid":  uuid.uuid4().hex,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Core API health banner ─────────────────────────────────────────────
def _get_status():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2.0) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None

status = _get_status()
if status:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Device",       status.get("device", "?"))
    c2.metric("Packets In",   status.get("packets_received", 0))
    c3.metric("PPS (10s)",    f"{status.get('pps_10s', 0.0):.2f}")
    c4.metric("Queue",        status.get("queue_depth", 0))
    c5.metric("Obs Threshold", f"{status.get('obs_threshold', DEFAULT_THRESHOLD):.4f}")
else:
    st.warning("⚠️ Core API not reachable — start `core_api.py` first.")

st.divider()

# ── 4-panel layout ─────────────────────────────────────────────────────
row1 = st.columns(2)
row2 = st.columns(2)

with row1[0]:
    st.subheader("👁️ Observer — Anomaly Detector")
    obs_alert   = st.empty()
    obs_chart   = st.empty()
    obs_footer  = st.empty()

with row1[1]:
    st.subheader("🔮 Prophet — Traffic Forecaster")
    proph_alert  = st.empty()
    proph_chart  = st.empty()
    proph_footer = st.empty()

with row2[0]:
    st.subheader("🔬 Analyst — Traffic Classifier")
    analyst_alert  = st.empty()
    analyst_chart  = st.empty()
    analyst_footer = st.empty()

with row2[1]:
    st.subheader("🎛️ Manager — Route Controller")
    manager_alert  = st.empty()
    manager_chart  = st.empty()
    manager_footer = st.empty()


# ── Render functions ───────────────────────────────────────────────────
def _plot_defaults():
    fig = go.Figure()
    fig.update_layout(
        height=200,
        margin=dict(l=0, r=0, t=8, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_observer(fk: int):
    h = st.session_state.obs_h
    if not h:
        obs_alert.info("Waiting for Observer data...")
        obs_chart.plotly_chart(_plot_defaults(), use_container_width=True, key=f"obs_{fk}")
        return

    latest     = h[-1]
    score      = latest["score"]
    threshold  = latest.get("threshold", DEFAULT_THRESHOLD)
    is_attack  = latest["is_attack"]

    if is_attack:
        obs_alert.error(f"🚨 ANOMALY — Score: **{score:.4f}**   Threshold: {threshold:.4f}")
    else:
        obs_alert.success(f"✅ Normal — Score: **{score:.4f}**   Threshold: {threshold:.4f}")

    scores = [p["score"] for p in h]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=scores, mode="lines", fill="tozeroy",
        line=dict(color="#ef4444", width=2), name="Anomaly Score",
    ))
    fig.add_hline(
        y=threshold,
        line=dict(color="#facc15", dash="dash", width=1.5),
        annotation_text="Threshold",
        annotation_position="top right",
        annotation_font=dict(color="#facc15"),
    )
    fig.update_layout(
        height=200, margin=dict(l=0, r=0, t=8, b=0),
        yaxis_title="Score", xaxis_title=f"Last {len(scores)} packets",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    obs_chart.plotly_chart(fig, use_container_width=True, key=f"obs_{fk}")

    attacks = sum(1 for p in h if p["is_attack"])
    pct     = 100 * attacks / len(h)
    obs_footer.caption(
        f"Window: **{len(h)}** packets — Anomalies: **{attacks}** ({pct:.1f}%)"
    )


def render_prophet(fk: int):
    h = st.session_state.proph_h
    if not h:
        proph_alert.info("Waiting for Prophet data...")
        proph_chart.plotly_chart(_plot_defaults(), use_container_width=True, key=f"proph_{fk}")
        return

    fcast = h[-1]["forecast"]
    trend = ("📈 Rising"  if len(h) > 1 and h[-1]["forecast"] > h[-2]["forecast"]
             else "📉 Falling")
    if fcast > 0.6:
        intensity, col = "🔴 High Load",   "#ef4444"
    elif fcast > 0.3:
        intensity, col = "🟡 Medium Load", "#facc15"
    else:
        intensity, col = "🟢 Low Load",    "#22c55e"

    proph_alert.markdown(
        f"**Predicted Load:** "
        f"<span style='color:{col}; font-size:1.5em; font-weight:bold'>{fcast:.3f}</span>"
        f"  {intensity}  {trend}",
        unsafe_allow_html=True,
    )

    forecasts = [p["forecast"] for p in h]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=forecasts, mode="lines+markers",
        line=dict(color="#a78bfa", width=2),
        marker=dict(size=3, color="#a78bfa"),
    ))
    fig.add_hline(y=0.6, line=dict(color="#ef4444", dash="dot", width=1),
                  annotation_text="High", annotation_font=dict(color="#ef4444"))
    fig.add_hline(y=0.3, line=dict(color="#facc15", dash="dot", width=1),
                  annotation_text="Med",  annotation_font=dict(color="#facc15"))
    fig.update_layout(
        height=200, margin=dict(l=0, r=0, t=8, b=0),
        yaxis=dict(range=[0, 1], title="Predicted Load"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    proph_chart.plotly_chart(fig, use_container_width=True, key=f"proph_{fk}")

    avg = sum(p["forecast"] for p in h) / len(h)
    proph_footer.caption(
        f"Window avg: **{avg:.3f}**  |  Peak: **{max(forecasts):.3f}**  |  Min: **{min(forecasts):.3f}**"
    )


def render_analyst(fk: int):
    h = st.session_state.analyst_h
    if not h:
        analyst_alert.info("Waiting for Analyst data...")
        analyst_chart.plotly_chart(_plot_defaults(), use_container_width=True, key=f"analyst_{fk}")
        return

    ttype = h[-1]["traffic_type"]
    cid   = h[-1].get("cluster_id", "?")
    if ttype == 1.0:
        label, col = "🎬 Video / Heavy", "#f97316"
    else:
        label, col = "📦 Normal / Background", "#22c55e"

    analyst_alert.markdown(
        f"**Current Class:** "
        f"<span style='color:{col}; font-weight:bold'>{label}</span>"
        f"  &nbsp; Cluster ID: `{cid}`",
        unsafe_allow_html=True,
    )

    # Rolling 10-packet heavy-traffic ratio
    types  = [p["traffic_type"] for p in h]
    window = 10
    ratios = []
    for i in range(len(types)):
        chunk = types[max(0, i - window): i + 1]
        ratios.append(sum(chunk) / len(chunk))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=ratios, mode="lines", fill="tozeroy",
        line=dict(color="#f97316", width=2), name="Heavy %",
    ))
    fig.update_layout(
        height=200, margin=dict(l=0, r=0, t=8, b=0),
        yaxis=dict(range=[0, 1], title=f"Heavy Ratio (rolling {window})"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    analyst_chart.plotly_chart(fig, use_container_width=True, key=f"analyst_{fk}")

    normal = sum(1 for p in h if p["traffic_type"] == 0.0)
    heavy  = len(h) - normal
    analyst_footer.caption(
        f"Window: **{len(h)}** packets — "
        f"Normal: **{normal}** ({100*normal/len(h):.0f}%)  "
        f"Heavy: **{heavy}** ({100*heavy/len(h):.0f}%)"
    )


def render_manager(fk: int):
    mh = st.session_state.manager_h
    oh = st.session_state.obs_h
    ph = st.session_state.proph_h
    ah = st.session_state.analyst_h

    if not mh:
        manager_alert.info("Waiting for Manager data...")
        manager_chart.plotly_chart(_plot_defaults(), use_container_width=True, key=f"mgr_{fk}")
        return

    route = mh[-1]["route"]
    lat_a = mh[-1].get("lat_a", 0.0)
    lat_b = mh[-1].get("lat_b", 0.05)
    active_lat = lat_b if route == 1 else lat_a

    if route == 0:
        route_label = "🟢 Primary (Fiber)"
    else:
        route_label = "🔴 Backup (Satellite)"

    manager_alert.markdown(
        f"**Active Route:** {route_label}  "
        f"&nbsp; Experienced Latency: **{active_lat:.3f}**",
        unsafe_allow_html=True,
    )

    routes = [p["route"] for p in mh]
    switches = sum(1 for i in range(1, len(routes)) if routes[i] != routes[i - 1])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=routes,
        mode="lines+markers",
        line=dict(color="#3b82f6", width=2, shape="hv"),
        marker=dict(
            size=6,
            color=["#ef4444" if r == 1 else "#22c55e" for r in routes],
            line=dict(width=1, color="white"),
        ),
        name="Route",
    ))
    fig.update_layout(
        height=200, margin=dict(l=0, r=0, t=8, b=0),
        yaxis=dict(tickvals=[0, 1], ticktext=["Primary", "Backup"], title="Route"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    manager_chart.plotly_chart(fig, use_container_width=True, key=f"mgr_{fk}")

    # Decision context row
    ctx_parts = []
    if oh:
        a_score = oh[-1]["score"]
        thr     = oh[-1].get("threshold", DEFAULT_THRESHOLD)
        ctx_parts.append(f"Observer: `{a_score:.4f}` {'⚠️' if a_score > thr else '✅'}")
    if ph:
        ctx_parts.append(f"Prophet: `{ph[-1]['forecast']:.3f}`")
    if ah:
        ctx_parts.append(f"Analyst: `{'Heavy' if ah[-1]['traffic_type'] else 'Normal'}`")
    ctx_parts.append(f"lat_A: `{lat_a:.3f}`  lat_B: `{lat_b:.3f}`")

    manager_footer.caption(
        f"Route switches: **{switches}**  |  Inputs → " + "  |  ".join(ctx_parts)
    )


def render_all(fk: int):
    render_observer(fk)
    render_prophet(fk)
    render_analyst(fk)
    render_manager(fk)


# ── WebSocket listener ─────────────────────────────────────────────────
async def listen_brains():
    uri = f"ws://127.0.0.1:8000/ws/dashboard?client_id={st.session_state.brains_cid}"
    try:
        async with websockets.connect(uri) as ws:
            while True:
                if st.session_state.get("brains_stop"):
                    break
                raw  = await ws.recv()
                data = json.loads(raw)

                st.session_state.obs_h.append({
                    "score":     data.get("error", 0.0),
                    "is_attack": data.get("is_attack", False),
                    "threshold": data.get("obs_threshold", DEFAULT_THRESHOLD),
                })
                st.session_state.proph_h.append({
                    "forecast": data.get("forecast", 0.0),
                })
                st.session_state.analyst_h.append({
                    "traffic_type": data.get("traffic_type", 0.0),
                    "cluster_id":   data.get("cluster_id", "?"),
                })
                st.session_state.manager_h.append({
                    "route": data.get("route", 0),
                    "lat_a": data.get("lat_a", 0.0),
                    "lat_b": data.get("lat_b", 0.05),
                })

                for key in ["obs_h", "proph_h", "analyst_h", "manager_h"]:
                    while len(st.session_state[key]) > MAX_HISTORY:
                        st.session_state[key].pop(0)

                st.session_state.brains_frame += 1
                render_all(st.session_state.brains_frame)

    except Exception as e:
        st.error(f"Connection lost: {e}")
    finally:
        st.session_state.brains_conn = False
        st.session_state.brains_stop = False
        render_all(st.session_state.brains_frame)


# ── Connect / Disconnect buttons ──────────────────────────────────────
st.divider()
bc, bd = st.columns(2)
with bc:
    if st.button("🔌 Connect Brains", disabled=st.session_state.brains_conn):
        st.session_state.brains_conn = True
        asyncio.run(listen_brains())
with bd:
    if st.button("⏏️ Disconnect", disabled=not st.session_state.brains_conn):
        st.session_state.brains_stop = True

if not st.session_state.brains_auto:
    st.session_state.brains_auto = True
    st.session_state.brains_conn = True
    asyncio.run(listen_brains())
elif not status:
    render_all(st.session_state.brains_frame)
