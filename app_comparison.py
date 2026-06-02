"""
app_comparison.py — Head-to-Head: AI Digital Twin Router vs Standard Router

Shows both routers processing the same live packet stream simultaneously.
The standard router runs client-side using the lat_a/lat_b values broadcast
by core_api.py — no extra backend changes needed.

Mode selector:  OSPF  |  Static  |  Random
"""

import streamlit as st
import asyncio
import websockets
import json
import time
import plotly.graph_objects as go
import urllib.request
import uuid
import sys
import os

# Import standard_router from the project root
sys.path.insert(0, os.path.dirname(__file__))
from standard_router import StandardRouter

st.set_page_config(layout="wide", page_title="AI vs Standard Router")

# ── Title ──────────────────────────────────────────────────────────────
st.markdown("## ⚔️ Head-to-Head: AI Router vs Standard Router")
st.caption(
    "Both routers see the same packet stream. "
    "The AI Router uses Observer + Prophet + Analyst + Manager (proactive). "
    "The Standard Router uses only link latency (reactive)."
)

MAX_H = 60  # history window

# ── Mode selector ──────────────────────────────────────────────────────
mode_col, _, spacer = st.columns([2, 2, 4])
with mode_col:
    new_mode = st.radio(
        "Standard Router Mode",
        options=["ospf", "static", "random"],
        format_func=lambda m: {
            "ospf":   "OSPF (reactive failover)",
            "static": "Static (always primary)",
            "random": "Random (coin-flip)",
        }[m],
        horizontal=True,
        key="sr_mode_radio",
    )

# ── Session state ──────────────────────────────────────────────────────
_defaults = {
    "cmp_ai_routes":     [],
    "cmp_sr_routes":     [],
    "cmp_ai_latency":    [],
    "cmp_sr_latency":    [],
    "cmp_ai_attacks_caught": 0,
    "cmp_sr_attacks_caught": 0,
    "cmp_packets":       0,
    "cmp_sr_mode":       "ospf",
    "cmp_conn":          False,
    "cmp_stop":          False,
    "cmp_auto":          False,
    "cmp_frame":         0,
    "cmp_cid":           uuid.uuid4().hex,
    "cmp_sr":            StandardRouter(mode="ospf"),
    "cmp_sr_reason":     "",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Reset standard router when mode changes
if new_mode != st.session_state.cmp_sr_mode:
    st.session_state.cmp_sr_mode = new_mode
    st.session_state.cmp_sr.set_mode(new_mode)
    st.session_state.cmp_ai_routes     = []
    st.session_state.cmp_sr_routes     = []
    st.session_state.cmp_ai_latency    = []
    st.session_state.cmp_sr_latency    = []
    st.session_state.cmp_ai_attacks_caught = 0
    st.session_state.cmp_sr_attacks_caught = 0
    st.session_state.cmp_packets       = 0

st.divider()

# ── Core API health ─────────────────────────────────────────────────────
def _status():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2.0) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None

status = _status()
if status:
    st.caption(
        f"Core API online — Device: `{status.get('device')}` | "
        f"Packets: {status.get('packets_received', 0)} | "
        f"PPS: {status.get('pps_10s', 0.0):.2f}"
    )
else:
    st.warning("⚠️ Core API not reachable — start `core_api.py` first.")

# ── Score board ─────────────────────────────────────────────────────────
sb_cols = st.columns(6)
score_ai_lat    = sb_cols[0].empty()
score_sr_lat    = sb_cols[1].empty()
score_ai_att    = sb_cols[2].empty()
score_sr_att    = sb_cols[3].empty()
score_ai_sw     = sb_cols[4].empty()
score_sr_sw     = sb_cols[5].empty()

st.divider()

# ── Side-by-side live panels ────────────────────────────────────────────
ai_col, sr_col = st.columns(2)

with ai_col:
    st.subheader("🤖 AI Digital Twin Router")
    ai_route_badge  = st.empty()
    ai_route_chart  = st.empty()
    ai_lat_chart    = st.empty()
    ai_decision_box = st.empty()

with sr_col:
    st.subheader(f"🖥️ Standard Router ({new_mode.upper()})")
    sr_route_badge  = st.empty()
    sr_route_chart  = st.empty()
    sr_lat_chart    = st.empty()
    sr_decision_box = st.empty()

st.divider()
st.subheader("📊 Comparative Analysis")
delta_chart_placeholder = st.empty()


# ── Helper: route badge ─────────────────────────────────────────────────
def _route_md(route: int, latency: float) -> str:
    if route == 0:
        return (
            f"<div style='background:#14532d; border-radius:8px; padding:10px; text-align:center'>"
            f"<span style='font-size:1.8em'>🟢</span><br>"
            f"<b style='font-size:1.1em'>Primary (Fiber)</b><br>"
            f"<span style='color:#86efac'>lat={latency:.3f}</span></div>"
        )
    else:
        return (
            f"<div style='background:#7f1d1d; border-radius:8px; padding:10px; text-align:center'>"
            f"<span style='font-size:1.8em'>🔴</span><br>"
            f"<b style='font-size:1.1em'>Backup (Satellite)</b><br>"
            f"<span style='color:#fca5a5'>lat={latency:.3f}</span></div>"
        )


def _route_chart(routes, fk, key_prefix):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=routes, mode="lines+markers",
        line=dict(color="#3b82f6", width=2, shape="hv"),
        marker=dict(
            size=6,
            color=["#ef4444" if r == 1 else "#22c55e" for r in routes],
        ),
    ))
    fig.update_layout(
        height=130, margin=dict(l=0, r=0, t=4, b=0),
        yaxis=dict(tickvals=[0, 1], ticktext=["Primary", "Backup"]),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def _lat_chart(lats, color, fk, key_prefix):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=lats, mode="lines", fill="tozeroy",
        line=dict(color=color, width=2),
    ))
    fig.update_layout(
        height=100, margin=dict(l=0, r=0, t=4, b=0),
        yaxis_title="lat",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def render_comparison(fk: int):
    ai_r  = st.session_state.cmp_ai_routes
    sr_r  = st.session_state.cmp_sr_routes
    ai_l  = st.session_state.cmp_ai_latency
    sr_l  = st.session_state.cmp_sr_latency
    sr    = st.session_state.cmp_sr
    n     = st.session_state.cmp_packets

    if not ai_r:
        return

    # ── Score board update ──────────────────────────────────────────
    ai_avg = sum(ai_l) / len(ai_l) if ai_l else 0
    sr_avg = sum(sr_l) / len(sr_l) if sr_l else 0

    delta_lat = sr_avg - ai_avg
    lat_winner = "🤖 AI wins" if delta_lat > 0 else ("🖥️ SR wins" if delta_lat < 0 else "Tied")

    score_ai_lat.metric("AI Avg Latency",  f"{ai_avg:.4f}", delta=f"{-delta_lat:+.4f}")
    score_sr_lat.metric("SR Avg Latency",  f"{sr_avg:.4f}")
    score_ai_att.metric("AI Correct Routes", st.session_state.cmp_ai_attacks_caught)
    score_sr_att.metric("SR Correct Routes", st.session_state.cmp_sr_attacks_caught)
    ai_sw = sum(1 for i in range(1, len(ai_r)) if ai_r[i] != ai_r[i - 1])
    sr_sw = sum(1 for i in range(1, len(sr_r)) if sr_r[i] != sr_r[i - 1])
    score_ai_sw.metric("AI Route Switches", ai_sw)
    score_sr_sw.metric("SR Route Switches", sr_sw)

    # ── AI panel ────────────────────────────────────────────────────
    ai_lat_now = ai_l[-1] if ai_l else 0
    ai_route_badge.markdown(_route_md(ai_r[-1], ai_lat_now), unsafe_allow_html=True)
    ai_route_chart.plotly_chart(
        _route_chart(ai_r, fk, "ai_r"),
        use_container_width=True, key=f"ai_r_{fk}",
    )
    ai_lat_chart.plotly_chart(
        _lat_chart(ai_l, "#22c55e", fk, "ai_l"),
        use_container_width=True, key=f"ai_l_{fk}",
    )
    ai_decision_box.caption(
        f"Proactive — Observer+Prophet+Analyst drove decision  |  "
        f"Avg lat: **{ai_avg:.4f}**  |  Switches: **{ai_sw}**"
    )

    # ── Standard Router panel ────────────────────────────────────────
    sr_lat_now = sr_l[-1] if sr_l else 0
    sr_route_badge.markdown(_route_md(sr_r[-1], sr_lat_now), unsafe_allow_html=True)
    sr_route_chart.plotly_chart(
        _route_chart(sr_r, fk, "sr_r"),
        use_container_width=True, key=f"sr_r_{fk}",
    )
    sr_lat_chart.plotly_chart(
        _lat_chart(sr_l, "#f97316", fk, "sr_l"),
        use_container_width=True, key=f"sr_l_{fk}",
    )
    sr_decision_box.caption(
        f"Reactive ({st.session_state.cmp_sr_mode.upper()}) — {st.session_state.cmp_sr_reason[:80]}  |  "
        f"Avg lat: **{sr_avg:.4f}**  |  Switches: **{sr_sw}**"
    )

    # ── Delta chart ─────────────────────────────────────────────────
    if len(ai_l) >= 2 and len(sr_l) >= 2:
        deltas = [s - a for a, s in zip(ai_l, sr_l)]
        colors = ["#22c55e" if d > 0 else "#ef4444" for d in deltas]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=deltas,
            marker_color=colors,
            name="SR lat − AI lat",
        ))
        fig.add_hline(y=0, line=dict(color="white", width=1))
        fig.update_layout(
            title="Latency Advantage: SR − AI  (green = AI wins, red = AI loses)",
            height=200,
            margin=dict(l=0, r=0, t=30, b=0),
            yaxis_title="Δ latency",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        delta_chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"delta_{fk}")


# ── WebSocket listener ─────────────────────────────────────────────────
async def listen_comparison():
    uri = f"ws://127.0.0.1:8000/ws/dashboard?client_id={st.session_state.cmp_cid}"
    try:
        async with websockets.connect(uri) as ws:
            while True:
                if st.session_state.get("cmp_stop"):
                    break

                raw  = await ws.recv()
                data = json.loads(raw)

                lat_a     = data.get("lat_a", 0.0)
                lat_b     = data.get("lat_b", 0.05)
                is_attack = data.get("is_attack", False)

                # AI router decision (from backend)
                ai_route    = data.get("route", 0)
                ai_exp_lat  = lat_b if ai_route == 1 else lat_a

                # AI "correct" = on Backup during attack OR on Primary during normal
                if is_attack and ai_route == 1:
                    st.session_state.cmp_ai_attacks_caught += 1
                elif not is_attack and ai_route == 0:
                    st.session_state.cmp_ai_attacks_caught += 1

                # Standard router decision (local simulation)
                sr_result = st.session_state.cmp_sr.decide(lat_a, lat_b)
                sr_route   = sr_result["route"]
                sr_exp_lat = sr_result["experienced_latency"]
                st.session_state.cmp_sr_reason = sr_result["reason"]

                if is_attack and sr_route == 1:
                    st.session_state.cmp_sr_attacks_caught += 1
                elif not is_attack and sr_route == 0:
                    st.session_state.cmp_sr_attacks_caught += 1

                # Append histories
                st.session_state.cmp_ai_routes.append(ai_route)
                st.session_state.cmp_sr_routes.append(sr_route)
                st.session_state.cmp_ai_latency.append(ai_exp_lat)
                st.session_state.cmp_sr_latency.append(sr_exp_lat)
                st.session_state.cmp_packets += 1

                for key in ["cmp_ai_routes", "cmp_sr_routes",
                            "cmp_ai_latency", "cmp_sr_latency"]:
                    while len(st.session_state[key]) > MAX_H:
                        st.session_state[key].pop(0)

                st.session_state.cmp_frame += 1
                render_comparison(st.session_state.cmp_frame)

    except Exception as e:
        st.error(f"Connection lost: {e}")
    finally:
        st.session_state.cmp_conn = False
        st.session_state.cmp_stop = False


# ── Connect / Disconnect buttons ──────────────────────────────────────
cc, cd = st.columns(2)
with cc:
    if st.button("🔌 Connect", disabled=st.session_state.cmp_conn):
        st.session_state.cmp_conn = True
        asyncio.run(listen_comparison())
with cd:
    if st.button("⏏️ Disconnect", disabled=not st.session_state.cmp_conn):
        st.session_state.cmp_stop = True

if not st.session_state.cmp_auto:
    st.session_state.cmp_auto = True
    st.session_state.cmp_conn = True
    asyncio.run(listen_comparison())
elif not status:
    render_comparison(st.session_state.cmp_frame)
