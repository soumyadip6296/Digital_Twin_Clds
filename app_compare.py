"""
app_compare.py  —  AI Router vs Normal Router: Head-to-Head Comparison
-----------------------------------------------------------------------
Run with:
    streamlit run app_compare.py

Requires core_api.py to be running:
    python core_api.py
"""

import streamlit as st
import asyncio
import websockets
from websockets.exceptions import ConnectionClosedError
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import threading
import os
import time
from collections import deque

from streamlit.runtime.scriptrunner import add_script_run_ctx
import nest_asyncio
nest_asyncio.apply()

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(page_title="Router Comparison", layout="wide")
st.title("⚔️ AI Router vs Normal Router — Head-to-Head")
st.caption("Both routers see the exact same packets. Ground truth comes from the dataset labels.")

# =============================================================================
# CONSTANTS
# =============================================================================
FEATURES = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
    'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min',
    'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
    'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
    'Flow IAT Min', 'Fwd IAT Total', 'Bwd IAT Total', 'Fwd Header Length',
    'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length',
    'Max Packet Length', 'Packet Length Mean', 'Packet Length Std',
    'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
    'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'Down/Up Ratio',
    'Average Packet Size', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward'
]

PRIMARY_LATENCY_S = 0.01   # 10 ms
BACKUP_LATENCY_S  = 0.05   # 50 ms
HISTORY_LEN       = 80


# =============================================================================
# SESSION STATE
# =============================================================================
def _init_state():
    defaults = {
        "running":        False,
        "finished":       False,
        "packets_sent":   0,
        "total_packets":  0,
        # Per-router rolling history for charts
        "ai_errors":      deque(maxlen=HISTORY_LEN),
        "ai_routes":      deque(maxlen=HISTORY_LEN),
        "norm_latscores": deque(maxlen=HISTORY_LEN),
        "norm_routes":    deque(maxlen=HISTORY_LEN),
        "gt_history":       deque(maxlen=HISTORY_LEN),
        # New AI signal deques (from upgraded core_api)
        "ai_confidence":    deque(maxlen=HISTORY_LEN),
        "ai_error_deltas":  deque(maxlen=HISTORY_LEN),
        "nm_rate_zscores":  deque(maxlen=HISTORY_LEN),
        # Cluster transition event counter
        "ai_cluster_transitions": 0,
        # Confusion-matrix counters (TP / FP / FN / TN)
        "ai_tp": 0, "ai_fp": 0, "ai_fn": 0, "ai_tn": 0,
        "nm_tp": 0, "nm_fp": 0, "nm_fn": 0, "nm_tn": 0,
        # Route-switch counters
        "ai_switches":   0,
        "nm_switches":   0,
        "ai_last_route": 0,
        "nm_last_route": 0,
        # Latest raw result
        "latest": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# =============================================================================
# HELPERS
# =============================================================================
def _update_confusion(prefix, predicted_attack, ground_truth):
    if ground_truth and predicted_attack:
        st.session_state[f"{prefix}_tp"] += 1
    elif not ground_truth and predicted_attack:
        st.session_state[f"{prefix}_fp"] += 1
    elif ground_truth and not predicted_attack:
        st.session_state[f"{prefix}_fn"] += 1
    else:
        st.session_state[f"{prefix}_tn"] += 1


def _safe_div(n, d):
    return n / d if d else 0.0


def _compute_metrics(prefix):
    tp = st.session_state[f"{prefix}_tp"]
    fp = st.session_state[f"{prefix}_fp"]
    fn = st.session_state[f"{prefix}_fn"]
    tn = st.session_state[f"{prefix}_tn"]
    accuracy  = _safe_div(tp + tn, tp + fp + fn + tn)
    fpr       = _safe_div(fp, fp + tn)

    # FIX: When the dataset has zero positive ground-truth labels (e.g. the
    # CICIDS2017 Monday file is 100% benign), Precision/Recall/F1 are
    # mathematically undefined (0/0).  Returning 0.0 silently makes the
    # scorecard look "stuck" at 0%.  Use None to signal N/A to the UI.
    has_positives = (tp + fn) > 0
    if has_positives:
        precision = _safe_div(tp, tp + fp)
        recall    = _safe_div(tp, tp + fn)
        f1        = _safe_div(2 * precision * recall, precision + recall)
    else:
        precision = recall = f1 = None   # undefined — no attacks in dataset

    return dict(precision=precision, recall=recall, f1=f1, accuracy=accuracy, fpr=fpr,
                tp=tp, fp=fp, fn=fn, tn=tn, has_positives=has_positives)


# =============================================================================
# ASYNC INJECTION LOOP  (runs in a background thread — does NOT block the UI)
# =============================================================================
async def run_comparison(df: pd.DataFrame, speed: int):
    uri   = "ws://localhost:8000/ws/compare"
    delay = 1.0 / speed
    total = len(df)

    st.session_state.total_packets = total

    try:
        async with websockets.connect(uri, ping_interval=None) as ws:
            for i in range(total):
                if not st.session_state.running:
                    break

                row      = df.iloc[i]
                features = pd.to_numeric(row[FEATURES], errors='coerce').fillna(0).tolist()
                vol      = float(row.get('Total Length of Fwd Packets', 0) +
                                 row.get('Total Length of Bwd Packets', 0))
                label    = str(row.get('Label', 'BENIGN')).strip().upper()
                is_atk   = label != 'BENIGN'
                lat_a    = 0.95 if is_atk else np.random.uniform(PRIMARY_LATENCY_S,
                                                                   PRIMARY_LATENCY_S + 0.02)

                payload = {
                    "features":            features,
                    "volume":              vol,
                    "lat_a":               lat_a,
                    "lat_b":               BACKUP_LATENCY_S,
                    "ground_truth_attack": is_atk,
                }
                await ws.send(json.dumps(payload))
                result = json.loads(await ws.recv())

                ai = result["ai"]
                nm = result["normal"]
                gt = result["ground_truth"]

                _update_confusion("ai", ai["is_attack"], gt)
                _update_confusion("nm", nm["is_attack"], gt)

                if ai["route"] != st.session_state.ai_last_route:
                    st.session_state.ai_switches += 1
                    st.session_state.ai_last_route = ai["route"]

                if nm["route"] != st.session_state.nm_last_route:
                    st.session_state.nm_switches += 1
                    st.session_state.nm_last_route = nm["route"]

                # Core deques — keep in sync
                st.session_state.ai_errors.append(ai["error"])
                st.session_state.ai_routes.append(ai["route"])
                st.session_state.norm_latscores.append(nm["lat_score"])
                st.session_state.norm_routes.append(nm["route"])
                st.session_state.gt_history.append(1 if gt else 0)
                # New signal deques
                st.session_state.ai_confidence.append(ai.get("attack_confidence", 0.0))
                st.session_state.ai_error_deltas.append(ai.get("error_delta", 0.0))
                st.session_state.nm_rate_zscores.append(nm.get("rate_zscore", 0.0))
                # Cluster transition counter
                if ai.get("cluster_transition", False):
                    st.session_state.ai_cluster_transitions += 1

                st.session_state.packets_sent = i + 1
                st.session_state.latest = result

                # At high speed skip the sleep entirely;
                # yield control briefly so the event loop stays alive
                if delay > 0.005:
                    await asyncio.sleep(delay)
                else:
                    await asyncio.sleep(0)

    except ConnectionClosedError as e:
        # FIX: Code 1001 = "Going Away" — the server closed this socket because
        # a newer session superseded it (user clicked Start again). The stop_event
        # was already set by _start_background_thread before the new thread
        # launched, so this is a clean exit. Do NOT store it as an error or the
        # UI will show "Connection error" and "Simulation complete!" prematurely.
        if e.rcvd is not None and e.rcvd.code == 1001:
            pass  # clean supersession — discard silently
        else:
            st.session_state.latest = {"_error": str(e)}
    except Exception as e:
        st.session_state.latest = {"_error": str(e)}

    st.session_state.running  = False
    st.session_state.finished = True


# FIX: Injection runs in its own thread+event loop so Streamlit's UI thread
# is never blocked — live charts, banners and metric cards update every rerun.
def _start_background_thread(df: pd.DataFrame, speed: int):
    def _worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_comparison(df, speed))

    t = threading.Thread(target=_worker, daemon=True)
    add_script_run_ctx(t)
    t.start()


# =============================================================================
# SIDEBAR — CONTROLS
# =============================================================================
with st.sidebar:
    st.header("⚙️ Simulation Settings")

    if not os.path.exists("data"):
        os.makedirs("data")
    datasets = [f for f in os.listdir("data") if f.endswith('.csv')]
    selected = st.selectbox("Dataset", datasets)
    speed    = st.slider("Injection speed (packets/sec)", 1, 500, 50)
    sample_size = st.select_slider(
        "Dataset sample size",
        options=[1000, 2000, 5000, 10000, 20000, 50000, "All"],
        value=10000,
        help="Stratified sample preserving attack/benign ratio. 'All' = full dataset."
    )

    st.divider()
    col_a, col_b = st.columns(2)

    start_clicked = col_a.button("▶️ Start", type="primary",
                                  width='stretch',
                                  disabled=st.session_state.running)
    stop_clicked  = col_b.button("🛑 Stop",
                                  width='stretch',
                                  disabled=not st.session_state.running)
    reset_clicked = st.button("🔄 Reset All", width='stretch')

    if start_clicked and selected:
        # Wipe all state so counters start fresh
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        _init_state()
        st.session_state.running  = True
        st.session_state.finished = False

        df_loaded = pd.read_csv(f"data/{selected}")
        df_loaded.columns = [c.strip() for c in df_loaded.columns]
        df_loaded = df_loaded.replace(['Infinity', 'inf', 'NaN'], np.nan).fillna(0)

        # Stratified sample — preserves attack/benign ratio
        if sample_size != 'All':
            label_col = 'Label' if 'Label' in df_loaded.columns else None
            if label_col and df_loaded[label_col].nunique() > 1:
                df_loaded = (
                    df_loaded
                    .groupby(label_col, group_keys=False)
                    .apply(lambda g: g.sample(
                        min(len(g), max(1, int(sample_size * len(g) / len(df_loaded)))),
                        random_state=42
                    ))
                    .sample(frac=1, random_state=42)
                    .reset_index(drop=True)
                )
            else:
                df_loaded = df_loaded.sample(
                    min(sample_size, len(df_loaded)), random_state=42
                ).reset_index(drop=True)
            st.sidebar.caption(f"Sampled {len(df_loaded):,} rows from dataset.")

        # FIX: launch in background thread; hand control back to Streamlit immediately
        _start_background_thread(df_loaded, speed)
        st.rerun()

    if stop_clicked:
        st.session_state.running = False
        # FIX: rerun so the UI reflects the stopped state immediately
        st.rerun()

    if reset_clicked:
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        _init_state()
        st.rerun()

    st.divider()
    st.metric("Packets processed",
              f"{st.session_state.packets_sent} / {st.session_state.total_packets or '?'}")


# =============================================================================
# MAIN LAYOUT
# =============================================================================

# ── Progress bar ─────────────────────────────────────────────────────────────
prog_bar    = st.empty()
status_line = st.empty()

if st.session_state.total_packets > 0:
    frac = st.session_state.packets_sent / st.session_state.total_packets
    prog_bar.progress(frac, text=f"Progress: {st.session_state.packets_sent} / "
                                  f"{st.session_state.total_packets} packets")

if st.session_state.latest and "_error" in st.session_state.latest:
    st.error(f"Connection error: {st.session_state.latest['_error']}")

# ── Live status banners ───────────────────────────────────────────────────────
banner_left, banner_right = st.columns(2)
with banner_left:
    st.markdown("### 🤖 AI Router (LSTM + PPO)")
    ai_banner = st.empty()
with banner_right:
    st.markdown("### 🖧 Normal Router (Rule-Based)")
    nm_banner = st.empty()

if st.session_state.latest and "_error" not in st.session_state.latest:
    p    = st.session_state.latest
    conf = p["ai"].get("attack_confidence", 0.0)
    clus = p["ai"].get("cluster_transition", False)
    # FIX: Only show 🔀 tag when confidence is already elevated (> 0.15).
    # Previously it showed on every cluster flip, which fired on 54% of
    # benign packets — making the banner meaningless noise.
    clus_tag = " 🔀 pattern shift" if (clus and conf > 0.15) else ""
    if p["ai"]["is_attack"]:
        ai_banner.error(f"⚠️ ATTACK DETECTED — confidence {conf:.0%}{clus_tag}")
    else:
        ai_banner.success(f"✅ Traffic Normal — confidence {1-conf:.0%}{clus_tag}")
    rate_z = p["normal"].get("rate_zscore", 0.0)
    rate_tag = f" (rate z={rate_z:.1f})" if abs(rate_z) > 2 else ""
    if p["normal"]["is_attack"]:
        nm_banner.error(f"⚠️ ALERT — switching backup{rate_tag}")
    else:
        nm_banner.success(f"✅ Traffic Normal{rate_tag}")

st.divider()

# ── Live Metric Cards ─────────────────────────────────────────────────────────
ai_m = _compute_metrics("ai")
nm_m = _compute_metrics("nm")

# FIX: Format N/A for undefined metrics (when dataset has no attack labels)
def _fmt(v): return f"{v:.2%}" if v is not None else "N/A"

# Warn prominently when the loaded dataset has no attack traffic at all
# FIX: Only show the all-benign warning when we have a representative sample.
# Datasets like Wednesday start with many benign rows before attacks appear —
# showing "no attack traffic" at 1% progress misleads the user into thinking
# the dataset is all-benign when it isn't.
# Threshold: either the simulation is finished, or we've seen > 15% of packets.
_enough_data = (
    st.session_state.finished or
    (st.session_state.total_packets > 0 and
     st.session_state.packets_sent / st.session_state.total_packets > 0.15)
)
if not ai_m["has_positives"] and st.session_state.packets_sent > 50 and _enough_data:
    st.warning(
        "⚠️ **No attack traffic detected in this dataset.** "
        "Precision, Recall, and F1 are undefined (0/0). "
        "This is expected for all-benign datasets like Monday-WorkingHours. "
        "Use **False Positive Rate** and **Accuracy** to compare routers here."
    )

mc = st.columns(9)
mc[0].metric("AI Accuracy",    f"{ai_m['accuracy']:.2%}")
mc[1].metric("AI Precision",   _fmt(ai_m['precision']))
mc[2].metric("AI Recall",      _fmt(ai_m['recall']))
mc[3].metric("AI F1",          _fmt(ai_m['f1']))
mc[4].metric("AI FP Rate",     f"{ai_m['fpr']:.2%}")
mc[5].metric("Norm Accuracy",  f"{nm_m['accuracy']:.2%}")
mc[6].metric("Norm Precision", _fmt(nm_m['precision']))
mc[7].metric("Norm Recall",    _fmt(nm_m['recall']))
mc[8].metric("Norm FP Rate",   f"{nm_m['fpr']:.2%}")

# Second row — new AI signal metrics
mc2 = st.columns(4)
latest_conf  = list(st.session_state.ai_confidence)[-1]  if st.session_state.ai_confidence  else 0.0
latest_delta = list(st.session_state.ai_error_deltas)[-1] if st.session_state.ai_error_deltas else 0.0
latest_fcst  = st.session_state.latest["ai"].get("forecast_deviation", 0.0) if st.session_state.latest and "_error" not in st.session_state.latest else 0.0
mc2[0].metric("AI Attack Confidence", f"{latest_conf:.2%}", help="Unified score combining Observer + Prophet + Latency + Cluster signals")
mc2[1].metric("AI Error Trend",       f"{latest_delta:+.5f}", help="Positive = anomaly score rising (more suspicious)")
mc2[2].metric("AI Forecast Deviation",f"{latest_fcst:.4f}",  help="Prophet: how much actual volume deviates from predicted")
mc2[3].metric("Cluster Transitions",  str(st.session_state.ai_cluster_transitions), help="# times traffic pattern shifted abruptly")

st.divider()

# ── Live Charts ───────────────────────────────────────────────────────────────
chart_left, chart_right = st.columns(2)

with chart_left:
    st.subheader("🤖 AI: Attack Confidence + Error Trend vs Ground Truth")
    chart_ai = st.empty()

with chart_right:
    st.subheader("🖧 Normal: Rolling Latency Score vs Ground Truth")
    chart_nm = st.empty()


def _render_charts():
    gt         = list(st.session_state.gt_history)
    errors     = list(st.session_state.ai_errors)
    confidence = list(st.session_state.ai_confidence)
    lat_scores = list(st.session_state.norm_latscores)
    rate_zs    = list(st.session_state.nm_rate_zscores)
    n = min(len(gt), len(errors), len(confidence), len(lat_scores), len(rate_zs))
    if n == 0:
        return
    gt, errors, confidence = gt[:n], errors[:n], confidence[:n]
    lat_scores, rate_zs    = lat_scores[:n], rate_zs[:n]
    x = list(range(n))

    # ── AI chart: confidence (primary) + raw error (secondary) ──────────────
    fig_ai = go.Figure()
    fig_ai.add_trace(go.Scatter(
        x=x, y=confidence, mode='lines', name='Attack Confidence',
        line=dict(color='#8b5cf6', width=2),
        fill='tozeroy', fillcolor='rgba(139,92,246,0.15)'
    ))
    fig_ai.add_trace(go.Scatter(
        x=x, y=errors, mode='lines', name='Raw Anomaly Score',
        line=dict(color='#a78bfa', width=1, dash='dot'),
    ))
    fig_ai.add_trace(go.Scatter(
        x=x, y=gt, mode='lines', name='Ground Truth Attack',
        line=dict(color='#ef4444', width=1, dash='dot'),
        yaxis='y2'
    ))
    # FIX: Draw both thresholds — sustained (0.35) and spike (0.28) —
    # so the chart reflects the actual dual-mode detection logic in core_api.
    fig_ai.add_hline(y=0.35, line_dash='dot', line_color='orange',
                     annotation_text='Sustained Threshold (0.35)',
                     annotation_position='top left')
    fig_ai.add_hline(y=0.28, line_dash='dash', line_color='gold',
                     annotation_text='Spike Threshold (0.28, requires lat>60ms)',
                     annotation_position='bottom left')
    fig_ai.update_layout(
        height=300, margin=dict(l=50, r=20, t=20, b=40),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(title='Attack Confidence', range=[0, 1.1]),
        yaxis2=dict(title='Attack (0/1)', overlaying='y', side='right',
                    range=[-0.1, 1.5], showgrid=False),
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    chart_ai.plotly_chart(fig_ai, width='stretch')

    # ── Normal chart: latency (primary) + packet-rate z-score (secondary) ───
    # Normalise rate_z to [0, ~0.12] range for co-display with latency
    rate_zs_norm = [min(max(z / 10.0, 0), 0.12) for z in rate_zs]
    fig_nm = go.Figure()
    fig_nm.add_trace(go.Scatter(
        x=x, y=lat_scores, mode='lines', name='Avg Latency',
        line=dict(color='#3b82f6', width=2),
        fill='tozeroy', fillcolor='rgba(59,130,246,0.15)'
    ))
    fig_nm.add_trace(go.Scatter(
        x=x, y=rate_zs_norm, mode='lines', name='Pkt-Rate z-score (÷10)',
        line=dict(color='#06b6d4', width=1, dash='dash'),
    ))
    fig_nm.add_trace(go.Scatter(
        x=x, y=gt, mode='lines', name='Ground Truth Attack',
        line=dict(color='#ef4444', width=1, dash='dot'),
        yaxis='y2'
    ))
    fig_nm.add_hline(y=0.08, line_dash='dot', line_color='orange',
                     annotation_text='Latency Threshold', annotation_position='top left')
    fig_nm.update_layout(
        height=300, margin=dict(l=50, r=20, t=20, b=40),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(title='Latency (s) / Rate-z÷10', range=[0, 0.15]),
        yaxis2=dict(title='Attack (0/1)', overlaying='y', side='right',
                    range=[-0.1, 1.5], showgrid=False),
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    chart_nm.plotly_chart(fig_nm, width='stretch')


if st.session_state.packets_sent > 0:
    _render_charts()

st.divider()

# ── Route Decision Timeline ───────────────────────────────────────────────────
st.subheader("📡 Route Decisions Over Time  (0 = Primary · 1 = Backup)")
route_chart = st.empty()


def _render_route_chart():
    # FIX: trim to shortest to avoid x-axis misalignment
    ai_r = list(st.session_state.ai_routes)
    nm_r = list(st.session_state.norm_routes)
    gt   = list(st.session_state.gt_history)
    n    = min(len(ai_r), len(nm_r), len(gt))
    if n == 0:
        return
    ai_r, nm_r, gt = ai_r[:n], nm_r[:n], gt[:n]
    x = list(range(n))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=ai_r, mode='lines+markers',
                             name='AI Route',
                             line=dict(color='#8b5cf6', width=2),
                             marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=x, y=nm_r, mode='lines+markers',
                             name='Normal Route',
                             line=dict(color='#3b82f6', width=2, dash='dash'),
                             marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=x, y=gt, mode='lines',
                             name='Ground Truth',
                             line=dict(color='#ef4444', width=1, dash='dot'),
                             yaxis='y2'))
    fig.update_layout(
        height=250, margin=dict(l=50, r=20, t=20, b=40),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(title='Route', tickvals=[0, 1],
                   ticktext=['Primary', 'Backup'], range=[-0.2, 1.4]),
        yaxis2=dict(title='Attack', overlaying='y', side='right',
                    range=[-0.1, 1.5], showgrid=False),
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    route_chart.plotly_chart(fig, width='stretch')


if st.session_state.packets_sent > 0:
    _render_route_chart()

st.divider()

# ── Scorecard ─────────────────────────────────────────────────────────────────
st.subheader("🏆 Scorecard")
sc_left, sc_right, sc_verdict = st.columns([2, 2, 1])


def _scorecard_table(m, switches, label):
    # FIX: Show "N/A" for Precision/Recall/F1 when no positive ground-truth
    # labels exist — displaying 0.00% implies the router is bad at detection
    # when in reality detection is simply undefined for this dataset.
    return pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall (Detection Rate)",
                   "F1 Score", "False Positive Rate", "Route Switches",
                   "True Positives", "False Positives", "False Negatives", "True Negatives"],
        label: [
            f"{m['accuracy']:.2%}", _fmt(m['precision']), _fmt(m['recall']),
            _fmt(m['f1']),          f"{m['fpr']:.2%}",    str(switches),
            str(m['tp']), str(m['fp']), str(m['fn']), str(m['tn'])
        ]
    })


with sc_left:
    st.markdown("**🤖 AI Router**")
    st.dataframe(_scorecard_table(ai_m, st.session_state.ai_switches, "AI Router"),
                 hide_index=True, width='stretch')

with sc_right:
    st.markdown("**🖧 Normal Router**")
    st.dataframe(_scorecard_table(nm_m, st.session_state.nm_switches, "Normal Router"),
                 hide_index=True, width='stretch')

with sc_verdict:
    st.markdown("**🥇 Winner**")
    if st.session_state.packets_sent > 10:
        # FIX: Use a mode-aware scoring formula.
        #
        # All-benign dataset (has_positives=False):
        #   F1 and Recall are undefined (0/0). Scoring them as 0 would make the
        #   winner look "stuck" at +0.00%. Instead score purely on FPR and
        #   Accuracy, which are the only meaningful metrics here.
        #
        # Mixed dataset (has_positives=True):
        #   Use the original F1/Recall/FPR composite.
        delta_fpr = ai_m["fpr"] - nm_m["fpr"]          # negative = AI is better
        delta_acc = ai_m["accuracy"] - nm_m["accuracy"]

        if not ai_m["has_positives"]:
            # All-benign: lower FPR wins; tie-break on accuracy
            ai_score = (1 - ai_m["fpr"]) * 0.7 + ai_m["accuracy"] * 0.3
            nm_score = (1 - nm_m["fpr"]) * 0.7 + nm_m["accuracy"] * 0.3
            if ai_score > nm_score:
                st.success("🤖 AI Router wins!")
            elif nm_score > ai_score:
                st.info("🖧 Normal Router wins!")
            else:
                st.warning("🤝 Tie!")
            st.caption("ℹ️ No attack traffic — scored on FPR + Accuracy only.")
            st.metric("AI FPR advantage",  f"{-delta_fpr:+.2%}",
                      help="Negative FPR delta = fewer false alarms")
            st.metric("AI Acc advantage",  f"{delta_acc:+.2%}")
        else:
            # Normal mixed dataset — F1/Recall/FPR composite
            ai_score  = (ai_m["f1"] or 0) * 0.5 + (ai_m["recall"] or 0) * 0.3 + (1 - ai_m["fpr"]) * 0.2
            nm_score  = (nm_m["f1"] or 0) * 0.5 + (nm_m["recall"] or 0) * 0.3 + (1 - nm_m["fpr"]) * 0.2
            delta_f1  = (ai_m["f1"] or 0) - (nm_m["f1"] or 0)
            delta_rec = (ai_m["recall"] or 0) - (nm_m["recall"] or 0)
            if ai_score > nm_score:
                st.success("🤖 AI Router wins!")
            elif nm_score > ai_score:
                st.info("🖧 Normal Router wins!")
            else:
                st.warning("🤝 Tie!")

            # FIX: When attacks exist (has_positives=True) but both routers
            # have TP=0, F1/Recall advantages are 0.00% for both — which
            # looks broken. Add a contextual note explaining the situation,
            # and only show FPR advantage (the only real differentiator).
            both_missed = ai_m["has_positives"] and ai_m["tp"] == 0 and nm_m["tp"] == 0
            if both_missed:
                st.caption(
                    "⚠️ Attack traffic exists but neither router detected it yet. "
                    "F1/Recall are 0 for both — only FPR differentiates them."
                )
                st.metric("AI FPR advantage", f"{-delta_fpr:+.2%}")
            else:
                st.metric("AI F1 advantage",     f"{delta_f1:+.2%}")
                st.metric("AI Recall advantage", f"{delta_rec:+.2%}")
                st.metric("AI FPR advantage",    f"{-delta_fpr:+.2%}")
    else:
        st.info("Awaiting data…")

# ── Confusion Matrices ────────────────────────────────────────────────────────
st.divider()
st.subheader("🔢 Confusion Matrices")
cm_left, cm_right = st.columns(2)


def _confusion_fig(m, title):
    z    = [[m['tn'], m['fp']], [m['fn'], m['tp']]]
    text = [[f"TN\n{m['tn']}", f"FP\n{m['fp']}"],
            [f"FN\n{m['fn']}", f"TP\n{m['tp']}"]]
    fig  = px.imshow(z, text_auto=False,
                     color_continuous_scale='RdYlGn',
                     labels=dict(x="Predicted", y="Actual"),
                     x=["Benign", "Attack"], y=["Benign", "Attack"])
    for i in range(2):
        for j in range(2):
            fig.add_annotation(x=j, y=i, text=text[i][j],
                               showarrow=False, font=dict(size=16, color="black"))
    fig.update_layout(title=title, height=280,
                      margin=dict(l=20, r=20, t=40, b=20),
                      coloraxis_showscale=False)
    return fig


# FIX: use st.plotly_chart() inside with-column blocks, not col.plotly_chart()
with cm_left:
    st.plotly_chart(_confusion_fig(ai_m, "🤖 AI Router"), width='stretch')
with cm_right:
    st.plotly_chart(_confusion_fig(nm_m, "🖧 Normal Router"), width='stretch')

# ── Auto-rerun while simulation is live ──────────────────────────────────────
if st.session_state.running:
    time.sleep(0.4)
    st.rerun()
elif st.session_state.finished:
    st.balloons()
    status_line.success("✅ Simulation complete! Full scorecard above.")