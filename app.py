from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from edgecast.datasets import (
    BURSTGPT_INFO,
    FINESERVE_INFO,
    FINESERVE_LITE_CLASSES,
    demand_from_window,
    load_preprocessed_trace,
    select_bursty_window,
    trace_metadata,
)
from edgecast.policies import (
    CacheAwarePolicy,
    NguyenPaperPolicy,
    optimize_initial_allocation,
)
from edgecast.scenarios import ramp_video_burst, sudden_spike
from edgecast.simulation import SimulationConfig, run_simulation
from edgecast.visualization import build_comparison_animation

st.set_page_config(
    page_title="EdgeCast · Live Dual-Method Demonstration",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
:root {
  --bg: #070707;
  --panel: #0d0d0d;
  --panel-card: #131313;
  --line: rgba(255,255,255,.10);
  --line-strong: rgba(255,255,255,.18);
  --text: #f5f5f5;
  --muted: #8e8e8e;
  --good: #34d399;
  --warn: #f59e0b;
  --cyan: #22d3ee;
}
html, body {
  background: #070707 !important;
  color: var(--text) !important;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  overflow-x: hidden;
}
.stApp {
  background:
    radial-gradient(circle at 60% 0%, rgba(34,211,238,.03), transparent 40%),
    linear-gradient(180deg, #090909 0%, #050505 100%);
  color: var(--text);
}
.block-container {
  padding-top: 0.6rem !important;
  padding-bottom: 0.8rem !important;
  padding-left: 1.2rem !important;
  padding-right: 1.2rem !important;
  max-width: 100% !important;
}
/* Compact Left Control Deck */
.ctrl-box {
  border: 1px solid var(--line);
  border-radius: 14px;
  background: var(--panel);
  padding: 12px 14px;
  margin-bottom: 10px;
}
.ctrl-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}
.ctrl-title {
  font-size: 1.05rem;
  font-weight: 800;
  letter-spacing: -.02em;
  color: #fff;
}
.ctrl-sub {
  color: var(--muted);
  font-size: .72rem;
  line-height: 1.35;
  margin-bottom: 8px;
}
.section-label {
  color: #a0a0a0;
  font-size: .66rem;
  text-transform: uppercase;
  letter-spacing: .10em;
  font-weight: 700;
  margin: 6px 0 4px;
}
/* Compact KPI Summary */
.kpi-mini-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 7px;
  margin-top: 8px;
}
.kpi-mini-card {
  border: 1px solid var(--line);
  border-radius: 10px;
  background: var(--panel-card);
  padding: 8px 10px;
}
.kpi-mini-label {
  color: #7a7a7a;
  font-size: .61rem;
  text-transform: uppercase;
  letter-spacing: .08em;
  font-weight: 600;
}
.kpi-mini-val {
  margin-top: 2px;
  font-size: .92rem;
  font-weight: 750;
  color: #f2f2f2;
  font-variant-numeric: tabular-nums;
}
.kpi-good {
  color: var(--good);
}
.kpi-mini-delta {
  font-size: .63rem;
  margin-top: 1px;
}
/* Right Simulation Area */
.sim-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 12px;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: var(--panel);
  margin-bottom: 8px;
}
.sim-badge {
  font-size: .74rem;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 6px;
}
.dot-paper {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #f59e0b;
}
.dot-edgecast {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #34d399;
}
[data-testid="stSlider"] [role="slider"] {
  background: #f5f5f5 !important;
  border-color: #f5f5f5 !important;
}
[data-testid="stSlider"] label p {
  font-size: .78rem !important;
  font-weight: 600 !important;
}
div[data-testid="stRadio"] > div {
  gap: 6px;
}
</style>
""",
    unsafe_allow_html=True,
)

CACHE_DIR = Path(".cache/edgecast")
DEPLOY_DATA_DIR = Path("data/preprocessed")
BURSTGPT_AGG_PATH = DEPLOY_DATA_DIR / "BurstGPT_1_60s.csv"
FINESERVE_LITE_AGG_PATH = DEPLOY_DATA_DIR / "FineServe_lite_60s.csv"

burstgpt_trace = load_preprocessed_trace(BURSTGPT_AGG_PATH) if BURSTGPT_AGG_PATH.exists() else None
fineserve_trace = load_preprocessed_trace(FINESERVE_LITE_AGG_PATH) if FINESERVE_LITE_AGG_PATH.exists() else None

# Split Layout: Left Controls (1), Right Simulation (3.3)
col_ctrl, col_sim = st.columns([1.05, 3.35], gap="small")

with col_ctrl:
    st.markdown(
        """
<div class="ctrl-box">
  <div class="ctrl-header">
    <div class="ctrl-title">◈ EdgeCast Lab</div>
    <span style="color:#22d3ee; font-size:.65rem; font-weight:700; border:1px solid rgba(34,211,238,.3); padding:2px 6px; border-radius:6px; background:rgba(34,211,238,.08);">DUAL POLICY</span>
  </div>
  <div class="ctrl-sub">Play with traffic & cluster scale on the left; watch side-by-side reactions on the right.</div>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown('<div class="section-label">1. Workload Source</div>', unsafe_allow_html=True)
        workload_source = st.radio(
            "Workload",
            [
                "🎛️ Manual Live Sliders",
                "🤖 Automated Synthetic",
                "📊 BurstGPT (Azure)",
                "🔬 FineServe (Multi-Class)",
            ],
            label_visibility="collapsed",
        )

        st.markdown('<div class="section-label">2. Cluster Scale</div>', unsafe_allow_html=True)
        node_count = st.slider(
            "Edge Servers",
            min_value=3,
            max_value=8,
            value=6,
            step=1,
            help="Number of heterogeneous edge inference nodes in the fabric (3 to 8).",
        )

        if workload_source.startswith("🎛️ Manual"):
            st.markdown('<div class="section-label">3. Live Model Traffic</div>', unsafe_allow_html=True)
            
            # Preset row
            p1, p2, p3 = st.columns(3)
            with p1:
                if st.button("⚡ Video", use_container_width=True, help="Surge video traffic"):
                    st.session_state["m_text"] = 5
                    st.session_state["m_image"] = 4
                    st.session_state["m_video"] = 24
                    st.rerun()
            with p2:
                if st.button("💬 Text", use_container_width=True, help="Spike text chat traffic"):
                    st.session_state["m_text"] = 32
                    st.session_state["m_image"] = 4
                    st.session_state["m_video"] = 3
                    st.rerun()
            with p3:
                if st.button("⚖️ Reset", use_container_width=True, help="Reset to balanced baseline"):
                    st.session_state["m_text"] = 5
                    st.session_state["m_image"] = 4
                    st.session_state["m_video"] = 3
                    st.rerun()

            if "m_text" not in st.session_state:
                st.session_state["m_text"] = 5
            if "m_image" not in st.session_state:
                st.session_state["m_image"] = 4
            if "m_video" not in st.session_state:
                st.session_state["m_video"] = 18

            live_text = st.slider("TEXT Traffic", 0, 50, key="m_text")
            live_image = st.slider("IMAGE Traffic", 0, 50, key="m_image")
            live_video = st.slider("VIDEO Traffic", 0, 50, key="m_video")

            baseline_slot = {"text": 5, "image": 4, "video": 3}
            live_slot = {
                "text": int(live_text),
                "image": int(live_image),
                "video": int(live_video),
            }
            warmup_slots = 12
            live_slots = 40
            allocation_basis_demand = [dict(baseline_slot) for _ in range(warmup_slots)]
            demand = (
                [dict(baseline_slot) for _ in range(warmup_slots)]
                + [dict(live_slot) for _ in range(live_slots)]
            )
            models = ("text", "image", "video")
            workload_label = f"Manual live · text={live_text} image={live_image} video={live_video}"

        elif workload_source.startswith("🤖 Automated"):
            st.markdown('<div class="section-label">3. Synthetic Scenario</div>', unsafe_allow_html=True)
            scenario = st.selectbox("Scenario", ["Gradual video ramp", "Sudden spike"], label_visibility="collapsed")
            seed = st.number_input("Random Seed", min_value=0, max_value=99999, value=7)
            demand = (
                ramp_video_burst(seed=int(seed))
                if scenario == "Gradual video ramp"
                else sudden_spike(seed=int(seed))
            )
            models = tuple(demand[0].keys())
            allocation_basis_demand = demand
            workload_label = f"Synthetic · {scenario}"

        elif workload_source.startswith("📊 BurstGPT"):
            st.markdown('<div class="section-label">3. BurstGPT Replay</div>', unsafe_allow_html=True)
            if burstgpt_trace is None:
                st.error("BurstGPT trace file missing.")
                st.stop()
            bg_mode = st.selectbox("Mode", ["requests", "token_work"], format_func=lambda x: "Request counts" if x == "requests" else "Token work")
            window_slots = st.slider("Window Duration", 20, 60, 40, 5)
            bg_start, bg_end = select_bursty_window(burstgpt_trace, window_slots=int(window_slots))
            demand, models, _ = demand_from_window(
                burstgpt_trace,
                start_slot=bg_start,
                window_slots=int(window_slots),
                mode=bg_mode,
            )
            allocation_basis_demand = demand
            workload_label = f"BurstGPT ({bg_mode})"

        else:
            st.markdown('<div class="section-label">3. FineServe Replay</div>', unsafe_allow_html=True)
            if fineserve_trace is None:
                st.error("FineServe trace file missing.")
                st.stop()
            fs_mode = st.selectbox("Mode", ["requests", "token_work"], format_func=lambda x: "Request counts" if x == "requests" else "Token work")
            fs_window_slots = st.slider("Window Duration", 20, 60, 40, 5)
            fs_start, fs_end = select_bursty_window(fineserve_trace, window_slots=int(fs_window_slots))
            demand, models, _ = demand_from_window(
                fineserve_trace,
                start_slot=fs_start,
                window_slots=int(fs_window_slots),
                mode=fs_mode,
            )
            allocation_basis_demand = demand
            workload_label = f"FineServe ({fs_mode})"

# Run Simulation synchronously
capacity_basis = allocation_basis_demand if workload_source.startswith("🎛️ Manual") else demand
total_per_slot = [sum(slot.values()) for slot in capacity_basis]
p70 = float(pd.Series(total_per_slot).quantile(.70)) if total_per_slot else 1.0
capacity_scale = 0.85
base_capacity = max(1, int(math.ceil((p70 / max(1, node_count - 1)) * capacity_scale)))

factors_pool = [.74, .88, 1.0, 1.14, 1.31, .81, 1.05, .92]
factors = factors_pool[:node_count]
node_capacities = tuple(
    max(1, int(round(base_capacity * factors[i])))
    for i in range(node_count)
)

allocation = optimize_initial_allocation(
    allocation_basis_demand,
    models,
    node_capacities,
)

sim_config = SimulationConfig(
    models=models,
    initial_allocation=tuple(allocation),
    node_capacity=base_capacity,
    node_capacities=node_capacities,
    switch_time=4,
    warm_switch_time=1,
    prefetch_time=4,
    cache_slots=2,
    max_parallel_prefetch=1,
    cooldown=6,
    max_drain_slots=1000,
)

base_result = run_simulation(
    demand,
    NguyenPaperPolicy(max_exact_nodes=8),
    sim_config,
)
edgecast_result = run_simulation(
    demand,
    CacheAwarePolicy(max_exact_nodes=8, min_reallocation_interval=6),
    sim_config,
)

latency_gain = (
    100.0 * (base_result.avg_latency - edgecast_result.avg_latency) / base_result.avg_latency
    if base_result.avg_latency else 0.0
)
p95_gain = (
    100.0 * (base_result.p95_latency - edgecast_result.p95_latency) / base_result.p95_latency
    if base_result.p95_latency else 0.0
)
downtime_diff = base_result.switch_downtime_node_slots - edgecast_result.switch_downtime_node_slots
downtime_gain = (
    100.0 * downtime_diff / base_result.switch_downtime_node_slots
    if base_result.switch_downtime_node_slots else 0.0
)
backlog_base = sum(base_result.queues[m][-1] for m in models) if base_result.queues else 0
backlog_edge = sum(edgecast_result.queues[m][-1] for m in models) if edgecast_result.queues else 0

with col_ctrl:
    st.markdown(
        f"""
<div class="kpi-mini-grid">
  <div class="kpi-mini-card">
    <div class="kpi-mini-label">Avg Latency</div>
    <div class="kpi-mini-val">{base_result.avg_latency:.2f} → <span class="kpi-good">{edgecast_result.avg_latency:.2f}</span></div>
    <div class="kpi-mini-delta kpi-good">{latency_gain:+.1f}% lower</div>
  </div>
  <div class="kpi-mini-card">
    <div class="kpi-mini-label">P95 Latency</div>
    <div class="kpi-mini-val">{base_result.p95_latency:.1f} → <span class="kpi-good">{edgecast_result.p95_latency:.1f}</span></div>
    <div class="kpi-mini-delta kpi-good">{p95_gain:+.1f}% lower</div>
  </div>
  <div class="kpi-mini-card">
    <div class="kpi-mini-label">Downtime</div>
    <div class="kpi-mini-val">{base_result.switch_downtime_node_slots} → <span class="kpi-good">{edgecast_result.switch_downtime_node_slots}</span></div>
    <div class="kpi-mini-delta kpi-good">{downtime_gain:+.1f}% saved</div>
  </div>
  <div class="kpi-mini-card">
    <div class="kpi-mini-label">EdgeCast Staging</div>
    <div class="kpi-mini-val"><span class="kpi-good">{edgecast_result.prefetches}</span> staged</div>
    <div class="kpi-mini-delta" style="color:#888;">0 downtime</div>
  </div>
  <div class="kpi-mini-card">
    <div class="kpi-mini-label">Switches</div>
    <div class="kpi-mini-val">{base_result.switches} → <span class="kpi-good">{edgecast_result.switches}</span></div>
    <div class="kpi-mini-delta" style="color:#aaa;">{edgecast_result.warm_switches} warm / {edgecast_result.cold_switches} cold</div>
  </div>
  <div class="kpi-mini-card">
    <div class="kpi-mini-label">Stability Gate</div>
    <div class="kpi-mini-val"><span class="kpi-good">Active</span></div>
    <div class="kpi-mini-delta" style="color:#888;">6-slot persistence</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

with col_sim:
    st.markdown(
        f"""
<div class="sim-header">
  <div class="sim-badge"><span class="dot-paper"></span> <b>Nguyen et al. Base Paper</b> <span style="color:#777; font-weight:400;">(P2b queue service + P3 cold reallocation)</span></div>
  <div class="sim-badge" style="color:#888; font-size:.70rem;">{workload_label} · {node_count} Edge Nodes</div>
  <div class="sim-badge"><span class="dot-edgecast"></span> <b>EdgeCast Improved Model</b> <span style="color:#777; font-weight:400;">(Cache-aware warm cutover + background staging)</span></div>
</div>
""",
        unsafe_allow_html=True,
    )

    anim_html = build_comparison_animation(
        base_result,
        edgecast_result,
        height=690,
        layout="side",
    )

    components.html(
        anim_html,
        height=730,
        scrolling=False,
    )

    with st.expander("📊 Queue Backlog Progression & Audit Decision Log", expanded=False):
        max_slots = max(len(base_result.slots), len(edgecast_result.slots))
        slots = list(range(max_slots))
        base_backlog_series = [
            sum(base_result.queues[m][i] for m in models) if i < len(base_result.slots) else 0
            for i in slots
        ]
        edge_backlog_series = [
            sum(edgecast_result.queues[m][i] for m in models) if i < len(edgecast_result.slots) else 0
            for i in slots
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=slots,
                y=base_backlog_series,
                name="Base Paper Backlog",
                line=dict(color="#f59e0b", width=2.5),
                mode="lines",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=slots,
                y=edge_backlog_series,
                name="EdgeCast Backlog",
                line=dict(color="#34d399", width=2.8),
                mode="lines",
            )
        )

        fig.update_layout(
            height=250,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#0d0d0d",
            plot_bgcolor="#0d0d0d",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color="#ddd"),
            ),
            xaxis=dict(
                title="Simulation Slot",
                color="#888",
                gridcolor="rgba(255,255,255,.05)",
                zeroline=False,
            ),
            yaxis=dict(
                title="Total Queued Units",
                color="#888",
                gridcolor="rgba(255,255,255,.05)",
                zeroline=False,
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

        decisions_list = []
        for s_idx, (b_dec, e_dec) in enumerate(zip(base_result.decisions, edgecast_result.decisions)):
            b_act = b_dec.get("action", "hold") if b_dec else "hold"
            e_act = e_dec.get("action", "hold") if e_dec else "hold"
            if b_act != "hold" or e_act != "hold" or e_dec.get("stability_gate") == "minimum-persistence":
                decisions_list.append(
                    {
                        "Slot": s_idx,
                        "Paper Action": b_act,
                        "Paper Mode": b_dec.get("mode", "cold") if b_act != "hold" else "-",
                        "EdgeCast Action": e_act,
                        "EdgeCast Mode": e_dec.get("mode", "-"),
                        "EdgeCast Staged Model": e_dec.get("target", "-") if e_act == "prefetch" else "-",
                        "Stability Gate": e_dec.get("stability_gate", "passed" if e_act == "multi_switch" else "-"),
                    }
                )
        if decisions_list:
            st.dataframe(pd.DataFrame(decisions_list), use_container_width=True, hide_index=True)

    with st.expander("Live Run Verification & Export", expanded=False):
        st.caption(
            "These files are generated from the current server-side simulation run. "
            "The result metrics are not stored as display constants."
        )

        manifest = {
            "workload": workload_label,
            "node_count": int(node_count),
            "models": list(models),
            "simulation_config": asdict(sim_config),
            "base_summary": base_result.summary(),
            "edgecast_summary": edgecast_result.summary(),
            "method": {
                "base": "Nguyen et al. aggregate P2b + exact reduced P3 cold reallocation",
                "edgecast": "stabilized cache-aware P3 + background checkpoint staging",
            },
        }

        trace_rows = []
        slot_count = max(len(base_result.slots), len(edgecast_result.slots))
        for i in range(slot_count):
            base_decision = base_result.decisions[i] if i < len(base_result.decisions) else {}
            edge_decision = edgecast_result.decisions[i] if i < len(edgecast_result.decisions) else {}
            row = {
                "slot": i,
                "base_total_queue": (
                    sum(base_result.queues[m][i] for m in models)
                    if i < len(base_result.slots)
                    else None
                ),
                "edgecast_total_queue": (
                    sum(edgecast_result.queues[m][i] for m in models)
                    if i < len(edgecast_result.slots)
                    else None
                ),
                "base_action": base_decision.get("action", "hold"),
                "edgecast_action": edge_decision.get("action", "hold"),
                "edgecast_Tp": edge_decision.get("Tp"),
                "edgecast_Tc": edge_decision.get("Tc"),
                "edgecast_sigma": edge_decision.get("sigma"),
                "edgecast_net_benefit": edge_decision.get("net_benefit"),
                "edgecast_stability_gate": edge_decision.get("stability_gate"),
            }
            for model in models:
                row[f"base_queue_{model}"] = (
                    base_result.queues[model][i] if i < len(base_result.slots) else None
                )
                row[f"edgecast_queue_{model}"] = (
                    edgecast_result.queues[model][i] if i < len(edgecast_result.slots) else None
                )
            trace_rows.append(row)

        trace_df = pd.DataFrame(trace_rows)
        v1, v2 = st.columns(2)
        with v1:
            st.json(manifest, expanded=False)
        with v2:
            st.metric("Requests in this run", f"{base_result.total_arrivals:,}")
            st.metric("Simulation slots", f"{slot_count:,}")
            st.metric("Completed requests", f"{edgecast_result.completed:,}")

        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "Download run manifest (JSON)",
                data=json.dumps(manifest, indent=2),
                file_name="edgecast_run_manifest.json",
                mime="application/json",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                "Download slot trace (CSV)",
                data=trace_df.to_csv(index=False),
                file_name="edgecast_slot_trace.csv",
                mime="text/csv",
                use_container_width=True,
            )
