"""
doctor_console.py
NeuroVistaFM Phase 2 — Doctor Console (paired with tv_display.py)

Goals (your requirements):
- Keeps existing components: procedure profile setup, detection settings, viewer controls, intraop synopsis, scoring.
- Uses MedGemma Phase-2 client (medgemma_client_phase2_v2.py) when available.
- Prompts should not repeat (hard anti-repeat guard).
- Brain highlights expected lobes for the CURRENT prompt:
    Correct -> RED, Partial -> AMBER, Incorrect -> BLUE
  (pending/unscored prompt: highlight is shown in CYAN)
- Writes shared state continuously for TV display via state_store.py.
- End Test button present (TV shows END/ENDED).
- TV should not hard-refresh (handled in tv_display.py).

Run:
  streamlit run doctor_console.py --server.port 8501
  streamlit run tv_display.py --server.port 8502
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from state_store import write_state
from procedure_profile import procedure_profile_ui
from prompt_state import (
    ensure_state,
    set_current_prompt,
    log_score,
    append_sii_sample,
    reset_prompt_session,
)
from alert_engine import detect_alert
from generate_data import generate_case
from metrics import current_state
from brain_mesh import load_brain_mesh
from brain_view_phase2 import make_brain_figure

# Prefer Phase-2 v2 MedGemma client; fall back gracefully
try:
    from medgemma_client_phase2_v2 import next_prompt, make_synopsis  # type: ignore
except Exception:
    try:
        from medgemma_client_phase2 import next_prompt, make_synopsis  # type: ignore
    except Exception:
        from medgemma_client import next_prompt, make_synopsis  # type: ignore


st.set_page_config(page_title="NeuroVistaFM — Doctor Console", layout="wide")


# -----------------------------
# Helpers
# -----------------------------
def _prompt_fingerprint(p: Dict[str, Any]) -> str:
    """Stable fingerprint for anti-repeat. Includes color tile if present."""
    p = p or {}
    txt = str(p.get("prompt_text", "")).strip().lower()
    dom = str(p.get("domain", "")).strip().lower()
    ptype = str(p.get("prompt_type", "")).strip().lower()
    color = ""
    payload = p.get("payload") or {}
    if isinstance(payload, dict):
        color = str(payload.get("color_hex", "")).strip().lower()
    raw = f"{dom}|{ptype}|{color}|{txt}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _norm_lobe(x: str) -> str:
    s = str(x or "").strip().upper()
    if not s:
        return ""
    if "TEMP" in s:
        return "TEMPORAL"
    if "FRON" in s:
        return "FRONTAL"
    if "PARI" in s:
        return "PARIETAL"
    if "OCCI" in s:
        return "OCCIPITAL"
    return s


def _expected_lobes_from_prompt(p: Dict[str, Any]) -> List[str]:
    """Extract expected lobes from prompt dict or infer from domain."""
    p = p or {}
    lobes = p.get("expected_lobes") or p.get("lobes") or []
    if isinstance(lobes, str):
        lobes = [lobes]
    out: List[str] = []
    for x in (lobes or []):
        lx = _norm_lobe(x)
        if lx:
            out.append(lx)
    if out:
        # de-dup keep order
        seen = set()
        uniq = []
        for l in out:
            if l not in seen:
                uniq.append(l)
                seen.add(l)
        return uniq

    dom = str(p.get("domain", "")).upper()
    if dom in ("AUD", "AUDIO"):
        return ["TEMPORAL"]
    if dom in ("LANG", "SPEECH"):
        return ["TEMPORAL", "FRONTAL"]
    if dom in ("VIS", "VISION"):
        return ["OCCIPITAL"]
    if dom in ("MOT", "MOTOR"):
        return ["FRONTAL", "PARIETAL"]
    if dom in ("EXEC", "ATTN"):
        return ["FRONTAL"]
    return []


def _score_to_overlay(score: Optional[int]) -> Optional[int]:
    """
    brain_view_phase2 expects highlight_score to choose color:
      2=red, 1=amber, 0=blue
    We'll use:
      None -> show pending cyan via a special value (-1) handled in patched brain_view_phase2
    """
    return score


def _write_tv_state(
    session_id: str,
    *,
    status: str,
    prompt: Optional[Dict[str, Any]],
    sii_now: float,
    in_episode: bool,
    domains_impacted: List[str],
    paused: bool,
    prompt_counter: int,
    note: str = "",
    alert: Optional[Dict[str, Any]] = None,
) -> None:
    write_state(
        session_id,
        {
            "status": status,
            "prompt": prompt,
            "sii_now": float(sii_now),
            "sii_series": st.session_state.get("sii_series", []),
            "in_episode": bool(in_episode),
            "domains_impacted": domains_impacted,
            "paused": bool(paused),
            "prompt_counter": int(prompt_counter),
            "note": note,
            "alert": alert,
            "_server_ts": time.time(),
            "session_ended": bool(status in ("END","ENDED")),
        },
    )


def _safe_now_t(case: Dict[str, Any]) -> int:
    ts = case.get("timeseries")
    if isinstance(ts, pd.DataFrame) and not ts.empty:
        return int(ts["t"].max())
    return 0


# -----------------------------
# Sidebar: session + core toggles
# -----------------------------
st.sidebar.header("Sharing")
session_id = st.sidebar.text_input("Session ID (must match TV display)", value="OR1")


# -----------------------------
# Session state init
# -----------------------------
ensure_state(st.session_state)
st.session_state.setdefault("phase", "SETUP")  # SETUP | LIVE
st.session_state.setdefault("paused", False)
st.session_state.setdefault("procedure_profile", {})
st.session_state.setdefault("domain_score_state", {})  # domain->score
st.session_state.setdefault("served_prompt_hashes", [])  # anti-repeat list
st.session_state.setdefault("last_alert", None)


# -----------------------------
# Sidebar: case + detection settings
# -----------------------------
st.sidebar.header("Case")
seed = int(st.sidebar.number_input("Synthetic case seed", min_value=0, max_value=10_000, value=7))
duration_s = int(st.sidebar.number_input("Duration (s)", min_value=60, max_value=3600, value=900, step=60))

if st.sidebar.button("Regenerate case", use_container_width=True):
    st.session_state["case"] = generate_case(seed=seed, duration_s=duration_s)
    reset_prompt_session(st.session_state)
    st.session_state["domain_score_state"] = {}
    st.session_state["served_prompt_hashes"] = []
    st.session_state["last_alert"] = None
    st.session_state["paused"] = False

if "case" not in st.session_state:
    st.session_state["case"] = generate_case(seed=seed, duration_s=duration_s)

case = st.session_state["case"]

st.sidebar.header("Detection settings")
tau = float(st.sidebar.slider("Episode SII threshold (tau)", 0.05, 0.70, 0.35, 0.01))
tau_low = float(st.sidebar.slider("Episode release threshold (tau_low)", 0.02, 0.60, 0.26, 0.01))
k_start = int(st.sidebar.slider("Episode start persistence (k_start)", 1, 20, 5, 1))
m_end = int(st.sidebar.slider("Episode end persistence (m_end)", 1, 30, 8, 1))
calm_gate = float(st.sidebar.slider("Calm gate SII (dim overlay below)", 0.0, 0.50, 0.18, 0.01))

st.sidebar.header("Viewer")
hemisphere = st.sidebar.selectbox("Hemisphere view", ["BOTH", "LEFT", "RIGHT"], index=0)
view_mode = st.sidebar.selectbox("View mode", ["FSI overlay", "Glass shell"], index=0)
show_recent = st.sidebar.checkbox("Show recent event dots", value=True)


# -----------------------------
# Compute SII state using metrics.current_state + generate_case timeseries
# -----------------------------
ts = case.get("timeseries")
if not isinstance(ts, pd.DataFrame) or ts.empty:
    # Emergency fallback (shouldn't happen)
    sii_now = 0.0
    in_episode = False
    domains_impacted: List[str] = []
    now_t = 0
else:
    now_t = _safe_now_t(case)
    cs = current_state(ts, t_now=now_t, tau=tau, tau_low=tau_low, k_start=k_start, m_end=m_end)
    sii_now = float(cs.get("sii_now", 0.0))
    in_episode = bool(cs.get("in_episode", False))
    domains_impacted = list(cs.get("domains_impacted", []) or [])

append_sii_sample(st.session_state, float(sii_now), t_now=float(now_t))


# -----------------------------
# Cached brain mesh
# -----------------------------
@st.cache_resource
def _get_mesh():
    # include_lobes=True is important for lobe overlays
    return load_brain_mesh(include_lobes=True)

mesh = _get_mesh()


# -----------------------------
# Prompt generation (MedGemma) + anti-repeat hard guard
# -----------------------------
def generate_next_prompt() -> Optional[Dict[str, Any]]:
    profile = st.session_state.get("procedure_profile", {}) or {}
    history = st.session_state.get("prompt_history", []) or []

    context = {
        "sii_now": float(sii_now),
        "in_episode": bool(in_episode),
        "domains_impacted": domains_impacted,
        "avoid_prompt_hashes": st.session_state.get("served_prompt_hashes", [])[-25:],
    }

    p = next_prompt(profile, history, context)  # type: ignore
    if not isinstance(p, dict) or not p.get("prompt_text"):
        return None

    fp = _prompt_fingerprint(p)
    recent = set(st.session_state.get("served_prompt_hashes", [])[-25:])

    if fp in recent:
        # try a few times to force novelty (even if the model repeats)
        for _ in range(6):
            p2 = next_prompt(profile, history, {**context, "force_new": True})  # type: ignore
            if isinstance(p2, dict) and p2.get("prompt_text"):
                fp2 = _prompt_fingerprint(p2)
                if fp2 not in recent:
                    p, fp = p2, fp2
                    break

    # store fingerprint and set prompt (prompt_state increments prompt_counter)
    st.session_state.setdefault("served_prompt_hashes", []).append(fp)
    set_current_prompt(st.session_state, p)
    return p


# -----------------------------
# Layout
# -----------------------------
st.title("NeuroVistaFM — Brain-wide Functional Stability Viewer")
st.caption("Doctor Console (Phase 2) — procedure setup + adaptive prompts + scoring + warnings")

left, right = st.columns([1.55, 1.0], gap="large")


# -----------------------------
# RIGHT COLUMN: Workflow / setup / live controls
# -----------------------------
with right:
    st.subheader("Workflow")
    cA, cB = st.columns(2)
    with cA:
        if st.button("🛠️ Setup", use_container_width=True):
            st.session_state["phase"] = "SETUP"
    with cB:
        if st.button("▶ Start test", use_container_width=True):
            st.session_state["phase"] = "LIVE"

    if st.session_state["phase"] == "SETUP":
        profile = procedure_profile_ui(defaults=st.session_state.get("procedure_profile") or {})
        st.session_state["procedure_profile"] = profile

        st.info("Adjust settings + procedure profile. Then click **Start test**.")
        _write_tv_state(
            session_id,
            status="SETUP",
            prompt=None,
            sii_now=sii_now,
            in_episode=in_episode,
            domains_impacted=domains_impacted,
            paused=bool(st.session_state.get("paused", False)),
            prompt_counter=int(st.session_state.get("prompt_counter", 0)),
            note="Waiting for test start…",
        )

    else:
        # LIVE phase
        st.subheader("TV Preview (what the patient sees)")
        p = st.session_state.get("current_prompt")

        def render_tv_preview(prompt: Optional[Dict[str, Any]]):
            if not prompt:
                st.info("Click **Next prompt** to begin.")
                return
            st.markdown(
                f"<div style='font-size:34px;font-weight:950;line-height:1.12'>{prompt.get('prompt_text','')}</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"Domain: {prompt.get('domain','?')} • SII={sii_now:.3f} • {'episode' if in_episode else 'stable'}")

            ptype = str(prompt.get("prompt_type", "TEXT")).upper()
            payload = prompt.get("payload", {}) or {}
            if ptype == "COLOR":
                color = payload.get("color_hex", "#777777")
                st.markdown(
                    f"<div style='height:170px;border-radius:18px;border:1px solid rgba(255,255,255,0.14);background:{color};'></div>",
                    unsafe_allow_html=True,
                )
            elif ptype == "IMAGE":
                emoji = payload.get("emoji", "🦊")
                st.markdown(f"<div style='font-size:90px;margin-top:10px'>{emoji}</div>", unsafe_allow_html=True)

        render_tv_preview(p)

        st.markdown("---")
        st.subheader("Test Controls")
        c1, c2 = st.columns(2)
        with c1:
            next_btn = st.button("▶ Next prompt", use_container_width=True)
        with c2:
            pause_btn = st.button("⏸ Pause/Resume", use_container_width=True)

        end_btn = st.button("⏹ End test", use_container_width=True)
        if end_btn:
            st.session_state["phase"] = "SETUP"
            _write_tv_state(
                session_id,
                status="END",
                prompt=None,
                sii_now=sii_now,
                in_episode=in_episode,
                domains_impacted=domains_impacted,
                paused=True,
                prompt_counter=int(st.session_state.get("prompt_counter", 0)),
                note="Test ended by doctor.",
            )
            st.rerun()

        if pause_btn:
            st.session_state["paused"] = not st.session_state.get("paused", False)
            st.rerun()

        if next_btn and not st.session_state.get("paused", False):
            generate_next_prompt()
            st.rerun()

        st.subheader("Observer Scoring")
        st.caption("Enter=✅(2) • Space=⚠️(1) • Backspace=❌(0) • U=Undo • R=Reset • Esc=Pause")

        def score_and_advance(score_int: int):
            cur = st.session_state.get("current_prompt") or {}
            dom = cur.get("domain")
            if dom:
                st.session_state["domain_score_state"][dom] = int(score_int)

            # Log score
            log_score(st.session_state, int(score_int), t_now=float(now_t))

            # Alerts
            profile2 = st.session_state.get("procedure_profile", {}) or {}
            crit = profile2.get("critical_domains", ["LANG"])
            alert = detect_alert(
                st.session_state.get("responses_log", []),
                sii_now=float(sii_now),
                sii_series=st.session_state.get("sii_series", []),
                critical_domains=crit,
            )
            st.session_state["last_alert"] = alert

            # Generate synopsis (MedGemma)
            try:
                syn = make_synopsis(
                    profile2,
                    st.session_state.get("responses_log", []),
                    {"sii_now": float(sii_now), "in_episode": bool(in_episode), "domains_impacted": domains_impacted},
                    alert=alert,
                )
            except TypeError:
                syn = make_synopsis(
                    profile2,
                    st.session_state.get("responses_log", []),
                    {"sii_now": float(sii_now), "in_episode": bool(in_episode), "domains_impacted": domains_impacted},
                )
            st.session_state["last_synopsis"] = syn

            if not st.session_state.get("paused", False):
                generate_next_prompt()

        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("✅ Correct", use_container_width=True):
                score_and_advance(2)
                st.rerun()
        with b2:
            if st.button("⚠ Partial", use_container_width=True):
                score_and_advance(1)
                st.rerun()
        with b3:
            if st.button("❌ Incorrect", use_container_width=True):
                score_and_advance(0)
                st.rerun()

        alert = st.session_state.get("last_alert")
        if alert:
            sev = alert.get("severity", "warning")
            msg = alert.get("msg", "Alert")
            if sev == "high":
                st.error("⚠️ " + msg)
            elif sev == "warning":
                st.warning("⚠️ " + msg)
            else:
                st.info("ℹ️ " + msg)

        st.markdown("---")
        st.subheader("Intraop Synopsis")
        syn = st.session_state.get("last_synopsis")
        if isinstance(syn, dict):
            for line in syn.get("situation_summary", []) or []:
                st.markdown(f"- {line}")
            if syn.get("warning_explanation"):
                st.caption("Model note: " + str(syn["warning_explanation"]))
        else:
            st.info("Score at least one prompt to populate synopsis.")

        with st.expander("Safety + Risk Mitigation", expanded=False):
            st.markdown("- Decision support only; clinician interprets and leads care.")
            st.markdown("- Confounds: sedation, pain, anxiety, motor restriction, distraction.")
            st.markdown("- If repeated failures occur, pause and re-check readiness.")
            st.markdown("- Use persistence/hysteresis settings to reduce flicker.")

        with st.expander("Escalation", expanded=False):
            st.markdown("- If 3 consecutive incorrects in a critical domain → pause and confirm with low-load probes.")
            st.markdown("- If SII rising / episode onset → reduce task load or switch modality.")
            st.markdown("- Escalate to attending request: confirmatory mapping before proceeding.")

        with st.expander("Keyboard Controls + Button Mapping", expanded=False):
            st.markdown("- ✅ Correct: Enter")
            st.markdown("- ⚠ Partial: Space")
            st.markdown("- ❌ Incorrect: Backspace")
            st.markdown("- Pause/Resume: Esc")
            st.markdown("- Undo: U")
            st.markdown("- Reset: R")

        # Always push TV state in LIVE
        _write_tv_state(
            session_id,
            status="LIVE",
            prompt=st.session_state.get("current_prompt"),
            sii_now=sii_now,
            in_episode=in_episode,
            domains_impacted=domains_impacted,
            paused=bool(st.session_state.get("paused", False)),
            prompt_counter=int(st.session_state.get("prompt_counter", 0)),
            note="",
            alert=st.session_state.get("last_alert"),
        )


# -----------------------------
# LEFT COLUMN: Brain render
# -----------------------------
with left:
    st.subheader("3D Functional Stability Overlay")

    curp = st.session_state.get("current_prompt") or {}
    highlight_lobes = _expected_lobes_from_prompt(curp)

    # Determine color based on last score for current domain (if any)
    hs: Optional[int] = None
    d = curp.get("domain")
    if d:
        hs = st.session_state.get("domain_score_state", {}).get(d)

    # Pending/unscored prompt: show cyan overlay (brain_view_phase2 patched supports highlight_score=None by drawing cyan)
    fig = make_brain_figure(
        mesh=mesh,
        hemisphere=hemisphere,
        domain_scores=st.session_state.get("domain_score_state", {}),
        highlight_lobes=highlight_lobes,
        highlight_score=hs,  # None => pending cyan
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"SII={sii_now:.3f} • {'episode' if in_episode else 'stable'} • impacted: {', '.join(domains_impacted) if domains_impacted else 'none'}")

    # Keep TV alive even in SETUP if user scrolls
    if st.session_state["phase"] == "SETUP":
        _write_tv_state(
            session_id,
            status="SETUP",
            prompt=None,
            sii_now=sii_now,
            in_episode=in_episode,
            domains_impacted=domains_impacted,
            paused=bool(st.session_state.get("paused", False)),
            prompt_counter=int(st.session_state.get("prompt_counter", 0)),
            note="Waiting for test start…",
        )
