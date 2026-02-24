
"""tv_display.py (v2)
TV display paired with doctor_console.py via state_store.py.
"""

from __future__ import annotations
import streamlit as st
from state_store import read_state

import time as _time

STALE_TTL_S = 6.0  # if no update from doctor console within this window, TV shows 'waiting'

st.set_page_config(page_title="NeuroVistaFM — TV Display", layout="wide")

st.sidebar.header("Sharing")
session_id = st.sidebar.text_input("Session ID (must match doctor console)", value="default")
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
refresh_s = st.sidebar.slider("Refresh interval (seconds)", 0.5, 5.0, 1.0, 0.5)

data = read_state(session_id)

# If the doctor console hasn't written state recently, don't show a stale prompt.
ts = data.get('_server_ts')
stale = True
try:
    if ts is not None and (_time.time() - float(ts)) <= STALE_TTL_S:
        stale = False
except Exception:
    stale = True

# IMPORTANT: if the session already ended, we still want to show the end summary even if updates stop.
ended_flag = bool(data.get("session_ended", False)) or (data.get("status") in ("END", "ENDED"))

if stale and (not ended_flag):
    data = {"status": "DISCONNECTED", "note": "Waiting for doctor console…"}

status = data.get("status", "SETUP")

def summarize_sii(series, thr1=0.35, thr2=0.50):
    """Summarize SII/SSI time-series from doctor console.
    `series` expected as list of dicts like {"t": <float>, "sii": <float>} or {"t":..., "sii_now":...}.
    """
    if not series:
        return None
    # normalize keys
    pts = []
    for p in series:
        if not isinstance(p, dict):
            continue
        t = p.get("t")
        v = p.get("sii")
        if v is None:
            v = p.get("sii_now")
        if t is None or v is None:
            continue
        try:
            pts.append((float(t), float(v)))
        except Exception:
            pass
    if len(pts) < 2:
        return None
    pts.sort(key=lambda x: x[0])
    t0, t1 = pts[0][0], pts[-1][0]
    duration = max(0.0, t1 - t0)
    vals = [v for _, v in pts]
    mean_v = sum(vals) / len(vals)
    max_v = max(vals)

    time_above_1 = 0.0
    time_above_2 = 0.0
    spikes_1 = 0
    spikes_2 = 0

    for i in range(1, len(pts)):
        dt = max(0.0, pts[i][0] - pts[i-1][0])
        prev, cur = pts[i-1][1], pts[i][1]
        if cur >= thr1:
            time_above_1 += dt
        if cur >= thr2:
            time_above_2 += dt
        if prev < thr1 <= cur:
            spikes_1 += 1
        if prev < thr2 <= cur:
            spikes_2 += 1

    return {
        "duration_s": duration,
        "mean": mean_v,
        "max": max_v,
        "thr1": thr1,
        "thr2": thr2,
        "time_above_thr1_s": time_above_1,
        "time_above_thr2_s": time_above_2,
        "spikes_thr1": spikes_1,
        "spikes_thr2": spikes_2,
        "n": len(pts),
    }

st.markdown("<div style='font-size:44px; font-weight:950; margin-top:6px;'>Patient Prompt</div>", unsafe_allow_html=True)
st.caption(f"Session: {session_id} • Status: {status}")

# Show last update age (helps debug sync)
ts = data.get('_server_ts')
if ts is not None:
    try:
        import time as _time
        st.caption(f"Last update: {_time.time()-float(ts):.1f}s ago")
    except Exception:
        pass


if status in ("END", "ENDED"):
    st.success(data.get("note", "Test ended."))
    series = data.get("sii_series", []) or data.get("ssi_history", []) or []
    summary = summarize_sii(series)
    st.markdown("### SSI Summary")
    if not summary:
        st.info("No SSI data captured for this session.")
    else:
        st.metric("Duration (min)", round(summary["duration_s"] / 60.0, 1))
        st.metric("Mean SSI", round(summary["mean"], 3))
        st.metric("Max SSI", round(summary["max"], 3))
        st.write(f'Time SSI ≥ {summary["thr1"]}: {round(summary["time_above_thr1_s"], 1)} s (spikes: {summary["spikes_thr1"]})')
        st.write(f'Time SSI ≥ {summary["thr2"]}: {round(summary["time_above_thr2_s"], 1)} s (spikes: {summary["spikes_thr2"]})')
        st.caption(f'Samples: {summary["n"]}')
else:
    prompt = data.get("prompt")
    sii_now = float(data.get("sii_now", 0.0))
    in_episode = bool(data.get("in_episode", False))
    paused = bool(data.get("paused", False))
    ctr = int(data.get("prompt_counter", 0))

    st.caption(
        f"Prompt #{ctr} • SII={sii_now:.3f} • {'EPISODE' if in_episode else 'stable'} • "
        f"{'PAUSED' if paused else 'running'}"
    )

    if not prompt:
        st.info(data.get("note", "Waiting for doctor console…"))
    else:
        ptype = prompt.get("prompt_type", "TEXT")
        text = prompt.get("prompt_text", "")
        payload = prompt.get("payload", {}) or {}

        if ptype == "COLOR":
            color = payload.get("color_hex", "#777777")
            st.markdown(
                f"""<div style="height:520px;border-radius:30px;border:2px solid rgba(255,255,255,0.14);
                background:{color};display:flex;align-items:center;justify-content:center;text-align:center;padding:28px;">
                <div style="font-size:62px;font-weight:950;color:rgba(0,0,0,0.78);line-height:1.08;">{text}</div>
                </div>""",
                unsafe_allow_html=True
            )
        elif ptype == "IMAGE":
            emoji = payload.get("emoji", "🦊")
            st.markdown(
                f"""<div style="height:520px;border-radius:30px;border:2px solid rgba(255,255,255,0.14);
                display:flex;align-items:center;justify-content:center;flex-direction:column;text-align:center;padding:28px;">
                <div style="font-size:140px">{emoji}</div>
                <div style="font-size:58px;font-weight:950;line-height:1.08;margin-top:10px;">{text}</div>
                </div>""",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""<div style="height:520px;border-radius:30px;border:2px solid rgba(255,255,255,0.14);
                display:flex;align-items:center;justify-content:center;text-align:center;padding:28px;">
                <div style="font-size:62px;font-weight:950;line-height:1.08;">{text}</div>
                </div>""",
                unsafe_allow_html=True
            )

def _rerun():
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

# Smooth rerun refresh (no full browser reload)
if auto_refresh:
    import time as _time
    _time.sleep(float(refresh_s))
    _rerun()
