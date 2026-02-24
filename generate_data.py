from __future__ import annotations

import numpy as np
import pandas as pd

# Support both repo layouts:
# - flat: metrics.py at repo root
# - package: src/metrics.py
try:  # pragma: no cover
    from metrics import compute_sii_from_stress, StressSIIParams  # type: ignore
except Exception:  # pragma: no cover
    from src.metrics import compute_sii_from_stress, StressSIIParams  # type: ignore

DOMAINS = ["LANG", "VIS", "AUD", "MOT"]


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return int(max(lo, min(int(x), hi)))


def _sanitize_episode_window(episode_window: tuple[int, int], duration_s: int) -> tuple[int, int]:
    """Ensure (start,end) is ordered and fits within [0, duration_s]."""
    a, b = int(episode_window[0]), int(episode_window[1])
    if a > b:
        a, b = b, a
    a = _clamp_int(a, 0, max(0, duration_s - 1))
    b = _clamp_int(b, 0, max(0, duration_s))
    if b <= a:
        b = min(duration_s, a + max(10, duration_s // 12))
    return a, b


def _rolling_domain_fsi_from_events(
    events: pd.DataFrame,
    t_grid: np.ndarray,
    domains: list[str],
    window_s: int = 90,
    decay_s: float = 45.0,
) -> pd.DataFrame:
    """Build per-domain FSI time series from event stream.

    Returns dataframe with:
      t, FSI_LANG, FSI_VIS, FSI_AUD, FSI_MOT

    Interpretation:
      - Higher FSI means more recent instability signals in that domain.
      - Uses a decayed sum of "instability weight" per event in [0..1].
    """

    e = events.copy()
    if e.empty:
        # Return a valid (all-zero) FSI frame if there are no events
        out = {"t": t_grid.astype(int)}
        for d in domains:
            out[f"FSI_{d}"] = np.zeros(len(t_grid), dtype=float)
        return pd.DataFrame(out)

    # Instability weight per event:
    # - low accuracy increases weight
    # - longer RT increases weight (relative)
    # - error type adds a small bump
    # - stim adds a small bump (optional)
    e["t"] = e["t"].astype(float)

    # Normalize RT within domain for a fair RT penalty
    rt_norm = np.zeros(len(e), dtype=float)
    for d in domains:
        idx = e["domain"] == d
        if idx.any():
            rt = e.loc[idx, "rt_ms"].to_numpy(dtype=float)
            # robust RT scaling even if distribution is tight
            lo, hi = np.quantile(rt, 0.10), np.quantile(rt, 0.90)
            den = max(float(hi - lo), 1.0)
            rt_norm[idx.to_numpy()] = np.clip((rt - lo) / den, 0.0, 1.0)

    # Error bump
    err_bump = np.where(e["err_type"].astype(str).values != "none", 0.15, 0.0)

    # Base weight: accuracy + RT + error + stim
    w_inst = (
        0.70 * (1.0 - e["acc"].to_numpy(dtype=float))
        + 0.25 * rt_norm
        + err_bump
        + 0.05 * e["stim"].to_numpy(dtype=float)
    )
    w_inst = np.clip(w_inst, 0.0, 1.0)

    # Pre-split events by domain for speed
    ev_by_dom = {d: e[e["domain"] == d] for d in domains}
    w_by_dom = {d: w_inst[e["domain"].to_numpy() == d] for d in domains}

    out: dict[str, np.ndarray] = {"t": t_grid.astype(int)}
    for d in domains:
        out[f"FSI_{d}"] = np.zeros(len(t_grid), dtype=float)

    # O(T * D * events_in_window) loop — OK for demo sizes
    for i, t in enumerate(t_grid.astype(float)):
        t0 = t - float(window_s)
        for d in domains:
            ed = ev_by_dom[d]
            if ed.empty:
                continue

            tt = ed["t"].to_numpy(dtype=float)
            mask = (tt >= t0) & (tt <= t)
            if not mask.any():
                continue

            ages = t - tt[mask]
            w_time = np.exp(-ages / float(decay_s))
            w_inst_sel = w_by_dom[d][mask]

            # decayed sum → normalize into [0..1] with a soft saturating transform
            score = float(np.sum(w_time * w_inst_sel))
            out[f"FSI_{d}"][i] = 1.0 - np.exp(-score)  # saturates smoothly

    return pd.DataFrame(out)


def generate_synthetic_case(
    seed: int = 7,
    duration_s: int = 900,
    n_parcels: int = 120,
    event_rate_hz: float = 0.25,  # ~1 event every 4s
    stress_dt: int = 10,          # stress sample every 10s
    # --- optional knobs for realism ---
    episode_window: tuple[int, int] = (520, 650),
    fsi_window_s: int = 90,
    fsi_decay_s: float = 45.0,
    sii_params: StressSIIParams | None = None,
) -> dict:
    """Generate a deterministic synthetic case for the demo.

    Returns dict with:
      - events: DataFrame (event-level)
      - stress: DataFrame (stress-level, sampled every stress_dt)
      - timeseries: DataFrame (1 Hz) with t, SII, FSI_* columns
      - episode_truth: dict(start,end)
      - sii_components: DataFrame returned from compute_sii_from_stress

    Notes:
      - Designed to be "demo-realistic" (clear episode injection) while remaining
        lightweight enough for Streamlit/Kaggle.
      - Uses a scripted episode where LANG/AUD performance degrades and stress rises.
    """

    # --- basic parameter hygiene (prevents confusing runtime errors) ---
    duration_s = int(max(60, duration_s))
    n_parcels = int(max(10, n_parcels))
    event_rate_hz = float(max(0.01, event_rate_hz))
    stress_dt = int(max(1, stress_dt))
    ep_start, ep_end = _sanitize_episode_window(episode_window, duration_s)

    rng = np.random.default_rng(int(seed))

    # --- Events ---
    n_events = max(1, int(duration_s * event_rate_hz))
    t_events = np.sort(rng.uniform(0, duration_s, size=n_events))

    # keep parcel_id in [1..n_parcels] (cleaner than 0-index for overlays)
    parcel_ids = rng.integers(1, n_parcels + 1, size=n_events)

    domains = rng.choice(DOMAINS, size=n_events, p=[0.35, 0.20, 0.20, 0.25])

    # baseline performance by domain
    base_acc = {"LANG": 0.95, "VIS": 0.96, "AUD": 0.94, "MOT": 0.98}
    base_rt = {"LANG": 900, "VIS": 750, "AUD": 820, "MOT": 500}

    acc = np.array([np.clip(rng.normal(base_acc[d], 0.03), 0, 1) for d in domains])
    rt = np.array([np.clip(rng.normal(base_rt[d], 120), 150, 2500) for d in domains])

    stim = (rng.uniform(0, 1, size=n_events) < 0.35).astype(int)

    err_type = np.array(["none"] * n_events, dtype=object)
    err_prob = np.clip(0.03 + (1 - acc) * 0.8, 0, 0.6)
    err_mask = rng.uniform(0, 1, size=n_events) < err_prob
    err_choices = np.array(["semantic", "phonemic", "omission", "delay"], dtype=object)
    if err_mask.any():
        err_type[err_mask] = rng.choice(err_choices, size=int(err_mask.sum()))

    # --- Scripted instability episode ---
    in_ep = (t_events >= ep_start) & (t_events <= ep_end)

    # Apply domain-specific degradation during episode
    degrade = in_ep & np.isin(domains, ["LANG", "AUD"])
    if degrade.any():
        acc[degrade] = np.clip(acc[degrade] - rng.uniform(0.12, 0.25, size=int(degrade.sum())), 0, 1)
        rt[degrade] = np.clip(rt[degrade] + rng.uniform(200, 450, size=int(degrade.sum())), 150, 2500)

    # Slight global jitter during episode
    if in_ep.any():
        rt[in_ep] = np.clip(rt[in_ep] + rng.normal(0, 80, size=int(in_ep.sum())), 150, 2500)

    # --- Proximity (optional, simulated) ---
    prox_mm = np.clip(rng.normal(6.0, 2.0, size=n_events), 0.5, 15.0)

    # Task labels
    task_map = {
        "LANG": ["naming", "reading", "repetition"],
        "VIS": ["picture_id", "object_match"],
        "AUD": ["sound_id", "word_comp"],
        "MOT": ["finger_tap", "hand_squeeze"],
    }
    task = np.array([rng.choice(task_map[d]) for d in domains], dtype=object)

    events = pd.DataFrame(
        {
            "t": t_events.astype(float),
            "parcel_id": parcel_ids.astype(int),
            "domain": domains.astype(str),
            "task": task.astype(str),
            "stim": stim.astype(int),
            "acc": acc.astype(float),
            "rt_ms": rt.astype(float),
            "err_type": err_type.astype(str),
            "prox_mm": prox_mm.astype(float),
        }
    )

    # --- Stress stream ---
    t_stress = np.arange(0, duration_s + 1, stress_dt, dtype=float)

    hr0 = 82.0
    hrv0 = 38.0
    disf0 = 0.8

    hr = rng.normal(hr0, 3, size=len(t_stress))
    hrv = rng.normal(hrv0, 2.5, size=len(t_stress))
    disf = np.clip(rng.normal(disf0, 0.2, size=len(t_stress)), 0.1, 3.0)

    # Global RT variance proxy over last ~minute (approx from events)
    rt_var_global = np.zeros_like(t_stress, dtype=float)
    for i, tt in enumerate(t_stress):
        w = events[(events["t"] >= tt - 60) & (events["t"] <= tt)]
        if len(w) >= 3:
            rt_var_global[i] = float(np.var(w["rt_ms"].to_numpy()))
        else:
            rt_var_global[i] = float(np.var(events["rt_ms"].head(min(30, len(events))).to_numpy()))

    # Inject stress change aligned with episode
    stress_ep = (t_stress >= ep_start) & (t_stress <= ep_end)
    if stress_ep.any():
        hr[stress_ep] += rng.uniform(8, 16, size=int(stress_ep.sum()))
        hrv[stress_ep] -= rng.uniform(8, 14, size=int(stress_ep.sum()))
        disf[stress_ep] += rng.uniform(0.6, 1.3, size=int(stress_ep.sum()))
        rt_var_global[stress_ep] *= rng.uniform(1.3, 1.8, size=int(stress_ep.sum()))

    stress = pd.DataFrame(
        {
            "t": t_stress,
            "hr": hr.astype(float),
            "hrv_rmssd": np.clip(hrv, 5, None).astype(float),
            "disfluency": disf.astype(float),
            "rt_var_global": rt_var_global.astype(float),
        }
    )

    # --- Compute SII from stress (cohesive with metrics.py) ---
    if sii_params is None:
        sii_params = StressSIIParams(
            baseline_t_max=180,
            robust_baseline=True,
            ema_alpha=0.25,
            clip_z=6.0,
            clip_raw=10.0,
            # default weights tuned for clear-but-not-crazy episode
            w_hr=1.0,
            w_hrv=1.0,
            w_disfluency=1.2,
            w_rtvar=1.1,
        )

    sii_df = compute_sii_from_stress(stress, sii_params)

    # --- Timeseries (1 Hz): SII + domain FSI from events ---
    t_ts = np.arange(0, duration_s, 1, dtype=int)

    # Interpolate SII from stress sampling grid to 1 Hz
    sii_interp = np.interp(
        t_ts.astype(float),
        sii_df["t"].to_numpy(dtype=float),
        sii_df["SII"].to_numpy(dtype=float),
    )

    fsi_df = _rolling_domain_fsi_from_events(
        events=events,
        t_grid=t_ts,
        domains=DOMAINS,
        window_s=int(fsi_window_s),
        decay_s=float(fsi_decay_s),
    )

    timeseries = fsi_df.copy()
    timeseries["SII"] = sii_interp.astype(float)

    # reorder columns nicely
    timeseries = timeseries[["t", "SII"] + [f"FSI_{d}" for d in DOMAINS]]

    return {
        "events": events,
        "stress": stress,
        "timeseries": timeseries,
        "episode_truth": {"start": int(ep_start), "end": int(ep_end)},
        # optional: expose components for explainability
        "sii_components": sii_df,
    }


# ---------------------------------------------------------------------------
# Backwards/compatibility aliases expected by the Streamlit app.
# The app looks for one of:
#   - generate_synthetic_case
#   - generate_case
#   - generate_synthetic_data
# Keep these names stable so the submission runs across refactors.


def generate_case(**kwargs) -> dict:
    """Alias for generate_synthetic_case (kept for compatibility)."""
    return generate_synthetic_case(**kwargs)


def generate_synthetic_data(**kwargs) -> dict:
    """Alias for generate_synthetic_case (kept for compatibility)."""
    return generate_synthetic_case(**kwargs)
