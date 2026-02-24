from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


# ============================================================
# Utilities
# ============================================================

def _safe_std(x: np.ndarray, eps: float) -> float:
    sd = float(np.nanstd(x, ddof=1)) if len(x) > 1 else 0.0
    return max(sd, eps)

def _mad_sigma(x: np.ndarray, eps: float) -> float:
    """
    Robust sigma estimator using MAD scaled by 1.4826 (normal-consistent).
    """
    med = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - med)))
    sigma = 1.4826 * mad
    return max(float(sigma), eps)

def _zscore(x: np.ndarray, mu: float, sigma: float, eps: float = 1e-6) -> np.ndarray:
    sigma = max(float(sigma), eps)
    return (x - float(mu)) / sigma

def _clip(x: np.ndarray, clip_val: float) -> np.ndarray:
    if clip_val is None or clip_val <= 0:
        return x
    return np.clip(x, -clip_val, clip_val)


# ============================================================
# Stress → SII
# ============================================================

@dataclass
class StressSIIParams:
    """
    Parameters for Stress–Instability Index (SII) computed from a stress stream.
    """
    baseline_t_max: int = 180   # baseline window for stress normalization [0..baseline_t_max]
    eps: float = 1e-6

    # robust baseline option (winner-grade)
    robust_baseline: bool = True

    # z-score clipping (prevents blow-ups / sigmoid saturation)
    clip_z: float = 6.0
    clip_raw: float = 10.0

    # feature weights (tune as needed)
    w_hr: float = 1.0
    w_hrv: float = 1.0              # applied to inverted HRV z-score
    w_disfluency: float = 1.2
    w_rtvar: float = 1.1

    # sigmoid mapping for bounding [0,1]
    sigmoid_center: float = 0.0
    sigmoid_scale: float = 1.0

    # smoothing
    ema_alpha: float = 0.25         # higher = less smoothing (more responsive)

def compute_sii_from_stress(
    stress: pd.DataFrame,
    params: StressSIIParams | None = None,
) -> pd.DataFrame:
    """
    Compute Stress–Instability Index (SII) from stress stream.

    Assumes stress has columns:
      t, hr, hrv_rmssd, disfluency, rt_var_global

    Returns DataFrame with columns:
      t, SII, sii_raw,
      z_hr, z_hrv_inv, z_disfluency, z_rt_var_global,
      baseline_* provenance fields
    """
    if params is None:
        params = StressSIIParams()

    required = ["t", "hr", "hrv_rmssd", "disfluency", "rt_var_global"]
    missing = [c for c in required if c not in stress.columns]
    if missing:
        raise ValueError(f"stress is missing columns: {missing}")

    s = stress.copy().sort_values("t")
    s["t"] = s["t"].astype(int)

    base = s[s["t"] <= int(params.baseline_t_max)]
    if base.empty:
        raise ValueError("Baseline window empty for stress. Increase baseline_t_max or check stress['t'].")

    def _mu_sd(col: str) -> tuple[float, float]:
        x = base[col].to_numpy(dtype=float)
        mu = float(np.nanmedian(x)) if params.robust_baseline else float(np.nanmean(x))
        sd = _mad_sigma(x, params.eps) if params.robust_baseline else _safe_std(x, params.eps)
        return mu, sd

    mu_hr,  sd_hr  = _mu_sd("hr")
    mu_hrv, sd_hrv = _mu_sd("hrv_rmssd")
    mu_dis, sd_dis = _mu_sd("disfluency")
    mu_rtv, sd_rtv = _mu_sd("rt_var_global")

    # z-scores with correct directionality
    z_hr = _zscore(s["hr"].to_numpy(float), mu_hr, sd_hr, params.eps)                 # HR ↑ worse
    z_hrv = _zscore(s["hrv_rmssd"].to_numpy(float), mu_hrv, sd_hrv, params.eps)
    z_hrv_inv = -z_hrv                                                                  # HRV ↓ worse
    z_dis = _zscore(s["disfluency"].to_numpy(float), mu_dis, sd_dis, params.eps)      # ↑ worse
    z_rtv = _zscore(s["rt_var_global"].to_numpy(float), mu_rtv, sd_rtv, params.eps)   # ↑ worse

    # clip per-feature z to prevent spikes
    z_hr = _clip(z_hr, params.clip_z)
    z_hrv_inv = _clip(z_hrv_inv, params.clip_z)
    z_dis = _clip(z_dis, params.clip_z)
    z_rtv = _clip(z_rtv, params.clip_z)

    # raw instability score
    sii_raw = (
        params.w_hr * z_hr
        + params.w_hrv * z_hrv_inv
        + params.w_disfluency * z_dis
        + params.w_rtvar * z_rtv
    )
    sii_raw = _clip(sii_raw, params.clip_raw)

    # sigmoid -> [0,1]
    x = (sii_raw - float(params.sigmoid_center)) / max(float(params.sigmoid_scale), params.eps)
    sii = 1.0 / (1.0 + np.exp(-x))

    # EMA smoothing
    sii_smooth = np.empty_like(sii)
    a = float(params.ema_alpha)
    sii_smooth[0] = sii[0]
    for i in range(1, len(sii)):
        sii_smooth[i] = a * sii[i] + (1.0 - a) * sii_smooth[i - 1]

    out = pd.DataFrame({
        "t": s["t"].to_numpy(int),
        "SII": sii_smooth.astype(float),
        "sii_raw": sii_raw.astype(float),
        "z_hr": z_hr.astype(float),
        "z_hrv_inv": z_hrv_inv.astype(float),
        "z_disfluency": z_dis.astype(float),
        "z_rt_var_global": z_rtv.astype(float),

        # provenance: baseline stats used
        "baseline_mu_hr": mu_hr, "baseline_sd_hr": sd_hr,
        "baseline_mu_hrv": mu_hrv, "baseline_sd_hrv": sd_hrv,
        "baseline_mu_disfluency": mu_dis, "baseline_sd_disfluency": sd_dis,
        "baseline_mu_rtvar": mu_rtv, "baseline_sd_rtvar": sd_rtv,
        "robust_baseline": bool(params.robust_baseline),
        "clip_z": float(params.clip_z),
        "clip_raw": float(params.clip_raw),
    })
    return out


# ============================================================
# Domain baseline from early stable window
# ============================================================

@dataclass
class BaselineStats:
    mu: float
    sigma: float
    n: int
    robust: bool

def compute_domain_baseline(
    ts: pd.DataFrame,
    baseline_t_max: int = 180,
    sii_col: str = "SII",
    stable_tau: float = 0.20,
    robust: bool = True,
) -> dict[str, BaselineStats]:
    """
    Compute baseline per domain using early stable window [0, baseline_t_max].

    Strategy:
      - Consider samples where t <= baseline_t_max
      - Prefer stable samples where SII <= stable_tau
      - If no stable samples exist, fall back to all samples in the window

    Expects domain columns in ts: FSI_LANG, FSI_AUD, ...
    Returns dict: domain -> BaselineStats(mu, sigma, n, robust)
    """
    w = ts[ts["t"] <= int(baseline_t_max)].copy()
    if w.empty:
        raise ValueError("Baseline window empty. Check baseline_t_max vs ts range.")

    stable = w[w[sii_col] <= float(stable_tau)] if sii_col in w.columns else w
    use = stable if not stable.empty else w

    dom_cols = [c for c in ts.columns if c.startswith("FSI_")]
    if not dom_cols:
        raise ValueError("No domain columns found (expected columns starting with 'FSI_').")

    out: dict[str, BaselineStats] = {}
    for c in dom_cols:
        dom = c.replace("FSI_", "")
        vals = use[c].to_numpy(dtype=float)

        mu = float(np.nanmedian(vals)) if robust else float(np.nanmean(vals))
        sigma = _mad_sigma(vals, 1e-6) if robust else max(float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 1e-6, 1e-6)
        out[dom] = BaselineStats(mu=mu, sigma=float(sigma), n=int(len(vals)), robust=bool(robust))
    return out


# ============================================================
# Parcel × domain FSI over sliding event window
# ============================================================

def compute_parcel_domain_fsi(
    events: pd.DataFrame,
    t_grid: np.ndarray,
    domains: list[str],
    n_parcels: int,
    window_s: int = 240,
    decay_s: float = 120.0,
    domain_focus: list[str] | None = None,
    focus_boost: float = 1.6,
    nonfocus_weight: float = 0.7,
    baseline: dict[str, BaselineStats] | None = None,
    # NEW: provide an absolute normalizer so you can show "calm looks calm"
    abs_normalizer: float | None = None,
) -> pd.DataFrame:
    """
    Compute parcel×domain FSI over sliding event window per parcel/domain.

    events must have columns: t, parcel_id, domain

    Returns DataFrame columns:
      t, parcel_id, domain,
      FSI_norm, FSI_abs, FSI, raw_score, n_events,
      w_time_sum, w_dom_sum,
      baseline_mu, baseline_sigma, z

    Notes:
      - raw_score is the unnormalized weighted sum.
      - FSI_norm is normalized within (t, domain) by max raw_score across parcels in that domain at that time.
      - FSI_abs is scaled using abs_normalizer if provided; else it uses raw_score (unscaled).
      - FSI is kept as an alias of FSI_norm for backward compatibility.
      - If baseline provided for domain, z = (FSI_norm - mu)/sigma using baseline stats.
    """
    if domain_focus is None:
        domain_focus = []

    cols = ["t", "parcel_id", "domain"]
    for c in cols:
        if c not in events.columns:
            raise ValueError(f"events missing required column '{c}'")

    if events.empty:
        return pd.DataFrame(columns=[
            "t", "parcel_id", "domain",
            "FSI_norm", "FSI_abs", "FSI", "raw_score", "n_events",
            "w_time_sum", "w_dom_sum",
            "baseline_mu", "baseline_sigma", "z",
        ])

    ev = events.copy()
    ev["t"] = ev["t"].astype(int)
    ev["parcel_id"] = ev["parcel_id"].astype(int)
    ev["domain"] = ev["domain"].astype(str)

    # filter known parcels/domains
    ev = ev[(ev["parcel_id"] >= 1) & (ev["parcel_id"] <= int(n_parcels))]
    ev = ev[ev["domain"].isin(domains)]

    rows = []
    t_grid = np.asarray(t_grid, dtype=int)

    for t in t_grid:
        t0 = int(max(0, int(t) - int(window_s)))
        w = ev[(ev["t"] >= t0) & (ev["t"] <= int(t))]
        if w.empty:
            continue

        for dom in domains:
            wd = w[w["domain"] == dom]
            if wd.empty:
                continue

            age = (int(t) - wd["t"].to_numpy(dtype=float))
            w_time = np.exp(-age / float(decay_s))

            w_dom = np.full_like(w_time, float(nonfocus_weight), dtype=float)
            if dom in domain_focus:
                w_dom *= float(focus_boost)

            pids = wd["parcel_id"].to_numpy(dtype=int)

            raw = {}
            counts = {}
            w_time_sum = {}
            w_dom_sum = {}

            for pid, wt, wdm in zip(pids, w_time, w_dom):
                raw[pid] = raw.get(pid, 0.0) + float(wt * wdm)
                counts[pid] = counts.get(pid, 0) + 1
                w_time_sum[pid] = w_time_sum.get(pid, 0.0) + float(wt)
                w_dom_sum[pid] = w_dom_sum.get(pid, 0.0) + float(wdm)

            max_raw = max(raw.values()) if raw else 0.0
            if max_raw <= 0:
                continue

            # abs scaling: either raw_score itself, or scaled to [0..1] using abs_normalizer
            if abs_normalizer is not None and abs_normalizer > 0:
                abs_den = float(abs_normalizer)
            else:
                abs_den = None

            for pid, raw_score in raw.items():
                fsi_norm = float(raw_score / max_raw)

                if abs_den is None:
                    fsi_abs = float(raw_score)  # true absolute magnitude (not bounded)
                else:
                    fsi_abs = float(min(raw_score / abs_den, 1.0))  # bounded for display

                if baseline and dom in baseline:
                    mu = baseline[dom].mu
                    sigma = baseline[dom].sigma
                else:
                    mu = 0.0
                    sigma = 1.0

                z = float((fsi_norm - mu) / sigma) if sigma > 0 else 0.0

                rows.append({
                    "t": int(t),
                    "parcel_id": int(pid),
                    "domain": dom,

                    "FSI_norm": fsi_norm,
                    "FSI_abs": fsi_abs,
                    "FSI": fsi_norm,  # backward compatible alias

                    "raw_score": float(raw_score),
                    "n_events": int(counts[pid]),
                    "w_time_sum": float(w_time_sum[pid]),
                    "w_dom_sum": float(w_dom_sum[pid]),
                    "baseline_mu": float(mu),
                    "baseline_sigma": float(sigma),
                    "z": z,
                })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["t", "domain", "parcel_id"]).reset_index(drop=True)
    return out


# ============================================================
# Episode detection: hysteresis + persistence + summaries
# ============================================================

def detect_episodes_hysteresis(
    ts: pd.DataFrame,
    sii_col: str = "SII",
    tau: float = 0.35,
    tau_low: float | None = None,
    k_start: int = 5,
    m_end: int = 8,
) -> list[tuple[int, int]]:
    """
    Episode start when SII > tau for k consecutive samples.
    End when SII < tau_low for m consecutive samples.

    Returns list of (t_start, t_end).

    If tau_low is None, uses tau_low = tau * 0.85 (default hysteresis).
    """
    if sii_col not in ts.columns:
        raise ValueError(f"ts missing '{sii_col}' column")

    if tau_low is None:
        tau_low = float(tau) * 0.85

    t = ts["t"].to_numpy(dtype=int)
    sii = ts[sii_col].to_numpy(dtype=float)

    episodes: list[tuple[int, int]] = []
    in_ep = False
    start_t = None

    above_run = 0
    below_run = 0

    for i in range(len(sii)):
        if not in_ep:
            if sii[i] > float(tau):
                above_run += 1
            else:
                above_run = 0

            if above_run >= int(k_start):
                start_idx = i - int(k_start) + 1
                start_t = int(t[start_idx])
                in_ep = True
                below_run = 0
        else:
            if sii[i] < float(tau_low):
                below_run += 1
            else:
                below_run = 0

            if below_run >= int(m_end):
                end_t = int(t[i])
                episodes.append((int(start_t), int(end_t)))
                in_ep = False
                start_t = None
                above_run = 0

    if in_ep and start_t is not None:
        episodes.append((int(start_t), int(t[-1])))

    return episodes

def summarize_episodes(
    ts: pd.DataFrame,
    episodes: list[tuple[int, int]],
    sii_col: str = "SII",
    tau: float = 0.35,
) -> list[dict]:
    """
    Winner-friendly episode summaries for explainability:
      start, end, duration, mean_sii, peak_sii, peak_t, confidence
    """
    out = []
    if not episodes:
        return out

    for a, b in episodes:
        seg = ts[(ts["t"] >= a) & (ts["t"] <= b)]
        if seg.empty:
            continue
        sii = seg[sii_col].to_numpy(dtype=float)
        t = seg["t"].to_numpy(dtype=int)

        peak_idx = int(np.argmax(sii))
        peak_sii = float(sii[peak_idx])
        peak_t = int(t[peak_idx])

        mean_sii = float(np.mean(sii))
        duration = int(b - a)

        # simple confidence: how far above tau the peak goes + mean above tau
        conf = float(0.5 * max(0.0, peak_sii - tau) + 0.5 * max(0.0, mean_sii - tau))
        conf = float(np.clip(conf / 0.5, 0.0, 1.0))  # scale into [0,1] for UI

        out.append({
            "t_start": int(a),
            "t_end": int(b),
            "duration_s": duration,
            "mean_sii": mean_sii,
            "peak_sii": peak_sii,
            "peak_t": peak_t,
            "confidence": conf,
        })
    return out


# ============================================================
# Convenience: current state at time t_now
# ============================================================

def current_state(
    ts: pd.DataFrame,
    t_now: int,
    tau: float,
    tau_low: float | None = None,
    k_start: int = 5,
    m_end: int = 8,
) -> dict:
    """
    Returns:
      t_now, sii_now, in_episode, episodes,
      domain_scores (sorted list of (domain, score)),
      domains_impacted (top 1-3 with score threshold),
      episode_summaries (rich list of dicts)
    """
    t_now = int(np.clip(int(t_now), int(ts["t"].min()), int(ts["t"].max())))
    row = ts.loc[ts["t"] == t_now].iloc[0]
    sii_now = float(row["SII"]) if "SII" in ts.columns else float("nan")

    episodes = detect_episodes_hysteresis(ts, tau=tau, tau_low=tau_low, k_start=k_start, m_end=m_end)
    in_episode = any(a <= t_now <= b for a, b in episodes)

    dom_cols = [c for c in ts.columns if c.startswith("FSI_")]
    dom_scores = [(c.replace("FSI_", ""), float(row[c])) for c in dom_cols]
    dom_scores.sort(key=lambda x: x[1], reverse=True)

    domains_impacted = [d for d, s in dom_scores if s >= 0.12][:3]
    if not domains_impacted and dom_scores:
        domains_impacted = [dom_scores[0][0]]

    episode_summaries = summarize_episodes(ts, episodes, tau=tau) if episodes else []

    return {
        "t_now": t_now,
        "sii_now": sii_now,
        "in_episode": bool(in_episode),
        "episodes": episodes,
        "episode_summaries": episode_summaries,
        "domain_scores": dom_scores,
        "domains_impacted": domains_impacted,
    }