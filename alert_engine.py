"""
alert_engine.py
Phase 2: Deterministic anomaly / "answers seem off" detector.

Inputs:
- responses_log: list of dicts {t, domain, score, prompt_text, ...}
  score: 2=Correct, 1=Partial, 0=Incorrect
- sii_now: float
- sii_series: optional list of dicts {t, sii}

Output:
- None, or an alert dict:
  { severity: "info"|"warning"|"high", code: str, msg: str, evidence: dict }
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple


def _mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))


def _recent(responses: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    return responses[-n:] if len(responses) >= n else responses[:]


def detect_alert(
    responses_log: List[Dict[str, Any]],
    sii_now: float,
    sii_series: Optional[List[Dict[str, Any]]] = None,
    *,
    critical_domains: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Main detector.
    """
    critical_domains = critical_domains or ["LANG"]

    if not responses_log:
        return None

    # --- Rule 1: two consecutive incorrect in same domain (especially critical) ---
    if len(responses_log) >= 2:
        a = responses_log[-2]
        b = responses_log[-1]
        if a.get("domain") == b.get("domain") and a.get("score") == 0 and b.get("score") == 0:
            dom = b.get("domain", "?")
            sev = "high" if dom in critical_domains else "warning"
            return {
                "severity": sev,
                "code": "TWO_CONSEC_FAIL",
                "msg": f"Two consecutive failures in {dom}. Consider re-baseline or confound check.",
                "evidence": {"last2": [a, b]},
            }

    # --- Rule 2: worsening trend across last 3 prompts (mean drop) ---
    # We compute a simple rolling mean drift: last3 vs previous3 (if exists)
    if len(responses_log) >= 6:
        prev3 = _recent(responses_log[:-3], 3)
        last3 = _recent(responses_log, 3)
        prev_mean = _mean([r.get("score", 0) for r in prev3])
        last_mean = _mean([r.get("score", 0) for r in last3])

        if last_mean <= prev_mean - 0.75:  # substantial drop
            return {
                "severity": "warning",
                "code": "THREE_PROMPT_DRIFT",
                "msg": f"Performance drift detected (mean {prev_mean:.2f} → {last_mean:.2f}).",
                "evidence": {"prev3": prev3, "last3": last3},
            }

    # --- Rule 3: low recent mean score (general off-ness) ---
    last4 = _recent(responses_log, 4)
    mean4 = _mean([r.get("score", 0) for r in last4])
    if len(last4) >= 4 and mean4 <= 0.75:
        return {
            "severity": "warning",
            "code": "LOW_RECENT_MEAN",
            "msg": f"Recent responses look off (last4 mean={mean4:.2f}). Consider fatigue/confound check.",
            "evidence": {"last4": last4},
        }

    # --- Rule 4: SII high + degraded recent performance (stronger signal) ---
    # If SII is above ~episode threshold and recent mean <= 1.0, raise high alert.
    if sii_now >= 0.35 and len(last4) >= 3:
        mean3 = _mean([r.get("score", 0) for r in _recent(responses_log, 3)])
        if mean3 <= 1.0:
            return {
                "severity": "high",
                "code": "HIGH_SII_AND_DEGRADED",
                "msg": f"High instability (SII={sii_now:.2f}) with degraded performance (last3 mean={mean3:.2f}).",
                "evidence": {"sii_now": sii_now, "last3": _recent(responses_log, 3)},
            }

    # --- Rule 5: SII rising quickly (optional) ---
    if sii_series and len(sii_series) >= 6:
        last = sii_series[-1]["sii"]
        prev = sii_series[-6]["sii"]
        if (last - prev) >= 0.12 and sii_now >= 0.25:
            return {
                "severity": "info",
                "code": "SII_RISING",
                "msg": f"SII rising (Δ≈{(last-prev):.2f} over recent window). Monitor for confounds.",
                "evidence": {"sii_prev": prev, "sii_now": last},
            }

    return None