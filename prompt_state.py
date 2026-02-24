"""
prompt_state.py
Phase 2: Streamlit session-state helpers for prompts + scoring logs.

This file keeps the app clean:
- current_prompt
- prompt_history (prompts shown)
- responses_log (scores given)
- sii_series (optional rolling SII samples)
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import time


def ensure_state(session_state: Any) -> None:
    """
    Initialize all required session keys.
    Pass st.session_state.
    """
    if "current_prompt" not in session_state:
        session_state["current_prompt"] = None

    if "prompt_history" not in session_state:
        session_state["prompt_history"] = []  # list of prompt dicts, in order shown

    if "responses_log" not in session_state:
        session_state["responses_log"] = []  # list of dicts {t, domain, score, prompt_text, ...}

    if "sii_series" not in session_state:
        session_state["sii_series"] = []  # list of dicts {t, sii}

    if "prompt_counter" not in session_state:
        session_state["prompt_counter"] = 0

    if "last_prompt_ts" not in session_state:
        session_state["last_prompt_ts"] = None


def set_current_prompt(session_state: Any, prompt: Dict[str, Any]) -> None:
    """
    Set current prompt and append to prompt_history.
    """
    session_state["current_prompt"] = prompt
    session_state["prompt_history"].append(prompt)
    session_state["prompt_counter"] = int(session_state.get("prompt_counter", 0)) + 1
    session_state["last_prompt_ts"] = time.time()


def log_score(session_state: Any, score: int, *, t_now: Optional[float] = None, note: str = "") -> Dict[str, Any]:
    """
    Log clinician scoring for the current prompt.
    score: 2=Correct, 1=Partial, 0=Incorrect
    """
    prompt = session_state.get("current_prompt") or {}
    entry = {
        "t": float(t_now) if t_now is not None else time.time(),
        "domain": prompt.get("domain", "?"),
        "score": int(score),
        "prompt_text": prompt.get("prompt_text", ""),
        "prompt_source": prompt.get("source", ""),
        "note": note,
    }
    session_state["responses_log"].append(entry)
    return entry


def undo_last_score(session_state: Any) -> Optional[Dict[str, Any]]:
    """
    Pop last response log entry, if any.
    """
    log = session_state.get("responses_log", [])
    if not log:
        return None
    return log.pop()


def reset_prompt_session(session_state: Any) -> None:
    """
    Clears prompt session state (prompts + responses) but does NOT touch procedure profile.
    """
    session_state["current_prompt"] = None
    session_state["prompt_history"] = []
    session_state["responses_log"] = []
    session_state["sii_series"] = []
    session_state["prompt_counter"] = 0
    session_state["last_prompt_ts"] = None


def append_sii_sample(session_state: Any, sii: float, *, t_now: Optional[float] = None) -> None:
    """
    Append SII sample into sii_series for trend display / alerting.
    """
    session_state["sii_series"].append({
        "t": float(t_now) if t_now is not None else time.time(),
        "sii": float(sii),
    })


def get_recent_scores(session_state: Any, n: int = 10) -> List[Dict[str, Any]]:
    """
    Convenience: last N response logs.
    """
    log = session_state.get("responses_log", [])
    return log[-n:] if len(log) > n else log[:]