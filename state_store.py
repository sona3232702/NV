
"""state_store.py
Simple file-based shared state between Streamlit apps (doctor console + TV display).

How it works:
- Doctor app writes the "TV payload" (current prompt + minimal status) to a JSON file.
- TV app reads that JSON file and renders it.

This avoids trying to share Streamlit session_state across processes.
"""

from __future__ import annotations
from typing import Any, Dict, Optional
import json
import os
import time
import tempfile

def _safe_session_id(session_id: str) -> str:
    # keep only safe filename chars
    keep = []
    for ch in (session_id or "default"):
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
    return "".join(keep) or "default"

def state_path(session_id: str = "default") -> str:
    sid = _safe_session_id(session_id)
    return os.path.join(os.path.dirname(__file__), f".neurovista_shared_state_{sid}.json")

def read_state(session_id: str = "default") -> Dict[str, Any]:
    path = state_path(session_id)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        # if partial write / corrupted, return empty and let doctor rewrite next tick
        return {}

def write_state(session_id: str, data: Dict[str, Any]) -> None:
    """Atomic write to avoid half-written JSON."""
    path = state_path(session_id)
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)

    # add/update server timestamp
    payload = dict(data)
    payload["_server_ts"] = time.time()

    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_nv_", suffix=".json", dir=directory, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
