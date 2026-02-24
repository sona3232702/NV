"""
medgemma_client.py
Phase 2: HAI-DEF (MedGemma) adapter for prompt planning + next-prompt generation + synopsis.

- Starts in STUB mode so the app always runs.
- Later, set USE_MEDGEMMA = True to enable Hugging Face inference.

Design goal:
Your Streamlit UI never calls HF directly — it calls this file.
That keeps integration stable and competition-proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import os
import random
import time


# -----------------------------
# Toggle: STUB vs REAL MedGemma
# -----------------------------
USE_MEDGEMMA = False  # <-- keep False for now (safe). Flip to True when ready.

MODEL_ID = os.environ.get("MEDGEMMA_MODEL_ID", "google/medgemma-1.5-4b-it")


# -----------------------------
# Lightweight prompt structures
# -----------------------------
@dataclass
class PromptItem:
    domain: str
    prompt_text: str
    title: str = ""
    expected: str = ""
    rationale: str = ""


# -----------------------------------
# STUB prompt bank (always available)
# -----------------------------------
_STUB_BANK: Dict[str, List[str]] = {
    "LANG": [
        "Name this object.",
        "Repeat: 'No ifs, ands, or buts.'",
        "Point to the picture I say.",
        "Say the days of the week.",
        "What does a key do?",
    ],
    "MOT": [
        "Tap your fingers for 5 seconds.",
        "Open and close your hand.",
        "Lift your right thumb.",
        "Touch your nose, then your ear.",
        "Hold your arms out straight.",
    ],
    "VIS": [
        "Point to the triangle.",
        "Find the circle.",
        "Match the two identical shapes.",
        "Which one is larger?",
        "Point to the red square.",
    ],
    "AUD": [
        "Repeat the word I say.",
        "Point to the picture I say.",
        "Do you hear a high or low tone?",
        "Repeat these numbers: 7-2-9.",
        "Say the word 'hospital'.",
    ],
    "EXEC": [
        "Count backwards from 20.",
        "Name the months backwards.",
        "Repeat these numbers: 4-1-8.",
        "Clap twice, then tap once.",
        "Spell 'WORLD' backwards.",
    ],
}


def _anti_repeat(history: List[Dict[str, Any]], candidates: List[str], k: int = 6) -> List[str]:
    recent = [h.get("prompt_text", "") for h in history[-k:]]
    filtered = [c for c in candidates if c not in recent]
    return filtered if filtered else candidates


def _choose_domain(profile: Dict[str, Any], history: List[Dict[str, Any]], rng: random.Random) -> str:
    allowed = profile.get("allowed_domains") or ["LANG"]

    # Optional: bias toward domains with recent failures (adaptive), but keep randomness.
    # We keep this simple for now: soft weighting.
    weights = {d: 1.0 for d in allowed}
    recent = history[-8:]
    for r in recent:
        dom = r.get("domain")
        score = r.get("score")
        if dom in weights and score in (0, 1):  # incorrect or partial
            weights[dom] += 0.35

    # Convert weights -> sampling
    doms = list(weights.keys())
    w = [weights[d] for d in doms]
    total = sum(w)
    probs = [x / total for x in w]
    return rng.choices(doms, weights=probs, k=1)[0]


# -----------------------------
# Public API used by your app
# -----------------------------
def make_prompt_plan(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Phase 2: Given doctor-entered procedure profile, create an initial prompt plan.
    In STUB mode: returns a simple balanced plan.
    In MedGemma mode: will call model to create plan.
    """
    if USE_MEDGEMMA:
        return _make_prompt_plan_medgemma(profile)

    allowed = profile.get("allowed_domains") or ["LANG", "MOT"]
    plan = []
    for i in range(12):
        dom = allowed[i % len(allowed)]
        text = random.choice(_STUB_BANK.get(dom, _STUB_BANK["LANG"]))
        plan.append({"domain": dom, "prompt_text": text, "difficulty": 2})

    return {
        "source": "stub",
        "prompt_plan": plan,
        "coverage_targets": {d: max(1, 12 // len(allowed)) for d in allowed},
        "notes": ["Stub plan: balanced rotation across allowed domains."],
    }


def next_prompt(profile: Dict[str, Any], history: List[Dict[str, Any]], sii_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Phase 2: Generate the next prompt adaptively.
    Must be susceptible to change based on answers but remain random enough.
    """
    if USE_MEDGEMMA:
        return _next_prompt_medgemma(profile, history, sii_state)

    rng = random.Random()
    # Optional deterministic seed support (useful for demos)
    seed = profile.get("seed")
    if seed is not None:
        rng.seed(int(seed) + len(history))

    dom = _choose_domain(profile, history, rng)
    candidates = _STUB_BANK.get(dom, _STUB_BANK["LANG"])
    candidates = _anti_repeat(history, candidates, k=7)

    text = rng.choice(candidates)

    return {
        "source": "stub",
        "domain": dom,
        "title": "Prompt",
        "prompt_text": text,
        "expected": "Clinician judged (Correct / Partial / Incorrect).",
        "rationale": "Stub mode: domain weighted by recent difficulty + anti-repeat + random sampling.",
        "timestamp": time.time(),
    }


def make_synopsis(profile: Dict[str, Any], history: List[Dict[str, Any]], sii_state: Dict[str, Any], alert: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Phase 2: Produce clinician-facing synopsis text (bullets) + optional warning explanation.
    In STUB mode: deterministic bullets.
    In MedGemma mode: model generates narrative.
    """
    if USE_MEDGEMMA:
        return _make_synopsis_medgemma(profile, history, sii_state, alert)

    sii_now = float(sii_state.get("sii_now", 0.0))
    in_ep = bool(sii_state.get("in_episode", False))
    doms = sii_state.get("domains_impacted") or []
    last = history[-1] if history else {}

    bullets = []
    bullets.append(f"SII={sii_now:.3f} • {'active episode' if in_ep else 'no active episode'}")
    if doms:
        bullets.append(f"Domain focus: {', '.join(doms[:3])}")
    if last:
        bullets.append(f"Last prompt: [{last.get('domain','?')}] {last.get('prompt_text','')[:80]}")

    warn_text = None
    if alert:
        warn_text = alert.get("msg", "Performance anomaly detected.")

    return {
        "source": "stub",
        "situation_summary": bullets,
        "warning_explanation": warn_text,
    }


# -----------------------------
# MedGemma (real) placeholders
# -----------------------------
# We keep these in the file, but they won't run until USE_MEDGEMMA=True.
# The Hugging Face page you uploaded shows pipeline + AutoModel/AutoProcessor usage. :contentReference[oaicite:1]{index=1}

_HF_READY = False
_PIPE = None

def _lazy_init_medgemma() -> None:
    global _HF_READY, _PIPE
    if _HF_READY:
        return
    try:
        from transformers import pipeline
        # MedGemma supports image-text-to-text; we use text-only messages for now.
        _PIPE = pipeline(
            "image-text-to-text",
            model=MODEL_ID,
        )
        _HF_READY = True
    except Exception as e:
        raise RuntimeError(f"MedGemma init failed: {e}") from e


def _make_prompt_plan_medgemma(profile: Dict[str, Any]) -> Dict[str, Any]:
    _lazy_init_medgemma()
    # Text-only prompt: provide procedure profile and ask for JSON.
    msg = (
        "You are an intraoperative awake mapping assistant.\n"
        "Create a prompt plan (12 items) in JSON with keys: domain, prompt_text, difficulty.\n"
        f"Procedure profile:\n{profile}\n"
        "Constraints:\n- Respect allowed_domains\n- Avoid repetition\n- Keep prompts short and patient-friendly\n"
        "Return ONLY valid JSON."
    )
    out = _PIPE(text=msg)
    # HF pipeline returns a list; try to pull generated text robustly
    gen = out[0].get("generated_text") if isinstance(out, list) and out else str(out)
    return {"source": "medgemma", "raw": gen}


def _next_prompt_medgemma(profile: Dict[str, Any], history: List[Dict[str, Any]], sii_state: Dict[str, Any]) -> Dict[str, Any]:
    _lazy_init_medgemma()
    msg = (
        "You generate the NEXT intraoperative prompt for awake mapping.\n"
        "Return JSON with keys: domain, prompt_text, rationale.\n"
        f"Procedure profile:\n{profile}\n"
        f"Current stability:\n{sii_state}\n"
        f"Recent history (most recent last):\n{history[-8:]}\n"
        "Constraints:\n- Must be adaptive to recent performance\n- Must remain random enough (do not repeat recent prompts)\n"
        "- Keep prompt <= max_prompt_len words if provided\n"
        "Return ONLY valid JSON."
    )
    out = _PIPE(text=msg)
    gen = out[0].get("generated_text") if isinstance(out, list) and out else str(out)
    return {"source": "medgemma", "raw": gen}


def _make_synopsis_medgemma(profile: Dict[str, Any], history: List[Dict[str, Any]], sii_state: Dict[str, Any], alert: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    _lazy_init_medgemma()
    msg = (
        "You are a clinician-facing intraoperative assistant.\n"
        "Write 4-6 short bullet points summarizing current status.\n"
        "Also, if an alert exists, provide a 1-2 sentence warning explanation.\n"
        f"Procedure profile:\n{profile}\n"
        f"Current stability:\n{sii_state}\n"
        f"Recent history:\n{history[-10:]}\n"
        f"Alert:\n{alert}\n"
        "Return JSON with keys: situation_summary (list of strings), warning_explanation (string or null).\n"
        "Return ONLY valid JSON."
    )
    out = _PIPE(text=msg)
    gen = out[0].get("generated_text") if isinstance(out, list) and out else str(out)
    return {"source": "medgemma", "raw": gen}