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
from typing import Any, Dict, List, Optional
import os
import random
import time


# -----------------------------
# Toggle: STUB vs REAL MedGemma
# -----------------------------
USE_MEDGEMMA = False  # keep False for now (safe). Flip to True when ready.
MODEL_ID = os.environ.get("MEDGEMMA_MODEL_ID", "google/medgemma-1.5-4b-it")


# -----------------------------
# Prompt schema (lightweight)
# -----------------------------
@dataclass
class PromptItem:
    domain: str
    prompt_text: str
    prompt_type: str = "TEXT"  # TEXT | COLOR | IMAGE | MATH | MOTOR | AUDIO
    payload: Optional[Dict[str, Any]] = None
    title: str = "Prompt"
    expected: str = ""
    rationale: str = ""


# -----------------------------------
# STUB prompt bank (always available)
# -----------------------------------
# In stub mode we still provide MULTI-MODAL-ish prompts:
# - COLOR renders a colored swatch in the TV preview
# - IMAGE uses a big emoji placeholder (later swapped to real images/audio)
# - MATH uses simple arithmetic
# - MOTOR uses reps/time cues
_STUB_TEMPLATES: Dict[str, List[Dict[str, Any]]] = {
    "LANG": [
        {"prompt_type": "TEXT", "prompt_text": "Say the days of the week.", "payload": {}},
        {"prompt_type": "TEXT", "prompt_text": "Repeat: 'No ifs, ands, or buts.'", "payload": {}},
        {"prompt_type": "TEXT", "prompt_text": "Name three things you might find in a kitchen.", "payload": {}},
    ],
    "MOT": [
        {"prompt_type": "MOTOR", "prompt_text": "Tap your fingers 5 times.", "payload": {"reps": 5}},
        {"prompt_type": "MOTOR", "prompt_text": "Open and close your hand for 5 seconds.", "payload": {"seconds": 5}},
        {"prompt_type": "MOTOR", "prompt_text": "Lift your right thumb and hold for 3 seconds.", "payload": {"seconds": 3}},
    ],
    "VIS": [
        {"prompt_type": "COLOR", "prompt_text": "What color is this?", "payload": {"color_hex": "#ff3b30", "answer": "red"}},
        {"prompt_type": "COLOR", "prompt_text": "What color is this?", "payload": {"color_hex": "#34c759", "answer": "green"}},
        {"prompt_type": "COLOR", "prompt_text": "What color is this?", "payload": {"color_hex": "#007aff", "answer": "blue"}},
        {"prompt_type": "IMAGE", "prompt_text": "What animal is this?", "payload": {"emoji": "🦁", "answer": "lion"}},
        {"prompt_type": "IMAGE", "prompt_text": "What animal is this?", "payload": {"emoji": "🐢", "answer": "turtle"}},
        {"prompt_type": "IMAGE", "prompt_text": "What animal is this?", "payload": {"emoji": "🐘", "answer": "elephant"}},
    ],
    "AUD": [
        # Placeholder until you add actual audio playback
        {"prompt_type": "TEXT", "prompt_text": "Repeat the last word you heard.", "payload": {"note": "audio placeholder"}},
        {"prompt_type": "TEXT", "prompt_text": "Did you hear a high or low tone?", "payload": {"note": "audio placeholder"}},
    ],
    "EXEC": [
        {"prompt_type": "MATH", "prompt_text": "What is 7 + 5?", "payload": {"answer": 12}},
        {"prompt_type": "MATH", "prompt_text": "What is 9 − 4?", "payload": {"answer": 5}},
        {"prompt_type": "TEXT", "prompt_text": "Count backwards from 20.", "payload": {}},
    ],
}


def _anti_repeat(history: List[Dict[str, Any]], candidates: List[Dict[str, Any]], k: int = 7) -> List[Dict[str, Any]]:
    recent = [h.get("prompt_text", "") for h in history[-k:]]
    filtered = [c for c in candidates if c.get("prompt_text", "") not in recent]
    return filtered if filtered else candidates


def _choose_domain(profile: Dict[str, Any], history: List[Dict[str, Any]], rng: random.Random) -> str:
    allowed = profile.get("allowed_domains") or ["LANG"]

    # Bias toward domains with recent difficulty, but keep randomness.
    weights = {d: 1.0 for d in allowed}
    recent = history[-8:]
    for r in recent:
        dom = r.get("domain")
        score = r.get("score")
        if dom in weights and score in (0, 1):
            weights[dom] += 0.35

    doms = list(weights.keys())
    w = [weights[d] for d in doms]
    return rng.choices(doms, weights=w, k=1)[0]


def _tailor_interest(text: str, interest: str) -> str:
    interest = (interest or "").strip()
    if not interest:
        return text

    # Keep it subtle: 1 short clause.
    # Example: "Say the days of the week." -> "Say the days of the week (like scheduling your lab notes)."
    lower = interest.lower()
    if "botan" in lower or "plant" in lower:
        return text + " (think plants/gardens)."
    if "space" in lower or "astro" in lower:
        return text + " (think space/planets)."
    if "music" in lower:
        return text + " (think music/rhythm)."
    if "cook" in lower or "chef" in lower:
        return text + " (think cooking)."
    return text + f" (theme: {interest})."


# -----------------------------
# Public API used by your app
# -----------------------------
def make_prompt_plan(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Create an initial prompt plan. In stub mode, returns a balanced plan."""
    if USE_MEDGEMMA:
        return _make_prompt_plan_medgemma(profile)

    allowed = profile.get("allowed_domains") or ["LANG", "MOT"]
    interest = profile.get("patient_interest") or profile.get("profile_interest") or ""

    plan: List[Dict[str, Any]] = []
    for i in range(12):
        dom = allowed[i % len(allowed)]
        template = random.choice(_STUB_TEMPLATES.get(dom, _STUB_TEMPLATES["LANG"]))
        plan.append({
            "domain": dom,
            "prompt_type": template.get("prompt_type", "TEXT"),
            "prompt_text": _tailor_interest(template.get("prompt_text", ""), str(interest)),
            "payload": template.get("payload", {}) or {},
            "difficulty": 2,
        })

    return {
        "source": "stub",
        "prompt_plan": plan,
        "coverage_targets": {d: max(1, 12 // len(allowed)) for d in allowed},
        "notes": ["Stub plan: balanced rotation across allowed domains with light interest tailoring."],
    }


def next_prompt(profile: Dict[str, Any], history: List[Dict[str, Any]], sii_state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate the next prompt adaptively (answer-sensitive, still random)."""
    if USE_MEDGEMMA:
        return _next_prompt_medgemma(profile, history, sii_state)

    rng = random.Random()
    seed = profile.get("seed")
    if seed is not None:
        rng.seed(int(seed) + len(history))

    dom = _choose_domain(profile, history, rng)
    candidates = _STUB_TEMPLATES.get(dom, _STUB_TEMPLATES["LANG"])
    candidates = _anti_repeat(history, candidates, k=7)

    tmpl = rng.choice(candidates)
    interest = profile.get("patient_interest") or profile.get("profile_interest") or ""

    prompt_text = _tailor_interest(str(tmpl.get("prompt_text", "")), str(interest))
    prompt_type = str(tmpl.get("prompt_type", "TEXT"))
    payload = dict(tmpl.get("payload", {}) or {})

    # Respect max_prompt_len if set (soft trim for stub)
    max_words = int(profile.get("max_prompt_len", 18))
    words = prompt_text.split()
    if len(words) > max_words:
        prompt_text = " ".join(words[:max_words])

    return {
        "source": "stub",
        "domain": dom,
        "title": "Prompt",
        "prompt_type": prompt_type,
        "prompt_text": prompt_text,
        "payload": payload,
        "expected": "Clinician judged (Correct / Partial / Incorrect).",
        "rationale": "Stub: weighted domain sampling (recent difficulty) + anti-repeat + controlled randomness + light interest tailoring.",
        "timestamp": time.time(),
    }


def make_synopsis(
    profile: Dict[str, Any],
    history: List[Dict[str, Any]],
    sii_state: Dict[str, Any],
    alert: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Clinician-facing synopsis bullets + optional warning explanation."""
    if USE_MEDGEMMA:
        return _make_synopsis_medgemma(profile, history, sii_state, alert)

    sii_now = float(sii_state.get("sii_now", 0.0))
    in_ep = bool(sii_state.get("in_episode", False))
    doms = sii_state.get("domains_impacted") or []
    last = history[-1] if history else {}

    bullets: List[str] = []
    bullets.append(f"SII={sii_now:.3f} • {'active episode' if in_ep else 'no active episode'}")
    if doms:
        bullets.append(f"Domain focus: {', '.join(doms[:3])}")
    if last:
        bullets.append(f"Last prompt: [{last.get('domain','?')}] {str(last.get('prompt_text',''))[:80]}")

    warn_text = alert.get("msg") if alert else None

    return {
        "source": "stub",
        "situation_summary": bullets,
        "warning_explanation": warn_text,
    }


# -----------------------------
# MedGemma (real) placeholders
# -----------------------------
_HF_READY = False
_PIPE = None

def _lazy_init_medgemma() -> None:
    global _HF_READY, _PIPE
    if _HF_READY:
        return
    try:
        from transformers import pipeline
        _PIPE = pipeline(
            "image-text-to-text",
            model=MODEL_ID,
        )
        _HF_READY = True
    except Exception as e:
        raise RuntimeError(f"MedGemma init failed: {e}") from e


def _make_prompt_plan_medgemma(profile: Dict[str, Any]) -> Dict[str, Any]:
    _lazy_init_medgemma()
    msg = (
        "You are an intraoperative awake mapping assistant.\n"
        "Create a prompt plan (12 items) in JSON with keys: domain, prompt_type, prompt_text, payload, difficulty.\n"
        f"Procedure profile:\n{profile}\n"
        "Constraints:\n- Respect allowed_domains\n- Avoid repetition\n- Keep prompts short and patient-friendly\n"
        "Return ONLY valid JSON."
    )
    out = _PIPE(text=msg)
    gen = out[0].get("generated_text") if isinstance(out, list) and out else str(out)
    return {"source": "medgemma", "raw": gen}


def _next_prompt_medgemma(profile: Dict[str, Any], history: List[Dict[str, Any]], sii_state: Dict[str, Any]) -> Dict[str, Any]:
    _lazy_init_medgemma()
    msg = (
        "You generate the NEXT intraoperative prompt for awake mapping.\n"
        "Return JSON with keys: domain, prompt_type, prompt_text, payload, rationale.\n"
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
