from __future__ import annotations

import json
from typing import Any, Dict, List


# ------------------------------------------------------------
# Scoring schema (UI + OR-friendly)
# ------------------------------------------------------------
SCORING_SCHEMA = {
    "keys": {
        "Enter": {"label": "Correct", "score": 2},
        "Space": {"label": "Partial", "score": 1},
        "Backspace": {"label": "Incorrect", "score": 0},
        "U": {"label": "Undo last", "score": None},
        "R": {"label": "Reset session", "score": None},
        "Escape": {"label": "Pause/Resume", "score": None},
    },
    "interpretation": {
        "2": "Response is accurate/complete for prompt.",
        "1": "Response is incomplete/delayed/effortful but interpretable.",
        "0": "Response is incorrect/absent/unsafe to proceed without recheck.",
    },
    "risk_rules": [
        "Two consecutive 0 scores in the same critical domain (e.g., LANG) triggers a risk flag.",
        "If failures persist across 3 prompts in one domain, recommend brief pause + fatigue/confound check.",
    ],
}


# ------------------------------------------------------------
# Prompt library (deterministic, can later be replaced by MedGemma)
# ------------------------------------------------------------
PROMPT_BANK = {
    "LANG": [
        {
            "title": "Object Naming",
            "subdomain": "Naming",
            "prompt_text": "Name this object.",
            "modality": "verbal",
            "low_load_probe": "Count from 1 to 10.",
        },
        {
            "title": "Repetition",
            "subdomain": "Phonology",
            "prompt_text": "Repeat: 'No ifs, ands, or buts.'",
            "modality": "verbal",
            "low_load_probe": "Repeat: 'Today is a good day.'",
        },
        {
            "title": "Comprehension Command",
            "subdomain": "Comprehension",
            "prompt_text": "Touch your right ear.",
            "modality": "motor",
            "low_load_probe": "Close your eyes, then open them.",
        },
    ],
    "AUD": [
        {
            "title": "Auditory Comprehension",
            "subdomain": "Comprehension",
            "prompt_text": "Point to the picture I say.",
            "modality": "motor",
            "low_load_probe": "Raise your hand.",
        },
        {
            "title": "Sound Recognition",
            "subdomain": "Auditory Recognition",
            "prompt_text": "What sound is this?",
            "modality": "verbal",
            "low_load_probe": "Say your name.",
        },
    ],
    "VIS": [
        {
            "title": "Visual Matching",
            "subdomain": "Visual Discrimination",
            "prompt_text": "Match the two identical shapes.",
            "modality": "visual",
            "low_load_probe": "Look at the center dot.",
        }
    ],
    "MOT": [
        {
            "title": "Finger Tapping",
            "subdomain": "Motor",
            "prompt_text": "Tap your fingers quickly for 5 seconds.",
            "modality": "motor",
            "low_load_probe": "Wiggle your fingers once.",
        },
        {
            "title": "Hand Squeeze",
            "subdomain": "Motor",
            "prompt_text": "Squeeze my hand now.",
            "modality": "motor",
            "low_load_probe": "Lift your thumb.",
        },
    ],
    "EXEC": [
        {
            "title": "Digit Span",
            "subdomain": "Working Memory",
            "prompt_text": "Repeat these numbers: 7 - 2 - 9.",
            "modality": "verbal",
            "low_load_probe": "Repeat: 3 - 8.",
        }
    ],
}


def _pick_domains(doms: List[str], fallback: str = "LANG") -> List[str]:
    doms = [d for d in doms if d in PROMPT_BANK]
    if not doms:
        return [fallback]
    # keep top 2 (OR-friendly)
    return doms[:2]


def _build_prompt_plan(
    domains: List[str],
    top_regions: List[int],
    target_site: str,
    phase: str,
    max_prompts: int = 6,
) -> List[Dict[str, Any]]:
    """
    Deterministically build a prompt plan that feels patient/site-specific.
    """
    plan: List[Dict[str, Any]] = []
    domains = _pick_domains(domains)

    # lightweight “site specificity” tag
    site_hint = target_site.strip() if target_site else "planned target"
    region_hint = ", ".join(str(x) for x in top_regions[:3]) if top_regions else "none"

    for d in domains:
        for item in PROMPT_BANK.get(d, [])[:3]:
            plan.append({
                "domain": d,
                "subdomain": item["subdomain"],
                "title": item["title"],
                "prompt_text": item["prompt_text"],
                "modality": item["modality"],
                "scoring": {
                    "keys": ["Enter", "Space", "Backspace", "U", "R", "Escape"],
                    "meaning": "Enter=Correct(2), Space=Partial(1), Backspace=Incorrect(0), U=Undo, R=Reset, Esc=Pause",
                },
                "rationale": (
                    f"Selected because {d} shows elevated instability; "
                    f"regions of interest include parcels [{region_hint}] near {site_hint} during {phase}."
                ),
                "low_load_probe": item.get("low_load_probe"),
                "tags": {
                    "target_site": site_hint,
                    "phase": phase,
                    "roi_parcels": top_regions[:5],
                }
            })

    return plan[:max_prompts]


def _risk_mitigation_block(domains: List[str]) -> Dict[str, List[str]]:
    critical = "LANG" in domains  # simplistic but OK for stub
    return {
        "clinical": [
            "Decision support only; does not diagnose deficits and does not recommend surgical actions.",
            "Use prompts consistent with awake mapping practice and clinician judgment.",
            "If patient discomfort/fatigue is suspected, pause prompts and reassess readiness.",
        ],
        "technical": [
            "Overlay and indices are derived from task/stress proxies; false positives/negatives can occur.",
            "Use hysteresis/persistence settings to reduce flicker; verify with repeated low-load probes.",
        ],
        "human_factors": [
            "Sedation level, anxiety, pain, distraction, and motor constraints can mimic instability.",
            "Observer workload: scoring is keyboard-first to reduce attention switching.",
        ],
        "escalation": [
            ("If 2 consecutive failures occur in a critical domain (LANG), repeat a low-load probe and reassess trend."
             if critical else
             "If 2 consecutive failures occur in a domain, repeat a low-load probe and reassess trend."),
            "If failures persist across 3 prompts in one domain, recommend a brief pause and confound check.",
        ],
    }


def generate_structured_note(payload: dict) -> dict:
    """
    Deterministic stub that mimics structured output we will later generate with MedGemma.

    Payload should (optionally) include:
      - t_now, sii_now, in_episode
      - domains_impacted (list)
      - top_regions (list of parcel ids)
      - target_site (str), phase (str), hemisphere (str)
      - evidence (dict) including thresholds/baselines/windows
    """
    top = (payload.get("top_regions") or [])[:8]
    doms = payload.get("domains_impacted") or []
    sii = payload.get("sii_now", None)
    in_ep = bool(payload.get("in_episode", False))

    t_now = payload.get("t_now", None)
    hemisphere = payload.get("hemisphere", "BOTH")
    target_site = payload.get("target_site", "unspecified")
    phase = payload.get("phase", "Mapping")

    # Summary lines (short + OR-friendly)
    if sii is not None and isinstance(sii, (int, float)):
        status_line = f"{'Instability episode ongoing' if in_ep else 'No active instability episode'} at t={t_now}s (SII={float(sii):.3f})."
    else:
        status_line = ("Instability episode ongoing." if in_ep else "No active instability episode.")

    situation_summary = [
        status_line,
        f"Hemisphere view: {hemisphere}. Planned target: {target_site}. Phase: {phase}.",
        f"Primary impacted domains: {', '.join(doms) if doms else 'none detected'}.",
        f"Top regions of interest (coarse parcels): {', '.join(str(r) for r in top) if top else 'none'}.",
    ]

    # Build prompt plan
    prompt_plan = _build_prompt_plan(
        domains=doms,
        top_regions=top,
        target_site=target_site,
        phase=phase,
        max_prompts=6,
    )

    # Monitoring focus (deterministic)
    monitoring_focus = [
        "Repeat brief tasks in the most affected domain to confirm stability trends.",
        "Use low-load probes if the patient appears fatigued or distracted.",
        "If instability increases, pause scoring and reassess confounds (sedation, pain, anxiety).",
    ]

    # MedGemma-ready “model contract” section
    # Later, this becomes the explicit prompt/response interface with MedGemma.
    medgemma_contract = {
        "intended_model": "MedGemma (future)",
        "input_schema": {
            "structured_signals": ["SII", "domain_FSI", "episode_summaries", "ROI parcels", "target_site", "phase"],
            "free_text_context": ["surgeon notes", "neuropsych notes (optional)"],
        },
        "output_schema": [
            "situation_summary",
            "prompt_plan",
            "risk_mitigation",
            "provenance",
        ],
        "stub_mode": True,
    }

    note = {
        "schema_version": "v2",
        "situation_summary": situation_summary,
        "primary_domains_impacted": doms,
        "regions_of_interest": top,

        # NEW: prompt correspondence mini-view support
        "prompt_plan": prompt_plan,

        # NEW: explicit scoring schema so UI is consistent
        "scoring_schema": SCORING_SCHEMA,

        "monitoring_focus": monitoring_focus,

        # NEW: risk mitigation block
        "risk_mitigation": _risk_mitigation_block(doms),

        # Provenance: pass through and encourage full traceability
        "provenance": {
            "timepoint_s": t_now,
            "hemisphere": hemisphere,
            "target_site": target_site,
            "phase": phase,
            "signals_used": ["SII", "FSI by domain", "events-derived ROI parcels"],
            "evidence": payload.get("evidence", {}),
        },

        # Keeps your original disclaimer, expanded slightly
        "limitations_note": (
            "This is a functional stability overlay derived from task performance and stress proxies. "
            "It does not infer neural activation, does not diagnose deficits, and does not provide surgical recommendations."
        ),

        # MedGemma integration placeholder
        "medgemma": medgemma_contract,
    }
    return note


def note_as_json(note: dict) -> str:
    return json.dumps(note, indent=2)