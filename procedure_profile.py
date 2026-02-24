"""
procedure_profile.py
Phase 2: Doctor-facing intake form for patient + procedure specifics.

Returns a single dict ("procedure_profile") that:
- drives MedGemma prompt planning
- constrains domains/randomness
- sets stop rules / safety preferences
"""

from __future__ import annotations
from typing import Any, Dict
import streamlit as st


def procedure_profile_ui(defaults: Dict[str, Any] | None = None) -> Dict[str, Any]:
    defaults = defaults or {}

    st.subheader("Procedure Setup")

    with st.expander("Patient + Procedure Profile", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            patient_id = st.text_input("Patient ID (de-identified)", value=defaults.get("patient_id", "PT-001"))
        with c2:
            age_group = st.selectbox("Age group", ["Adult", "Pediatric"], index=0 if defaults.get("age_group", "Adult") == "Adult" else 1)

        language = st.text_input("Primary language", value=defaults.get("language", "English"))

        patient_interest = st.text_input(
            "Patient interest / identity (e.g., botanist, space scientist, musician)",
            value=defaults.get("patient_interest", defaults.get("profile_interest", "")),
            help="Used to theme prompts in a patient-friendly way.",
        )

        baseline_notes = st.text_area(
            "Baseline / history notes (optional)",
            value=defaults.get("baseline_notes", ""),
            height=90,
            placeholder="e.g., mild expressive aphasia pre-op; anxious; hearing intact; left-handed",
        )

        st.markdown("---")

        procedure_type = st.selectbox(
            "Procedure type",
            ["Awake mapping", "Tumor resection", "Epilepsy mapping", "Other"],
            index=["Awake mapping", "Tumor resection", "Epilepsy mapping", "Other"].index(defaults.get("procedure_type", "Awake mapping")),
        )

        planned_target = st.text_input("Planned target / focus (free text)", value=defaults.get("planned_target", "Left temporal (planned)"))

        procedure_description = st.text_area(
            "Procedure description (what will be done, in clinician terms)",
            value=defaults.get("procedure_description", ""),
            height=110,
            placeholder="Describe steps/goal and what you want to monitor during mapping...",
        )

        constraints = st.multiselect(
            "Constraints / confounds to consider",
            [
                "Fatigue risk",
                "Hearing limitation",
                "Vision limitation",
                "Motor limitation",
                "Anxiety/pain confound",
                "Sedation/med confound",
                "Bilingual / language proficiency variable",
            ],
            default=defaults.get("constraints", ["Fatigue risk"]),
        )

        st.markdown("---")

        allowed_domains = st.multiselect(
            "Allowed prompt domains",
            ["LANG", "AUD", "VIS", "MOT", "EXEC"],
            default=defaults.get("allowed_domains", ["LANG", "MOT"]),
        )

        randomness = st.slider("Randomness level (higher = more variety)", 0.0, 1.0, float(defaults.get("randomness", 0.45)), 0.05)
        max_prompt_len = st.slider("Max prompt length (words)", 6, 40, int(defaults.get("max_prompt_len", 18)), 1)

        use_seed = st.checkbox("Use deterministic seed (demo reproducibility)", value=bool(defaults.get("use_seed", False)))
        seed = None
        if use_seed:
            seed = st.number_input("Seed", min_value=0, max_value=10_000, value=int(defaults.get("seed", 7)))

        st.markdown("---")
        st.markdown("### Safety / Stop rules")

        critical_domains = st.multiselect(
            "Critical domains (stricter escalation if failures occur)",
            ["LANG", "MOT", "VIS", "AUD", "EXEC"],
            default=defaults.get("critical_domains", ["LANG"]),
        )

        two_fails = st.checkbox("Escalate if 2 consecutive failures in a critical domain", value=bool(defaults.get("stop_two_fails", True)))
        three_trend = st.checkbox("Escalate if worsening persists across 3 prompts", value=bool(defaults.get("stop_three_trend", True)))
        fatigue_pause_hint = st.checkbox("Suggest pause if fatigue/confound suspected", value=bool(defaults.get("fatigue_pause_hint", True)))

    profile: Dict[str, Any] = {
        "patient_id": patient_id.strip(),
        "age_group": age_group,
        "language": language.strip(),
        "patient_interest": patient_interest.strip(),
        "baseline_notes": baseline_notes.strip(),
        "procedure_type": procedure_type,
        "planned_target": planned_target.strip(),
        "procedure_description": procedure_description.strip(),
        "constraints": constraints,
        "allowed_domains": allowed_domains,
        "randomness": float(randomness),
        "max_prompt_len": int(max_prompt_len),
        "critical_domains": critical_domains,
        "stop_rules": {
            "two_fails": bool(two_fails),
            "three_trend": bool(three_trend),
            "fatigue_pause_hint": bool(fatigue_pause_hint),
        },
    }

    if use_seed and seed is not None:
        profile["seed"] = int(seed)

    return profile
