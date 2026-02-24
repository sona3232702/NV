import random
from typing import Dict, List, Any

# Expanded bank so it doesn't feel repetitive in a demo
PROMPTS: List[Dict[str, Any]] = [
    {"domain":"LANG","text":"Say the days of the week.","lobes":["FRONTAL","TEMPORAL"]},
    {"domain":"LANG","text":"Name three fruits.","lobes":["FRONTAL","TEMPORAL"]},
    {"domain":"LANG","text":"Repeat: 'No ifs, ands, or buts.'","lobes":["FRONTAL","TEMPORAL"]},
    {"domain":"LANG","text":"What does a key do?","lobes":["FRONTAL","TEMPORAL"]},
    {"domain":"MOT","text":"Tap your fingers 5 times.","lobes":["FRONTAL","PARIETAL"]},
    {"domain":"MOT","text":"Open and close your hand 3 times.","lobes":["FRONTAL","PARIETAL"]},
    {"domain":"MOT","text":"Lift your right thumb.","lobes":["FRONTAL","PARIETAL"]},
    {"domain":"VIS","text":"What color is this?","type":"COLOR","color":"#F4C542","lobes":["OCCIPITAL"]},
    {"domain":"VIS","text":"What color is this?","type":"COLOR","color":"#2F80ED","lobes":["OCCIPITAL"]},
    {"domain":"VIS","text":"What color is this?","type":"COLOR","color":"#EB5757","lobes":["OCCIPITAL"]},
    {"domain":"EXEC","text":"What is 7 + 5?","lobes":["FRONTAL"]},
    {"domain":"EXEC","text":"What is 9 - 4?","lobes":["FRONTAL"]},
    {"domain":"EXEC","text":"Count backwards from 10.","lobes":["FRONTAL"]},
]

def _fingerprint(p: Dict[str, Any]) -> str:
    # include color so different swatches count as different prompts
    return f"{p.get('domain','')}|{p.get('text','')}|{p.get('type','TEXT')}|{p.get('color','')}"

def next_prompt(history: List[Dict[str, Any]], *, anti_repeat_k: int = 8) -> Dict[str, Any]:
    recent = set(_fingerprint(h) for h in history[-anti_repeat_k:])
    candidates = [p for p in PROMPTS if _fingerprint(p) not in recent]
    if not candidates:
        candidates = PROMPTS
    return random.choice(candidates)
