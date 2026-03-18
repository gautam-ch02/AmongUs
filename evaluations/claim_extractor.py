"""
claim_extractor.py  —  spaCy-based claim extraction for Among Us SPEAK events.

Replaces the two-regex extraction in structured_v1_inference.py.

Claim types:
  location   — "I was in Admin", "I went to Cafeteria"
  task       — "I completed Fix Wiring", "doing my wiring task"
  sighting   — "I saw Player 3", "Player 5 was with me in Admin"
  accusation — "I think Player 2 is the impostor", "vote Blue"
  denial     — "I didn't kill anyone", "I'm not the impostor"
  witness    — "I found Player 4's body in O2", "I saw Blue kill Orange"
  alibi      — "I was doing tasks", "I was nowhere near the body"

Usage:
    from analysis.claim_extractor import ClaimExtractor
    extractor = ClaimExtractor()
    claims = extractor.extract(text, actor, actor_location=None)
"""

import re
import spacy
from typing import List, Dict, Any, Optional

# =============================================================================
# Game constants
# =============================================================================

ROOMS = {
    "Cafeteria", "Weapons", "Navigation", "O2", "Shields", "Communications",
    "Storage", "Admin", "Electrical", "Lower Engine", "Security", "Reactor",
    "Upper Engine", "Medbay",
}
ROOM_LOWER: Dict[str, str] = {r.lower(): r for r in ROOMS}

TASKS = {
    "Download Data", "Empty Garbage", "Fix Wiring", "Accept Diverted Power",
    "Clear Asteroids", "Chart Course", "Stabilize Steering", "Clean O2 Filter",
    "Empty Chute", "Prime Shields", "Swipe Card", "Upload Data",
    "Calibrate Distributor", "Divert Power", "Align Engine Output", "Fuel Engines",
    "Start Reactor", "Unlock Manifolds", "Inspect Sample", "Submit Scan",
}

# Single-word / short-phrase → canonical task name
TASK_KEYWORDS: Dict[str, str] = {
    "wiring":               "Fix Wiring",
    "garbage":              "Empty Garbage",
    "asteroids":            "Clear Asteroids",
    "o2 filter":            "Clean O2 Filter",
    "swipe card":           "Swipe Card",
    "upload data":          "Upload Data",
    "download data":        "Download Data",
    "chart course":         "Chart Course",
    "stabilize steering":   "Stabilize Steering",
    "empty chute":          "Empty Chute",
    "calibrate distributor":"Calibrate Distributor",
    "divert power":         "Divert Power",
    "align engine":         "Align Engine Output",
    "fuel engines":         "Fuel Engines",
    "fuel engine":          "Fuel Engines",
    "start reactor":        "Start Reactor",
    "unlock manifolds":     "Unlock Manifolds",
    "inspect sample":       "Inspect Sample",
    "submit scan":          "Submit Scan",
    "medbay scan":          "Submit Scan",
    "prime shields":        "Prime Shields",
    "accept diverted":      "Accept Diverted Power",
}

COLOR_NAMES = {
    "blue", "red", "green", "purple", "yellow", "black", "white", "orange",
    "brown", "lime", "cyan", "pink", "maroon", "rose", "coral", "banana",
    "tan", "gray", "grey",
}

# =============================================================================
# Compiled regex patterns  (compiled once at module import)
# =============================================================================

_ROOM_ALT   = '|'.join(re.escape(r) for r in sorted(ROOMS, key=len, reverse=True))
_TASK_ALT   = '|'.join(re.escape(t) for t in sorted(TASKS, key=len, reverse=True))
_KWORD_ALT  = '|'.join(re.escape(k) for k in sorted(TASK_KEYWORDS, key=len, reverse=True))
_COLOR_ALT  = '|'.join(re.escape(c) for c in sorted(COLOR_NAMES, key=len, reverse=True))

ROOM_RE      = re.compile(r'\b(' + _ROOM_ALT + r')\b', re.IGNORECASE)
TASK_RE      = re.compile(r'\b(' + _TASK_ALT + r')\b', re.IGNORECASE)
TASK_KWORD_RE= re.compile(r'\b(' + _KWORD_ALT + r')\b', re.IGNORECASE)
COLOR_RE     = re.compile(r'\b(' + _COLOR_ALT + r')\b', re.IGNORECASE)
PLAYER_RE    = re.compile(r'\bPlayer\s+(\d+)(?:\s*:\s*([a-zA-Z]+))?\b', re.IGNORECASE)

# Player reference anywhere: "Player N" or standalone color not preceded by "Player N:"
ANY_PLAYER_RE = re.compile(
    r'\bPlayer\s+\d+(?:\s*:\s*[a-zA-Z]+)?\b|(?<!\:\s)\b(?:' + _COLOR_ALT + r')\b',
    re.IGNORECASE,
)

# --------------------------------------------------------------------------
# Self-location patterns  (subject = "I")
# --------------------------------------------------------------------------
SELF_LOC_RE = re.compile(
    r"(?i)"
    r"\b(?:"
    r"I\s+(?:was|am|'m)\s+(?:in|at|inside|near|around|by)"
    r"|I\s+(?:went|moved|headed|traveled|walked|came|arrived|got)\s+(?:to|in|into|at|near|toward(?:s)?)"
    r"|I\s+(?:was|'m)\s+currently\s+(?:in|at)"
    r"|I\s+(?:passed\s+through|passed\s+by|passed\s+near)"
    r"|my\s+last\s+location\s+was"
    r"|I\s+was\s+last\s+(?:in|at|seen\s+in)"
    r"|I\s+(?:stayed|remained|sat|stood)\s+(?:in|at)"
    r"|I\s+left\s+(?:[a-zA-Z\s]+?\s+)?(?:to|and\s+went)"
    r"|I\s+(?:moved?\s+from|came\s+from|went\s+from)"
    r"|I\s+was\s+coming\s+from"
    r")\b"
)

# Other-player location: "Player N was in X", "Blue is in X"
_ANY_P = r"(?:Player\s+\d+(?:\s*:\s*[a-zA-Z]+)?|" + _COLOR_ALT + r")"
_OTHER_LOC_PAT = (
    r"(?i)(?:"
    # "Player X was/is/seemed in/at/near ROOM"
    + r"\b" + _ANY_P + r"\b"
    + r"\s+(?:was|is|were|had\s+been|got\s+caught|appeared?|seems?)\s+"
    + r"(?:in|at|near|around|by|inside|coming\s+from|heading\s+to(?:ward(?:s)?)?)"
    # "Player X headed/moved/walked to/from ROOM"
    + r"|\b" + _ANY_P + r"\b"
    + r"\s+(?:headed?|moved?|walked?|went|travel(?:ed)?|heading|running|came?)\s+"
    + r"(?:to(?:ward(?:s)?)?|from|into|through)\s"
    # "Player X's movement(s) from/to/between ROOM"
    + r"|\b" + _ANY_P + r"(?:'s)?"
    + r"\s+movements?\s+(?:from|to(?:ward(?:s)?)?|between)"
    + r")"
)
OTHER_LOC_RE = re.compile(_OTHER_LOC_PAT)

# Task-doing: completing/did some task
TASK_DOING_RE = re.compile(
    r"(?i)\b(?:"
    r"(?:was\s+)?(?:complet(?:ing|ed)|finish(?:ing|ed)|work(?:ing|ed)\s+on|doing|did)\s+"
    r"(?:my\s+|the\s+|a\s+)?(?:task|wiring|scan|fix|divert|fuel|calibrat|inspect|unlock|start|align|empty|clear|download|upload|swipe|chart|stabilize|prime|submit)"
    r"|completing\s+(?:my\s+)?tasks?"
    r")\b"
)

# Accusation: vote / think-believe-suspect / [player] is impostor/sus
_P = r"(?:Player\s+\d+(?:\s*:\s*[a-zA-Z]+)?|" + _COLOR_ALT + r")"
_ACCUSE_PAT = (
    r"(?i)(?:"
    + r"vote\s+(?:for\s+)?" + _P
    + r"|(?:I\s+(?:think|believe|suspect|feel)|(?:probably|definitely|clearly|obviously|likely))\s+(?:that\s+)?" + _P + r"\s+(?:is|could be|might be|has to be|must be)\s+(?:the\s+)?(?:impostor|imposter|sus|suspicious|killer)"
    + r"|" + _P + r"\s+(?:is|seems?|must\s+be|has\s+to\s+be|could\s+be|might\s+be)\s+(?:the\s+)?(?:impostor|imposter|sus(?:picious)?|killer|lying)"
    + r"|" + _P + r"\s+(?:killed|murdered|attacked|did\s+it|is\s+acting\s+sus)"
    + r"|(?:I\s+)?(?:suspect|accus(?:e|ed|ing))\s+" + _P
    + r"|" + _P + r"\s+is\s+(?:the\s+one|suspicious|sketchy|acting\s+weird)"
    + r"|sus\s+(?:of\s+)?" + _P
    # possessive: "Player X's behavior/movements are/seem suspicious/unusual"
    + r"|" + _P + r"(?:'s)?\s+(?:movements?|behavior|actions?|activities?)\s+(?:(?:are|is|were|seem|looks?|appears?)\s+)?(?:suspicious|sus|unusual|strange|weird|sketchy)"
    # "Player X has been acting suspiciously/strangely"
    + r"|" + _P + r"\s+(?:has\s+been|is)\s+acting\s+(?:suspiciously|strangely|weird(?:ly)?|oddly|unusually)"
    # vent accusation: "Player X vented/is venting"
    + r"|" + _P + r"\s+(?:vent(?:ed|ing)|used?\s+(?:a\s+)?vent)"
    + r")"
)
ACCUSE_RE = re.compile(_ACCUSE_PAT)

# Denial: "I didn't kill", "I'm not the impostor", "it wasn't me"
DENY_RE = re.compile(
    r"(?i)\b(?:"
    r"(?:I\s+)?did(?:n'?t)?\s+kill"
    r"|(?:I\s+)?haven'?t\s+kill(?:ed)?"
    r"|(?:I\s+)?never\s+kill(?:ed)?"
    r"|(?:I(?:'m|\s+am)\s+)?not\s+(?:the\s+)?(?:impostor|imposter|killer)"
    r"|it\s+wasn'?t\s+(?:me|I)"
    r"|(?:I\s+)?wasn'?t\s+(?:me|the\s+one|involved|responsible)"
    r"|(?:I\s+)?have\s+nothing\s+to\s+(?:do\s+with|hide)"
    r"|I\s+(?:am\s+)?innocent"
    r"|I\s+(?:swear|promise)\s+I\s+(?:didn'?t|haven'?t|never)"
    r"|I'?m\s+not\s+(?:the\s+)?(?:impostor|killer|sus)"
    r")\b"
)

# Witness — body found or kill observed
BODY_RE = re.compile(
    r"(?i)(?:"
    r"(?:found|saw|spotted|discovered|reported)\s+(?:(?:Player\s+\d+|" + _COLOR_ALT + r")'s?\s+)?(?:the\s+)?(?:body|corpse|dead\s+body)"
    r"|(?:body|corpse|dead\s+body)\s+(?:was|is|found|in|at|near)"
    r"|(?:Player\s+\d+|" + _COLOR_ALT + r")\s+(?:was\s+found\s+dead|is\s+dead|died|got\s+killed)"
    r"|(?:Player\s+\d+|" + _COLOR_ALT + r")(?:'s)?\s+(?:death|murder|killing)\b"
    r")"
)

_PREF = r"(?:Player\s+\d+(?:\s*:\s*[a-zA-Z]+)?|" + _COLOR_ALT + r")"
_KILL_SEEN_PAT = (
    r"(?i)(?:"
    r"(?:saw|watched|witnessed|observed)\s+" + _PREF + r"\s+kill"
    + r"|witnessed\s+(?:the\s+)?kill"
    + r"|" + _PREF + r"\s+(?:\w+\s+)?killed\s+(?:" + _PREF + r"|someone|anyone|a\s+\w+|them|everybody)"
    + r"|(?:you|they|he|she)\s+killed\s+" + _PREF
    + r"|the\s+fact\s+that\s+" + _PREF + r"\s+killed"
    + r")"
)
KILL_SEEN_RE = re.compile(_KILL_SEEN_PAT)

# Alibi
ALIBI_RE = re.compile(
    r"(?i)\b(?:"
    r"(?:was\s+)?doing\s+(?:my\s+)?tasks?"
    r"|working\s+on\s+(?:my\s+)?tasks?"
    r"|completing\s+(?:my\s+)?tasks?"
    r"|(?:tried?\s+to|trying\s+to|just)\s+(?:complete|finish|do)\s+(?:my\s+)?tasks?"
    r"|had\s+tasks?\s+(?:to\s+do|to\s+complete)"
    r"|nowhere\s+near"
    r"|wasn'?t\s+near"
    r"|(?:I\s+)?wasn'?t\s+(?:in|around|at|there|near)\s+(?:the\s+)?(?:body|area|room)"
    r"|(?:I\s+was\s+)?(?:alone|by\s+myself|on\s+my\s+own)(?:\s+in)?"
    r")\b"
)

# Sighting: "I saw/was with [player]"
SEE_PLAYER_RE = re.compile(
    r"(?i)(?:"
    r"(?:I\s+)?(?:saw|spotted|noticed|observed|met|encountered)\s+(?:Player\s+\d+|" + _COLOR_ALT + r")"
    r"|(?:I\s+was|we\s+were)\s+(?:with|near|alongside|next\s+to)\s+(?:Player\s+\d+|" + _COLOR_ALT + r")"
    r"|(?:Player\s+\d+|" + _COLOR_ALT + r")\s+(?:and\s+I|was\s+with\s+me|can\s+(?:vouch|confirm))"
    r"|(?:Player\s+\d+|" + _COLOR_ALT + r")\s+(?:was\s+nearby|was\s+following|walked\s+with)"
    r")"
)


# =============================================================================
# Helpers
# =============================================================================

def _find_rooms(text: str) -> List[str]:
    """Return canonical room names found in text."""
    return [ROOM_LOWER.get(m.group(1).lower(), m.group(1))
            for m in ROOM_RE.finditer(text)]


def _find_tasks(text: str) -> List[str]:
    tasks: set = set()
    for m in TASK_RE.finditer(text):
        tasks.add(m.group(1).title() if m.group(1).lower() in {t.lower() for t in TASKS}
                  else m.group(1))
    # canonical task lookup
    tasks2: set = set()
    for t in tasks:
        low = t.lower()
        match = next((canonical for canonical in TASKS if canonical.lower() == low), t)
        tasks2.add(match)
    # keyword fragments
    for m in TASK_KWORD_RE.finditer(text):
        kw = m.group(1).lower()
        if kw in TASK_KEYWORDS:
            tasks2.add(TASK_KEYWORDS[kw])
    return list(tasks2)


def _find_players(text: str) -> List[str]:
    """Return player references in text (normalised strings)."""
    players = []
    seen_spans = []
    for m in PLAYER_RE.finditer(text):
        num   = m.group(1)
        color = (m.group(2) or "").lower()
        label = f"Player {num}: {color}" if color else f"Player {num}"
        players.append(label)
        seen_spans.append((m.start(), m.end()))
    # standalone colors not already part of "Player N: color"
    for m in COLOR_RE.finditer(text):
        c = m.group(1).lower()
        # skip if inside an already-matched Player span
        overlap = any(s <= m.start() <= e for s, e in seen_spans)
        if not overlap:
            players.append(c)
    return players


def _extract_player_from_match(m_text: str) -> Optional[str]:
    """Extract the first player reference from a matched substring."""
    pm = PLAYER_RE.search(m_text)
    if pm:
        color = (pm.group(2) or "").lower()
        return f"Player {pm.group(1)}: {color}" if color else f"Player {pm.group(1)}"
    cm = COLOR_RE.search(m_text)
    if cm:
        return cm.group(1).lower()
    return None


def _claim(claim_type, claim_text, subject, predicate, obj, confidence):
    return {
        "claim_type":  claim_type,
        "claim_text":  claim_text.strip()[:300],
        "subject":     subject,
        "predicate":   predicate,
        "object":      obj,
        "confidence":  round(confidence, 2),
    }


# =============================================================================
# Main extraction logic — sentence-level
# =============================================================================

def _extract_from_sentence(
    sent_text: str,
    actor: str,
) -> List[Dict[str, Any]]:
    """Extract all claims from a single sentence."""
    claims: List[Dict[str, Any]] = []
    t = sent_text.strip()
    if not t:
        return claims

    rooms   = _find_rooms(t)
    tasks   = _find_tasks(t)
    players = _find_players(t)

    # ── 1. Witness: kill observed or body found (highest priority) ──────
    if KILL_SEEN_RE.search(t):
        m = KILL_SEEN_RE.search(t)
        killer = _extract_player_from_match(m.group(0))
        claims.append(_claim(
            "witness", t, actor, "saw_kill",
            obj=killer,
            confidence=0.9,
        ))

    if BODY_RE.search(t):
        m = BODY_RE.search(t)
        room_obj = rooms[0] if rooms else None
        claims.append(_claim(
            "witness", t, actor, "found_body",
            obj=room_obj,
            confidence=0.85,
        ))

    # ── 2. Accusation ───────────────────────────────────────────────────
    if ACCUSE_RE.search(t):
        m = ACCUSE_RE.search(t)
        target = _extract_player_from_match(m.group(0))
        claims.append(_claim(
            "accusation", t, actor, "accuses_impostor",
            obj=target,
            confidence=0.85,
        ))
    elif players and re.search(r'\b(sus(?:picious)?|vote|impostor|imposter|killer|guilty|blame|accus|unusual|strange|weird|sketchy|vented?|vent(?:ing)?|explain\s+yourself)\b', t, re.I):
        # Looser accusation: player name + suspicious keyword
        claims.append(_claim(
            "accusation", t, actor, "accuses_impostor",
            obj=players[0] if players else None,
            confidence=0.65,
        ))

    # ── 3. Denial ───────────────────────────────────────────────────────
    if DENY_RE.search(t):
        claims.append(_claim(
            "denial", t, actor, "deny_impostor_or_kill",
            obj=None,
            confidence=0.85,
        ))

    # ── 4. Self-location ─────────────────────────────────────────────────
    # Skip location inference if a witness claim already explains room mention
    _has_witness = any(c["claim_type"] == "witness" for c in claims)
    if not _has_witness:
        if SELF_LOC_RE.search(t) and rooms:
            for room in rooms:
                claims.append(_claim(
                    "location", t, actor, "was_in",
                    obj=room,
                    confidence=0.85,
                ))
        elif rooms and re.search(r"\b(I|my|me)\b", t, re.I) and not OTHER_LOC_RE.search(t):
            # First-person sentence mentioning a room without explicit location verb
            for room in rooms:
                claims.append(_claim(
                    "location", t, actor, "was_in",
                    obj=room,
                    confidence=0.65,
                ))

    # ── 5. Other-player location sighting ────────────────────────────────
    if OTHER_LOC_RE.search(t) and rooms:
        m = OTHER_LOC_RE.search(t)
        target = _extract_player_from_match(m.group(0))
        room = rooms[0] if rooms else None
        if target:
            claims.append(_claim(
                "sighting", t, actor, "saw_player_in_room",
                obj=target,
                confidence=0.80,
            ))

    # ── 6. Sighting: "I saw/was with [player]" ──────────────────────────
    if SEE_PLAYER_RE.search(t):
        m = SEE_PLAYER_RE.search(t)
        target = _extract_player_from_match(m.group(0))
        if target and not any(c.get("claim_type") == "sighting" for c in claims):
            claims.append(_claim(
                "sighting", t, actor, "saw_player",
                obj=target,
                confidence=0.80,
            ))

    # ── 7. Task claims ───────────────────────────────────────────────────
    if tasks and (TASK_DOING_RE.search(t) or re.search(r'\b(task|complet|finish|doing|work|fix|repair)\b', t, re.I)):
        for task in tasks:
            claims.append(_claim(
                "task", t, actor, "completed_task",
                obj=task,
                confidence=0.80,
            ))
    elif tasks:
        # Task name mentioned without explicit doing-verb
        for task in tasks:
            claims.append(_claim(
                "task", t, actor, "mentioned_task",
                obj=task,
                confidence=0.55,
            ))

    # ── 8. Alibi ─────────────────────────────────────────────────────────
    if ALIBI_RE.search(t) and not any(c.get("claim_type") in ("location", "task") for c in claims):
        claims.append(_claim(
            "alibi", t, actor, "was_doing_tasks_or_alibi",
            obj=None,
            confidence=0.75,
        ))

    return claims


# =============================================================================
# Public API
# =============================================================================

class ClaimExtractor:
    """
    spaCy-based claim extractor.  Load once and reuse across events.

    extractor = ClaimExtractor()
    claims = extractor.extract(text, actor)
    """

    def __init__(self):
        # Disable heavy pipeline components we don't need; keep sentencizer
        self._nlp = spacy.load("en_core_web_sm",
                               disable=["ner", "lemmatizer", "attribute_ruler"])

    def extract(
        self,
        text: str,
        actor: str = "unknown",
        actor_location: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract claims from a SPEAK event's raw_text.

        Returns a list of claim dicts (may be empty if no claims found).
        """
        if not text or not text.strip():
            return []

        doc = self._nlp(text)
        all_claims: List[Dict[str, Any]] = []

        for sent in doc.sents:
            all_claims.extend(_extract_from_sentence(sent.text, actor))

        # De-duplicate: same (claim_type, predicate, object) within same event
        seen: set = set()
        deduped: List[Dict[str, Any]] = []
        for c in all_claims:
            key = (c["claim_type"], c["predicate"], str(c["object"]))
            if key not in seen:
                seen.add(key)
                deduped.append(c)

        return deduped


# =============================================================================
# Standalone test
# =============================================================================

if __name__ == "__main__":
    extractor = ClaimExtractor()

    tests = [
        ("I was in Admin and then Cafeteria. Where was everyone?",           "Player 3: orange"),
        ("I just found Player 2's body in O2! This is terrible.",            "Player 1: blue"),
        ("I think Player 7 is the impostor — they were acting really sus.",  "Player 4: lime"),
        ("I didn't kill anyone. I'm not the impostor!",                      "Player 5: black"),
        ("I was completing my wiring task in Electrical.",                   "Player 6: pink"),
        ("Vote Blue. They were near Weapons right before the body was found.","Player 3: cyan"),
        ("I saw Player 1: blue kill Player 2: red near Navigation.",         "Player 5: brown"),
        ("I was doing tasks the whole time. I was nowhere near the body.",   "Player 4: green"),
        ("I was with Player 6 in Admin, we can vouch for each other.",       "Player 2: yellow"),
    ]

    for text, actor in tests:
        claims = extractor.extract(text, actor)
        print(f"ACTOR: {actor}")
        print(f"TEXT:  {text[:100]}")
        for c in claims:
            print(f"  [{c['claim_type']:10}] pred={c['predicate']:<25} obj={str(c['object']):<30} conf={c['confidence']}")
        print()
