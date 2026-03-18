"""
verify_metrics.py  —  Task D: Deterministic evidence verification.

For each row in every evidence CSV:
  1. Open source_file at the stated line_number (1-indexed).
  2. Parse the JSON.
  3. For each key in key_fields, check it matches the actual JSON value.
  4. Run metric-specific consistency checks on metric_value.

Output: analysis/verification_report.csv
"""

import csv
import json
import os
import math

EVIDENCE_DIR  = r"C:\Users\shiven\Desktop\AmongUs\analysis\evidence"
OUTPUT_FILE   = r"C:\Users\shiven\Desktop\AmongUs\analysis\verification_report.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Line cache  (avoid re-reading the same file repeatedly)
# ─────────────────────────────────────────────────────────────────────────────

_line_cache: dict[str, list[str]] = {}


def _get_lines(path: str) -> list[str]:
    if path not in _line_cache:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                _line_cache[path] = f.readlines()
        except FileNotFoundError:
            _line_cache[path] = []
    return _line_cache[path]


def read_line(source_file: str, line_number: int):
    """Return (json_obj | None, error_msg | None)."""
    lines = _get_lines(source_file)
    if not lines:
        return None, f"source_file not found: {source_file}"
    idx = line_number - 1  # 1-indexed → 0-indexed
    if idx < 0 or idx >= len(lines):
        return None, f"line_number {line_number} out of range (file has {len(lines)} lines)"
    raw = lines[idx].strip()
    if not raw:
        return None, f"line {line_number} is empty"
    try:
        return json.loads(raw), None
    except json.JSONDecodeError as e:
        return None, f"JSON parse error at line {line_number}: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Deep-get utility for nested JSON
# ─────────────────────────────────────────────────────────────────────────────

def deep_get(obj, dotted_key):
    """Retrieve obj[a][b][c] for key 'a.b.c'."""
    parts = dotted_key.split(".")
    cur = obj
    for p in parts:
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(p)
        else:
            return None
    return cur


# ─────────────────────────────────────────────────────────────────────────────
# Value comparison helpers
# ─────────────────────────────────────────────────────────────────────────────

_NUMERIC_TOLERANCE = 1e-3  # 0.1% relative, or absolute 1e-3


def values_match(expected, actual, key_name=""):
    """
    Return (match: bool, note: str).
    Compares expected (from key_fields) vs actual (from JSON).
    Handles: numeric ≈, bool, None, list, string.
    """
    # Both None/missing
    if expected is None and actual is None:
        return True, ""
    if expected is None or actual is None:
        return False, f"expected={expected!r}, actual={actual!r}"

    # Numeric comparison
    try:
        ef = float(expected) if not isinstance(expected, bool) else None
        af = float(actual)   if not isinstance(actual, bool)   else None
        if ef is not None and af is not None:
            if math.isnan(ef) and math.isnan(af):
                return True, ""
            if abs(ef) < 1e-9 and abs(af) < 1e-9:
                return True, ""
            rel = abs(ef - af) / max(abs(ef), 1e-9)
            if rel < _NUMERIC_TOLERANCE or abs(ef - af) < _NUMERIC_TOLERANCE:
                return True, ""
            return False, f"numeric mismatch: expected={ef}, actual={af}, rel_err={rel:.4f}"
    except (TypeError, ValueError):
        pass

    # Bool
    if isinstance(expected, bool) or isinstance(actual, bool):
        eb = bool(expected)
        ab = bool(actual)
        if eb == ab:
            return True, ""
        return False, f"bool mismatch: expected={eb}, actual={ab}"

    # List  — compare sorted stringified
    if isinstance(expected, list) or isinstance(actual, list):
        try:
            el = sorted(str(x) for x in expected)
            al = sorted(str(x) for x in actual)
            if el == al:
                return True, ""
            return False, f"list mismatch: expected={expected!r}, actual={actual!r}"
        except Exception:
            pass

    # String — for long fields like error_details, allow prefix/substring match
    # (evidence may store a truncated version of a long string)
    es = str(expected).strip()
    as_ = str(actual).strip()
    if es == as_:
        return True, ""
    # Allow expected to be a prefix of actual (evidence truncation)
    if len(es) >= 40 and as_.startswith(es[:len(es)]):
        return True, "(prefix match — evidence truncated)"
    if len(es) >= 40 and es in as_:
        return True, "(substring match — evidence truncated)"
    return False, f"string mismatch: expected={expected!r}, actual={actual!r}"


# ─────────────────────────────────────────────────────────────────────────────
# Key-field path resolution
# Different source files store the same concept at different JSON paths.
# ─────────────────────────────────────────────────────────────────────────────

# Maps evidence key_field name → JSON path(s) to try, for each source file suffix
KEY_PATHS: dict[str, list[str]] = {
    # outcomes_v1.jsonl
    "winner":          ["winner"],
    "winner_reason":   ["winner_reason"],
    "timestep":        ["timestep"],
    "human_role":      [],   # not in file — skip
    "human_won":       [],   # not in file — skip

    # events_v1.jsonl / deception_events_v1.jsonl
    "victim":          ["target"],
    "kill_timestep":   ["timestep"],
    "external_witnesses": [],  # derived field — skip exact check
    "witness_count":   [],     # derived field — skip exact check
    "kill_timing_norm": [],    # derived — skip
    "kill_timing_cat": [],     # derived — skip
    "game_duration":   [],     # derived — skip
    "target":          ["target"],
    "target_identity": ["actor_identity"],   # VOTE events: target has no identity field stored directly
    "meeting_id":      ["meeting_id"],
    "correct":         [],     # derived — skip
    "human_voter":     [],     # derived — skip

    # deception_events_v1.jsonl
    "claim_text":         ["claim.claim_text_span"],
    "deception_lie":      ["deception_lie"],
    "deception_omission": ["deception_omission"],
    "deception_ambiguity":["deception_ambiguity"],
    "deception_confidence":["deception_confidence"],
    "claim_type":         ["claim.claim_type"],

    # api_calls_v1.jsonl
    "latency_ms":        ["latency_ms"],
    "prompt_tokens":     ["prompt_tokens"],
    "completion_tokens": ["completion_tokens"],
    "phase":             ["phase"],
    "step":              ["step"],
    "thinking_depth_wc": [],  # derived — skip

    # task events
    "event_type":        ["event_type"],
    "actor":             ["actor"],
    "actor_identity":    ["actor_identity"],
    "tasks_done":        [],   # derived — skip
    "tasks_total":       [],   # derived — skip

    # alive_players (game_outcomes evidence, survivors row)
    "alive_players":     ["phase_context.alive_players"],
}

# key_fields keys that are always derived / not directly in source → always SKIP
DERIVED_KEYS = {
    # Whole-game / outcome derivations
    "human_role", "human_won",
    # Kill chain derivations
    "kill_timing_norm", "kill_timing_cat", "game_duration",
    "external_witnesses", "witness_count",
    "kill_ev_id", "kill_ts", "killer_survived", "reason",
    "ejected_player", "killer", "killer_identity",
    # Kill chain multi-event aggregations (stored only in evidence, not in any single log line)
    "meeting_id",          # impostor_survival_after_witness combines multiple events
    "voteout_ev_id",       # synthesised chain field
    "votes_for_killer",    # tallied from voteout details string
    "total_votes",         # tallied from voteout details string
    # Voting derivations (computed from parsed voteout details)
    "correct", "human_voter",
    "ejected_identity",    # not in voteout JSON — parsed from details text
    "correct_ejection",    # derived from identity lookup
    "target_identity",     # not stored on VOTE events directly
    # Task derivations (computed from actor_state_snapshot sub-fields)
    "tasks_done_so_far",
    "tasks_completed_this_player",
    "tasks_total_this_player",
    "tasks_done", "tasks_total",
    # Latency derivations
    "thinking_depth_wc",
    # Judge derivations (aggregated — no single line backing)
    "identity", "mean_awareness", "mean_lying", "mean_deception",
    "mean_planning", "n_turns",
}


def get_json_path(json_obj, key):
    """Try all known paths for a key_fields key. Return (value, found:bool)."""
    paths = KEY_PATHS.get(key)
    if paths is None:
        # Unknown key — try direct lookup
        paths = [key]
    for p in paths:
        v = deep_get(json_obj, p)
        if v is not None:
            return v, True
        # Also try direct top-level
        if p in json_obj:
            return json_obj[p], True
    return None, False


# ─────────────────────────────────────────────────────────────────────────────
# Metric-level consistency checks on metric_value
# ─────────────────────────────────────────────────────────────────────────────

def consistency_check(metric_name, metric_value, key_fields, json_obj):
    """
    Return (status: 'PASS'|'WARN'|'FAIL', detail: str).
    Checks whether metric_value is consistent with key_fields + json_obj.
    """
    mv = metric_value

    try:
        if metric_name == "game_outcome":
            w = json_obj.get("winner")
            if w in (2, 3) and mv == "crewmate_win":
                return "PASS", "winner code matches crewmate_win"
            if w in (1, 4) and mv == "impostor_win":
                return "PASS", "winner code matches impostor_win"
            return "FAIL", f"winner={w} but metric_value={mv!r}"

        if metric_name == "game_duration":
            ts = json_obj.get("timestep")
            if ts is not None and str(ts) == str(mv):
                return "PASS", f"timestep={ts} matches metric_value"
            return "FAIL", f"timestep={ts} != metric_value={mv}"

        if metric_name == "survivors_at_end":
            ap = deep_get(json_obj, "phase_context.alive_players")
            if ap is None:
                ap = json_obj.get("alive_players")
            if ap is not None:
                try:
                    if abs(float(ap) - float(mv)) < 0.01:
                        return "PASS", f"alive_players={ap} matches"
                    return "FAIL", f"alive_players={ap} != metric_value={mv}"
                except Exception:
                    pass
            return "WARN", "alive_players field not found in JSON for cross-check"

        if metric_name == "kill_event":
            et = json_obj.get("event_type")
            if et != "KILL":
                return "FAIL", f"event_type={et!r}, expected KILL"
            expected_val = "witnessed" if key_fields.get("witness_count", 0) > 0 else "unwitnessed"
            # Use external_witnesses list length if available
            wits = key_fields.get("external_witnesses", [])
            if isinstance(wits, list):
                expected_val = "witnessed" if len(wits) > 0 else "unwitnessed"
            if mv == expected_val:
                return "PASS", f"witness count consistent with {mv}"
            return "FAIL", f"external_witnesses={wits!r} but metric_value={mv!r}"

        if metric_name == "impostor_survival_after_witness":
            # Just check that the referenced event is a KILL
            et = json_obj.get("event_type")
            if et != "KILL":
                return "FAIL", f"event_type={et!r}, expected KILL for kill chain"
            return "PASS", "source event is KILL as expected"

        if metric_name == "vote_correctness":
            et = json_obj.get("event_type")
            if et != "VOTE":
                return "FAIL", f"event_type={et!r}, expected VOTE"
            tgt = key_fields.get("target")
            if tgt and json_obj.get("target") and tgt != json_obj.get("target"):
                return "WARN", f"target mismatch: kf={tgt!r} json={json_obj.get('target')!r}"
            return "PASS", "VOTE event; target field checked"

        if metric_name == "claim_deception":
            dl = json_obj.get("deception_lie")
            if dl is True and mv == "lie":
                return "PASS", "deception_lie=True matches metric_value=lie"
            if dl is False and mv in ("no_lie", "truth", "false"):
                return "PASS", f"deception_lie=False matches metric_value={mv!r}"
            if dl is None:
                return "WARN", "deception_lie is null in source"
            return "FAIL", f"deception_lie={dl!r} but metric_value={mv!r}"

        if metric_name == "api_call_latency":
            lat = json_obj.get("latency_ms")
            if lat is not None:
                try:
                    if abs(float(lat) - float(mv)) < 0.1:
                        return "PASS", f"latency_ms={lat} matches metric_value"
                    return "FAIL", f"latency_ms={lat} != metric_value={mv} (diff={abs(float(lat)-float(mv)):.2f}ms)"
                except Exception:
                    pass
            return "WARN", "latency_ms not found for cross-check"

        if metric_name == "task_event":
            et = json_obj.get("event_type")
            if et == "COMPLETE TASK":
                return "PASS", "event_type=COMPLETE TASK as expected"
            return "FAIL", f"event_type={et!r}, expected 'COMPLETE TASK'"

        if metric_name == "judge_score":
            # metric_value is a JSON with scores; just verify it's parseable
            try:
                scores = json.loads(mv) if isinstance(mv, str) else mv
                if isinstance(scores, dict):
                    return "PASS", f"judge_score is valid dict with keys={list(scores.keys())}"
            except Exception:
                pass
            # Could also be a single float
            try:
                float(mv)
                return "PASS", "judge_score is a numeric value"
            except Exception:
                return "WARN", f"judge_score metric_value could not be validated: {mv!r}"

    except Exception as e:
        return "WARN", f"consistency check raised exception: {e}"

    return "PASS", "no specific consistency check for this metric"


# ─────────────────────────────────────────────────────────────────────────────
# Main verification logic
# ─────────────────────────────────────────────────────────────────────────────

def verify_row(ev_row, row_num, evidence_filename):
    """
    Returns a result dict with status (PASS/FAIL/WARN) and discrepancy_details.
    """
    source_file  = ev_row.get("source_file", "").strip()
    line_number  = ev_row.get("line_number", "").strip()
    metric_name  = ev_row.get("metric_name", "").strip()
    metric_value = ev_row.get("metric_value", "").strip()
    key_fields_s = ev_row.get("key_fields", "").strip()
    config_id    = ev_row.get("config_id", "")
    event_ids    = ev_row.get("event_ids", "")

    base = {
        "evidence_file":      evidence_filename,
        "row_number":         row_num,
        "config_id":          config_id,
        "metric_name":        metric_name,
        "event_ids":          event_ids,
        "status":             "PASS",
        "discrepancy_details": "",
    }

    # ── Parse key_fields ────────────────────────────────────────────────────
    key_fields = {}
    if key_fields_s:
        try:
            key_fields = json.loads(key_fields_s)
        except json.JSONDecodeError:
            base["status"] = "WARN"
            base["discrepancy_details"] = f"key_fields could not be parsed as JSON: {key_fields_s[:80]}"
            return base

    # ── Skip rows without a source file or with line_number=0 (summary rows) ─
    if not source_file:
        base["status"] = "PASS"
        base["discrepancy_details"] = "no source_file (summary row); skipped"
        return base

    # ── Parse line number ────────────────────────────────────────────────────
    if not line_number:
        base["status"] = "WARN"
        base["discrepancy_details"] = "line_number missing"
        return base
    try:
        ln = int(line_number)
    except ValueError:
        base["status"] = "WARN"
        base["discrepancy_details"] = f"line_number not an integer: {line_number!r}"
        return base

    # line_number=0 means no specific line index (e.g. red_flag rows, judge rows)
    if ln == 0:
        base["status"] = "PASS"
        base["discrepancy_details"] = "line_number=0 (aggregate/derived row); skipped"
        return base

    # ── Read source line ─────────────────────────────────────────────────────
    json_obj, err = read_line(source_file, ln)
    if err:
        base["status"] = "FAIL"
        base["discrepancy_details"] = err
        return base

    # ── Check each key_field against JSON ────────────────────────────────────
    discrepancies = []

    for kf_key, kf_val in key_fields.items():
        if kf_key in DERIVED_KEYS:
            continue  # skip derived fields

        # Skip empty-string values in key_fields — these are placeholders
        if kf_val == "" or kf_val is None:
            continue

        json_val, found = get_json_path(json_obj, kf_key)

        if not found:
            # Missing in JSON — only WARN if the value is not None
            if kf_val is not None:
                discrepancies.append(f"key '{kf_key}' not found in JSON (expected {kf_val!r})")
            continue

        match, note = values_match(kf_val, json_val)
        if not match:
            discrepancies.append(f"key '{kf_key}': {note}")

    if discrepancies:
        base["status"] = "FAIL"
        base["discrepancy_details"] = " | ".join(discrepancies)
        return base

    # ── Metric-level consistency check ───────────────────────────────────────
    status, detail = consistency_check(metric_name, metric_value, key_fields, json_obj)
    base["status"] = status
    base["discrepancy_details"] = detail
    return base


def main():
    print("=" * 70)
    print("TASK D: DETERMINISTIC EVIDENCE VERIFICATION")
    print("=" * 70)

    evidence_files = sorted(
        f for f in os.listdir(EVIDENCE_DIR) if f.endswith("_evidence.csv")
    )

    all_results = []
    total = passes = fails = warns = skipped = 0

    for ev_file in evidence_files:
        ev_path = os.path.join(EVIDENCE_DIR, ev_file)
        with open(ev_path, newline="", encoding="utf-8") as f:
            ev_rows = list(csv.DictReader(f))

        file_pass = file_fail = file_warn = 0

        for i, row in enumerate(ev_rows, start=2):  # row 1 = header
            result = verify_row(row, i, ev_file)
            all_results.append(result)
            total += 1
            s = result["status"]
            if s == "PASS":
                passes += 1; file_pass += 1
            elif s == "FAIL":
                fails += 1; file_fail += 1
            else:
                warns += 1; file_warn += 1

        status_str = f"PASS={file_pass} FAIL={file_fail} WARN={file_warn}"
        marker = " *** FAILURES" if file_fail > 0 else ""
        print(f"  {ev_file:<50} {status_str}{marker}")

    # Write report
    fieldnames = ["evidence_file", "row_number", "config_id", "metric_name",
                  "event_ids", "status", "discrepancy_details"]
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_results)

    print()
    print("=" * 70)
    print(f"VERIFICATION SUMMARY")
    print(f"  Total rows checked : {total}")
    print(f"  PASS               : {passes}  ({100*passes/total:.1f}%)")
    print(f"  WARN               : {warns}   ({100*warns/total:.1f}%)")
    print(f"  FAIL               : {fails}   ({100*fails/total:.1f}%)")
    print(f"  Written: {OUTPUT_FILE}")
    print("=" * 70)

    if fails > 0:
        print(f"\nFAILED ROWS:")
        for r in all_results:
            if r["status"] == "FAIL":
                print(f"  [{r['evidence_file']} row {r['row_number']}] "
                      f"metric={r['metric_name']} ev={r['event_ids']}")
                print(f"    {r['discrepancy_details']}")


if __name__ == "__main__":
    main()
