#!/usr/bin/env python3
"""
step1_validate_configs.py

Validates all completed Among Us human-vs-LLM experiments from Feb 26, 2026 onwards.

Reads exclusively from structured-v1/ JSONL files and agent-logs-compact.json
(for prompt_profile, which is not captured in structured-v1/runs.jsonl).

Outputs: analysis/config_mapping.csv

CRITICAL RULES:
- No in-head computation. All values come from actual file reads.
- Every claim traces to a specific file path.
- Missing files are logged as warnings; values are left as None/False, never imputed.
"""

import csv
import json
import os
import re
import sys
from collections import defaultdict, Counter
from datetime import date

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

CUTOFF_DATE = "2026-02-26"  # inclusive

# All directories to scan. Each entry: (label, path, forced_human_role or None)
# forced_human_role is used for the llama experiment folders which encode the
# human role in the folder name rather than having the human in agent_turns.
SCAN_ROOTS = [
    ("shiven_expt_logs",  r"C:\Users\shiven\Desktop\AmongUs\expt-logs",                            None),
    ("aadi_expt_logs",    r"C:\Users\shiven\Desktop\AmongUs\aadi-expt-logs\expt-logs",              None),
    ("llama_crewmate",    r"C:\Users\shiven\Desktop\AmongUs\amongus_llama_human_crewmate",          "crewmate"),
    ("llama_impostor",    r"C:\Users\shiven\Desktop\AmongUs\amongus_llama_human_impostor",          "impostor"),
    ("human_trials",      r"C:\Users\shiven\Desktop\AmongUs\human_trials",                          None),
]

# Full expected set of structured-v1 files
CORE_FILES = {
    "runs.jsonl", "events_v1.jsonl", "agent_turns_v1.jsonl",
    "api_calls_v1.jsonl", "outcomes_v1.jsonl",
}
INFERENCE_FILES = {
    "deception_events_v1.jsonl", "deception_opportunities_v1.jsonl",
    "listener_outcomes_v1.jsonl", "red_flags_v1.jsonl",
    "round_covariates_v1.jsonl", "replay_consistency_checks_v1.jsonl",
    "annotation_v1.jsonl", "inference_metrics_v1.json",
}
ALL_EXPECTED_FILES = CORE_FILES | INFERENCE_FILES

# Config ID mapping: (normalized_model, human_role, prompt_class) -> config_id
# normalized_model strips provider prefix and -instruct suffix
CONFIG_MAP = {
    ("claude-3.5-haiku",  "crewmate", "baseline"):    "C01",
    ("claude-3.5-haiku",  "impostor", "baseline"):    "C02",
    ("gemini-2.5-flash",  "crewmate", "baseline"):    "C03",
    ("gemini-2.5-flash",  "impostor", "baseline"):    "C04",
    ("gemini-2.5-flash",  "crewmate", "aggressive"):  "C05",
    ("gemini-2.5-flash",  "impostor", "aggressive"):  "C06",
    ("llama-3.1-8b",      "crewmate", "baseline"):    "C07",
    ("llama-3.1-8b",      "impostor", "baseline"):    "C08",
}

CONFIG_EXPECTED_N = {
    "C01": 10, "C02": 10, "C03": 10, "C04": 10,
    "C05": 5,  "C06": 5,  "C07": 10, "C08": 10,
}

OUTPUT_DIR = r"C:\Users\shiven\Desktop\AmongUs\analysis"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "config_mapping.csv")

WARNINGS = []

def warn(msg):
    WARNINGS.append(msg)
    print(f"  [WARN] {msg}", file=sys.stderr)

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def read_jsonl_first(path):
    """Return first valid JSON object from a JSONL file, or None with a warning."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    return json.loads(raw)
                except json.JSONDecodeError as e:
                    warn(f"JSON parse error in {path}: {e}")
                    continue
    except OSError as e:
        warn(f"Cannot open {path}: {e}")
    return None

def read_jsonl_all(path):
    """Return all valid JSON objects from a JSONL file. Returns [] if missing."""
    if not os.path.exists(path):
        return []
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for i, raw in enumerate(f, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rows.append(json.loads(raw))
                except json.JSONDecodeError as e:
                    warn(f"JSON parse error at line {i} in {path}: {e}")
    except OSError as e:
        warn(f"Cannot open {path}: {e}")
    return rows

def normalize_model(raw_model):
    """
    Strip provider prefix and normalize to a canonical short name.
    Returns the normalized name and a flag indicating if it's the human model.
    """
    if raw_model is None:
        return None, False
    m = str(raw_model).strip().lower()
    if "homosapiens" in m or "brain-1.0" in m:
        return "human", True
    # Strip provider prefix (e.g. "anthropic/", "google/", "meta-llama/", "openai/")
    if "/" in m:
        m = m.split("/", 1)[1]
    # Strip trailing -instruct, :free, etc.
    m = re.sub(r"[-:]instruct$", "", m)
    m = re.sub(r":free$", "", m)
    return m, False

def normalize_prompt(raw_profile):
    """Map prompt_profile strings to 'baseline' or 'aggressive'."""
    if raw_profile is None:
        return None
    p = str(raw_profile).strip().lower()
    if p.startswith("aggressive"):
        return "aggressive"
    if p.startswith("baseline") or p == "":
        return "baseline"
    return p  # unknown

def get_experimenter(source_label, runs_data):
    """Infer experimenter from source label or runs.jsonl experiment_path."""
    if "aadi" in source_label.lower():
        return "aadi"
    if runs_data:
        exp_path = runs_data.get("experiment_path", "")
        if "Aadi" in exp_path or "aadi" in exp_path.lower():
            return "aadi"
    return "shiven"

# ---------------------------------------------------------------------------
# PROMPT PROFILE DETECTION
# ---------------------------------------------------------------------------

def detect_prompt_profile(exp_dir, runs_data):
    """
    Detect prompt profile from multiple sources in priority order:
    1. agent-logs-compact.json (JSONL) — most reliable; logs prompt_config per turn
    2. runs.jsonl env_snapshot (rarely populated in current codebase)
    3. Default to 'baseline_v1' with a warning
    Returns (raw_profile_string, source_used)
    """
    # Source 1: agent-logs-compact.json
    compact_path = os.path.join(exp_dir, "agent-logs-compact.json")
    if os.path.exists(compact_path):
        try:
            with open(compact_path, encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    entries = obj if isinstance(obj, list) else [obj]
                    for e in entries:
                        pc = e.get("prompt_config") or {}
                        prof = pc.get("prompt_profile")
                        if prof:
                            return prof, "agent-logs-compact.json"
        except OSError:
            pass

    # Source 2: runs.jsonl env_snapshot
    if runs_data:
        env = runs_data.get("env_snapshot", {})
        prof = env.get("prompt_profile") or env.get("PROMPT_PROFILE")
        if prof:
            return prof, "runs.jsonl:env_snapshot"

    # Source 3: default
    warn(f"{exp_dir}: prompt_profile not found in any source; defaulting to baseline_v1")
    return "baseline_v1", "default"

# ---------------------------------------------------------------------------
# CORE EXTRACTION
# ---------------------------------------------------------------------------

def extract_from_agent_turns(agent_turns_path, forced_human_role=None):
    """
    Read agent_turns_v1.jsonl and extract:
    - human_role: identity of the player with model=homosapiens/brain-1.0
    - llm_model: canonical model name of non-human agents
    - human_found: bool
    Returns dict with keys: llm_model, human_role, human_found, all_models_seen
    """
    result = {
        "llm_model": None,
        "human_role": None,
        "human_found": False,
        "all_models_seen": set(),
    }
    if not os.path.exists(agent_turns_path):
        warn(f"MISSING: {agent_turns_path}")
        if forced_human_role:
            result["human_role"] = forced_human_role
        return result

    llm_models = Counter()
    human_identity = None

    rows = read_jsonl_all(agent_turns_path)
    for r in rows:
        agent = r.get("agent", {})
        raw_model = agent.get("model", "")
        norm_model, is_human = normalize_model(raw_model)
        result["all_models_seen"].add(raw_model)
        if is_human:
            result["human_found"] = True
            identity = agent.get("identity", "").strip().lower()
            if identity in ("crewmate", "impostor"):
                human_identity = identity
        else:
            if norm_model:
                llm_models[norm_model] += 1

    if human_identity:
        result["human_role"] = human_identity
    elif forced_human_role:
        result["human_role"] = forced_human_role

    if llm_models:
        result["llm_model"] = llm_models.most_common(1)[0][0]
    elif not result["human_found"]:
        warn(f"No human or LLM agents found in {agent_turns_path}")

    return result


def extract_outcome(outcomes_path):
    """
    Read outcomes_v1.jsonl and return (game_complete, winner, winner_reason, timestep).
    """
    if not os.path.exists(outcomes_path):
        return False, None, None, None
    rows = read_jsonl_all(outcomes_path)
    if not rows:
        warn(f"Empty outcomes file: {outcomes_path}")
        return False, None, None, None
    row = rows[0]
    winner = row.get("winner")
    winner_reason = row.get("winner_reason", "")
    timestep = row.get("timestep")
    game_complete = winner is not None
    return game_complete, winner, winner_reason, timestep


def check_files(sv_dir):
    """Return dict of {filename: bool} for all expected structured-v1 files."""
    result = {}
    for fname in sorted(ALL_EXPECTED_FILES):
        result[fname] = os.path.exists(os.path.join(sv_dir, fname))
    return result


def map_config_id(llm_model, human_role, prompt_class):
    """Look up config_id. Returns (config_id, mapped) where mapped=False means no match."""
    key = (llm_model, human_role, prompt_class)
    cid = CONFIG_MAP.get(key)
    return cid, cid is not None

# ---------------------------------------------------------------------------
# MAIN SCAN
# ---------------------------------------------------------------------------

def scan_all():
    records = []
    exp_dir_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2})_exp_(\d+)$")

    for source_label, root, forced_human_role in SCAN_ROOTS:
        if not os.path.exists(root):
            warn(f"Root directory not found: {root}")
            continue

        entries = sorted(os.listdir(root))
        for entry in entries:
            m = exp_dir_pattern.match(entry)
            if not m:
                continue

            date_str = m.group(1)
            exp_num  = int(m.group(2))
            exp_dir  = os.path.join(root, entry)

            # Date filter
            if date_str < CUTOFF_DATE:
                continue

            sv_dir           = os.path.join(exp_dir, "structured-v1")
            runs_path        = os.path.join(sv_dir, "runs.jsonl")
            agent_turns_path = os.path.join(sv_dir, "agent_turns_v1.jsonl")
            outcomes_path    = os.path.join(sv_dir, "outcomes_v1.jsonl")

            if not os.path.isdir(sv_dir):
                warn(f"No structured-v1/ dir in {exp_dir}; skipping")
                continue

            notes_parts = []

            # --- runs.jsonl ---
            runs_data = read_jsonl_first(runs_path)
            run_id = None
            if runs_data:
                run_id = runs_data.get("run_id")
                tournament = runs_data.get("tournament", {})
                run_label  = tournament.get("run_label", "")
                if run_label:
                    notes_parts.append(f"run_label={run_label}")
                # Check for human_role in tournament block
                tourn_human_role = tournament.get("human_role", "")
            else:
                warn(f"{entry}: runs.jsonl missing or empty")
                run_label = ""
                tourn_human_role = ""

            if run_id is None:
                run_id = f"{date_str}_exp_{exp_num}"
                notes_parts.append("run_id inferred from dir name")

            # --- agent_turns_v1.jsonl ---
            turn_info = extract_from_agent_turns(agent_turns_path, forced_human_role)
            llm_model   = turn_info["llm_model"]
            human_role  = turn_info["human_role"]
            all_models  = turn_info["all_models_seen"]

            # Fallback: human_role from tournament block in runs.jsonl
            if human_role is None and tourn_human_role:
                human_role = tourn_human_role.lower()
                notes_parts.append("human_role from runs.jsonl:tournament")

            # Fallback: llm_model from runs.jsonl env_snapshot
            if llm_model is None and runs_data:
                env = runs_data.get("env_snapshot", {})
                raw_crew = env.get("openrouter_crewmate_model")
                raw_imp  = env.get("openrouter_impostor_model")
                if human_role == "crewmate" and raw_imp:
                    llm_model, _ = normalize_model(raw_imp)
                    notes_parts.append("llm_model from runs.jsonl env_snapshot")
                elif human_role == "impostor" and raw_crew:
                    llm_model, _ = normalize_model(raw_crew)
                    notes_parts.append("llm_model from runs.jsonl env_snapshot")
                elif raw_crew:
                    llm_model, _ = normalize_model(raw_crew)
                    notes_parts.append("llm_model from runs.jsonl env_snapshot (human role unknown)")

            # Mixed-model warning
            non_human_models = {
                normalize_model(m)[0]
                for m in all_models
                if not normalize_model(m)[1] and normalize_model(m)[0]
            }
            if len(non_human_models) > 1:
                notes_parts.append(f"mixed_models:{sorted(non_human_models)}")
                warn(f"{entry}: multiple LLM models detected: {non_human_models}")

            # --- prompt profile ---
            raw_profile, profile_source = detect_prompt_profile(exp_dir, runs_data)
            prompt_class = normalize_prompt(raw_profile)
            if profile_source == "default":
                notes_parts.append("prompt_profile=default_assumption")

            # --- outcomes ---
            game_complete, winner, winner_reason, timestep = extract_outcome(outcomes_path)

            # --- file presence ---
            file_flags = check_files(sv_dir)

            # --- experimenter ---
            experimenter = get_experimenter(source_label, runs_data)

            # --- config ID mapping ---
            config_id, mapped = map_config_id(llm_model, human_role, prompt_class)
            if not mapped:
                notes_parts.append(
                    f"NO_CONFIG_MATCH(model={llm_model},role={human_role},prompt={prompt_class})"
                )
                warn(f"{entry}: does not match any config: model={llm_model} role={human_role} prompt={prompt_class}")

            # --- missing inference files ---
            missing_inference = [
                f for f in INFERENCE_FILES if not file_flags.get(f, False)
            ]
            if missing_inference:
                notes_parts.append(f"missing_inference_files={len(missing_inference)}")

            # --- missing core files ---
            missing_core = [
                f for f in CORE_FILES if not file_flags.get(f, False)
            ]
            if missing_core:
                notes_parts.append(f"missing_core_files:{missing_core}")
                warn(f"{entry}: missing core structured-v1 files: {missing_core}")

            records.append({
                "run_id":                   run_id,
                "date":                     date_str,
                "config_id":                config_id or "UNMAPPED",
                "llm_model":                llm_model or "UNKNOWN",
                "human_role":               human_role or "UNKNOWN",
                "prompt_profile":           raw_profile,
                "prompt_class":             prompt_class,
                "game_complete":            game_complete,
                "winner":                   winner,
                "winner_reason":            winner_reason or "",
                "game_duration_timesteps":  timestep,
                "experimenter":             experimenter,
                "source_label":             source_label,
                "experiment_dir":           entry,
                "has_events":               file_flags.get("events_v1.jsonl", False),
                "has_agent_turns":          file_flags.get("agent_turns_v1.jsonl", False),
                "has_api_calls":            file_flags.get("api_calls_v1.jsonl", False),
                "has_outcomes":             file_flags.get("outcomes_v1.jsonl", False),
                "has_deception_events":     file_flags.get("deception_events_v1.jsonl", False),
                "has_listener_outcomes":    file_flags.get("listener_outcomes_v1.jsonl", False),
                "has_red_flags":            file_flags.get("red_flags_v1.jsonl", False),
                "notes":                    " | ".join(notes_parts),
            })

    return records

# ---------------------------------------------------------------------------
# REPORTING
# ---------------------------------------------------------------------------

def print_section(title):
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)

def print_table(records):
    """Print a compact summary table."""
    hdr = (
        f"{'#':<4} {'Config':<7} {'Date':<12} {'Exp Dir':<22} "
        f"{'LLM Model':<22} {'Role':<10} {'Prompt':<12} "
        f"{'Complete':<10} {'Winner':<8} {'Steps':<6} {'Experimenter'}"
    )
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(records, 1):
        print(
            f"{i:<4} {r['config_id']:<7} {r['date']:<12} {r['experiment_dir']:<22} "
            f"{r['llm_model']:<22} {r['human_role']:<10} {r['prompt_class']:<12} "
            f"{str(r['game_complete']):<10} {str(r['winner']):<8} "
            f"{str(r['game_duration_timesteps']):<6} {r['experimenter']}"
        )

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Scanning experiment directories from {} onwards...".format(CUTOFF_DATE))
    records = scan_all()

    # Sort by config_id then date then exp dir
    records.sort(key=lambda r: (r["config_id"], r["date"], r["experiment_dir"]))

    # -----------------------------------------------------------------------
    # SECTION 1: Full table
    # -----------------------------------------------------------------------
    print_section("FULL CONFIG MAPPING TABLE")
    print_table(records)

    # -----------------------------------------------------------------------
    # SECTION 2: Count per config_id (complete games only)
    # -----------------------------------------------------------------------
    print_section("GAME COUNTS PER CONFIG (complete games only)")
    config_counts  = defaultdict(list)
    for r in records:
        if r["game_complete"]:
            config_counts[r["config_id"]].append(r)

    expected_configs = ["C01","C02","C03","C04","C05","C06","C07","C08","UNMAPPED"]
    print(f"{'Config':<10} {'Expected N':<12} {'Actual N':<10} {'Status'}")
    print("-" * 50)
    for cid in expected_configs:
        exp_n = CONFIG_EXPECTED_N.get(cid, "?")
        actual = len(config_counts.get(cid, []))
        if cid == "UNMAPPED":
            if actual:
                print(f"{'UNMAPPED':<10} {'N/A':<12} {actual:<10} UNEXPECTED GAMES")
        else:
            status = "OK" if actual == exp_n else f"SHORTFALL={exp_n - actual}" if actual < exp_n else f"EXCESS=+{actual - exp_n}"
            print(f"{cid:<10} {str(exp_n):<12} {actual:<10} {status}")

    # -----------------------------------------------------------------------
    # SECTION 3: Incomplete games
    # -----------------------------------------------------------------------
    incomplete = [r for r in records if not r["game_complete"]]
    print_section(f"INCOMPLETE GAMES ({len(incomplete)} total)")
    if incomplete:
        for r in incomplete:
            print(f"  {r['experiment_dir']}  config={r['config_id']}  model={r['llm_model']}  "
                  f"role={r['human_role']}  prompt={r['prompt_class']}")
    else:
        print("  None — all scanned games are complete.")

    # -----------------------------------------------------------------------
    # SECTION 4: Games missing inference-derived files
    # -----------------------------------------------------------------------
    missing_inf = [r for r in records if r["game_complete"] and not r["has_deception_events"]]
    print_section(f"COMPLETE GAMES MISSING INFERENCE FILES ({len(missing_inf)} games)")
    if missing_inf:
        for r in missing_inf:
            print(f"  {r['experiment_dir']}  config={r['config_id']}  source={r['source_label']}")
        print()
        print(f"  ACTION REQUIRED: Run structured_v1_inference.py on the above {len(missing_inf)} directories.")
    else:
        print("  All complete games have inference-derived files.")

    # -----------------------------------------------------------------------
    # SECTION 5: Unmapped games (don't fit any of the 8 configs)
    # -----------------------------------------------------------------------
    unmapped = [r for r in records if r["config_id"] == "UNMAPPED"]
    print_section(f"UNMAPPED GAMES (do not match any of the 8 configs) — {len(unmapped)} total")
    if unmapped:
        for r in unmapped:
            print(f"  {r['experiment_dir']}  model={r['llm_model']}  role={r['human_role']}  "
                  f"prompt={r['prompt_class']}  complete={r['game_complete']}  notes={r['notes']}")
    else:
        print("  None.")

    # -----------------------------------------------------------------------
    # SECTION 6: Data quality warnings
    # -----------------------------------------------------------------------
    print_section(f"DATA QUALITY WARNINGS ({len(WARNINGS)} total)")
    if WARNINGS:
        for i, w in enumerate(WARNINGS, 1):
            print(f"  {i:>3}. {w}")
    else:
        print("  No warnings.")

    # -----------------------------------------------------------------------
    # Write CSV
    # -----------------------------------------------------------------------
    fieldnames = [
        "run_id", "date", "config_id", "llm_model", "human_role",
        "prompt_profile", "prompt_class", "game_complete", "winner",
        "winner_reason", "game_duration_timesteps", "experimenter",
        "source_label", "experiment_dir",
        "has_events", "has_agent_turns", "has_api_calls", "has_outcomes",
        "has_deception_events", "has_listener_outcomes", "has_red_flags",
        "notes",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)

    print_section("OUTPUT")
    print(f"  Saved {len(records)} rows to: {OUTPUT_CSV}")
    print(f"  Total experiments scanned: {len(records)}")
    print(f"  Complete games: {sum(1 for r in records if r['game_complete'])}")
    print(f"  Incomplete games: {len(incomplete)}")
    print(f"  Unmapped games: {len(unmapped)}")
    print(f"  Warnings: {len(WARNINGS)}")


if __name__ == "__main__":
    main()
