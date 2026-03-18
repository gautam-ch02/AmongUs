#!/usr/bin/env python3
"""
step2_fill_gaps.py

Task A: Run structured_v1_inference.py on all complete games that are missing
        deception_events_v1.jsonl, listener_outcomes_v1.jsonl, or red_flags_v1.jsonl.

Task B: Re-scan all experiment directories and verify every complete game now
        has all expected structured-v1 files. Update config_mapping.csv in place.

Task C: Check evaluations/results/ for LLM-judge scores, report coverage,
        and ask the user whether to proceed with eval.py.

Input:  analysis/config_mapping.csv  (written by step1_validate_configs.py)
Output: analysis/config_mapping.csv  (updated in-place with refreshed file flags)
        analysis/step2_inference_log.jsonl  (per-experiment inference run record)

CRITICAL RULES:
- No in-head computation. All values come from actual file reads or subprocess calls.
- Failures are logged and printed; we never skip silently.
- config_mapping.csv is only written after all verification passes complete.
"""

import csv
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

ANALYSIS_DIR     = r"C:\Users\shiven\Desktop\AmongUs\analysis"
CONFIG_CSV       = os.path.join(ANALYSIS_DIR, "config_mapping.csv")
INFERENCE_LOG    = os.path.join(ANALYSIS_DIR, "step2_inference_log.jsonl")
INFERENCE_SCRIPT = r"C:\Users\shiven\Desktop\AmongUs\evaluations\structured_v1_inference.py"
EVALS_DIR        = r"C:\Users\shiven\Desktop\AmongUs\evaluations"
EVALS_RESULTS    = os.path.join(EVALS_DIR, "results")
EVAL_SCRIPT      = os.path.join(EVALS_DIR, "eval.py")

SOURCE_ROOTS = {
    "shiven_expt_logs": r"C:\Users\shiven\Desktop\AmongUs\expt-logs",
    "aadi_expt_logs":   r"C:\Users\shiven\Desktop\AmongUs\aadi-expt-logs\expt-logs",
    "llama_crewmate":   r"C:\Users\shiven\Desktop\AmongUs\amongus_llama_human_crewmate",
    "llama_impostor":   r"C:\Users\shiven\Desktop\AmongUs\amongus_llama_human_impostor",
}

# Full set of inference-derived files we expect after running the pipeline
INFERENCE_FILES = {
    "deception_events_v1.jsonl",
    "deception_opportunities_v1.jsonl",
    "listener_outcomes_v1.jsonl",
    "red_flags_v1.jsonl",
    "round_covariates_v1.jsonl",
    "replay_consistency_checks_v1.jsonl",
    "annotation_v1.jsonl",
    "inference_metrics_v1.json",
}

CORE_FILES = {
    "runs.jsonl",
    "events_v1.jsonl",
    "agent_turns_v1.jsonl",
    "api_calls_v1.jsonl",
    "outcomes_v1.jsonl",
}

ALL_EXPECTED = CORE_FILES | INFERENCE_FILES

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def print_section(title):
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)

def log_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")

def get_sv_dir(source_label, experiment_dir):
    root = SOURCE_ROOTS.get(source_label)
    if root is None:
        return None
    return os.path.join(root, experiment_dir, "structured-v1")

def check_file_flags(sv_dir):
    """Return dict {filename: bool} for all expected files."""
    return {fname: os.path.exists(os.path.join(sv_dir, fname))
            for fname in sorted(ALL_EXPECTED)}

def count_turns_in_experiment(sv_dir):
    """Count lines in agent_turns_v1.jsonl. Returns (total_turns, llm_turns)."""
    at_path = os.path.join(sv_dir, "agent_turns_v1.jsonl")
    if not os.path.exists(at_path):
        return 0, 0
    total = 0
    llm = 0
    try:
        with open(at_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    obj = json.loads(line)
                    model = obj.get("agent", {}).get("model", "")
                    if "homosapiens" not in model and "brain" not in model:
                        llm += 1
                except json.JSONDecodeError:
                    pass
    except OSError:
        pass
    return total, llm

# ---------------------------------------------------------------------------
# TASK A: Run inference on experiments missing inference files
# ---------------------------------------------------------------------------

def task_a_run_inference(rows):
    """
    Identify complete games missing any inference-derived file,
    run structured_v1_inference.py on each, and return a results list.
    """
    print_section("TASK A — Running Inference on Missing Experiments")

    needs_inference = [
        r for r in rows
        if r["game_complete"] == "True" and (
            r["has_deception_events"]  == "False" or
            r["has_listener_outcomes"] == "False" or
            r["has_red_flags"]         == "False"
        )
    ]

    print(f"  Complete games missing inference files: {len(needs_inference)}")
    print()

    if not os.path.exists(INFERENCE_SCRIPT):
        print(f"  ERROR: Inference script not found at {INFERENCE_SCRIPT}")
        sys.exit(1)

    # Clear previous log (fresh run)
    if os.path.exists(INFERENCE_LOG):
        os.remove(INFERENCE_LOG)

    results = []
    success_count = 0
    failure_count = 0

    for i, row in enumerate(needs_inference, 1):
        exp_dir     = row["experiment_dir"]
        src_label   = row["source_label"]
        config_id   = row["config_id"]
        sv_dir      = get_sv_dir(src_label, exp_dir)

        prefix = f"  [{i:>3}/{len(needs_inference)}] {exp_dir} (config={config_id}, src={src_label})"

        if sv_dir is None:
            msg = f"Unknown source_label '{src_label}'"
            print(f"{prefix} — SKIP: {msg}")
            record = {"exp_dir": exp_dir, "status": "skip", "reason": msg,
                      "timestamp": datetime.now(timezone.utc).isoformat()}
            log_jsonl(INFERENCE_LOG, record)
            results.append((exp_dir, "skip", msg))
            failure_count += 1
            continue

        if not os.path.isdir(sv_dir):
            msg = f"structured-v1 dir not found: {sv_dir}"
            print(f"{prefix} — SKIP: {msg}")
            record = {"exp_dir": exp_dir, "status": "skip", "reason": msg,
                      "timestamp": datetime.now(timezone.utc).isoformat()}
            log_jsonl(INFERENCE_LOG, record)
            results.append((exp_dir, "skip", msg))
            failure_count += 1
            continue

        # Check events_v1.jsonl exists (required input for inference)
        if not os.path.exists(os.path.join(sv_dir, "events_v1.jsonl")):
            msg = "events_v1.jsonl missing — cannot run inference"
            print(f"{prefix} — SKIP: {msg}")
            record = {"exp_dir": exp_dir, "status": "skip", "reason": msg,
                      "sv_dir": sv_dir,
                      "timestamp": datetime.now(timezone.utc).isoformat()}
            log_jsonl(INFERENCE_LOG, record)
            results.append((exp_dir, "skip", msg))
            failure_count += 1
            continue

        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, INFERENCE_SCRIPT, "--structured-dir", sv_dir],
                capture_output=True,
                text=True,
                timeout=120,
            )
            elapsed = time.monotonic() - t0

            if proc.returncode == 0:
                # Verify the expected files were actually created
                flags = check_file_flags(sv_dir)
                created = [f for f in INFERENCE_FILES if flags.get(f, False)]
                missing = [f for f in INFERENCE_FILES if not flags.get(f, False)]
                status = "ok" if not missing else "partial"
                msg = f"created={len(created)} missing={missing}"
                print(f"{prefix} — {status.upper()} ({elapsed:.1f}s) {msg}")
                record = {
                    "exp_dir": exp_dir, "config_id": config_id,
                    "source_label": src_label, "sv_dir": sv_dir,
                    "status": status, "elapsed_s": round(elapsed, 2),
                    "files_created": created, "files_missing": missing,
                    "returncode": 0, "stderr": proc.stderr.strip()[:500],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                log_jsonl(INFERENCE_LOG, record)
                results.append((exp_dir, status, msg))
                if status == "ok":
                    success_count += 1
                else:
                    failure_count += 1
            else:
                msg = (proc.stderr or proc.stdout or "non-zero exit").strip()[:300]
                print(f"{prefix} — FAIL (rc={proc.returncode}, {elapsed:.1f}s): {msg}")
                record = {
                    "exp_dir": exp_dir, "config_id": config_id,
                    "source_label": src_label, "sv_dir": sv_dir,
                    "status": "fail", "elapsed_s": round(elapsed, 2),
                    "returncode": proc.returncode,
                    "stderr": proc.stderr.strip()[:500],
                    "stdout": proc.stdout.strip()[:500],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                log_jsonl(INFERENCE_LOG, record)
                results.append((exp_dir, "fail", msg))
                failure_count += 1

        except subprocess.TimeoutExpired:
            msg = "Subprocess timed out after 120s"
            print(f"{prefix} — FAIL: {msg}")
            record = {"exp_dir": exp_dir, "status": "fail", "reason": msg,
                      "timestamp": datetime.now(timezone.utc).isoformat()}
            log_jsonl(INFERENCE_LOG, record)
            results.append((exp_dir, "fail", msg))
            failure_count += 1

        except Exception as e:
            msg = str(e)
            print(f"{prefix} — FAIL: {msg}")
            record = {"exp_dir": exp_dir, "status": "fail", "reason": msg,
                      "timestamp": datetime.now(timezone.utc).isoformat()}
            log_jsonl(INFERENCE_LOG, record)
            results.append((exp_dir, "fail", msg))
            failure_count += 1

    print()
    print(f"  Task A summary: {success_count} succeeded, {failure_count} failed/skipped")
    print(f"  Full log: {INFERENCE_LOG}")
    return results

# ---------------------------------------------------------------------------
# TASK B: Re-scan and verify completeness; update config_mapping.csv
# ---------------------------------------------------------------------------

def task_b_verify_and_update(rows):
    """
    Re-check every experiment's structured-v1 files against disk.
    Update the boolean flag columns in rows[]. Write updated CSV.
    """
    print_section("TASK B — Re-scanning & Verifying File Completeness")

    BOOL_FLAG_COLS = {
        "has_events":            "events_v1.jsonl",
        "has_agent_turns":       "agent_turns_v1.jsonl",
        "has_api_calls":         "api_calls_v1.jsonl",
        "has_outcomes":          "outcomes_v1.jsonl",
        "has_deception_events":  "deception_events_v1.jsonl",
        "has_listener_outcomes": "listener_outcomes_v1.jsonl",
        "has_red_flags":         "red_flags_v1.jsonl",
    }

    still_missing_inference = []
    still_missing_core = []
    fully_complete = 0

    for row in rows:
        if row["game_complete"] != "True":
            continue

        sv_dir = get_sv_dir(row["source_label"], row["experiment_dir"])
        if sv_dir is None or not os.path.isdir(sv_dir):
            still_missing_core.append(row["experiment_dir"])
            continue

        flags = check_file_flags(sv_dir)

        # Update each boolean flag column
        for col, fname in BOOL_FLAG_COLS.items():
            row[col] = str(flags.get(fname, False))

        # Classify
        missing_core = [f for f in CORE_FILES if not flags.get(f, False)]
        missing_inf  = [f for f in INFERENCE_FILES if not flags.get(f, False)]

        if missing_core:
            still_missing_core.append(f"{row['experiment_dir']} core={missing_core}")
        if missing_inf:
            still_missing_inference.append(
                f"{row['experiment_dir']} (config={row['config_id']}): {missing_inf}"
            )
        if not missing_core and not missing_inf:
            fully_complete += 1

    # Re-write CSV
    fieldnames = list(rows[0].keys())
    with open(CONFIG_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Report
    complete_rows = [r for r in rows if r["game_complete"] == "True"]
    print(f"  Total complete games: {len(complete_rows)}")
    print(f"  Fully complete (all files present): {fully_complete}")
    print()

    if still_missing_inference:
        print(f"  Still missing inference files ({len(still_missing_inference)} games):")
        for entry in still_missing_inference:
            print(f"    - {entry}")
    else:
        print("  All complete games now have inference-derived files. [OK]")

    if still_missing_core:
        print()
        print(f"  WARNING — missing CORE files ({len(still_missing_core)} games):")
        for entry in still_missing_core:
            print(f"    - {entry}")

    print()
    print(f"  Updated CSV written to: {CONFIG_CSV}")
    return still_missing_inference, still_missing_core

# ---------------------------------------------------------------------------
# TASK C: LLM Judge Evaluation check (report only — do NOT run without consent)
# ---------------------------------------------------------------------------

def task_c_eval_status(rows):
    """
    Check evaluations/results/ for existing eval scores.
    Count experiments needing evaluation.
    Print a detailed report and ask the user whether to proceed.
    """
    print_section("TASK C — LLM Judge Evaluation Status")

    # Check existing results
    existing_results = []
    if os.path.exists(EVALS_RESULTS):
        for fname in sorted(os.listdir(EVALS_RESULTS)):
            fpath = os.path.join(EVALS_RESULTS, fname)
            if os.path.isfile(fpath):
                size = os.path.getsize(fpath)
                # Count non-empty scored rows
                n_rows = 0
                if fname.endswith(".json"):
                    try:
                        with open(fpath, encoding="utf-8") as f:
                            for line in f:
                                if line.strip():
                                    n_rows += 1
                    except OSError:
                        pass
                existing_results.append((fname, size, n_rows))
        if existing_results:
            print(f"  Existing results in evaluations/results/:")
            for fname, size, n in existing_results:
                print(f"    {fname}  ({size} bytes, {n} scored rows)")
        else:
            print("  evaluations/results/ exists but is empty.")
    else:
        print("  evaluations/results/ does NOT exist — no eval has been run yet.")

    print()

    # Count complete in-config experiments needing eval
    in_config_complete = [
        r for r in rows
        if r["game_complete"] == "True" and r["config_id"] != "UNMAPPED"
    ]

    # For each, count agent turns and LLM turns (excluding human)
    total_llm_turns = 0
    total_all_turns = 0
    exp_stats = []

    already_evaled_exps = {fname.replace("_all_skill_scores.json", "")
                           for fname, _, _ in existing_results}

    needs_eval = []
    for row in in_config_complete:
        sv_dir = get_sv_dir(row["source_label"], row["experiment_dir"])
        if sv_dir is None:
            continue
        total_t, llm_t = count_turns_in_experiment(sv_dir)
        exp_stats.append({
            "exp_dir":     row["experiment_dir"],
            "config_id":   row["config_id"],
            "total_turns": total_t,
            "llm_turns":   llm_t,
        })
        total_all_turns += total_t
        total_llm_turns += llm_t
        if row["experiment_dir"] not in already_evaled_exps:
            needs_eval.append((row["experiment_dir"], row["config_id"], total_t, llm_t))

    print(f"  In-config complete experiments: {len(in_config_complete)}")
    print(f"  Already evaluated:              {len(already_evaled_exps)}")
    print(f"  Needing evaluation:             {len(needs_eval)}")
    print()
    print(f"  Total agent turns across all in-config games: {total_all_turns:,}")
    print(f"  Estimated LLM turns (excl. human ~15%):       {total_llm_turns:,}")
    print(f"    (eval.py sends 1 API call per LLM turn to the judge model)")
    print()

    # Cost estimate: default judge = meta-llama/llama-3.3-70b-instruct (via OpenRouter)
    # Typical prompt ~2000 tokens + completion ~50 tokens per eval call
    # llama-3.3-70b on OpenRouter ≈ $0.12/M input + $0.30/M output (rough estimate)
    # Or the user can pick gemini-flash which is ~$0.075/M
    est_input_tokens  = total_llm_turns * 2000
    est_output_tokens = total_llm_turns * 60
    est_cost_llama_usd = (est_input_tokens * 0.12 + est_output_tokens * 0.30) / 1_000_000
    est_cost_flash_usd = (est_input_tokens * 0.075 + est_output_tokens * 0.30) / 1_000_000

    print(f"  Estimated API calls for eval: {total_llm_turns:,}")
    print(f"  Estimated tokens (input):     {est_input_tokens:,}")
    print(f"  Estimated cost (llama-3.3-70b judge):  ~${est_cost_llama_usd:.2f}")
    print(f"  Estimated cost (gemini-2.0-flash judge): ~${est_cost_flash_usd:.2f}")
    print()

    # Eval command format
    print("  Eval command format (one experiment):")
    print("    cd evaluations/")
    print("    python eval.py --expt_name <experiment_dir> \\")
    print("                   --evaluator <model>")
    print()
    print("  Example for first 3 needing eval:")
    for exp_dir, cid, tt, lt in needs_eval[:3]:
        print(f"    python eval.py --expt_name {exp_dir} --evaluator meta-llama/llama-3.3-70b-instruct")
    print()

    if needs_eval:
        print("  *** AWAITING YOUR DECISION ***")
        print("  eval.py has NOT been run. Please confirm before proceeding.")
        print("  Options:")
        print("    1. Run eval on ALL in-config complete experiments")
        print("       (evaluates all 8 configs, ~{} API calls)".format(total_llm_turns))
        print("    2. Run eval on a subset (specify which configs)")
        print("    3. Skip eval for now")
        print()
        print("  When ready, reply with your choice and I will execute eval.py.")
    else:
        print("  All experiments already have evaluation results.")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    if not os.path.exists(CONFIG_CSV):
        print(f"ERROR: {CONFIG_CSV} not found. Run step1_validate_configs.py first.")
        sys.exit(1)

    # Load CSV
    with open(CONFIG_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Loaded {len(rows)} rows from {CONFIG_CSV}")

    # -----------------------------------------------------------------------
    # TASK A
    # -----------------------------------------------------------------------
    task_a_results = task_a_run_inference(rows)

    # -----------------------------------------------------------------------
    # TASK B
    # -----------------------------------------------------------------------
    still_missing_inf, still_missing_core = task_b_verify_and_update(rows)

    # -----------------------------------------------------------------------
    # TASK A summary (post-run)
    # -----------------------------------------------------------------------
    print_section("TASK A — Final Inference Run Summary")
    by_status = defaultdict(list)
    for exp_dir, status, msg in task_a_results:
        by_status[status].append((exp_dir, msg))

    print(f"  OK:      {len(by_status['ok'])}")
    print(f"  Partial: {len(by_status['partial'])}")
    print(f"  Fail:    {len(by_status['fail'])}")
    print(f"  Skip:    {len(by_status['skip'])}")

    if by_status["fail"]:
        print()
        print("  FAILURES:")
        for exp_dir, msg in by_status["fail"]:
            print(f"    {exp_dir}: {msg}")

    if by_status["partial"]:
        print()
        print("  PARTIAL (some files still missing):")
        for exp_dir, msg in by_status["partial"]:
            print(f"    {exp_dir}: {msg}")

    if by_status["skip"]:
        print()
        print("  SKIPPED:")
        for exp_dir, msg in by_status["skip"]:
            print(f"    {exp_dir}: {msg}")

    # -----------------------------------------------------------------------
    # TASK C
    # -----------------------------------------------------------------------
    task_c_eval_status(rows)

    # -----------------------------------------------------------------------
    # Final completeness summary
    # -----------------------------------------------------------------------
    print_section("FINAL FILE COMPLETENESS SUMMARY")
    complete_rows = [r for r in rows if r["game_complete"] == "True"]
    in_config     = [r for r in complete_rows if r["config_id"] != "UNMAPPED"]

    def count_with(field):
        return sum(1 for r in in_config if r.get(field) == "True")

    print(f"  In-config complete games: {len(in_config)}")
    print()
    print(f"  {'File':<35} {'Present':>10} {'Missing':>10}")
    print("  " + "-" * 57)
    for fname, col in [
        ("events_v1.jsonl",                    "has_events"),
        ("agent_turns_v1.jsonl",               "has_agent_turns"),
        ("api_calls_v1.jsonl",                 "has_api_calls"),
        ("outcomes_v1.jsonl",                  "has_outcomes"),
        ("deception_events_v1.jsonl",          "has_deception_events"),
        ("listener_outcomes_v1.jsonl",         "has_listener_outcomes"),
        ("red_flags_v1.jsonl",                 "has_red_flags"),
    ]:
        n = count_with(col)
        print(f"  {fname:<35} {n:>10} {len(in_config)-n:>10}")

    if not still_missing_inf and not still_missing_core:
        print()
        print("  All in-config complete games are fully populated. [OK]")


if __name__ == "__main__":
    main()
