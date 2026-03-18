#!/usr/bin/env python3
"""
step5_latency_tasks_judge.py

Compute Groups 5 (Latency & Model Performance), 6 (Task Efficiency),
and 7 (LLM Judge Scores) for all 8 configs.
Only uses COMPLETE games from analysis/config_mapping.csv.

Group 5 — Latency & Model Performance (from api_calls_v1.jsonl + agent_turns_v1.jsonl):
  mean/median/p90/std latency_ms, latency by phase, mean prompt/completion tokens,
  prompt_growth_slope (linear regression: prompt_tokens ~ step),
  api_failure_rate, thinking_depth (mean, by identity)
  Correlations: latency_vs_win, latency_vs_lie_rate, thinking_depth_vs_lie_rate

Group 6 — Task Efficiency (from events_v1.jsonl COMPLETE TASK events):
  total_tasks_completed, tasks_per_game, task_completion_rate,
  tasks_per_crewmate_per_timestep, task_completion_in_wins_vs_losses

Group 7 — LLM Judge Scores (from evaluations/results/):
  mean_awareness, mean_lying, mean_deception, mean_planning (overall + by identity)
  lying_score vs claim_lie_rate correlation per config

Outputs:
  analysis/results/C{01-08}_latency_metrics.csv
  analysis/results/C{01-08}_correlation_metrics.csv
  analysis/results/C{01-08}_task_metrics.csv
  analysis/results/C{01-08}_judge_metrics.csv
  analysis/evidence/C{01-08}_latency_metrics_evidence.csv
  analysis/evidence/C{01-08}_task_metrics_evidence.csv
  analysis/evidence/C{01-08}_judge_metrics_evidence.csv
"""

import csv
import json
import os
import re
import sys
import statistics
import math
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, linregress

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent.parent
CONFIG_MAPPING = BASE_DIR / "analysis" / "config_mapping.csv"
EVAL_RESULTS_DIR = BASE_DIR / "evaluations" / "results"
DECEPTION_EV_DIR = BASE_DIR / "analysis" / "evidence"  # for lie_rate cross-ref

SOURCE_ROOTS = {
    "shiven_expt_logs": BASE_DIR / "expt-logs",
    "aadi_expt_logs":   BASE_DIR / "aadi-expt-logs" / "expt-logs",
    "llama_crewmate":   BASE_DIR / "amongus_llama_human_crewmate",
    "llama_impostor":   BASE_DIR / "amongus_llama_human_impostor",
}

RESULTS_DIR = BASE_DIR / "analysis" / "results"
EVIDENCE_DIR = BASE_DIR / "analysis" / "evidence"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

EVIDENCE_COLS = [
    "config_id", "run_id", "metric_name", "metric_value",
    "event_ids", "actor", "actor_identity", "key_fields",
    "source_file", "line_number", "timestamp", "notes",
]

THINKING_RE = re.compile(
    r'\[Thinking Process\](.*?)(?=\[Action\]|\[SPEAK Strategy\]|\[SPEAK\]|$)',
    re.DOTALL | re.IGNORECASE
)

WINNER_IMPOSTORS = {1, 4}
HUMAN_MARKERS    = ("homosapiens", "brain")

N_CREWMATES_DEFAULT = 5   # 7 players - 2 impostors


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def read_jsonl(path):
    records = []
    p = str(path)
    if not os.path.exists(p):
        return records
    with open(p, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                records.append((lineno, json.loads(raw)))
            except json.JSONDecodeError:
                pass
    return records


def load_config_mapping(csv_path):
    games = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            complete = row.get("game_complete", "").strip().lower() == "true"
            cid = row.get("config_id", "").strip()
            if complete and cid and cid.startswith("C"):
                games.append(row)
    return games


def get_sv_dir(source_label, exp_dir):
    root = SOURCE_ROOTS.get(source_label)
    return (root / exp_dir / "structured-v1") if root else None


def is_human(model_str):
    m = (model_str or "").lower()
    return any(h in m for h in HUMAN_MARKERS)


def extract_thinking_wc(text):
    """Word count of [Thinking Process] section in raw_response_text."""
    if not text:
        return None
    m = THINKING_RE.search(text)
    if not m:
        return None
    thought = m.group(1).strip()
    return len(thought.split()) if thought else None


def safe_stat(lst, fn):
    return round(fn(lst), 4) if lst else None


def percentile(lst, p):
    if not lst:
        return None
    s = sorted(lst)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 4)


def corr_report(x_vals, y_vals, label):
    """Pearson r with p-value. Returns dict."""
    n = len(x_vals)
    if n < 3:
        return {"r": None, "p": None, "n": n,
                "low_power_flag": True, "label": label}
    try:
        r, p = pearsonr(x_vals, y_vals)
        return {"r": round(float(r), 4), "p": round(float(p), 4),
                "n": n, "low_power_flag": n < 10, "label": label}
    except Exception:
        return {"r": None, "p": None, "n": n,
                "low_power_flag": True, "label": label}


def ev_row(config_id, run_id, metric_name, metric_value,
           event_ids, actor, actor_identity, key_fields,
           source_file, line_number, timestamp, notes=""):
    if not isinstance(event_ids, str):
        event_ids = json.dumps(event_ids)
    if not isinstance(key_fields, str):
        key_fields = json.dumps(key_fields, default=str)
    return {
        "config_id":      config_id,
        "run_id":         run_id,
        "metric_name":    metric_name,
        "metric_value":   str(metric_value),
        "event_ids":      event_ids,
        "actor":          actor,
        "actor_identity": actor_identity,
        "key_fields":     key_fields,
        "source_file":    str(source_file),
        "line_number":    line_number,
        "timestamp":      timestamp,
        "notes":          notes,
    }


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"    Written: {path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# GROUP 5: LATENCY & MODEL PERFORMANCE
# ---------------------------------------------------------------------------

def process_latency(games):
    """
    Returns per_game_lat list and evidence list.
    Each per_game_lat entry has raw data needed for summary and correlations.
    """
    per_game = []
    evidence  = []

    for game in games:
        run_id       = game["run_id"]
        config_id    = game["config_id"]
        source_label = game["source_label"]
        exp_dir      = game["experiment_dir"]
        human_role   = game["human_role"].strip()
        winner_raw   = game["winner"]

        sv_dir = get_sv_dir(source_label, exp_dir)
        if sv_dir is None:
            continue

        api_path = sv_dir / "api_calls_v1.jsonl"
        at_path  = sv_dir / "agent_turns_v1.jsonl"
        outcomes_path = sv_dir / "outcomes_v1.jsonl"

        # Game won from human's perspective (consistent with Group 1)
        try:
            w_int = int(winner_raw)
        except (ValueError, TypeError):
            w_int = -1
        is_impostor_win = w_int in WINNER_IMPOSTORS
        human_won = (
            (human_role == "crewmate" and not is_impostor_win) or
            (human_role == "impostor" and is_impostor_win)
        )
        # LLM won = opposite of human won (LLM plays the other side)
        llm_won = not human_won

        # ---- Agent turns: extract thinking depth per turn ----
        thinking_by_turn = {}   # turn_id -> {"depth": int, "identity": str}
        at_records = read_jsonl(at_path)
        for ln, turn in at_records:
            agent = turn.get("agent", {})
            model = agent.get("model", "")
            if is_human(model):
                continue
            ident  = agent.get("identity", "")
            tid    = turn.get("turn_id", "")
            text   = turn.get("raw_response_text", "") or ""
            depth  = extract_thinking_wc(text)
            if tid and depth is not None:
                thinking_by_turn[tid] = {"depth": depth, "identity": ident}

        # ---- API calls ----
        api_records = read_jsonl(api_path)

        latencies       = []
        lat_task        = []
        lat_meeting     = []
        prompt_toks     = []
        comp_toks       = []
        n_failed        = 0
        n_total         = 0

        # For prompt growth regression: per call (step, prompt_tokens)
        reg_steps  = []
        reg_ptoks  = []

        for ln, call in api_records:
            n_total += 1
            success = call.get("success", True)
            if not success:
                n_failed += 1

            agent  = call.get("agent", {})
            model  = agent.get("model", "")
            if is_human(model):
                continue

            ident  = agent.get("identity", "")
            lat    = call.get("latency_ms")
            pt     = call.get("prompt_tokens")
            ct     = call.get("completion_tokens")
            phase  = call.get("phase", "")
            step   = call.get("step")
            tid    = call.get("turn_id", "")
            ts_str = call.get("timestamp", "")

            if not success:
                # Still count failure; skip metric accumulation
                evidence.append(ev_row(
                    config_id=config_id, run_id=run_id,
                    metric_name="api_call_failure",
                    metric_value=call.get("error_type", "unknown"),
                    event_ids=tid, actor=agent.get("name", ""),
                    actor_identity=ident,
                    key_fields={
                        "http_status": call.get("http_status"),
                        "error_type": call.get("error_type"),
                        "error_details": str(call.get("error_details",""))[:100],
                        "phase": phase, "step": step,
                    },
                    source_file=api_path, line_number=ln, timestamp=ts_str,
                    notes="failed_api_call",
                ))
                continue

            if lat is not None:
                latencies.append(lat)
                if "Task" in phase:
                    lat_task.append(lat)
                elif "Meeting" in phase:
                    lat_meeting.append(lat)

            if pt is not None:
                prompt_toks.append(pt)
                if step is not None:
                    reg_steps.append(step)
                    reg_ptoks.append(pt)

            if ct is not None:
                comp_toks.append(ct)

            # Thinking depth from agent_turns (joined by turn_id)
            td_info = thinking_by_turn.get(tid, {})
            td      = td_info.get("depth")

            # Evidence per API call
            evidence.append(ev_row(
                config_id=config_id, run_id=run_id,
                metric_name="api_call_latency",
                metric_value=round(lat, 2) if lat is not None else "N/A",
                event_ids=tid, actor=agent.get("name", ""),
                actor_identity=ident,
                key_fields={
                    "latency_ms": lat,
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "phase": phase, "step": step,
                    "thinking_depth_wc": td,
                },
                source_file=api_path, line_number=ln, timestamp=ts_str,
            ))

        # Prompt growth regression (per game)
        slope, r2, n_reg = None, None, len(reg_steps)
        if n_reg >= 3:
            try:
                lr = linregress(reg_steps, reg_ptoks)
                slope = round(float(lr.slope), 4)
                r2    = round(float(lr.rvalue ** 2), 4)
            except Exception:
                pass

        # Thinking depth aggregates
        all_depths  = [v["depth"] for v in thinking_by_turn.values()]
        imp_depths  = [v["depth"] for v in thinking_by_turn.values()
                       if v["identity"] == "Impostor"]
        crew_depths = [v["depth"] for v in thinking_by_turn.values()
                       if v["identity"] == "Crewmate"]

        per_game.append({
            "config_id":    config_id,
            "run_id":       run_id,
            "source_label": source_label,
            "exp_dir":      exp_dir,
            "human_won":    human_won,
            "llm_won":      llm_won,
            "latencies":    latencies,
            "lat_task":     lat_task,
            "lat_meeting":  lat_meeting,
            "prompt_toks":  prompt_toks,
            "comp_toks":    comp_toks,
            "n_failed":     n_failed,
            "n_total":      n_total,
            "slope":        slope,
            "r2":           r2,
            "n_reg":        n_reg,
            "all_depths":   all_depths,
            "imp_depths":   imp_depths,
            "crew_depths":  crew_depths,
            # per-game aggregates for correlations
            "mean_lat":     statistics.mean(latencies) if latencies else None,
            "mean_depth":   statistics.mean(all_depths) if all_depths else None,
        })

        print(f"  {config_id} {run_id}: "
              f"calls={n_total}(fail={n_failed}) "
              f"lat={round(statistics.mean(latencies),0) if latencies else 'N/A'}ms "
              f"slope={slope} "
              f"think_depth={round(statistics.mean(all_depths),0) if all_depths else 'N/A'}wc")

    return per_game, evidence


def compute_latency_summary(per_game):
    by_config = defaultdict(list)
    for g in per_game:
        by_config[g["config_id"]].append(g)

    summaries = {}
    for config_id, games in by_config.items():
        n = len(games)

        all_lat   = [l for g in games for l in g["latencies"]]
        all_task  = [l for g in games for l in g["lat_task"]]
        all_meet  = [l for g in games for l in g["lat_meeting"]]
        all_pt    = [t for g in games for t in g["prompt_toks"]]
        all_ct    = [t for g in games for t in g["comp_toks"]]

        total_calls  = sum(g["n_total"]  for g in games)
        total_failed = sum(g["n_failed"] for g in games)

        slopes = [g["slope"] for g in games if g["slope"] is not None]
        r2s    = [g["r2"]    for g in games if g["r2"]    is not None]

        all_depth   = [d for g in games for d in g["all_depths"]]
        imp_depth   = [d for g in games for d in g["imp_depths"]]
        crew_depth  = [d for g in games for d in g["crew_depths"]]

        summaries[config_id] = {
            "config_id":                 config_id,
            "n_games":                   n,
            "total_api_calls":           total_calls,
            "api_failure_rate":          round(total_failed / total_calls, 4) if total_calls else None,
            "mean_latency_ms":           safe_stat(all_lat, statistics.mean),
            "median_latency_ms":         safe_stat(all_lat, statistics.median),
            "p90_latency_ms":            percentile(all_lat, 90),
            "std_latency_ms":            safe_stat(all_lat, statistics.stdev) if len(all_lat)>1 else None,
            "latency_task_phase":        safe_stat(all_task, statistics.mean),
            "latency_meeting_phase":     safe_stat(all_meet, statistics.mean),
            "mean_prompt_tokens":        safe_stat(all_pt, statistics.mean),
            "mean_completion_tokens":    safe_stat(all_ct, statistics.mean),
            "prompt_growth_slope_mean":  round(statistics.mean(slopes), 4) if slopes else None,
            "prompt_growth_r2_mean":     round(statistics.mean(r2s), 4) if r2s else None,
            "thinking_depth_mean":       safe_stat(all_depth, statistics.mean),
            "thinking_depth_impostor":   safe_stat(imp_depth, statistics.mean),
            "thinking_depth_crewmate":   safe_stat(crew_depth, statistics.mean),
        }

        print(f"\n  Config {config_id}: {n} games | {total_calls} calls "
              f"(fail_rate={round(total_failed/total_calls,3) if total_calls else 'N/A'})")
        print(f"    Latency: mean={round(statistics.mean(all_lat),0) if all_lat else 'N/A'}ms "
              f"median={round(statistics.median(all_lat),0) if all_lat else 'N/A'}ms "
              f"p90={percentile(all_lat,90)}ms "
              f"std={round(statistics.stdev(all_lat),0) if len(all_lat)>1 else 'N/A'}ms")
        print(f"    Task phase={round(statistics.mean(all_task),0) if all_task else 'N/A'}ms | "
              f"Meeting phase={round(statistics.mean(all_meet),0) if all_meet else 'N/A'}ms")
        print(f"    Tokens: prompt={round(statistics.mean(all_pt),0) if all_pt else 'N/A'} "
              f"completion={round(statistics.mean(all_ct),0) if all_ct else 'N/A'}")
        print(f"    Prompt growth slope={round(statistics.mean(slopes),2) if slopes else 'N/A'} "
              f"r²={round(statistics.mean(r2s),3) if r2s else 'N/A'}")
        print(f"    Thinking depth: all={round(statistics.mean(all_depth),0) if all_depth else 'N/A'}wc "
              f"impostor={round(statistics.mean(imp_depth),0) if imp_depth else 'N/A'}wc "
              f"crewmate={round(statistics.mean(crew_depth),0) if crew_depth else 'N/A'}wc")

    return summaries


def compute_correlations(per_game, lie_rate_by_run):
    """
    Compute per-game correlations within each config.
    lie_rate_by_run: {run_id: float} — impostor claim lie rate per game
    """
    by_config = defaultdict(list)
    for g in per_game:
        by_config[g["config_id"]].append(g)

    corr_summaries = {}
    for config_id, games in by_config.items():
        n = len(games)

        # ---- latency_vs_win ----
        lats = [g["mean_lat"] for g in games if g["mean_lat"] is not None]
        wins = [float(g["llm_won"]) for g in games if g["mean_lat"] is not None]
        c_lw = corr_report(lats, wins, "latency_vs_llm_win")

        # ---- latency_vs_lie_rate ----
        lr_lats, lr_lies = [], []
        for g in games:
            lr = lie_rate_by_run.get(g["run_id"])
            if g["mean_lat"] is not None and lr is not None:
                lr_lats.append(g["mean_lat"])
                lr_lies.append(lr)
        c_ll = corr_report(lr_lats, lr_lies, "latency_vs_lie_rate")

        # ---- thinking_depth_vs_lie_rate ----
        td_depths, td_lies = [], []
        for g in games:
            lr = lie_rate_by_run.get(g["run_id"])
            if g["mean_depth"] is not None and lr is not None:
                td_depths.append(g["mean_depth"])
                td_lies.append(lr)
        c_dl = corr_report(td_depths, td_lies, "thinking_depth_vs_lie_rate")

        corr_summaries[config_id] = {
            "config_id": config_id,
            "n_games": n,
            # latency vs win
            "latency_vs_win_r":          c_lw["r"],
            "latency_vs_win_p":          c_lw["p"],
            "latency_vs_win_n":          c_lw["n"],
            # latency vs lie rate
            "latency_vs_lie_rate_r":     c_ll["r"],
            "latency_vs_lie_rate_p":     c_ll["p"],
            "latency_vs_lie_rate_n":     c_ll["n"],
            # thinking depth vs lie rate
            "thinking_vs_lie_rate_r":    c_dl["r"],
            "thinking_vs_lie_rate_p":    c_dl["p"],
            "thinking_vs_lie_rate_n":    c_dl["n"],
            "low_power_warning":         "n<=10 per config; correlations have low statistical power",
        }

        print(f"\n  Config {config_id} correlations (n={n}): "
              f"[WARNING: low power with n<=10]")
        print(f"    latency_vs_win:        r={c_lw['r']}  p={c_lw['p']}")
        print(f"    latency_vs_lie_rate:   r={c_ll['r']}  p={c_ll['p']} "
              f"(n={c_ll['n']} games with lie data)")
        print(f"    thinking_vs_lie_rate:  r={c_dl['r']}  p={c_dl['p']} "
              f"(n={c_dl['n']} games with both)")

    return corr_summaries


def load_lie_rate_by_run(all_games):
    """
    Compute per-run impostor claim lie rate from deception_events_v1.jsonl.
    Returns {run_id: float or None}.
    """
    result = {}
    for game in all_games:
        run_id       = game["run_id"]
        source_label = game["source_label"]
        exp_dir      = game["experiment_dir"]
        sv_dir = get_sv_dir(source_label, exp_dir)
        if sv_dir is None:
            continue

        at_path    = sv_dir / "agent_turns_v1.jsonl"
        dec_path   = sv_dir / "deception_events_v1.jsonl"

        # Build identity map
        identity_map = {}
        for _, rec in read_jsonl(at_path):
            agent = rec.get("agent", {})
            n, i  = agent.get("name",""), agent.get("identity","")
            if n and i:
                identity_map[n] = i

        dec_records = read_jsonl(dec_path)
        n_imp_total = 0
        n_imp_lie   = 0
        for _, rec in dec_records:
            actor = rec.get("actor","")
            ident = identity_map.get(actor,"Unknown")
            if ident == "Impostor":
                n_imp_total += 1
                if rec.get("deception_lie"):
                    n_imp_lie += 1
        result[run_id] = (n_imp_lie / n_imp_total) if n_imp_total else None
    return result


# ---------------------------------------------------------------------------
# GROUP 6: TASK EFFICIENCY
# ---------------------------------------------------------------------------

def process_tasks(games):
    """
    Returns per_game_task list and evidence list.
    """
    per_game = []
    evidence  = []

    for game in games:
        run_id       = game["run_id"]
        config_id    = game["config_id"]
        source_label = game["source_label"]
        exp_dir      = game["experiment_dir"]
        human_role   = game["human_role"].strip()
        winner_raw   = game["winner"]
        duration_raw = game.get("game_duration_timesteps", "")

        try:
            w_int = int(winner_raw)
        except (ValueError, TypeError):
            w_int = -1
        is_impostor_win = w_int in WINNER_IMPOSTORS
        human_won = (
            (human_role == "crewmate" and not is_impostor_win) or
            (human_role == "impostor" and is_impostor_win)
        )

        try:
            duration = int(duration_raw)
        except (ValueError, TypeError):
            duration = None

        sv_dir = get_sv_dir(source_label, exp_dir)
        if sv_dir is None:
            continue

        events_path = sv_dir / "events_v1.jsonl"
        ev_records  = read_jsonl(events_path)

        # COMPLETE TASK events by crewmates only
        tasks_done    = 0
        tasks_total_per_player = {}   # player_name -> tasks_total

        for ln, ev in ev_records:
            if ev.get("event_type") != "COMPLETE TASK":
                continue
            ident = ev.get("actor_identity", "")
            if ident != "Crewmate":
                continue

            actor = ev.get("actor", "")
            snap  = ev.get("actor_state_snapshot", {}) or {}
            actor_snap = snap.get("actor", {}) or {}
            t_done  = actor_snap.get("tasks_completed")
            t_total = actor_snap.get("tasks_total")
            ts      = ev.get("timestep", "")
            ev_id   = ev.get("event_id", "")
            ev_ts   = ev.get("timestamp", "")

            tasks_done += 1

            if t_total and actor not in tasks_total_per_player:
                tasks_total_per_player[actor] = int(t_total)

            evidence.append(ev_row(
                config_id=config_id, run_id=run_id,
                metric_name="task_completed",
                metric_value=tasks_done,
                event_ids=ev_id, actor=actor,
                actor_identity="Crewmate",
                key_fields={
                    "tasks_done_so_far": tasks_done,
                    "tasks_completed_this_player": t_done,
                    "tasks_total_this_player": t_total,
                    "timestep": ts,
                    "human_won": human_won,
                },
                source_file=events_path, line_number=ln, timestamp=ev_ts,
            ))

        # Total possible tasks: N_CREWMATES × tasks_per_player
        if tasks_total_per_player:
            tasks_per_player = max(tasks_total_per_player.values())
        else:
            tasks_per_player = 3   # default from observed data

        total_possible = N_CREWMATES_DEFAULT * tasks_per_player
        task_completion_rate = (tasks_done / total_possible
                                if total_possible else None)

        # tasks_per_crewmate_per_timestep
        tpcpt = None
        if duration and duration > 0:
            tpcpt = round(tasks_done / (N_CREWMATES_DEFAULT * duration), 6)

        per_game.append({
            "config_id":         config_id,
            "run_id":            run_id,
            "tasks_done":        tasks_done,
            "total_possible":    total_possible,
            "task_completion_rate": task_completion_rate,
            "tpcpt":             tpcpt,
            "duration":          duration,
            "human_won":         human_won,
            "is_impostor_win":   is_impostor_win,
        })

        print(f"  {config_id} {run_id}: tasks={tasks_done}/{total_possible} "
              f"({task_completion_rate*100:.0f}% done) "
              f"dur={duration} human_won={human_won}")

    return per_game, evidence


def compute_task_summary(per_game_task):
    by_config = defaultdict(list)
    for g in per_game_task:
        by_config[g["config_id"]].append(g)

    summaries = {}
    for config_id, games in by_config.items():
        n = len(games)

        tasks_list  = [g["tasks_done"] for g in games]
        tcr_list    = [g["task_completion_rate"] for g in games
                       if g["task_completion_rate"] is not None]
        tpcpt_list  = [g["tpcpt"] for g in games if g["tpcpt"] is not None]

        # tasks in wins vs losses (from human's perspective)
        win_tasks  = [g["tasks_done"] for g in games if g["human_won"]]
        loss_tasks = [g["tasks_done"] for g in games if not g["human_won"]]

        # Also split by game outcome type
        imp_win_tasks  = [g["tasks_done"] for g in games if g["is_impostor_win"]]
        crew_win_tasks = [g["tasks_done"] for g in games if not g["is_impostor_win"]]

        summaries[config_id] = {
            "config_id":                      config_id,
            "n_games":                        n,
            "total_tasks_completed":          sum(tasks_list),
            "tasks_per_game":                 round(statistics.mean(tasks_list), 3) if tasks_list else None,
            "task_completion_rate":           round(statistics.mean(tcr_list), 4) if tcr_list else None,
            "tasks_per_crewmate_per_timestep": round(statistics.mean(tpcpt_list), 6) if tpcpt_list else None,
            "tasks_in_crewmate_wins":         round(statistics.mean(crew_win_tasks), 2) if crew_win_tasks else None,
            "tasks_in_impostor_wins":         round(statistics.mean(imp_win_tasks), 2) if imp_win_tasks else None,
            "n_crewmate_win_games":           len(crew_win_tasks),
            "n_impostor_win_games":           len(imp_win_tasks),
        }

        print(f"\n  Config {config_id}: {n} games | "
              f"total_tasks={sum(tasks_list)} | tasks/game={round(statistics.mean(tasks_list),1) if tasks_list else 'N/A'}")
        print(f"    Completion rate: {round(statistics.mean(tcr_list),1)*100:.0f}%  | "
              f"tasks/crewmate/ts={round(statistics.mean(tpcpt_list),4) if tpcpt_list else 'N/A'}")
        if crew_win_tasks or imp_win_tasks:
            cw = round(statistics.mean(crew_win_tasks),1) if crew_win_tasks else "N/A"
            iw = round(statistics.mean(imp_win_tasks),1)  if imp_win_tasks else "N/A"
            print(f"    Tasks in crewmate wins: {cw} | Tasks in impostor wins: {iw}")

    return summaries


# ---------------------------------------------------------------------------
# GROUP 7: LLM JUDGE SCORES
# ---------------------------------------------------------------------------

def load_eval_results(eval_dir, all_games):
    """
    Load evaluations/results/ JSONL files.
    Returns {run_key: list_of_score_rows}
    where run_key = (source_label, experiment_dir)
    """
    # Build set of valid run keys from config games
    valid_runs = {(g["source_label"], g["experiment_dir"]): g
                  for g in all_games}

    results = {}
    for path in glob.glob(str(eval_dir / "*.json")):
        fname = os.path.basename(path)
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        if not rows:
            continue
        # Extract source_label and experiment_dir from filename convention:
        # <source_label>__<exp_dir>_all_skill_scores.json
        parts = fname.replace("_all_skill_scores.json", "").split("__", 1)
        if len(parts) == 2:
            sl, ed = parts
        else:
            continue
        rk = (sl, ed)
        if rk in valid_runs:
            results[rk] = rows

    return results, valid_runs


def compute_judge_summary(eval_results, valid_runs, all_games, lie_rate_by_run):
    by_config = defaultdict(list)  # config_id -> list of per-game dicts

    for (sl, ed), rows in eval_results.items():
        meta = valid_runs.get((sl, ed))
        if not meta:
            continue
        config_id = meta["config_id"]
        run_id    = meta["run_id"]

        # Per-identity aggregates for this experiment
        scores_by_ident = defaultdict(lambda: defaultdict(list))
        for row in rows:
            if not row.get("parse_ok"):
                continue
            ident = row.get("player_identity", "Unknown")
            for dim in ["awareness", "lying", "deception", "planning"]:
                v = row.get(dim)
                if v is not None:
                    scores_by_ident[ident][dim].append(v)

        game_entry = {
            "config_id":  config_id,
            "run_id":     run_id,
            "source_label": sl,
            "exp_dir":    ed,
            "scores_by_ident": scores_by_ident,
            "lie_rate":   lie_rate_by_run.get(run_id),
        }
        # Per-game mean lying score (all identities)
        all_lying = [r.get("lying") for r in rows
                     if r.get("parse_ok") and r.get("lying") is not None]
        game_entry["mean_lying"] = statistics.mean(all_lying) if all_lying else None
        by_config[config_id].append(game_entry)

    summaries = {}
    evidence  = []

    for config_id in sorted(by_config):
        games  = by_config[config_id]
        n      = len(games)

        # Aggregate across all games for this config
        all_scores = defaultdict(list)
        imp_scores  = defaultdict(list)
        crew_scores = defaultdict(list)

        for g in games:
            for ident, dims in g["scores_by_ident"].items():
                for dim, vals in dims.items():
                    all_scores[dim].extend(vals)
                    if ident == "Impostor":
                        imp_scores[dim].extend(vals)
                    elif ident == "Crewmate":
                        crew_scores[dim].extend(vals)

            # Per-game evidence row
            for ident, dims in g["scores_by_ident"].items():
                evidence.append(ev_row(
                    config_id=config_id, run_id=g["run_id"],
                    metric_name="judge_scores_per_game",
                    metric_value=ident,
                    event_ids="", actor="", actor_identity=ident,
                    key_fields={
                        "identity": ident,
                        "mean_awareness": round(statistics.mean(dims["awareness"]),3) if dims["awareness"] else None,
                        "mean_lying":     round(statistics.mean(dims["lying"]),3)     if dims["lying"]     else None,
                        "mean_deception": round(statistics.mean(dims["deception"]),3) if dims["deception"] else None,
                        "mean_planning":  round(statistics.mean(dims["planning"]),3)  if dims["planning"]  else None,
                        "n_turns":        len(dims["awareness"]),
                        "lie_rate":       g["lie_rate"],
                    },
                    source_file=str(EVAL_RESULTS_DIR / f"{g['source_label']}__{g['exp_dir']}_all_skill_scores.json"),
                    line_number=0, timestamp="",
                ))

        def ms(d, dim):
            return round(statistics.mean(d[dim]), 3) if d[dim] else None

        # Correlation: lying_score vs claim_lie_rate (per game)
        lying_scores = [g["mean_lying"]  for g in games
                        if g["mean_lying"] is not None and g["lie_rate"] is not None]
        lie_rates    = [g["lie_rate"]     for g in games
                        if g["mean_lying"] is not None and g["lie_rate"] is not None]
        c_lying = corr_report(lying_scores, lie_rates, "lying_score_vs_claim_lie_rate")

        summaries[config_id] = {
            "config_id":                   config_id,
            "n_games_evaluated":           n,
            "mean_awareness":              ms(all_scores, "awareness"),
            "mean_lying":                  ms(all_scores, "lying"),
            "mean_deception":              ms(all_scores, "deception"),
            "mean_planning":               ms(all_scores, "planning"),
            "impostor_mean_awareness":     ms(imp_scores,  "awareness"),
            "impostor_mean_lying":         ms(imp_scores,  "lying"),
            "impostor_mean_deception":     ms(imp_scores,  "deception"),
            "impostor_mean_planning":      ms(imp_scores,  "planning"),
            "crewmate_mean_awareness":     ms(crew_scores, "awareness"),
            "crewmate_mean_lying":         ms(crew_scores, "lying"),
            "crewmate_mean_deception":     ms(crew_scores, "deception"),
            "crewmate_mean_planning":      ms(crew_scores, "planning"),
            "lying_vs_lie_rate_r":         c_lying["r"],
            "lying_vs_lie_rate_p":         c_lying["p"],
            "lying_vs_lie_rate_n":         c_lying["n"],
            "low_power_warning":           "n<=10 per config; correlations have low statistical power",
        }

        print(f"\n  Config {config_id}: {n} games evaluated")
        print(f"    All:      awareness={ms(all_scores,'awareness')} lying={ms(all_scores,'lying')} "
              f"deception={ms(all_scores,'deception')} planning={ms(all_scores,'planning')}")
        print(f"    Impostor: awareness={ms(imp_scores,'awareness')} lying={ms(imp_scores,'lying')} "
              f"deception={ms(imp_scores,'deception')} planning={ms(imp_scores,'planning')}")
        print(f"    Crewmate: awareness={ms(crew_scores,'awareness')} lying={ms(crew_scores,'lying')} "
              f"deception={ms(crew_scores,'deception')} planning={ms(crew_scores,'planning')}")
        print(f"    lying_score vs claim_lie_rate: r={c_lying['r']} p={c_lying['p']} n={c_lying['n']}")

    return summaries, evidence


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("STEP 5: LATENCY + TASKS + JUDGE METRICS (GROUPS 5, 6, 7)")
    print("=" * 70)

    print(f"\nLoading: {CONFIG_MAPPING}")
    all_games = load_config_mapping(CONFIG_MAPPING)
    print(f"  Loaded {len(all_games)} complete in-config games\n")

    config_counts = defaultdict(int)
    for g in all_games:
        config_counts[g["config_id"]] += 1
    for cid in sorted(config_counts):
        print(f"  {cid}: {config_counts[cid]} games")

    # Pre-load per-run lie rates for correlation metrics
    print("\nPre-loading per-run impostor lie rates for correlations...")
    lie_rate_by_run = load_lie_rate_by_run(all_games)
    n_with_lr = sum(1 for v in lie_rate_by_run.values() if v is not None)
    print(f"  Lie rates available for {n_with_lr}/{len(lie_rate_by_run)} games")

    # -----------------------------------------------------------------------
    # GROUP 5: LATENCY
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("GROUP 5: LATENCY & MODEL PERFORMANCE")
    print("=" * 70 + "\n")

    per_game_lat, lat_evidence = process_latency(all_games)
    lat_summaries = compute_latency_summary(per_game_lat)

    # Print summary table
    print("\n  Per-Config Latency Summary:")
    hdr = (f"  {'Cfg':<6} {'MeanLat':>8} {'MedLat':>8} {'P90':>8} "
           f"{'TaskLat':>8} {'MtgLat':>8} {'PromTok':>8} {'Slope':>8} {'TkDpth':>7} {'FailPct':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for cid in sorted(lat_summaries):
        s = lat_summaries[cid]
        fmt = lambda v: f"{v:.0f}" if v is not None else "N/A"
        fmtf = lambda v: f"{v:.2f}" if v is not None else "N/A"
        fmtp = lambda v: f"{v:.1%}" if v is not None else "N/A"
        print(f"  {cid:<6} {fmt(s['mean_latency_ms']):>8} {fmt(s['median_latency_ms']):>8} "
              f"{fmt(s['p90_latency_ms']):>8} {fmt(s['latency_task_phase']):>8} "
              f"{fmt(s['latency_meeting_phase']):>8} {fmt(s['mean_prompt_tokens']):>8} "
              f"{fmtf(s['prompt_growth_slope_mean']):>8} {fmt(s['thinking_depth_mean']):>7} "
              f"{fmtp(s['api_failure_rate']):>8}")

    # Correlation analysis
    print("\n" + "-" * 70)
    print("CORRELATIONS (within each config across games)")
    print("-" * 70)
    corr_summaries = compute_correlations(per_game_lat, lie_rate_by_run)

    print()
    for cid in sorted(lat_summaries):
        write_csv(RESULTS_DIR / f"{cid}_latency_metrics.csv",
                  [lat_summaries[cid]], list(lat_summaries[cid].keys()))
    for cid in sorted(corr_summaries):
        write_csv(RESULTS_DIR / f"{cid}_correlation_metrics.csv",
                  [corr_summaries[cid]], list(corr_summaries[cid].keys()))

    # Consolidate evidence by config
    for cid in sorted(lat_summaries):
        ev_rows = [r for r in lat_evidence if r["config_id"] == cid]
        write_csv(EVIDENCE_DIR / f"{cid}_latency_metrics_evidence.csv",
                  ev_rows, EVIDENCE_COLS)

    # -----------------------------------------------------------------------
    # GROUP 6: TASK EFFICIENCY
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("GROUP 6: TASK EFFICIENCY")
    print("=" * 70 + "\n")

    per_game_task, task_evidence = process_tasks(all_games)
    task_summaries = compute_task_summary(per_game_task)

    # Print summary table
    print("\n  Per-Config Task Efficiency Summary:")
    thdr = (f"\n  {'Cfg':<6} {'TotTasks':>9} {'T/Game':>7} {'CmpltPct':>9} "
            f"{'T/Crew/TS':>10} {'T@CrewWin':>10} {'T@ImpWin':>9}")
    print(thdr)
    print("  " + "-" * (len(thdr) - 2))
    for cid in sorted(task_summaries):
        s = task_summaries[cid]
        fmt  = lambda v: f"{v:.2f}" if v is not None else "N/A"
        fmtc = lambda v: f"{v:.1%}" if v is not None else "N/A"
        fmts = lambda v: f"{v:.6f}" if v is not None else "N/A"
        print(f"  {cid:<6} {s['total_tasks_completed']:>9} {fmt(s['tasks_per_game']):>7} "
              f"{fmtc(s['task_completion_rate']):>9} "
              f"{fmts(s['tasks_per_crewmate_per_timestep']):>10} "
              f"{fmt(s['tasks_in_crewmate_wins']):>10} "
              f"{fmt(s['tasks_in_impostor_wins']):>9}")

    print()
    for cid in sorted(task_summaries):
        write_csv(RESULTS_DIR / f"{cid}_task_metrics.csv",
                  [task_summaries[cid]], list(task_summaries[cid].keys()))
        ev_rows = [r for r in task_evidence if r["config_id"] == cid]
        write_csv(EVIDENCE_DIR / f"{cid}_task_metrics_evidence.csv",
                  ev_rows, EVIDENCE_COLS)

    # -----------------------------------------------------------------------
    # GROUP 7: LLM JUDGE SCORES
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("GROUP 7: LLM JUDGE SCORES")
    print("=" * 70)

    eval_results, valid_runs = load_eval_results(EVAL_RESULTS_DIR, all_games)
    print(f"\n  Eval result files found: {len(eval_results)} / {len(all_games)}")

    if not eval_results:
        print("  [NOTE] No eval results found. Writing placeholder CSVs.")
        placeholder = {"config_id": "N/A", "status": "eval_not_run",
                       "note": "Run step3_run_evals.py first"}
        for cid in sorted(config_counts):
            write_csv(RESULTS_DIR / f"{cid}_judge_metrics.csv",
                      [placeholder], list(placeholder.keys()))
            write_csv(EVIDENCE_DIR / f"{cid}_judge_metrics_evidence.csv",
                      [], EVIDENCE_COLS)
    else:
        judge_summaries, judge_evidence = compute_judge_summary(
            eval_results, valid_runs, all_games, lie_rate_by_run)

        # Print summary table
        print("\n  Per-Config Judge Score Summary (1-10):")
        jhdr = (f"\n  {'Cfg':<6} {'Awrns':>7} {'Lying':>7} {'Decep':>7} {'Plannng':>8} "
                f"| {'Imp:Aw':>7} {'Imp:Ly':>7} {'Imp:De':>7} {'Imp:Pl':>8} "
                f"| {'Cw:Aw':>6} {'Cw:Ly':>6} {'Cw:De':>6}")
        print(jhdr)
        print("  " + "-" * (len(jhdr) - 2))
        for cid in sorted(judge_summaries):
            s = judge_summaries[cid]
            fv = lambda v: f"{v:.2f}" if v is not None else "N/A"
            print(f"  {cid:<6} {fv(s['mean_awareness']):>7} {fv(s['mean_lying']):>7} "
                  f"{fv(s['mean_deception']):>7} {fv(s['mean_planning']):>8} "
                  f"| {fv(s['impostor_mean_awareness']):>7} {fv(s['impostor_mean_lying']):>7} "
                  f"{fv(s['impostor_mean_deception']):>7} {fv(s['impostor_mean_planning']):>8} "
                  f"| {fv(s['crewmate_mean_awareness']):>6} {fv(s['crewmate_mean_lying']):>6} "
                  f"{fv(s['crewmate_mean_deception']):>6}")

        print()
        for cid in sorted(judge_summaries):
            write_csv(RESULTS_DIR / f"{cid}_judge_metrics.csv",
                      [judge_summaries[cid]], list(judge_summaries[cid].keys()))
        for cid in sorted(config_counts):
            ev_rows = [r for r in judge_evidence if r["config_id"] == cid]
            write_csv(EVIDENCE_DIR / f"{cid}_judge_metrics_evidence.csv",
                      ev_rows, EVIDENCE_COLS)

    print("\n" + "=" * 70)
    print(f"DONE")
    print(f"  Results  : {RESULTS_DIR}")
    print(f"  Evidence : {EVIDENCE_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
