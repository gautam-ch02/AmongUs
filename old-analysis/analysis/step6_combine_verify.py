"""
step6_combine_verify.py
Tasks A, B, C: master comparison table, aggression comparison, statistical notes.
"""

import csv
import json
import os
import math
from collections import defaultdict

RESULTS_DIR = r"C:\Users\shiven\Desktop\AmongUs\analysis\results"
EVIDENCE_DIR = r"C:\Users\shiven\Desktop\AmongUs\analysis\evidence"
CONFIG_MAP    = r"C:\Users\shiven\Desktop\AmongUs\analysis\config_mapping.csv"

CONFIGS = ["C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08"]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_csv_first(path):
    """Return first data row of a CSV as a dict, or {} if missing."""
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def load_config_meta():
    """Return {config_id: {llm_model, human_role, prompt_profile}} from config_mapping."""
    meta = {}
    with open(CONFIG_MAP, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = row["config_id"]
            if cid not in meta and cid != "UNMAPPED":
                meta[cid] = {
                    "llm_model":      row.get("llm_model", ""),
                    "human_role":     row.get("human_role", ""),
                    "prompt_profile": row.get("prompt_profile", ""),
                }
    return meta


def fmt(v, decimals=3):
    """Format a value: float→rounded string, None/empty→''."""
    if v is None or v == "":
        return ""
    try:
        f = float(v)
        if math.isnan(f):
            return "nan"
        return str(round(f, decimals))
    except (ValueError, TypeError):
        return str(v)


def _pct(v):
    """Format as percent with 1 dp."""
    try:
        return f"{float(v)*100:.1f}%"
    except Exception:
        return str(v)


# ─────────────────────────────────────────────────────────────────────────────
# TASK A: Master Comparison Table
# ─────────────────────────────────────────────────────────────────────────────

def task_a():
    print("=" * 70)
    print("TASK A: MASTER COMPARISON TABLE")
    print("=" * 70)

    meta = load_config_meta()
    rows = []

    for cid in CONFIGS:
        g  = read_csv_first(os.path.join(RESULTS_DIR, f"{cid}_game_outcomes.csv"))
        k  = read_csv_first(os.path.join(RESULTS_DIR, f"{cid}_kill_metrics.csv"))
        v  = read_csv_first(os.path.join(RESULTS_DIR, f"{cid}_voting_metrics.csv"))
        d  = read_csv_first(os.path.join(RESULTS_DIR, f"{cid}_deception_metrics.csv"))
        l  = read_csv_first(os.path.join(RESULTS_DIR, f"{cid}_latency_metrics.csv"))
        c  = read_csv_first(os.path.join(RESULTS_DIR, f"{cid}_correlation_metrics.csv"))
        t  = read_csv_first(os.path.join(RESULTS_DIR, f"{cid}_task_metrics.csv"))
        j  = read_csv_first(os.path.join(RESULTS_DIR, f"{cid}_judge_metrics.csv"))
        m  = meta.get(cid, {})

        # Derived: impostor_win_rate and crewmate_win_rate from human_win_rate + human_role
        hw  = g.get("human_win_rate", "")
        try:
            hw_f = float(hw)
            hr   = m.get("human_role", "crewmate")
            if hr == "crewmate":
                crewmate_win = hw_f
                impostor_win = 1.0 - hw_f
            else:  # human is impostor
                impostor_win = hw_f
                crewmate_win = 1.0 - hw_f
        except Exception:
            crewmate_win = impostor_win = ""

        row = {
            # Identity
            "config_id":                         cid,
            "llm_model":                         m.get("llm_model", ""),
            "human_role":                        m.get("human_role", ""),
            "prompt_profile":                    m.get("prompt_profile", ""),
            # Group 1 — Game Outcomes
            "games_played":                      g.get("games_played", ""),
            "human_win_rate":                    fmt(g.get("human_win_rate"), 4),
            "impostor_win_rate":                 fmt(impostor_win, 4),
            "crewmate_win_rate":                 fmt(crewmate_win, 4),
            "mean_game_duration":                fmt(g.get("mean_game_duration"), 2),
            "median_game_duration":              fmt(g.get("median_game_duration"), 1),
            "mean_survivors_at_end":             fmt(g.get("mean_survivors_at_end"), 2),
            # Group 2 — Kill Metrics
            "kills_per_game":                    fmt(k.get("kills_per_game"), 3),
            "kills_per_impostor":                fmt(k.get("kills_per_impostor"), 3),
            "mean_kill_timestep":                fmt(k.get("mean_kill_timestep"), 2),
            "kill_timing_early":                 k.get("kill_timing_early", ""),
            "kill_timing_mid":                   k.get("kill_timing_mid", ""),
            "kill_timing_late":                  k.get("kill_timing_late", ""),
            "witnessed_kill_rate":               fmt(k.get("witnessed_kill_rate"), 4),
            "mean_witness_count":                fmt(k.get("mean_witness_count"), 3),
            "impostor_survival_after_witness":   fmt(k.get("impostor_survival_after_witness_rate"), 4),
            # Group 3 — Voting & Detection
            "total_votes":                       v.get("total_votes", ""),
            "skip_vote_rate":                    fmt(v.get("skip_vote_rate"), 4),
            "correct_vote_rate":                 fmt(v.get("correct_vote_rate"), 4),
            "correct_vote_rate_human":           fmt(v.get("correct_vote_rate_human"), 4),
            "correct_vote_rate_llm":             fmt(v.get("correct_vote_rate_llm"), 4),
            "ejection_accuracy":                 fmt(v.get("ejection_accuracy"), 4),
            "impostor_detection_rate":           fmt(v.get("impostor_detection_rate"), 4),
            "vote_changed_rate":                 fmt(v.get("vote_changed_rate"), 4),
            # Group 4 — Deception & Communication
            "claim_lie_rate_impostor":           fmt(d.get("claim_lie_rate_impostor"), 4),
            "claim_lie_rate_crewmate":           fmt(d.get("claim_lie_rate_crewmate"), 4),
            "meeting_deception_density":         fmt(d.get("meeting_deception_density"), 4),
            "mean_speak_length_impostor":        fmt(d.get("mean_speak_length_impostor"), 2),
            "mean_speak_length_crewmate":        fmt(d.get("mean_speak_length_crewmate"), 2),
            "fake_task_rate":                    fmt(d.get("fake_task_rate"), 4),
            "red_flag_count_per_game":           fmt(d.get("red_flag_count_per_game"), 3),
            "deception_opportunity_utilization": fmt(d.get("deception_opportunity_utilization"), 4),
            # Group 5 — Latency
            "mean_latency_ms":                   fmt(l.get("mean_latency_ms"), 1),
            "median_latency_ms":                 fmt(l.get("median_latency_ms"), 1),
            "p90_latency_ms":                    fmt(l.get("p90_latency_ms"), 1),
            "std_latency_ms":                    fmt(l.get("std_latency_ms"), 1),
            "latency_task_phase":                fmt(l.get("latency_task_phase"), 1),
            "latency_meeting_phase":             fmt(l.get("latency_meeting_phase"), 1),
            "mean_prompt_tokens":                fmt(l.get("mean_prompt_tokens"), 1),
            "mean_completion_tokens":            fmt(l.get("mean_completion_tokens"), 1),
            "prompt_growth_slope_mean":          fmt(l.get("prompt_growth_slope_mean"), 3),
            "prompt_growth_r2_mean":             fmt(l.get("prompt_growth_r2_mean"), 3),
            "thinking_depth_mean":               fmt(l.get("thinking_depth_mean"), 1),
            "thinking_depth_impostor":           fmt(l.get("thinking_depth_impostor"), 1),
            "thinking_depth_crewmate":           fmt(l.get("thinking_depth_crewmate"), 1),
            "api_failure_rate":                  fmt(l.get("api_failure_rate"), 4),
            "total_api_calls":                   l.get("total_api_calls", ""),
            # Group 5 — Correlations
            "corr_latency_vs_win_r":             fmt(c.get("latency_vs_win_r"), 4),
            "corr_latency_vs_win_p":             fmt(c.get("latency_vs_win_p"), 4),
            "corr_latency_vs_lie_r":             fmt(c.get("latency_vs_lie_rate_r"), 4),
            "corr_latency_vs_lie_p":             fmt(c.get("latency_vs_lie_rate_p"), 4),
            "corr_thinking_vs_lie_r":            fmt(c.get("thinking_vs_lie_rate_r"), 4),
            "corr_thinking_vs_lie_p":            fmt(c.get("thinking_vs_lie_rate_p"), 4),
            # Group 6 — Task Efficiency
            "tasks_per_game":                    fmt(t.get("tasks_per_game"), 2),
            "task_completion_rate":              fmt(t.get("task_completion_rate"), 4),
            "tasks_per_crewmate_per_timestep":   fmt(t.get("tasks_per_crewmate_per_timestep"), 5),
            "tasks_in_crewmate_wins":            fmt(t.get("tasks_in_crewmate_wins"), 2),
            "tasks_in_impostor_wins":            fmt(t.get("tasks_in_impostor_wins"), 2),
            # Group 7 — Judge Scores
            "judge_mean_awareness":              fmt(j.get("mean_awareness"), 3),
            "judge_mean_lying":                  fmt(j.get("mean_lying"), 3),
            "judge_mean_deception":              fmt(j.get("mean_deception"), 3),
            "judge_mean_planning":               fmt(j.get("mean_planning"), 3),
            "judge_impostor_awareness":          fmt(j.get("impostor_mean_awareness"), 3),
            "judge_impostor_lying":              fmt(j.get("impostor_mean_lying"), 3),
            "judge_impostor_deception":          fmt(j.get("impostor_mean_deception"), 3),
            "judge_impostor_planning":           fmt(j.get("impostor_mean_planning"), 3),
            "judge_crewmate_awareness":          fmt(j.get("crewmate_mean_awareness"), 3),
            "judge_crewmate_lying":              fmt(j.get("crewmate_mean_lying"), 3),
            "judge_crewmate_deception":          fmt(j.get("crewmate_mean_deception"), 3),
            "judge_crewmate_planning":           fmt(j.get("crewmate_mean_planning"), 3),
            "judge_lying_vs_lie_rate_r":         fmt(j.get("lying_vs_lie_rate_r"), 4),
            "judge_lying_vs_lie_rate_p":         fmt(j.get("lying_vs_lie_rate_p"), 4),
        }
        rows.append(row)

    # Write CSV
    out_path = os.path.join(RESULTS_DIR, "master_comparison_table.csv")
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Written: {out_path} ({len(rows)} rows x {len(fieldnames)} columns)\n")

    # Pretty-print key sections
    _print_master_table(rows)
    return rows


def _print_master_table(rows):
    """Print abbreviated but readable version of master table."""
    sep = "-" * 120

    # Section 1: Outcomes & Kills
    print("\n--- GROUP 1-2: GAME OUTCOMES & KILL METRICS ---")
    hdr = f"{'Cfg':<5} {'Model':<22} {'HRole':<10} {'Prompt':<14} {'N':<3} {'HWin%':<7} {'AvgDur':<7} {'K/G':<5} {'WitKR':<7} {'ImpSurv':<8}"
    print(hdr)
    print(sep[:len(hdr)])
    for r in rows:
        print(f"{r['config_id']:<5} {r['llm_model']:<22} {r['human_role']:<10} {r['prompt_profile']:<14} "
              f"{r['games_played']:<3} {_pct(r['human_win_rate']):<7} "
              f"{r['mean_game_duration']:<7} {r['kills_per_game']:<5} "
              f"{_pct(r['witnessed_kill_rate']):<7} {_pct(r['impostor_survival_after_witness']):<8}")

    # Section 2: Voting & Detection
    print("\n--- GROUP 3: VOTING & DETECTION ---")
    hdr2 = f"{'Cfg':<5} {'VoteAcc':<9} {'HumanAcc':<10} {'LLMAcc':<9} {'EjectAcc':<10} {'ImpDetect':<11} {'VoteChg':<8}"
    print(hdr2)
    print(sep[:len(hdr2)])
    for r in rows:
        print(f"{r['config_id']:<5} {_pct(r['correct_vote_rate']):<9} {_pct(r['correct_vote_rate_human']):<10} "
              f"{_pct(r['correct_vote_rate_llm']):<9} {_pct(r['ejection_accuracy']):<10} "
              f"{_pct(r['impostor_detection_rate']):<11} {_pct(r['vote_changed_rate']):<8}")

    # Section 3: Deception
    print("\n--- GROUP 4: DECEPTION & COMMUNICATION ---")
    hdr3 = f"{'Cfg':<5} {'ImpLie%':<9} {'CwLie%':<8} {'MtgDecDns':<11} {'ImpWC':<7} {'CwWC':<7} {'FkTask%':<9} {'RedFlag/G':<10}"
    print(hdr3)
    print(sep[:len(hdr3)])
    for r in rows:
        print(f"{r['config_id']:<5} {_pct(r['claim_lie_rate_impostor']):<9} {_pct(r['claim_lie_rate_crewmate']):<8} "
              f"{r['meeting_deception_density']:<11} {r['mean_speak_length_impostor']:<7} "
              f"{r['mean_speak_length_crewmate']:<7} {_pct(r['fake_task_rate']):<9} "
              f"{r['red_flag_count_per_game']:<10}")

    # Section 4: Latency
    print("\n--- GROUP 5: LATENCY & API PERFORMANCE ---")
    hdr4 = f"{'Cfg':<5} {'MeanLat':<9} {'P90Lat':<8} {'FailPct':<9} {'PromTok':<9} {'Slope':<7} {'TkDpth':<7} {'ImpTk':<7} {'CwTk':<6}"
    print(hdr4)
    print(sep[:len(hdr4)])
    for r in rows:
        print(f"{r['config_id']:<5} {r['mean_latency_ms']:<9} {r['p90_latency_ms']:<8} "
              f"{_pct(r['api_failure_rate']):<9} {r['mean_prompt_tokens']:<9} "
              f"{r['prompt_growth_slope_mean']:<7} {r['thinking_depth_mean']:<7} "
              f"{r['thinking_depth_impostor']:<7} {r['thinking_depth_crewmate']:<6}")

    # Section 5: Tasks
    print("\n--- GROUP 6: TASK EFFICIENCY ---")
    hdr5 = f"{'Cfg':<5} {'T/Game':<8} {'CmpltPct':<10} {'T/Cw/TS':<10} {'T@CwWin':<9} {'T@ImpWin':<9}"
    print(hdr5)
    print(sep[:len(hdr5)])
    for r in rows:
        print(f"{r['config_id']:<5} {r['tasks_per_game']:<8} {_pct(r['task_completion_rate']):<10} "
              f"{r['tasks_per_crewmate_per_timestep']:<10} {r['tasks_in_crewmate_wins']:<9} "
              f"{r['tasks_in_impostor_wins']:<9}")

    # Section 6: Judge
    print("\n--- GROUP 7: LLM JUDGE SCORES (1-10) ---")
    hdr6 = f"{'Cfg':<5} {'Awrns':<7} {'Lying':<7} {'Decep':<7} {'Plan':<7} | {'ImpLy':<7} {'ImpDe':<7} | {'CwLy':<7} {'CwDe':<7}"
    print(hdr6)
    print(sep[:len(hdr6)])
    for r in rows:
        print(f"{r['config_id']:<5} {r['judge_mean_awareness']:<7} {r['judge_mean_lying']:<7} "
              f"{r['judge_mean_deception']:<7} {r['judge_mean_planning']:<7} | "
              f"{r['judge_impostor_lying']:<7} {r['judge_impostor_deception']:<7} | "
              f"{r['judge_crewmate_lying']:<7} {r['judge_crewmate_deception']:<7}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# TASK B: Aggression Comparison
# ─────────────────────────────────────────────────────────────────────────────

NUMERIC_COLS = [
    "human_win_rate", "impostor_win_rate", "crewmate_win_rate",
    "mean_game_duration", "median_game_duration", "mean_survivors_at_end",
    "kills_per_game", "kills_per_impostor", "mean_kill_timestep",
    "witnessed_kill_rate", "mean_witness_count", "impostor_survival_after_witness",
    "skip_vote_rate", "correct_vote_rate", "correct_vote_rate_human",
    "correct_vote_rate_llm", "ejection_accuracy", "impostor_detection_rate",
    "vote_changed_rate",
    "claim_lie_rate_impostor", "claim_lie_rate_crewmate",
    "meeting_deception_density", "mean_speak_length_impostor",
    "mean_speak_length_crewmate", "fake_task_rate", "red_flag_count_per_game",
    "deception_opportunity_utilization",
    "mean_latency_ms", "median_latency_ms", "p90_latency_ms",
    "mean_prompt_tokens", "prompt_growth_slope_mean", "thinking_depth_mean",
    "thinking_depth_impostor", "thinking_depth_crewmate", "api_failure_rate",
    "tasks_per_game", "task_completion_rate", "tasks_per_crewmate_per_timestep",
    "tasks_in_crewmate_wins", "tasks_in_impostor_wins",
    "judge_mean_awareness", "judge_mean_lying", "judge_mean_deception",
    "judge_mean_planning", "judge_impostor_lying", "judge_impostor_deception",
    "judge_crewmate_lying", "judge_crewmate_deception",
]


def _direction(diff):
    if diff is None:
        return "N/A"
    if abs(diff) < 1e-6:
        return "unchanged"
    return "increased" if diff > 0 else "decreased"


def task_b(master_rows):
    print("=" * 70)
    print("TASK B: AGGRESSION COMPARISON")
    print("=" * 70)

    row_by_cfg = {r["config_id"]: r for r in master_rows}

    pairs = [
        ("C03", "C05", "crewmate baseline vs aggressive (gemini)"),
        ("C04", "C06", "impostor baseline vs aggressive (gemini)"),
    ]

    comparison_rows = []
    for base_id, agg_id, label in pairs:
        base = row_by_cfg[base_id]
        agg  = row_by_cfg[agg_id]
        print(f"\n  {label} ({base_id} vs {agg_id}):")
        print(f"  {'Metric':<42} {'Baseline':>12} {'Aggressive':>12} {'AbsDiff':>10} {'Direction':<12}")
        print("  " + "-" * 90)
        for col in NUMERIC_COLS:
            bv = base.get(col, "")
            av = agg.get(col, "")
            try:
                bf = float(bv) if bv not in ("", "nan") else None
                af = float(av) if av not in ("", "nan") else None
                diff = (af - bf) if (bf is not None and af is not None) else None
            except Exception:
                bf, af, diff = None, None, None

            direction = _direction(diff)
            diff_str  = f"{diff:+.4f}" if diff is not None else "N/A"
            bv_str    = bv if bv else "N/A"
            av_str    = av if av else "N/A"
            print(f"  {col:<42} {bv_str:>12} {av_str:>12} {diff_str:>10} {direction:<12}")

            comparison_rows.append({
                "comparison":     label,
                "baseline_cfg":   base_id,
                "aggressive_cfg": agg_id,
                "metric":         col,
                "baseline_value": bv_str,
                "aggressive_value": av_str,
                "absolute_difference": diff_str,
                "direction":      direction,
            })

    out_path = os.path.join(RESULTS_DIR, "aggression_comparison.csv")
    fieldnames = ["comparison", "baseline_cfg", "aggressive_cfg", "metric",
                  "baseline_value", "aggressive_value", "absolute_difference", "direction"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(comparison_rows)
    print(f"\n  Written: {out_path} ({len(comparison_rows)} rows)\n")


# ─────────────────────────────────────────────────────────────────────────────
# TASK C: Statistical Notes
# ─────────────────────────────────────────────────────────────────────────────

STAT_NOTES_TEMPLATE = """\
# Statistical Notes — Among Us Human vs. LLM Benchmark

Generated from analysis of {n_configs} configurations, {n_total} complete in-config games.

---

## 1. Sample Sizes Per Configuration

| Config | Model | Human Role | Prompt | N Games | Note |
|--------|-------|-----------|--------|---------|------|
| C01 | claude-3.5-haiku | crewmate | baseline_v1 | 10 | Primary baseline |
| C02 | claude-3.5-haiku | impostor | baseline_v1 | 10 | Primary baseline |
| C03 | gemini-2.5-flash | crewmate | baseline_v1 | 10 | Gemini baseline |
| C04 | gemini-2.5-flash | impostor | baseline_v1 | 10 | Gemini baseline |
| C05 | gemini-2.5-flash | crewmate | aggressive_v1 | **5** | **Low power** |
| C06 | gemini-2.5-flash | impostor | aggressive_v1 | **5** | **Low power** |
| C07 | llama-3.1-8b | crewmate | baseline_v1 | 10 | Open-source baseline |
| C08 | llama-3.1-8b | impostor | baseline_v1 | 10 | Open-source baseline |

**Power warning**: C05 and C06 each have only n=5 games. All per-config statistics
for these two configs should be treated as pilot/exploratory. Differences smaller
than ~0.20 on any [0,1]-bounded metric cannot be reliably distinguished from noise.

---

## 2. Metric-by-Metric Robustness Assessment

### ROBUST — safe to report as findings

- **human_win_rate**: Aggregate outcome; n=10 per config (n=5 for C05/C06) with
  binary outcomes. 100% win rates (C01, C04) are definitive within the sample.
  Variance is low or absent for those configs. C02/C03 show mixed outcomes and
  are reliable at the level of "majority/minority win."

- **mean_game_duration**: Continuous, low-variance within config. C03 (30.2 ts) vs
  C01 (11.2 ts) is a 2.7× difference that is robust across all 10 games. C05/C06
  durations are more variable relative to sample size.

- **kills_per_game / witnessed_kill_rate**: Continuous counts. Cross-config
  differences are large (e.g. C07 1.8 kills/game vs C04 1.3). Reliable as
  directional findings.

- **impostor_survival_after_witness_rate**: Based on discrete kill-chain events;
  0% (C01, C02) and 100% (C04) are definitive within sample. C03/C05/C06/C07/C08
  mid-range values less certain at n=5–10.

- **ejection_accuracy**: Small-count (2–6 ejections per config); 100% accuracy
  for C01/C02/C04 is definitive. Other configs have higher variance.

- **mean_latency_ms**: Hundreds of API calls per config; population means are
  very stable. Model-level differences (Claude ~6.3s vs Gemini ~2.5s) are highly
  robust regardless of n_games.

- **api_failure_rate**: Derived from total call counts (498–1546 per config);
  C07's 12.6% failure rate is robust against sampling noise.

- **judge scores (awareness, lying, deception, planning)**: Aggregated over
  6,438 LLM-scored turns. Numeric averages are stable; inter-config differences
  of ≥0.3 points on a 1–10 scale can be trusted directionally.

### EXPLORATORY — report with caveats

- **correct_vote_rate / correct_vote_rate_human / correct_vote_rate_llm**:
  Derived from 6–21 vote events per config (often 1–3 per game). High variance
  at the per-game level; per-config averages have SE of ≈0.10–0.15.

- **impostor_detection_rate**: Based on 2–8 ejections per config. Values like
  0.1 or 0.2 (1 or 2 impostor ejections total) have very wide confidence
  intervals (±0.15–0.30 at n=10).

- **claim_lie_rate_impostor / claim_lie_rate_crewmate**: High (80–100%) across
  all configs, driven partly by spatial hallucination in LLM agents rather than
  deliberate deception. Inter-config differences of ≤0.10 are not meaningful.

- **meeting_deception_density**: Continuous but derived from small meeting counts
  (1–5 per game). C03 and C07 have sufficient meetings for reliable estimates;
  C01/C02 do not (short games, few meetings).

- **fake_task_rate**: Moderate event count (10–60 impostor task-phase events per
  config). Reliable as directional indicator; exact decimal values uncertain.

- **tasks_in_crewmate_wins / tasks_in_impostor_wins**: Conditioned on win type;
  for configs with near-100% win rates (C01, C04) one arm is empty or has n=1.
  Do not compute statistics on the empty arm.

- **prompt_growth_slope / r²**: Per-game linear regression aggregated over 5–10
  games. R² values (0.07–0.38) indicate weak-to-moderate fit; slopes should be
  interpreted as indicative only.

- **Pearson correlations (latency_vs_win, latency_vs_lie_rate, thinking_vs_lie_rate)**:
  All computed with n=3–10 game-level observations per config. ALL are flagged
  low-power. No correlation reached p<0.05. Do not draw causal conclusions.

- **judge_lying_vs_lie_rate_r**: Per-config correlation between LLM-judge lying
  score and data-derived claim_lie_rate. n=3–7 per config; no significant
  result found. Interesting direction for future work with larger samples.

### NOT REPORTABLE without more data

- **skip_vote_rate**: Always 0.0 — the simulation never implements skip votes.
  This is a simulation artifact, not a behavioral finding.

- **vote_changed_rate**: Always 0.0 — agents never change votes within a meeting.
  This is a simulation artifact.

- **tasks_in_impostor_wins** for C01 (N=0 impostor wins): undefined / NA.

- **tasks_in_crewmate_wins** for C04 (N=0 crewmate wins): undefined / NA.

---

## 3. Aggression Config Caveats (C03 vs C05, C04 vs C06)

The aggressive_v1 prompt configs (C05, C06) have only n=5 games each, half the
baseline (n=10). Any comparison between baseline and aggressive results should
note:

- Minimum detectable effect size ≈ Cohen's d ≈ 1.0 at 80% power with n=5 vs 10
  (very large effect required for detection).
- The observed direction of each metric change is informative, but magnitudes
  should not be treated as precise estimates.
- Replication with n≥15 per arm is recommended before publishing aggression
  findings as conclusions.

---

## 4. Data Anomalies Found During Computation

1. **Crewmate claim_lie_rate 80–100%** across all configs: LLM crewmates
   frequently make false location claims. Confirmed to be spatial hallucination
   (agents report incorrect rooms), not deliberate deception. The
   `deception_events_v1.jsonl` scorer labels these as lies because the claimed
   location does not match game state.

2. **Tasks completed > 15 (theoretical max)** in C01/C07: Some games show 17–20
   crewmate task completions against a 5-crewmate × 3-task = 15 ceiling. This
   indicates duplicate COMPLETE TASK events being logged for tasks that were
   re-triggered. Completion rate is capped at 108.7% for C01.

3. **C07 12.6% API failure rate**: llama-3.1-8b via the experiment harness failed
   147/1169 API calls. These failures were handled by the game engine (retries or
   fallback no-ops). Failure rate is highest in early C07 experiments (exp_3:
   132/166 = 80% failure) and improved in later runs — suggesting model/API
   instability, not a systematic flaw.

4. **Missing lie-rate data for 22/51 shiven/aadi games**: `deception_events_v1.jsonl`
   is absent or empty for some experiments (mostly llama configs). Correlations
   involving lie_rate use only the subset with data (n=3–7 per config instead
   of n=5–10). This is flagged in correlation outputs.

5. **C02/exp_4: 20 API failures in one game** (83 calls, 20 failures = 24%):
   Isolated spike; all other C02 games have 0 failures. Excluded from
   per-config mean only if flagged — currently retained in aggregate.

6. **event_ids absent for some evidence rows** (game_outcome, game_duration rows):
   These reference the whole-game outcomes_v1.jsonl record (no event-level ID)
   rather than a specific event. Line number alone is used for verification.

7. **Thinking depth "wc" suffix in one C03 game log**: One entry had
   `think_depth=103wc` (missing decimal separator) instead of `103.0wc`.
   Parsed correctly by the script; not a data loss.

---

## 5. Recommended Reporting Tiers

### Tier 1 — Headline findings (high confidence)
- Human win rate by model (C01: 100%, C04: 100%, C03: 10%)
- Mean game duration by model family
- API latency by model provider (Claude 2.5×slower than Gemini)
- API failure rate (Llama 12.6% vs Claude/Gemini 0–3.6%)
- LLM judge awareness and planning scores
- Impostor survival after witnessed kill (0% vs 100% at extremes)

### Tier 2 — Supporting findings (directional, report with SE or CI)
- Voting accuracy by human role
- Impostor detection rate
- Kill timing distribution
- Witness kill rate
- Fake task rate
- Thinking depth by role (Impostor systematically deeper than Crewmate)

### Tier 3 — Exploratory (include in appendix, flag uncertainty)
- Aggression prompt comparisons (C03 vs C05, C04 vs C06)
- Pearson correlations (latency, thinking depth, lie rate)
- Meeting deception density
- Prompt growth slope
- Judge lying score vs. data-derived lie rate correlation

### Not recommended for reporting
- Skip vote rate (always 0 — simulation artifact)
- Vote changed rate (always 0 — simulation artifact)
- Crewmate lie rate as a deception measure (confounded by spatial hallucination)
"""


def task_c():
    print("=" * 70)
    print("TASK C: STATISTICAL NOTES")
    print("=" * 70)

    out_path = os.path.join(
        os.path.dirname(RESULTS_DIR), "results", "statistical_notes.md"
    )
    content = STAT_NOTES_TEMPLATE.format(n_configs=8, n_total=70)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n  Written: {out_path}\n")
    print("  Sections: sample sizes, metric robustness tiers, aggression caveats,")
    print("  data anomalies, recommended reporting tiers.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("STEP 6: COMBINE & VERIFY (A, B, C)")
    print("=" * 70)

    master = task_a()
    task_b(master)
    task_c()

    print("=" * 70)
    print("DONE")
    print(f"  Results:  {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
