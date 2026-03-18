#!/usr/bin/env python3
"""
run_v2_inference.py  —  Batch runner for v2 deception inference on all 70 games.

For each complete in-config game from config_mapping.csv:
  1. Runs inference.py --v2
  2. Tracks v1 vs v2 claim coverage
  3. Prints per-config comparison table
  4. Prints verification checks (Step 1F)

Usage:
    python analysis/run_v2_inference.py
"""

import csv
import json
import os
import sys
import pathlib
from collections import defaultdict
from typing import Dict, Any, List

BASE_DIR     = pathlib.Path(__file__).parent.parent
CONFIG_MAP   = BASE_DIR / "analysis" / "config_mapping.csv"
EVAL_DIR     = BASE_DIR / "evaluations"

SOURCE_ROOTS = {
    "shiven_expt_logs": BASE_DIR / "expt-logs",
    "aadi_expt_logs":   BASE_DIR / "aadi-expt-logs" / "expt-logs",
    "llama_crewmate":   BASE_DIR / "amongus_llama_human_crewmate",
    "llama_impostor":   BASE_DIR / "analysis" / "amongus_llama_human_impostor",
}

# Add evaluations/ to path so we can import structured_v1_inference directly
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

# Add analysis/ to path for claim_extractor
analysis_dir = str(BASE_DIR / "analysis")
if analysis_dir not in sys.path:
    sys.path.insert(0, analysis_dir)

from inference import build_tables, read_jsonl  # noqa: E402
from claim_extractor import ClaimExtractor                     # noqa: E402


def load_games() -> List[Dict[str, Any]]:
    games = []
    with open(CONFIG_MAP, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("game_complete", "").strip().lower() == "true"
                    and row.get("config_id", "").startswith("C")):
                games.append(row)
    return games


def get_sv_dir(source_label: str, exp_dir: str) -> pathlib.Path | None:
    root = SOURCE_ROOTS.get(source_label)
    return (root / exp_dir / "structured-v1") if root else None


def count_speak_events(events_path: pathlib.Path) -> int:
    n = 0
    for ev in read_jsonl(str(events_path)):
        if ev.get("event_type") == "SPEAK":
            n += 1
    return n


def count_events(events_path: pathlib.Path) -> tuple[int, int]:
    """Returns (total_speak, speak_with_raw_text)."""
    total = raw = 0
    for ev in read_jsonl(str(events_path)):
        if ev.get("event_type") == "SPEAK":
            total += 1
            if ev.get("raw_text") or ev.get("action_repr"):
                raw += 1
    return total, raw


def get_v1_coverage(sv_dir: pathlib.Path) -> tuple[int, int]:
    """Returns (speak_events_with_claims_v1, total_speak_events)."""
    dec = read_jsonl(str(sv_dir / "deception_events_v1.jsonl"))
    # count unique event_ids that are in v1
    ev_ids_in_v1 = set()
    for r in dec:
        eid = r.get("event_id", "")
        if eid:
            ev_ids_in_v1.add(eid)

    events = read_jsonl(str(sv_dir / "events_v1.jsonl"))
    total_speak = 0
    speak_covered_v1 = 0
    for ev in events:
        if ev.get("event_type") == "SPEAK":
            total_speak += 1
            if ev.get("event_id") in ev_ids_in_v1:
                speak_covered_v1 += 1
    return speak_covered_v1, total_speak


def get_v2_coverage(sv_dir: pathlib.Path) -> tuple[int, int, Dict[str, int], int, int, int, int]:
    """
    Returns:
      (speak_covered_v2, total_speak,
       claim_type_counts,
       imp_lie_v2, crew_lie_v2, imp_speculation, crew_accusation_lie)
    """
    dec = read_jsonl(str(sv_dir / "deception_events_v2.jsonl"))
    events = read_jsonl(str(sv_dir / "events_v1.jsonl"))

    ev_ids_with_claim = set()
    claim_types: Dict[str, int] = defaultdict(int)
    imp_lie = crew_lie = imp_speculation = crew_acc_lie = 0

    for r in dec:
        eid = r.get("event_id", "")
        if eid:
            ev_ids_with_claim.add(eid)
        claim = r.get("claim") or {}
        ct = claim.get("claim_type", "unknown")
        claim_types[ct] += 1

        is_lie = r.get("deception_lie", False)
        ident  = r.get("actor_identity", "")
        if ident == "Impostor" and is_lie:
            imp_lie += 1
        if ident == "Crewmate" and is_lie:
            crew_lie += 1
        # Track crewmate accusations (should never be lie)
        if ident == "Crewmate" and ct == "accusation" and is_lie:
            crew_acc_lie += 1
        if ident == "Impostor" and (claim.get("truth_status") == "speculation"):
            imp_speculation += 1

    total_speak = speak_covered = 0
    for ev in events:
        if ev.get("event_type") == "SPEAK":
            total_speak += 1
            if ev.get("event_id") in ev_ids_with_claim:
                speak_covered += 1

    return speak_covered, total_speak, dict(claim_types), imp_lie, crew_lie, imp_speculation, crew_acc_lie


def main():
    print("=" * 72)
    print("V2 INFERENCE BATCH RUNNER")
    print("=" * 72)

    games = load_games()
    print(f"\nLoaded {len(games)} complete in-config games\n")

    extractor = ClaimExtractor()
    print("ClaimExtractor loaded (spaCy en_core_web_sm)\n")

    # Per-config aggregation
    by_config: Dict[str, Dict] = defaultdict(lambda: {
        "n": 0,
        "speak_total": 0,
        "covered_v1": 0,
        "covered_v2": 0,
        "claim_types": defaultdict(int),
        "imp_lie": 0,
        "crew_lie": 0,
        "crew_acc_lie": 0,
        "opp_v1": 0,
        "opp_v2": 0,
        "errors": [],
    })

    total_games = len(games)
    for idx, game in enumerate(games, 1):
        run_id       = game["run_id"]
        config_id    = game["config_id"]
        source_label = game["source_label"]
        exp_dir      = game["experiment_dir"]

        sv_dir = get_sv_dir(source_label, exp_dir)
        if sv_dir is None or not sv_dir.exists():
            by_config[config_id]["errors"].append(f"{run_id}: sv_dir not found")
            continue

        try:
            build_tables(str(sv_dir), claim_extractor=extractor)
        except Exception as e:
            by_config[config_id]["errors"].append(f"{run_id}: {e}")
            print(f"  [{idx:3}/{total_games}] {config_id} {run_id}  ERROR: {e}", flush=True)
            continue

        # Gather stats
        cov_v1, total_sp = get_v1_coverage(sv_dir)
        cov_v2, _, ctypes, imp_lie, crew_lie, _, crew_acc_lie = get_v2_coverage(sv_dir)

        # Opportunities
        opp_v1 = len(read_jsonl(str(sv_dir / "deception_opportunities_v1.jsonl")))
        opp_v2 = len(read_jsonl(str(sv_dir / "deception_opportunities_v2.jsonl")))

        cfg = by_config[config_id]
        cfg["n"]           += 1
        cfg["speak_total"] += total_sp
        cfg["covered_v1"]  += cov_v1
        cfg["covered_v2"]  += cov_v2
        cfg["imp_lie"]     += imp_lie
        cfg["crew_lie"]    += crew_lie
        cfg["crew_acc_lie"]+= crew_acc_lie
        cfg["opp_v1"]      += opp_v1
        cfg["opp_v2"]      += opp_v2
        for k, v in ctypes.items():
            cfg["claim_types"][k] += v

        v1_pct = f"{100*cov_v1/max(total_sp,1):.0f}%"
        v2_pct = f"{100*cov_v2/max(total_sp,1):.0f}%"
        print(f"  [{idx:3}/{total_games}] {config_id} {run_id:25s}  "
              f"speak={total_sp:3d}  v1={v1_pct:4s}  v2={v2_pct:4s}  "
              f"imp_lie={imp_lie:3d}  crew_lie={crew_lie}", flush=True)

    # ── Summary table ────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("COVERAGE COMPARISON: V1 vs V2 (per config, all games combined)")
    print("=" * 72)
    hdr = (f"  {'Cfg':<6}  {'N':>3}  {'Speak':>6}  {'V1%':>5}  {'V2%':>5}  "
           f"{'Delta':>6}  {'OppV1':>6}  {'OppV2':>6}  {'ImpLie':>7}  {'CrewLie':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    totals = {"n": 0, "speak": 0, "cv1": 0, "cv2": 0,
              "opp1": 0, "opp2": 0, "imp_lie": 0, "crew_lie": 0}

    for cid in sorted(by_config):
        cfg = by_config[cid]
        n   = cfg["n"]
        sp  = cfg["speak_total"]
        cv1 = cfg["covered_v1"]
        cv2 = cfg["covered_v2"]
        opp1 = cfg["opp_v1"]
        opp2 = cfg["opp_v2"]
        il   = cfg["imp_lie"]
        cl   = cfg["crew_lie"]
        p1   = 100 * cv1 / max(sp, 1)
        p2   = 100 * cv2 / max(sp, 1)
        delta = p2 - p1
        print(f"  {cid:<6}  {n:>3}  {sp:>6}  {p1:>4.0f}%  {p2:>4.0f}%  "
              f"{delta:>+6.1f}  {opp1:>6}  {opp2:>6}  {il:>7}  {cl:>8}")
        totals["n"]       += n
        totals["speak"]   += sp
        totals["cv1"]     += cv1
        totals["cv2"]     += cv2
        totals["opp1"]    += opp1
        totals["opp2"]    += opp2
        totals["imp_lie"] += il
        totals["crew_lie"]+= cl

    sp  = totals["speak"]
    cv1 = totals["cv1"]
    cv2 = totals["cv2"]
    p1  = 100 * cv1 / max(sp, 1)
    p2  = 100 * cv2 / max(sp, 1)
    print("  " + "-" * (len(hdr) - 2))
    print(f"  {'TOTAL':<6}  {totals['n']:>3}  {sp:>6}  {p1:>4.0f}%  {p2:>4.0f}%  "
          f"{p2-p1:>+6.1f}  {totals['opp1']:>6}  {totals['opp2']:>6}  "
          f"{totals['imp_lie']:>7}  {totals['crew_lie']:>8}")

    # ── Claim type breakdown ─────────────────────────────────────────────
    print()
    print("CLAIM TYPE BREAKDOWN (v2, all configs)")
    all_ctypes: Dict[str, int] = defaultdict(int)
    for cfg in by_config.values():
        for k, v in cfg["claim_types"].items():
            all_ctypes[k] += v
    total_claims = sum(all_ctypes.values())
    for ct, cnt in sorted(all_ctypes.items(), key=lambda x: -x[1]):
        print(f"  {ct:<15}  {cnt:>6}  ({100*cnt/max(total_claims,1):.1f}%)")
    print(f"  {'TOTAL':<15}  {total_claims:>6}")

    # ── Step 1F: Verification ─────────────────────────────────────────────
    print()
    print("=" * 72)
    print("STEP 1F: VERIFICATION CHECKS")
    print("=" * 72)

    all_pass = True

    # Check 1: overall v2 coverage > 80%
    pct_v2 = 100 * totals["cv2"] / max(totals["speak"], 1)
    ok1 = pct_v2 >= 80
    all_pass = all_pass and ok1
    status1 = "PASS" if ok1 else "FAIL"
    print(f"\n[{status1}] Claim coverage >= 80%:  {pct_v2:.1f}%  "
          f"({totals['cv2']}/{totals['speak']} SPEAK events have >= 1 v2 claim)")

    # Check 2: crewmate lie rate near 0
    total_crew_acc_lie = sum(c["crew_acc_lie"] for c in by_config.values())
    ok2 = total_crew_acc_lie == 0
    all_pass = all_pass and ok2
    status2 = "PASS" if ok2 else "FAIL"
    print(f"\n[{status2}] No crewmate accusation lies: "
          f"{total_crew_acc_lie} crewmate accusation rows with deception_lie=True  "
          f"(should be 0)")

    # Check 3: crew_lie should be near 0 (only from location/task/denial mislabels)
    total_crew_lie = totals["crew_lie"]
    ok3 = total_crew_lie == 0
    status3 = "PASS" if ok3 else "WARN"
    if not ok3:
        print(f"\n[{status3}] Crewmate lie count: {total_crew_lie}  "
              f"(non-zero may indicate location/denial false positives — review manually)")
    else:
        print(f"\n[{status3}] Crewmate total lie count: 0")

    # Check 4: v2 coverage > v1 coverage
    ok4 = totals["cv2"] > totals["cv1"]
    all_pass = all_pass and ok4
    status4 = "PASS" if ok4 else "FAIL"
    p1_tot = 100 * totals["cv1"] / max(totals["speak"], 1)
    p2_tot = 100 * totals["cv2"] / max(totals["speak"], 1)
    print(f"\n[{status4}] V2 coverage > V1 coverage: "
          f"v1={p1_tot:.1f}%  v2={p2_tot:.1f}%  delta={p2_tot-p1_tot:+.1f}pp")

    # Check 5: opp_v2 < opp_v1 (SPEAK-only denominator is smaller)
    ok5 = totals["opp2"] < totals["opp1"]
    all_pass = all_pass and ok5
    status5 = "PASS" if ok5 else "FAIL"
    print(f"\n[{status5}] V2 opportunities < V1 (SPEAK-only fix): "
          f"v1={totals['opp1']}  v2={totals['opp2']}  "
          f"delta={totals['opp2']-totals['opp1']}")

    # Check 6: impostor lie count > 0 (they should have lies)
    ok6 = totals["imp_lie"] > 0
    all_pass = all_pass and ok6
    status6 = "PASS" if ok6 else "WARN"
    print(f"\n[{status6}] Impostor lies > 0: {totals['imp_lie']} impostor deception_lie=True rows")

    # Check 7: v2 total claims > 0
    ok7 = total_claims > 0
    all_pass = all_pass and ok7
    status7 = "PASS" if ok7 else "FAIL"
    print(f"\n[{status7}] Total v2 claims > 0: {total_claims}")

    # Error report
    all_errors = []
    for cid, cfg in by_config.items():
        for e in cfg["errors"]:
            all_errors.append(f"  {cid}: {e}")
    if all_errors:
        print(f"\n[WARN] {len(all_errors)} game(s) had errors:")
        for e in all_errors[:10]:
            print(e)

    print()
    print("=" * 72)
    if all_pass:
        print("ALL CHECKS PASSED — v2 inference pipeline is ready.")
    else:
        print("SOME CHECKS FAILED — review results above before proceeding.")
    print("=" * 72)


if __name__ == "__main__":
    main()
