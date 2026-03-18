#!/usr/bin/env python3
"""
step3_outcomes_kills.py

Compute Group 1 (Game Outcomes) and Group 2 (Kill Metrics) for all 8 configs.
Only uses COMPLETE games from analysis/config_mapping.csv.

Group 1 — Game Outcomes (per config):
  games_played, impostor_wins, crewmate_wins, human_win_rate,
  mean_game_duration, median_game_duration, mean_survivors_at_end

Group 2 — Kill Metrics (per config, from events_v1.jsonl KILL events):
  total_kills, kills_per_game, kills_per_impostor, mean_kill_timestep,
  kill_timing_normalized (early/mid/late), witnessed_kill_rate,
  mean_witness_count, impostor_survival_after_witness

Outputs:
  analysis/results/C{01-08}_game_outcomes.csv
  analysis/results/C{01-08}_kill_metrics.csv
  analysis/evidence/C{01-08}_game_outcomes_evidence.csv
  analysis/evidence/C{01-08}_kill_metrics_evidence.csv

Evidence CSV columns:
  config_id, run_id, metric_name, metric_value, event_ids,
  actor, actor_identity, key_fields (JSON), source_file,
  line_number, timestamp, notes
"""

import csv
import json
import os
import re
import sys
import statistics
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent.parent
CONFIG_MAPPING = BASE_DIR / "analysis" / "config_mapping.csv"

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

# winner codes
WINNER_IMPOSTORS = {1, 4}   # 1=outnumbered, 4=time limit
WINNER_CREWMATES = {2, 3}   # 2=ejection, 3=tasks

EVIDENCE_COLS = [
    "config_id", "run_id", "metric_name", "metric_value",
    "event_ids", "actor", "actor_identity", "key_fields",
    "source_file", "line_number", "timestamp", "notes",
]

VOTEOUT_RE    = re.compile(r"^(.+?) was voted out!")
WITNESS_RE    = re.compile(r"Witness:\s*(\[.*?\])")


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def read_jsonl(path):
    """Read a JSONL file; return list of (lineno, dict). Warn on errors."""
    records = []
    p = str(path)
    if not os.path.exists(p):
        print(f"  [WARN] Missing: {p}", file=sys.stderr)
        return records
    with open(p, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                records.append((lineno, json.loads(raw)))
            except json.JSONDecodeError as e:
                print(f"  [WARN] JSON error {p}:{lineno}: {e}", file=sys.stderr)
    return records


def load_config_mapping(csv_path):
    """Return rows for complete in-config games."""
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
    if root is None:
        return None
    return root / exp_dir / "structured-v1"


def parse_witnesses(additional_info):
    """
    Parse "Location: X, Witness: ['A', 'B']" from additional_info.
    Returns list of witness name strings.
    """
    if not additional_info:
        return []
    m = WITNESS_RE.search(additional_info)
    if not m:
        return []
    raw = m.group(1)
    try:
        return json.loads(raw.replace("'", '"'))
    except Exception:
        inner = raw.strip("[]")
        return [s.strip().strip("'\"") for s in inner.split(",") if s.strip()]


def parse_ejected_player(voteout_details):
    """
    Parse voteout details string.
    Returns ejected player name if voted out, else None.
    """
    if not voteout_details:
        return None
    m = VOTEOUT_RE.match(voteout_details.strip())
    return m.group(1).strip() if m else None


def ev_row(config_id, run_id, metric_name, metric_value,
           event_ids, actor, actor_identity, key_fields,
           source_file, line_number, timestamp, notes=""):
    """Build one evidence row dict."""
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
# GROUP 1: GAME OUTCOMES
# ---------------------------------------------------------------------------

def process_game_outcomes(games):
    """
    For each game, read outcomes_v1.jsonl and events_v1.jsonl.
    Returns (per_game_data list, evidence list).
    """
    per_game_data = []
    evidence      = []

    for game in games:
        run_id       = game["run_id"]
        config_id    = game["config_id"]
        source_label = game["source_label"]
        exp_dir      = game["experiment_dir"]
        human_role   = game["human_role"].strip()

        sv_dir = get_sv_dir(source_label, exp_dir)
        if sv_dir is None:
            print(f"  [WARN] Unknown source '{source_label}' for {run_id}", file=sys.stderr)
            continue

        outcomes_path = sv_dir / "outcomes_v1.jsonl"
        events_path   = sv_dir / "events_v1.jsonl"

        # ---- Outcome record ----
        out_records = read_jsonl(outcomes_path)
        if not out_records:
            print(f"  [WARN] No outcome records: {run_id}", file=sys.stderr)
            continue
        out_lineno, outcome = out_records[-1]   # last record = final outcome

        winner_raw    = outcome.get("winner")
        winner_reason = outcome.get("winner_reason", "")
        timestep      = outcome.get("timestep")
        out_ts        = outcome.get("timestamp", "")

        if winner_raw is None or timestep is None:
            print(f"  [WARN] Missing winner/timestep: {run_id}", file=sys.stderr)
            continue

        winner_int       = int(winner_raw)
        is_impostor_win  = winner_int in WINNER_IMPOSTORS
        outcome_label    = "impostor_win" if is_impostor_win else "crewmate_win"
        human_won = (
            (human_role == "impostor" and is_impostor_win) or
            (human_role == "crewmate" and not is_impostor_win)
        )

        # ---- Survivors at end: last event with valid alive_players ----
        ev_records = read_jsonl(events_path)
        alive_at_end   = None
        alive_ev_id    = ""
        alive_ev_ln    = None
        alive_ev_ts    = ""
        for ln, ev in reversed(ev_records):
            pc = ev.get("phase_context") or {}
            ap = pc.get("alive_players")
            if ap is not None:
                alive_at_end = int(ap)
                alive_ev_id  = ev.get("event_id", "")
                alive_ev_ln  = ln
                alive_ev_ts  = ev.get("timestamp", "")
                break

        per_game_data.append({
            "config_id":              config_id,
            "run_id":                 run_id,
            "source_label":           source_label,
            "exp_dir":                exp_dir,
            "human_role":             human_role,
            "winner":                 winner_int,
            "winner_reason":          winner_reason,
            "game_duration_timesteps": int(timestep),
            "is_impostor_win":        is_impostor_win,
            "human_won":              human_won,
            "alive_at_end":           alive_at_end,
        })

        # Evidence: game_outcome
        evidence.append(ev_row(
            config_id=config_id, run_id=run_id,
            metric_name="game_outcome", metric_value=outcome_label,
            event_ids="", actor="", actor_identity="",
            key_fields={
                "winner": winner_int, "winner_reason": winner_reason,
                "timestep": int(timestep), "human_role": human_role,
                "human_won": human_won,
            },
            source_file=outcomes_path, line_number=out_lineno,
            timestamp=out_ts,
        ))

        # Evidence: game_duration
        evidence.append(ev_row(
            config_id=config_id, run_id=run_id,
            metric_name="game_duration", metric_value=int(timestep),
            event_ids="", actor="", actor_identity="",
            key_fields={"timestep": int(timestep)},
            source_file=outcomes_path, line_number=out_lineno,
            timestamp=out_ts,
        ))

        # Evidence: survivors_at_end
        if alive_at_end is not None:
            evidence.append(ev_row(
                config_id=config_id, run_id=run_id,
                metric_name="survivors_at_end", metric_value=alive_at_end,
                event_ids=alive_ev_id, actor="", actor_identity="",
                key_fields={"alive_players": alive_at_end},
                source_file=events_path, line_number=alive_ev_ln,
                timestamp=alive_ev_ts,
            ))

    return per_game_data, evidence


def compute_outcome_summary(per_game_data):
    by_config = defaultdict(list)
    for g in per_game_data:
        by_config[g["config_id"]].append(g)

    summaries = {}
    for config_id, games in by_config.items():
        n             = len(games)
        imp_wins      = sum(1 for g in games if g["is_impostor_win"])
        crew_wins     = n - imp_wins
        human_wins    = sum(1 for g in games if g["human_won"])
        durations     = [g["game_duration_timesteps"] for g in games]
        survivors     = [g["alive_at_end"] for g in games if g["alive_at_end"] is not None]

        summaries[config_id] = {
            "config_id":            config_id,
            "games_played":         n,
            "impostor_wins":        imp_wins,
            "crewmate_wins":        crew_wins,
            "human_win_rate":       round(human_wins / n, 4) if n else None,
            "mean_game_duration":   round(statistics.mean(durations), 2) if durations else None,
            "median_game_duration": statistics.median(durations) if durations else None,
            "mean_survivors_at_end": round(statistics.mean(survivors), 2) if survivors else None,
        }
    return summaries


# ---------------------------------------------------------------------------
# GROUP 2: KILL METRICS
# ---------------------------------------------------------------------------

def process_kill_metrics(games):
    """
    For each game, read events_v1.jsonl and parse KILL + voteout events.

    Key chain for impostor_survival_after_witness:
      KILL (with external witnesses)
        -> next VOTE event after kill_ts  -> meeting_id M
        -> voteout event with meeting_id M
        -> parse ejected player from details
        -> killer_survived = (ejected != killer)

    Returns (per_game_kills, chain_data, evidence).
    """
    per_game_kills = []
    chain_data     = []
    evidence       = []

    for game in games:
        run_id       = game["run_id"]
        config_id    = game["config_id"]
        source_label = game["source_label"]
        exp_dir      = game["experiment_dir"]

        sv_dir = get_sv_dir(source_label, exp_dir)
        if sv_dir is None:
            continue

        events_path  = sv_dir / "events_v1.jsonl"
        outcomes_path = sv_dir / "outcomes_v1.jsonl"

        ev_records = read_jsonl(events_path)
        if not ev_records:
            continue

        # Game duration from outcomes
        out_records   = read_jsonl(outcomes_path)
        game_duration = None
        if out_records:
            _, outcome = out_records[-1]
            td = outcome.get("timestep")
            if td is not None:
                game_duration = int(td)

        # Partition events by type
        kill_events    = []   # (lineno, ev)
        vote_events    = []   # (lineno, ev)
        voteout_events = []   # (lineno, ev)

        for ln, ev in ev_records:
            et = ev.get("event_type", "")
            if et == "KILL":
                kill_events.append((ln, ev))
            elif et == "VOTE":
                vote_events.append((ln, ev))
            elif et == "voteout":
                voteout_events.append((ln, ev))

        # Build lookup: meeting_id -> voteout event
        voteout_by_meeting = {}
        for ln, ev in voteout_events:
            mid = ev.get("meeting_id")
            if mid is not None:
                voteout_by_meeting[mid] = (ln, ev)

        # Build sorted list of (timestep, meeting_id) from VOTE events
        # so we can find the next meeting after a kill
        meeting_first_ts = {}   # meeting_id -> min timestep of first vote
        for _, ev in vote_events:
            mid = ev.get("meeting_id")
            ts  = ev.get("timestep", 0)
            if mid is not None:
                if mid not in meeting_first_ts or ts < meeting_first_ts[mid]:
                    meeting_first_ts[mid] = ts

        # sorted list: [(ts, meeting_id), ...]
        # Sort by timestep so we can find next meeting after a kill.
        # meeting_first_ts = {meeting_id: min_timestep}
        # sorted as [(min_timestep, meeting_id), ...] → iterate by timestep
        sorted_meetings = sorted(
            ((ts, mid) for mid, ts in meeting_first_ts.items()),
            key=lambda x: x[0]
        )

        # ---- Process each KILL event ----
        game_kills = []
        n_witnessed = 0

        for ln, kill_ev in kill_events:
            killer       = kill_ev.get("actor", "")
            victim       = kill_ev.get("target", "")
            kill_ts      = kill_ev.get("timestep", 0)
            kill_ev_id   = kill_ev.get("event_id", "")
            actor_ident  = kill_ev.get("actor_identity", "")
            add_info     = kill_ev.get("additional_info", "") or ""
            kill_ts_str  = kill_ev.get("timestamp", "")

            all_witnesses      = parse_witnesses(add_info)
            external_witnesses = [w for w in all_witnesses
                                  if w != killer and w != victim]
            has_witnesses      = len(external_witnesses) > 0

            # Timing normalisation
            kill_timing_norm = None
            kill_timing_cat  = None
            if game_duration and game_duration > 0:
                kill_timing_norm = round(kill_ts / game_duration, 4)
                if kill_timing_norm < 0.33:
                    kill_timing_cat = "early"
                elif kill_timing_norm <= 0.66:
                    kill_timing_cat = "mid"
                else:
                    kill_timing_cat = "late"

            game_kills.append({
                "kill_ev_id":        kill_ev_id,
                "killer":            killer,
                "victim":            victim,
                "kill_ts":           kill_ts,
                "actor_identity":    actor_ident,
                "external_witnesses": external_witnesses,
                "witness_count":     len(external_witnesses),
                "has_witnesses":     has_witnesses,
                "kill_timing_norm":  kill_timing_norm,
                "kill_timing_cat":   kill_timing_cat,
                "lineno":            ln,
                "kill_ts_str":       kill_ts_str,
            })

            if has_witnesses:
                n_witnessed += 1

            # Evidence: kill_event
            evidence.append(ev_row(
                config_id=config_id, run_id=run_id,
                metric_name="kill_event",
                metric_value="witnessed" if has_witnesses else "unwitnessed",
                event_ids=kill_ev_id, actor=killer, actor_identity=actor_ident,
                key_fields={
                    "victim": victim,
                    "kill_timestep": kill_ts,
                    "external_witnesses": external_witnesses,
                    "witness_count": len(external_witnesses),
                    "kill_timing_norm": kill_timing_norm,
                    "kill_timing_cat": kill_timing_cat,
                    "game_duration": game_duration,
                },
                source_file=events_path, line_number=ln, timestamp=kill_ts_str,
            ))

            # ---- Impostor survival chain (witnessed kills only) ----
            if not has_witnesses:
                continue

            # Find next meeting after kill_ts
            next_mid = None
            for m_ts, mid in sorted_meetings:
                if m_ts > kill_ts:
                    next_mid = mid
                    break

            if next_mid is None:
                # No meeting after this kill — game ended
                chain_data.append({
                    "config_id": config_id, "run_id": run_id,
                    "kill_ev_id": kill_ev_id, "killer": killer,
                    "actor_identity": actor_ident, "victim": victim,
                    "kill_ts": kill_ts, "external_witnesses": external_witnesses,
                    "meeting_id": None, "ejected_player": None,
                    "votes_for_killer": 0, "total_votes": 0,
                    "killer_survived": True, "no_meeting": True,
                    "voteout_ev_id": None, "voteout_lineno": None,
                    "events_path": events_path, "kill_ts_str": kill_ts_str,
                })
                evidence.append(ev_row(
                    config_id=config_id, run_id=run_id,
                    metric_name="impostor_survival_after_witness",
                    metric_value="survived_no_meeting",
                    event_ids=kill_ev_id, actor=killer, actor_identity=actor_ident,
                    key_fields={
                        "kill_ev_id": kill_ev_id, "victim": victim, "kill_ts": kill_ts,
                        "external_witnesses": external_witnesses,
                        "meeting_id": None, "ejected_player": None,
                        "killer_survived": True, "reason": "no_meeting_after_kill",
                    },
                    source_file=events_path, line_number=ln, timestamp=kill_ts_str,
                    notes="game ended before next meeting",
                ))
                continue

            # Get voteout for that meeting
            vo_info = voteout_by_meeting.get(next_mid)
            voteout_ev_id = None
            voteout_ln    = None
            ejected_player = None

            if vo_info is not None:
                voteout_ln, vo_ev = vo_info
                voteout_ev_id  = vo_ev.get("event_id", "")
                details        = vo_ev.get("details", "") or ""
                ejected_player = parse_ejected_player(details)

            # Count votes for killer from voteout details (for evidence logging)
            votes_for_killer = 0
            total_votes      = 0
            if vo_info is not None:
                voteout_ln, vo_ev = vo_info
                details = vo_ev.get("details", "") or ""
                # Parse vote lines from details
                voted_for_re = re.compile(r"'([^']+) voted for ([^']+)'")
                vote_matches = voted_for_re.findall(details)
                total_votes = len(vote_matches)
                votes_for_killer = sum(
                    1 for voter, target in vote_matches
                    if target.strip() == killer.strip()
                )

            killer_survived = (ejected_player != killer)

            chain_data.append({
                "config_id": config_id, "run_id": run_id,
                "kill_ev_id": kill_ev_id, "killer": killer,
                "actor_identity": actor_ident, "victim": victim,
                "kill_ts": kill_ts, "external_witnesses": external_witnesses,
                "meeting_id": next_mid, "ejected_player": ejected_player,
                "votes_for_killer": votes_for_killer, "total_votes": total_votes,
                "killer_survived": killer_survived, "no_meeting": False,
                "voteout_ev_id": voteout_ev_id, "voteout_lineno": voteout_ln,
                "events_path": events_path, "kill_ts_str": kill_ts_str,
            })

            evidence.append(ev_row(
                config_id=config_id, run_id=run_id,
                metric_name="impostor_survival_after_witness",
                metric_value="survived" if killer_survived else "ejected",
                event_ids=json.dumps([kill_ev_id, voteout_ev_id]),
                actor=killer, actor_identity=actor_ident,
                key_fields={
                    "kill_ev_id": kill_ev_id,
                    "victim": victim, "kill_ts": kill_ts,
                    "external_witnesses": external_witnesses,
                    "meeting_id": next_mid,
                    "voteout_ev_id": voteout_ev_id,
                    "ejected_player": ejected_player,
                    "votes_for_killer": votes_for_killer,
                    "total_votes": total_votes,
                    "killer_survived": killer_survived,
                },
                source_file=events_path, line_number=ln, timestamp=kill_ts_str,
                notes="kill_event_id -> witness_names -> meeting_id -> votes_for_killer -> total_votes -> killer_survived",
            ))

        per_game_kills.append({
            "config_id":    config_id,
            "run_id":       run_id,
            "kills":        game_kills,
            "game_duration": game_duration,
        })

        print(f"  {config_id} {run_id}: {len(kill_events)} kills, "
              f"{n_witnessed} witnessed")

    return per_game_kills, chain_data, evidence


def compute_kill_summary(per_game_kills, chain_data):
    by_config      = defaultdict(list)
    chains_by_cfg  = defaultdict(list)

    for g in per_game_kills:
        by_config[g["config_id"]].append(g)
    for c in chain_data:
        chains_by_cfg[c["config_id"]].append(c)

    summaries = {}
    for config_id, games in by_config.items():
        all_kills = [k for g in games for k in g["kills"]]
        n_games   = len(games)

        total_kills        = len(all_kills)
        kills_per_game     = total_kills / n_games if n_games else 0
        kills_per_impostor = kills_per_game / 2      # always 2 impostors

        kill_ts_list  = [k["kill_ts"] for k in all_kills]
        mean_kill_ts  = statistics.mean(kill_ts_list) if kill_ts_list else None

        timing_counts = {"early": 0, "mid": 0, "late": 0, "unknown": 0}
        for k in all_kills:
            cat = k["kill_timing_cat"] or "unknown"
            timing_counts[cat] = timing_counts.get(cat, 0) + 1

        witnessed      = [k for k in all_kills if k["has_witnesses"]]
        wit_kill_rate  = len(witnessed) / total_kills if total_kills else 0
        wit_counts     = [k["witness_count"] for k in all_kills]
        mean_wit_count = statistics.mean(wit_counts) if wit_counts else 0

        chains = chains_by_cfg[config_id]
        chains_with_meeting = [c for c in chains if not c.get("no_meeting")]
        n_survived = sum(1 for c in chains_with_meeting if c["killer_survived"])
        n_ejected  = sum(1 for c in chains_with_meeting if not c["killer_survived"])
        n_no_meeting = sum(1 for c in chains if c.get("no_meeting"))
        surv_rate  = (n_survived / len(chains_with_meeting)
                      if chains_with_meeting else None)

        summaries[config_id] = {
            "config_id":                          config_id,
            "n_games":                            n_games,
            "total_kills":                        total_kills,
            "kills_per_game":                     round(kills_per_game, 3),
            "kills_per_impostor":                 round(kills_per_impostor, 3),
            "mean_kill_timestep":                 round(mean_kill_ts, 2) if mean_kill_ts else None,
            "kill_timing_early":                  timing_counts["early"],
            "kill_timing_mid":                    timing_counts["mid"],
            "kill_timing_late":                   timing_counts["late"],
            "witnessed_kills":                    len(witnessed),
            "unwitnessed_kills":                  total_kills - len(witnessed),
            "witnessed_kill_rate":                round(wit_kill_rate, 4),
            "mean_witness_count":                 round(mean_wit_count, 3),
            "witnessed_kills_with_next_meeting":  len(chains_with_meeting),
            "witnessed_kills_no_meeting":         n_no_meeting,
            "killer_survived_count":              n_survived,
            "killer_ejected_count":               n_ejected,
            "impostor_survival_after_witness_rate":
                round(surv_rate, 4) if surv_rate is not None else None,
        }

        print(f"\n  Config {config_id}: {n_games} games | {total_kills} kills "
              f"({kills_per_game:.2f}/game, {kills_per_impostor:.2f}/impostor)")
        print(f"    Timing: early={timing_counts['early']} "
              f"mid={timing_counts['mid']} late={timing_counts['late']}")
        print(f"    Witnessed: {len(witnessed)}/{total_kills} "
              f"({wit_kill_rate:.1%}) | mean witnesses: {mean_wit_count:.2f}")
        print(f"    Survival chains: {len(chains_with_meeting)} with meeting "
              f"-> {n_survived} survived, {n_ejected} ejected "
              f"({n_no_meeting} kills had no meeting after)")
        if surv_rate is not None:
            print(f"    Impostor survival rate after witnessed kill: {surv_rate:.1%}")

    return summaries


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("STEP 3: COMPUTE METRIC GROUPS 1 & 2 (OUTCOMES + KILLS)")
    print("=" * 70)

    print(f"\nLoading: {CONFIG_MAPPING}")
    all_games = load_config_mapping(CONFIG_MAPPING)
    print(f"  Loaded {len(all_games)} complete in-config games\n")

    config_counts = defaultdict(int)
    for g in all_games:
        config_counts[g["config_id"]] += 1
    for cid in sorted(config_counts):
        print(f"  {cid}: {config_counts[cid]} games")

    # -----------------------------------------------------------------------
    # GROUP 1: GAME OUTCOMES
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("GROUP 1: GAME OUTCOMES")
    print("=" * 70)

    per_game_data, out_evidence = process_game_outcomes(all_games)
    print(f"\n  Processed {len(per_game_data)} games successfully")

    outcome_summaries = compute_outcome_summary(per_game_data)

    # Print summary table
    hdr = (f"  {'Config':<8} {'N':>4} {'Imp_W':>6} {'Crew_W':>7} "
           f"{'HumanWR':>8} {'MeanDur':>8} {'MedDur':>8} {'MeanSurv':>9}")
    print("\n  Per-Config Game Outcome Summary:")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for cid in sorted(outcome_summaries):
        s   = outcome_summaries[cid]
        hwr = f"{s['human_win_rate']:.1%}"       if s['human_win_rate'] is not None       else "N/A"
        md  = f"{s['mean_game_duration']:.1f}"   if s['mean_game_duration'] is not None   else "N/A"
        mdd = f"{s['median_game_duration']:.0f}" if s['median_game_duration'] is not None else "N/A"
        ms  = f"{s['mean_survivors_at_end']:.1f}" if s['mean_survivors_at_end'] is not None else "N/A"
        print(f"  {cid:<8} {s['games_played']:>4} {s['impostor_wins']:>6} "
              f"{s['crewmate_wins']:>7} {hwr:>8} {md:>8} {mdd:>8} {ms:>9}")

    # Write per-config files
    print()
    for cid in sorted(outcome_summaries):
        s = outcome_summaries[cid]
        write_csv(RESULTS_DIR / f"{cid}_game_outcomes.csv", [s], list(s.keys()))
        ev_rows = [r for r in out_evidence if r["config_id"] == cid]
        write_csv(EVIDENCE_DIR / f"{cid}_game_outcomes_evidence.csv",
                  ev_rows, EVIDENCE_COLS)

    # -----------------------------------------------------------------------
    # GROUP 2: KILL METRICS
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("GROUP 2: KILL METRICS")
    print("=" * 70 + "\n")

    per_game_kills, chain_data, kill_evidence = process_kill_metrics(all_games)

    kill_summaries = compute_kill_summary(per_game_kills, chain_data)

    # Print kill summary table
    khdr = (f"\n  {'Config':<8} {'Kills':>6} {'K/Gm':>6} {'K/Imp':>6} "
            f"{'MeanTS':>7} {'WitPct':>7} {'MnWit':>6} {'SurvRate':>9}")
    print(khdr)
    print("  " + "-" * (len(khdr) - 2))
    for cid in sorted(kill_summaries):
        s  = kill_summaries[cid]
        mt = f"{s['mean_kill_timestep']:.1f}" if s['mean_kill_timestep'] else "N/A"
        wp = f"{s['witnessed_kill_rate']:.1%}"
        mw = f"{s['mean_witness_count']:.2f}"
        sr = (f"{s['impostor_survival_after_witness_rate']:.1%}"
              if s['impostor_survival_after_witness_rate'] is not None else "N/A")
        print(f"  {cid:<8} {s['total_kills']:>6} {s['kills_per_game']:>6.2f} "
              f"{s['kills_per_impostor']:>6.2f} {mt:>7} {wp:>7} {mw:>6} {sr:>9}")

    # Write per-config files
    print()
    for cid in sorted(kill_summaries):
        s = kill_summaries[cid]
        write_csv(RESULTS_DIR / f"{cid}_kill_metrics.csv", [s], list(s.keys()))
        ev_rows = [r for r in kill_evidence if r["config_id"] == cid]
        write_csv(EVIDENCE_DIR / f"{cid}_kill_metrics_evidence.csv",
                  ev_rows, EVIDENCE_COLS)

    print("\n" + "=" * 70)
    print(f"DONE")
    print(f"  Results  : {RESULTS_DIR}")
    print(f"  Evidence : {EVIDENCE_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
