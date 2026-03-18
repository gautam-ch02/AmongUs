# Metric Verification Log

## Metadata

| Field          | Value                                    |
| -------------- | ---------------------------------------- |
| Generated      | 2026-03-02T02:19:13.632025               |
| Model          | GPT-5.2 (via GitHub Copilot)             |
| Python         | 3.13.12                                  |
| Platform       | Windows-11-10.0.26200-SP0                |
| Workspace      | C:\Users\shiven\Desktop\AmongUs          |
| Git HEAD       | 425b9bc6c4333aee33235a9e7c943ba8a9f7da1f |
| Duration       | 1.37 seconds                             |
| Checks Passed  | 415                                      |
| Checks Failed  | 0                                        |
| Warnings       | 1                                        |
| Overall Status | ✅ PASS                                  |

---

## Summary

**All verification checks passed.** The metrics in `final_analysis/` are correctly compiled with valid evidence trails.

## Warnings

| Category    | Message                                                                                  |
| ----------- | ---------------------------------------------------------------------------------------- |
| v2_pipeline | verify_v2_pipeline.py: ✅ PASS with 2 warnings. Review warnings before paper submission. |

## Passed Checks (Sample)

| Category         | Message                                                              |
| ---------------- | -------------------------------------------------------------------- |
| cross_ref        | 40 checks passed                                                     |
|                  | - C01.games_played: master=10.0 == per_config=10.0                   |
|                  | - C01.human_win_rate: master=1.0 == per_config=1.0                   |
|                  | - C01.judge_mean_awareness: values match (~8.96)                     |
|                  | ... and 37 more                                                      |
| deception_logic  | 20 checks passed                                                     |
|                  | - C01: factual_lie_rate_crewmate=0.0 (correct - crewmates don't lie) |
|                  | - C01: factual_lie_rate_human_impostor is empty (human is crewmate)  |
|                  | - C01: kills_per_game_human=0.0 (human is crewmate)                  |
|                  | ... and 17 more                                                      |
| evidence_rows    | 48 checks passed                                                     |
|                  | - C01_kill_metrics_v2_evidence.csv: 15 rows                          |
|                  | - C01_vote_metrics_v2_evidence.csv: 11 rows                          |
|                  | - C01_deception_metrics_v2_evidence.csv: 116 rows                    |
|                  | ... and 45 more                                                      |
| file_exists      | 82 checks passed                                                     |
|                  | - master_comparison_table_v2.csv exists                              |
|                  | - config_mapping.csv exists                                          |
|                  | - C01_game_outcomes.csv exists                                       |
|                  | ... and 79 more                                                      |
| master_integrity | 27 checks passed                                                     |
|                  | - Master table has exactly 8 rows (one per config)                   |
|                  | - Config C01 present in master table                                 |
|                  | - Config C02 present in master table                                 |
|                  | ... and 24 more                                                      |
| missing_values   | 88 checks passed                                                     |
|                  | - C01.config_id=C01                                                  |
|                  | - C01.llm_model=claude-3.5-haiku                                     |
|                  | - C01.human_role=crewmate                                            |
|                  | ... and 85 more                                                      |
| spot_check       | 38 checks passed                                                     |
|                  | - C01: games_played=10 matches config_mapping count=10               |
|                  | - C02: games_played=10 matches config_mapping count=10               |
|                  | - C03: games_played=10 matches config_mapping count=10               |
|                  | ... and 35 more                                                      |
| value_range      | 72 checks passed                                                     |
|                  | - C01.human_win_rate=1.0 in [0,1]                                    |
|                  | - C01.impostor_win_rate=0.0 in [0,1]                                 |
|                  | - C01.crewmate_win_rate=1.0 in [0,1]                                 |
|                  | ... and 69 more                                                      |

---

## Verification Steps Performed

1. **File Existence**: Verified all expected files exist (master table, per-config CSVs, evidence CSVs)
2. **Master Table Integrity**: Checked row count (8), column presence (84 cols), value ranges
3. **Evidence Row Counts**: Verified all evidence files have expected row counts
4. **Cross-Reference**: Compared master table values against per-config files
5. **Spot-Check Raw Logs**: Sampled 10 games and verified raw log existence/validity
6. **Deception Logic**: Verified crewmate_lie_rate=0, role-specific metrics match human_role
7. **Missing Values**: Checked core columns for NaN/empty values
8. **v2 Pipeline**: Ran existing verify_v2_pipeline.py (232 checks)

---

_This verification was performed by GPT-5.2 (via GitHub Copilot) via GitHub Copilot on request from the user._
