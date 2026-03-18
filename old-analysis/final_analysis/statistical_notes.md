# Statistical Notes — Among Us Human vs. LLM Benchmark

Generated from analysis of 8 configurations, 70 complete in-config games.

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
