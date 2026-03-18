# Prompt And Decision-Policy Changelog

This file tracks prompt and runtime decision-policy changes that can affect experiment comparability.

## Scope
- Included: system/user prompt composition, hard rules, action parsing constraints, fallback guards, prompt-profile flags, and prompt-related logging.
- Excluded: pure UI/frontend changes unless they alter agent prompting or executed action logic.

## Chronological Changes (Verified)

| Date | Commit | Files | What changed | Comparability impact |
|---|---|---|---|---|
| 2026-02-24 | `0798b53` (`version 2`) | `among-agents/amongagents/agent/agent.py` | Added witnessed-kill extraction and `Structured Evidence` block. Added meeting `Hard Rule` scaffolding tied to witnessed kills. | Prompt content changed in meeting/task turns where kill evidence exists. |
| 2026-02-26 | `cc347f2` (`Version 3`) | `among-agents/amongagents/agent/agent.py` | Added strict action protocol with `FINAL_ACTION_INDEX` and `FINAL_SPEAK_MESSAGE`. Added alive/eliminated player constraints in user prompt. Added meeting-specific guidance and filtering for alive suspects. | Output contract tightened; parsing behavior and meeting pressure changed. |
| 2026-02-26 | `0f210ea` (`v3.4`) | `among-agents/amongagents/agent/agent.py` (+ game/server helpers) | Refined runtime fallback/selection behavior around meeting actions and safeguards. | Can alter executed action when model output is partial/invalid. |
| 2026-02-28 | `f266303` (`fixed fallback bug with imp`) | `among-agents/amongagents/agent/agent.py`, `tournament_env_templates.env` | Added prompt versioning: `PROMPT_PROFILE` (`baseline_v1`/`aggressive_v1`) and `AGGRESSION_LEVEL` (1..5). Added aggressive addendum (system prompt + per-turn hard-rule suffix). Logged `prompt_profile` and `aggression_level` in structured logs. Fixed impostor fallback so forced witnessed-kill speech is not injected for impostors. | New experimental condition introduced without mutating baseline by default. |

## Historical Prompt Milestones (By Commit Subject)

These are earlier prompt-related milestones inferred from commit subjects:

| Date | Commit | Subject |
|---|---|---|
| 2025-01-19 | `b3932fe` | convert among-agents from submodule to folder |
| 2025-01-25 | `919793e` / `4dcb0a4` / `d4b88f8` | first llama vs phi battles and full-game following |
| 2025-01-30 | `776c20d` | first signs of deception game (v3 with llama) |
| 2026-02-24 | `9a27939` | added better logging |
| 2026-02-26 | `cc347f2` | Version 3 |
| 2026-02-27 | `6429123` | v4 |
| 2026-02-28 | `f266303` | fixed fallback bug with imp |

## Current Prompt Profiles

### `baseline_v1`
- Legacy/default behavior.
- No aggression addendum applied.
- Use for continuity with prior runs.

### `aggressive_v1`
- Adds aggressive directives while preserving output schema and parser contract.
- Controlled by `AGGRESSION_LEVEL` (1..5).

## Environment Controls

Add to `.env`:

```env
PROMPT_PROFILE=baseline_v1
AGGRESSION_LEVEL=3
```

For aggressive condition:

```env
PROMPT_PROFILE=aggressive_v1
AGGRESSION_LEVEL=4
```

## Reproducibility And Audit Fields

Use these logged fields to stratify results:

- `system_prompt_hash`
- `prompt_profile`
- `aggression_level`
- `raw_prompt_text`
- `raw_response_text`
- `turn_id`, `game_id`, `run_id`

Primary locations:
- `expt-logs/.../structured-v1/api_calls_v1.jsonl`
- `expt-logs/.../structured-v1/agent_turns_v1.jsonl`
- `expt-logs/.../agent-logs.json`

## Recommended Reporting Practice

1. Report baseline and aggressive results separately.
2. Do not pool runs across different `prompt_profile` values.
3. For each table/plot, include `system_prompt_hash` (or list hashes per condition).
4. If fallback guards are active, report both raw model text and executed event text when analyzing dialogue behavior.

## Quick Commands To Rebuild This Changelog

```powershell
git log --date=short --pretty=format:"%h %ad %s" -- among-agents/amongagents/agent/agent.py among-agents/amongagents/agent/neutral_prompts.py tournament_env_templates.env

git show --patch --unified=3 cc347f2 -- among-agents/amongagents/agent/agent.py
git show --patch --unified=3 0798b53 -- among-agents/amongagents/agent/agent.py
git show --patch --unified=3 f266303 -- among-agents/amongagents/agent/agent.py tournament_env_templates.env
```

