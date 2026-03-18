"""
verify_with_llm.py  —  Task E: LLM-assisted evidence verification.

For each config, samples up to SAMPLE_PER_GROUP evidence rows per metric group,
loads the raw log entry from source_file using event_ids / line_number,
constructs a verification prompt, sends it to a configurable LLM judge,
and outputs: analysis/llm_verification_report.csv

DO NOT RUN automatically — run manually after reviewing deterministic verification.

Usage:
    python analysis/verify_with_llm.py \
        [--model gemini-2.5-flash] \
        [--sample 15] \
        [--groups game_outcomes kill_metrics voting_metrics deception_metrics]

Environment:
    GOOGLE_API_KEY  — for Gemini models (default)
    ANTHROPIC_API_KEY  — for Claude models
    OPENAI_API_KEY  — for OpenAI models

Configure LLM_PROVIDER and LLM_MODEL below, or pass via --model flag.
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

EVIDENCE_DIR  = r"C:\Users\shiven\Desktop\AmongUs\analysis\evidence"
OUTPUT_FILE   = r"C:\Users\shiven\Desktop\AmongUs\analysis\llm_verification_report.csv"

DEFAULT_MODEL    = "gemini-2.5-flash"
DEFAULT_SAMPLE   = 15          # max rows sampled per metric group per config
RETRY_DELAY_SEC  = 2.0         # seconds between API retries on rate-limit
MAX_RETRIES      = 3

# Metric groups to verify (map to evidence file suffix)
METRIC_GROUPS = {
    "game_outcomes":      "game_outcomes_evidence.csv",
    "kill_metrics":       "kill_metrics_evidence.csv",
    "voting_metrics":     "voting_metrics_evidence.csv",
    "deception_metrics":  "deception_metrics_evidence.csv",
    "task_metrics":       "task_metrics_evidence.csv",
    "judge_metrics":      "judge_metrics_evidence.csv",
    # latency_metrics_evidence has thousands of rows — sample more carefully
    "latency_metrics":    "latency_metrics_evidence.csv",
}

CONFIGS = ["C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08"]

# ─────────────────────────────────────────────────────────────────────────────
# Source file reading
# ─────────────────────────────────────────────────────────────────────────────

_file_cache: dict[str, list[str]] = {}


def read_source_line(source_file: str, line_number: int) -> tuple[Optional[dict], Optional[str]]:
    """Return (json_obj, error). 1-indexed line_number."""
    if source_file not in _file_cache:
        try:
            with open(source_file, encoding="utf-8", errors="replace") as f:
                _file_cache[source_file] = f.readlines()
        except FileNotFoundError:
            _file_cache[source_file] = []
    lines = _file_cache[source_file]
    if not lines:
        return None, f"File not found: {source_file}"
    idx = line_number - 1
    if idx < 0 or idx >= len(lines):
        return None, f"Line {line_number} out of range (file has {len(lines)} lines)"
    raw = lines[idx].strip()
    if not raw:
        return None, f"Line {line_number} is empty"
    try:
        return json.loads(raw), None
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"


def load_raw_log_entry(ev_row: dict) -> tuple[Optional[str], Optional[str]]:
    """
    Load the raw log entry for an evidence row.
    Returns (raw_json_string, error_message).
    """
    source_file = ev_row.get("source_file", "").strip()
    line_number  = ev_row.get("line_number", "").strip()

    if not source_file or not line_number:
        return None, "Missing source_file or line_number"

    try:
        ln = int(line_number)
    except ValueError:
        return None, f"Invalid line_number: {line_number!r}"

    obj, err = read_source_line(source_file, ln)
    if err:
        return None, err

    # Return a concise but complete JSON string (strip huge fields like raw_prompt_text)
    trimmed = {k: v for k, v in obj.items()
               if k not in ("raw_prompt_text", "normalized_prompt_text",
                            "raw_response_text", "normalized_response_text",
                            "response_headers", "actor_state_snapshot_hash",
                            "audit_flags")}
    return json.dumps(trimmed, indent=2)[:4000], None  # cap at 4000 chars


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────────

VERIFICATION_PROMPT_TEMPLATE = """\
You are a careful data verifier for an Among Us human-vs-LLM experiment benchmark.

Given the raw log entry below (JSON), determine whether the metric computation claim is correct.

## Raw log entry (from {source_file_name}, line {line_number}):
```json
{raw_json}
```

## Metric computation claim:
- metric_name: {metric_name}
- metric_value: {metric_value}
- event_ids: {event_ids}
- key_fields claimed: {key_fields}

## Verification task:
1. Check whether the key_fields values match what is actually in the raw log entry.
2. Check whether the metric_value is consistent with the key_fields and the raw log entry.
3. Note any discrepancy, even minor ones.

Reply with EXACTLY one of:
  PASS — the raw log supports the claim; key_fields match and metric_value is consistent.
  FAIL — there is a clear discrepancy between the raw log and the claim.
  WARN — the raw log partially supports the claim but something is uncertain or ambiguous.

Then provide a one-sentence explanation (max 120 chars).

Format your reply as:
VERDICT: <PASS|FAIL|WARN>
REASON: <one sentence>
"""


def build_prompt(ev_row: dict, raw_json: str) -> str:
    source_name = os.path.basename(ev_row.get("source_file", "unknown"))
    return VERIFICATION_PROMPT_TEMPLATE.format(
        source_file_name = source_name,
        line_number      = ev_row.get("line_number", "?"),
        raw_json         = raw_json,
        metric_name      = ev_row.get("metric_name", "?"),
        metric_value     = ev_row.get("metric_value", "?"),
        event_ids        = ev_row.get("event_ids", ""),
        key_fields       = ev_row.get("key_fields", "{}"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM client abstraction
# ─────────────────────────────────────────────────────────────────────────────

def detect_provider(model: str) -> str:
    m = model.lower()
    if "gemini" in m:
        return "gemini"
    if "claude" in m or "anthropic" in m:
        return "anthropic"
    if "gpt" in m or "openai" in m:
        return "openai"
    raise ValueError(f"Cannot auto-detect provider for model: {model!r}. "
                     "Set LLM_PROVIDER env var or specify --model with a recognizable name.")


def call_gemini(prompt: str, model: str) -> str:
    import google.generativeai as genai
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY not set")
    genai.configure(api_key=api_key)
    m = genai.GenerativeModel(model)
    resp = m.generate_content(prompt)
    return resp.text.strip()


def call_anthropic(prompt: str, model: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    msg = client.messages.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def call_openai(prompt: str, model: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
    )
    return resp.choices[0].message.content.strip()


def call_llm(prompt: str, model: str, provider: str) -> str:
    """Call the LLM with retries on rate-limit errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if provider == "gemini":
                return call_gemini(prompt, model)
            elif provider == "anthropic":
                return call_anthropic(prompt, model)
            elif provider == "openai":
                return call_openai(prompt, model)
            else:
                raise ValueError(f"Unknown provider: {provider!r}")
        except Exception as e:
            err_str = str(e).lower()
            if any(x in err_str for x in ("rate", "quota", "429", "timeout")):
                if attempt < MAX_RETRIES:
                    print(f"    [rate limit / timeout, retry {attempt}/{MAX_RETRIES}]")
                    time.sleep(RETRY_DELAY_SEC * attempt)
                    continue
            raise
    raise RuntimeError(f"LLM call failed after {MAX_RETRIES} retries")


# ─────────────────────────────────────────────────────────────────────────────
# Response parsing
# ─────────────────────────────────────────────────────────────────────────────

_VERDICT_RE = re.compile(r"VERDICT\s*[:=]\s*(PASS|FAIL|WARN)", re.IGNORECASE)
_REASON_RE  = re.compile(r"REASON\s*[:=]\s*(.+)", re.IGNORECASE)


def parse_response(response_text: str) -> tuple[str, str]:
    """Return (verdict, reason). verdict in {PASS, FAIL, WARN, UNKNOWN}."""
    vm = _VERDICT_RE.search(response_text)
    rm = _REASON_RE.search(response_text)
    verdict = vm.group(1).upper() if vm else "UNKNOWN"
    reason  = rm.group(1).strip()[:200] if rm else response_text[:200]
    return verdict, reason


# ─────────────────────────────────────────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_evidence_rows(config_id: str, group_name: str, suffix: str,
                          n: int) -> list[tuple[dict, str]]:
    """
    Return up to n sampled (evidence_row, evidence_filename) pairs for
    the given config and metric group.
    """
    fname = f"{config_id}_{suffix}"
    fpath = os.path.join(EVIDENCE_DIR, fname)
    if not os.path.exists(fpath):
        return []

    with open(fpath, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Filter to rows that have a source_file and line_number (skip summary rows)
    rows = [r for r in rows if r.get("source_file", "").strip()
            and r.get("line_number", "").strip()]

    if len(rows) <= n:
        return [(r, fname) for r in rows]

    random.seed(42)  # reproducible sample
    return [(r, fname) for r in random.sample(rows, n)]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM-assisted evidence verification")
    parser.add_argument("--model",  default=DEFAULT_MODEL,
                        help=f"LLM model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--provider", default=None,
                        help="LLM provider: gemini|anthropic|openai (auto-detected from model if omitted)")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE,
                        help=f"Max evidence rows per metric group per config (default: {DEFAULT_SAMPLE})")
    parser.add_argument("--groups", nargs="*", default=list(METRIC_GROUPS.keys()),
                        help="Metric groups to verify")
    parser.add_argument("--configs", nargs="*", default=CONFIGS,
                        help="Config IDs to verify")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    random.seed(args.seed)
    model    = args.model
    provider = args.provider or detect_provider(model)
    sample_n = args.sample

    print("=" * 70)
    print("LLM EVIDENCE VERIFICATION")
    print(f"  Model:    {model}  (provider: {provider})")
    print(f"  Sample:   up to {sample_n} rows per group per config")
    print(f"  Groups:   {args.groups}")
    print(f"  Configs:  {args.configs}")
    print("=" * 70)

    report_rows = []
    total = passes = fails = warns = errors = 0

    for config_id in args.configs:
        for group_name in args.groups:
            suffix = METRIC_GROUPS.get(group_name)
            if not suffix:
                print(f"  Unknown group: {group_name} — skipping")
                continue

            samples = sample_evidence_rows(config_id, group_name, suffix, sample_n)
            if not samples:
                continue

            print(f"\n  {config_id} / {group_name}: {len(samples)} rows")

            for ev_row, ev_fname in samples:
                total += 1
                metric_name  = ev_row.get("metric_name", "?")
                metric_value = ev_row.get("metric_value", "?")

                # Load raw log entry
                raw_json, load_err = load_raw_log_entry(ev_row)
                if load_err:
                    print(f"    [LOAD_ERR] {metric_name}: {load_err}")
                    report_rows.append({
                        "evidence_file":    ev_fname,
                        "config_id":        config_id,
                        "metric_group":     group_name,
                        "metric_name":      metric_name,
                        "metric_value":     metric_value,
                        "event_ids":        ev_row.get("event_ids", ""),
                        "source_file":      ev_row.get("source_file", ""),
                        "line_number":      ev_row.get("line_number", ""),
                        "llm_verdict":      "ERROR",
                        "llm_reason":       load_err,
                        "model_used":       model,
                        "prompt_preview":   "",
                    })
                    errors += 1
                    continue

                # Build prompt
                prompt = build_prompt(ev_row, raw_json)

                # Call LLM
                try:
                    response = call_llm(prompt, model, provider)
                    verdict, reason = parse_response(response)
                except Exception as e:
                    verdict, reason = "ERROR", str(e)[:200]
                    errors += 1

                # Track counts
                if verdict == "PASS":
                    passes += 1
                elif verdict == "FAIL":
                    fails += 1
                elif verdict == "WARN":
                    warns += 1

                marker = "  FAIL ***" if verdict == "FAIL" else (
                         "  WARN   " if verdict == "WARN" else ""
                )
                print(f"    [{verdict}] {metric_name}={metric_value!r:.30}{marker}")
                if verdict in ("FAIL", "WARN"):
                    print(f"           reason: {reason}")

                report_rows.append({
                    "evidence_file":    ev_fname,
                    "config_id":        config_id,
                    "metric_group":     group_name,
                    "metric_name":      metric_name,
                    "metric_value":     metric_value,
                    "event_ids":        ev_row.get("event_ids", ""),
                    "source_file":      ev_row.get("source_file", ""),
                    "line_number":      ev_row.get("line_number", ""),
                    "llm_verdict":      verdict,
                    "llm_reason":       reason,
                    "model_used":       model,
                    "prompt_preview":   prompt[:300].replace("\n", " "),
                })

    # Write report
    fieldnames = ["evidence_file", "config_id", "metric_group", "metric_name",
                  "metric_value", "event_ids", "source_file", "line_number",
                  "llm_verdict", "llm_reason", "model_used", "prompt_preview"]
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(report_rows)

    print()
    print("=" * 70)
    print("LLM VERIFICATION SUMMARY")
    print(f"  Total rows verified : {total}")
    print(f"  PASS                : {passes}  ({100*passes/max(total,1):.1f}%)")
    print(f"  WARN                : {warns}   ({100*warns/max(total,1):.1f}%)")
    print(f"  FAIL                : {fails}   ({100*fails/max(total,1):.1f}%)")
    print(f"  ERROR (load/API)    : {errors}  ({100*errors/max(total,1):.1f}%)")
    print(f"  Model used          : {model}")
    print(f"  Written: {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
