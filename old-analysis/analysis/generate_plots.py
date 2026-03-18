"""
generate_plots.py  —  Generate all paper figures from analysis/results/ CSVs.

Saves PNG (300 DPI) and PDF to analysis/plots/.
"""

import csv
import json
import math
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
RESULTS_DIR  = r"C:\Users\shiven\Desktop\AmongUs\analysis\results"
EVIDENCE_DIR = r"C:\Users\shiven\Desktop\AmongUs\analysis\evidence"
CONFIG_MAP   = r"C:\Users\shiven\Desktop\AmongUs\analysis\config_mapping.csv"
PLOTS_DIR    = r"C:\Users\shiven\Desktop\AmongUs\analysis\plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

CONFIGS = ["C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08"]

# ─────────────────────────────────────────────────────────────────────────────
# Style & Palette
# ─────────────────────────────────────────────────────────────────────────────
CB_PALETTE = sns.color_palette("colorblind", 10)

MODEL_COLORS = {
    "claude-3.5-haiku":   CB_PALETTE[0],   # blue
    "gemini-2.5-flash":   CB_PALETTE[1],   # orange
    "llama-3.1-8b":       CB_PALETTE[2],   # green
}
MODEL_LABELS = {
    "claude-3.5-haiku":   "Claude 3.5 Haiku",
    "gemini-2.5-flash":   "Gemini 2.5 Flash",
    "llama-3.1-8b":       "Llama 3.1 8B",
}
ROLE_COLORS = {
    "crewmate": CB_PALETTE[0],
    "impostor":  CB_PALETTE[3],
}
PROMPT_HATCH = {
    "baseline_v1":    "",
    "aggressive_v1":  "//",
}

LABEL_FS   = 12
TICK_FS    = 10
TITLE_FS   = 13
ANNOT_FS   = 8.5

CONFIG_XLABELS = {
    "C01": "C01\nClaude\nCrewmate\nBaseline",
    "C02": "C02\nClaude\nImpostor\nBaseline",
    "C03": "C03\nGemini\nCrewmate\nBaseline",
    "C04": "C04\nGemini\nImpostor\nBaseline",
    "C05": "C05\nGemini\nCrewmate\nAggressive",
    "C06": "C06\nGemini\nImpostor\nAggressive",
    "C07": "C07\nLlama\nCrewmate\nBaseline",
    "C08": "C08\nLlama\nImpostor\nBaseline",
}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "axes.grid.axis":    "y",
    "grid.alpha":        0.35,
    "grid.linewidth":    0.6,
    "figure.dpi":        150,
})


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_master():
    path = os.path.join(RESULTS_DIR, "master_comparison_table.csv")
    with open(path, newline="", encoding="utf-8") as f:
        rows = {r["config_id"]: r for r in csv.DictReader(f)}
    return rows


def fv(val, default=float("nan")):
    """Safe float parse."""
    try:
        v = float(val)
        return v if not math.isnan(v) else default
    except Exception:
        return default


def load_config_meta():
    with open(CONFIG_MAP, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    meta = {}
    for r in rows:
        cid = r["config_id"]
        if cid not in meta and cid != "UNMAPPED":
            meta[cid] = r
    return meta


def load_duration_stats():
    """Compute mean and std of game_duration_timesteps per config from config_mapping."""
    with open(CONFIG_MAP, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_cfg = {}
    for r in rows:
        cid = r["config_id"]
        if cid == "UNMAPPED" or r.get("game_complete", "").lower() != "true":
            continue
        try:
            dur = float(r["game_duration_timesteps"])
            by_cfg.setdefault(cid, []).append(dur)
        except Exception:
            pass
    result = {}
    for cid, durs in by_cfg.items():
        result[cid] = {
            "mean": float(np.mean(durs)),
            "std":  float(np.std(durs, ddof=1)) if len(durs) > 1 else 0.0,
            "n":    len(durs),
        }
    return result


def load_per_game_scatter():
    """
    Build per-game (mean_latency, impostor_lie_rate, config_id, model) for scatter.
    Returns list of dicts.
    """
    meta = load_config_meta()
    games = []

    for cfg in CONFIGS:
        # Mean latency per run from evidence
        lat_by_run = {}
        ev_path = os.path.join(EVIDENCE_DIR, f"{cfg}_latency_metrics_evidence.csv")
        if os.path.exists(ev_path):
            with open(ev_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("metric_name") != "api_call_latency":
                        continue
                    run = row["run_id"]
                    try:
                        lat_by_run.setdefault(run, []).append(float(row["metric_value"]))
                    except Exception:
                        pass

        # Impostor lie rate per run from deception evidence
        lie_by_run = {}
        dec_path = os.path.join(EVIDENCE_DIR, f"{cfg}_deception_metrics_evidence.csv")
        if os.path.exists(dec_path):
            with open(dec_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("metric_name") != "claim_deception":
                        continue
                    run = row["run_id"]
                    try:
                        kf = json.loads(row.get("key_fields", "{}"))
                        actor_id = row.get("actor_identity", "")
                    except Exception:
                        continue
                    if actor_id != "Impostor":
                        continue
                    lie_by_run.setdefault(run, {"lie": 0, "total": 0})
                    if kf.get("deception_lie") is True:
                        lie_by_run[run]["lie"] += 1
                    lie_by_run[run]["total"] += 1

        model = meta.get(cfg, {}).get("llm_model", "?")
        for run, lats in lat_by_run.items():
            mean_lat = float(np.mean(lats)) / 1000  # convert to seconds
            lie_rate = None
            if run in lie_by_run and lie_by_run[run]["total"] >= 2:
                d = lie_by_run[run]
                lie_rate = d["lie"] / d["total"]
            games.append({
                "config_id": cfg,
                "run_id":    run,
                "model":     model,
                "mean_lat_s": mean_lat,
                "lie_rate":   lie_rate,
            })

    return games


# ─────────────────────────────────────────────────────────────────────────────
# Shared draw helpers
# ─────────────────────────────────────────────────────────────────────────────

def annotate_bars(ax, bars, fmt="{:.2f}", offset_frac=0.01, fontsize=ANNOT_FS):
    """Add value labels on top of each bar."""
    ymax = max((b.get_height() for b in bars if not math.isnan(b.get_height())), default=1)
    offset = ymax * offset_frac + 0.005
    for b in bars:
        h = b.get_height()
        if math.isnan(h) or h == 0:
            continue
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + offset,
            fmt.format(h),
            ha="center", va="bottom",
            fontsize=fontsize, color="0.25",
        )


def save_fig(fig, name):
    png_path = os.path.join(PLOTS_DIR, f"{name}.png")
    pdf_path = os.path.join(PLOTS_DIR, f"{name}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}.png / {name}.pdf")
    return png_path


def model_legend(ax, models_shown):
    seen = []
    handles = []
    for m in models_shown:
        if m in MODEL_COLORS and m not in seen:
            seen.append(m)
            handles.append(mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_LABELS[m]))
    ax.legend(handles=handles, fontsize=TICK_FS, framealpha=0.85,
              edgecolor="0.7", loc="best")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Human win rate by config
# ─────────────────────────────────────────────────────────────────────────────

def plot_human_win_rate(master):
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(len(CONFIGS))
    meta = load_config_meta()

    bar_colors = [MODEL_COLORS[master[c]["llm_model"]] for c in CONFIGS]
    vals = [fv(master[c]["human_win_rate"]) for c in CONFIGS]
    bars = ax.bar(xs, vals, color=bar_colors, edgecolor="white", linewidth=0.8,
                  zorder=3)
    annotate_bars(ax, bars, fmt="{:.0%}")

    ax.axhline(0.5, color="0.4", linewidth=1.2, linestyle="--", label="50% reference", zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels([CONFIG_XLABELS[c] for c in CONFIGS], fontsize=TICK_FS)
    ax.set_ylabel("Human Win Rate", fontsize=LABEL_FS)
    ax.set_ylim(0, 1.15)
    ax.set_title("Human Win Rate by Configuration", fontsize=TITLE_FS, fontweight="bold")

    models_shown = sorted({master[c]["llm_model"] for c in CONFIGS})
    handles = [mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_LABELS[m])
               for m in models_shown if m in MODEL_COLORS]
    handles.append(plt.Line2D([0], [0], color="0.4", linewidth=1.2,
                               linestyle="--", label="50% baseline"))
    ax.legend(handles=handles, fontsize=TICK_FS, framealpha=0.85, edgecolor="0.7")
    fig.tight_layout()
    return save_fig(fig, "01_human_win_rate_by_config")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Win rate by role per model (baseline only)
# ─────────────────────────────────────────────────────────────────────────────

def plot_win_rate_by_role_per_model(master):
    # baseline_v1 only
    baseline_cfgs = [c for c in CONFIGS if master[c]["prompt_profile"] == "baseline_v1"]
    models_order = ["claude-3.5-haiku", "gemini-2.5-flash", "llama-3.1-8b"]

    # For each model, find crewmate-role and impostor-role configs
    data = {}
    for m in models_order:
        cfgs = [c for c in baseline_cfgs if master[c]["llm_model"] == m]
        crew = next((c for c in cfgs if master[c]["human_role"] == "crewmate"), None)
        imp  = next((c for c in cfgs if master[c]["human_role"] == "impostor"),  None)
        data[m] = {
            "crewmate": fv(master[crew]["human_win_rate"]) if crew else float("nan"),
            "impostor":  fv(master[imp]["human_win_rate"])  if imp  else float("nan"),
            "crew_cfg": crew,
            "imp_cfg":  imp,
        }

    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(len(models_order))
    w = 0.35

    bars_crew = ax.bar(xs - w/2,
                       [data[m]["crewmate"] for m in models_order],
                       width=w, color=ROLE_COLORS["crewmate"],
                       label="Human = Crewmate", edgecolor="white", zorder=3)
    bars_imp  = ax.bar(xs + w/2,
                       [data[m]["impostor"] for m in models_order],
                       width=w, color=ROLE_COLORS["impostor"],
                       label="Human = Impostor", edgecolor="white", zorder=3)

    annotate_bars(ax, bars_crew, fmt="{:.0%}")
    annotate_bars(ax, bars_imp,  fmt="{:.0%}")

    ax.axhline(0.5, color="0.4", linewidth=1.2, linestyle="--", zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models_order], fontsize=LABEL_FS)
    ax.set_ylabel("Human Win Rate", fontsize=LABEL_FS)
    ax.set_ylim(0, 1.18)
    ax.set_title("Human Win Rate by Model × Role (Baseline Configs)", fontsize=TITLE_FS, fontweight="bold")
    ax.legend(fontsize=TICK_FS, framealpha=0.85, edgecolor="0.7")
    fig.tight_layout()
    return save_fig(fig, "02_win_rate_by_role_per_model")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Impostor claim lie rate
# ─────────────────────────────────────────────────────────────────────────────

def plot_claim_lie_rate_impostor(master):
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(len(CONFIGS))
    bar_colors = [MODEL_COLORS[master[c]["llm_model"]] for c in CONFIGS]
    vals = [fv(master[c]["claim_lie_rate_impostor"]) for c in CONFIGS]

    bars = ax.bar(xs, vals, color=bar_colors, edgecolor="white", linewidth=0.8, zorder=3)
    annotate_bars(ax, bars, fmt="{:.0%}")

    ax.set_xticks(xs)
    ax.set_xticklabels([CONFIG_XLABELS[c] for c in CONFIGS], fontsize=TICK_FS)
    ax.set_ylabel("Impostor Claim Lie Rate", fontsize=LABEL_FS)
    ax.set_ylim(0, 1.15)
    ax.set_title("Impostor Claim Lie Rate by Configuration\n(% of impostor speech claims flagged as lies)",
                 fontsize=TITLE_FS, fontweight="bold")
    model_legend(ax, [master[c]["llm_model"] for c in CONFIGS])
    fig.tight_layout()
    return save_fig(fig, "03_claim_lie_rate_impostor")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Ejection accuracy by config
# ─────────────────────────────────────────────────────────────────────────────

def plot_ejection_accuracy(master):
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(len(CONFIGS))
    bar_colors = [MODEL_COLORS[master[c]["llm_model"]] for c in CONFIGS]
    vals = [fv(master[c]["ejection_accuracy"]) for c in CONFIGS]

    bars = ax.bar(xs, vals, color=bar_colors, edgecolor="white", zorder=3)
    annotate_bars(ax, bars, fmt="{:.0%}")

    ax.axhline(0.5, color="0.4", linewidth=1.2, linestyle="--", zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels([CONFIG_XLABELS[c] for c in CONFIGS], fontsize=TICK_FS)
    ax.set_ylabel("Ejection Accuracy", fontsize=LABEL_FS)
    ax.set_ylim(0, 1.18)
    ax.set_title("Ejection Accuracy by Configuration\n(% of ejections that targeted an actual impostor)",
                 fontsize=TITLE_FS, fontweight="bold")
    model_legend(ax, [master[c]["llm_model"] for c in CONFIGS])
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], color="0.4", linewidth=1.2,
                               linestyle="--", label="50% baseline"))
    labels.append("50% baseline")
    ax.legend(handles=handles, labels=labels, fontsize=TICK_FS, framealpha=0.85, edgecolor="0.7")
    fig.tight_layout()
    return save_fig(fig, "04_ejection_accuracy_by_config")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5: Human vs LLM correct vote rate
# ─────────────────────────────────────────────────────────────────────────────

def plot_correct_vote_human_vs_llm(master):
    fig, ax = plt.subplots(figsize=(12, 5))
    xs = np.arange(len(CONFIGS))
    w = 0.35

    human_vals = [fv(master[c]["correct_vote_rate_human"]) for c in CONFIGS]
    llm_vals   = [fv(master[c]["correct_vote_rate_llm"])   for c in CONFIGS]

    bars_h = ax.bar(xs - w/2, human_vals, width=w, color=CB_PALETTE[0],
                    label="Human voter", edgecolor="white", zorder=3)
    bars_l = ax.bar(xs + w/2, llm_vals,   width=w, color=CB_PALETTE[4],
                    label="LLM voter", edgecolor="white", zorder=3)

    annotate_bars(ax, bars_h, fmt="{:.0%}")
    annotate_bars(ax, bars_l, fmt="{:.0%}")

    ax.axhline(0.5, color="0.4", linewidth=1.2, linestyle="--", zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels([CONFIG_XLABELS[c] for c in CONFIGS], fontsize=TICK_FS)
    ax.set_ylabel("Correct Vote Rate", fontsize=LABEL_FS)
    ax.set_ylim(0, 1.18)
    ax.set_title("Correct Vote Rate: Human vs LLM Voters",
                 fontsize=TITLE_FS, fontweight="bold")
    ax.legend(fontsize=TICK_FS, framealpha=0.85, edgecolor="0.7")
    fig.tight_layout()
    return save_fig(fig, "05_correct_vote_rate_human_vs_llm")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6: Kills per game
# ─────────────────────────────────────────────────────────────────────────────

def plot_kills_per_game(master):
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(len(CONFIGS))
    bar_colors = [MODEL_COLORS[master[c]["llm_model"]] for c in CONFIGS]
    vals = [fv(master[c]["kills_per_game"]) for c in CONFIGS]

    bars = ax.bar(xs, vals, color=bar_colors, edgecolor="white", zorder=3)
    annotate_bars(ax, bars, fmt="{:.2f}")

    ax.set_xticks(xs)
    ax.set_xticklabels([CONFIG_XLABELS[c] for c in CONFIGS], fontsize=TICK_FS)
    ax.set_ylabel("Kills per Game", fontsize=LABEL_FS)
    ax.set_ylim(0, max(vals) * 1.22)
    ax.set_title("Kills per Game by Configuration", fontsize=TITLE_FS, fontweight="bold")
    model_legend(ax, [master[c]["llm_model"] for c in CONFIGS])
    fig.tight_layout()
    return save_fig(fig, "06_kills_per_game_by_config")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 7: Impostor survival after witnessed kill
# ─────────────────────────────────────────────────────────────────────────────

def plot_impostor_survival_after_witness(master):
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(len(CONFIGS))
    bar_colors = [MODEL_COLORS[master[c]["llm_model"]] for c in CONFIGS]
    vals = [fv(master[c]["impostor_survival_after_witness"]) for c in CONFIGS]

    bars = ax.bar(xs, vals, color=bar_colors, edgecolor="white", zorder=3)
    annotate_bars(ax, bars, fmt="{:.0%}")

    ax.set_xticks(xs)
    ax.set_xticklabels([CONFIG_XLABELS[c] for c in CONFIGS], fontsize=TICK_FS)
    ax.set_ylabel("Impostor Survival Rate", fontsize=LABEL_FS)
    ax.set_ylim(0, 1.18)
    ax.set_title("Impostor Survival After Witnessed Kill\n(Key deception-under-pressure metric)",
                 fontsize=TITLE_FS, fontweight="bold")
    model_legend(ax, [master[c]["llm_model"] for c in CONFIGS])
    fig.tight_layout()
    return save_fig(fig, "07_impostor_survival_after_witness")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 8: Latency by model — bar chart with std error bars
# ─────────────────────────────────────────────────────────────────────────────

def plot_latency_by_model(master):
    # Aggregate across all configs per model (weighted by n_calls)
    model_data = {}
    for c in CONFIGS:
        m   = master[c]["llm_model"]
        lat = fv(master[c]["mean_latency_ms"])
        std = fv(master[c]["std_latency_ms"])
        n   = fv(master[c]["total_api_calls"], 1)
        if math.isnan(lat):
            continue
        model_data.setdefault(m, []).append((lat, std, n))

    models = ["claude-3.5-haiku", "gemini-2.5-flash", "llama-3.1-8b"]
    means, errs = [], []
    for m in models:
        entries = model_data.get(m, [])
        if not entries:
            means.append(float("nan"))
            errs.append(0)
            continue
        total_n = sum(e[2] for e in entries)
        w_mean  = sum(e[0] * e[2] for e in entries) / total_n
        # pooled std estimate (simple average of per-config stds)
        avg_std = float(np.mean([e[1] for e in entries]))
        means.append(w_mean / 1000)   # ms → s
        errs.append(avg_std / 1000)

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(len(models))
    colors = [MODEL_COLORS[m] for m in models]
    bars = ax.bar(xs, means, yerr=errs, capsize=6, color=colors,
                  edgecolor="white", linewidth=0.8, error_kw={"linewidth": 1.5},
                  zorder=3)
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width()/2, v + max(errs)*0.06 + 0.05,
                f"{v:.2f}s", ha="center", va="bottom", fontsize=ANNOT_FS, color="0.25")

    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], fontsize=LABEL_FS)
    ax.set_ylabel("Mean API Latency (seconds)", fontsize=LABEL_FS)
    ax.set_title("API Latency by Model\n(mean ± std, pooled across configs)",
                 fontsize=TITLE_FS, fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, "08_latency_by_model")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 9: Latency vs lie rate scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_latency_vs_lie_rate_scatter(per_game):
    pts = [g for g in per_game
           if g["lie_rate"] is not None and not math.isnan(g["mean_lat_s"])]

    if len(pts) < 5:
        print("  [SKIP] latency_vs_lie_rate_scatter: insufficient data")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))

    models_shown = sorted({p["model"] for p in pts if p["model"] in MODEL_COLORS})
    for m in models_shown:
        mpts = [p for p in pts if p["model"] == m]
        xs = [p["mean_lat_s"] for p in mpts]
        ys = [p["lie_rate"]   for p in mpts]
        ax.scatter(xs, ys, color=MODEL_COLORS[m], label=MODEL_LABELS.get(m, m),
                   alpha=0.75, s=60, zorder=4, edgecolors="white", linewidths=0.6)

    # Trend line across all points
    all_x = np.array([p["mean_lat_s"] for p in pts])
    all_y = np.array([p["lie_rate"]   for p in pts])
    if len(all_x) >= 5:
        m_coef, b_coef = np.polyfit(all_x, all_y, 1)
        x_line = np.linspace(all_x.min(), all_x.max(), 100)
        ax.plot(x_line, m_coef * x_line + b_coef,
                color="0.35", linewidth=1.5, linestyle="--",
                label=f"OLS trend (pooled, n={len(pts)})", zorder=5)

    ax.set_xlabel("Mean Game Latency (seconds/call)", fontsize=LABEL_FS)
    ax.set_ylabel("Impostor Claim Lie Rate", fontsize=LABEL_FS)
    ax.set_ylim(-0.05, 1.15)
    ax.set_title("Mean API Latency vs Impostor Lie Rate (per game)\n"
                 "[NOTE: all per-config correlations are low-power, n≤10]",
                 fontsize=TITLE_FS, fontweight="bold")
    ax.legend(fontsize=TICK_FS, framealpha=0.85, edgecolor="0.7")
    fig.tight_layout()
    return save_fig(fig, "09_latency_vs_lie_rate_scatter")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 10: Aggression comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_aggression_comparison(master):
    """
    4-cluster grouped bar chart: win_rate, impostor_lie_rate, kills_per_game,
    impostor_detection_rate.
    Each cluster: 4 bars — C03 (crew base), C05 (crew aggr), C04 (imp base), C06 (imp aggr).
    """
    metrics = [
        ("human_win_rate",                  "Human Win Rate"),
        ("claim_lie_rate_impostor",         "Impostor Lie Rate"),
        ("kills_per_game",                  "Kills / Game"),
        ("impostor_detection_rate",         "Impostor Detection Rate"),
    ]

    # Normalise kills to [0,1] range for display on shared y-axis
    # Instead keep separate y-axes — simpler: just do multi-subplot
    fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharey=False)

    pairs = [
        ("C03", "C05", "crewmate\nbaseline", "crewmate\naggressive"),
        ("C04", "C06", "impostor\nbaseline",  "impostor\naggressive"),
    ]

    for ax_idx, (met_key, met_label) in enumerate(metrics):
        ax = axes[ax_idx]
        xs = np.arange(2)   # baseline, aggressive
        w = 0.35

        vals_crew = [fv(master["C03"][met_key]), fv(master["C05"][met_key])]
        vals_imp  = [fv(master["C04"][met_key]), fv(master["C06"][met_key])]

        b1 = ax.bar(xs - w/2, vals_crew, width=w, color=ROLE_COLORS["crewmate"],
                    label="Human = Crewmate", edgecolor="white", zorder=3)
        b2 = ax.bar(xs + w/2, vals_imp,  width=w, color=ROLE_COLORS["impostor"],
                    label="Human = Impostor", edgecolor="white", zorder=3)

        for b, v in zip(list(b1) + list(b2), vals_crew + vals_imp):
            if not math.isnan(v):
                ax.text(b.get_x() + b.get_width()/2,
                        v + max([v for v in vals_crew + vals_imp if not math.isnan(v)] or [1]) * 0.03 + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color="0.25")

        ax.set_xticks(xs)
        ax.set_xticklabels(["Baseline", "Aggressive"], fontsize=TICK_FS)
        ax.set_title(met_label, fontsize=TICK_FS + 1, fontweight="bold")
        ax.tick_params(axis="y", labelsize=TICK_FS)

        if ax_idx == 0:
            ax.set_ylabel("Value", fontsize=LABEL_FS)
        if ax_idx == 0:
            ax.legend(fontsize=8, framealpha=0.85, edgecolor="0.7")

    fig.suptitle("Gemini Baseline vs Aggressive Prompt — Key Metrics (n=10 vs n=5, low power)",
                 fontsize=TITLE_FS, fontweight="bold", y=1.02)
    fig.tight_layout()
    return save_fig(fig, "10_aggression_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 11: Thinking depth by role
# ─────────────────────────────────────────────────────────────────────────────

def plot_thinking_depth_by_role(master):
    """Bar chart per model: impostor vs crewmate thinking depth."""
    baseline_cfgs = [c for c in CONFIGS if master[c]["prompt_profile"] == "baseline_v1"]
    models_order  = ["claude-3.5-haiku", "gemini-2.5-flash", "llama-3.1-8b"]

    # Average thinking depth per model across roles (impostor configs + crewmate configs)
    model_imp, model_crew = {}, {}
    for c in baseline_cfgs:
        m = master[c]["llm_model"]
        imp_d  = fv(master[c]["thinking_depth_impostor"])
        crew_d = fv(master[c]["thinking_depth_crewmate"])
        model_imp.setdefault(m,  []).append(imp_d)
        model_crew.setdefault(m, []).append(crew_d)

    imp_means  = [float(np.nanmean(model_imp.get(m, [float("nan")])))  for m in models_order]
    crew_means = [float(np.nanmean(model_crew.get(m, [float("nan")]))) for m in models_order]

    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(len(models_order))
    w  = 0.35

    b_imp  = ax.bar(xs - w/2, imp_means,  width=w, color=ROLE_COLORS["impostor"],
                    label="Impostor turns", edgecolor="white", zorder=3)
    b_crew = ax.bar(xs + w/2, crew_means, width=w, color=ROLE_COLORS["crewmate"],
                    label="Crewmate turns", edgecolor="white", zorder=3)

    for b, v in zip(list(b_imp) + list(b_crew), imp_means + crew_means):
        if not math.isnan(v):
            ax.text(b.get_x() + b.get_width()/2, v + 1.5,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=ANNOT_FS, color="0.25")

    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models_order], fontsize=LABEL_FS)
    ax.set_ylabel("Mean Thinking Depth (words)", fontsize=LABEL_FS)
    ax.set_ylim(0, max(imp_means + crew_means) * 1.20)
    ax.set_title("Thinking Depth by Role × Model\n([Thinking Process] section word count)",
                 fontsize=TITLE_FS, fontweight="bold")
    ax.legend(fontsize=TICK_FS, framealpha=0.85, edgecolor="0.7")
    fig.tight_layout()
    return save_fig(fig, "11_thinking_depth_by_role")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 12: Game duration by config with error bars
# ─────────────────────────────────────────────────────────────────────────────

def plot_game_duration_by_config(master, dur_stats):
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(len(CONFIGS))
    bar_colors = [MODEL_COLORS[master[c]["llm_model"]] for c in CONFIGS]

    means = [dur_stats.get(c, {}).get("mean", float("nan")) for c in CONFIGS]
    stds  = [dur_stats.get(c, {}).get("std",  0.0)          for c in CONFIGS]

    bars = ax.bar(xs, means, yerr=stds, capsize=5, color=bar_colors,
                  edgecolor="white", linewidth=0.8,
                  error_kw={"linewidth": 1.5, "ecolor": "0.3"},
                  zorder=3)

    for b, v in zip(bars, means):
        if not math.isnan(v):
            ax.text(b.get_x() + b.get_width()/2,
                    v + max(stds) * 0.06 + 0.3,
                    f"{v:.1f}", ha="center", va="bottom",
                    fontsize=ANNOT_FS, color="0.25")

    ax.set_xticks(xs)
    ax.set_xticklabels([CONFIG_XLABELS[c] for c in CONFIGS], fontsize=TICK_FS)
    ax.set_ylabel("Mean Game Duration (timesteps)", fontsize=LABEL_FS)
    ax.set_ylim(0, max(m + s for m, s in zip(means, stds) if not math.isnan(m)) * 1.18)
    ax.set_title("Mean Game Duration by Configuration (±1 SD)",
                 fontsize=TITLE_FS, fontweight="bold")
    model_legend(ax, [master[c]["llm_model"] for c in CONFIGS])
    fig.tight_layout()
    return save_fig(fig, "12_game_duration_by_config")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 13: Task completion rate by config
# ─────────────────────────────────────────────────────────────────────────────

def plot_task_completion_by_config(master):
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(len(CONFIGS))
    bar_colors = [MODEL_COLORS[master[c]["llm_model"]] for c in CONFIGS]

    # task_completion_rate can exceed 1.0 (duplicate log events); cap display at 1.1
    vals = [min(fv(master[c]["task_completion_rate"]), 1.1) for c in CONFIGS]
    raw  = [fv(master[c]["task_completion_rate"]) for c in CONFIGS]

    bars = ax.bar(xs, vals, color=bar_colors, edgecolor="white", zorder=3)

    for b, v_raw in zip(bars, raw):
        label = f"{v_raw:.0%}" if not math.isnan(v_raw) else "N/A"
        ax.text(b.get_x() + b.get_width()/2,
                min(v_raw, 1.1) + 0.015,
                label, ha="center", va="bottom",
                fontsize=ANNOT_FS, color="0.25")

    ax.axhline(1.0, color="0.4", linewidth=1.2, linestyle="--",
               label="100% (all tasks complete)", zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels([CONFIG_XLABELS[c] for c in CONFIGS], fontsize=TICK_FS)
    ax.set_ylabel("Task Completion Rate", fontsize=LABEL_FS)
    ax.set_ylim(0, 1.22)
    ax.set_title("Crewmate Task Completion Rate by Configuration\n"
                 "(>100% = duplicate task-complete events in logs; see anomaly note)",
                 fontsize=TITLE_FS, fontweight="bold")

    handles = [mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_LABELS[m])
               for m in sorted({master[c]["llm_model"] for c in CONFIGS}) if m in MODEL_COLORS]
    handles.append(plt.Line2D([0], [0], color="0.4", linewidth=1.2,
                               linestyle="--", label="100% completion"))
    ax.legend(handles=handles, fontsize=TICK_FS, framealpha=0.85, edgecolor="0.7")
    fig.tight_layout()
    return save_fig(fig, "13_task_completion_by_config")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("GENERATING PLOTS")
    print(f"  Output: {PLOTS_DIR}")
    print("=" * 65)

    master    = load_master()
    dur_stats = load_duration_stats()
    per_game  = load_per_game_scatter()

    generated = []
    skipped   = []

    def run(label, fn, *args):
        try:
            result = fn(*args)
            if result:
                generated.append(label)
            else:
                skipped.append((label, "insufficient data"))
        except Exception as e:
            import traceback
            skipped.append((label, str(e)))
            print(f"  [ERROR] {label}: {e}")
            traceback.print_exc()

    run("01_human_win_rate_by_config",          plot_human_win_rate,                   master)
    run("02_win_rate_by_role_per_model",         plot_win_rate_by_role_per_model,       master)
    run("03_claim_lie_rate_impostor",            plot_claim_lie_rate_impostor,          master)
    run("04_ejection_accuracy_by_config",        plot_ejection_accuracy,                master)
    run("05_correct_vote_rate_human_vs_llm",     plot_correct_vote_human_vs_llm,        master)
    run("06_kills_per_game_by_config",           plot_kills_per_game,                   master)
    run("07_impostor_survival_after_witness",    plot_impostor_survival_after_witness,  master)
    run("08_latency_by_model",                   plot_latency_by_model,                 master)
    run("09_latency_vs_lie_rate_scatter",        plot_latency_vs_lie_rate_scatter,      per_game)
    run("10_aggression_comparison",              plot_aggression_comparison,            master)
    run("11_thinking_depth_by_role",             plot_thinking_depth_by_role,           master)
    run("12_game_duration_by_config",            plot_game_duration_by_config,          master, dur_stats)
    run("13_task_completion_by_config",          plot_task_completion_by_config,        master)

    print()
    print("=" * 65)
    print(f"DONE  |  Generated: {len(generated)}  |  Skipped: {len(skipped)}")
    print("=" * 65)
    if skipped:
        print("\nSkipped / failed:")
        for name, reason in skipped:
            print(f"  {name}: {reason}")

    # List all files
    print("\nFiles in plots/:")
    for f in sorted(os.listdir(PLOTS_DIR)):
        size_kb = os.path.getsize(os.path.join(PLOTS_DIR, f)) // 1024
        print(f"  {f}  ({size_kb} KB)")


if __name__ == "__main__":
    main()
