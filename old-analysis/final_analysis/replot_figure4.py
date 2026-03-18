#!/usr/bin/env python3
"""
Replot Figure 4: Factual Lie Rate for LLM Impostors across configurations.
Excludes human impostor lie rate.  Bars are colour-coded by LLM model.
Output: final_analysis/plots/figure4_llm_impostor_lie_rate.{png,pdf}
"""

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ── paths ───────────────────────────────────────────────
BASE = Path(__file__).parent.parent
MASTER = BASE / "final_analysis" / "master_comparison_table_v2.csv"
OUT_DIR = BASE / "final_analysis" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── load data ───────────────────────────────────────────
with open(MASTER, encoding="utf-8") as f:
    rows = {r["config_id"]: r for r in csv.DictReader(f)}

CONFIGS = ["C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08"]

def fv(val, default=float("nan")):
    try:
        v = float(val)
        return v if not math.isnan(v) else default
    except (ValueError, TypeError):
        return default

# ── style (matching existing v2 plots) ──────────────────
CB = sns.color_palette("colorblind", 10)
MODEL_COLORS = {
    "claude-3.5-haiku": CB[0],
    "gemini-2.5-flash":  CB[1],
    "llama-3.1-8b":      CB[2],
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

LABEL_FS = 12
TICK_FS  = 10
TITLE_FS = 13
ANNOT_FS = 9

CONFIG_XLABELS = {
    "C01": "C01\nClaude\nCrew\nBase",
    "C02": "C02\nClaude\nImp\nBase",
    "C03": "C03\nGemini\nCrew\nBase",
    "C04": "C04\nGemini\nImp\nBase",
    "C05": "C05\nGemini\nCrew\nAggr",
    "C06": "C06\nGemini\nImp\nAggr",
    "C07": "C07\nLlama\nCrew\nBase",
    "C08": "C08\nLlama\nImp\nBase",
}

# ── data vectors ────────────────────────────────────────
llm_vals = [fv(rows[c].get("factual_lie_rate_llm_impostor")) for c in CONFIGS]
models   = [rows[c].get("llm_model", "") for c in CONFIGS]
colors   = [MODEL_COLORS.get(m, CB[4]) for m in models]

# ── plot ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.5))
xs = np.arange(len(CONFIGS))
bars = ax.bar(xs, llm_vals, width=0.55, color=colors, edgecolor="white", zorder=3)

# annotate values on bars
for b, v in zip(bars, llm_vals):
    if not math.isnan(v) and v != 0:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                f"{v:.2f}", ha="center", va="bottom", fontsize=ANNOT_FS,
                color="0.25")

ax.set_xticks(xs)
ax.set_xticklabels([CONFIG_XLABELS[c] for c in CONFIGS], fontsize=TICK_FS)
ax.set_ylabel("Factual Lie Rate", fontsize=LABEL_FS)
ax.set_title("Factual Lie Rate for LLM Impostors Across Configurations",
             fontsize=TITLE_FS, fontweight="bold")
ax.set_ylim(0, max(v for v in llm_vals if not math.isnan(v)) * 1.18)

# legend by model
from matplotlib.patches import Patch
legend_handles = [Patch(facecolor=MODEL_COLORS[m], edgecolor="white", label=m)
                  for m in ["claude-3.5-haiku", "gemini-2.5-flash", "llama-3.1-8b"]]
ax.legend(handles=legend_handles, fontsize=TICK_FS, loc="upper left")

fig.tight_layout()

# ── save ────────────────────────────────────────────────
stem = "figure4_llm_impostor_lie_rate"
fig.savefig(str(OUT_DIR / f"{stem}.png"), dpi=300, bbox_inches="tight")
fig.savefig(str(OUT_DIR / f"{stem}.pdf"), bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR / stem}.png")
print(f"Saved: {OUT_DIR / stem}.pdf")
