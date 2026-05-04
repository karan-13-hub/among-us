#!/usr/bin/env python3
"""NeurIPS-paper figures from the cross-play sweep.

Produces (under results/):

  fig8_within_vs_across_family.png  — strip+box of crew/imp win rate by family relation
  fig9_size_scaling.png             — per-role skill metric vs model size, family-grouped
  fig10_win_categories_per_matchup.png — stacked bar of Ejection / Tasks / Outnumber / Timeout per matchup
  fig11_crossplay_vs_selfplay_delta.png — forest plot of (CP - SP) per (model, role) for the named skill
  fig12_sample_sizes.png            — sample-size table as a heatmap (matchup x config)

Reads the CSVs already produced by run_full_analysis_crossplay.py.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

RES = Path("/home/yjangir1/scratchhbharad2/users/yjangir1/karan/"
           "eval-among-us-crossplay-open-source/results")
DATE = "2026-05-03"

FAMILY_OF = {
    "gemma-4-E4B-it":               "gemma",
    "gemma-4-26B-A4B-it":           "gemma",
    "gemma-4-31b":                  "gemma",
    "qwen3-4b":                     "qwen3",
    "qwen3-8B":                     "qwen3",
    "qwen3-32b":                    "qwen3",
    "llama-3.2-3b-instruct":        "llama",
    "llama-3.1-8b":                 "llama",
    "llama-3.3-70b":                "llama",
    "deepseek-r1-distill-llama-8B": "deepseek-distill",
}
FAMILY_COLOR = {
    "gemma":            "#1f77b4",
    "qwen3":            "#2ca02c",
    "llama":            "#d62728",
    "deepseek-distill": "#9467bd",
}
SIZE_OF = {
    "llama-3.2-3b-instruct":         3,
    "qwen3-4b":                      4,
    "gemma-4-E4B-it":                4,
    "deepseek-r1-distill-llama-8B":  8,
    "llama-3.1-8b":                  8,
    "qwen3-8B":                      8,
    "gemma-4-26B-A4B-it":            26,
    "gemma-4-31b":                   31,
    "qwen3-32b":                     32,
    "llama-3.3-70b":                 70,
}
SHORT = {
    "gemma-4-E4B-it":               "gemma-E4B",
    "gemma-4-26B-A4B-it":           "gemma-26B",
    "gemma-4-31b":                  "gemma-31B",
    "qwen3-4b":                     "qwen3-4B",
    "qwen3-8B":                     "qwen3-8B",
    "qwen3-32b":                    "qwen3-32B",
    "llama-3.2-3b-instruct":        "llama-3B",
    "llama-3.1-8b":                 "llama-8B",
    "llama-3.3-70b":                "llama-70B",
    "deepseek-r1-distill-llama-8B": "ds-llama-8B",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.labelsize": 10.5,
    "axes.titlesize": 11.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})


def save(fig, name):
    fig.savefig(RES / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(RES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.png  /  .pdf")


# =====================================================================
# fig8 — within vs across family
# =====================================================================
print("[fig8] within-vs-across family")
pooled = pd.read_csv(RES / f"{DATE}_winrate_matchup_pooled.csv")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, target, color, lbl in [
    (axes[0], "crew_win_rate", "#1f77b4", "Crewmate win rate"),
    (axes[1], "imp_win_rate",  "#d62728", "Impostor win rate"),
]:
    groups = ["within_family", "across_family"]
    data = [pooled.loc[pooled["family_relation"] == g, target].values
            for g in groups]
    bp = ax.boxplot(data, positions=[0, 1], widths=0.55,
                    showfliers=False, patch_artist=True,
                    boxprops=dict(facecolor=color, alpha=0.25,
                                  edgecolor=color),
                    medianprops=dict(color=color, lw=2),
                    whiskerprops=dict(color=color),
                    capprops=dict(color=color))
    for i, g in enumerate(groups):
        vals = pooled.loc[pooled["family_relation"] == g, target].values
        n = len(vals)
        jitter = (np.random.RandomState(0).uniform(-0.12, 0.12, n))
        ax.scatter(np.full(n, i) + jitter, vals, s=55,
                   c=color, edgecolor="black", linewidth=0.5,
                   alpha=0.85, zorder=3)
        # Mean line
        ax.hlines(np.mean(vals), i - 0.30, i + 0.30,
                  color="black", lw=1.6, linestyles="-",
                  zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["within family", "across family"])
    ax.set_ylabel(lbl)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="0.6", lw=0.8, ls="--", zorder=0)
    ax.grid(True, axis="y", alpha=0.25); ax.set_axisbelow(True)
    # Sample-size annotation
    for i, g in enumerate(groups):
        n_pairs = (pooled["family_relation"] == g).sum()
        ax.text(i, 0.04, f"n = {n_pairs} matchup pairs",
                ha="center", fontsize=8.5, color="0.3")

axes[0].set_title("(a) Crewmate side", loc="left")
axes[1].set_title("(b) Impostor side", loc="left")
fig.suptitle("Win rates: within-family vs across-family matchups",
             fontsize=12, y=1.02)
fig.tight_layout()
save(fig, "fig8_within_vs_across_family")


# =====================================================================
# fig9 — size scaling: detection skill (crew) and deceptive_efficacy (imp)
# =====================================================================
print("[fig9] size scaling")
crew_pooled = pd.read_csv(
    RES / f"{DATE}_crewmate_x_model_pooled_numeric.csv", index_col=0)
imp_pooled = pd.read_csv(
    RES / f"{DATE}_impostor_x_model_pooled_numeric.csv", index_col=0)

# crew detection skill = 1 - MSE
crew_skill = (1.0 - crew_pooled.loc["detection_accuracy"]).to_dict()
imp_skill = imp_pooled.loc["deceptive_efficacy"].to_dict()

# Per-model offset overrides for points that cluster at the same x.
SIZE_LABEL_OVR = {
    # gemma-26B at size=26 and gemma-31B at size=31 are visually
    # adjacent on log axis and their labels collide.
    "gemma-26B":   (-58, -3),     # left
    "gemma-31B":   (  6,  6),     # upper-right (default-ish)
    # qwen3-8B / llama-8B / ds-llama-8B all at size=8
    "qwen3-8B":    (  6,   6),
    "llama-8B":    ( -52,  6),    # left-up
    "ds-llama-8B": (  6,  -14),   # below
    # 4B cluster (qwen3-4B, gemma-E4B both at ~4)
    "qwen3-4B":    ( -50, -3),    # left
    "gemma-E4B":   (  6,   6),    # upper-right
    # 70B / 3B / 32B alone — default
}

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
for ax, skill_dict, label, ylim in [
    (axes[0], crew_skill, "Detection skill  (1 − MSE)", (0.74, 0.87)),
    (axes[1], imp_skill,  "deceptive_efficacy",          (-0.135, 0.00)),
]:
    rows = []
    for m, v in skill_dict.items():
        rows.append({"model": m, "size": SIZE_OF.get(m, np.nan),
                     "family": FAMILY_OF.get(m, "?"),
                     "y": v, "short": SHORT.get(m, m)})
    df = pd.DataFrame(rows).dropna()
    for fam, sub in df.groupby("family"):
        sub = sub.sort_values("size")
        ax.plot(sub["size"], sub["y"], "-",
                color=FAMILY_COLOR.get(fam, "0.4"),
                lw=1.2, alpha=0.55, zorder=1)
        ax.scatter(sub["size"], sub["y"],
                   c=FAMILY_COLOR.get(fam, "0.4"),
                   s=95, edgecolor="black", linewidth=0.6, zorder=3)
        for _, row in sub.iterrows():
            dx, dy = SIZE_LABEL_OVR.get(row["short"], (6, 6))
            ax.annotate(row["short"], (row["size"], row["y"]),
                        xytext=(dx, dy), textcoords="offset points",
                        fontsize=8.5, color="0.2", zorder=4)
    ax.set_xscale("log")
    ax.set_xticks([3, 4, 8, 16, 32, 70])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Model size  (B params, log scale)")
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.25); ax.set_axisbelow(True)
    ax.set_ylim(*ylim)

handles = [Patch(facecolor=col, edgecolor="black", label=fam)
           for fam, col in FAMILY_COLOR.items()]
axes[0].set_title("(a) Crewmate role  —  detection skill", loc="left")
axes[1].set_title("(b) Impostor role  —  deceptive efficacy", loc="left")
fig.suptitle("Skill metrics scale with model size  "
             f"(per-role pooled means, n = {len(df)} models)",
             fontsize=12, y=0.99)
# Family legend at the bottom
fig.legend(handles=handles, loc="lower center",
           bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=False,
           title="family", title_fontsize=9, fontsize=9)
fig.tight_layout(rect=[0, 0.05, 1, 0.95])
save(fig, "fig9_size_scaling")


# =====================================================================
# fig10 — win-category stacked bars per matchup
# =====================================================================
print("[fig10] win-category breakdown")
cats = pd.read_csv(RES / f"{DATE}_win_categories_matchup.csv")
# Sort by crewmate-wins share (Ejection + Tasks) descending
cats["crew_share"] = cats["Ejection"] + cats["Tasks"]
cats = cats.sort_values("crew_share", ascending=True).reset_index(drop=True)


def short_pair(matchup):
    # matchup = "crew=A__imp=B"
    parts = matchup.replace("crew=", "").split("__imp=")
    a, b = parts[0], parts[1] if len(parts) == 2 else ""
    return f"{SHORT.get(a, a)} → {SHORT.get(b, b)}"


cats["label"] = cats["matchup"].map(short_pair)

fig, ax = plt.subplots(figsize=(11.5, 7))
y = np.arange(len(cats))
left = np.zeros(len(cats))
CAT_COLORS = {
    "Ejection": "#2ca02c",   # crew win, detection-driven
    "Tasks":    "#9bd49b",   # crew win, task-driven
    "Outnumber":"#ff7f0e",   # imp win, deception-driven
    "Timeout":  "#d62728",   # imp win, timeout
}
for cat in ["Ejection", "Tasks", "Outnumber", "Timeout"]:
    ax.barh(y, cats[cat], left=left, color=CAT_COLORS[cat],
            edgecolor="white", linewidth=0.5, label=cat,
            height=0.78)
    left += cats[cat]
ax.axvline(0.5, color="0.3", lw=1.0, ls="--", zorder=4)
ax.set_yticks(y)
ax.set_yticklabels(cats["label"], fontsize=9)
ax.set_xlim(0, 1)
ax.set_xlabel("Share of games  (Ejection + Tasks = Crewmate wins,  Outnumber + Timeout = Impostor wins)")
ax.set_title("Win-category breakdown per matchup",
             loc="left")
ax.legend(loc="lower right", title="Win category", title_fontsize=9,
          frameon=True, framealpha=0.95)
ax.grid(True, axis="x", alpha=0.25); ax.set_axisbelow(True)
fig.tight_layout()
save(fig, "fig10_win_categories_per_matchup")


# =====================================================================
# fig11 — cross-play vs self-play delta (per model, per role, named skill)
# =====================================================================
print("[fig11] cross-play vs self-play delta")
delta = pd.read_csv(RES / f"{DATE}_crossplay_vs_selfplay_metrics.csv")
# Use detection skill (1 - MSE) for crewmate, deceptive_efficacy for imp.
# The CSV already has *_selfplay, *_crossplay, *_delta columns for each metric.

rows = []
for _, r in delta.iterrows():
    role = r["role"]
    if role == "Crewmate":
        sp = 1.0 - r.get("detection_accuracy_selfplay", np.nan)
        cp = 1.0 - r.get("detection_accuracy_crossplay", np.nan)
    else:
        sp = r.get("deceptive_efficacy_selfplay", np.nan)
        cp = r.get("deceptive_efficacy_crossplay", np.nan)
    rows.append({"model": r["model"], "role": role,
                 "selfplay": sp, "crossplay": cp, "delta": cp - sp,
                 "n_cp": r["n_crossplay"], "n_sp": r["n_selfplay"]})
df = pd.DataFrame(rows).dropna(subset=["delta"])

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), sharex=False)
for ax, role, color, label in [
    (axes[0], "Crewmate", "#1f77b4",
     "Δ detection skill   (cross-play  −  self-play)"),
    (axes[1], "Impostor", "#d62728",
     "Δ deceptive_efficacy   (cross-play  −  self-play)"),
]:
    sub = df[df["role"] == role].copy()
    sub = sub.sort_values("delta")
    sub["short"] = sub["model"].map(SHORT.get)
    sub["family"] = sub["model"].map(FAMILY_OF.get)
    y = np.arange(len(sub))
    for i, (_, row) in enumerate(sub.iterrows()):
        c = FAMILY_COLOR.get(row["family"], "0.4")
        ax.scatter(row["delta"], i, s=130, c=c,
                   edgecolor="black", linewidth=0.6, zorder=3)
        ax.plot([0, row["delta"]],
                [i, i], color=c, lw=1.6, alpha=0.5, zorder=2)
    ax.axvline(0, color="0.3", lw=1.0, ls="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["short"], fontsize=10)
    ax.set_xlabel(label, fontsize=11.5)
    ax.set_title(f"({'a' if role == 'Crewmate' else 'b'}) {role}",
                 loc="left", fontsize=12)
    ax.grid(True, axis="x", alpha=0.25); ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=10)

fig.suptitle(
    "Behavior shift when the opponent is a different model\n"
    "(positive Δ = model performs better in cross-play than self-play)",
    fontsize=13, y=1.02)
fig.tight_layout()
save(fig, "fig11_crossplay_vs_selfplay_delta")


# =====================================================================
# fig12 — sample-size heatmap (matchup × config)
# =====================================================================
print("[fig12] sample-size heatmap")
long = pd.read_csv(RES / f"{DATE}_winrate_matchup_x_config.csv")
piv = (long
       .pivot_table(index="matchup", columns="config",
                    values="n_games", aggfunc="sum", fill_value=0)
       .reindex(columns=["4C_1I", "4C_2I", "5C_1I", "5C_2I"]))
piv.index = [short_pair(m) for m in piv.index]

fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(piv) + 1.5))
im = ax.imshow(piv.values, cmap="Greens", vmin=0, vmax=30, aspect="auto")
ax.set_xticks(range(piv.shape[1]))
ax.set_xticklabels(piv.columns, fontsize=10)
ax.set_yticks(range(piv.shape[0]))
ax.set_yticklabels(piv.index, fontsize=9)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.values[i, j]
        ax.text(j, i, f"{v}", ha="center", va="center",
                color="black" if v < 18 else "white", fontsize=9)
ax.set_title("Sample size  (games per matchup × config)", loc="left")
fig.colorbar(im, ax=ax, label="n games", fraction=0.04, pad=0.02)
fig.tight_layout()
save(fig, "fig12_sample_sizes")

print("\nDone.")
