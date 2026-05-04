#!/usr/bin/env python3
"""Separate plots for the ELO-vs-skill question, one PNG/PDF per figure.

Outputs (in results/):

  fig1_crew_elo_vs_detection.png           — scatter, crew ELO vs detection skill
  fig2_crew_elo_skill_correlations.png     — bar chart, r(crew ELO, every crewmate skill, sign-corrected)
  fig3_imp_elo_vs_deception.png            — scatter, imp ELO vs deceptive_efficacy
  fig4_imp_elo_vs_survival.png             — scatter, imp ELO vs objective_viability  (the strongest correlate)
  fig5_imp_elo_skill_correlations.png      — bar chart, r(imp ELO, every impostor skill, sign-corrected)

Sign convention: every "skill" axis is oriented so HIGHER = better at the
named ability. For crewmate that means
   detection skill   = 1 − detection_accuracy   (raw column is MSE, lower=better)
   belief stability  = 1 − belief_volatility    (less volatile → more stable)
For impostor it means
   alibi opacity     = 1 − alibi_grounding      (less grounded → harder to corroborate)
   belief stability  = 1 − belief_volatility
All other columns are raw (higher = better at the named ability).

Reads: results/2026-05-03_elo_vs_skill_per_model.csv,
       results/2026-05-03_crewmate_x_model_pooled_numeric.csv,
       results/2026-05-03_impostor_x_model_pooled_numeric.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

RES = Path("/home/yjangir1/scratchhbharad2/users/yjangir1/karan/"
           "eval-among-us-crossplay-open-source/results")
DATE = "2026-05-03"

per_model = pd.read_csv(RES / f"{DATE}_elo_vs_skill_per_model.csv")
crew_pooled = pd.read_csv(
    RES / f"{DATE}_crewmate_x_model_pooled_numeric.csv", index_col=0)
imp_pooled = pd.read_csv(
    RES / f"{DATE}_impostor_x_model_pooled_numeric.csv", index_col=0)


def _row(df, metric):
    return df.loc[metric] if metric in df.index else pd.Series(dtype=float)


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
SHORT_NAME = {
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


def pearson(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    return float(np.corrcoef(x, y)[0, 1]) if len(x) >= 2 else float("nan")


def spearman(x, y):
    s = pd.Series(x).rank()
    t = pd.Series(y).rank()
    return float(s.corr(t))


# ============================================================== per-model
crew = per_model[per_model["role"] == "Crewmate"][
    ["model", "log_odds_rating", "win_rate"]
].rename(columns={"log_odds_rating": "crew_elo", "win_rate": "crew_winrate"})
imp = per_model[per_model["role"] == "Impostor"][
    ["model", "log_odds_rating", "win_rate"]
].rename(columns={"log_odds_rating": "imp_elo", "win_rate": "imp_winrate"})
dfm = crew.merge(imp, on="model", how="outer")

# Sign-corrected per-model skills (CREWMATE)
crew_skills = {
    "detection skill\n(1 − MSE)":        ("detection_accuracy",  "invert"),
    "objective_viability":               ("objective_viability", "raw"),
    "alibi_grounding":                   ("alibi_grounding",     "raw"),
    "alibi_corroboration":               ("alibi_corroboration", "raw"),
    "faction_consensus":                 ("faction_consensus",   "raw"),
    "social_influence":                  ("social_influence",    "raw"),
    "belief stability\n(1 − volatility)":("belief_volatility",   "invert"),
    "spatial_dispersion":                ("spatial_dispersion",  "raw"),
}
for label, (col, mode) in crew_skills.items():
    base = _row(crew_pooled, col)
    dfm[f"crew::{label}"] = dfm["model"].map(
        lambda m, c=col, mo=mode: (1.0 - base.get(m, np.nan))
        if mo == "invert" else base.get(m, np.nan))

# Sign-corrected per-model skills (IMPOSTOR)
imp_skills = {
    "deceptive_efficacy":                ("deceptive_efficacy",  "raw"),
    "objective_viability\n(survival)":   ("objective_viability", "raw"),
    "alibi opacity\n(1 − grounding)":    ("alibi_grounding",     "invert"),
    "belief stability\n(1 − volatility)":("belief_volatility",   "invert"),
    "alibi_corroboration":               ("alibi_corroboration", "raw"),
    "faction_consensus":                 ("faction_consensus",   "raw"),
    "social_influence":                  ("social_influence",    "raw"),
    "spatial_dispersion":                ("spatial_dispersion",  "raw"),
}
for label, (col, mode) in imp_skills.items():
    base = _row(imp_pooled, col)
    dfm[f"imp::{label}"] = dfm["model"].map(
        lambda m, c=col, mo=mode: (1.0 - base.get(m, np.nan))
        if mo == "invert" else base.get(m, np.nan))

dfm["family"] = dfm["model"].map(FAMILY_OF)
dfm["short"]  = dfm["model"].map(SHORT_NAME)


# ============================================================== style
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
})


def family_legend(ax, loc="upper left"):
    handles = [Line2D([0], [0], marker="o", color="white",
                      markerfacecolor=col, markeredgecolor="black",
                      markersize=8, label=fam)
               for fam, col in FAMILY_COLOR.items()]
    ax.legend(handles=handles, loc=loc, fontsize=8.5,
              frameon=True, framealpha=0.9, title="family",
              title_fontsize=8.5, handletextpad=0.4,
              borderpad=0.5, labelspacing=0.3)


def _iterative_label_offsets(points_df, xcol, ycol, ax, max_iter=80):
    """Simple repulsion: place each label at default (6, 6) offset, then
    iteratively push labels that visually overlap. Returns dict
    {model_short -> (dx, dy)} in display offset points.
    """
    # Start every label at (6, 6) — top-right of point.
    offsets = {row["short"]: [6.0, 6.0] for _, row in points_df.iterrows()}
    # Convert each point's data coordinate to display pixels.
    inv = ax.transData
    pts = {row["short"]: inv.transform((row[xcol], row[ycol]))
           for _, row in points_df.iterrows()}

    def label_box(name):
        cx, cy = pts[name]
        dx, dy = offsets[name]
        # Approx label half-extents (display px). Treat label as ~70x14 px box.
        lx = cx + dx * 1.3
        ly = cy + dy * 1.3
        return (lx - 38, lx + 38, ly - 8, ly + 8)

    def boxes_overlap(b1, b2):
        return not (b1[1] < b2[0] or b2[1] < b1[0]
                    or b1[3] < b2[2] or b2[3] < b1[2])

    names = list(offsets.keys())
    for _ in range(max_iter):
        moved = False
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                bi = label_box(names[i]); bj = label_box(names[j])
                if not boxes_overlap(bi, bj):
                    continue
                # Push them apart vertically by 6 px each in opposite signs.
                ci_y = (bi[2] + bi[3]) / 2
                cj_y = (bj[2] + bj[3]) / 2
                if ci_y >= cj_y:
                    offsets[names[i]][1] += 6
                    offsets[names[j]][1] -= 6
                else:
                    offsets[names[i]][1] -= 6
                    offsets[names[j]][1] += 6
                # Also nudge horizontally a bit to break ties.
                offsets[names[i]][0] += 1
                offsets[names[j]][0] -= 1
                moved = True
        if not moved:
            break
    return offsets


def make_scatter(xcol, ycol, xlabel, ylabel, title, fname,
                 fitline_color="0.4", label_overrides=None):
    sub = dfm.dropna(subset=[xcol, ycol])
    x = sub[xcol].values
    y = sub[ycol].values
    r_p = pearson(x, y)
    r_s = spearman(x, y)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    b1, b0 = np.polyfit(x, y, 1)
    xr = np.linspace(x.min() - 0.15, x.max() + 0.15, 50)
    ax.plot(xr, b0 + b1 * xr, color=fitline_color, lw=1.3, ls="--",
            zorder=1, label=f"OLS  (slope = {b1:+.3f})")
    # Plot points first so transData is available for label placement.
    for _, row in sub.iterrows():
        c = FAMILY_COLOR.get(row["family"], "0.4")
        ax.scatter(row[xcol], row[ycol], s=110, c=c,
                   edgecolor="black", linewidth=0.7, zorder=3)

    # Pre-set axis limits before placing labels (so display→data is stable)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.grid(True, alpha=0.25); ax.set_axisbelow(True)
    # Hard-set axis ranges so even outlier points are visible AND so the
    # data→display transform used for label placement is stable.
    x_pad = max(0.05, 0.10 * (x.max() - x.min()))
    y_pad = max(0.005, 0.10 * (y.max() - y.min()))
    ax.set_xlim(x.min() - x_pad, x.max() + x_pad)
    ax.set_ylim(y.min() - y_pad, y.max() + y_pad)

    # Label placement: per-model overrides, then iterative repulsion for the rest.
    overrides = label_overrides or {}
    auto_offsets = _iterative_label_offsets(sub, xcol, ycol, ax)
    for _, row in sub.iterrows():
        nm = row["short"]
        if nm in overrides:
            dx, dy = overrides[nm]
        else:
            dx, dy = auto_offsets[nm]
        # Draw a thin leader if the label was pushed far from the point.
        far = (abs(dx) > 12 or abs(dy) > 12)
        ax.annotate(nm, (row[xcol], row[ycol]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=9, color="0.2", zorder=4,
                    arrowprops=dict(arrowstyle="-",
                                    color="0.55", lw=0.6,
                                    shrinkA=2, shrinkB=4) if far else None)

    ax.text(0.97, 0.05,
            f"Pearson r  = {r_p:+.3f}\n"
            f"Spearman r = {r_s:+.3f}\n"
            f"n = {len(sub)} models",
            transform=ax.transAxes, fontsize=10, ha="right", va="bottom",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="0.55", alpha=0.95))
    family_legend(ax, loc="upper left")
    fig.tight_layout()
    fig.savefig(RES / f"{fname}.png", dpi=300, bbox_inches="tight")
    fig.savefig(RES / f"{fname}.pdf",            bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}.png   r = {r_p:+.3f}")
    return r_p, r_s


def make_correlation_bar(elo_col, skills_dict, role_prefix,
                         named_skill_label, title, fname):
    """Horizontal bar chart of r(ELO, every sign-corrected skill).

    Color logic:
      - If the named skill IS the strongest correlate     → purple bar,
        legend says "named skill = strongest correlate".
      - Otherwise the named skill is red, the strongest is blue,
        legend shows both entries.
    Other bars are grey.
    """
    rows = []
    for label in skills_dict:
        col = f"{role_prefix}::{label}"
        rows.append({"label": label,
                     "r": pearson(dfm[elo_col], dfm[col])})
    df = pd.DataFrame(rows)
    df["abs_r"] = df["r"].abs()
    df = df.sort_values("abs_r", ascending=False).reset_index(drop=True)

    top_label = df.iloc[0]["label"]
    same = (top_label == named_skill_label)
    colors = []
    for lbl in df["label"]:
        if same and lbl == named_skill_label:
            colors.append("#7c1f7c")
        elif lbl == top_label:
            colors.append("#1f77b4")
        elif lbl == named_skill_label:
            colors.append("#d62728")
        else:
            colors.append("0.65")

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ypos = np.arange(len(df))
    ax.barh(ypos, df["r"], color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(df["label"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(-1, 1)
    ax.set_xlabel(f"Pearson r  with {role_prefix} ELO  (logit of {role_prefix} win rate)")
    ax.set_title(title, loc="left")
    for i, r in enumerate(df["r"]):
        xtxt = r + (0.03 if r >= 0 else -0.03)
        ha = "left" if r >= 0 else "right"
        ax.text(xtxt, i, f"{r:+.2f}", va="center", ha=ha,
                fontsize=10, color="0.15")
    ax.grid(True, axis="x", alpha=0.25); ax.set_axisbelow(True)

    # Legend reflects only the colors actually drawn.
    if same:
        legend_handles = [Line2D([0], [0], marker="s", color="white",
                                 markerfacecolor="#7c1f7c",
                                 markeredgecolor="black", markersize=11,
                                 label="named skill = strongest correlate")]
    else:
        legend_handles = [
            Line2D([0], [0], marker="s", color="white",
                   markerfacecolor="#1f77b4", markeredgecolor="black",
                   markersize=11, label="strongest correlate"),
            Line2D([0], [0], marker="s", color="white",
                   markerfacecolor="#d62728", markeredgecolor="black",
                   markersize=11, label="named skill"),
        ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9,
              frameon=True, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(RES / f"{fname}.png", dpi=300, bbox_inches="tight")
    fig.savefig(RES / f"{fname}.pdf",            bbox_inches="tight")
    plt.close(fig)
    df.to_csv(RES / f"{fname}_table.csv", index=False)
    print(f"  saved {fname}.png   top correlate = {top_label!r}  "
          f"r = {df.iloc[0]['r']:+.3f}")
    return df


# ============================================================== run
print("=" * 70)
print("CREWMATE side")
print("=" * 70)

# Per-model label-offset overrides for the points that hand-tuning
# resolves more cleanly than the auto-repulsion.
# Each entry below is the (dx, dy) offset (display pixels) of the model's
# label relative to its data point. Same offset for two different models
# means same *direction* off the point — not same screen position, since
# the points themselves are at different (x, y).
IMP_DECEPTION_OVR = {
    # isolated points on the left half — default upper-right placement
    "ds-llama-8B":  (  8,   8),     # data point at (-2.15, -0.063)
    "llama-3B":     (  8,   8),     # data point at (-1.59, -0.054)
    # bottom cluster: 3 points within ~0.25 of each other
    "qwen3-32B":    (  8, -16),     # (-1.41, -0.107) — below-right
    "gemma-31B":    (  8,   8),     # (-1.24, -0.103) — above-right
    "gemma-26B":    (-78,  -3),     # (-1.47, -0.118) — left
    # mid:
    "qwen3-4B":     (  8, -14),     # (-0.77, -0.059) — below
    "qwen3-8B":     (  8,   8),     # (-0.03, -0.054) — above
    # top cluster (at y ≈ -0.02 to -0.03)
    "llama-8B":     (  8,   8),     # (-0.08, -0.025) — above
    "gemma-E4B":    (  8, -14),     # (-0.12, -0.032) — below
    "llama-70B":    (  8, -14),     # (+0.77, -0.021) — below (ample x room)
}
IMP_SURVIVAL_OVR = {
    # bottom-left outlier
    "ds-llama-8B":  (  8,   8),     # (-2.15, 0.244)
    # mid-left
    "llama-3B":     ( -68,  -3),    # (-1.59, 0.489) — left to avoid gemma-26B
    # bottom cluster: gemma-26B (-1.47, 0.490), qwen3-32B (-1.41, 0.561), gemma-31B (-1.24, 0.522)
    "gemma-26B":    (  8, -14),
    "qwen3-32B":    (  8,   8),
    "gemma-31B":    (  8,  -2),
    # mid:
    "qwen3-4B":     (  8,   6),     # (-0.77, 0.607)
    # top cluster: qwen3-8B (-0.03, 0.708), llama-8B (-0.08, 0.651), gemma-E4B (-0.12, 0.617)
    "qwen3-8B":     (  8,   8),
    "llama-8B":     (  8,   8),
    "gemma-E4B":    (  8, -14),
    "llama-70B":    ( -72, -3),     # (+0.77, 0.635) — left
}
IMP_SURVIVAL_OVR = {
    # x ≈ -2.15, y ≈ 0.244 — corner, push above-right
    "ds-llama-8B":  ( 9,   8),
    # x ≈ -1.59, y ≈ 0.489
    "llama-3B":     ( 9,   8),
    # nearby cluster:  gemma-26B (-1.47, 0.490), qwen3-32B (-1.41, 0.561)
    "gemma-26B":    ( 9, -14),       # below-right
    "qwen3-32B":    ( 9,   8),       # above
    "gemma-31B":    ( 9,  -2),       # right (-1.24, 0.522)
    # mid:
    "qwen3-4B":     ( 9,   6),       # (-0.77, 0.607)
    # top cluster:  qwen3-8B (-0.03, 0.708), llama-8B (-0.08, 0.651), gemma-E4B (-0.12, 0.617)
    "qwen3-8B":     ( 9,   8),
    "llama-8B":     ( 9,   8),       # above-right
    "gemma-E4B":    ( 9, -14),       # below-right
    "llama-70B":    (-72, -3),       # left  (0.77, 0.635)
}

# Fig 1 — Crew ELO vs detection skill (the "named" comparison)
make_scatter(
    xcol="crew_elo", ycol="crew::detection skill\n(1 − MSE)",
    xlabel="Crewmate ELO  (logit of crew win rate)",
    ylabel="Detection skill  (1 − MSE)",
    title="Crewmate ELO vs detection skill",
    fname="fig1_crew_elo_vs_detection",
)

# Fig 2 — what is crew ELO most correlated with?
crew_corr = make_correlation_bar(
    elo_col="crew_elo",
    skills_dict=crew_skills,
    role_prefix="crew",
    named_skill_label="detection skill\n(1 − MSE)",
    title="Crewmate ELO — correlation with each skill",
    fname="fig2_crew_elo_skill_correlations",
)

print()
print("=" * 70)
print("IMPOSTOR side")
print("=" * 70)

# Fig 3 — Imp ELO vs deceptive_efficacy
make_scatter(
    xcol="imp_elo", ycol="imp::deceptive_efficacy",
    xlabel="Impostor ELO  (logit of imp win rate)",
    ylabel="deceptive_efficacy  (named skill)",
    title="Impostor ELO vs deception skill",
    fname="fig3_imp_elo_vs_deception",
    label_overrides=IMP_DECEPTION_OVR,
)

# Fig 4 — Imp ELO vs objective_viability (strongest correlate)
make_scatter(
    xcol="imp_elo", ycol="imp::objective_viability\n(survival)",
    xlabel="Impostor ELO  (logit of imp win rate)",
    ylabel="objective_viability  (survival)",
    title="Impostor ELO vs survival",
    fname="fig4_imp_elo_vs_survival",
    fitline_color="#1f77b4",
    label_overrides=IMP_SURVIVAL_OVR,
)

# Fig 5 — what is imp ELO most correlated with?
imp_corr = make_correlation_bar(
    elo_col="imp_elo",
    skills_dict=imp_skills,
    role_prefix="imp",
    named_skill_label="deceptive_efficacy",
    title="Impostor ELO — correlation with each skill",
    fname="fig5_imp_elo_skill_correlations",
)

print("\nDone.")
