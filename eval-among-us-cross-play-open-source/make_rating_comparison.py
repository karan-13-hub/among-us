#!/usr/bin/env python3
"""Compare three rating systems for the cross-play sweep:

  1. Win-rate ELO  (logit of role-win rate)        — what we've used so far
  2. Bradley-Terry MLE  (per role, fit on per-game outcomes)
  3. TrueSkill          (per role, online updates)

For each rating system we report Pearson r against the named skill and
against the strongest correlate.  If all three give similar correlations,
the case "ELO is misleading on the impostor side" generalizes to the
broader family of pairwise rating models.

Outputs:
  results/2026-05-03_rating_comparison_table.csv
  results/2026-05-03_rating_comparison_per_model.csv
  fig13_rating_comparison.png / .pdf
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trueskill

RES = Path("/home/yjangir1/scratchhbharad2/users/yjangir1/karan/"
           "eval-among-us-crossplay-open-source/results")
DATE = "2026-05-03"


def pearson(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    return float(np.corrcoef(x, y)[0, 1]) if len(x) >= 2 else float("nan")


# ===============================================================
# Load data
# ===============================================================
games = pd.read_csv(RES / f"{DATE}_game_outcomes_long.csv")
games = games.dropna(subset=["crew_model", "imp_model"])
crew_pool = pd.read_csv(
    RES / f"{DATE}_crewmate_x_model_pooled_numeric.csv", index_col=0)
imp_pool = pd.read_csv(
    RES / f"{DATE}_impostor_x_model_pooled_numeric.csv", index_col=0)


# ===============================================================
# 1. Win-rate ELO (logit of win rate per role)
# ===============================================================
def winrate_elo(games):
    rows = []
    for actor_col, target, role in [
        ("crew_model", "Crewmate", "Crewmate"),
        ("imp_model",  "Impostor", "Impostor"),
    ]:
        for m, g in games.groupby(actor_col):
            wr = (g["winner_faction"] == target).mean()
            wr_clip = min(max(wr, 1e-3), 1 - 1e-3)
            rows.append({"model": m, "role": role,
                         "n_games": len(g),
                         "win_rate": wr,
                         "rating": math.log(wr_clip / (1 - wr_clip))})
    return pd.DataFrame(rows)


# ===============================================================
# 2. Bradley-Terry MLE (per role)
# ===============================================================
def bradley_terry(games, role):
    """Fit BT log-strengths theta_i such that P(i beats j) = σ(θ_i − θ_j).

    For "crewmate role" we treat each game as: crew_model wins iff
    winner_faction == "Crewmate".  The "match" is between crew_model
    (in role) and imp_model (in role).  We fit theta_i for each model
    in this role only.

    Iterative MLE update (Hunter, 2004).
    """
    if role == "Crewmate":
        # Player i = crew_model, opponent j = imp_model, win iff crew wins.
        actor_col = "crew_model"; opp_col = "imp_model"
        won = games["winner_faction"] == "Crewmate"
    else:
        actor_col = "imp_model"; opp_col = "crew_model"
        won = games["winner_faction"] == "Impostor"

    g = games.copy()
    g["actor_won"] = won.astype(int)
    models = sorted(set(g[actor_col].unique()) | set(g[opp_col].unique()))
    idx = {m: i for i, m in enumerate(models)}
    n = len(models)

    # Build win-count matrix W[i, j] = times i beat j (in actor role).
    W = np.zeros((n, n))
    N = np.zeros((n, n))
    for _, row in g.iterrows():
        i = idx[row[actor_col]]; j = idx[row[opp_col]]
        if pd.isna(i) or pd.isna(j) or i == j:
            continue
        N[i, j] += 1
        if row["actor_won"]:
            W[i, j] += 1

    # Iterative MLE for the BT strengths p_i (un-normalized).
    p = np.ones(n)
    for it in range(500):
        new_p = np.zeros(n)
        for i in range(n):
            num = W[i].sum()
            den = 0.0
            for j in range(n):
                if i == j: continue
                # BT pairwise contribution
                den += (N[i, j] + N[j, i]) / (p[i] + p[j])
            new_p[i] = (num / den) if den > 0 else p[i]
        # Normalize
        if new_p.sum() > 0:
            new_p = new_p / new_p.sum() * n
        if np.allclose(new_p, p, atol=1e-9):
            break
        p = new_p
    theta = np.log(np.clip(p, 1e-9, None))
    return pd.DataFrame({"model": models, "role": role, "rating": theta})


# ===============================================================
# 3. TrueSkill (per role, online from game stream)
# ===============================================================
def trueskill_ratings(games):
    env = trueskill.TrueSkill(draw_probability=0)
    crew_r = {}
    imp_r = {}
    rows = []
    # Iterate games in chronological order (use "experiment" + "game_index"
    # as a coarse ordering; doesn't matter for correlation).
    g_sorted = games.sort_values(["experiment", "game_index"])
    for _, row in g_sorted.iterrows():
        cm = row["crew_model"]; im = row["imp_model"]
        if pd.isna(cm) or pd.isna(im):
            continue
        if cm not in crew_r: crew_r[cm] = env.create_rating()
        if im not in imp_r:  imp_r[im]  = env.create_rating()
        r_crew = crew_r[cm]; r_imp = imp_r[im]
        crew_won = (row["winner_faction"] == "Crewmate")
        # Treat each "match" as 2-team 1v1 (one rating per side).
        if crew_won:
            new_crew, new_imp = env.rate_1vs1(r_crew, r_imp)
        else:
            new_imp, new_crew = env.rate_1vs1(r_imp, r_crew)
        crew_r[cm] = new_crew
        imp_r[im] = new_imp
    for m, r in crew_r.items():
        rows.append({"model": m, "role": "Crewmate",
                     "rating": r.mu - 3 * r.sigma,
                     "mu": r.mu, "sigma": r.sigma})
    for m, r in imp_r.items():
        rows.append({"model": m, "role": "Impostor",
                     "rating": r.mu - 3 * r.sigma,
                     "mu": r.mu, "sigma": r.sigma})
    return pd.DataFrame(rows)


# ===============================================================
# Run all three
# ===============================================================
print("Fitting win-rate ELO ...")
elo = winrate_elo(games);             elo["system"] = "Win-rate ELO"
print("Fitting Bradley-Terry (per role) ...")
bt_crew = bradley_terry(games, "Crewmate")
bt_imp  = bradley_terry(games, "Impostor")
bt = pd.concat([bt_crew, bt_imp], ignore_index=True); bt["system"] = "Bradley-Terry"
print("Fitting TrueSkill (per role) ...")
ts = trueskill_ratings(games);        ts["system"] = "TrueSkill"

all_ratings = pd.concat([elo, bt, ts], ignore_index=True)
all_ratings.to_csv(RES / f"{DATE}_rating_comparison_per_model.csv", index=False)


# ===============================================================
# Correlate with skill metrics
# ===============================================================
def detection_skill(m): return 1.0 - crew_pool.loc["detection_accuracy"].get(m, np.nan)
def deceptive_eff(m):    return imp_pool.loc["deceptive_efficacy"].get(m, np.nan)
def obj_viability(m):    return imp_pool.loc["objective_viability"].get(m, np.nan)
def alibi_opacity(m):    return 1.0 - imp_pool.loc["alibi_grounding"].get(m, np.nan)

rows = []
for system in ["Win-rate ELO", "Bradley-Terry", "TrueSkill"]:
    sub = all_ratings[all_ratings["system"] == system]
    crew = sub[sub["role"] == "Crewmate"].copy()
    imp  = sub[sub["role"] == "Impostor"].copy()
    crew["det_skill"] = crew["model"].map(detection_skill)
    imp["dec_eff"]    = imp["model"].map(deceptive_eff)
    imp["obj_via"]    = imp["model"].map(obj_viability)
    imp["alibi_op"]   = imp["model"].map(alibi_opacity)

    rows.append({
        "system": system,
        "crew_n_models":     len(crew),
        "r(crew_rating, detection_skill)":
            pearson(crew["rating"], crew["det_skill"]),
        "imp_n_models":      len(imp),
        "r(imp_rating, deceptive_efficacy)":
            pearson(imp["rating"], imp["dec_eff"]),
        "r(imp_rating, objective_viability)":
            pearson(imp["rating"], imp["obj_via"]),
        "r(imp_rating, alibi_opacity)":
            pearson(imp["rating"], imp["alibi_op"]),
    })

table = pd.DataFrame(rows)
print("\n=== rating-vs-skill correlations (n = 10 models) ===")
print(table.to_string(index=False))
table.to_csv(RES / f"{DATE}_rating_comparison_table.csv", index=False)


# ===============================================================
# Figure: grouped bar of the four correlations across rating systems
# ===============================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.labelsize": 10.5,
    "axes.titlesize": 11.5,
})
fig, ax = plt.subplots(figsize=(11, 4.8))

systems = list(table["system"])
metric_keys = [
    ("r(crew_rating, detection_skill)",   "crew rating ↔ detection skill",  "#1f77b4"),
    ("r(imp_rating, deceptive_efficacy)", "imp rating ↔ deceptive_efficacy", "#d62728"),
    ("r(imp_rating, objective_viability)","imp rating ↔ objective_viability\n(survival)", "#ff7f0e"),
    ("r(imp_rating, alibi_opacity)",      "imp rating ↔ alibi opacity\n(1 − grounding)", "#7c1f7c"),
]

x = np.arange(len(systems))
width = 0.20
for i, (k, lbl, color) in enumerate(metric_keys):
    vals = table[k].values
    bars = ax.bar(x + (i - 1.5) * width, vals, width=width,
                  color=color, edgecolor="black", linewidth=0.5,
                  label=lbl)
    for j, v in enumerate(vals):
        ax.text(x[j] + (i - 1.5) * width, v + (0.02 if v >= 0 else -0.06),
                f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=8.5)

ax.axhline(0, color="0.3", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(systems)
ax.set_ylim(-1, 1.05)
ax.set_ylabel(f"Pearson r  (n = {table['crew_n_models'].iloc[0]} models)")
ax.set_title("Rating systems compared — none recover deception",
             loc="left")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30),
          ncol=4, frameon=False, fontsize=9)
ax.grid(True, axis="y", alpha=0.25); ax.set_axisbelow(True)
fig.tight_layout(rect=[0, 0.08, 1, 1])
fig.savefig(RES / "fig13_rating_comparison.png", dpi=300, bbox_inches="tight")
fig.savefig(RES / "fig13_rating_comparison.pdf", bbox_inches="tight")
plt.close(fig)
print(f"\nsaved fig13_rating_comparison.png / .pdf")
