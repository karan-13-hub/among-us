#!/usr/bin/env python3
"""Bootstrap CIs for rating-vs-skill correlations + per-config breakdown.

Outputs (in results/):
  fig14_correlation_bootstrap_CIs.png — bootstrapped 95% CIs for fig5/fig2 bars
  fig15_per_config_correlations.png   — heatmap of r(rating, skill) per config × metric
  2026-05-03_correlation_bootstrap.csv — per-(role, metric) point + 95% CI
  2026-05-03_per_config_correlations.csv — full per-config × metric table
"""
import math
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

sweep = pd.read_csv(RES / f"{DATE}_tom_metrics_sweep_crossplay.csv")
games = pd.read_csv(RES / f"{DATE}_game_outcomes_long.csv")
games = games.dropna(subset=["crew_model", "imp_model"])

CREW_SKILLS = [
    ("detection skill\n(1 − MSE)",       "detection_accuracy",  "invert"),
    ("objective_viability",              "objective_viability", "raw"),
    ("alibi_grounding",                  "alibi_grounding",     "raw"),
    ("alibi_corroboration",              "alibi_corroboration", "raw"),
    ("faction_consensus",                "faction_consensus",   "raw"),
    ("social_influence",                 "social_influence",    "raw"),
    ("belief stability\n(1 − volatility)","belief_volatility",  "invert"),
    ("spatial_dispersion",               "spatial_dispersion",  "raw"),
]
IMP_SKILLS = [
    ("deceptive_efficacy",               "deceptive_efficacy",  "raw"),
    ("objective_viability\n(survival)",  "objective_viability", "raw"),
    ("alibi opacity\n(1 − grounding)",   "alibi_grounding",     "invert"),
    ("belief stability\n(1 − volatility)","belief_volatility",  "invert"),
    ("alibi_corroboration",              "alibi_corroboration", "raw"),
    ("faction_consensus",                "faction_consensus",   "raw"),
    ("social_influence",                 "social_influence",    "raw"),
    ("spatial_dispersion",               "spatial_dispersion",  "raw"),
]


def pearson(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    return float(np.corrcoef(x, y)[0, 1]) if len(x) >= 2 else float("nan")


# ======================================================================
# Per-game rows for each role's skill metric, by model.
# We bootstrap by resampling rows within each (model, role) group.
# ======================================================================
def role_rows(role_id):
    sub = sweep[sweep["identity"] == role_id].copy()
    if role_id == 0:
        sub["actor"] = sub["crew_model"]
    else:
        sub["actor"] = sub["imp_model"]
    return sub.dropna(subset=["actor"])


crew_rows = role_rows(0)
imp_rows  = role_rows(1)


def games_per_model_role(games, role):
    """Return DataFrame {model, n_games, n_wins} for the given role."""
    if role == "Crewmate":
        actor = "crew_model"; tgt = "Crewmate"
    else:
        actor = "imp_model"; tgt = "Impostor"
    g = games.copy()
    g["win"] = (g["winner_faction"] == tgt).astype(int)
    return (g.groupby(actor)
            .agg(n_games=("win", "size"), n_wins=("win", "sum"))
            .reset_index()
            .rename(columns={actor: "model"}))


def boot_correlation(role_rows_df, role_games, skill_col, mode,
                     n_boot=1000, seed=0):
    """Bootstrap Pearson r between (per-model log-odds rating) and
    (per-model mean of skill_col, sign-corrected)."""
    rng = np.random.default_rng(seed)

    # Pre-group rows by model for fast resampling.
    skill_by_model = {m: g[skill_col].dropna().values
                      for m, g in role_rows_df.groupby("actor")
                      if skill_col in g.columns}
    # role_games is the pre-aggregated DataFrame from games_per_model_role:
    #   columns = model, n_games, n_wins
    games_by_model = {row["model"]: row for _, row in role_games.iterrows()}

    models = sorted(set(skill_by_model.keys()) & set(games_by_model.keys()))

    rs = []
    for b in range(n_boot):
        skills = []
        rates = []
        for m in models:
            arr = skill_by_model[m]
            if len(arr) == 0:
                skills.append(np.nan)
            else:
                samp = arr[rng.integers(0, len(arr), len(arr))]
                v = samp.mean()
                if mode == "invert":
                    v = 1.0 - v
                skills.append(v)
            row = games_by_model[m]
            n = int(row["n_games"]); w = int(row["n_wins"])
            # Resample game outcomes via Beta-Binomial-ish: just resample
            # the n bernoullis with the same empirical rate.
            outcomes = rng.binomial(1, p=w / n, size=n)
            wr = outcomes.mean()
            wr_clip = min(max(wr, 1e-3), 1 - 1e-3)
            rates.append(math.log(wr_clip / (1 - wr_clip)))
        rs.append(pearson(skills, rates))
    rs = np.array(rs)
    return (np.nanmean(rs),
            np.nanpercentile(rs, 2.5),
            np.nanpercentile(rs, 97.5),
            len(models))


# ======================================================================
# Compute bootstrap CIs for both roles
# ======================================================================
print("[bootstrap] crewmate role")
crew_games = games_per_model_role(games, "Crewmate")
crew_results = []
for label, col, mode in CREW_SKILLS:
    mean_r, lo, hi, n = boot_correlation(crew_rows, crew_games, col, mode)
    crew_results.append({"role": "Crewmate", "metric": label,
                         "mean_r": mean_r, "ci_lo": lo, "ci_hi": hi,
                         "n_models": n})
    print(f"  {label.replace(chr(10), ' ')[:35]:35s}  "
          f"r = {mean_r:+.2f}  CI [{lo:+.2f}, {hi:+.2f}]")

print("\n[bootstrap] impostor role")
imp_games = games_per_model_role(games, "Impostor")
imp_results = []
for label, col, mode in IMP_SKILLS:
    mean_r, lo, hi, n = boot_correlation(imp_rows, imp_games, col, mode)
    imp_results.append({"role": "Impostor", "metric": label,
                        "mean_r": mean_r, "ci_lo": lo, "ci_hi": hi,
                        "n_models": n})
    print(f"  {label.replace(chr(10), ' ')[:35]:35s}  "
          f"r = {mean_r:+.2f}  CI [{lo:+.2f}, {hi:+.2f}]")

boot = pd.DataFrame(crew_results + imp_results)
boot.to_csv(RES / f"{DATE}_correlation_bootstrap.csv", index=False)


# ======================================================================
# Plot: forest-style 95% CI bars for both roles
# ======================================================================
plt.rcParams.update({"font.family": "sans-serif",
                     "axes.labelsize": 10.5,
                     "axes.titlesize": 11.5})
fig, axes = plt.subplots(1, 2, figsize=(15, 6.0),
                         gridspec_kw=dict(wspace=0.55))

for ax, role_results, named, role_name, role_top in [
    (axes[0], crew_results, "detection skill\n(1 − MSE)",  "Crewmate",
     "detection skill\n(1 − MSE)"),
    (axes[1], imp_results,  "deceptive_efficacy",          "Impostor",
     "objective_viability\n(survival)"),
]:
    df = pd.DataFrame(role_results)
    df["abs"] = df["mean_r"].abs()
    df = df.sort_values("abs", ascending=False).reset_index(drop=True)
    y = np.arange(len(df))
    colors = []
    for lbl in df["metric"]:
        if lbl == named and lbl == role_top:
            colors.append("#7c1f7c")
        elif lbl == role_top:
            colors.append("#1f77b4")
        elif lbl == named:
            colors.append("#d62728")
        else:
            colors.append("0.65")
    err_low = df["mean_r"] - df["ci_lo"]
    err_high = df["ci_hi"] - df["mean_r"]
    ax.errorbar(df["mean_r"], y,
                xerr=[err_low, err_high],
                fmt="none",
                capsize=5, elinewidth=1.6, mew=0.8,
                ecolor="0.45", zorder=2)
    for i, c in enumerate(colors):
        ax.scatter(df["mean_r"].iloc[i], i, s=140, c=c,
                   edgecolor="black", linewidth=0.7, zorder=3)
    ax.axvline(0, color="0.3", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(df["metric"], fontsize=10)
    ax.invert_yaxis()
    # Extra room on the right so r-value labels don't overlap right axis
    ax.set_xlim(-1.18, 1.18)
    ax.set_xlabel(
        f"Pearson r  with {role_name.lower()} ELO\n"
        "(point + 95% bootstrap CI, 1000 resamples)",
        fontsize=10)
    ax.set_title(f"({'a' if role_name=='Crewmate' else 'b'}) "
                 f"{role_name} ELO ↔ skill",
                 loc="left", fontsize=12)
    ax.grid(True, axis="x", alpha=0.25); ax.set_axisbelow(True)
    # Place the r-value labels at the OUTER end of the CI (past the cap),
    # not on top of the marker, so they never overlap.
    for i, r in enumerate(df["mean_r"]):
        ci_outer = df["ci_hi"].iloc[i] if r >= 0 else df["ci_lo"].iloc[i]
        # +/- margin so the text sits clearly past the whisker cap.
        x_text = ci_outer + (0.04 if r >= 0 else -0.04)
        ax.text(x_text, i, f"{r:+.2f}",
                va="center",
                ha="left" if r >= 0 else "right",
                fontsize=10, color="0.15")

fig.suptitle("Bootstrapped 95% CIs on rating-vs-skill correlations",
             fontsize=13, y=1.0)
fig.tight_layout()
fig.savefig(RES / "fig14_correlation_bootstrap_CIs.png", dpi=300,
            bbox_inches="tight")
fig.savefig(RES / "fig14_correlation_bootstrap_CIs.pdf",
            bbox_inches="tight")
plt.close(fig)
print("saved fig14_correlation_bootstrap_CIs.png / .pdf")


# ======================================================================
# Per-config breakdown
# ======================================================================
print("\n[per-config] computing role × metric × config correlation table")


def per_config_correlations(role_rows_df, role, games, skill_pairs):
    rows = []
    for cfg, gsub in role_rows_df.groupby("config"):
        gw = games[games["config"] == cfg]
        rg = games_per_model_role(gw, role)
        # Per-model means in this config
        for label, col, mode in skill_pairs:
            if col not in gsub.columns:
                continue
            means = (gsub.dropna(subset=[col, "actor"])
                     .groupby("actor")[col].mean())
            if mode == "invert":
                means = 1.0 - means
            merged = pd.merge(rg, means.reset_index().rename(
                columns={"actor": "model"}),
                on="model", how="inner")
            if len(merged) < 3:
                continue
            wr = merged["n_wins"] / merged["n_games"]
            wr_clip = wr.clip(1e-3, 1 - 1e-3)
            elo = np.log(wr_clip / (1 - wr_clip))
            r = pearson(elo, merged[col])
            rows.append({"role": role, "config": cfg,
                         "metric": label, "n_models": len(merged),
                         "r": r})
    return rows


crew_per_cfg = per_config_correlations(crew_rows, "Crewmate", games,
                                       CREW_SKILLS)
imp_per_cfg  = per_config_correlations(imp_rows,  "Impostor", games,
                                       IMP_SKILLS)
per_cfg = pd.DataFrame(crew_per_cfg + imp_per_cfg)
per_cfg.to_csv(RES / f"{DATE}_per_config_correlations.csv", index=False)
print(f"saved per_config_correlations.csv ({len(per_cfg)} rows)")

# Heatmap: rows = (role + metric), cols = config
piv_crew = (per_cfg[per_cfg["role"] == "Crewmate"]
            .pivot_table(index="metric", columns="config", values="r"))
piv_imp = (per_cfg[per_cfg["role"] == "Impostor"]
           .pivot_table(index="metric", columns="config", values="r"))
order_cfg = ["4C_1I", "4C_2I", "5C_1I", "5C_2I"]
piv_crew = piv_crew.reindex(columns=order_cfg)
piv_imp  = piv_imp.reindex(columns=order_cfg)
piv_crew = piv_crew.reindex(
    [lbl for lbl, _, _ in CREW_SKILLS if lbl in piv_crew.index])
piv_imp = piv_imp.reindex(
    [lbl for lbl, _, _ in IMP_SKILLS if lbl in piv_imp.index])

fig, axes = plt.subplots(1, 2, figsize=(13, 5.6),
                         gridspec_kw=dict(wspace=0.55))
for ax, piv, title in [
    (axes[0], piv_crew, "(a) Crewmate ELO ↔ skill, per config"),
    (axes[1], piv_imp,  "(b) Impostor ELO ↔ skill, per config"),
]:
    if piv.empty:
        ax.set_visible(False); continue
    arr = piv.values.astype(float)
    # Display data with NaNs masked so they show as the cmap's "bad" color
    masked = np.ma.array(arr, mask=np.isnan(arr))
    cmap = matplotlib.cm.get_cmap("RdBu_r").copy()
    cmap.set_bad("#dcdcdc")
    im = ax.imshow(masked, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns,
                                                           fontsize=10)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index,
                                                           fontsize=10)
    ax.set_title(title, loc="left", fontsize=12)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = arr[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center",
                        color="0.4", fontsize=9, style="italic")
            else:
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        color="black" if abs(v) < 0.6 else "white",
                        fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

# Note about the n/a cells (alibi_corroboration / spatial_dispersion are
# identically zero in 1-impostor configs, so r is undefined).
fig.text(0.5, -0.03,
         "n/a cells: metric has zero variance in 1-impostor configs "
         "(e.g., alibi_corroboration requires ≥2 impostors).",
         ha="center", va="top", fontsize=9, color="0.3", style="italic")
fig.suptitle("Per-config rating-vs-skill correlations  "
             "(Pearson r at each game configuration)",
             fontsize=12.5, y=1.02)
fig.tight_layout()
fig.savefig(RES / "fig15_per_config_correlations.png",
            dpi=300, bbox_inches="tight")
fig.savefig(RES / "fig15_per_config_correlations.pdf",
            bbox_inches="tight")
plt.close(fig)
print("saved fig15_per_config_correlations.png / .pdf")
