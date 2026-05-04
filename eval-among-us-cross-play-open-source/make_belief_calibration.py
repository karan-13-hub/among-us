#!/usr/bin/env python3
"""Belief-calibration reliability diagrams for cross-play.

For every crewmate snapshot we extract:
  - p = predicted P(player j is impostor)  (verbalized OR logprob)
  - y = ground-truth indicator (1 if j was impostor, 0 if crewmate)

We then plot:
  fig_calibration_paired.png       — verbalized vs logprob, all crewmates pooled
  fig_calibration_per_crew.png     — reliability diagram per crew-model
  fig_calibration_within_vs_across.png — within-family vs across-family pair calibration
  + ECE/Brier/LogLoss per (crew_model, role) tables.

This is the cross-play analogue of the self-play
`belief_calibration_*.png` artefacts.
"""
import importlib
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/yjangir1/scratchhbharad2/users/yjangir1/karan")
SWEEP_ROOT = ROOT / "eval-among-us-crossplay"
OUT_ROOT = ROOT / "eval-among-us-crossplay-open-source"
RESULTS_PATH = OUT_ROOT / "results"
DATE = "2026-05-03"

EXCLUDE_MATCHUPS = {
    "eval-cross-play-among-us-deepseek-qwen32b-vs-qwen3_32b",
}

AMONG_US_PATH = ROOT / "among-us"
sys.path.insert(0, str(AMONG_US_PATH / "among-agents"))
sys.path.insert(0, str(AMONG_US_PATH))
sys.path.insert(0, str(AMONG_US_PATH / "evaluations"))

print("Importing metrics_calculator ...")
_mc = importlib.import_module("evaluations.metrics_calculator")
EpistemicLogParser = _mc.EpistemicLogParser
_extract_ground_truth_from_summary = _mc._extract_ground_truth_from_summary
_load_all_summaries = _mc._load_all_summaries


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
FAMILY_COLOR = {
    "gemma": "#1f77b4", "qwen3": "#2ca02c",
    "llama": "#d62728", "deepseek-distill": "#9467bd",
}


def canonical(name):
    if name is None:
        return name
    aliases = {
        "qwen3-4b-instruct":            "qwen3-4b",
        "qwen3-8b":                     "qwen3-8B",
        "deepseek-r1-distill-llama-8b": "deepseek-r1-distill-llama-8B",
        "llama-3.1-8b-instruct":        "llama-3.1-8b",
        "gemma-4-31B-it":               "gemma-4-31b",
        "llama-3.3-70b-instruct":       "llama-3.3-70b",
    }
    return aliases.get(name, name)


def _norm(s):
    return s.lower().replace(":", "").replace(" ", "").replace("-", "_")


def _ece(p, y, n_bins=10):
    if len(p) == 0:
        return float("nan")
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        e += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
    return float(e)


def _brier(p, y):
    if len(p) == 0:
        return float("nan")
    return float(((p - y) ** 2).mean())


def _logloss(p, y, eps=1e-7):
    if len(p) == 0:
        return float("nan")
    p = np.clip(p, eps, 1 - eps)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _reliability(p, y, n_bins=10):
    if len(p) == 0:
        return []
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        out.append((float(p[m].mean()), float(y[m].mean()),
                    float(m.sum() / len(p))))
    return out


def parse_compact_log_models(path: Path) -> dict:
    if not path.exists():
        return {}
    txt = path.read_text()
    out = {}
    dec = json.JSONDecoder()
    i, n = 0, len(txt)
    while i < n:
        while i < n and txt[i].isspace():
            i += 1
        if i >= n:
            break
        try:
            obj, j = dec.raw_decode(txt, i)
        except json.JSONDecodeError:
            break
        i = j
        g = obj.get("game_index")
        pl = obj.get("player", {}) or {}
        ident = pl.get("identity")
        mp = pl.get("model")
        if not (g and ident and mp):
            continue
        m = canonical(os.path.basename(mp) if isinstance(mp, str) else mp)
        slot = out.setdefault(g, {"crew": set(), "imp": set(),
                                  "by_player": {}})
        if ident == "Crewmate":
            slot["crew"].add(m)
        elif ident == "Impostor":
            slot["imp"].add(m)
        # also remember which player had which model
        nm = pl.get("name")
        if nm:
            slot["by_player"][_norm(nm)] = (ident, m)
    return out


# ===============================================================
# Walk every exp dir, accumulate (p, y, crew_model, imp_model, role,
# config, signal) tuples
# ===============================================================
print(f"sweep root: {SWEEP_ROOT}")
verb_records = []
logp_records = []
n_exps = 0
n_skipped = 0
t0 = time.time()

for mdir in sorted(SWEEP_ROOT.iterdir()):
    if not mdir.is_dir() or mdir.name in EXCLUDE_MATCHUPS:
        continue
    for edir in sorted(mdir.glob("*_exp_*")):
        n_exps += 1
        summary = edir / "summary.json"
        epi = edir / "epistemic-states.jsonl"
        comp = edir / "agent-logs-compact.json"
        if not (summary.exists() and epi.exists() and comp.exists()):
            n_skipped += 1
            continue
        # ground truth per game
        summaries = _load_all_summaries(str(summary))
        if not summaries:
            n_skipped += 1
            continue
        games_gt = {}
        all_players = set()
        for s in summaries:
            gk = next((k for k in s if k.startswith("Game")), None)
            if not gk:
                continue
            gt = _extract_ground_truth_from_summary(s, gk)
            if gt is None:
                continue
            games_gt[gk] = gt
            all_players.update(gt.player_names)
        if not games_gt:
            n_skipped += 1
            continue

        # game-level model assignments
        gm = parse_compact_log_models(comp)

        # parse epistemic snapshots
        try:
            parser = EpistemicLogParser(all_player_names=sorted(all_players))
            snaps = parser.parse_file(str(epi))
        except Exception:
            n_skipped += 1
            continue
        if not snaps:
            n_skipped += 1
            continue
        # assign game index by player roster
        players_per_game = {gk: set(gt.player_names)
                            for gk, gt in games_gt.items()}
        _mc._assign_game_index_by_players(snaps, players_per_game)

        # config
        try:
            details = (edir / "experiment-details.txt").read_text()
            import re
            m = re.search(r"'num_players':\s*(\d+),\s*'num_impostors':\s*(\d+)",
                          details)
            n_players, n_imp = int(m.group(1)), int(m.group(2))
            cfg = f"{n_players - n_imp}C_{n_imp}I"
        except Exception:
            cfg = "?"

        for snap in snaps:
            gi = snap.game_index
            gk = f"Game {gi}" if isinstance(gi, int) else gi
            gt = games_gt.get(gk)
            if gt is None:
                continue
            slot = gm.get(gk, {})
            crew_set = slot.get("crew", set())
            imp_set = slot.get("imp", set())
            crew_model = sorted(crew_set)[0] if len(crew_set) == 1 else None
            imp_model = sorted(imp_set)[0] if len(imp_set) == 1 else None
            if not (crew_model and imp_model):
                continue
            imp_norm = {_norm(p) for p in gt.impostor_names}

            # Only collect crewmates' beliefs (the "detection" task).
            # snap.player is the player whose snapshot this is — only
            # include if that player was a crewmate.
            this_player_id = slot.get("by_player", {}).get(_norm(snap.player),
                                                           ("?", "?"))[0]
            if this_player_id != "Crewmate":
                continue

            bd = snap.belief_distribution or {}
            for j, prob in bd.items():
                y = 1.0 if _norm(j) in imp_norm else 0.0
                verb_records.append({
                    "p": float(prob), "y": y,
                    "crew_model": crew_model, "imp_model": imp_model,
                    "config": cfg, "experiment": edir.name,
                    "matchup_dir": mdir.name,
                })
            bd_lp = snap.belief_distribution_logprobs or {}
            for j, prob in bd_lp.items():
                y = 1.0 if _norm(j) in imp_norm else 0.0
                logp_records.append({
                    "p": float(prob), "y": y,
                    "crew_model": crew_model, "imp_model": imp_model,
                    "config": cfg, "experiment": edir.name,
                    "matchup_dir": mdir.name,
                })

print(f"\n[parse] exps={n_exps}  skipped={n_skipped}  "
      f"wall={time.time() - t0:.1f}s")
print(f"  verbalized predictions: {len(verb_records):,}")
print(f"  logprob    predictions: {len(logp_records):,}")

verb = pd.DataFrame(verb_records)
logp = pd.DataFrame(logp_records)
verb["crew_model"] = verb["crew_model"].map(canonical)
verb["imp_model"] = verb["imp_model"].map(canonical)
logp["crew_model"] = logp["crew_model"].map(canonical)
logp["imp_model"] = logp["imp_model"].map(canonical)
verb["matchup"] = "crew=" + verb["crew_model"] + "__imp=" + verb["imp_model"]
logp["matchup"] = "crew=" + logp["crew_model"] + "__imp=" + logp["imp_model"]
verb.to_csv(RESULTS_PATH / f"{DATE}_belief_calibration_verbal_long.csv",
            index=False)
logp.to_csv(RESULTS_PATH / f"{DATE}_belief_calibration_logprob_long.csv",
            index=False)
print("saved long-form CSVs")


# ===============================================================
# Per-crew-model + per-matchup ECE/Brier/LogLoss
# ===============================================================
def aggregate(df, key="crew_model", signal="?"):
    rows = []
    for k, g in df.groupby(key):
        rows.append({
            key: k, "signal": signal, "n": len(g),
            "ECE": _ece(g["p"].values, g["y"].values),
            "Brier": _brier(g["p"].values, g["y"].values),
            "LogLoss": _logloss(g["p"].values, g["y"].values),
            "prior": float(g["y"].mean()),
        })
    return pd.DataFrame(rows)

agg_crew_v = aggregate(verb, "crew_model", "verbal")
agg_crew_l = aggregate(logp, "crew_model", "logprob") if not logp.empty else pd.DataFrame()
pd.concat([agg_crew_v, agg_crew_l], ignore_index=True).to_csv(
    RESULTS_PATH / f"{DATE}_belief_calibration_by_crew_model.csv",
    index=False)

agg_mu_v = aggregate(verb, "matchup", "verbal")
agg_mu_l = aggregate(logp, "matchup", "logprob") if not logp.empty else pd.DataFrame()
pd.concat([agg_mu_v, agg_mu_l], ignore_index=True).to_csv(
    RESULTS_PATH / f"{DATE}_belief_calibration_by_matchup.csv", index=False)


# ===============================================================
# Plot 1 — paired calibration (all crewmates, verbal vs logprob)
# ===============================================================
plt.rcParams.update({"font.family": "sans-serif",
                     "axes.labelsize": 11, "axes.titlesize": 12,
                     "xtick.labelsize": 10, "ytick.labelsize": 10})

fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
ax = axes[0]
ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
for label, df, color in [
    ("Verbalized", verb, "#ff7f0e"),
    ("Logprob",    logp, "#1f77b4"),
]:
    if df.empty:
        continue
    p = df["p"].values; y = df["y"].values
    rows = _reliability(p, y, n_bins=10)
    if not rows:
        continue
    xs, ys, ws = zip(*rows)
    ax.plot(xs, ys, "o-", color=color,
            label=f"{label}  (ECE={_ece(p, y):.3f}, n={len(p):,})")
    ax.scatter(xs, ys, s=[max(20, 800 * w) for w in ws],
               color=color, alpha=0.25)
ax.set_xlabel("Predicted P(player j is Impostor)")
ax.set_ylabel("Observed frequency")
ax.set_title("(a) Crewmate belief calibration — verbal vs logprob",
             loc="left")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3); ax.legend(loc="lower right")

ax = axes[1]
bins = np.linspace(0, 1, 21)
if not verb.empty:
    ax.hist(verb["p"].values, bins=bins, alpha=0.5, color="#ff7f0e",
            label=f"Verbalized  (n={len(verb):,})", density=True)
if not logp.empty:
    ax.hist(logp["p"].values, bins=bins, alpha=0.5, color="#1f77b4",
            label=f"Logprob  (n={len(logp):,})", density=True)
ax.set_xlabel("Predicted P(Impostor)")
ax.set_ylabel("Density")
ax.set_title("(b) Distribution of predictions",
             loc="left")
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(RESULTS_PATH / "fig16_belief_calibration_paired.png",
            dpi=300, bbox_inches="tight")
fig.savefig(RESULTS_PATH / "fig16_belief_calibration_paired.pdf",
            bbox_inches="tight")
plt.close(fig)
print("saved fig16_belief_calibration_paired.png / .pdf")


# ===============================================================
# Plot 2 — per-crew-model reliability diagrams (verbalized only)
# ===============================================================
crews = sorted(verb["crew_model"].dropna().unique())
ncols = 3
nrows = (len(crews) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.0 * nrows))
axes = np.atleast_1d(axes).ravel()
for i, m in enumerate(crews):
    ax = axes[i]
    sub = verb[verb["crew_model"] == m]
    p = sub["p"].values; y = sub["y"].values
    prior = float(y.mean()) if len(y) else float("nan")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.axhline(prior, color="#d62728", ls=":", alpha=0.7,
               label=f"Prior={prior:.2f}")
    rows = _reliability(p, y, n_bins=10)
    if rows:
        xs, ys, ws = zip(*rows)
        c = FAMILY_COLOR.get(FAMILY_OF.get(m, "?"), "0.4")
        ax.plot(xs, ys, "o-", color=c,
                label=f"ECE={_ece(p, y):.3f}")
        ax.scatter(xs, ys, s=[max(15, 600 * w) for w in ws],
                   color=c, alpha=0.30)
    ax.set_title(f"{SHORT.get(m, m)}   n={len(sub):,}",
                 fontsize=11, loc="left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8.5, loc="upper left")
    if i % ncols == 0:
        ax.set_ylabel("Observed frequency")
    if i // ncols == nrows - 1:
        ax.set_xlabel("Predicted P(Impostor)")
for j in range(len(crews), len(axes)):
    axes[j].axis("off")
fig.suptitle("Per-crew-model belief calibration (verbalized)",
             fontsize=13, y=1.0)
fig.tight_layout()
fig.savefig(RESULTS_PATH / "fig17_belief_calibration_per_crew_model.png",
            dpi=300, bbox_inches="tight")
fig.savefig(RESULTS_PATH / "fig17_belief_calibration_per_crew_model.pdf",
            bbox_inches="tight")
plt.close(fig)
print("saved fig17_belief_calibration_per_crew_model.png / .pdf")


# ===============================================================
# Plot 3 — within-family vs across-family pair calibration
# ===============================================================
verb["crew_family"] = verb["crew_model"].map(FAMILY_OF.get)
verb["imp_family"] = verb["imp_model"].map(FAMILY_OF.get)
verb["family_relation"] = np.where(
    verb["crew_family"] == verb["imp_family"],
    "within_family", "across_family")

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
for rel, color in [
    ("within_family", "#1f77b4"),
    ("across_family", "#d62728"),
]:
    sub = verb[verb["family_relation"] == rel]
    if sub.empty:
        continue
    p = sub["p"].values; y = sub["y"].values
    rows = _reliability(p, y, n_bins=10)
    if not rows:
        continue
    xs, ys, ws = zip(*rows)
    ax.plot(xs, ys, "o-", color=color,
            label=f"{rel.replace('_', ' ')}  (ECE={_ece(p, y):.3f}, n={len(sub):,})")
    ax.scatter(xs, ys, s=[max(20, 800 * w) for w in ws],
               color=color, alpha=0.25)
ax.set_xlabel("Predicted P(player j is Impostor)")
ax.set_ylabel("Observed frequency")
ax.set_title("Belief calibration: within-family vs across-family")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3); ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(RESULTS_PATH / "fig18_belief_calibration_family.png",
            dpi=300, bbox_inches="tight")
fig.savefig(RESULTS_PATH / "fig18_belief_calibration_family.pdf",
            bbox_inches="tight")
plt.close(fig)
print("saved fig18_belief_calibration_family.png / .pdf")
print("\nDone.")
