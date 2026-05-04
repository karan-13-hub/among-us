#!/usr/bin/env python3
"""Cross-play analysis driver — Among Us.

Walks every matchup folder under
``/home/yjangir1/scratchhbharad2/users/yjangir1/karan/eval-among-us-crossplay/``,
backfills ToM metrics for matchups missing them, then produces:

  Stage 0 : per-experiment ToM metrics (reuse if cached)
  Stage 1 : combined sweep CSV with matchup metadata
  Stage 2 : win rates (long-form outcomes, matchup x config table,
            crew x imp matrix, win-category breakdown)
  Stage 3 : per-role metric tables — by matchup and model-pooled
  Stage 4 : size-bucket triangle round-robins (small 3-4B, medium 8B, large)
  Stage 5 : within-family vs across-family comparison
  Stage 6 : cross-play vs self-play delta (per model, per role)
  Stage 7 : Pareto radar plots for the triangles
  Stage 8 : ELO-vs-skill empirical demonstration
            (game-outcome derived ratings vs detection_accuracy /
             deceptive_efficacy)

All outputs land in ``eval-among-us-crossplay-open-source/results/``.
"""
import importlib
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
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
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

SELFPLAY_RESULTS = ROOT / "eval-among-us-selfplay" / "results"
SELFPLAY_DATE = "2026-05-03"

DATE = datetime.now().strftime("%Y-%m-%d")

LOG_PATH = RESULTS_PATH / f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
_LOG_FH = open(LOG_PATH, "w", buffering=1)


def log(msg: str = "") -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    _LOG_FH.write(line + "\n")


def banner(msg: str) -> None:
    log("")
    log("=" * 80)
    log(msg)
    log("=" * 80)


log(f"Log file: {LOG_PATH}")
log(f"Sweep root:   {SWEEP_ROOT}")
log(f"Output root:  {OUT_ROOT}")
log(f"Results path: {RESULTS_PATH}")
_t_start = time.time()

AMONG_US_PATH = ROOT / "among-us"
sys.path.insert(0, str(AMONG_US_PATH / "among-agents"))
sys.path.insert(0, str(AMONG_US_PATH))
sys.path.insert(0, str(AMONG_US_PATH / "evaluations"))

log("Importing metrics_calculator from among-us/evaluations ...")
_mc = importlib.import_module("evaluations.metrics_calculator")
process_experiment = _mc.process_experiment
log("  metrics_calculator imported OK")

# -------------------------------------------------------------------------
# Family / size taxonomy (consistent with self-play scripts)
# -------------------------------------------------------------------------
FAMILY_OF = {
    "gemma-4-E4B-it":               "gemma",
    "gemma-4-26B-A4B-it":           "gemma",
    "gemma-4-31b":                  "gemma",
    "qwen3-4b":                     "qwen3",
    "qwen3-4b-instruct":            "qwen3",
    "qwen3-8B":                     "qwen3",
    "qwen3-8b":                     "qwen3",
    "qwen3-32b":                    "qwen3",
    "llama-3.2-3b-instruct":        "llama",
    "llama-3.1-8b":                 "llama",
    "llama-3.1-8b-instruct":        "llama",
    "llama-3.3-70b":                "llama",
    "deepseek-r1-distill-llama-8B": "deepseek-distill",
    "deepseek-r1-distill-llama-8b": "deepseek-distill",
    "deepseek-r1-distill-qwen-32b": "deepseek-distill",
}
SIZE_OF = {
    "llama-3.2-3b-instruct":        "3-4B",
    "qwen3-4b":                     "3-4B",
    "qwen3-4b-instruct":            "3-4B",
    "gemma-4-E4B-it":               "3-4B",
    "deepseek-r1-distill-llama-8B": "8B",
    "deepseek-r1-distill-llama-8b": "8B",
    "llama-3.1-8b":                 "8B",
    "llama-3.1-8b-instruct":        "8B",
    "qwen3-8B":                     "8B",
    "qwen3-8b":                     "8B",
    "gemma-4-26B-A4B-it":           "26-32B",
    "gemma-4-31b":                  "26-32B",
    "deepseek-r1-distill-qwen-32b": "26-32B",
    "qwen3-32b":                    "26-32B",
    "llama-3.3-70b":                "70B",
}


def fam(m): return FAMILY_OF.get(m, "unknown")
def size(m): return SIZE_OF.get(m, "unknown")


def canonical_model(name: str) -> str:
    """Normalize the few model-name spellings that vary across matchups."""
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


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
WINNER_CATEGORY = {
    1: ("Impostor", "Outnumber"),
    2: ("Crewmate", "Ejection"),
    3: ("Crewmate", "Tasks"),
    4: ("Impostor", "Timeout"),
}
CATEGORY_ORDER = ["Ejection", "Tasks", "Outnumber", "Timeout"]
CONFIG_ORDER = ["4C_1I", "4C_2I", "5C_1I", "5C_2I"]

IMPOSTOR_METRICS = [
    "deceptive_efficacy", "spatial_dispersion", "alibi_corroboration",
    "alibi_grounding", "social_influence", "belief_volatility",
    "faction_consensus", "objective_viability",
]
CREWMATE_METRICS = [
    "detection_accuracy", "spatial_dispersion", "alibi_corroboration",
    "alibi_grounding", "social_influence", "belief_volatility",
    "faction_consensus", "objective_viability",
]


def _cell(vals):
    vals = vals.dropna()
    if len(vals) == 0:
        return "—"
    if len(vals) == 1:
        return f"{vals.iloc[0]:.4f}"
    return f"{vals.mean():.4f} ± {vals.std():.4f}"


def parse_compact_log_models(path: Path) -> dict:
    """Return {game_index_str -> {'crew': set[model], 'imp': set[model]}}.

    Reads ``agent-logs-compact.json`` (concatenated JSON objects, one per
    snapshot) and extracts the {identity, model} pairs each game saw.
    """
    if not path.exists():
        return {}
    try:
        txt = path.read_text()
    except Exception as ex:
        log(f"    [WARN] failed to read {path}: {ex}")
        return {}
    out: dict[str, dict[str, set]] = {}
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
        model_path = pl.get("model")
        if not (g and ident and model_path):
            continue
        m = canonical_model(os.path.basename(model_path) if isinstance(model_path, str) else model_path)
        slot = out.setdefault(g, {"crew": set(), "imp": set()})
        if ident == "Crewmate":
            slot["crew"].add(m)
        elif ident == "Impostor":
            slot["imp"].add(m)
    return out


def parse_config_from_details(details_path: Path) -> str | None:
    if not details_path.exists():
        return None
    text = details_path.read_text()
    m = re.search(r"'num_players':\s*(\d+),\s*'num_impostors':\s*(\d+)", text)
    if not m:
        return None
    n_players = int(m.group(1))
    n_imp = int(m.group(2))
    return f"{n_players - n_imp}C_{n_imp}I"


# -------------------------------------------------------------------------
# Stage 0 — Discover experiments + ensure tom_metrics CSVs exist
# -------------------------------------------------------------------------
banner("STAGE 0 — Discovering matchups & experiments")


EXCLUDE_MATCHUPS: set[str] = set()
# Add a matchup folder name here to skip it (e.g. when its exp dirs are
# being actively written by a rerun and may be partial).


def discover():
    """Yield {matchup_dir, exp_dir, config, exp_name} dicts."""
    out = []
    skipped = []
    for mdir in sorted(SWEEP_ROOT.iterdir()):
        if not mdir.is_dir():
            continue
        if mdir.name in EXCLUDE_MATCHUPS:
            log(f"  [exclude] {mdir.name} — skipping (see EXCLUDE_MATCHUPS)")
            continue
        for edir in sorted(mdir.glob("*_exp_*")):
            details = edir / "experiment-details.txt"
            agent_log = edir / "agent-logs-compact.json"
            summary = edir / "summary.json"
            epi = edir / "epistemic-states.jsonl"
            missing = []
            for p, lbl in [(details, "details"), (agent_log, "agent-logs-compact"),
                           (summary, "summary"), (epi, "epistemic")]:
                if not p.exists():
                    missing.append(lbl)
            if missing:
                skipped.append((str(edir), missing))
                continue
            cfg = parse_config_from_details(details)
            if not cfg:
                skipped.append((str(edir), ["unparseable_details"]))
                continue
            out.append({
                "matchup_dir": str(mdir),
                "matchup_name": mdir.name,
                "exp_dir": str(edir),
                "exp_name": edir.name,
                "config": cfg,
            })
    return out, skipped


experiments, skipped = discover()
log(f"Found {len(experiments)} experiments across "
    f"{len({e['matchup_name'] for e in experiments})} matchups.")
if skipped:
    log(f"Skipped {len(skipped)}:")
    for path, reasons in skipped:
        log(f"  - {path}  ({','.join(reasons)})")

# Per-matchup summary
_by_matchup = defaultdict(list)
for e in experiments:
    _by_matchup[e["matchup_name"]].append(e["config"])
for mu in sorted(_by_matchup):
    log(f"  {mu:<70s}  n_exp={len(_by_matchup[mu]):2d}  cfgs={sorted(_by_matchup[mu])}")


# -------------------------------------------------------------------------
# Stage 1 — Per-experiment ToM metrics + matchup metadata
# -------------------------------------------------------------------------
banner(f"STAGE 1 — Per-experiment ToM metrics  [{len(experiments)} experiments]")


def attach_matchup_metadata(df: pd.DataFrame, exp_dir: str, exp_name: str,
                            config: str) -> pd.DataFrame | None:
    """Add crew_model / imp_model / matchup / config columns by joining
    on game_index against the per-game model assignments parsed from
    agent-logs-compact.json.
    """
    if df is None or df.empty:
        return df
    game_models = parse_compact_log_models(Path(exp_dir) / "agent-logs-compact.json")
    if not game_models:
        log(f"    [WARN] no agent-logs-compact mapping in {exp_dir}")
        return None
    # df has 'game' = "Game N" string. Map to (crew_model, imp_model, matchup).
    crew_col, imp_col, mu_col = [], [], []
    bad = 0
    for g in df["game"].astype(str):
        slot = game_models.get(g)
        if not slot:
            crew_col.append(None); imp_col.append(None); mu_col.append(None); bad += 1
            continue
        crews = slot["crew"]; imps = slot["imp"]
        # In well-formed cross-play exps a single model plays each role.
        cm = sorted(crews)[0] if len(crews) == 1 else "/".join(sorted(crews)) if crews else None
        im = sorted(imps)[0] if len(imps) == 1 else "/".join(sorted(imps)) if imps else None
        crew_col.append(cm); imp_col.append(im)
        mu_col.append(f"crew={cm}__imp={im}" if cm and im else None)
    if bad:
        log(f"    [WARN] {bad}/{len(df)} rows had no game-match in agent-logs ({exp_dir})")
    df = df.copy()
    df["crew_model"] = crew_col
    df["imp_model"] = imp_col
    df["matchup"] = mu_col
    df["config"] = config
    df["experiment"] = exp_name
    return df


per_exp_dfs = []
for i, e in enumerate(experiments, 1):
    cached = (Path(e["matchup_dir"]) / "results" /
              f"{e['exp_name']}_tom_metrics.csv")
    df = None
    if cached.exists():
        try:
            df = pd.read_csv(cached)
            # Some cached CSVs already have matchup columns; some don't.
            need = {"crew_model", "imp_model", "matchup", "config"} - set(df.columns)
            if need:
                df = attach_matchup_metadata(df, e["exp_dir"],
                                             e["exp_name"], e["config"])
        except Exception as ex:
            log(f"  [{i:3d}/{len(experiments)}] cache read failed for "
                f"{cached.name}: {ex} -- recomputing")
            df = None
    if df is None:
        t0 = time.time()
        try:
            df = process_experiment(e["exp_dir"], source="verbalized")
        except Exception as ex:
            log(f"  [{i:3d}/{len(experiments)}] process_experiment failed "
                f"for {e['exp_name']}: {ex}")
            continue
        if df is None or df.empty:
            log(f"  [{i:3d}/{len(experiments)}] {e['matchup_name']} "
                f"{e['exp_name']}  EMPTY  ({time.time()-t0:.1f}s)")
            continue
        df = attach_matchup_metadata(df, e["exp_dir"], e["exp_name"],
                                     e["config"])
        # Persist back to the matchup's results dir so future runs are fast.
        cached.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cached, index=False)
        log(f"  [{i:3d}/{len(experiments)}] computed {e['matchup_name']} "
            f"{e['exp_name']}  rows={len(df):4d}  ({time.time()-t0:.1f}s)")
    if df is None or df.empty:
        continue
    df["matchup_dir"] = e["matchup_name"]
    per_exp_dfs.append(df)

df_metrics = pd.concat(per_exp_dfs, ignore_index=True) if per_exp_dfs else pd.DataFrame()
if not df_metrics.empty:
    # Canonicalize model names — the small-triangle matchup uses
    # "qwen3-4b-instruct" while elsewhere it's "qwen3-4b", etc.
    df_metrics["crew_model"] = df_metrics["crew_model"].map(canonical_model)
    df_metrics["imp_model"] = df_metrics["imp_model"].map(canonical_model)
    df_metrics["matchup"] = (
        "crew=" + df_metrics["crew_model"].astype(str) +
        "__imp=" + df_metrics["imp_model"].astype(str)
    )
    df_metrics["role"] = df_metrics["identity"].map({0: "Crewmate", 1: "Impostor"})
    df_metrics["crew_family"] = df_metrics["crew_model"].map(fam)
    df_metrics["imp_family"] = df_metrics["imp_model"].map(fam)
    df_metrics["crew_size"] = df_metrics["crew_model"].map(size)
    df_metrics["imp_size"] = df_metrics["imp_model"].map(size)
    df_metrics["family_relation"] = np.where(
        df_metrics["crew_family"] == df_metrics["imp_family"],
        "within_family", "across_family",
    )
    out = RESULTS_PATH / f"{DATE}_tom_metrics_sweep_crossplay.csv"
    df_metrics.to_csv(out, index=False)
    log(f"Combined sweep: {len(df_metrics)} rows | "
        f"{df_metrics['matchup'].nunique()} matchups | "
        f"{df_metrics['crew_model'].nunique()} crew-models | "
        f"{df_metrics['imp_model'].nunique()} imp-models -> {out.name}")
else:
    log("[FATAL] no metrics rows accumulated; aborting downstream stages.")
    sys.exit(1)


# -------------------------------------------------------------------------
# Stage 2 — Win rates
# -------------------------------------------------------------------------
banner("STAGE 2 — Win rates / categories / matrix")


def collect_outcomes() -> pd.DataFrame:
    rows = []
    for e in experiments:
        sp = Path(e["exp_dir"]) / "summary.json"
        if not sp.exists():
            continue
        gm = parse_compact_log_models(
            Path(e["exp_dir"]) / "agent-logs-compact.json"
        )
        with open(sp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for k, v in obj.items():
                    if not k.startswith("Game") or not isinstance(v, dict):
                        continue
                    w = v.get("winner")
                    fac, cat = WINNER_CATEGORY.get(w, ("Unknown", "Unknown"))
                    slot = gm.get(k, {"crew": set(), "imp": set()})
                    cm = sorted(slot["crew"])[0] if len(slot["crew"]) == 1 else None
                    im = sorted(slot["imp"])[0] if len(slot["imp"]) == 1 else None
                    cm = canonical_model(cm); im = canonical_model(im)
                    rows.append({
                        "matchup_dir": e["matchup_name"],
                        "experiment": e["exp_name"],
                        "config": e["config"],
                        "game_index": int(k.split()[-1]),
                        "crew_model": cm,
                        "imp_model": im,
                        "matchup": (f"crew={cm}__imp={im}"
                                    if cm and im else None),
                        "winner_code": w,
                        "winner_faction": fac,
                        "winner_category": cat,
                    })
    return pd.DataFrame(rows)


df_wins = collect_outcomes()
if df_wins.empty:
    log("[WARN] no game outcomes recovered.")
else:
    df_wins.to_csv(RESULTS_PATH / f"{DATE}_game_outcomes_long.csv", index=False)
    log(f"  saved game_outcomes_long.csv  ({len(df_wins)} games)")

    # 2a) win rate per (matchup, config) — long form
    grp = df_wins.groupby(["matchup", "crew_model", "imp_model", "config"],
                          observed=True, dropna=False)
    n_games = grp.size().rename("n_games")
    crew_rate = grp["winner_faction"].apply(
        lambda s: (s == "Crewmate").mean()).rename("crew_win_rate")
    imp_rate = grp["winner_faction"].apply(
        lambda s: (s == "Impostor").mean()).rename("imp_win_rate")
    long = pd.concat([n_games, crew_rate, imp_rate], axis=1).reset_index()
    long.to_csv(RESULTS_PATH / f"{DATE}_winrate_matchup_x_config.csv",
                index=False)
    log("  saved winrate_matchup_x_config.csv")

    # 2b) win rate per matchup pooled across configs
    grp = df_wins.groupby(["matchup", "crew_model", "imp_model"],
                          observed=True, dropna=False)
    pooled = pd.DataFrame({
        "n_games": grp.size(),
        "crew_win_rate": grp["winner_faction"].apply(
            lambda s: (s == "Crewmate").mean()),
        "imp_win_rate": grp["winner_faction"].apply(
            lambda s: (s == "Impostor").mean()),
    }).reset_index()
    pooled["crew_family"] = pooled["crew_model"].map(fam)
    pooled["imp_family"] = pooled["imp_model"].map(fam)
    pooled["crew_size"] = pooled["crew_model"].map(size)
    pooled["imp_size"] = pooled["imp_model"].map(size)
    pooled["family_relation"] = np.where(
        pooled["crew_family"] == pooled["imp_family"],
        "within_family", "across_family")
    pooled.to_csv(RESULTS_PATH / f"{DATE}_winrate_matchup_pooled.csv",
                  index=False)
    log("  saved winrate_matchup_pooled.csv")

    # 2c) crew x imp win-rate matrix
    matrix = (pooled
              .pivot_table(index="crew_model", columns="imp_model",
                           values="crew_win_rate", aggfunc="mean"))
    matrix.to_csv(RESULTS_PATH / f"{DATE}_winrate_matrix_crew_x_imp.csv")
    log(f"  saved winrate_matrix_crew_x_imp.csv  "
        f"({matrix.shape[0]}x{matrix.shape[1]}, NaN where pair not run)")

    # 2d) win-category share per matchup
    cat_counts = (df_wins.groupby(["matchup", "winner_category"],
                                  observed=True, dropna=False).size()
                  .unstack("winner_category", fill_value=0)
                  .reindex(columns=CATEGORY_ORDER, fill_value=0))
    cat_share = cat_counts.div(cat_counts.sum(axis=1), axis=0).fillna(0.0)
    cat_share.reset_index().to_csv(
        RESULTS_PATH / f"{DATE}_win_categories_matchup.csv", index=False)
    log("  saved win_categories_matchup.csv")


# -------------------------------------------------------------------------
# Stage 3 — Per-role metric tables (matchup-level + model-pooled)
# -------------------------------------------------------------------------
banner("STAGE 3 — Per-role ToM metric tables")


def build_role_x_matchup(df, role_id, metrics):
    """Rows = metrics; columns = (crew_model, imp_model) ordered.

    For role 'Crewmate' we vary the crew_model first (it's the actor); for
    role 'Impostor' we vary imp_model first.
    """
    sub = df[df["identity"] == role_id]
    if sub.empty:
        return pd.DataFrame()
    if role_id == 0:  # crewmate role
        actor_col, opp_col = "crew_model", "imp_model"
    else:
        actor_col, opp_col = "imp_model", "crew_model"
    pairs = (sub.groupby([actor_col, opp_col], dropna=False)
             .size().reset_index().drop(columns=0))
    pairs = pairs.sort_values([actor_col, opp_col])
    cols = [(r[actor_col], r[opp_col]) for _, r in pairs.iterrows()]
    cell = {c: {} for c in cols}
    ns = {}
    for actor, opp in cols:
        s = sub[(sub[actor_col] == actor) & (sub[opp_col] == opp)]
        ns[(actor, opp)] = len(s)
        for met in metrics:
            cell[(actor, opp)][met] = _cell(s[met]) if met in s.columns else "—"
    data = {c: [cell[c][m] for m in metrics] for c in cols}
    out = pd.DataFrame(data, index=metrics)
    out.columns = pd.MultiIndex.from_tuples(
        cols, names=[f"{actor_col}", f"{opp_col}"])
    out.index.name = "metric"
    out.loc["n (rows)"] = [str(ns[c]) for c in cols]
    return out


def build_role_pooled_by_model(df, role_id, metrics):
    """Rows = metrics, cols = each model that appeared as the actor for this role,
    averaging over all opponents (pooled cross-play view)."""
    sub = df[df["identity"] == role_id]
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame()
    actor_col = "crew_model" if role_id == 0 else "imp_model"
    models = sorted(sub[actor_col].dropna().unique())
    fmt = {m: [] for m in models}
    num = {m: [] for m in models}
    ns = {}
    for m in models:
        s = sub[sub[actor_col] == m]
        ns[m] = len(s)
        for met in metrics:
            if met not in s.columns:
                fmt[m].append("—"); num[m].append(np.nan); continue
            fmt[m].append(_cell(s[met]))
            num[m].append(float(s[met].mean()) if s[met].notna().any() else np.nan)
    tbl = pd.DataFrame(fmt, index=metrics); tbl.index.name = "metric"
    tbl.loc["n (rows)"] = [str(ns[m]) for m in models]
    means = pd.DataFrame(num, index=metrics)
    return tbl, means


crew_x_matchup = build_role_x_matchup(df_metrics, 0, CREWMATE_METRICS)
imp_x_matchup = build_role_x_matchup(df_metrics, 1, IMPOSTOR_METRICS)
if not crew_x_matchup.empty:
    crew_x_matchup.to_csv(RESULTS_PATH / f"{DATE}_crewmate_x_matchup.csv")
    log("  saved crewmate_x_matchup.csv")
if not imp_x_matchup.empty:
    imp_x_matchup.to_csv(RESULTS_PATH / f"{DATE}_impostor_x_matchup.csv")
    log("  saved impostor_x_matchup.csv")

crew_pooled_tbl, crew_pooled_num = build_role_pooled_by_model(
    df_metrics, 0, CREWMATE_METRICS)
imp_pooled_tbl, imp_pooled_num = build_role_pooled_by_model(
    df_metrics, 1, IMPOSTOR_METRICS)
if not crew_pooled_tbl.empty:
    crew_pooled_tbl.to_csv(RESULTS_PATH / f"{DATE}_crewmate_x_model_pooled.csv")
    crew_pooled_num.to_csv(
        RESULTS_PATH / f"{DATE}_crewmate_x_model_pooled_numeric.csv")
    log("  saved crewmate_x_model_pooled (+ numeric)")
if not imp_pooled_tbl.empty:
    imp_pooled_tbl.to_csv(RESULTS_PATH / f"{DATE}_impostor_x_model_pooled.csv")
    imp_pooled_num.to_csv(
        RESULTS_PATH / f"{DATE}_impostor_x_model_pooled_numeric.csv")
    log("  saved impostor_x_model_pooled (+ numeric)")


# -------------------------------------------------------------------------
# Stage 4 — Size-bucket triangle round-robins
# -------------------------------------------------------------------------
banner("STAGE 4 — Size-bucket triangle round-robins")

TRIANGLES = {
    "small_3-4B":      ["qwen3-4b", "llama-3.2-3b-instruct", "gemma-4-E4B-it"],
    "medium_8B":       ["llama-3.1-8b", "qwen3-8B",
                        "deepseek-r1-distill-llama-8B"],
    "large_26-32B":    ["qwen3-32b", "gemma-4-26B-A4B-it",
                        "gemma-4-31b", "deepseek-r1-distill-qwen-32b"],
}


def write_triangle(name: str, models: list[str]):
    log(f"\n  triangle: {name}  models={models}")
    sub_df = df_metrics[
        df_metrics["crew_model"].isin(models) &
        df_metrics["imp_model"].isin(models)
    ]
    sub_wins = df_wins[
        df_wins["crew_model"].isin(models) &
        df_wins["imp_model"].isin(models)
    ] if not df_wins.empty else pd.DataFrame()
    if sub_df.empty and sub_wins.empty:
        log(f"    [skip] no matchups in data for {name}")
        return None, None, None

    pair_set = sorted({(c, i) for c, i in zip(
        sub_df["crew_model"].fillna(""), sub_df["imp_model"].fillna(""))
        if c and i})
    log(f"    matchups present ({len(pair_set)}):")
    for c, i in pair_set:
        log(f"      crew={c:30s}  imp={i}")

    if not sub_wins.empty:
        wm = (sub_wins.groupby(["crew_model", "imp_model"], observed=True)
              ["winner_faction"].apply(lambda s: (s == "Crewmate").mean())
              .unstack("imp_model"))
        wm = wm.reindex(index=models, columns=models)
        wm.to_csv(RESULTS_PATH / f"{DATE}_triangle_{name}_winrate_matrix.csv")
        log(f"    saved triangle_{name}_winrate_matrix.csv")

    crew_tbl = build_role_x_matchup(sub_df, 0, CREWMATE_METRICS)
    imp_tbl = build_role_x_matchup(sub_df, 1, IMPOSTOR_METRICS)
    if not crew_tbl.empty:
        crew_tbl.to_csv(
            RESULTS_PATH / f"{DATE}_triangle_{name}_crewmate_x_matchup.csv")
    if not imp_tbl.empty:
        imp_tbl.to_csv(
            RESULTS_PATH / f"{DATE}_triangle_{name}_impostor_x_matchup.csv")

    # pooled-by-model (averaged over opponents within the triangle)
    crew_pool, crew_num = build_role_pooled_by_model(sub_df, 0, CREWMATE_METRICS)
    imp_pool, imp_num = build_role_pooled_by_model(sub_df, 1, IMPOSTOR_METRICS)
    if not crew_pool.empty:
        crew_pool.to_csv(
            RESULTS_PATH / f"{DATE}_triangle_{name}_crewmate_pooled.csv")
        crew_num.to_csv(
            RESULTS_PATH / f"{DATE}_triangle_{name}_crewmate_pooled_numeric.csv")
    if not imp_pool.empty:
        imp_pool.to_csv(
            RESULTS_PATH / f"{DATE}_triangle_{name}_impostor_pooled.csv")
        imp_num.to_csv(
            RESULTS_PATH / f"{DATE}_triangle_{name}_impostor_pooled_numeric.csv")
    return sub_df, crew_num, imp_num


triangle_data = {}
for name, models in TRIANGLES.items():
    triangle_data[name] = write_triangle(name, models)


# -------------------------------------------------------------------------
# Stage 5 — Within-family vs across-family comparison
# -------------------------------------------------------------------------
banner("STAGE 5 — Within-family vs across-family")

if not df_wins.empty:
    wins = df_wins.dropna(subset=["crew_model", "imp_model"]).copy()
    wins["crew_family"] = wins["crew_model"].map(fam)
    wins["imp_family"] = wins["imp_model"].map(fam)
    wins["family_relation"] = np.where(
        wins["crew_family"] == wins["imp_family"],
        "within_family", "across_family")
    wins["crew_size"] = wins["crew_model"].map(size)
    wins["imp_size"] = wins["imp_model"].map(size)
    wins["size_relation"] = np.where(
        wins["crew_size"] == wins["imp_size"],
        "same_size", "diff_size")

    summary = []
    for relkey, sub in wins.groupby("family_relation"):
        summary.append({
            "partition": "family",
            "value": relkey,
            "n_games": len(sub),
            "n_matchups": sub["matchup"].nunique(),
            "crew_win_rate": (sub["winner_faction"] == "Crewmate").mean(),
            "imp_win_rate": (sub["winner_faction"] == "Impostor").mean(),
        })
    for sk, sub in wins.groupby("size_relation"):
        summary.append({
            "partition": "size",
            "value": sk,
            "n_games": len(sub),
            "n_matchups": sub["matchup"].nunique(),
            "crew_win_rate": (sub["winner_faction"] == "Crewmate").mean(),
            "imp_win_rate": (sub["winner_faction"] == "Impostor").mean(),
        })
    pd.DataFrame(summary).to_csv(
        RESULTS_PATH / f"{DATE}_within_vs_across_family_winrates.csv",
        index=False)
    log("  saved within_vs_across_family_winrates.csv")

    # per-role metrics by family relation
    df_meta = df_metrics.dropna(subset=["crew_model", "imp_model"]).copy()
    rows = []
    for role_id, role_name, mets, actor in [
        (0, "Crewmate", CREWMATE_METRICS, "crew"),
        (1, "Impostor", IMPOSTOR_METRICS, "imp"),
    ]:
        sub = df_meta[df_meta["identity"] == role_id]
        for relkey, s in sub.groupby("family_relation"):
            row = {"role": role_name, "family_relation": relkey,
                   "n_rows": len(s)}
            for m in mets:
                row[m] = float(s[m].mean()) if (m in s.columns and
                                                s[m].notna().any()) else np.nan
            rows.append(row)
    pd.DataFrame(rows).to_csv(
        RESULTS_PATH / f"{DATE}_within_vs_across_family_metrics.csv",
        index=False)
    log("  saved within_vs_across_family_metrics.csv")


# -------------------------------------------------------------------------
# Stage 6 — Cross-play vs self-play delta (per model, per role)
# -------------------------------------------------------------------------
banner("STAGE 6 — Cross-play vs self-play delta")

selfplay_csv = SELFPLAY_RESULTS / f"{SELFPLAY_DATE}_tom_metrics_sweep.csv"
if not selfplay_csv.exists():
    log(f"  [skip] self-play sweep CSV not found: {selfplay_csv}")
else:
    sp = pd.read_csv(selfplay_csv)
    sp["model"] = sp["model"].map(canonical_model)
    sp["role"] = sp["identity"].map({0: "Crewmate", 1: "Impostor"})
    log(f"  self-play loaded: {len(sp)} rows | "
        f"{sp['model'].nunique()} models")

    cp = df_metrics.copy()
    cp["actor_crew"] = cp["crew_model"]
    cp["actor_imp"] = cp["imp_model"]

    rows = []
    for role_id, role_name, mets, actor_col in [
        (0, "Crewmate", CREWMATE_METRICS, "actor_crew"),
        (1, "Impostor", IMPOSTOR_METRICS, "actor_imp"),
    ]:
        sp_role = sp[sp["identity"] == role_id]
        cp_role = cp[cp["identity"] == role_id]
        models = sorted(set(cp_role[actor_col].dropna().unique()) &
                        set(sp_role["model"].unique()))
        for m in models:
            sp_m = sp_role[sp_role["model"] == m]
            cp_m = cp_role[cp_role[actor_col] == m]
            row = {"model": m, "role": role_name,
                   "n_selfplay": len(sp_m), "n_crossplay": len(cp_m)}
            for met in mets:
                if met not in sp_m.columns or met not in cp_m.columns:
                    continue
                sp_v = sp_m[met].mean() if sp_m[met].notna().any() else np.nan
                cp_v = cp_m[met].mean() if cp_m[met].notna().any() else np.nan
                row[f"{met}_selfplay"] = sp_v
                row[f"{met}_crossplay"] = cp_v
                row[f"{met}_delta"] = (cp_v - sp_v
                                       if pd.notna(sp_v) and pd.notna(cp_v)
                                       else np.nan)
            rows.append(row)
    pd.DataFrame(rows).to_csv(
        RESULTS_PATH / f"{DATE}_crossplay_vs_selfplay_metrics.csv",
        index=False)
    log("  saved crossplay_vs_selfplay_metrics.csv")


# -------------------------------------------------------------------------
# Stage 7 — Pareto radar plots for the triangles
# -------------------------------------------------------------------------
banner("STAGE 7 — Triangle radar plots (within-triangle Pareto)")

CREW_RADAR_AXES = [
    ("detection_accuracy",  "Detection\n(1-MSE)",   "invert"),
    ("objective_viability", "Objective\nviability", "raw"),
    ("alibi_grounding",     "Alibi\ngrounding",     "raw"),
    ("alibi_corroboration", "Alibi\ncorroboration", "raw"),
    ("faction_consensus",   "Faction\nconsensus",   "raw"),
    ("social_influence",    "Social\ninfluence",    "raw"),
]
IMP_RADAR_AXES = [
    ("deceptive_efficacy",  "Deceptive\nefficacy",          "raw"),
    ("objective_viability", "Objective\nviability",         "raw"),
    ("alibi_grounding",     "Alibi opacity\n(1-grounding)", "invert"),
    ("belief_volatility",   "Stability\n(1-volatility)",    "invert"),
    ("faction_consensus",   "Faction\nconsensus",           "raw"),
    ("social_influence",    "Social\ninfluence",            "raw"),
]


def _aggregate_for_radar(df, axes_spec, role_id, models, actor_col):
    sub = df[(df["identity"] == role_id) & (df[actor_col].isin(models))]
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for m, g in sub.groupby(actor_col):
        row = {"model": m, "n": len(g)}
        for col, _, mode in axes_spec:
            v = g[col].mean() if col in g.columns and g[col].notna().any() else np.nan
            if mode == "invert" and pd.notna(v):
                v = 1.0 - v
            row[col] = v
        rows.append(row)
    out = pd.DataFrame(rows)
    out = pd.concat(
        [out[out["model"] == m] for m in models if (out["model"] == m).any()],
        ignore_index=True,
    )
    return out


def _pareto_set_max(M):
    out = []
    for i in range(M.shape[0]):
        dominated = False
        for j in range(M.shape[0]):
            if i == j:
                continue
            if np.all(M[j] >= M[i]) and np.any(M[j] > M[i]):
                dominated = True
                break
        if not dominated:
            out.append(i)
    return out


def _normalize(mat, cols, margin=0.05):
    normed = mat.copy()
    for c in cols:
        vals = normed[c].dropna()
        if len(vals) == 0:
            continue
        lo, hi = vals.min(), vals.max()
        span = hi - lo
        normed[c] = 0.5 if span < 1e-9 else margin + (1.0 - 2 * margin) * (normed[c] - lo) / span
    return normed


def _plot_radar(ax, mat, axes_spec, title, pareto_idx):
    cols = [c for c, _, _ in axes_spec]
    labels = [lbl for _, lbl, _ in axes_spec]
    n_axes = len(cols)
    angles = np.linspace(0, 2 * math.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(["", "", ""])
    cmap = plt.colormaps.get_cmap("tab10")
    for i, (_, row) in enumerate(mat.iterrows()):
        vals = [row[c] if pd.notna(row[c]) else 0 for c in cols]
        vals += vals[:1]
        is_pareto = i in pareto_idx
        color = cmap(i % 10)
        ax.plot(angles, vals, color=color, lw=2 if is_pareto else 1.2,
                marker="o" if is_pareto else None, markersize=5,
                label=f"{row['model']}{' *' if is_pareto else ''}")
        ax.fill(angles, vals, color=color, alpha=0.10 if is_pareto else 0.05)
    ax.set_title(title, fontsize=11, pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.05),
              fontsize=8, frameon=False)


def render_triangle_radar(name, models):
    sub = df_metrics[
        df_metrics["crew_model"].isin(models) &
        df_metrics["imp_model"].isin(models)
    ]
    if sub.empty:
        log(f"  [skip radar] no rows for {name}")
        return
    crew_mat = _aggregate_for_radar(sub, CREW_RADAR_AXES, 0, models,
                                    "crew_model")
    imp_mat = _aggregate_for_radar(sub, IMP_RADAR_AXES, 1, models,
                                   "imp_model")
    if crew_mat.empty and imp_mat.empty:
        log(f"  [skip radar] empty aggregation for {name}")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 6),
                             subplot_kw=dict(polar=True))

    if not crew_mat.empty:
        crew_norm = _normalize(crew_mat, [c for c, _, _ in CREW_RADAR_AXES])
        crew_M = crew_norm[[c for c, _, _ in CREW_RADAR_AXES]].fillna(0).values
        crew_par = _pareto_set_max(crew_M)
        _plot_radar(axes[0], crew_norm, CREW_RADAR_AXES,
                    f"Crewmate role — {name}", crew_par)
        crew_mat.to_csv(
            RESULTS_PATH / f"{DATE}_radar_{name}_crewmate_raw.csv",
            index=False)
    else:
        axes[0].set_visible(False)

    if not imp_mat.empty:
        imp_norm = _normalize(imp_mat, [c for c, _, _ in IMP_RADAR_AXES])
        imp_M = imp_norm[[c for c, _, _ in IMP_RADAR_AXES]].fillna(0).values
        imp_par = _pareto_set_max(imp_M)
        _plot_radar(axes[1], imp_norm, IMP_RADAR_AXES,
                    f"Impostor role — {name}", imp_par)
        imp_mat.to_csv(
            RESULTS_PATH / f"{DATE}_radar_{name}_impostor_raw.csv",
            index=False)
    else:
        axes[1].set_visible(False)

    fig.suptitle(f"Cross-play radar — {name} (within-triangle Pareto markers)",
                 fontsize=13)
    fig.tight_layout()
    out = RESULTS_PATH / f"{DATE}_radar_{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  saved {out.name}")


for name, models in TRIANGLES.items():
    render_triangle_radar(name, models)


# -------------------------------------------------------------------------
# Stage 8 — Why ELO is the wrong evaluation lens
# -------------------------------------------------------------------------
banner("STAGE 8 — ELO-vs-skill demonstration")

# Build "ELO-ish" rating: invert win rate to logits per role.
# We compute *separate* crew and imp ratings (already an admission of
# role asymmetry) so the comparison is as charitable to ELO as possible.

def role_ratings_from_winrates(wins: pd.DataFrame) -> pd.DataFrame:
    """Bradley-Terry-style log-odds rating per model per role.

    For each model in the crewmate role, log-odds(crew win rate over all
    its opponents). Symmetrically for impostor. Both are clipped to keep
    log-odds finite.
    """
    if wins.empty:
        return pd.DataFrame()
    rows = []
    for actor_col, role_name, target in [
        ("crew_model", "Crewmate", "Crewmate"),
        ("imp_model", "Impostor", "Impostor"),
    ]:
        sub = wins.dropna(subset=[actor_col])
        for m, g in sub.groupby(actor_col):
            wr = (g["winner_faction"] == target).mean()
            wr_clip = min(max(wr, 1e-3), 1 - 1e-3)
            rows.append({
                "model": m, "role": role_name,
                "n_games": len(g),
                "win_rate": wr,
                "log_odds_rating": math.log(wr_clip / (1 - wr_clip)),
            })
    return pd.DataFrame(rows)


elo_df = role_ratings_from_winrates(df_wins) if not df_wins.empty else pd.DataFrame()

# Pull per-model per-role metric means (already pooled) for correlation
crew_means = (df_metrics[df_metrics["identity"] == 0]
              .groupby("crew_model")[CREWMATE_METRICS].mean()
              .reset_index().rename(columns={"crew_model": "model"}))
crew_means["role"] = "Crewmate"
imp_means = (df_metrics[df_metrics["identity"] == 1]
             .groupby("imp_model")[IMPOSTOR_METRICS].mean()
             .reset_index().rename(columns={"imp_model": "model"}))
imp_means["role"] = "Impostor"

if not elo_df.empty:
    crew_join = elo_df[elo_df["role"] == "Crewmate"].merge(
        crew_means, on=["model", "role"], how="left")
    imp_join = elo_df[elo_df["role"] == "Impostor"].merge(
        imp_means, on=["model", "role"], how="left")

    # 8a) Save per-model per-role rating + skill-metric table
    pd.concat([crew_join, imp_join], ignore_index=True).to_csv(
        RESULTS_PATH / f"{DATE}_elo_vs_skill_per_model.csv", index=False)
    log("  saved elo_vs_skill_per_model.csv")

    # 8b) Correlate log-odds rating against skill metrics
    corr_rows = []
    for role_name, joined, skill_metric, all_metrics in [
        ("Crewmate", crew_join, "detection_accuracy", CREWMATE_METRICS),
        ("Impostor", imp_join, "deceptive_efficacy", IMPOSTOR_METRICS),
    ]:
        for m in all_metrics:
            if m not in joined.columns:
                continue
            d = joined[["log_odds_rating", m]].dropna()
            if len(d) < 3:
                continue
            r_p = d.corr(method="pearson").iloc[0, 1]
            r_s = d.corr(method="spearman").iloc[0, 1]
            corr_rows.append({
                "role": role_name,
                "metric": m,
                "is_primary_skill": (m == skill_metric),
                "n_models": len(d),
                "pearson_r": r_p,
                "spearman_r": r_s,
            })
    corr_df = pd.DataFrame(corr_rows)
    if not corr_df.empty:
        corr_df.to_csv(RESULTS_PATH / f"{DATE}_elo_vs_skill_correlations.csv",
                       index=False)
        log("  saved elo_vs_skill_correlations.csv")

    # 8c) Detect non-transitivity (rock-paper-scissors cycles) in the
    #     pooled crew x imp matrix — but it's directional (A's crew vs B's imp).
    #     We instead check: does "model X (as crew) wins more than 50% vs Y (as imp)"
    #     remain consistent when we swap roles? Cycles in the directed
    #     win graph would invalidate any single rating.
    if not df_wins.empty:
        wins = df_wins.dropna(subset=["crew_model", "imp_model"])
        # For each unordered pair {A, B} with both directions present, check
        # whether A beats B by symmetric "average crew win rate" — and flag
        # if the role-swap flips the outcome.
        pair_dir = (wins.groupby(["crew_model", "imp_model"], observed=True)
                    ["winner_faction"]
                    .apply(lambda s: (s == "Crewmate").mean())
                    .reset_index().rename(columns={"winner_faction": "crew_wr"}))
        pair_dir["unordered_pair"] = pair_dir.apply(
            lambda r: tuple(sorted([r["crew_model"], r["imp_model"]])), axis=1)
        flips = []
        for pair, g in pair_dir.groupby("unordered_pair"):
            if len(g) < 2:
                continue
            a, b = pair
            # direction 1: a as crew vs b as imp
            d1 = g[(g["crew_model"] == a) & (g["imp_model"] == b)]
            d2 = g[(g["crew_model"] == b) & (g["imp_model"] == a)]
            if d1.empty or d2.empty:
                continue
            # If a "deserves" higher crew rating than b, we'd expect:
            #   d1.crew_wr (a-crew vs b-imp) > 0.5 OR d2.crew_wr (b-crew vs a-imp) < 0.5
            # In the role-symmetric ELO-friendly case, these two would agree.
            # We flag a flip when both d1 and d2 favor the same *role*
            # (i.e. crewmate wins both directions — meaning who-crews dominates,
            # not who-the-model-is).
            r1 = float(d1["crew_wr"].iloc[0])
            r2 = float(d2["crew_wr"].iloc[0])
            crew_advantage_1 = r1 > 0.5
            crew_advantage_2 = r2 > 0.5
            flips.append({
                "model_A": a, "model_B": b,
                "crew=A_imp=B_crew_winrate": r1,
                "crew=B_imp=A_crew_winrate": r2,
                "crew_advantage_when_A_crews": crew_advantage_1,
                "crew_advantage_when_B_crews": crew_advantage_2,
                "role_dominates_identity": crew_advantage_1 == crew_advantage_2,
            })
        pd.DataFrame(flips).to_csv(
            RESULTS_PATH / f"{DATE}_elo_role_vs_identity_flips.csv",
            index=False)
        log("  saved elo_role_vs_identity_flips.csv")

# Final summary
banner("DONE")
log(f"Total wall: {time.time() - _t_start:.1f}s")
log(f"Outputs in {RESULTS_PATH}")
_LOG_FH.close()
