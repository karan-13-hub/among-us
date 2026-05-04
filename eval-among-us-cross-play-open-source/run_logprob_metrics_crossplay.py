#!/usr/bin/env python3
"""Compute logprob-derived ToM metrics for every cross-play experiment.

Cross-play exp dirs already store top-logprobs alongside the verbalized
belief / vote distributions in epistemic-states.jsonl.  We've been
running the analysis on the verbalized fields only; this script
re-runs `process_experiment(..., source="logprobs")` over the same
exp dirs, writes per-exp CSVs to <matchup>/results/, and concatenates
into a sweep-wide CSV so the verbalized vs logprob disparity figure
becomes possible.

Output:
  <matchup>/results/<exp_name>_tom_metrics_logprob.csv  (per exp)
  results/2026-05-03_tom_metrics_sweep_logprob_crossplay.csv (combined)
"""
import importlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/yjangir1/scratchhbharad2/users/yjangir1/karan")
SWEEP_ROOT = ROOT / "eval-among-us-crossplay"
OUT_ROOT = ROOT / "eval-among-us-crossplay-open-source"
RESULTS_PATH = OUT_ROOT / "results"
DATE = "2026-05-03"

EXCLUDE_MATCHUPS = {
    "eval-cross-play-among-us-deepseek-qwen32b-vs-qwen3_32b",  # in flight
}

AMONG_US_PATH = ROOT / "among-us"
sys.path.insert(0, str(AMONG_US_PATH / "among-agents"))
sys.path.insert(0, str(AMONG_US_PATH))
sys.path.insert(0, str(AMONG_US_PATH / "evaluations"))

print("Importing metrics_calculator ...")
_mc = importlib.import_module("evaluations.metrics_calculator")
process_experiment = _mc.process_experiment


def parse_compact_log_models(path: Path) -> dict:
    if not path.exists():
        return {}
    txt = path.read_text()
    out: dict = {}
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
        m = os.path.basename(mp) if isinstance(mp, str) else mp
        slot = out.setdefault(g, {"crew": set(), "imp": set()})
        if ident == "Crewmate":
            slot["crew"].add(m)
        elif ident == "Impostor":
            slot["imp"].add(m)
    return out


def canonical_model(name):
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


def parse_config(p):
    text = (Path(p) / "experiment-details.txt").read_text()
    m = re.search(r"'num_players':\s*(\d+),\s*'num_impostors':\s*(\d+)", text)
    if not m:
        return None
    n_players, n_imp = int(m.group(1)), int(m.group(2))
    return f"{n_players - n_imp}C_{n_imp}I"


def attach_metadata(df, exp_dir, exp_name, cfg):
    if df is None or df.empty:
        return df
    gm = parse_compact_log_models(Path(exp_dir) / "agent-logs-compact.json")
    if not gm:
        return None
    crew_col, imp_col, mu_col = [], [], []
    for g in df["game"].astype(str):
        slot = gm.get(g, {"crew": set(), "imp": set()})
        cm = sorted(slot["crew"])[0] if len(slot["crew"]) == 1 else None
        im = sorted(slot["imp"])[0] if len(slot["imp"]) == 1 else None
        cm = canonical_model(cm); im = canonical_model(im)
        crew_col.append(cm); imp_col.append(im)
        mu_col.append(f"crew={cm}__imp={im}" if cm and im else None)
    df = df.copy()
    df["crew_model"] = crew_col
    df["imp_model"] = imp_col
    df["matchup"] = mu_col
    df["config"] = cfg
    df["experiment"] = exp_name
    return df


print(f"Sweep root: {SWEEP_ROOT}")
all_rows = []
n_exps = 0
n_done = 0
n_skipped = 0
t0 = time.time()

for mdir in sorted(SWEEP_ROOT.iterdir()):
    if not mdir.is_dir() or mdir.name in EXCLUDE_MATCHUPS:
        continue
    print(f"\n[matchup] {mdir.name}")
    for edir in sorted(mdir.glob("*_exp_*")):
        n_exps += 1
        exp_name = edir.name
        cached = mdir / "results" / f"{exp_name}_tom_metrics_logprob.csv"
        if cached.exists():
            try:
                df = pd.read_csv(cached)
                df = attach_metadata(df, str(edir), exp_name,
                                     parse_config(edir))
                if df is not None and not df.empty:
                    df["matchup_dir"] = mdir.name
                    all_rows.append(df)
                    n_done += 1
                    continue
            except Exception:
                pass

        cfg = parse_config(edir)
        if not cfg:
            print(f"  SKIP {exp_name} (unparseable details)")
            n_skipped += 1
            continue
        ts = time.time()
        try:
            df = process_experiment(str(edir), source="logprobs")
        except Exception as ex:
            print(f"  FAIL {exp_name}: {ex}")
            n_skipped += 1
            continue
        if df is None or df.empty:
            print(f"  EMPTY {exp_name} (no logprob data)")
            n_skipped += 1
            continue
        df = attach_metadata(df, str(edir), exp_name, cfg)
        if df is None or df.empty:
            print(f"  meta-fail {exp_name}")
            n_skipped += 1
            continue
        cached.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cached, index=False)
        df["matchup_dir"] = mdir.name
        all_rows.append(df)
        n_done += 1
        print(f"  OK  {exp_name}  cfg={cfg}  rows={len(df):4d}  "
              f"({time.time() - ts:.1f}s)")

print(f"\n=== summary ===")
print(f"experiments: {n_exps}   logprob-OK: {n_done}   skipped: {n_skipped}")
print(f"wall: {time.time() - t0:.1f}s")

if all_rows:
    big = pd.concat(all_rows, ignore_index=True)
    big["crew_model"] = big["crew_model"].map(canonical_model)
    big["imp_model"] = big["imp_model"].map(canonical_model)
    big["matchup"] = ("crew=" + big["crew_model"].astype(str)
                      + "__imp=" + big["imp_model"].astype(str))
    big["role"] = big["identity"].map({0: "Crewmate", 1: "Impostor"})
    out = RESULTS_PATH / f"{DATE}_tom_metrics_sweep_logprob_crossplay.csv"
    big.to_csv(out, index=False)
    print(f"saved {out}  ({len(big)} rows, "
          f"{big['matchup'].nunique()} matchups, "
          f"{big['crew_model'].nunique()} crew-models, "
          f"{big['imp_model'].nunique()} imp-models)")
