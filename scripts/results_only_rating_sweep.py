"""Sweep the RATING-side knobs (margin cap, ridge strength, memory horizons).

Each config means a full weekly refit of every rating system, so we score on the
validation window (2006-2017) only and keep the test seasons untouched.
"""
from __future__ import annotations
import os, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.cfbrank import metrics
from src.cfbrank.data import build_games, is_fbs
from src.cfbrank.elo import run_elo
from scripts.results_only_model import (form_pass, rating_pass, season_refit, mk_lr,
                                        mk_gbm, logit, RES)

VAL = (2006, 2017)
ORIG = ["massey_short", "massey_mid", "massey_long", "offdef_margin", "offdef_total",
        "bt_diff", "pr_diff", "pyth_diff", "sos_diff", "home_ind",
        "elo_diff", "elo_fast_diff", "elo_slow_diff", "wp_diff", "avgmargin_diff",
        "form3_diff", "gp_min", "gp_diff", "rest_diff", "neutral", "week"]
LR_FEATS = ["massey_mid", "offdef_margin", "bt_diff", "elo_diff", "pr_diff", "home_ind"]

CFGS = {
    "base":      dict(),
    "cap21":     dict(margin_cap=21.0),
    "cap35":     dict(margin_cap=35.0),
    "lam8":      dict(lam_massey=8.0),
    "lam2":      dict(lam_massey=2.0),
    "decay_lo":  dict(decays=(0.15, 0.35, 0.70)),
    "decay_hi":  dict(decays=(0.30, 0.55, 0.90)),
    "cap35_lam2": dict(margin_cap=35.0, lam_massey=2.0),
    "cap35_dhi":  dict(margin_cap=35.0, decays=(0.30, 0.55, 0.90)),
}

g = build_games(); g = g[g.season <= 2025].reset_index(drop=True)
pool = g[(g.home_division == "fbs") | (g.away_division == "fbs")].reset_index(drop=True)
eval_ids = set(g[is_fbs(g)].game_id)
e = run_elo(pool, k=40, home_field=50, mov=True, preseason_regress=0.15)
ef = run_elo(pool, k=65, home_field=50, mov=True, preseason_regress=0.15)
es = run_elo(pool, k=20, home_field=50, mov=True, preseason_regress=0.15)
F = form_pass(pool)
skel = g[g.game_id.isin(eval_ids)][["game_id", "season", "week", "home_team",
                                    "away_team", "home_win"]].copy()
skel["elo_prob"] = skel.game_id.map(e["elo_prob"])
skel["elo_diff"] = skel.game_id.map(e["elo_home_pre"] - e["elo_away_pre"])
skel["elo_fast_diff"] = skel.game_id.map(ef["elo_home_pre"] - ef["elo_away_pre"])
skel["elo_slow_diff"] = skel.game_id.map(es["elo_home_pre"] - es["elo_away_pre"])
skel = skel.merge(F, left_on="game_id", right_index=True, how="left")

rows = []
for name, cfg in CFGS.items():
    t = time.time()
    R = rating_pass(pool, eval_ids, **cfg)
    d = skel.merge(R, left_on="game_id", right_index=True, how="left")
    d["gbm"] = season_refit(d, ORIG, mk_gbm)
    d["lr"] = season_refit(d, LR_FEATS, mk_lr)
    d["bl"] = 1 / (1 + np.exp(-(0.75 * logit(d.gbm) + 0.25 * logit(d.lr))))
    v = d[(d.season >= VAL[0]) & (d.season <= VAL[1])]
    m = metrics.evaluate(v.home_win.values, v.bl.values)
    rows.append(dict(cfg=name, val_ll=m["logloss"], val_acc=m["acc"], secs=round(time.time() - t)))
    print(f"  {name:10s} val_ll {m['logloss']:.4f} acc {m['acc']:.4f}  [{time.time()-t:.0f}s]", flush=True)
r = pd.DataFrame(rows).sort_values("val_ll")
r.to_csv(os.path.join(RES, "results_only_rating_sweep.csv"), index=False)
print("\n" + r.to_string(index=False))
print(f"\nBEST rating config: {r.iloc[0].cfg}")
