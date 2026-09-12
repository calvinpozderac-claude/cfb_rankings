"""Follow-up experiments on top of experiments.py output:

  * elo_dynk        Elo with a larger K for each team's first 4 games (40 -> 30)
  * gbm_T4_margin   GBM *regression* on point margin with the T4 features, converted
                    to P(home win) by a walk-forward logistic on predicted margin
                    (uses more information per game than the win/loss label)
  * stack_final     stacker over the strongest results-only signals
  * stack_final_ol  the same plus the opening line (market-informed)

Appends to results/experiments_ladder.csv and experiments_by_week.csv.
"""
from __future__ import annotations
import os, sys
from collections import defaultdict
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.cfbrank import metrics
from src.cfbrank.data import build_games, is_fbs
from src.cfbrank.elo import expected, INIT
from src.cfbrank.mlmodels import ML_FEATURES
sys.path.insert(0, os.path.dirname(__file__))
from experiments import (ev, logit, market_tests, season_stack, EVAL_LO, EVAL_HI, FEAT_START, RES)

base = pd.read_csv(os.path.join(RES, "experiments_predictions.csv.gz"))
g = build_games(); g = g[g.season <= 2025].reset_index(drop=True)

# ---- dynamic-K Elo ----
def elo_dynk(k_early=40, k_late=30, switch=4, hf=50, reg=0.15):
    rating = defaultdict(lambda: INIT); gp = defaultdict(int); cur = None; out = {}; diff = {}
    for r in g.itertuples(index=False):
        if cur is None: cur = r.season
        elif r.season != cur:
            for t in list(rating): rating[t] = INIT + (1 - reg) * (rating[t] - INIT)
            gp.clear(); cur = r.season
        h, a = r.home_team, r.away_team; rh, ra = rating[h], rating[a]
        p = expected((rh + (0 if r.neutral_site else hf)) - ra); out[r.game_id] = p; diff[r.game_id] = rh - ra
        k = k_early if min(gp[h], gp[a]) < switch else k_late
        m = abs(r.margin); wd = (rh - ra) if r.home_win else (ra - rh)
        d = k * np.log(m + 1) * (2.2 / (0.001 * wd + 2.2)) * (r.home_win - p)
        rating[h] = rh + d; rating[a] = ra - d; gp[h] += 1; gp[a] += 1
    return out, diff
o, d = elo_dynk(); base["elo_dynk"] = base.game_id.map(o); base["elo_dynk_diff"] = base.game_id.map(d)

# ---- margin-regression GBM on T4 features ----
T4 = ML_FEATURES + ["week", "ridge_margin_pred", "pr_keep_logit", "bt_logit"] + \
     [f"ridge_{t}_pred" for t in ["ypp_margin", "success_margin", "explosive_margin", "turnover_margin", "havoc_margin"]] + \
     ["ret_off_diff", "ret_def_diff", "continuity_diff", "elo_rp_diff", "home_ret_avg", "away_ret_avg",
      "away_travel_km", "tz_shift", "elev_diff"]
pm = {}
for s in sorted(base.season.unique()):
    tr = base[(base.season >= FEAT_START) & (base.season < s)]; te = base[base.season == s]
    if len(tr) < 1500 or len(te) == 0: continue
    m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.04, max_iter=350, l2_regularization=1.0,
                                      min_samples_leaf=50, random_state=0)
    m.fit(tr[T4].values, np.clip(tr.margin.values, -35, 35))
    for gid, p in zip(te.game_id.values, m.predict(te[T4].values)): pm[gid] = float(p)
base["gbm_T4_margin_pred"] = base.game_id.map(pm)
# calibrate predicted margin -> prob, walk-forward
pp = {}
d = base.dropna(subset=["gbm_T4_margin_pred"])
for s in sorted(d.season.unique()):
    tr = d[d.season < s]; te = d[d.season == s]
    if len(tr) < 800 or len(te) == 0: continue
    clf = LogisticRegression(C=1e6, max_iter=1000).fit(tr[["gbm_T4_margin_pred"]].values, tr.home_win.values)
    for gid, p in zip(te.game_id.values, clf.predict_proba(te[["gbm_T4_margin_pred"]].values)[:, 1]): pp[gid] = float(p)
base["gbm_T4_margin"] = base.game_id.map(pp)

# ---- final stacks ----
final_cols = ["elo_dynk", "elo_rp", "ridge_margin_prob", "ridge_ypp_margin_prob", "pr_points_keep",
              "bradley_terry", "gbm_T2_efficiency", "gbm_T4_situational", "gbm_T4_margin"]
base["stack_final"] = season_stack(base, final_cols, C=0.5)
base["stack_final_wk"] = season_stack(base, final_cols, C=0.5, by_week_bucket=True)
base["stack_final_ol"] = season_stack(base, final_cols + ["open_prob", "gbm_T5_openline"], C=0.5)
base.to_csv(os.path.join(RES, "experiments_predictions.csv.gz"), index=False, compression="gzip")

lad = pd.read_csv(os.path.join(RES, "experiments_ladder.csv"))
rows = []
for col, label, tier in [("elo_dynk", "Elo, dynamic K", "T1"), ("gbm_T4_margin", "GBM T4 margin-regression", "T4"),
                         ("stack_final", "Stack FINAL (results-only)", "T4"), ("stack_final_wk", "Stack FINAL, week-bucketed", "T4"),
                         ("stack_final_ol", "Stack FINAL + opening line", "T5")]:
    rows.append({"tier": tier, "model": label, "col": col, **ev(base, col), **market_tests(base, col)})
lad = pd.concat([lad[~lad.col.isin([r["col"] for r in rows])], pd.DataFrame(rows)], ignore_index=True)
lad.to_csv(os.path.join(RES, "experiments_ladder.csv"), index=False)

bw = pd.read_csv(os.path.join(RES, "experiments_by_week.csv"))
rw = base[(base.season >= EVAL_LO) & (base.season <= EVAL_HI) & (base.season_type == "regular")]
add = []
for wk in range(1, 16):
    s = rw[rw.week == wk]
    if len(s) < 30: continue
    for col in ["stack_final", "stack_final_wk", "stack_final_ol", "gbm_T4_margin"]:
        add.append({"week": wk, "model": col, **metrics.evaluate(s.home_win.values, s[col].values)})
bw = pd.concat([bw[~bw.model.isin(["stack_final", "stack_final_wk", "stack_final_ol", "gbm_T4_margin"])], pd.DataFrame(add)])
bw.to_csv(os.path.join(RES, "experiments_by_week.csv"), index=False)
pd.set_option("display.width", 200)
print(lad[["tier", "model", "acc", "logloss", "brier", "market_gain", "ats_pct", "model_right_when_disagree"]].round(4).to_string(index=False))
