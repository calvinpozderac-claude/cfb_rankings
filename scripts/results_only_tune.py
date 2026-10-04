"""Tune the model side of the results-only predictor on the cached feature table.

Validation = 2006-2017, held-out test = 2018-2025. Nothing about the test
seasons informs the choice of feature set, GBM hyper-parameters or blend weight.
"""
from __future__ import annotations
import itertools, os, sys, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.cfbrank import metrics
from scripts.results_only_model import (ALL_FEATS, LR_FEATS, RATING_FEATS, FORM_FEATS,
                                        season_refit, mk_lr, logit, RES)

VAL, TEST = (2006, 2017), (2018, 2025)
base = pd.read_csv(os.path.join(RES, "results_only_features.csv.gz"))

ORIG = ["massey_short", "massey_mid", "massey_long", "offdef_margin", "offdef_total",
        "bt_diff", "pr_diff", "pyth_diff", "sos_diff", "home_ind",
        "elo_diff", "elo_fast_diff", "elo_slow_diff", "wp_diff", "avgmargin_diff",
        "form3_diff", "gp_min", "gp_diff", "rest_diff", "neutral", "week"]
COMPACT = ["massey_short", "massey_mid", "offdef_margin", "offdef_total", "elo_diff",
           "bt_diff", "pr_diff", "sos_diff", "wp_diff", "form3_diff", "gp_min",
           "rest_diff", "neutral", "home_ind", "week"]
SETS = {"all": ALL_FEATS, "orig": ORIG, "compact": COMPACT, "ratings": RATING_FEATS}
GRIDS = {
    "d3_lr04": dict(max_depth=3, learning_rate=0.04, max_iter=400),
    "d3_lr03_600": dict(max_depth=3, learning_rate=0.03, max_iter=600),
    "d4_lr03": dict(max_depth=4, learning_rate=0.03, max_iter=500),
    "d2_lr05": dict(max_depth=2, learning_rate=0.05, max_iter=700),
}


def ev(df, col, era):
    s = df[(df.season >= era[0]) & (df.season <= era[1])]
    return metrics.evaluate(s.home_win.values, s[col].values)


def gbm_maker(gp, seed=0):
    return lambda: HistGradientBoostingClassifier(
        l2_regularization=1.0, min_samples_leaf=40, random_state=seed, **gp)


rows = []
t0 = time.time()
for (sname, feats), (gname, gp) in itertools.product(SETS.items(), GRIDS.items()):
    col = f"g_{sname}_{gname}"
    base[col] = season_refit(base, feats, gbm_maker(gp))
    v, t = ev(base, col, VAL), ev(base, col, TEST)
    rows.append(dict(feats=sname, gbm=gname, val_ll=v["logloss"], test_ll=t["logloss"],
                     test_acc=t["acc"]))
    print(f"  {sname:8s} {gname:12s} val {v['logloss']:.4f} | test {t['logloss']:.4f} "
          f"[{time.time()-t0:.0f}s]", flush=True)
res = pd.DataFrame(rows).sort_values("val_ll")
res.to_csv(os.path.join(RES, "results_only_tuning.csv"), index=False)
best = res.iloc[0]
print(f"\nBEST on validation: feats={best.feats} gbm={best.gbm} (val {best.val_ll:.4f})")

# blend weight between the stable LR and the chosen GBM, also chosen on validation
base["ro_lr"] = season_refit(base, LR_FEATS, mk_lr)
bcol = f"g_{best.feats}_{best.gbm}"
blend = []
for w in (0.0, 0.25, 0.5, 0.75, 1.0):
    base["_b"] = 1 / (1 + np.exp(-(w * logit(base[bcol]) + (1 - w) * logit(base.ro_lr))))
    blend.append(dict(w=w, val_ll=ev(base, "_b", VAL)["logloss"], test_ll=ev(base, "_b", TEST)["logloss"]))
bl = pd.DataFrame(blend)
print("\nblend (w on GBM):"); print(bl.round(4).to_string(index=False))
bw = bl.sort_values("val_ll").iloc[0].w

# seed-bagged final model at the chosen settings
cols = []
for sd in range(5):
    c = f"_seed{sd}"
    base[c] = season_refit(base, SETS[best.feats], gbm_maker(GRIDS[best.gbm], seed=sd))
    cols.append(c)
base["gbm_bag"] = 1 / (1 + np.exp(-np.mean([logit(base[c]) for c in cols], axis=0)))
base["ro_best"] = 1 / (1 + np.exp(-(bw * logit(base.gbm_bag) + (1 - bw) * logit(base.ro_lr))))
for name in ["ro_lr", bcol, "gbm_bag", "ro_best", "ensemble", "market"]:
    v, t = ev(base, name, VAL), ev(base, name, TEST)
    print(f"  {name:22s} val {v['logloss']:.4f} | TEST {t['logloss']:.4f} acc {t['acc']:.4f} n={t['n']}")
base[["game_id", "season", "week", "home_win", "ro_lr", "gbm_bag", "ro_best",
      "market", "ensemble", "cfbd_elo", "spread_home", "margin",
      "home_team", "away_team"]].to_csv(
    os.path.join(RES, "results_only_best.csv.gz"), index=False, compression="gzip")
print(f"\nchosen blend weight {bw} | done {time.time()-t0:.0f}s")
