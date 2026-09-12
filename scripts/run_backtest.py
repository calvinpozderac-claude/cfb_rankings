"""End-to-end walk-forward backtest across all ranking methods.

Outputs (under results/):
  predictions.csv.gz   one row per FBS-vs-FBS game with each model's P(home win)
  metrics_overall.csv  metrics per method over several eras
  metrics_by_season.csv
  tuning_elo.csv, tuning_pagerank.csv
  calibration_<era>.csv
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.cfbrank import backtest, elo as elomod, features as feat, metrics, rankers
from src.cfbrank.data import build_games, is_fbs
from src.cfbrank.mlmodels import ML_FEATURES, make_gbm, make_mlp

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
os.makedirs(RESULTS, exist_ok=True)

START_SEASON = 2006        # betting era; base preds still generated from 2001
ERAS = [(2006, 2025, "2006-2025 (betting era)"),
        (2014, 2025, "2014-2025 (playoff era)"),
        (2019, 2025, "2019-2025 (recent)"),
        (2023, 2025, "2023-2025 (last 3)")]


def tune_elo(g, fbs_ids):
    """Grid-search Elo on 2007-2015, minimise log loss on FBS games."""
    val = g[(g.season >= 2007) & (g.season <= 2015)]
    val_ids = set(val[is_fbs(val)].game_id)
    rows = []
    best = None
    for k in (25, 40, 55):
        for hf in (50, 65, 80):
            for pr in (0.05, 0.10, 0.15, 0.25, 0.40):
                e = elomod.run_elo(g, k=k, home_field=hf, mov=True, preseason_regress=pr)
                sub = g[g.game_id.isin(val_ids)].copy()
                sub["p"] = sub.game_id.map(e["elo_prob"])
                m = metrics.evaluate(sub.home_win.values, sub.p.values)
                rows.append({"k": k, "home_field": hf, "preseason_regress": pr,
                             "acc": m["acc"], "logloss": m["logloss"]})
                if best is None or m["logloss"] < best["logloss"]:
                    best = rows[-1]
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "tuning_elo.csv"), index=False)
    print(f"  best Elo: {best}")
    return best


def tune_pagerank(g):
    """Tune PR margin edge cap + damping on 2010-2018 (log loss)."""
    val = g[(g.season >= 2013) & (g.season <= 2018)]
    val_ids = set(val[is_fbs(val)].game_id)
    rows = []
    best = None
    for cap in (14, 28, 45):
        for damp in (0.85, 0.90):
            fit = rankers.make_pagerank("margin", damping=damp, margin_cap=cap)
            preds = backtest.batch_backtest(g[g.season >= 2012], fit, val_ids)
            sub = g[g.game_id.isin(preds)].copy()
            sub["p"] = sub.game_id.map(preds)
            m = metrics.evaluate(sub.home_win.values, sub.p.values)
            rows.append({"margin_cap": cap, "damping": damp,
                         "acc": m["acc"], "logloss": m["logloss"]})
            if best is None or m["logloss"] < best["logloss"]:
                best = rows[-1]
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "tuning_pagerank.csv"), index=False)
    print(f"  best PR-margin: {best}")
    return best


def tune_decay(g):
    """Tune the cross-season recency decay per batch-model family on 2013-2018.

    A game d seasons back is weighted decay**d. Different families want different
    memory: PageRank favours recency, Bradley-Terry wants more history.
    Returns (best_pr_decay, best_bt_decay); blade-chest uses the PR-side optimum.
    """
    val = g[(g.season >= 2013) & (g.season <= 2018)]
    val_ids = set(val[is_fbs(val)].game_id)
    gg = g[g.season >= 2006]
    grid = [0.0, 0.35, 0.5, 0.65, 0.8, 1.0]
    rows = []
    best = {"pr": (None, 9), "bt": (None, 9)}
    for decay in grid:
        for fam, fit in [("pr_points_keep", rankers.make_pagerank("points_keep", decay=decay)),
                         ("bradley_terry", rankers.make_bradley_terry(decay=decay))]:
            preds = backtest.batch_backtest(gg, fit, val_ids)
            sub = g[g.game_id.isin(preds)].copy()
            sub["p"] = sub.game_id.map(preds)
            m = metrics.evaluate(sub.home_win.values, sub.p.values)
            rows.append({"family": fam, "decay": decay, "acc": m["acc"], "logloss": m["logloss"]})
            key = "pr" if fam == "pr_points_keep" else "bt"
            if m["logloss"] < best[key][1]:
                best[key] = (decay, m["logloss"])
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "tuning_decay.csv"), index=False)
    print(f"  best decay: PageRank={best['pr'][0]}  Bradley-Terry={best['bt'][0]}")
    return best["pr"][0], best["bt"][0]


def main():
    t0 = time.time()
    g = build_games()
    g = g[g.season <= 2025].reset_index(drop=True)
    fbs_ids = set(g[is_fbs(g)].game_id)
    print(f"games={len(g):,}  FBS-vs-FBS={len(fbs_ids):,}")

    # ---- hyperparameter tuning (answers 'learn the weights') ----
    print("Tuning Elo ...")
    be = tune_elo(g, fbs_ids)
    print("Tuning PageRank margin ...")
    bp = tune_pagerank(g)
    print("Tuning cross-season decay ...")
    pr_decay, bt_decay = tune_decay(g)
    blade_decay = 0.5  # blade-chest optimum from a separate sweep (needs a little more history)

    # ---- base table ----
    base = g[g.game_id.isin(fbs_ids)][
        ["game_id", "season", "week", "season_type", "home_team", "away_team",
         "home_win", "margin", "spread_home"]
    ].copy().reset_index(drop=True)

    # ---- Elo (online) with tuned params ----
    print("Running Elo ...")
    e = elomod.run_elo(g, k=be["k"], home_field=be["home_field"], mov=True,
                       preseason_regress=be["preseason_regress"])
    base["elo"] = base.game_id.map(e["elo_prob"])
    elo_diff = (e["elo_home_pre"] - e["elo_away_pre"])

    # ---- features for ML ----
    print("Building features ...")
    F = feat.build_features(g)
    F["elo_prob"] = e["elo_prob"]
    F["elo_diff"] = elo_diff
    fmerged = F  # indexed by game_id

    # ---- batch rankers (walk-forward over ALL fbs games) ----
    dmp = bp["damping"]
    batch_specs = [
        ("pr_win", rankers.make_pagerank("win", damping=dmp, decay=pr_decay)),
        ("pr_margin", rankers.make_pagerank("margin", damping=dmp,
                                            margin_cap=bp["margin_cap"], decay=pr_decay)),
        ("pr_points", rankers.make_pagerank("points", damping=dmp, decay=pr_decay)),
        ("pr_points_against", rankers.make_pagerank("points_against", damping=dmp, decay=pr_decay)),
        ("pr_points_keep", rankers.make_pagerank("points_keep", damping=dmp, decay=pr_decay)),
        ("bradley_terry", rankers.make_bradley_terry(decay=bt_decay)),
        ("blade_chest", rankers.make_blade_chest(dim=3, decay=blade_decay)),
    ]
    for name, fit in batch_specs:
        t = time.time()
        preds = backtest.batch_backtest(g, fit, fbs_ids)
        base[name] = base.game_id.map(preds)
        print(f"  {name:14s} done [{time.time()-t:.0f}s]")

    # ---- ML models (per-season refit) ----
    for name, mk in [("gbm", make_gbm), ("mlp", make_mlp)]:
        t = time.time()
        preds = backtest.season_refit_ml(g, fmerged, mk, ML_FEATURES, fbs_ids,
                                          start_season=2004)
        base[name] = base.game_id.map(preds)
        print(f"  {name:14s} done [{time.time()-t:.0f}s]")

    # ---- benchmarks ----
    base["market"] = base.game_id.map(backtest.market_prob(g, fbs_ids, START_SEASON))
    base["cfbd_elo"] = base.game_id.map(backtest.cfbd_elo_prob(g, fbs_ids))

    # ---- stacked ensemble (our signals only), per-season logistic ----
    from sklearn.linear_model import LogisticRegression
    stack_cols = ["elo", "pr_points_keep", "bradley_terry", "blade_chest", "gbm", "mlp"]

    def _logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    ens = {}
    Xall = base[stack_cols].apply(_logit)
    for s in sorted(base.season.unique()):
        if s < START_SEASON:
            continue
        tr = base[(base.season < s)].dropna(subset=stack_cols + ["home_win"])
        te = base[base.season == s].dropna(subset=stack_cols)
        if len(tr) < 800 or len(te) == 0:
            continue
        clf = LogisticRegression(C=1.0, max_iter=1000)
        clf.fit(_logit(tr[stack_cols].values), tr["home_win"].values)
        pr = clf.predict_proba(_logit(te[stack_cols].values))[:, 1]
        for gid, p in zip(te.game_id.values, pr):
            ens[gid] = float(p)
    base["ensemble"] = base.game_id.map(ens)

    # ---- market+model ensemble (does our model add to the market?) ----
    ens2 = {}
    mm_cols = ["market", "ensemble"]
    for s in sorted(base.season.unique()):
        if s < START_SEASON:
            continue
        tr = base[(base.season < s)].dropna(subset=mm_cols + ["home_win"])
        te = base[base.season == s].dropna(subset=mm_cols)
        if len(tr) < 800 or len(te) == 0:
            continue
        clf = LogisticRegression(C=1.0, max_iter=1000)
        clf.fit(_logit(tr[mm_cols].values), tr["home_win"].values)
        pr = clf.predict_proba(_logit(te[mm_cols].values))[:, 1]
        for gid, p in zip(te.game_id.values, pr):
            ens2[gid] = float(p)
    base["market_plus_model"] = base.game_id.map(ens2)

    base.to_csv(os.path.join(RESULTS, "predictions.csv.gz"), index=False,
                compression="gzip")

    # ---- metrics ----
    model_cols = ["elo", "pr_win", "pr_margin", "pr_points", "pr_points_against",
                  "pr_points_keep", "bradley_terry", "blade_chest", "gbm", "mlp",
                  "ensemble", "cfbd_elo", "market", "market_plus_model"]
    overall_rows = []
    for lo, hi, label in ERAS:
        sub = base[(base.season >= lo) & (base.season <= hi)]
        for mc in model_cols:
            m = metrics.evaluate(sub.home_win.values, sub[mc].values)
            overall_rows.append({"era": label, "model": mc, **m})
    pd.DataFrame(overall_rows).to_csv(os.path.join(RESULTS, "metrics_overall.csv"),
                                      index=False)

    by_season = []
    for s in sorted(base.season.unique()):
        if s < START_SEASON:
            continue
        sub = base[base.season == s]
        for mc in model_cols:
            m = metrics.evaluate(sub.home_win.values, sub[mc].values)
            by_season.append({"season": s, "model": mc, **m})
    pd.DataFrame(by_season).to_csv(os.path.join(RESULTS, "metrics_by_season.csv"),
                                   index=False)

    # by regular-season week (timing effects), 2014-2025
    by_week = []
    rw = base[(base.season >= 2014) & (base.season <= 2025) & (base.season_type == "regular")]
    for wk in range(1, 16):
        sub = rw[rw.week == wk]
        if len(sub) < 30:
            continue
        for mc in model_cols:
            m = metrics.evaluate(sub.home_win.values, sub[mc].values)
            by_week.append({"week": wk, "model": mc, **m})
    pd.DataFrame(by_week).to_csv(os.path.join(RESULTS, "metrics_by_week.csv"), index=False)

    # calibration for a few key models, recent era
    rec = base[(base.season >= 2019) & (base.season <= 2025)]
    for mc in ["elo", "ensemble", "market", "gbm"]:
        ct = metrics.calibration_table(rec.home_win.values, rec[mc].values)
        ct.to_csv(os.path.join(RESULTS, f"calibration_{mc}.csv"), index=False)

    print(f"\nDONE in {time.time()-t0:.0f}s")
    # quick console summary for recent era
    print("\n=== 2019-2025 (recent) ===")
    sub = base[(base.season >= 2019) & (base.season <= 2025)]
    for mc in model_cols:
        m = metrics.evaluate(sub.home_win.values, sub[mc].values)
        print(f"  {mc:18s} n={m['n']:>5} acc={m['acc']:.4f} ll={m['logloss']:.4f} brier={m['brier']:.4f}")


if __name__ == "__main__":
    main()
