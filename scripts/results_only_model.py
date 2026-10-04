"""Best predictor using ONLY previous game results (win/loss + final score).

Allowed inputs for any feature: the identities of the two teams, where the game
is played (home / away / neutral), its date, and the final scores of games that
already finished. Nothing else -- no box scores, no rosters, no betting lines,
no conference/division labels as features. (Division is used once, as *scope*,
to decide which league we rate and evaluate on: games involving >=1 FBS team.)

Pipeline, all strictly walk-forward:
  1. one chronological pass  -> Elo (fast + slow) and running form/record/rest
  2. a weekly refit          -> Massey (3 horizons), off/def, Bradley-Terry,
                                points-flow PageRank, Pythagorean, SOS
  3. per-season refit        -> logistic regression and gradient boosting on the
                                combined feature table, then a logistic stack
Evaluated against the Vegas closing line and CFBD's Elo.
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.cfbrank import metrics, resultsonly as ro
from src.cfbrank.data import build_games, is_fbs
from src.cfbrank.elo import run_elo

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
START_PRED = 2004          # first season we emit predictions for
ERAS = [(2006, 2025, "2006-2025 (betting era)"), (2014, 2025, "2014-2025"),
        (2019, 2025, "2019-2025 (recent)"), (2023, 2025, "2023-2025 (last 3)")]

RATING_FEATS = ["massey_short", "massey_mid", "massey_long", "massey_cap14", "massey_cap50",
                "massey_recent", "offdef_margin", "offdef_total", "bt_diff", "pr_diff",
                "pyth_diff", "sos_diff", "scoresd_sum", "scoresd_diff", "gpw_min", "home_ind"]
FORM_FEATS = ["elo_diff", "elo_fast_diff", "elo_slow_diff", "wp_diff", "avgmargin_diff",
              "form3_diff", "gp_min", "gp_diff", "rest_diff", "neutral", "week",
              "h2h_margin", "h2h_games"]
LR_FEATS = ["massey_mid", "offdef_margin", "bt_diff", "elo_diff", "pr_diff", "home_ind"]
ALL_FEATS = RATING_FEATS + FORM_FEATS

# ---- settings chosen on the 2006-2017 validation window (see
# results_only_tuning.csv / results_only_rating_sweep.csv); the 2018-2025 test
# seasons played no part in any of these choices.
RATING_CFG = dict(margin_cap=35.0, lam_massey=2.0)
GBM_FEATS = ["massey_short", "massey_mid", "massey_long", "offdef_margin", "offdef_total",
             "bt_diff", "pr_diff", "pyth_diff", "sos_diff", "home_ind",
             "elo_diff", "elo_fast_diff", "elo_slow_diff", "wp_diff", "avgmargin_diff",
             "form3_diff", "gp_min", "gp_diff", "rest_diff", "neutral", "week"]
BLEND_W = 0.75      # weight on the bagged GBM vs the logistic regression
N_SEEDS = 5
TEST = (2018, 2025)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def form_pass(pool: pd.DataFrame) -> pd.DataFrame:
    """One chronological pass: season record, scoring form, rest, experience."""
    wins = defaultdict(int); gp = defaultdict(int)
    pf = defaultdict(float); pa = defaultdict(float)
    last3 = defaultdict(lambda: deque(maxlen=3))
    last_date = {}
    h2h = defaultdict(lambda: deque(maxlen=5))   # NOT reset per season
    cur = None
    rows = []
    for r in pool.itertuples(index=False):
        if cur is None:
            cur = r.season
        elif r.season != cur:
            wins.clear(); gp.clear(); pf.clear(); pa.clear(); last3.clear(); last_date.clear()
            cur = r.season
        h, a = r.home_team, r.away_team

        def wp(t):
            return wins[t] / gp[t] if gp[t] else 0.5

        def am(t):
            return (pf[t] - pa[t]) / gp[t] if gp[t] else 0.0

        def f3(t):
            return float(np.mean(last3[t])) if len(last3[t]) else 0.0

        key = (h, a) if h < a else (a, h)
        prior = list(h2h[key])
        sign = 1.0 if h == key[0] else -1.0
        d = pd.Timestamp(r.start_date) if pd.notna(r.start_date) else None
        rh = (d - last_date[h]).days if (d is not None and h in last_date) else np.nan
        ra = (d - last_date[a]).days if (d is not None and a in last_date) else np.nan
        rows.append({
            "game_id": r.game_id,
            "wp_diff": wp(h) - wp(a),
            "avgmargin_diff": np.clip(am(h), -28, 28) - np.clip(am(a), -28, 28),
            "form3_diff": np.clip(f3(h), -28, 28) - np.clip(f3(a), -28, 28),
            "gp_min": min(gp[h], gp[a]), "gp_diff": gp[h] - gp[a],
            "rest_diff": (rh - ra) if (rh == rh and ra == ra) else 0.0,
            "neutral": 1.0 if r.neutral_site else 0.0,
            "h2h_margin": float(np.mean(prior)) * sign if prior else 0.0,
            "h2h_games": len(prior),
        })
        m = r.home_points - r.away_points
        h2h[key].append(m * sign)
        gp[h] += 1; gp[a] += 1
        pf[h] += r.home_points; pa[h] += r.away_points
        pf[a] += r.away_points; pa[a] += r.home_points
        last3[h].append(m); last3[a].append(-m)
        if r.home_win == 1:
            wins[h] += 1
        else:
            wins[a] += 1
        if d is not None:
            last_date[h] = d; last_date[a] = d
    return pd.DataFrame(rows).set_index("game_id")


def rating_pass(pool: pd.DataFrame, eval_ids: set, **cfg) -> pd.DataFrame:
    """Weekly refit of every batch rating system; features for that week's games."""
    out = []
    by_p = {p: d for p, d in pool.groupby("period")}
    t0 = time.time()
    for k, p in enumerate(sorted(by_p)):
        cur = by_p[p]
        cur = cur[cur.game_id.isin(eval_ids)]
        if len(cur) == 0:
            continue
        hist = pool[pool.period < p]
        snap = ro.fit_all(hist, **cfg)
        if snap is None:
            continue
        for r in cur.itertuples(index=False):
            f = ro.features(snap, r.home_team, r.away_team, bool(r.neutral_site))
            f["game_id"] = r.game_id
            out.append(f)
        if k % 80 == 0:
            print(f"    period {p} ({k}/{len(by_p)}) {time.time()-t0:.0f}s", flush=True)
    return pd.DataFrame(out).set_index("game_id")


def season_refit(df, feats, make, start=START_PRED, min_train=2000):
    """Refit once per season on every earlier season; predict that season."""
    preds = {}
    for s in sorted(df.season.unique()):
        if s < start:
            continue
        tr = df[df.season < s].dropna(subset=feats + ["home_win"])
        te = df[df.season == s].dropna(subset=feats)
        if len(tr) < min_train or len(te) == 0:
            continue
        m = make()
        m.fit(tr[feats].values, tr.home_win.values)
        for gid, p in zip(te.game_id.values, m.predict_proba(te[feats].values)[:, 1]):
            preds[gid] = float(p)
    return df.game_id.map(preds)


def mk_lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def mk_gbm(seed=0):
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.04, max_iter=400,
                                          l2_regularization=1.0, min_samples_leaf=40,
                                          random_state=seed)


FEAT_CACHE = os.path.join(RES, "results_only_features.csv.gz")


def build_feature_table(force=False) -> pd.DataFrame:
    if os.path.exists(FEAT_CACHE) and not force:
        return pd.read_csv(FEAT_CACHE)
    g = build_games()
    g = g[g.season <= 2025].reset_index(drop=True)
    pool = g[(g.home_division == "fbs") | (g.away_division == "fbs")].reset_index(drop=True)
    eval_ids = set(g[is_fbs(g)].game_id)
    print(f"rating pool {len(pool):,} games | eval set {len(eval_ids):,} FBS-vs-FBS")

    print("Elo passes ...")
    e = run_elo(pool, k=40, home_field=50, mov=True, preseason_regress=0.15)
    ef = run_elo(pool, k=65, home_field=50, mov=True, preseason_regress=0.15)
    es = run_elo(pool, k=20, home_field=50, mov=True, preseason_regress=0.15)

    print("form pass ...")
    F = form_pass(pool)
    print("weekly rating refits ...")
    R = rating_pass(pool, eval_ids, **RATING_CFG)

    base = g[g.game_id.isin(eval_ids)][
        ["game_id", "season", "week", "season_type", "home_team", "away_team",
         "home_win", "margin", "spread_home"]].copy()
    base["elo_prob"] = base.game_id.map(e["elo_prob"])
    base["elo_diff"] = base.game_id.map(e["elo_home_pre"] - e["elo_away_pre"])
    base["elo_fast_diff"] = base.game_id.map(ef["elo_home_pre"] - ef["elo_away_pre"])
    base["elo_slow_diff"] = base.game_id.map(es["elo_home_pre"] - es["elo_away_pre"])
    base = base.merge(F, left_on="game_id", right_index=True, how="left")
    base = base.merge(R, left_on="game_id", right_index=True, how="left")
    prev = pd.read_csv(os.path.join(RES, "predictions.csv.gz"),
                       usecols=["game_id", "market", "cfbd_elo", "ensemble"])
    base = base.merge(prev, on="game_id", how="left")
    base.to_csv(FEAT_CACHE, index=False, compression="gzip")
    return base


def main():
    t0 = time.time()
    base = build_feature_table(force="--rebuild" in sys.argv)
    print(f"feature table {len(base):,} rows")

    print("walk-forward models ...")
    base["ro_lr"] = season_refit(base, LR_FEATS, mk_lr)
    seeds = []
    for sd in range(N_SEEDS):
        c = f"_s{sd}"
        base[c] = season_refit(base, GBM_FEATS, lambda sd=sd: mk_gbm(sd))
        seeds.append(c)
    base["ro_gbm"] = 1 / (1 + np.exp(-np.mean([logit(base[c]) for c in seeds], axis=0)))
    base["ro_best"] = 1 / (1 + np.exp(-(BLEND_W * logit(base.ro_gbm)
                                        + (1 - BLEND_W) * logit(base.ro_lr))))
    base.drop(columns=seeds, inplace=True)

    base.to_csv(os.path.join(RES, "results_only_predictions.csv.gz"), index=False,
                compression="gzip")

    cols = ["market", "cfbd_elo", "ensemble", "elo_prob", "ro_lr", "ro_gbm", "ro_best"]
    rows = []
    for lo, hi, lab in ERAS:
        sub = base[(base.season >= lo) & (base.season <= hi)]
        for c in cols:
            rows.append({"era": lab, "model": c, **metrics.evaluate(sub.home_win.values, sub[c].values)})
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(RES, "results_only_metrics.csv"), index=False)
    for lo, hi, lab in ERAS:
        print(f"\n=== {lab} ===")
        t = tab[tab.era == lab].set_index("model")
        print(t[["n", "acc", "logloss", "brier"]].round(4).to_string())
    sub = base[(base.season >= TEST[0]) & (base.season <= TEST[1])]
    print(f"\n=== HELD-OUT TEST {TEST[0]}-{TEST[1]} (no tuning used these seasons) ===")
    for c in cols:
        m = metrics.evaluate(sub.home_win.values, sub[c].values)
        print(f"  {c:10s} n={m['n']:5d} acc={m['acc']:.4f} logloss={m['logloss']:.4f} brier={m['brier']:.4f}")
    print(f"\nDONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
