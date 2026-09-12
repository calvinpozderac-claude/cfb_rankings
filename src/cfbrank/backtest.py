"""Walk-forward backtest orchestration.

Two prediction protocols, both strictly out-of-sample:
  * online models (Elo): one chronological predict-then-update pass.
  * batch models (PageRank/BT/blade-chest): for each week (period), fit on all
    games strictly before it, then predict that week's games.
  * ML models: refit once per season on all prior completed FBS games.

All predictions are aligned on the FBS-vs-FBS evaluation set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def batch_backtest(games: pd.DataFrame, fit_fn, eval_ids: set, min_period=0) -> dict:
    """Fit `fit_fn` per period on history<period, predict that period's eval games.

    `games` is the full chronological table (all divisions inform ratings).
    `eval_ids` is the set of game_ids to predict (FBS-vs-FBS).
    Returns {game_id: p_home}.
    """
    preds = {}
    periods = sorted(games["period"].unique())
    # index games by period for speed
    by_period = {p: df for p, df in games.groupby("period")}
    for p in periods:
        cur = by_period[p]
        cur_eval = cur[cur["game_id"].isin(eval_ids)]
        if len(cur_eval) == 0 or p < min_period:
            continue
        hist = games[games["period"] < p]
        if len(hist) < 50:
            for gid in cur_eval["game_id"]:
                preds[gid] = 0.5
            continue
        predictor = fit_fn(hist)
        for r in cur_eval.itertuples(index=False):
            preds[r.game_id] = predictor(r.home_team, r.away_team, bool(r.neutral_site))
    return preds


def season_refit_ml(games: pd.DataFrame, feats: pd.DataFrame, make_model,
                    feat_cols, eval_ids: set, start_season: int) -> dict:
    """Refit an sklearn model once per season on all prior FBS-vs-FBS games.

    make_model() -> fresh estimator with fit/predict_proba.
    """
    from .data import is_fbs
    fbs = games[is_fbs(games)].copy()
    df = fbs.merge(feats, left_on="game_id", right_index=True, how="left")
    preds = {}
    seasons = sorted(s for s in df["season"].unique() if s >= start_season)
    for s in seasons:
        train = df[df["season"] < s].dropna(subset=feat_cols)
        test = df[(df["season"] == s) & (df["game_id"].isin(eval_ids))]
        if len(train) < 500 or len(test) == 0:
            continue
        model = make_model()
        model.fit(train[feat_cols].values, train["home_win"].values)
        te = test.dropna(subset=feat_cols)
        pr = model.predict_proba(te[feat_cols].values)[:, 1]
        for gid, p in zip(te["game_id"].values, pr):
            preds[gid] = float(p)
    return preds


def market_prob(games: pd.DataFrame, eval_ids: set, start_season: int) -> dict:
    """Benchmark: calibrate closing spread -> P(home win) via per-season walk-forward
    logistic (fit on all prior seasons' spread/outcome pairs)."""
    from sklearn.linear_model import LogisticRegression
    from .data import is_fbs
    fbs = games[is_fbs(games)].dropna(subset=["spread_home"]).copy()
    preds = {}
    seasons = sorted(s for s in fbs["season"].unique() if s >= start_season)
    for s in seasons:
        train = fbs[fbs["season"] < s]
        test = fbs[(fbs["season"] == s) & (fbs["game_id"].isin(eval_ids))]
        if len(train) < 300 or len(test) == 0:
            continue
        clf = LogisticRegression(C=1e6, max_iter=1000)
        clf.fit(train[["spread_home"]].values, train["home_win"].values)
        pr = clf.predict_proba(test[["spread_home"]].values)[:, 1]
        for gid, p in zip(test["game_id"].values, pr):
            preds[gid] = float(p)
    return preds


def cfbd_elo_prob(games: pd.DataFrame, eval_ids: set, home_field=55.0) -> dict:
    """Benchmark: CFBD's own pregame Elo -> P(home win) (logistic, scale 400)."""
    preds = {}
    sub = games[games["game_id"].isin(eval_ids)]
    sub = sub.dropna(subset=["home_pregame_elo", "away_pregame_elo"])
    for r in sub.itertuples(index=False):
        hfa = 0.0 if r.neutral_site else home_field
        diff = (r.home_pregame_elo + hfa) - r.away_pregame_elo
        preds[r.game_id] = float(1.0 / (1.0 + 10.0 ** (-diff / 400.0)))
    return preds
