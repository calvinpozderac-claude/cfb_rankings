"""Batch ranking models: PageRank family, Bradley-Terry, and blade-chest.

Each model exposes ``fit(history) -> predictor`` where ``history`` is a games
DataFrame (chronological) and ``predictor(home, away, neutral) -> P(home win)``.
Fitting uses only the supplied history, so the walk-forward backtester can call
it with "all games before period p" and get leak-free predictions.

To give the batch models a prior season carried in from history (so a program
that was strong last year is rated strong in week 1), they fit on several past
seasons with an exponential recency weight: a game ``d`` seasons back gets
weight ``decay ** d`` (see ``_decayed_history``). The best ``decay`` is tuned.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _decayed_history(history: pd.DataFrame, decay: float, max_back: int = 6) -> pd.DataFrame:
    """All history from the last ``max_back+1`` seasons, with a recency weight.

    A game from ``d`` seasons before the one being predicted gets weight
    ``decay ** d`` (current season = 1). ``decay=1`` weights all seasons equally;
    ``decay->0`` collapses to the current season only. This is how every batch
    model "remembers" that a program was strong in prior years — heavily at the
    start of a new season, fading as fresh results arrive.
    """
    if len(history) == 0:
        return history.assign(w=[])
    cur = int(history["season"].max())
    h = history[history["season"] >= cur - max_back].copy()
    h["w"] = np.power(float(decay), (cur - h["season"]).astype(float))
    return h


def _calibrate(rating: dict, h: pd.DataFrame, hfa_default=True):
    """Fit logistic P(home) ~ rating_diff + home_indicator on history `h`.

    Returns a predictor(home, away, neutral) using `rating` (defaultdict) and the
    fitted logistic. Falls back gracefully for unseen teams (baseline rating).
    """
    base = np.median(list(rating.values())) if rating else 0.0
    diffs, homeind, y, w = [], [], [], []
    for r in h.itertuples(index=False):
        diffs.append(rating.get(r.home_team, base) - rating.get(r.away_team, base))
        homeind.append(0.0 if r.neutral_site else 1.0)
        y.append(r.home_win)
        w.append(getattr(r, "w", 1.0))
    X = np.column_stack([diffs, homeind])
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        # degenerate history; constant predictor
        p0 = float(np.mean(y)) if len(y) else 0.5
        return lambda home, away, neutral: p0
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    clf.fit(X, y, sample_weight=np.asarray(w))

    def predict(home, away, neutral):
        d = rating.get(home, base) - rating.get(away, base)
        hi = 0.0 if neutral else 1.0
        return float(clf.predict_proba([[d, hi]])[0, 1])

    return predict


# --------------------------------------------------------------------------- #
# PageRank family
# --------------------------------------------------------------------------- #
def _pagerank(nodes, edges, damping=0.85, iters=200, tol=1e-10):
    """Power-iteration PageRank. `edges` = list of (src, dst, weight); rank flows
    src -> dst (loser -> winner)."""
    idx = {t: i for i, t in enumerate(nodes)}
    n = len(nodes)
    if n == 0:
        return {}
    M = np.zeros((n, n))
    for s, d, w in edges:
        M[idx[d], idx[s]] += w  # column s -> distribute to row d
    colsum = M.sum(axis=0)
    dangling = colsum == 0
    colsum[dangling] = 1.0
    M = M / colsum
    r = np.full(n, 1.0 / n)
    teleport = np.full(n, 1.0 / n)
    for _ in range(iters):
        r_new = (1 - damping) * teleport + damping * (M @ r)
        # redistribute dangling nodes' mass uniformly
        r_new += damping * teleport * r[dangling].sum()
        r_new /= r_new.sum()
        if np.abs(r_new - r).sum() < tol:
            r = r_new
            break
        r = r_new
    return {t: r[idx[t]] for t in nodes}


def _result_edges(home, away, hp, ap, home_win, w, weight, margin_cap):
    """Directed edges (src, dst, weight) contributed by one game.

    Rank flows *out* of `src` toward `dst`; PageRank then rewards nodes that
    receive heavily-weighted flow. Weights from parallel games accumulate.
    """
    winner, loser = (home, away) if home_win == 1 else (away, home)
    if weight == "win":
        return [(loser, winner, w)]
    if weight == "margin":
        m = min(abs(hp - ap), margin_cap)
        return [(loser, winner, w * (1.0 + m))]
    if weight == "points":
        return [(home, away, w * (ap + 1)), (away, home, w * (hp + 1))]
    if weight == "points_against":
        # Each team sends rank to opponents in proportion to the points those
        # opponents scored on it (raw points). A close game swaps rank ~evenly;
        # a blowout ships rank to the team that scored.
        e = []
        if ap > 0:
            e.append((home, away, w * ap))   # away scored ap on home
        if hp > 0:
            e.append((away, home, w * hp))   # home scored hp on away
        return e
    if weight == "points_keep":
        # Same opponent flow, plus a self-loop weighted by the points a team
        # scored, so it RETAINS rank. Season keep/give split per team is
        # (points scored) / (points scored + points allowed).
        e = []
        if ap > 0:
            e.append((home, away, w * ap))
        if hp > 0:
            e.append((away, home, w * hp))
        if hp > 0:
            e.append((home, home, w * hp))   # home keeps in proportion to its scoring
        if ap > 0:
            e.append((away, away, w * ap))
        return e
    raise ValueError(weight)


def make_pagerank(weight="win", damping=0.85, margin_cap=28.0, decay=0.6):
    """Factory for a PageRank ranker.

    weight: 'win'            - unit loser->winner edge
            'margin'         - loser->winner edge weight grows with (capped) margin
            'points'         - bidirectional, edge i->j weight = points j scored on i (+1)
            'points_against' - i distributes rank to opponents by their points on i
            'points_keep'    - as points_against, plus a self-loop = own points scored
    """
    def fit(history: pd.DataFrame):
        h = _decayed_history(history, decay)
        nodes = sorted(set(h["home_team"]) | set(h["away_team"]))
        edges = []
        for r in h.itertuples(index=False):
            edges.extend(_result_edges(r.home_team, r.away_team, r.home_points,
                                       r.away_points, r.home_win, r.w, weight, margin_cap))
        pr = _pagerank(nodes, edges, damping=damping)
        # log scale so ratios become additive differences for the logistic
        rating = {t: np.log(max(v, 1e-12)) for t, v in pr.items()}
        return _calibrate(rating, h)

    return fit


# --------------------------------------------------------------------------- #
# Bradley-Terry (logistic MLE with home advantage + L2 shrinkage)
# --------------------------------------------------------------------------- #
def make_bradley_terry(reg=3.0, decay=0.6):
    """Bradley-Terry / logistic paired-comparison ratings.

    P(home win) = sigmoid(s_home - s_away + h*home_ind). Strengths s and home
    edge h are fit by weighted, L2-regularised maximum likelihood.
    """
    def fit(history: pd.DataFrame):
        h = _decayed_history(history, decay)
        teams = sorted(set(h["home_team"]) | set(h["away_team"]))
        tidx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        if n == 0:
            return lambda home, away, neutral: 0.5
        hi = tidx_arr(h["home_team"], tidx)
        ai = tidx_arr(h["away_team"], tidx)
        homeind = np.where(h["neutral_site"].values, 0.0, 1.0)
        y = h["home_win"].values.astype(float)
        w = h["w"].values

        def negll(theta):
            s = theta[:n]
            hadv = theta[n]
            z = s[hi] - s[ai] + hadv * homeind
            p = 1.0 / (1.0 + np.exp(-z))
            p = np.clip(p, 1e-12, 1 - 1e-12)
            ll = w * (y * np.log(p) + (1 - y) * np.log(1 - p))
            loss = -ll.sum() + reg * 0.5 * np.sum(s ** 2)
            # gradient
            g = np.zeros(n + 1)
            resid = w * (p - y)
            np.add.at(g, hi, resid)
            np.add.at(g, ai, -resid)
            g[:n] += reg * s
            g[n] = np.sum(resid * homeind)
            return loss, g

        theta0 = np.zeros(n + 1)
        res = minimize(negll, theta0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 300})
        s = res.x[:n]
        hadv = res.x[n]
        rating = {t: s[tidx[t]] for t in teams}
        base = float(np.median(s))

        def predict(home, away, neutral):
            z = rating.get(home, base) - rating.get(away, base) + (0.0 if neutral else hadv)
            return float(1.0 / (1.0 + np.exp(-z)))

        return predict

    return fit


def tidx_arr(series, tidx):
    return np.array([tidx[t] for t in series], dtype=int)


# --------------------------------------------------------------------------- #
# Blade-chest (low-rank, intransitive)
# --------------------------------------------------------------------------- #
def make_blade_chest(dim=3, reg=1.0, decay=0.6, gamma=1.0):
    """Blade-chest-inner model (Chen & Joachims, 2016).

    Each team has a blade b_t (offense) and chest c_t (defense) vector plus a
    scalar strength g_t. P(i beats j) = sigmoid( <b_i, c_j> - <b_j, c_i> + g_i - g_j ).
    Captures rock-paper-scissors (intransitive) matchup structure a single
    rating cannot. Fit by weighted regularised MLE (L-BFGS).
    """
    def fit(history: pd.DataFrame):
        h = _decayed_history(history, decay)
        teams = sorted(set(h["home_team"]) | set(h["away_team"]))
        tidx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        if n == 0:
            return lambda home, away, neutral: 0.5
        hi = tidx_arr(h["home_team"], tidx)
        ai = tidx_arr(h["away_team"], tidx)
        homeind = np.where(h["neutral_site"].values, 0.0, 1.0)
        y = h["home_win"].values.astype(float)
        w = h["w"].values
        rng = np.random.default_rng(0)

        # params: blade (n,dim), chest (n,dim), g (n,), home adv (1,)
        def unpack(theta):
            b = theta[: n * dim].reshape(n, dim)
            c = theta[n * dim: 2 * n * dim].reshape(n, dim)
            g = theta[2 * n * dim: 2 * n * dim + n]
            hadv = theta[-1]
            return b, c, g, hadv

        def negll(theta):
            b, c, g, hadv = unpack(theta)
            # z_ij = <b_i,c_j> - <b_j,c_i> + g_i - g_j + hadv*homeind
            bi, bj = b[hi], b[ai]
            ci, cj = c[hi], c[ai]
            z = np.einsum("kd,kd->k", bi, cj) - np.einsum("kd,kd->k", bj, ci) \
                + g[hi] - g[ai] + hadv * homeind
            p = np.clip(1.0 / (1.0 + np.exp(-z)), 1e-12, 1 - 1e-12)
            loss = -(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum()
            loss += reg * 0.5 * (np.sum(b ** 2) + np.sum(c ** 2) + np.sum(g ** 2))
            # gradients
            resid = w * (p - y)  # d loss / d z
            gb = np.zeros_like(b); gc = np.zeros_like(c); gg = np.zeros(n)
            # dz/db_i = c_j ; dz/db_j = -c_i ; dz/dc_j = b_i ; dz/dc_i = -b_j
            np.add.at(gb, hi, resid[:, None] * cj)
            np.add.at(gb, ai, -resid[:, None] * ci)
            np.add.at(gc, ai, resid[:, None] * bi)
            np.add.at(gc, hi, -resid[:, None] * bj)
            np.add.at(gg, hi, resid)
            np.add.at(gg, ai, -resid)
            gb += reg * b; gc += reg * c; gg += reg * g
            ghadv = np.sum(resid * homeind)
            grad = np.concatenate([gb.ravel(), gc.ravel(), gg, [ghadv]])
            return loss, grad

        theta0 = np.concatenate([
            0.1 * rng.standard_normal(n * dim),
            0.1 * rng.standard_normal(n * dim),
            np.zeros(n), [0.3],
        ])
        res = minimize(negll, theta0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 200})
        b, c, g, hadv = unpack(res.x)
        base_g = float(np.median(g))

        def predict(home, away, neutral):
            i = tidx.get(home); j = tidx.get(away)
            hh = 0.0 if neutral else hadv
            if i is None or j is None:
                gi = base_g if i is None else g[i]
                gj = base_g if j is None else g[j]
                z = gi - gj + hh
            else:
                z = b[i] @ c[j] - b[j] @ c[i] + g[i] - g[j] + hh
            return float(1.0 / (1.0 + np.exp(-z)))

        return predict

    return fit
