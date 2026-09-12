"""Batch ranking models: PageRank family, Bradley-Terry, and blade-chest.

Each model exposes ``fit(history) -> predictor`` where ``history`` is a games
DataFrame (chronological) and ``predictor(home, away, neutral) -> P(home win)``.
Fitting uses only the supplied history, so the walk-forward backtester can call
it with "all games before period p" and get leak-free predictions.

To give the batch (season-resume) models early-season stability comparable to
Elo's cross-season carry, they fit on the current season plus the prior season,
with prior-season games down-weighted by ``carry``.
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
def _history_window(history: pd.DataFrame, carry: float) -> pd.DataFrame:
    """Current season + prior season (down-weighted) with a `w` weight column."""
    if len(history) == 0:
        return history.assign(w=[])
    cur = int(history["season"].max())
    h = history[history["season"] >= cur - 1].copy()
    h["w"] = np.where(h["season"] == cur, 1.0, carry)
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


def make_pagerank(weight="win", damping=0.85, margin_cap=28.0, carry=0.35):
    """Factory for a PageRank ranker.

    weight: 'win'    - each result contributes a unit loser->winner edge
            'margin' - edge weight = min(margin, margin_cap)  (learned cap)
            'points' - bidirectional; edge i->j weight = points j scored on i
    """
    def fit(history: pd.DataFrame):
        h = _history_window(history, carry)
        nodes = sorted(set(h["home_team"]) | set(h["away_team"]))
        edges = []
        for r in h.itertuples(index=False):
            hp, ap, w = r.home_points, r.away_points, r.w
            winner = r.home_team if r.home_win == 1 else r.away_team
            loser = r.away_team if r.home_win == 1 else r.home_team
            if weight == "win":
                edges.append((loser, winner, w))
            elif weight == "margin":
                m = min(abs(hp - ap), margin_cap)
                edges.append((loser, winner, w * (1.0 + m)))
            elif weight == "points":
                # rank flows toward whoever scored: i -> j weighted by j's points on i
                edges.append((r.home_team, r.away_team, w * (ap + 1)))
                edges.append((r.away_team, r.home_team, w * (hp + 1)))
            else:
                raise ValueError(weight)
        pr = _pagerank(nodes, edges, damping=damping)
        # log scale so ratios become additive differences for the logistic
        rating = {t: np.log(max(v, 1e-12)) for t, v in pr.items()}
        return _calibrate(rating, h)

    return fit


# --------------------------------------------------------------------------- #
# Bradley-Terry (logistic MLE with home advantage + L2 shrinkage)
# --------------------------------------------------------------------------- #
def make_bradley_terry(reg=3.0, carry=0.35):
    """Bradley-Terry / logistic paired-comparison ratings.

    P(home win) = sigmoid(s_home - s_away + h*home_ind). Strengths s and home
    edge h are fit by weighted, L2-regularised maximum likelihood.
    """
    def fit(history: pd.DataFrame):
        h = _history_window(history, carry)
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
def make_blade_chest(dim=3, reg=1.0, carry=0.35, gamma=1.0):
    """Blade-chest-inner model (Chen & Joachims, 2016).

    Each team has a blade b_t (offense) and chest c_t (defense) vector plus a
    scalar strength g_t. P(i beats j) = sigmoid( <b_i, c_j> - <b_j, c_i> + g_i - g_j ).
    Captures rock-paper-scissors (intransitive) matchup structure a single
    rating cannot. Fit by weighted regularised MLE (L-BFGS).
    """
    def fit(history: pd.DataFrame):
        h = _history_window(history, carry)
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
