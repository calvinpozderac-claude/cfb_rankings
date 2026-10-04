"""Results-only rating systems.

HARD CONSTRAINT: every quantity here is derived from nothing but the sequence of
played games -- the two team identities, where it was played (home / away /
neutral), the date, and the two final scores. No box scores, no rosters, no
betting lines, no conference or division labels as features.

For each prediction week we fit, on a recency-weighted window of past games:

  massey   ridge least-squares team ratings on (clipped) point margin, at three
           memory horizons -- short (form), medium, long (program strength)
  offdef   ridge ratings on points *scored* and *allowed* separately, so the
           full scoreline is used: gives a predicted margin AND a predicted
           total (the total is a usable proxy for game variance)
  bt       Bradley-Terry logistic strengths from win/loss alone
  prkeep   points-flow PageRank with self-retention (this repo's best network
           ranker)
  pyth     Pythagorean win expectation from points for/against
  sos      strength of schedule = mean Massey rating of opponents faced

Normal equations are accumulated with sparse matrices (each design row has 3-4
non-zeros), so a full weekly refit of every system is milliseconds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize

from .rankers import _pagerank, _result_edges

PYTH_EXP = 2.37  # standard college-football Pythagorean exponent


def decayed(history: pd.DataFrame, decay: float, max_back: int = 7) -> pd.DataFrame:
    """Window of past seasons with weight decay**(seasons ago)."""
    if len(history) == 0:
        return history.assign(w=[])
    cur = int(history["season"].max())
    h = history[history["season"] >= cur - max_back].copy()
    h["w"] = np.power(float(decay), (cur - h["season"]).astype(float))
    return h


def _ridge_solve(A, b, lam, n_pen):
    """Solve (A + lam*I_{first n_pen}) x = b."""
    A = A.copy()
    idx = np.arange(n_pen)
    A[idx, idx] += lam
    A[np.arange(n_pen, A.shape[0]), np.arange(n_pen, A.shape[0])] += 1e-8
    return np.linalg.solve(A, b)


@dataclass
class Snapshot:
    teams: list
    tidx: dict
    massey: dict = field(default_factory=dict)        # decay -> (np.array ratings, hfa)
    massey_cap: dict = field(default_factory=dict)     # margin cap -> (ratings, hfa)
    massey_recent: tuple = None                        # day-decayed "current form"
    score_sd: dict = field(default_factory=dict)       # team -> sd of points scored
    gpw: dict = field(default_factory=dict)            # team -> weighted games in window
    offd: tuple = None                                 # (off, def, mu, hadv)
    bt: tuple = None                                   # (strength, hadv)
    prkeep: dict = field(default_factory=dict)         # team -> log pagerank
    pyth: dict = field(default_factory=dict)           # team -> pythag win exp
    sos: dict = field(default_factory=dict)            # team -> mean opp massey
    base: dict = field(default_factory=dict)           # fallbacks for unseen teams


def fit_all(history: pd.DataFrame, decays=(0.20, 0.45, 0.80), lam_massey=4.0,
            lam_offdef=6.0, margin_cap=28.0, bt_reg=3.0, bt_decay=0.65,
            pr_decay=0.35, pr_damping=0.85, form_halflife=45.0) -> Snapshot | None:
    """Fit every results-only rating system on `history`."""
    med = decayed(history, decays[1])
    if len(med) < 100:
        return None
    teams = sorted(set(med["home_team"]) | set(med["away_team"]))
    tidx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    snap = Snapshot(teams=teams, tidx=tidx)

    hi = med["home_team"].map(tidx).values
    ai = med["away_team"].map(tidx).values
    s = np.where(med["neutral_site"].values, 0.0, 1.0)
    y_margin = np.clip(med["margin"].values.astype(float), -margin_cap, margin_cap)
    hp = med["home_points"].values.astype(float)
    ap = med["away_points"].values.astype(float)
    g = len(med)
    rows = np.arange(g)

    # ---- Massey ridge on margin, at several memory horizons -----------------
    # design: +1 home, -1 away, home-field indicator
    X = sp.csr_matrix(
        (np.concatenate([np.ones(g), -np.ones(g), s]),
         (np.concatenate([rows, rows, rows]), np.concatenate([hi, ai, np.full(g, n)]))),
        shape=(g, n + 1))
    cur_season = int(med["season"].max())
    for d in decays:
        w = np.power(float(d), (cur_season - med["season"].values).astype(float))
        Xw = X.multiply(w[:, None]).tocsr()
        A = np.asarray((X.T @ Xw).todense())
        beta = _ridge_solve(A, np.asarray(Xw.T @ y_margin).ravel(), lam_massey, n)
        snap.massey[d] = (beta[:n], float(beta[n]))
        if d == decays[1]:
            # same normal matrix, different blowout treatment: a tight cap rates
            # close-game strength, a loose cap rewards dominance
            for cap in (14.0, 50.0):
                yc = np.clip(med["margin"].values.astype(float), -cap, cap)
                bc = _ridge_solve(A, np.asarray(Xw.T @ yc).ravel(), lam_massey, n)
                snap.massey_cap[cap] = (bc[:n], float(bc[n]))
            # within-window recency: half-life in days on top of the season decay
            dts = pd.to_datetime(med["start_date"], utc=True, errors="coerce")
            days_ago = (dts.max() - dts).dt.days.fillna(400).values
            wr = w * np.power(0.5, days_ago / form_halflife)
            Xr = X.multiply(wr[:, None]).tocsr()
            Ar = np.asarray((X.T @ Xr).todense())
            br = _ridge_solve(Ar, np.asarray(Xr.T @ y_margin).ravel(), lam_massey, n)
            snap.massey_recent = (br[:n], float(br[n]))

    # ---- offence / defence ridge on the two scorelines -----------------------
    # params: off(n), def(n), mu, home-scoring bonus
    r2 = np.arange(2 * g)
    cols = np.concatenate([hi, ai + n, np.full(g, 2 * n), np.full(g, 2 * n + 1),      # home scoring
                           ai, hi + n, np.full(g, 2 * n)])                            # away scoring
    rws = np.concatenate([rows, rows, rows, rows, rows + g, rows + g, rows + g])
    vals = np.concatenate([np.ones(g), -np.ones(g), np.ones(g), s,
                           np.ones(g), -np.ones(g), np.ones(g)])
    X2 = sp.csr_matrix((vals, (rws, cols)), shape=(2 * g, 2 * n + 2))
    w2 = np.concatenate([med["w"].values, med["w"].values])
    X2w = X2.multiply(w2[:, None]).tocsr()
    A2 = np.asarray((X2.T @ X2w).todense())
    b2 = np.asarray(X2w.T @ np.concatenate([hp, ap])).ravel()
    beta2 = _ridge_solve(A2, b2, lam_offdef, 2 * n)
    snap.offd = (beta2[:n], beta2[n:2 * n], float(beta2[2 * n]), float(beta2[2 * n + 1]))

    # ---- Bradley-Terry (win/loss only) --------------------------------------
    bt_h = decayed(history, bt_decay)
    bhi = bt_h["home_team"].map(tidx).values
    bai = bt_h["away_team"].map(tidx).values
    keep = ~(pd.isna(bhi) | pd.isna(bai))
    bhi = bhi[keep].astype(int); bai = bai[keep].astype(int)
    bs = np.where(bt_h["neutral_site"].values[keep], 0.0, 1.0)
    by = bt_h["home_win"].values[keep].astype(float)
    bw = bt_h["w"].values[keep]

    def negll(theta):
        st = theta[:n]; ha = theta[n]
        z = st[bhi] - st[bai] + ha * bs
        p = np.clip(1 / (1 + np.exp(-z)), 1e-12, 1 - 1e-12)
        loss = -(bw * (by * np.log(p) + (1 - by) * np.log(1 - p))).sum() + bt_reg * 0.5 * (st ** 2).sum()
        resid = bw * (p - by)
        gr = np.zeros(n + 1)
        np.add.at(gr, bhi, resid); np.add.at(gr, bai, -resid)
        gr[:n] += bt_reg * st
        gr[n] = (resid * bs).sum()
        return loss, gr

    res = minimize(negll, np.zeros(n + 1), jac=True, method="L-BFGS-B",
                   options={"maxiter": 120})
    snap.bt = (res.x[:n], float(res.x[n]))

    # ---- points-flow PageRank with self-retention ---------------------------
    pr_h = decayed(history, pr_decay)
    edges = []
    for r in pr_h.itertuples(index=False):
        edges.extend(_result_edges(r.home_team, r.away_team, r.home_points,
                                   r.away_points, r.home_win, r.w, "points_keep", 28.0))
    pr = _pagerank(teams, edges, damping=pr_damping)
    snap.prkeep = {t: float(np.log(max(v, 1e-12))) for t, v in pr.items()}

    # ---- Pythagorean expectation + strength of schedule ---------------------
    w = med["w"].values
    pf = np.zeros(n); pa = np.zeros(n); wt = np.zeros(n); sos_num = np.zeros(n)
    mr = snap.massey[decays[1]][0]
    np.add.at(pf, hi, w * hp); np.add.at(pa, hi, w * ap)
    np.add.at(pf, ai, w * ap); np.add.at(pa, ai, w * hp)
    np.add.at(wt, hi, w); np.add.at(wt, ai, w)
    np.add.at(sos_num, hi, w * mr[ai]); np.add.at(sos_num, ai, w * mr[hi])
    pyth = np.power(np.maximum(pf, 1e-6), PYTH_EXP) / (
        np.power(np.maximum(pf, 1e-6), PYTH_EXP) + np.power(np.maximum(pa, 1e-6), PYTH_EXP))
    # weighted scoring volatility (upset-propensity) and effective games played
    mean_f = pf / np.maximum(wt, 1e-6)
    sq = np.zeros(n)
    np.add.at(sq, hi, w * hp ** 2); np.add.at(sq, ai, w * ap ** 2)
    var_f = np.maximum(sq / np.maximum(wt, 1e-6) - mean_f ** 2, 0.0)
    snap.score_sd = {t: float(np.sqrt(var_f[tidx[t]])) for t in teams}
    snap.gpw = {t: float(wt[tidx[t]]) for t in teams}
    snap.pyth = {t: float(pyth[tidx[t]]) for t in teams}
    snap.sos = {t: float(sos_num[tidx[t]] / max(wt[tidx[t]], 1e-6)) for t in teams}
    snap.base = {
        "massey": {d: float(np.median(snap.massey[d][0])) for d in decays},
        "off": float(np.median(snap.offd[0])), "def": float(np.median(snap.offd[1])),
        "bt": float(np.median(snap.bt[0])),
        "pr": float(np.median(list(snap.prkeep.values()))),
        "pyth": float(np.median(list(snap.pyth.values()))),
        "sos": float(np.median(list(snap.sos.values()))),
        "sd": float(np.median(list(snap.score_sd.values()))),
        "gpw": float(np.median(list(snap.gpw.values()))),
    }
    return snap


def features(snap: Snapshot, home: str, away: str, neutral: bool) -> dict:
    """Rating-derived pre-game features for one upcoming matchup."""
    ti = snap.tidx
    s = 0.0 if neutral else 1.0
    f = {}
    # positional names so any set of decays works: short = quickest memory
    for nm, d in zip(("massey_short", "massey_mid", "massey_long"), sorted(snap.massey)):
        r, hfa = snap.massey[d]
        bh = r[ti[home]] if home in ti else snap.base["massey"][d]
        ba = r[ti[away]] if away in ti else snap.base["massey"][d]
        f[nm] = float(bh - ba + hfa * s)
    for cap, (r, hfa) in snap.massey_cap.items():
        bh = r[ti[home]] if home in ti else float(np.median(r))
        ba = r[ti[away]] if away in ti else float(np.median(r))
        f[f"massey_cap{int(cap)}"] = float(bh - ba + hfa * s)
    if snap.massey_recent is not None:
        r, hfa = snap.massey_recent
        bh = r[ti[home]] if home in ti else float(np.median(r))
        ba = r[ti[away]] if away in ti else float(np.median(r))
        f["massey_recent"] = float(bh - ba + hfa * s)
    sdh = snap.score_sd.get(home, snap.base["sd"]); sda = snap.score_sd.get(away, snap.base["sd"])
    f["scoresd_sum"] = float(sdh + sda); f["scoresd_diff"] = float(sdh - sda)
    f["gpw_min"] = float(min(snap.gpw.get(home, snap.base["gpw"]), snap.gpw.get(away, snap.base["gpw"])))
    off, dfn, mu, hadv = snap.offd
    oh = off[ti[home]] if home in ti else snap.base["off"]
    oa = off[ti[away]] if away in ti else snap.base["off"]
    dh = dfn[ti[home]] if home in ti else snap.base["def"]
    da = dfn[ti[away]] if away in ti else snap.base["def"]
    exp_h = mu + oh - da + hadv * s
    exp_a = mu + oa - dh
    f["offdef_margin"] = float(exp_h - exp_a)
    f["offdef_total"] = float(exp_h + exp_a)
    st, bha = snap.bt
    sh = st[ti[home]] if home in ti else snap.base["bt"]
    sa = st[ti[away]] if away in ti else snap.base["bt"]
    f["bt_diff"] = float(sh - sa + bha * s)
    f["pr_diff"] = float(snap.prkeep.get(home, snap.base["pr"]) - snap.prkeep.get(away, snap.base["pr"]))
    f["pyth_diff"] = float(snap.pyth.get(home, snap.base["pyth"]) - snap.pyth.get(away, snap.base["pyth"]))
    f["sos_diff"] = float(snap.sos.get(home, snap.base["sos"]) - snap.sos.get(away, snap.base["sos"]))
    f["home_ind"] = s
    return f
