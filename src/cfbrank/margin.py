"""Margin-based (least-squares / Massey-style) team ratings.

Solve  min_r  sum_g w_g (margin_g - (r_home - r_away + hfa*home_ind))^2 + lam*||r||^2

i.e. each team gets a rating in *points*, and the predicted margin of a game is
the rating difference plus home-field. This is the workhorse behind SRS, Massey
and Sagarin-style systems. Weights carry a cross-season recency decay and an
optional within-season "form" half-life (days).

The predictor object exposes both a win probability and a predicted margin so
downstream models (GBM / stacker) can use the margin directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .rankers import _decayed_history


class MarginPredictor:
    def __init__(self, rating: dict, hfa: float, calib, base: float):
        self.rating, self.hfa, self.calib, self.base = rating, hfa, calib, base

    def predict_margin(self, home, away, neutral) -> float:
        return (self.rating.get(home, self.base) - self.rating.get(away, self.base)
                + (0.0 if neutral else self.hfa))

    def __call__(self, home, away, neutral) -> float:
        if self.calib is None:
            return 0.5
        m = self.predict_margin(home, away, neutral)
        return float(self.calib.predict_proba([[m]])[0, 1])


def make_ridge_margin(lam: float = 4.0, decay: float = 0.5, cap: float | None = 35.0,
                      form_halflife_days: float | None = None, target: str = "margin"):
    """Factory: fit(history) -> MarginPredictor.

    target              : per-game home-minus-away quantity to rate teams on. "margin"
                          (points) by default; any efficiency margin column works
                          (yards/play, success rate, turnovers ...). Rows with a
                          missing target are ignored.
    lam                 : ridge strength on team ratings (target^2 scale).
    decay               : cross-season recency (see rankers._decayed_history).
    cap                 : clip margins to +/-cap so blowouts don't dominate.
    form_halflife_days  : if set, additionally down-weight older games within the
                          window by 0.5**(days_ago/halflife) -> a "recent form" model.
    """
    def fit(history: pd.DataFrame):
        h = _decayed_history(history, decay)
        if target in h.columns:
            h = h.dropna(subset=[target])
        if len(h) < 50 or target not in h.columns:
            return MarginPredictor({}, 0.0, None, 0.0)
        w = h["w"].values.astype(float)
        if form_halflife_days:
            d = pd.to_datetime(h["start_date"], utc=True, errors="coerce")
            days_ago = (d.max() - d).dt.days.fillna(365).values
            w = w * np.power(0.5, days_ago / form_halflife_days)
        teams = sorted(set(h["home_team"]) | set(h["away_team"]))
        tidx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        hi = np.array([tidx[t] for t in h["home_team"]])
        ai = np.array([tidx[t] for t in h["away_team"]])
        homeind = np.where(h["neutral_site"].values, 0.0, 1.0)
        y = h[target].values.astype(float)
        if cap:
            y = np.clip(y, -cap, cap)
        # design: X[:, tidx] = +1 home / -1 away ; last column = home indicator
        X = np.zeros((len(h), n + 1))
        X[np.arange(len(h)), hi] = 1.0
        X[np.arange(len(h)), ai] = -1.0
        X[:, n] = homeind
        XtW = X.T * w
        A = XtW @ X
        A[np.arange(n), np.arange(n)] += lam          # ridge on team ratings only
        A[n, n] += 1e-6
        b = XtW @ y
        theta = np.linalg.solve(A, b)
        r = theta[:n]
        hfa = float(theta[n])
        rating = {t: float(r[tidx[t]]) for t in teams}
        base = float(np.median(r))
        # calibrate predicted margin -> P(home win) on the training games
        pred_m = r[hi] - r[ai] + hfa * homeind
        calib = LogisticRegression(C=1.0, max_iter=500)
        calib.fit(pred_m.reshape(-1, 1), h["home_win"].values, sample_weight=w)
        return MarginPredictor(rating, hfa, calib, base)

    return fit
