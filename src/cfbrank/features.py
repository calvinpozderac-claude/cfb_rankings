"""Pre-game feature construction (single chronological pass, no leakage).

Every feature for a game is computed from information available *before* kickoff:
season-to-date records, scoring form, rest, and experience. Season counters
reset each year.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

MARGIN_CAP = 28.0


def build_features(games: pd.DataFrame) -> pd.DataFrame:
    """Return per-game pre-game features indexed by game_id."""
    wins = defaultdict(int)      # season wins
    losses = defaultdict(int)
    pts_for = defaultdict(float)
    pts_against = defaultdict(float)
    gp = defaultdict(int)        # games played this season
    last_date = {}               # last game datetime (for rest)
    cur_season = None

    rows = []
    for r in games.itertuples(index=False):
        if cur_season is None:
            cur_season = r.season
        elif r.season != cur_season:
            wins.clear(); losses.clear(); pts_for.clear()
            pts_against.clear(); gp.clear(); last_date.clear()
            cur_season = r.season
        h, a = r.home_team, r.away_team

        def wp(t):
            g = gp[t]
            return wins[t] / g if g else 0.5

        def avgm(t):  # avg (capped) scoring margin so far
            g = gp[t]
            if not g:
                return 0.0
            return np.clip((pts_for[t] - pts_against[t]) / g, -MARGIN_CAP, MARGIN_CAP)

        rest_h = rest_a = np.nan
        if isinstance(r.start_date, pd.Timestamp) or (r.start_date == r.start_date):
            d = pd.Timestamp(r.start_date)
            if h in last_date:
                rest_h = (d - last_date[h]).days
            if a in last_date:
                rest_a = (d - last_date[a]).days

        rows.append({
            "game_id": r.game_id,
            "home_ind": 0.0 if r.neutral_site else 1.0,
            "neutral": 1.0 if r.neutral_site else 0.0,
            "conf_game": 1.0 if getattr(r, "conference_game", False) else 0.0,
            "wp_diff": wp(h) - wp(a),
            "margin_diff": avgm(h) - avgm(a),
            "gp_home": gp[h],
            "gp_away": gp[a],
            "rest_diff": (rest_h - rest_a) if (rest_h == rest_h and rest_a == rest_a) else 0.0,
        })

        # update after recording pre-game state
        gp[h] += 1; gp[a] += 1
        pts_for[h] += r.home_points; pts_against[h] += r.away_points
        pts_for[a] += r.away_points; pts_against[a] += r.home_points
        if r.home_win == 1:
            wins[h] += 1; losses[a] += 1
        else:
            wins[a] += 1; losses[h] += 1
        if isinstance(r.start_date, pd.Timestamp) or (r.start_date == r.start_date):
            d = pd.Timestamp(r.start_date)
            last_date[h] = d; last_date[a] = d

    return pd.DataFrame(rows).set_index("game_id")
