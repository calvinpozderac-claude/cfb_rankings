"""Online Elo rating family.

A single chronological pass over every game: predict the home-win probability
from the ratings as they stand *before* kickoff, then update. Because the
prediction for each game uses only prior information, the per-game probabilities
are exactly the walk-forward (train-through-last-week) predictions we want.

Options
-------
k                 : update step size.
home_field        : Elo points added to the home team pre-game (0 at neutral sites).
mov              : if True, use a 538-style margin-of-victory multiplier.
preseason_regress : fraction pulled back toward the mean between seasons
                    (0 = carry ratings fully, 1 = full reset).
scale             : logistic scale (400 = classic Elo).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

INIT = 1500.0


def expected(diff: float, scale: float = 400.0) -> float:
    return 1.0 / (1.0 + 10.0 ** (-diff / scale))


def run_elo(
    games: pd.DataFrame,
    k: float = 40.0,
    home_field: float = 65.0,
    mov: bool = True,
    preseason_regress: float = 0.25,
    scale: float = 400.0,
    team_regress: dict | None = None,
) -> pd.DataFrame:
    """Return per-game DataFrame indexed by game_id.

    team_regress: optional {(season, team): regress} overriding the preseason
    regression per team (e.g. driven by returning production: a team that lost
    most of its production is pulled harder toward the mean).

    Columns: elo_prob (pre-game home win prob), elo_home_pre, elo_away_pre.
    ``games`` must be sorted chronologically (see data.build_games).
    """
    rating: dict[str, float] = defaultdict(lambda: INIT)
    cur_season = None
    out_gid, out_prob, out_h, out_a = [], [], [], []

    for row in games.itertuples(index=False):
        season = row.season
        if cur_season is None:
            cur_season = season
        elif season != cur_season:
            # between-season regression toward the mean
            for t in list(rating.keys()):
                reg = preseason_regress
                if team_regress is not None:
                    reg = team_regress.get((season, t), preseason_regress)
                rating[t] = INIT + (1 - reg) * (rating[t] - INIT)
            cur_season = season

        h, a = row.home_team, row.away_team
        rh, ra = rating[h], rating[a]
        hfa = 0.0 if row.neutral_site else home_field
        p_home = expected((rh + hfa) - ra, scale)

        out_gid.append(row.game_id)
        out_prob.append(p_home)
        out_h.append(rh)
        out_a.append(ra)

        # update
        actual = row.home_win  # 1 if home won
        if mov:
            margin = abs(row.margin)
            winner_diff = (rh - ra) if actual == 1 else (ra - rh)
            # +hfa affects the winner_diff too, keep it simple/robust:
            mult = np.log(margin + 1.0) * (2.2 / (0.001 * winner_diff + 2.2))
        else:
            mult = 1.0
        delta = k * mult * (actual - p_home)
        rating[h] = rh + delta
        rating[a] = ra - delta

    return pd.DataFrame(
        {"elo_prob": out_prob, "elo_home_pre": out_h, "elo_away_pre": out_a},
        index=pd.Index(out_gid, name="game_id"),
    )


def final_ratings(
    games: pd.DataFrame, **kwargs
) -> pd.Series:
    """Convenience: end-state Elo ratings after processing all ``games``."""
    rating: dict[str, float] = defaultdict(lambda: INIT)
    k = kwargs.get("k", 40.0)
    home_field = kwargs.get("home_field", 65.0)
    mov = kwargs.get("mov", True)
    preseason_regress = kwargs.get("preseason_regress", 0.25)
    scale = kwargs.get("scale", 400.0)
    cur_season = None
    for row in games.itertuples(index=False):
        if cur_season is None:
            cur_season = row.season
        elif row.season != cur_season:
            for t in list(rating.keys()):
                rating[t] = INIT + (1 - preseason_regress) * (rating[t] - INIT)
            cur_season = row.season
        h, a = row.home_team, row.away_team
        rh, ra = rating[h], rating[a]
        hfa = 0.0 if row.neutral_site else home_field
        p = expected((rh + hfa) - ra, scale)
        actual = row.home_win
        if mov:
            margin = abs(row.margin)
            wd = (rh - ra) if actual == 1 else (ra - rh)
            mult = np.log(margin + 1.0) * (2.2 / (0.001 * wd + 2.2))
        else:
            mult = 1.0
        d = k * mult * (actual - p)
        rating[h] = rh + d
        rating[a] = ra - d
    return pd.Series(rating).sort_values(ascending=False)
