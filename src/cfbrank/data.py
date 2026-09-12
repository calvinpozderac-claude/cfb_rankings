"""Data loading and cleaning for the CFB ranking project.

Source: cfbfastR-data (https://github.com/sportsdataverse/cfbfastR-data), a public
mirror of CollegeFootballData.com. We use:
  * schedules/csv/cfb_schedules_YYYY.csv  -> game results (2001-2025)
  * betting/csv/cfb_line_odds.csv.gz      -> sportsbook lines (2006-2025)

The betting file is in long format and carries only a per-row team *abbreviation*
(no team id), so we recover an abbreviation -> team-name map by majority vote
(the true team is the one common to every game an abbreviation appears in) in
order to place each spread on the home team's perspective.
"""
from __future__ import annotations

import glob
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# Repo checkout produced by the session's git proxy.
CFBFASTR = "/home/user/sportsdataverse/cfbfastr-data"
SCHED_GLOB = os.path.join(CFBFASTR, "schedules", "csv", "cfb_schedules_*.csv")
LINES_GZ = os.path.join(CFBFASTR, "betting", "csv", "cfb_line_odds.csv.gz")

# Cache location inside this project.
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
GAMES_CACHE = os.path.abspath(os.path.join(DATA_DIR, "games.csv.gz"))

SEASON_TYPE_ORDER = {"regular": 0, "postseason": 1}


def _load_schedules() -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(SCHED_GLOB)):
        year = int(os.path.basename(f).split("_")[-1].split(".")[0])
        df = pd.read_csv(f, low_memory=False)
        df["_year"] = year
        frames.append(df)
    sched = pd.concat(frames, ignore_index=True)
    return sched


def _consensus_spread(lines_path: str) -> pd.DataFrame:
    """Return game_id -> consensus closing spread from the HOME team's perspective.

    Convention: spread_home < 0 means the home team is favored (e.g. -7 = home by 7).
    """
    cols = [
        "game_id", "season", "market_type", "abbr", "lines",
        "home_team_id", "away_team_id", "week", "season_type",
    ]
    lo = pd.read_csv(lines_path, usecols=cols, low_memory=False)
    lo = lo[lo["market_type"] == "spread"].copy()
    lo["lines"] = pd.to_numeric(lo["lines"], errors="coerce")
    lo = lo.dropna(subset=["lines", "game_id", "abbr"])
    lo["game_id"] = lo["game_id"].astype(np.int64)
    return lo


def _abbr_to_team(sched: pd.DataFrame, lo: pd.DataFrame) -> dict:
    """Majority-vote map: betting abbreviation -> full team name.

    For each abbr we tally the team names of every game it appears in. The abbr's
    real team is the name present in (almost) all of those games; opponents vary.
    """
    gid_teams = (
        sched.dropna(subset=["game_id"])
        .assign(game_id=lambda d: d["game_id"].astype(np.int64))
        .set_index("game_id")[["home_team", "away_team"]]
        .to_dict("index")
    )
    votes: dict[str, Counter] = defaultdict(Counter)
    # one row per (game_id, abbr) is enough for voting
    pairs = lo[["game_id", "abbr"]].drop_duplicates()
    for gid, abbr in pairs.itertuples(index=False):
        teams = gid_teams.get(int(gid))
        if not teams:
            continue
        for name in (teams["home_team"], teams["away_team"]):
            if isinstance(name, str):
                votes[abbr][name] += 1
    return {abbr: c.most_common(1)[0][0] for abbr, c in votes.items() if c}


def build_games(force: bool = False) -> pd.DataFrame:
    """Build the clean, analysis-ready games table (with caching)."""
    if os.path.exists(GAMES_CACHE) and not force:
        return pd.read_csv(GAMES_CACHE, low_memory=False)

    sched = _load_schedules()
    keep = [
        "game_id", "season", "week", "season_type", "start_date", "neutral_site",
        "conference_game", "home_team", "home_conference", "home_division", "home_points",
        "away_team", "away_conference", "away_division", "away_points",
        "home_pregame_elo", "away_pregame_elo", "completed",
    ]
    g = sched[keep].copy()
    g = g.dropna(subset=["game_id", "home_team", "away_team"])
    g["game_id"] = g["game_id"].astype(np.int64)
    g = g[g["completed"] == True]  # noqa: E712
    g = g.dropna(subset=["home_points", "away_points"])
    g = g[g["home_points"] != g["away_points"]]  # drop the (rare) recorded ties
    g["home_points"] = g["home_points"].astype(int)
    g["away_points"] = g["away_points"].astype(int)
    g["home_win"] = (g["home_points"] > g["away_points"]).astype(int)
    g["margin"] = g["home_points"] - g["away_points"]  # home perspective
    g["neutral_site"] = g["neutral_site"].fillna(False).astype(bool)
    g["start_date"] = pd.to_datetime(g["start_date"], errors="coerce", utc=True)

    # ---- attach consensus closing spread (home perspective) ----
    lo = _consensus_spread(LINES_GZ)
    a2t = _abbr_to_team(sched, lo)
    lo["team"] = lo["abbr"].map(a2t)
    home_name = g.set_index("game_id")["home_team"].to_dict()
    lo["home_name"] = lo["game_id"].map(home_name)
    home_rows = lo[lo["team"] == lo["home_name"]]
    spread_home = home_rows.groupby("game_id")["lines"].median()
    g["spread_home"] = g["game_id"].map(spread_home)

    # ---- global chronological ordering for walk-forward ----
    g["st_order"] = g["season_type"].map(SEASON_TYPE_ORDER).fillna(0).astype(int)
    g = g.sort_values(["season", "st_order", "week", "start_date", "game_id"])
    g = g.reset_index(drop=True)
    # integer period id, unique & increasing per (season, season_type, week)
    period_keys = g[["season", "st_order", "week"]].drop_duplicates().reset_index(drop=True)
    period_keys["period"] = np.arange(len(period_keys))
    g = g.merge(period_keys, on=["season", "st_order", "week"], how="left")

    os.makedirs(DATA_DIR, exist_ok=True)
    g.to_csv(GAMES_CACHE, index=False, compression="gzip")
    return g


def is_fbs(g: pd.DataFrame) -> pd.Series:
    return (g["home_division"] == "fbs") & (g["away_division"] == "fbs")


if __name__ == "__main__":
    g = build_games(force=True)
    print(f"games: {len(g):,}  seasons {g.season.min()}-{g.season.max()}")
    fbs = g[is_fbs(g)]
    print(f"FBS-vs-FBS games: {len(fbs):,}")
    print(f"with closing spread: {fbs['spread_home'].notna().sum():,}")
    print(f"with CFBD pregame Elo: {fbs['home_pregame_elo'].notna().sum():,}")
    # quick sanity: does spread sign agree with outcome more often than not?
    sub = fbs.dropna(subset=["spread_home"])
    fav_home = sub["spread_home"] < 0
    acc = (fav_home == (sub["home_win"] == 1)).mean()
    print(f"spread-favorite straight-up accuracy (sanity): {acc:.3f} on {len(sub):,} games")
