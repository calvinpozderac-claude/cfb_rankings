"""Offseason and situational features.

* Returning production (per team-season, known preseason): share of last
  season's offensive yards / defensive plays produced by players on this
  season's roster, plus raw roster continuity. This is the kind of "offseason
  information" the betting market has in September and results-only ratings
  do not.
* Travel: great-circle distance the away team travels, time-zone shift and
  elevation change, from team_info venue coordinates.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

# Raw cfbfastR-data files (player_stats/csv, rosters/csv, team_info/parquet), fetched
# with `git show` from the public repo; override with CFB_RAW_DIR.
RAW = os.environ.get("CFB_RAW_DIR",
                     "/tmp/claude-0/-home-user-cfb-rankings/df3be1ef-a416-5c7a-878c-54267a1c0a5d/scratchpad/raw")

TZ_OFFSET = {  # standard-time UTC offsets (hours) for the time zones that occur
    "America/New_York": -5, "America/Detroit": -5, "America/Indiana/Indianapolis": -5,
    "America/Kentucky/Louisville": -5, "America/Chicago": -6, "America/Denver": -7,
    "America/Phoenix": -7, "America/Boise": -7, "America/Los_Angeles": -8,
    "America/Anchorage": -9, "Pacific/Honolulu": -10, "America/Indiana/Knox": -6,
    "America/Menominee": -6, "America/North_Dakota/Center": -6,
}


def returning_production(prod: pd.DataFrame, seasons=range(2015, 2026)) -> pd.DataFrame:
    """Per (season, team): ret_off, ret_def in [0,1] and roster_continuity."""
    rows = []
    prev_ro = None
    for s in seasons:
        ro = pd.read_csv(os.path.join(RAW, f"rosters_{s}.csv"), usecols=["athlete_id", "team"])
        ro = ro.drop_duplicates("athlete_id")
        on_roster = set(zip(ro["team"], ro["athlete_id"]))
        last = prod[prod["season"] == s - 1]
        if len(last) == 0:
            continue
        last = last.assign(back=[(t, a) in on_roster for t, a in zip(last["team"], last["athlete_id"])])
        agg = last.groupby("team").apply(
            lambda d: pd.Series({
                "ret_off": (d.off_yards * d.back).sum() / max(d.off_yards.sum(), 1.0),
                "ret_def": (d.def_plays * d.back).sum() / max(d.def_plays.sum(), 1.0),
            }), include_groups=False).reset_index()
        agg["season"] = s
        if prev_ro is not None:
            prev_sets = prev_ro.groupby("team")["athlete_id"].apply(set).to_dict()
            cur_sets = ro.groupby("team")["athlete_id"].apply(set).to_dict()
            agg["roster_continuity"] = agg["team"].map(
                lambda t: len(prev_sets.get(t, set()) & cur_sets.get(t, set()))
                / max(len(prev_sets.get(t, set())), 1))
        else:
            agg["roster_continuity"] = np.nan
        rows.append(agg)
        prev_ro = ro
    return pd.concat(rows, ignore_index=True)


def _haversine(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def travel_features(games: pd.DataFrame) -> pd.DataFrame:
    """Per game_id: away_travel_km, tz_shift (hours, away relative to home),
    elev_diff (m, home venue minus away home venue). Neutral sites -> 0."""
    info = {}
    for s in range(2013, 2026):
        f = os.path.join(RAW, f"team_info_{s}.parquet")
        if not os.path.exists(f):
            continue
        ti = pd.read_parquet(f, columns=["school", "latitude", "longitude", "timezone", "elevation"])
        for r in ti.itertuples(index=False):
            info[(s, r.school)] = r
            info.setdefault(("any", r.school), r)
    rows = []
    for r in games[["game_id", "season", "home_team", "away_team", "neutral_site"]].itertuples(index=False):
        h = info.get((r.season, r.home_team)) or info.get(("any", r.home_team))
        a = info.get((r.season, r.away_team)) or info.get(("any", r.away_team))
        if r.neutral_site or h is None or a is None or pd.isna(h.latitude) or pd.isna(a.latitude):
            rows.append({"game_id": r.game_id, "away_travel_km": 0.0, "tz_shift": 0.0, "elev_diff": 0.0})
            continue
        km = float(_haversine(a.latitude, a.longitude, h.latitude, h.longitude))
        tz = TZ_OFFSET.get(h.timezone, np.nan) - TZ_OFFSET.get(a.timezone, np.nan)
        el = (float(h.elevation) if pd.notna(h.elevation) else 0.0) - \
             (float(a.elevation) if pd.notna(a.elevation) else 0.0)
        rows.append({"game_id": r.game_id, "away_travel_km": km,
                     "tz_shift": 0.0 if pd.isna(tz) else float(tz), "elev_diff": el})
    return pd.DataFrame(rows).set_index("game_id")
