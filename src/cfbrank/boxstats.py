"""Team-game box-score efficiency and per-player production from cfbfastR
play-level player stats (2014-2025).

Every stat is attributed to the *player's* roster team (the file's `team`
column is not reliably the offense), so offense and defense for each game are
reconstructed from player ids.

Outputs (cached under data/):
  boxstats.csv.gz          one row per (game_id, team) with offensive efficiency
                           plus the same stats *allowed* (opponent's offense)
  player_production.csv.gz one row per (season, athlete_id, team) with offensive
                           yards and defensive plays -> returning production
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

# Raw cfbfastR-data files (player_stats/csv, rosters/csv, team_info/parquet), fetched
# with `git show` from the public repo; override with CFB_RAW_DIR.
RAW = os.environ.get("CFB_RAW_DIR",
                     "/tmp/claude-0/-home-user-cfb-rankings/df3be1ef-a416-5c7a-878c-54267a1c0a5d/scratchpad/raw")
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
BOX_CACHE = os.path.join(DATA_DIR, "boxstats.csv.gz")
PROD_CACHE = os.path.join(DATA_DIR, "player_production.csv.gz")

SEASONS = range(2014, 2026)


def _roster_map(season: int) -> dict:
    ro = pd.read_csv(os.path.join(RAW, f"rosters_{season}.csv"), usecols=["athlete_id", "team"])
    return ro.drop_duplicates("athlete_id").set_index("athlete_id")["team"].to_dict()


def _success(gain, down, dist):
    need = np.where(down == 1, 0.5 * dist, np.where(down == 2, 0.7 * dist, dist))
    return (gain >= need) & (dist > 0)


def _season_boxstats(season: int):
    ps = pd.read_csv(os.path.join(RAW, f"player_stats_{season}.csv"), low_memory=False)
    id2t = _roster_map(season)

    def team_of(col):
        return ps[col].map(id2t)

    # ---- offensive play attribution ----
    off_team = team_of("rush_player_id")
    for c in ["completion_player_id", "incompletion_player_id", "sack_taken_player_id",
              "interception_thrown_player_id"]:
        off_team = off_team.fillna(team_of(c))
    is_rush = ps["rush_player_id"].notna()
    is_pass = ps["completion_player_id"].notna() | ps["incompletion_player_id"].notna() \
        | ps["interception_thrown_player_id"].notna()
    is_sack = ps["sack_taken_player_id"].notna() & ~is_rush & ~is_pass
    is_play = is_rush | is_pass | is_sack
    gain = np.where(is_rush, ps["rush_yds"].fillna(0),
                    np.where(ps["completion_player_id"].notna(), ps["completion_yds"].fillna(0),
                             np.where(is_sack, -ps["sack_taken_stat"].fillna(0), 0.0)))
    down = ps["down"].fillna(1).values
    dist = ps["distance"].fillna(10).values
    succ = _success(gain, down, dist)
    explosive = (is_rush & (gain >= 12)) | (is_pass & (gain >= 20))
    int_thrown = ps["interception_thrown_player_id"].notna()
    # fumbles: lost if recovered by another team; unknown recovery -> 0.5 expected
    fum_team = team_of("fumble_player_id")
    rec_team = team_of("fumble_recovered_player_id")
    fum = ps["fumble_player_id"].notna()
    fum_lost = np.where(fum & rec_team.notna(), (rec_team != fum_team).astype(float),
                        np.where(fum, 0.5, 0.0))
    # a fumble row may not carry an offensive attribution; credit it to fumbler's team
    off_team = off_team.fillna(fum_team)

    plays = pd.DataFrame({
        "game_id": ps["game_id"], "team": off_team,
        "play": is_play.astype(float), "yards": np.where(is_play, gain, 0.0),
        "success": (succ & is_play).astype(float), "explosive": explosive.astype(float),
        "pass_play": is_pass.astype(float), "rush_play": is_rush.astype(float),
        "sack_taken": is_sack.astype(float), "int_thrown": int_thrown.astype(float),
        "fum_lost": fum_lost, "td": ps["touchdown_player_id"].notna().astype(float),
    }).dropna(subset=["team"])
    off = plays.groupby(["game_id", "team"]).sum().reset_index()
    off["season"] = season

    # ---- defensive havoc credited to the defender's team ----
    hav = pd.DataFrame({
        "game_id": ps["game_id"],
        "sack_team": team_of("sack_player_id"), "int_team": team_of("interception_player_id"),
        "pbu_team": team_of("pass_breakup_player_id"), "ff_team": team_of("fumble_forced_player_id"),
    })
    parts = []
    for c, name in [("sack_team", "sacks"), ("int_team", "ints"), ("pbu_team", "pbus"), ("ff_team", "ffs")]:
        d = hav.dropna(subset=[c]).groupby(["game_id", c]).size().rename(name).reset_index()
        parts.append(d.rename(columns={c: "team"}))
    havoc = parts[0]
    for p in parts[1:]:
        havoc = havoc.merge(p, on=["game_id", "team"], how="outer")
    off = off.merge(havoc, on=["game_id", "team"], how="left").fillna(
        {"sacks": 0, "ints": 0, "pbus": 0, "ffs": 0})

    # ---- per-player production (for returning production next season) ----
    prod_parts = []
    for c, ycol in [("rush_player_id", "rush_yds"), ("completion_player_id", "completion_yds"),
                    ("reception_player_id", "reception_yds")]:
        d = ps.dropna(subset=[c])[[c, ycol]].copy()
        d.columns = ["athlete_id", "yds"]
        d["yds"] = d["yds"].fillna(0)
        prod_parts.append(d)
    offp = pd.concat(prod_parts).groupby("athlete_id")["yds"].sum().rename("off_yards")
    defp_parts = []
    for c in ["sack_player_id", "interception_player_id", "pass_breakup_player_id",
              "fumble_forced_player_id"]:
        defp_parts.append(ps.dropna(subset=[c])[c].rename("athlete_id"))
    defp = pd.concat(defp_parts).value_counts().rename("def_plays")
    prod = pd.concat([offp, defp], axis=1).fillna(0).reset_index().rename(columns={"index": "athlete_id"})
    prod["team"] = prod["athlete_id"].map(id2t)
    prod["season"] = season
    return off, prod.dropna(subset=["team"])


def build_boxstats(force: bool = False):
    if os.path.exists(BOX_CACHE) and os.path.exists(PROD_CACHE) and not force:
        return pd.read_csv(BOX_CACHE), pd.read_csv(PROD_CACHE)
    offs, prods = [], []
    for s in SEASONS:
        o, p = _season_boxstats(s)
        offs.append(o); prods.append(p)
        print(f"  boxstats {s}: {len(o):,} team-games")
    off = pd.concat(offs, ignore_index=True)
    # attach the *allowed* side: opponent's offense in the same game
    opp = off[["game_id", "team", "play", "yards", "success", "explosive", "int_thrown",
               "fum_lost", "sack_taken"]].copy()
    opp.columns = ["game_id", "opp", "d_play", "d_yards", "d_success", "d_explosive",
                   "d_int_forced", "d_fum_forced", "d_sacks_made"]
    two = off.merge(off[["game_id", "team"]].rename(columns={"team": "opp"}), on="game_id")
    two = two[two["team"] != two["opp"]]
    box = two.merge(opp, on=["game_id", "opp"], how="left")
    box = box.drop_duplicates(["game_id", "team"])
    os.makedirs(DATA_DIR, exist_ok=True)
    box.to_csv(BOX_CACHE, index=False, compression="gzip")
    prod = pd.concat(prods, ignore_index=True)
    prod.to_csv(PROD_CACHE, index=False, compression="gzip")
    return box, prod


def game_margins(box: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Per-game home-minus-away efficiency margins, indexed by game_id."""
    b = box.set_index(["game_id", "team"])

    def rate(num, den):
        return np.where(den > 0, num / np.maximum(den, 1), np.nan)

    rows = []
    for r in games[["game_id", "home_team", "away_team"]].itertuples(index=False):
        try:
            h = b.loc[(r.game_id, r.home_team)]; a = b.loc[(r.game_id, r.away_team)]
        except KeyError:
            continue
        if h["play"] < 20 or a["play"] < 20:
            continue
        rows.append({
            "game_id": r.game_id,
            "ypp_margin": h["yards"] / h["play"] - a["yards"] / a["play"],
            "success_margin": h["success"] / h["play"] - a["success"] / a["play"],
            "explosive_margin": h["explosive"] / h["play"] - a["explosive"] / a["play"],
            "turnover_margin": (a["int_thrown"] + a["fum_lost"]) - (h["int_thrown"] + h["fum_lost"]),
            "havoc_margin": (h["sacks"] + h["pbus"] + h["ffs"]) / max(a["play"], 1)
                            - (a["sacks"] + a["pbus"] + a["ffs"]) / max(h["play"], 1),
            "plays_total": h["play"] + a["play"],
        })
    return pd.DataFrame(rows).set_index("game_id")


if __name__ == "__main__":
    box, prod = build_boxstats(force=True)
    print(f"boxstats rows {len(box):,}; player production rows {len(prod):,}")
    print(box.describe().T[["mean", "min", "max"]].round(2).to_string())
