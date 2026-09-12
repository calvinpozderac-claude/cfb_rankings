"""Where does our best results-only model out- or under-perform the Vegas line?

Model: stack_final (best results-only, 0.544 log loss). Benchmark: closing line.
We isolate games where the two DISAGREE on the straight-up winner and profile the
two buckets -- MODEL_WINS (our pick right, Vegas wrong) and VEGAS_WINS (vice
versa) -- across pre-game descriptors, then surface example games.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
MODEL = "stack_final"
LO, HI = 2019, 2025

b = pd.read_csv(os.path.join(RES, "experiments_predictions.csv.gz"))
r = b[(b.season >= LO) & (b.season <= HI)].dropna(subset=[MODEL, "market", "spread_home"]).copy()

r["m_home"] = r[MODEL] >= 0.5          # model picks home
r["v_home"] = r["market"] >= 0.5       # market picks home
r["home_won"] = r["home_win"] == 1
r["m_right"] = r["m_home"] == r["home_won"]
r["v_right"] = r["v_home"] == r["home_won"]
r["disagree"] = r["m_home"] != r["v_home"]
r["conf"] = (r[MODEL] - r["market"]).abs()          # how far apart the two are
r["fav_spread"] = r["spread_home"].abs()            # market's confidence, in points
# in a disagreement the model always backs the market underdog -> its edge in points:
r["model_edge_pts"] = np.where(r["m_home"], -r["spread_home"], r["spread_home"])

dis = r[r["disagree"]].copy()
dis["bucket"] = np.where(dis["m_right"], "MODEL_WINS", "VEGAS_WINS")
print(f"=== 2019-2025, {MODEL} vs closing line ===")
print(f"games: {len(r)}   agree: {(~r.disagree).sum()} ({(~r.disagree).mean()*100:.1f}%)   "
      f"disagree: {len(dis)} ({r.disagree.mean()*100:.1f}%)")
print(f"on AGREED games, both right: {r[~r.disagree].m_right.mean()*100:.1f}%")
print(f"on DISAGREED games: MODEL right {dis.m_right.mean()*100:.1f}%  ({dis.m_right.sum()} of {len(dis)})")
print()

# ---- profiles of the two buckets ----
def prof(col, fn=np.mean, pct=False):
    a = fn(dis[dis.bucket == "MODEL_WINS"][col]); v = fn(dis[dis.bucket == "VEGAS_WINS"][col])
    s = "{:.1%}".format if pct else "{:.2f}".format
    return s(a), s(v)

print("descriptor                       MODEL_WINS   VEGAS_WINS")
for lbl, col, pct in [
    ("market spread on game (|pts|)", "fav_spread", False),
    ("model's edge vs line (pts)", "model_edge_pts", False),
    ("model-market prob gap", "conf", False),
    ("week of season", "week", False),
    ("model backs the home team", "m_home", True),
    ("neutral site", "neutral_site", True),
    ("returning-prod diff (home-away)", "ret_off_diff", False),
    ("actual final margin (|pts|)", "margin", False),
]:
    d = dis.copy()
    if col == "margin":
        d["margin"] = d["margin"].abs()
    a = d[d.bucket == "MODEL_WINS"][col].mean()
    v = d[d.bucket == "VEGAS_WINS"][col].mean()
    if pct:
        print(f"{lbl:32s}   {a*100:8.1f}%   {v*100:8.1f}%")
    else:
        print(f"{lbl:32s}   {a:8.2f}   {v:8.2f}")

# ---- model win-rate on disagreements, by pre-game bucket ----
def winrate_by(col, edges, labels):
    dis["_b"] = pd.cut(dis[col], edges, labels=labels)
    g = dis.groupby("_b", observed=True).agg(n=("m_right", "size"), model_right=("m_right", "mean"))
    g["model_right"] = (g["model_right"] * 100).round(1)
    return g

print("\n--- model win-rate on disagreements, by market spread (line points) ---")
print(winrate_by("fav_spread", [0, 3, 7, 14, 100], ["pick'em 0-3", "3-7", "7-14", "14+"]).to_string())
print("\n--- by week of season ---")
print(winrate_by("week", [0, 3, 8, 20], ["1-3 (early)", "4-8 (mid)", "9+ (late)"]).to_string())
print("\n--- by model-market prob gap (model conviction) ---")
print(winrate_by("conf", [0, .05, .1, .2, 1], ["<.05", ".05-.10", ".10-.20", ".20+"]).to_string())

# home/away split of the disagreement
print("\n--- who the model backs on disagreements ---")
for side, sub in [("model backs HOME (line has road fav)", dis[dis.m_home]),
                  ("model backs ROAD (line has home fav)", dis[~dis.m_home])]:
    print(f"{side:40s} n={len(sub):3d}  model right {sub.m_right.mean()*100:.1f}%")

# ---- example games ----
def fmt(row):
    fav = row.home_team if row.spread_home < 0 else row.away_team
    dog = row.away_team if row.spread_home < 0 else row.home_team
    winner = row.home_team if row.home_won else row.away_team
    mp = row.home_team if row.m_home else row.away_team
    return (f"{int(row.season)} wk{int(row.week):2d}  {row.away_team} @ {row.home_team:22s} "
            f"line: {fav} -{abs(row.spread_home):.1f}  model->{mp:15s} "
            f"won: {winner:15s} ({row.home_points:.0f}-{row.away_points:.0f})")

_g = pd.read_csv(os.path.join(os.path.dirname(RES), "data", "games.csv.gz"),
                 usecols=["game_id", "home_points", "away_points"])
dis = dis.merge(_g, on="game_id", how="left")
print("\n=== biggest MODEL wins (model most confident, beat the line) ===")
for _, x in dis[dis.m_right].sort_values("conf", ascending=False).head(12).iterrows():
    print("  " + fmt(x))
print("\n=== biggest VEGAS wins (model most confident, but wrong) ===")
for _, x in dis[~dis.m_right].sort_values("conf", ascending=False).head(12).iterrows():
    print("  " + fmt(x))

dis.to_csv(os.path.join(RES, "vegas_vs_model_disagreements.csv"), index=False)
r.to_csv(os.path.join(RES, "vegas_vs_model_all.csv.gz"), index=False, compression="gzip")
print(f"\nsaved {len(dis)} disagreement rows -> results/vegas_vs_model_disagreements.csv")
