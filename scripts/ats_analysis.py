"""Market-efficiency tests: can any model beat the closing spread?"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
pred = pd.read_csv(os.path.join(RES, "predictions.csv.gz"))

rows = []
for lo, hi in [(2006, 2025), (2019, 2025)]:
    d = pred.dropna(subset=["spread_home", "market", "ensemble", "elo", "gbm"]).copy()
    d = d[(d.season >= lo) & (d.season <= hi)]
    d["home_cover"] = np.sign(d["margin"] + d["spread_home"])
    d = d[d["home_cover"] != 0]
    for model in ["ensemble", "elo", "gbm"]:
        pick_home = d[model] > d["market"]
        win = np.where(pick_home, d["home_cover"] > 0, d["home_cover"] < 0)
        rows.append({"era": f"{lo}-{hi}", "model": model, "n": len(d),
                     "ats_win_pct": round(win.mean() * 100, 2),
                     "break_even": 52.38})
ats = pd.DataFrame(rows)
ats.to_csv(os.path.join(RES, "ats_analysis.csv"), index=False)
print(ats.to_string(index=False))

# disagreement analysis, recent
d = pred.dropna(subset=["spread_home", "market", "ensemble"]).copy()
d = d[(d.season >= 2019) & (d.season <= 2025)]
dis = (d["ensemble"] >= 0.5) != (d["market"] >= 0.5)
sub = d[dis]
print(f"\nEnsemble vs market disagree on winner: {len(sub)}/{len(d)} ({len(sub)/len(d)*100:.1f}%)")
print(f"  ensemble correct: {((sub.ensemble>=0.5)==(sub.home_win==1)).mean()*100:.1f}%")
print(f"  market   correct: {((sub.market>=0.5)==(sub.home_win==1)).mean()*100:.1f}%")
