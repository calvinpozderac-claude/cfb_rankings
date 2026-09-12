"""Bankroll curves: what a $100-flat bettor would have made (lost) 2013-2025."""
from __future__ import annotations
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.betting_sim import moneyline_odds, simulate, MODEL, RES, STAKE

b = pd.read_csv(os.path.join(RES, "experiments_predictions.csv.gz"))
b = b.dropna(subset=[MODEL, "open_prob", "open_spread_home"]).merge(moneyline_odds(), on="game_id", how="left")


def curve(w, s, e, m):
    d = simulate(b, w, s, e, m).sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    d["cum"] = d.profit.cumsum()
    return d


plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(9, 5))
specs = [
    ("Bet model's pick on every game (ML)", 1, 99, -9, "ML", "#999999"),
    ("Bet all model-vs-line disagreements (ML)", 1, 99, 0.0001, "ML", "#D55E00"),
    ("Best in-sample rule: wk>=8, |open|<=3, edge>=.08 (ML)", 8, 3, 0.08, "ML", "#0072B2"),
    ("Same rule, against the spread (-110)", 8, 3, 0.08, "ATS", "#CC79A7"),
]
for lbl, w, s, e, m, c in specs:
    d = curve(w, s, e, m)
    if lbl.startswith("Bet all model-vs"):
        d = d[(d[MODEL] >= .5) != (d.open_prob >= .5)].copy(); d["cum"] = d.profit.cumsum()
    ax.plot(np.arange(len(d)), d.cum, color=c, lw=1.6,
            label=f"{lbl}  (n={len(d)}, {d.profit.sum()/(len(d)*STAKE)*100:+.1f}% ROI)")
    # mark where the out-of-sample test period begins (2021) for the tuned rule
    if lbl.startswith("Best in-sample"):
        k = (d.season <= 2019).sum()
        ax.axvline(k, color=c, ls=":", lw=1)
        ax.text(k, ax.get_ylim()[1], " tune | test", color=c, fontsize=8, va="top")
ax.axhline(0, color="k", lw=1)
ax.set_xlabel("Bet number (chronological, 2013-2025)")
ax.set_ylabel("Cumulative profit, $100 flat per bet")
ax.set_title("No threshold rule beats the vig: every strategy trends down")
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(RES, "figures", "fig9_betting_bankroll.png")); plt.close(fig)
print("wrote fig9_betting_bankroll.png")
