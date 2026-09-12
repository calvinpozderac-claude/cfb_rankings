"""Figure: where our best results-only model beats vs trails the closing line."""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
FIG = os.path.join(RES, "figures")
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})

a = pd.read_csv(os.path.join(RES, "vegas_vs_model_all.csv.gz"))
d = pd.read_csv(os.path.join(RES, "vegas_vs_model_disagreements.csv"))
a["m_ok"] = (a.stack_final >= .5) == (a.home_win == 1)
a["v_ok"] = (a.market >= .5) == (a.home_win == 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

# panel 1: accuracy by week bucket, model vs market (the crossover)
buckets = [("Wk 1", 1, 1), ("Wk 2-4", 2, 4), ("Wk 5-9", 5, 9), ("Wk 10+", 10, 20)]
xs = np.arange(len(buckets))
mv = [a[(a.week >= lo) & (a.week <= hi)].m_ok.mean() * 100 for _, lo, hi in buckets]
vv = [a[(a.week >= lo) & (a.week <= hi)].v_ok.mean() * 100 for _, lo, hi in buckets]
ax1.bar(xs - 0.2, vv, 0.4, label="Vegas closing line", color="#111111")
ax1.bar(xs + 0.2, mv, 0.4, label="Our model (results-only)", color="#009E73")
ax1.set_xticks(xs); ax1.set_xticklabels([b[0] for b in buckets])
ax1.set_ylim(60, 76); ax1.set_ylabel("Straight-up accuracy (%)")
ax1.set_title("Accuracy by point in season\n(model overtakes the line by November)")
for x, m, v in zip(xs, mv, vv):
    ax1.text(x + 0.2, m + 0.2, f"{m:.0f}", ha="center", fontsize=8)
    ax1.text(x - 0.2, v + 0.2, f"{v:.0f}", ha="center", fontsize=8)
ax1.legend(fontsize=8, loc="lower right")

# panel 2: on disagreements, model win-rate by market spread bucket
sb = [("Pick'em\n(0-3)", 0, 3), ("3-7", 3, 7), ("7-14", 7, 14), ("14+", 14, 100)]
xs2 = np.arange(len(sb))
rate, ns = [], []
for _, lo, hi in sb:
    s = d[(d.fav_spread >= lo) & (d.fav_spread < hi)]
    rate.append(s.m_right.mean() * 100 if len(s) else 0); ns.append(len(s))
cols = ["#009E73" if r >= 50 else "#D55E00" for r in rate]
ax2.bar(xs2, rate, color=cols)
ax2.axhline(50, color="k", ls="--", lw=1)
ax2.set_xticks(xs2); ax2.set_xticklabels([b[0] for b in sb])
ax2.set_ylim(0, 60); ax2.set_ylabel("Model's win-rate when it disagrees (%)")
ax2.set_xlabel("Vegas spread on the game (points)")
ax2.set_title("When the model fights the line and wins\n(only near pick'em; never vs a clear favorite)")
for x, r, n in zip(xs2, rate, ns):
    ax2.text(x, r + 1, f"{r:.0f}%\nn={n}", ha="center", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig8_vegas_vs_model.png")); plt.close(fig)
print("wrote fig8_vegas_vs_model.png")
