"""Figure: the results-only predictor vs benchmarks, overall and by week."""
from __future__ import annotations
import os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.cfbrank import metrics

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
FIG = os.path.join(RES, "figures")
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False})
b = pd.read_csv(os.path.join(RES, "results_only_predictions.csv.gz"))
t = b[(b.season >= 2018) & (b.season <= 2025)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
models = [("market", "Vegas closing line", "#111111"), ("ro_best", "Results-only model", "#009E73"),
          ("ensemble", "Previous ensemble", "#999999"), ("cfbd_elo", "CFBD Elo", "#E69F00"),
          ("elo_prob", "Elo alone", "#7aa6c2")]
ll = [metrics.evaluate(t.home_win.values, t[m].values)["logloss"] for m, _, _ in models]
ax1.barh([l for _, l, _ in models], ll, color=[c for _, _, c in models])
ax1.invert_yaxis(); ax1.axvline(ll[0], color="k", ls="--", lw=1)
ax1.set_xlim(0.50, 0.57); ax1.set_xlabel("Log loss, held-out 2018-2025 (lower = better)")
ax1.set_title("Using only past results, we close ~1/3 of the gap to Vegas")
for i, v in enumerate(ll):
    ax1.text(v + .0012, i, f"{v:.4f}", va="center", fontsize=8)

reg = t[t.season_type == "regular"]
wk = [w for w in range(1, 16) if (reg.week == w).sum() >= 40]
for m, lab, c in [("ro_best", "Results-only model", "#009E73"), ("ensemble", "Previous ensemble", "#999999")]:
    gaps = []
    for w in wk:
        s = reg[reg.week == w]
        gaps.append(metrics.evaluate(s.home_win.values, s[m].values)["logloss"]
                    - metrics.evaluate(s.home_win.values, s.market.values)["logloss"])
    ax2.plot(wk, gaps, "o-", color=c, ms=4, label=lab)
ax2.axhline(0, color="k", lw=1)
ax2.set_xlabel("Week of regular season"); ax2.set_ylabel("Log-loss gap to Vegas")
ax2.set_title("Where the gain came from (2018-2025)"); ax2.legend(fontsize=8)
ax2.set_xticks(range(1, 16, 2))
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig11_results_only.png")); plt.close(fig)

rows = []
for w in wk:
    s = reg[reg.week == w]
    rows.append({"week": w, **{m: metrics.evaluate(s.home_win.values, s[m].values)["logloss"]
                               for m in ["market", "ro_best", "ensemble"]}})
d = pd.DataFrame(rows)
d["gap_ro"] = d.ro_best - d.market; d["gap_prev"] = d.ensemble - d.market
d.round(4).to_csv(os.path.join(RES, "results_only_by_week.csv"), index=False)
print(d[["week", "market", "ro_best", "ensemble", "gap_ro", "gap_prev"]].round(3).to_string(index=False))
print("\nwrote fig11_results_only.png")
