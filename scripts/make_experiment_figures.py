"""Figures for the experiment suite: the tier ladder and the by-week gap to Vegas."""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
FIG = os.path.join(RES, "figures")
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})

lad = pd.read_csv(os.path.join(RES, "experiments_ladder.csv"))
show = ["market", "open_prob", "ensemble", "elo", "ridge_margin_prob", "stack_T1", "gbm_T2_efficiency",
        "stack_T2", "elo_rp", "gbm_T4_situational", "stack_T4", "stack_T4_weekly", "gbm_T4_margin", "stack_final",
        "gbm_T5_openline", "stack_T5"]
d = lad.set_index("col").loc[show].reset_index()
tier_col = {"benchmark": "#111111", "T0": "#999999", "T1": "#7aa6c2", "T2": "#0072B2",
            "T3": "#009E73", "T4": "#009E73", "T5": "#D55E00"}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.6))
cols = [tier_col[t] for t in d.tier]
mk = float(d[d.col == "market"].logloss.iloc[0]); mka = float(d[d.col == "market"].acc.iloc[0]) * 100
ax1.barh(d.model, d.logloss, color=cols); ax1.invert_yaxis(); ax1.axvline(mk, color="k", ls="--", lw=1)
ax1.set_xlim(0.50, 0.58); ax1.set_xlabel("Log loss, 2019-2025 (lower = better)"); ax1.set_title("Experiment ladder: probability quality")
for i, v in enumerate(d.logloss):
    ax1.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=8)
ax2.barh(d.model, d.acc * 100, color=cols); ax2.invert_yaxis(); ax2.axvline(mka, color="k", ls="--", lw=1)
ax2.set_xlim(69, 74.5); ax2.set_xlabel("Accuracy (%), 2019-2025"); ax2.set_title("Experiment ladder: pick accuracy")
for i, v in enumerate(d.acc * 100):
    ax2.text(v + 0.05, i, f"{v:.1f}", va="center", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig6_experiment_ladder.png")); plt.close(fig)

bw = pd.read_csv(os.path.join(RES, "experiments_by_week.csv"))
lab = {"ensemble": "Ensemble (T0)", "stack_T2": "Stack T2 (+efficiency)", "stack_final": "Stack FINAL (results-only)",
       "stack_T5": "Stack T5 (+opening line)"}
colr = {"ensemble": "#999999", "stack_T2": "#0072B2", "stack_final": "#009E73", "stack_T5": "#D55E00"}
mkw = bw[bw.model == "market"].set_index("week").logloss
fig, ax = plt.subplots(figsize=(8, 4.4))
for m in ["ensemble", "stack_T2", "stack_final", "stack_T5"]:
    s = bw[bw.model == m].set_index("week")
    ax.plot(s.index, s.logloss - mkw.loc[s.index], "o-", ms=4, color=colr[m], label=lab[m])
ax.axhline(0, color="k", lw=1)
ax.set_xlabel("Week of regular season"); ax.set_ylabel("Log loss gap to Vegas (model - market)")
ax.set_title("How far behind the closing line, by week (2019-2025)"); ax.legend(fontsize=8)
ax.set_xticks(range(1, 16, 2))
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig7_gap_by_week.png")); plt.close(fig)
print("figures written")
