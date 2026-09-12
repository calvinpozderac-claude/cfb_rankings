"""Generate report figures and final-season ranking tables."""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.cfbrank import elo as elomod, metrics, rankers
from src.cfbrank.data import build_games, is_fbs

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

# clean, colorblind-safe palette
C = {"market": "#111111", "ensemble": "#0072B2", "elo": "#009E73",
     "gbm": "#56B4E9", "cfbd_elo": "#E69F00", "bradley_terry": "#CC79A7",
     "pr_points": "#D55E00", "blade_chest": "#999999"}
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})

pred = pd.read_csv(os.path.join(RES, "predictions.csv.gz"))
byS = pd.read_csv(os.path.join(RES, "metrics_by_season.csv"))


# --- Fig 1: model comparison, recent era (log loss + accuracy) ---
rec = pred[(pred.season >= 2019) & (pred.season <= 2025)]
models = ["market", "ensemble", "gbm", "mlp", "cfbd_elo", "elo",
          "bradley_terry", "pr_points", "pr_margin", "pr_win", "blade_chest"]
labels = {"market": "Vegas line", "ensemble": "Ensemble", "gbm": "GBM",
          "mlp": "Neural net", "cfbd_elo": "CFBD Elo", "elo": "Elo (ours)",
          "bradley_terry": "Bradley-Terry", "pr_points": "PageRank-points",
          "pr_margin": "PageRank-margin", "pr_win": "PageRank-win",
          "blade_chest": "Blade-Chest"}
ll = [metrics.evaluate(rec.home_win.values, rec[m].values)["logloss"] for m in models]
ac = [metrics.evaluate(rec.home_win.values, rec[m].values)["acc"] for m in models]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
colors = ["#111111" if m == "market" else ("#0072B2" if m == "ensemble" else "#7aa6c2")
          for m in models]
ax1.barh([labels[m] for m in models], ll, color=colors)
ax1.invert_yaxis(); ax1.set_xlabel("Log loss (lower = better)")
ax1.set_xlim(0.48, 0.70); ax1.set_title("Probability quality, 2019-2025")
ax1.axvline(ll[0], color="#111111", ls="--", lw=1, alpha=0.6)
for i, v in enumerate(ll):
    ax1.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8)
ax2.barh([labels[m] for m in models], [a * 100 for a in ac], color=colors)
ax2.invert_yaxis(); ax2.set_xlabel("Straight-up accuracy (%)")
ax2.set_xlim(64, 75); ax2.set_title("Pick accuracy, 2019-2025")
ax2.axvline(ac[0] * 100, color="#111111", ls="--", lw=1, alpha=0.6)
for i, v in enumerate(ac):
    ax2.text(v * 100 + 0.05, i, f"{v*100:.1f}", va="center", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_model_comparison.png")); plt.close(fig)


# --- Fig 2: reliability curves ---
fig, ax = plt.subplots(figsize=(5.6, 5.4))
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="perfect")
for m in ["market", "ensemble", "elo"]:
    ct = metrics.calibration_table(rec.home_win.values, rec[m].values, bins=10)
    ax.plot(ct.pred_mean, ct.obs_rate, "o-", color=C[m], label=labels[m], ms=5)
ax.set_xlabel("Predicted P(home win)"); ax.set_ylabel("Observed home-win rate")
ax.set_title("Calibration, 2019-2025"); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_calibration.png")); plt.close(fig)


# --- Fig 3: accuracy & log loss by season ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
for m in ["market", "ensemble", "elo", "cfbd_elo"]:
    d = byS[byS.model == m].sort_values("season")
    ax1.plot(d.season, d.acc * 100, "o-", color=C[m], label=labels[m], ms=4)
    ax2.plot(d.season, d.logloss, "o-", color=C[m], label=labels[m], ms=4)
ax1.set_ylabel("Accuracy (%)"); ax1.set_title("Straight-up accuracy by season")
ax2.set_ylabel("Log loss"); ax2.set_title("Log loss by season")
for ax in (ax1, ax2):
    ax.set_xlabel("Season"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_by_season.png")); plt.close(fig)


# --- Fig 4: skill by week of (regular) season ---
reg = pred[(pred.season >= 2014) & (pred.season <= 2025) & (pred.season_type == "regular")]
wk = sorted(w for w in reg.week.unique() if 1 <= w <= 14)
fig, ax = plt.subplots(figsize=(7.2, 4.4))
for m in ["market", "ensemble", "elo"]:
    ys = []
    for w in wk:
        s = reg[reg.week == w]
        ys.append(metrics.evaluate(s.home_win.values, s[m].values)["acc"] * 100)
    ax.plot(wk, ys, "o-", color=C[m], label=labels[m], ms=4)
ax.set_xlabel("Week of regular season"); ax.set_ylabel("Accuracy (%)")
ax.set_title("Accuracy by week, 2014-2025 (early season is hardest)")
ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_by_week.png")); plt.close(fig)


# --- Final 2025 rankings (Elo + Bradley-Terry), FBS only ---
g = build_games(); g = g[g.season <= 2025]
g25 = g[g.season == 2025]
fbs_teams = set(g25[g25.home_division == "fbs"].home_team) | set(g25[g25.away_division == "fbs"].away_team)

elo_final = elomod.final_ratings(g, k=40, home_field=50, mov=True, preseason_regress=0.15)
elo_fbs = elo_final[elo_final.index.isin(fbs_teams)].head(25)

# Bradley-Terry on 2025 season (full-season strengths)
bt_fit = rankers.make_bradley_terry(reg=3.0, carry=0.35)
# extract ratings by probing: refit and read via closure is awkward; recompute directly
from scipy.optimize import minimize
h = g[(g.season >= 2024)].copy(); h["w"] = np.where(h.season == 2025, 1.0, 0.35)
teams = sorted(set(h.home_team) | set(h.away_team)); tidx = {t: i for i, t in enumerate(teams)}
n = len(teams)
hi = np.array([tidx[t] for t in h.home_team]); ai = np.array([tidx[t] for t in h.away_team])
homeind = np.where(h.neutral_site.values, 0.0, 1.0); y = h.home_win.values.astype(float); w = h.w.values


def nll(th):
    s = th[:n]; ha = th[n]
    z = s[hi] - s[ai] + ha * homeind
    p = np.clip(1 / (1 + np.exp(-z)), 1e-12, 1 - 1e-12)
    loss = -(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum() + 3.0 * 0.5 * np.sum(s ** 2)
    g_ = np.zeros(n + 1); resid = w * (p - y)
    np.add.at(g_, hi, resid); np.add.at(g_, ai, -resid); g_[:n] += 3.0 * s
    g_[n] = np.sum(resid * homeind)
    return loss, g_


res = minimize(nll, np.zeros(n + 1), jac=True, method="L-BFGS-B", options={"maxiter": 400})
bt = pd.Series(res.x[:n], index=teams)
bt_fbs = bt[bt.index.isin(fbs_teams)].sort_values(ascending=False).head(25)

rk = pd.DataFrame({
    "rank": range(1, 26),
    "Elo_team": elo_fbs.index, "Elo_rating": elo_fbs.values.round(0).astype(int),
    "BT_team": bt_fbs.index, "BT_strength": bt_fbs.values.round(2),
})
rk.to_csv(os.path.join(RES, "rankings_2025.csv"), index=False)
print("Saved figures and rankings_2025.csv")
print(rk.to_string(index=False))
