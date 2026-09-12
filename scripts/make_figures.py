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
     "pr_points": "#D55E00", "pr_points_keep": "#D55E00", "blade_chest": "#999999"}
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})

pred = pd.read_csv(os.path.join(RES, "predictions.csv.gz"))
byS = pd.read_csv(os.path.join(RES, "metrics_by_season.csv"))


# --- Fig 1: model comparison, recent era (log loss + accuracy) ---
rec = pred[(pred.season >= 2019) & (pred.season <= 2025)]
models = ["market", "ensemble", "gbm", "mlp", "cfbd_elo", "elo", "pr_points_keep",
          "bradley_terry", "pr_points", "pr_points_against", "pr_margin", "pr_win",
          "blade_chest"]
labels = {"market": "Vegas line", "ensemble": "Ensemble", "gbm": "GBM",
          "mlp": "Neural net", "cfbd_elo": "CFBD Elo", "elo": "Elo (ours)",
          "bradley_terry": "Bradley-Terry", "pr_points": "PageRank-points",
          "pr_points_against": "PageRank-points-against", "pr_points_keep": "PageRank-points+keep",
          "pr_margin": "PageRank-margin", "pr_win": "PageRank-win",
          "blade_chest": "Blade-Chest"}
ll = [metrics.evaluate(rec.home_win.values, rec[m].values)["logloss"] for m in models]
ac = [metrics.evaluate(rec.home_win.values, rec[m].values)["acc"] for m in models]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
def _bar_color(m):
    if m == "market":
        return "#111111"
    if m == "ensemble":
        return "#0072B2"
    if m == "pr_points_keep":
        return "#009E73"   # highlight the self-retention variant
    return "#7aa6c2"
colors = [_bar_color(m) for m in models]
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


# --- Fig 4: accuracy AND log loss by week of season (timing effects) ---
byW = pd.read_csv(os.path.join(RES, "metrics_by_week.csv"))
wk_models = ["market", "ensemble", "elo", "pr_points_keep", "bradley_terry"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
for m in wk_models:
    d = byW[byW.model == m].sort_values("week")
    ax1.plot(d.week, d.acc * 100, "o-", color=C[m], label=labels[m], ms=4)
    ax2.plot(d.week, d.logloss, "o-", color=C[m], label=labels[m], ms=4)
ax1.set_ylabel("Accuracy (%)"); ax1.set_title("Accuracy by week, 2014-2025")
ax2.set_ylabel("Log loss (lower = better)"); ax2.set_title("Log loss by week, 2014-2025")
for ax in (ax1, ax2):
    ax.set_xlabel("Week of regular season")
    ax.set_xticks(range(1, 16, 2))
ax1.legend(fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_by_week.png")); plt.close(fig)


# --- Fig 5: PageRank family focus (the point-flow variants) ---
pr_order = ["pr_win", "pr_margin", "pr_points", "pr_points_against", "pr_points_keep"]
pr_lab = ["wins", "margin", "points", "points-against", "points+keep (retain)"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.9))
acc = [metrics.evaluate(rec.home_win.values, rec[m].values)["acc"] * 100 for m in pr_order]
ll = [metrics.evaluate(rec.home_win.values, rec[m].values)["logloss"] for m in pr_order]
elo_acc = metrics.evaluate(rec.home_win.values, rec["elo"].values)["acc"] * 100
elo_ll = metrics.evaluate(rec.home_win.values, rec["elo"].values)["logloss"]
bar_c = ["#7aa6c2"] * 4 + ["#009E73"]
ax1.bar(pr_lab, acc, color=bar_c)
ax1.axhline(elo_acc, color="#111", ls="--", lw=1)
ax1.text(0.02, elo_acc + 0.15, "Elo", transform=ax1.get_yaxis_transform(), fontsize=8)
ax1.set_ylim(65, 72); ax1.set_ylabel("Accuracy (%)")
ax1.set_title("PageRank edge weighting: accuracy (2019-2025)")
ax1.tick_params(axis="x", rotation=25)
ax2.bar(pr_lab, ll, color=bar_c)
ax2.axhline(elo_ll, color="#111", ls="--", lw=1)
ax2.text(0.02, elo_ll + 0.004, "Elo", transform=ax2.get_yaxis_transform(), fontsize=8)
ax2.set_ylim(0.53, 0.70); ax2.set_ylabel("Log loss")
ax2.set_title("PageRank edge weighting: log loss (2019-2025)")
ax2.tick_params(axis="x", rotation=25)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_pagerank_family.png")); plt.close(fig)


# --- Final 2025 rankings (Elo + Bradley-Terry), FBS only ---
g = build_games(); g = g[g.season <= 2025]
g25 = g[g.season == 2025]
fbs_teams = set(g25[g25.home_division == "fbs"].home_team) | set(g25[g25.away_division == "fbs"].away_team)

elo_final = elomod.final_ratings(g, k=40, home_field=50, mov=True, preseason_regress=0.15)
elo_fbs = elo_final[elo_final.index.isin(fbs_teams)].head(25)

# Bradley-Terry end-of-2025 strengths, with the tuned cross-season recency decay
from scipy.optimize import minimize
BT_DECAY = 0.65
h = g[g.season >= 2019].copy()
h["w"] = np.power(BT_DECAY, (2025 - h["season"]).astype(float))
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
