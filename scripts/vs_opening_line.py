"""Does anything we built beat the Vegas OPENING line?

The opening line is the honest benchmark for a bet placed when week W-1 ends: it is
available at that moment and carries no week of sharp money (unlike the close).
Opening spreads exist for 2013-2019 and 2021-2025 (2020 missing). The opening
spread is converted to P(home win) with a walk-forward logistic fit on earlier
seasons only. All comparisons use the SAME games.
"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.cfbrank import metrics

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))


def ll_i(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


ro = pd.read_csv(os.path.join(RES, "results_only_predictions.csv.gz"))
ex = pd.read_csv(os.path.join(RES, "experiments_predictions.csv.gz"))
d = ro.merge(ex[["game_id", "open_spread_home", "open_prob", "gbm_T4_situational", "stack_final"]],
             on="game_id").dropna(subset=["open_prob", "ro_best", "open_spread_home"]).copy()
d["gap"] = lg(d.ro_best) - lg(d.open_prob)      # >0: model likes HOME more than the open does
out = []


def say(s=""):
    print(s); out.append(s)


say(f"opening-line games with our model: {len(d):,}  seasons {sorted(d.season.unique())}")
say("\n== 1. matched-sample log loss vs the OPENING line ==")
rows = []
for lab, mask in [("all", d.season > 0), ("2013-2019", d.season <= 2019), ("2021-2025", d.season >= 2021)]:
    s = d[mask]
    o = ll_i(s.home_win.values, s.open_prob.values).mean()
    for c, n in [("open_prob", "Vegas OPENING"), ("market", "Vegas CLOSING"), ("ro_best", "Results-only model"),
                 ("ensemble", "Previous ensemble"), ("gbm_T4_situational", "GBM T4 (box score+offseason)")]:
        z = s.dropna(subset=[c])
        if len(z) < 0.5 * len(s):
            continue
        m = metrics.evaluate(z.home_win.values, z[c].values)
        oo = ll_i(z.home_win.values, z.open_prob.values).mean()
        rows.append(dict(window=lab, model=n, n=m["n"], acc=m["acc"], logloss=m["logloss"], vs_open=m["logloss"] - oo))
        say(f"  {lab:10s} {n:30s} n={m['n']:5d} acc={m['acc']:.4f} ll={m['logloss']:.4f} vs open {m['logloss']-oo:+.4f}")
pd.DataFrame(rows).to_csv(os.path.join(RES, "vs_opening_line.csv"), index=False)

say("\n== 2. does the model add to the opening line? (walk-forward blend) ==")
d["lo"], d["lm"] = lg(d.open_prob), lg(d.ro_best)
d["blend"] = np.nan
for s in sorted(d.season.unique()):
    tr = d[d.season < s]
    if len(tr) < 800:
        continue
    c = LogisticRegression(C=1e6, max_iter=500).fit(tr[["lo", "lm"]].values, tr.home_win.values)
    m = d.season == s
    d.loc[m, "blend"] = c.predict_proba(d.loc[m, ["lo", "lm"]].values)[:, 1]
    say(f"  {s}: optimal weight on open {c.coef_[0][0]:.2f}, on model {c.coef_[0][1]:+.2f}")
b = d.dropna(subset=["blend"]); y = b.home_win.values
for c, nm in [("ro_best", "model alone vs open"), ("blend", "open+model blend vs open")]:
    diff = ll_i(y, b[c].values) - ll_i(y, b.open_prob.values)
    rng = np.random.default_rng(0)
    bs = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
    say(f"  {nm:26s} {diff.mean():+.4f}  95% CI [{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}]  n={len(diff)}")

say("\n== 3. slices: model - OPEN log loss (negative = model wins) ==")
reg = d[d.season_type == "regular"]
for lab, mask in [("week 1", reg.week == 1), ("weeks 2-4", reg.week.between(2, 4)), ("weeks 5-9", reg.week.between(5, 9)),
                  ("weeks 10-13", reg.week.between(10, 13)), ("weeks 14-15", reg.week >= 14),
                  ("|open|<=3", reg.open_spread_home.abs() <= 3), ("|open| 3-10", reg.open_spread_home.abs().between(3, 10)),
                  ("|open|>10", reg.open_spread_home.abs() > 10)]:
    s = reg[mask]
    a, o = ll_i(s.home_win.values, s.ro_best.values).mean(), ll_i(s.home_win.values, s.open_prob.values).mean()
    say(f"  {lab:14s} n={len(s):5d} model {a:.4f} open {o:.4f} diff {a-o:+.4f}")

say("\n== 4. does the market move toward the model after the open? ==")
d["move"] = d.spread_home - d.open_spread_home      # >0: line moved toward AWAY
for thr in (1.0, 2.0):
    m = d.dropna(subset=["move"]); m = m[m.move.abs() >= thr]
    say(f"  line moved >= {thr}pt: n={len(m)}  toward model's side {(np.sign(m.gap) == -np.sign(m.move)).mean()*100:.1f}%  (SE {np.sqrt(.25/len(m))*100:.1f}, chance 50)")

say("\n== 5. ATS vs the OPENING spread, bet the model's side ==")
d["cover"] = np.sign(d.margin + d.open_spread_home)
x = d[d["cover"] != 0]
roi = lambda w: (w * 100 / 110 - (1 - w)) * 100
for q, lab in [(0.0, "every game"), (0.75, "top quartile of disagreement"), (0.9, "top decile")]:
    z = x[x.gap.abs() >= x.gap.abs().quantile(q)]
    w = np.where(z.gap > 0, z["cover"] > 0, z["cover"] < 0)
    say(f"  {lab:30s} n={len(z):4d} win%={w.mean()*100:5.2f} (SE {np.sqrt(.25/len(z))*100:.1f})  roi@-110={roi(w.mean()):+.1f}%  (break-even 52.38%)")
open(os.path.join(RES, "vs_opening_line.txt"), "w").write("\n".join(out))
