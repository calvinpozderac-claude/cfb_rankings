"""Experiment suite: how close can we get to the Vegas closing line?

Tiers (each adds information on top of the last; all walk-forward):
  T0  baseline           previous ensemble / GBM / Elo (from results/predictions.csv.gz)
  T1  + ridge ratings    least-squares point-margin ratings (Massey-style)
  T2  + efficiency       ridge ratings on yards/play, success, explosive, turnover, havoc
                         margins built from play-level box scores (2014-2025)
  T3  + offseason        returning production & roster continuity (from rosters +
                         player production), and an Elo whose preseason regression is
                         driven by returning production
  T4  + situational      travel distance, time-zone shift, elevation
  T5  + opening line     market-informed: the *opening* consensus spread as a feature
                         (tests whether our model + the open beats the close)

For every tier we fit a per-season-refit GBM on the feature set and a per-season
logistic stacker over the base models; both are evaluated on 2019-2025 FBS games
against the closing line (log loss / accuracy / Brier, market+model incremental
value, ATS, disagreement win-rate) and by week.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.cfbrank import elo as elomod, features as feat, metrics, rankers
from src.cfbrank.boxstats import build_boxstats, game_margins
from src.cfbrank.data import build_games, is_fbs
from src.cfbrank.extras import returning_production, travel_features
from src.cfbrank.margin import make_ridge_margin
from src.cfbrank.mlmodels import ML_FEATURES

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
FEAT_START = 2015          # first season with all new features
EVAL_LO, EVAL_HI = 2019, 2025
ELO = dict(k=40, home_field=50, mov=True, preseason_regress=0.15)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def ev(df, col, lo=EVAL_LO, hi=EVAL_HI):
    s = df[(df.season >= lo) & (df.season <= hi)]
    return metrics.evaluate(s.home_win.values, s[col].values)


def batch_collect(games, fit_fn, eval_ids, want_margin=True):
    """Weekly refit; returns dict game_id -> (prob, predicted target)."""
    out = {}
    by_p = {p: d for p, d in games.groupby("period")}
    for p in sorted(by_p):
        cur = by_p[p]; cur = cur[cur.game_id.isin(eval_ids)]
        if len(cur) == 0:
            continue
        hist = games[games.period < p]
        if len(hist) < 50:
            continue
        pred = fit_fn(hist)
        for r in cur.itertuples(index=False):
            pr = pred(r.home_team, r.away_team, bool(r.neutral_site))
            m = pred.predict_margin(r.home_team, r.away_team, bool(r.neutral_site)) if want_margin else np.nan
            out[r.game_id] = (pr, m)
    return out


def season_gbm(df, feats, name, start=FEAT_START, min_train=1500):
    preds = {}
    for s in sorted(df.season.unique()):
        tr = df[(df.season >= start) & (df.season < s)]
        te = df[df.season == s]
        if len(tr) < min_train or len(te) == 0:
            continue
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.04, max_iter=350,
                                           l2_regularization=1.0, min_samples_leaf=50,
                                           random_state=0)
        m.fit(tr[feats].values, tr.home_win.values)
        for gid, p in zip(te.game_id.values, m.predict_proba(te[feats].values)[:, 1]):
            preds[gid] = float(p)
    return df.game_id.map(preds)


def season_stack(df, cols, start=FEAT_START, min_train=1500, C=1.0, by_week_bucket=False):
    preds = {}
    d = df.dropna(subset=cols)
    for s in sorted(d.season.unique()):
        tr = d[(d.season >= start) & (d.season < s)]
        te = d[d.season == s]
        if len(tr) < min_train or len(te) == 0:
            continue
        buckets = [(0, 99)] if not by_week_bucket else [(0, 3), (4, 8), (9, 99)]
        for lo, hi in buckets:
            trb = tr[(tr.week >= lo) & (tr.week <= hi)]
            teb = te[(te.week >= lo) & (te.week <= hi)]
            if len(trb) < 300 or len(teb) == 0:
                trb = tr
            clf = LogisticRegression(C=C, max_iter=1000)
            clf.fit(logit(trb[cols].values), trb.home_win.values)
            for gid, p in zip(teb.game_id.values, clf.predict_proba(logit(teb[cols].values))[:, 1]):
                preds[gid] = float(p)
    return df.game_id.map(preds)


def spread_prob(df, col, start=FEAT_START):
    """Walk-forward logistic calibration of a spread column to P(home win)."""
    preds = {}
    d = df.dropna(subset=[col])
    for s in sorted(d.season.unique()):
        tr = d[(d.season >= 2012) & (d.season < s)]; te = d[d.season == s]
        if len(tr) < 500 or len(te) == 0:
            continue
        clf = LogisticRegression(C=1e6, max_iter=1000).fit(tr[[col]].values, tr.home_win.values)
        for gid, p in zip(te.game_id.values, clf.predict_proba(te[[col]].values)[:, 1]):
            preds[gid] = float(p)
    return df.game_id.map(preds)


def market_tests(df, col, lo=EVAL_LO, hi=EVAL_HI):
    """Incremental value over the market, ATS vs close, disagreement win rate."""
    d = df[(df.season >= lo) & (df.season <= hi)].dropna(subset=[col, "market", "spread_home"])
    # market + model stack (per-season walk-forward)
    inc = {}
    for s in sorted(d.season.unique()):
        tr = df[(df.season < s) & (df.season >= FEAT_START)].dropna(subset=[col, "market"])
        te = d[d.season == s]
        if len(tr) < 800:
            continue
        clf = LogisticRegression(C=1.0, max_iter=1000).fit(logit(tr[["market", col]].values), tr.home_win.values)
        for gid, p in zip(te.game_id.values, clf.predict_proba(logit(te[["market", col]].values))[:, 1]):
            inc[gid] = p
    d = d.assign(mm=d.game_id.map(inc)).dropna(subset=["mm"])
    ll_mkt = metrics.log_loss(d.home_win.values, d.market.values)
    ll_mm = metrics.log_loss(d.home_win.values, d.mm.values)
    cover = np.sign(d.margin + d.spread_home); dd = d[cover != 0]; cover = cover[cover != 0]
    pick_home = dd[col] > dd.market
    ats = float(np.where(pick_home, cover > 0, cover < 0).mean())
    dis = (dd[col] >= 0.5) != (dd.market >= 0.5)
    sub = dd[dis]
    model_right = float(((sub[col] >= 0.5) == (sub.home_win == 1)).mean()) if len(sub) else np.nan
    return {"ll_market": ll_mkt, "ll_market_plus_model": ll_mm, "market_gain": ll_mkt - ll_mm,
            "ats_pct": ats * 100, "disagree_pct": float(dis.mean() * 100),
            "model_right_when_disagree": model_right * 100}


def main():
    t0 = time.time()
    g = build_games(); g = g[g.season <= 2025].reset_index(drop=True)
    fbs_ids = set(g[is_fbs(g)].game_id)
    box, prod = build_boxstats()
    gm = game_margins(box, g[is_fbs(g) & (g.season >= 2014)])
    g = g.merge(gm, left_on="game_id", right_index=True, how="left")
    prev = pd.read_csv(os.path.join(RES, "predictions.csv.gz"))
    base = g[g.game_id.isin(fbs_ids)][["game_id", "season", "week", "season_type", "home_team",
                                       "away_team", "home_win", "margin", "spread_home",
                                       "open_spread_home", "neutral_site"]].copy()
    base = base.merge(prev[["game_id", "elo", "pr_points_keep", "bradley_terry", "blade_chest",
                            "gbm", "mlp", "ensemble", "market", "cfbd_elo"]], on="game_id", how="left")
    print(f"base rows {len(base):,}")

    # ---------------- T1/T2: ridge ratings on points and efficiency margins ----------
    var_ref = {}
    ref = g[(g.season >= 2014) & (g.season <= 2016)]
    for t in ["margin", "ypp_margin", "success_margin", "explosive_margin", "turnover_margin", "havoc_margin"]:
        var_ref[t] = float(np.nanvar(np.clip(ref[t], -35, 35) if t == "margin" else ref[t]))
    ridge_specs = {"margin": dict(lam=4.0, cap=35.0)}
    for t in ["ypp_margin", "success_margin", "explosive_margin", "turnover_margin", "havoc_margin"]:
        ridge_specs[t] = dict(lam=4.0 * var_ref[t] / var_ref["margin"], cap=None)
    for t, sp in ridge_specs.items():
        tt = time.time()
        fit = make_ridge_margin(lam=sp["lam"], decay=0.35, cap=sp["cap"], target=t)
        out = batch_collect(g[g.season >= 2008], fit, set(base[base.season >= 2012].game_id))
        base[f"ridge_{t}_prob"] = base.game_id.map({k: v[0] for k, v in out.items()})
        base[f"ridge_{t}_pred"] = base.game_id.map({k: v[1] for k, v in out.items()})
        print(f"  ridge on {t:17s} done [{time.time()-tt:.0f}s]  ll19-25={ev(base, f'ridge_{t}_prob')['logloss']:.4f}")

    # ---------------- T3: offseason information ------------------------------------
    rp = returning_production(prod)
    rp["ret_off"] = rp.ret_off.clip(0, 1); rp["ret_def"] = rp.ret_def.clip(0, 1)
    rp["ret_avg"] = (rp.ret_off + rp.ret_def) / 2
    rp_key = rp.set_index(["season", "team"])
    for side in ["home", "away"]:
        for c in ["ret_off", "ret_def", "ret_avg", "roster_continuity"]:
            base[f"{side}_{c}"] = [rp_key[c].get((s, t), np.nan) for s, t in zip(base.season, base[f"{side}_team"])]
    base["ret_off_diff"] = base.home_ret_off - base.away_ret_off
    base["ret_def_diff"] = base.home_ret_def - base.away_ret_def
    base["continuity_diff"] = base.home_roster_continuity - base.away_roster_continuity

    # Elo whose preseason regression depends on returning production
    def elo_rp(slope):
        tr = {}
        for r in rp.itertuples(index=False):
            reg = ELO["preseason_regress"] + slope * (0.5 - r.ret_avg)
            tr[(int(r.season), r.team)] = float(np.clip(reg, 0.02, 0.7))
        return elomod.run_elo(g, team_regress=tr, **ELO)
    best = None
    for slope in [0.0, 0.3, 0.6, 0.9, 1.2]:
        e = elo_rp(slope)
        tmp = base.assign(p=base.game_id.map(e["elo_prob"]))
        ll = ev(tmp, "p", 2015, 2018)["logloss"]
        print(f"  elo_rp slope={slope}: ll(2015-18)={ll:.4f}")
        if best is None or ll < best[1]:
            best = (slope, ll)
    e = elo_rp(best[0])
    base["elo_rp"] = base.game_id.map(e["elo_prob"])
    base["elo_rp_diff"] = base.game_id.map(e["elo_home_pre"] - e["elo_away_pre"])
    print(f"  elo_rp best slope {best[0]} | ll19-25={ev(base, 'elo_rp')['logloss']:.4f} vs elo {ev(base, 'elo')['logloss']:.4f}")

    # ---------------- T4: situational -------------------------------------------------
    tv = travel_features(base)
    base = base.merge(tv, left_on="game_id", right_index=True, how="left")

    # ---------------- feature table ---------------------------------------------------
    F = feat.build_features(g)
    e0 = elomod.run_elo(g, **ELO)
    F["elo_prob"] = e0["elo_prob"]; F["elo_diff"] = e0["elo_home_pre"] - e0["elo_away_pre"]
    base = base.merge(F, left_on="game_id", right_index=True, how="left")
    base["pr_keep_logit"] = logit(base.pr_points_keep); base["bt_logit"] = logit(base.bradley_terry)
    base["open_prob"] = spread_prob(base, "open_spread_home")

    T1 = ML_FEATURES + ["week", "ridge_margin_pred", "pr_keep_logit", "bt_logit"]
    T2 = T1 + [f"ridge_{t}_pred" for t in ["ypp_margin", "success_margin", "explosive_margin", "turnover_margin", "havoc_margin"]]
    T3 = T2 + ["ret_off_diff", "ret_def_diff", "continuity_diff", "elo_rp_diff", "home_ret_avg", "away_ret_avg"]
    T4 = T3 + ["away_travel_km", "tz_shift", "elev_diff"]
    T5 = T4 + ["open_spread_home"]
    tiers = {"gbm_T1_ridge": T1, "gbm_T2_efficiency": T2, "gbm_T3_offseason": T3,
             "gbm_T4_situational": T4, "gbm_T5_openline": T5}
    for name, cols in tiers.items():
        tt = time.time(); base[name] = season_gbm(base, cols, name)
        print(f"  {name:20s} ll19-25={ev(base, name)['logloss']:.4f} [{time.time()-tt:.0f}s]")

    # ---------------- stackers -------------------------------------------------------
    base["stack_T1"] = season_stack(base, ["elo", "ridge_margin_prob", "pr_points_keep", "bradley_terry", "gbm_T1_ridge"])
    base["stack_T2"] = season_stack(base, ["elo", "ridge_margin_prob", "pr_points_keep", "bradley_terry", "gbm_T2_efficiency"])
    base["stack_T4"] = season_stack(base, ["elo_rp", "ridge_margin_prob", "pr_points_keep", "bradley_terry", "gbm_T2_efficiency", "gbm_T4_situational"])
    base["stack_T4_weekly"] = season_stack(base, ["elo_rp", "ridge_margin_prob", "pr_points_keep", "bradley_terry", "gbm_T2_efficiency", "gbm_T4_situational"], by_week_bucket=True)
    base["stack_T5"] = season_stack(base, ["elo_rp", "ridge_margin_prob", "gbm_T4_situational", "gbm_T5_openline", "open_prob"])

    base.to_csv(os.path.join(RES, "experiments_predictions.csv.gz"), index=False, compression="gzip")

    # ---------------- ladder ------------------------------------------------------------
    ladder_cols = [
        ("market", "Vegas closing line", "benchmark"), ("open_prob", "Vegas opening line", "benchmark"),
        ("cfbd_elo", "CFBD Elo", "benchmark"),
        ("elo", "Elo (T0)", "T0"), ("gbm", "GBM (T0)", "T0"), ("ensemble", "Ensemble (T0)", "T0"),
        ("ridge_margin_prob", "Ridge margin rating", "T1"), ("gbm_T1_ridge", "GBM T1", "T1"), ("stack_T1", "Stack T1", "T1"),
        ("ridge_ypp_margin_prob", "Ridge yards/play rating", "T2"), ("ridge_success_margin_prob", "Ridge success-rate rating", "T2"),
        ("ridge_turnover_margin_prob", "Ridge turnover rating", "T2"),
        ("gbm_T2_efficiency", "GBM T2 (+efficiency)", "T2"), ("stack_T2", "Stack T2", "T2"),
        ("elo_rp", "Elo + returning production", "T3"), ("gbm_T3_offseason", "GBM T3 (+offseason)", "T3"),
        ("gbm_T4_situational", "GBM T4 (+situational)", "T4"), ("stack_T4", "Stack T4", "T4"),
        ("stack_T4_weekly", "Stack T4, week-bucketed", "T4"),
        ("gbm_T5_openline", "GBM T5 (+opening line)", "T5"), ("stack_T5", "Stack T5 (+opening line)", "T5"),
    ]
    rows = []
    for col, label, tier in ladder_cols:
        m = ev(base, col)
        r = {"tier": tier, "model": label, "col": col, **m}
        if col not in ("market",):
            r.update(market_tests(base, col))
        rows.append(r)
    ladder = pd.DataFrame(rows)
    ladder.to_csv(os.path.join(RES, "experiments_ladder.csv"), index=False)

    # by-week for a few
    bw = []
    rw = base[(base.season >= EVAL_LO) & (base.season <= EVAL_HI) & (base.season_type == "regular")]
    for wk in range(1, 16):
        s = rw[rw.week == wk]
        if len(s) < 30:
            continue
        for col in ["market", "ensemble", "stack_T2", "stack_T4", "stack_T4_weekly", "stack_T5", "elo", "elo_rp"]:
            bw.append({"week": wk, "model": col, **metrics.evaluate(s.home_win.values, s[col].values)})
    pd.DataFrame(bw).to_csv(os.path.join(RES, "experiments_by_week.csv"), index=False)

    print(f"\nDONE in {time.time()-t0:.0f}s\n")
    pd.set_option("display.width", 200)
    print(ladder[["tier", "model", "n", "acc", "logloss", "brier", "market_gain", "ats_pct", "model_right_when_disagree"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
