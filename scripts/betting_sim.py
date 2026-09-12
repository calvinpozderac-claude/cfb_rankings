"""Could a simple threshold rule have made money?

Protocol (no leakage): we imagine placing bets on week W's games as soon as
week W-1 finishes. At that moment we know:
  * our walk-forward model's win probability (the ensemble -- trained only on
    prior seasons, features only pre-game),
  * the sportsbooks' OPENING spread for week W (available before kickoff),
  * the week number.
The bet decision uses ONLY those. Payouts use the consensus *closing* moneyline
(opening moneyline is only 7% populated); closing odds are sharper than the
opening odds we'd actually get, so if anything this understates dog payouts.

A "simple threshold" rule bets the model's pick when
    week >= W_min  AND  |opening spread| <= S_max  AND  edge >= E_min
where edge = model P(pick) - opening-line-implied P(pick).

To avoid curve-fitting the threshold to the same games, the rule is chosen on
TRAIN 2013-2019 and reported out-of-sample on TEST 2021-2025 (2020 has no
opening lines), plus full-window context and naive baselines. Flat $100 stake;
-110 assumed for the ATS variant.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.cfbrank.data import _abbr_to_team, _load_schedules, LINES_GZ

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
STAKE = 100.0
# Use the ensemble: our best results-only model that has a long backtest (2006-2025,
# within 0.005 log loss of stack_final, which only exists 2023-2025). Opening lines
# exist for 2013-2019 and 2021-2025 (2020 missing), giving a proper train/test split.
MODEL = "ensemble"


def moneyline_odds():
    """Consensus closing decimal moneyline per game, home & away."""
    sched = _load_schedules()
    lo = pd.read_csv(LINES_GZ, usecols=["game_id", "market_type", "abbr", "odds"], low_memory=False)
    ml = lo[lo.market_type == "money_line"].copy()
    ml["odds"] = pd.to_numeric(ml.odds, errors="coerce")
    ml = ml.dropna(subset=["odds", "game_id", "abbr"])
    ml["game_id"] = ml.game_id.astype(np.int64)
    ml["team"] = ml.abbr.map(_abbr_to_team(sched, ml))
    ml["dec"] = np.where(ml.odds < 0, 1 + 100 / ml.odds.abs(), 1 + ml.odds / 100)
    hz = sched.dropna(subset=["game_id"]).assign(game_id=lambda d: d.game_id.astype(np.int64))
    home = hz.set_index("game_id").home_team.to_dict()
    ml["side"] = np.where(ml.team == ml.game_id.map(home), "home", "away")
    con = ml.groupby(["game_id", "side"]).dec.median().unstack()
    return con.rename(columns={"home": "home_dec", "away": "away_dec"})


def simulate(df, w_min, s_max, e_min, market="ML"):
    """Return per-bet frame for the rule. market 'ML' bets moneyline, 'ATS' at -110."""
    d = df[(df.week >= w_min) & (df.open_spread_home.abs() <= s_max)].copy()
    d["pick_home"] = d[MODEL] >= 0.5
    # edge = model P(pick) - opening-implied P(pick)
    d["edge"] = np.where(d.pick_home, d[MODEL] - d.open_prob, (1 - d[MODEL]) - (1 - d.open_prob))
    d = d[d.edge >= e_min]
    if market == "ML":
        d = d.dropna(subset=["home_dec", "away_dec"])
        dec = np.where(d.pick_home, d.home_dec, d.away_dec)
        won = d.pick_home == (d.home_win == 1)
        d["profit"] = np.where(won, (dec - 1) * STAKE, -STAKE)
    else:  # ATS at -110 vs the CLOSING spread
        d = d.dropna(subset=["spread_home"])
        cover_home = np.sign(d.margin + d.spread_home)
        keep = cover_home != 0
        d = d[keep]; cover_home = cover_home[keep]
        pick_home = d[MODEL] >= 0.5
        won = np.where(pick_home, cover_home > 0, cover_home < 0)
        d["profit"] = np.where(won, STAKE * 100 / 110, -STAKE)
        d["pick_home"] = pick_home
    d["won"] = d["profit"] > 0
    return d


def summary(bets):
    n = len(bets)
    if n == 0:
        return dict(n=0, win=np.nan, profit=0, roi=np.nan, t=np.nan)
    prof = bets.profit.sum()
    roi = prof / (n * STAKE)
    t = bets.profit.mean() / (bets.profit.std(ddof=1) / np.sqrt(n)) if bets.profit.std() > 0 else 0
    return dict(n=n, win=bets.won.mean(), profit=prof, roi=roi, t=t)


def main():
    b = pd.read_csv(os.path.join(RES, "experiments_predictions.csv.gz"))
    b = b.dropna(subset=[MODEL, "open_prob", "open_spread_home"])
    b = b.merge(moneyline_odds(), on="game_id", how="left")
    print(f"betting window: seasons {sorted(b.season.unique())}")
    print(f"eval games with model + opening line: {len(b)}  ({b[['home_dec','away_dec']].notna().all(1).mean()*100:.0f}% have moneyline)\n")

    train = b[b.season <= 2019]; test = b[b.season >= 2021]   # 2020 has no opening lines

    # ---- baselines (full sample) ----
    print("=== baselines, full betting window, flat $100 ===")
    for name, rule in [("bet model pick on EVERY game (ML)", dict(w_min=1, s_max=99, e_min=-9)),
                       ("bet market FAVORITE every game (ML)", "fav"),
                       ("bet all model-vs-open disagreements (ML)", dict(w_min=1, s_max=99, e_min=0.0001, dis=True))]:
        if rule == "fav":
            d = b.dropna(subset=["home_dec", "away_dec"]).copy()
            d["pick_home"] = d.open_spread_home < 0
            dec = np.where(d.pick_home, d.home_dec, d.away_dec)
            won = d.pick_home == (d.home_win == 1)
            d["profit"] = np.where(won, (dec - 1) * STAKE, -STAKE); d["won"] = d.profit > 0
        else:
            dis = rule.pop("dis", False)
            d = simulate(b, market="ML", **rule)
            if dis:
                d = d[(d[MODEL] >= .5) != (d.open_prob >= .5)]
        s = summary(d)
        print(f"  {name:45s} n={s['n']:5d} win={s['win']*100:5.1f}% ROI={s['roi']*100:+6.2f}% profit=${s['profit']:+,.0f}")

    # ---- threshold grid, choose on TRAIN, report on TEST ----
    grid = [(w, s, e, m) for w in (1, 5, 8, 10) for s in (3, 7, 99)
            for e in (0.0, 0.03, 0.05, 0.08) for m in ("ML", "ATS")]
    rows = []
    for w, s, e, m in grid:
        tr = summary(simulate(train, w, s, e, m)); te = summary(simulate(test, w, s, e, m))
        rows.append(dict(w_min=w, s_max=s, e_min=e, market=m,
                         tr_n=tr["n"], tr_roi=tr["roi"], te_n=te["n"], te_roi=te["roi"],
                         te_win=te["win"], te_profit=te["profit"], te_t=te["t"]))
    gr = pd.DataFrame(rows)
    gr.to_csv(os.path.join(RES, "betting_grid.csv"), index=False)

    # pick best TRAIN ROI with >=40 train bets, separately per market
    print("\n=== rule chosen on TRAIN 2013-2019, evaluated OUT-OF-SAMPLE on TEST 2021-2025 ===")
    best_rules = {}
    for m in ("ML", "ATS"):
        cand = gr[(gr.market == m) & (gr.tr_n >= 40)].sort_values("tr_roi", ascending=False)
        if len(cand) == 0:
            continue
        r = cand.iloc[0]; best_rules[m] = r
        print(f"  [{m}] rule: week>={int(r.w_min)}, |open spread|<={int(r.s_max)}, edge>={r.e_min:.2f}")
        print(f"       TRAIN 2013-19: n={int(r.tr_n)}  ROI={r.tr_roi*100:+.2f}%")
        print(f"       TEST  2021-25: n={int(r.te_n)}  win={r.te_win*100:.1f}%  ROI={r.te_roi*100:+.2f}%  "
              f"profit=${r.te_profit:+,.0f}  t={r.te_t:.2f}")

    # ---- full-sample performance of the model's structural sweet spot ----
    print("\n=== the disagreement 'sweet spot' (late + near pick'em), full window ===")
    for lbl, w, s, e, m in [("ML week>=5, |open|<=3, edge>=.05", 5, 3, 0.05, "ML"),
                            ("ML week>=8, |open|<=7, edge>=.05", 8, 7, 0.05, "ML"),
                            ("ATS week>=8, |open|<=7, edge>=.05", 8, 7, 0.05, "ATS")]:
        s2 = summary(simulate(b, w, s, e, m))
        print(f"  {lbl:36s} n={s2['n']:4d} win={s2['win']*100:5.1f}% ROI={s2['roi']*100:+6.2f}% "
              f"profit=${s2['profit']:+,.0f} t={s2['t']:.2f}")

    # ---- bankroll curve for the chosen ML rule over the whole timeline ----
    if "ML" in best_rules:
        r = best_rules["ML"]
        allbets = simulate(b, int(r.w_min), int(r.s_max), r.e_min, "ML").sort_values(["season", "week"])
        allbets["cum_profit"] = allbets.profit.cumsum()
        allbets.to_csv(os.path.join(RES, "betting_curve.csv"), index=False)
        print(f"\nchosen ML rule over ALL seasons: n={len(allbets)} "
              f"ROI={summary(allbets)['roi']*100:+.2f}% profit=${allbets.profit.sum():+,.0f}")


if __name__ == "__main__":
    main()
