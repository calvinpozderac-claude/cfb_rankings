"""Where is the edge on Polymarket? A category-level 'ripeness' map.

Data: manja316/polymarket-historical-data (free tier) -- markets.csv (9,550
markets with category, volume, liquidity, settlement outcome, live price) and
prices_sample.csv (a 10-hour orderbook snapshot, 2026-04-17).

IMPORTANT LIMITATION. True probabilistic calibration needs (forecast price at
time T, eventual outcome) pairs. The free file gives outcomes (from
`outcome_prices` settlement) but only *settled* prices for resolved markets, and
the price sample is a single 10-hour window -- so only ~32 markets can be paired.
We therefore (a) report the aggregate calibration on that small poolable set as a
directional sanity check only, and (b) rank category 'ripeness' from the signals
that ARE robustly measurable on all 9,550 markets: liquidity depth (a proxy for
how much sharp money competes), the live longshot surface (favorite-longshot
harvesting), and the toss-up share (modelable uncertainty available) -- combined
with a modelability judgement grounded in domain knowledge and in this repo's own
finding that even sports is only marginally beatable where the market is liquid.
"""
from __future__ import annotations
import ast
import os
import numpy as np
import pandas as pd

PM = os.environ.get("POLYMARKET_DIR", "/home/user/pm")
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "polymarket"))
FIG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "figures"))
os.makedirs(OUT, exist_ok=True)

# Modelability from public data (0-3) and why -- an explicit, labelled judgement,
# NOT measured from the price data.
MODELABILITY = {
    "weather": (3, "pro forecasts (NWS/ECMWF) beat laypeople outright"),
    "economics": (3, "scheduled data + consensus models (CPI, rates, jobs)"),
    "sports": (3, "box-score/Elo models work; but liquid US markets track Vegas"),
    "science_tech": (2, "base rates + domain models (launches, approvals, releases)"),
    "geopolitics": (2, "structured expert/base-rate forecasting helps, noisy"),
    "other": (2, "mixed bag (sports props, commodities, misc.)"),
    "politics": (1, "polls/fundamentals already priced; crowd very sharp"),
    "entertainment": (1, "soft, idiosyncratic (awards, box office)"),
    "pop_culture": (1, "soft, thin"),
    "crypto": (0, "price paths ~ random walk; not modelable from public data"),
}


def load():
    m = pd.read_csv(os.path.join(PM, "markets.csv"))
    for c in ["volume", "liquidity", "last_trade_price", "spread"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m["end"] = pd.to_datetime(m.end_date, utc=True, errors="coerce")

    def wf(op):
        try:
            a = ast.literal_eval(op); return 1 if a[0] > 0.5 else 0
        except Exception:
            return np.nan
    m["y"] = m.outcome_prices.apply(wf)
    p = pd.read_csv(os.path.join(PM, "prices_sample.csv"))
    return m, p


def aggregate_calibration(m, p):
    cut = pd.Timestamp("2026-04-17", tz="UTC")
    yes = p[p.outcome.astype(str).str.lower() == "yes"].groupby("market_id").price.median().rename("fc")
    j = m[(m.active == 0) & m.y.notna()].merge(yes, on="market_id")
    j = j[(j.end > cut) & (j.fc > 0.02) & (j.fc < 0.98)]
    return j


def category_table(m):
    a = m[m.active == 1]
    g = a.groupby("category").agg(
        n_active=("market_id", "size"),
        longshot_pct=("last_trade_price", lambda s: (s < 0.10).mean() * 100),
        tossup_pct=("last_trade_price", lambda s: ((s > 0.35) & (s < 0.65)).mean() * 100),
        med_liquidity=("liquidity", "median"),
    )
    g["n_total"] = m.groupby("category").size()
    g = g[g.n_active >= 20].copy()
    g["modelability"] = g.index.map(lambda c: MODELABILITY.get(c, (1, ""))[0])
    g["why_modelable"] = g.index.map(lambda c: MODELABILITY.get(c, (1, ""))[1])
    # efficiency proxy: deeper liquidity => sharper competition => less edge (0-1, higher=more efficient)
    ll = np.log10(g.med_liquidity.clip(lower=1))
    g["efficiency"] = ((ll - ll.min()) / (ll.max() - ll.min())).round(2)
    # ripeness = modelability high AND competition (efficiency) low, with a bonus for
    # a large uncertain (toss-up) surface to actually deploy an information edge on.
    g["ripeness"] = (g.modelability / 3 * (1 - g.efficiency)
                     * (0.5 + 0.5 * g.tossup_pct / g.tossup_pct.max())).round(3)
    return g.sort_values("ripeness", ascending=False).round(1)


def main():
    m, p = load()
    j = aggregate_calibration(m, p)
    brier = float(np.mean((j.fc - j.y) ** 2))
    print(f"AGGREGATE calibration (n={len(j)} poolable): Brier {brier:.3f}  "
          f"priced {j.fc.mean():.3f} vs realized {j.y.mean():.3f}  "
          f"(faint longshot overpricing; directional only)")
    g = category_table(m)
    g.to_csv(os.path.join(OUT, "category_summary.csv"))
    print("\n=== category edge map (sorted by ripeness) ===")
    print(g[["n_active", "med_liquidity", "efficiency", "longshot_pct", "tossup_pct",
             "modelability", "ripeness"]].to_string())

    # ---- figure: efficiency (x) vs modelability (y), bubble=n, color=ripeness ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 6.2))
    sc = ax.scatter(g.efficiency, g.modelability + np.random.default_rng(0).normal(0, .03, len(g)),
                    s=40 + g.n_active, c=g.ripeness, cmap="viridis",
                    edgecolor="k", linewidth=.5, zorder=3)
    off = {"economics": (6, -12), "sports": (6, 5), "crypto": (6, -2)}
    for cat, r in g.iterrows():
        ax.annotate(cat, (r.efficiency, r.modelability), fontsize=9,
                    xytext=off.get(cat, (6, 4)), textcoords="offset points")
    ax.axvspan(-0.05, 0.5, color="#009E73", alpha=0.06)
    ax.text(0.02, 3.25, "RIPE\nmodelable + thin (less sharp money)", color="#00754e", fontsize=9, va="top")
    ax.text(0.72, 0.15, "EFFICIENT\ndeep + hard to model", color="#8a5a00", fontsize=9)
    ax.set_xlabel("Market efficiency  (median liquidity, normalized → sharper competition)")
    ax.set_ylabel("Modelability from public data  (0 = crypto … 3 = weather)")
    ax.set_title("Polymarket edge map: which categories are ripe\n"
                 "(bubble = # active markets; color = ripeness score)")
    ax.set_ylim(-0.4, 3.6); ax.set_xlim(-0.05, 1.05)
    cb = plt.colorbar(sc, ax=ax); cb.set_label("ripeness")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig10_polymarket_edge_map.png")); plt.close(fig)
    print("\nwrote polymarket/category_summary.csv and fig10_polymarket_edge_map.png")


if __name__ == "__main__":
    main()
