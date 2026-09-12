# cfb_rankings

Creative college-football team-ranking systems, judged by how well they **predict the next
week's games** out-of-sample against the **Vegas closing line**.

Three model classes are implemented and backtested walk-forward over **2001–2025**:

* **Network / PageRank** — win-, margin-, and points-weighted random walks over the
  win/loss graph, with tunable edge weighting — including a **points-flow-with-self-retention**
  variant that matches Elo.
* **Elo / paired-comparison** — margin-aware online Elo, Bradley–Terry logistic strengths,
  and the intransitive **Blade–Chest** low-rank model.
* **Machine learning** — gradient boosting and a neural net on pre-game features, plus a
  stacked **ensemble**.

All models carry information across seasons — Elo via between-season regression, the batch
models via a tuned exponential **recency decay** over the last several seasons — so a program
that was strong last year is rated strong in week 1.

**Benchmarks:** the betting market's consensus closing spread and CollegeFootballData's own
Elo.

👉 **Read [`REPORT.md`](REPORT.md) for the full write-up, figures, and findings.**

### TL;DR
Tuned Elo ≈ gradient boosting ≈ the ensemble, all at **~71% accuracy / 0.55 log loss** on
recent seasons. Most PageRanks predict poorly, but a points-flow walk with self-retention
(`points+keep`) matches Elo at **69.9%**. Adding play-level box-score efficiency, returning
production and Massey-style margin ratings (`scripts/experiments.py`) lifts the best
results-only model to **72.2% / 0.544** — about a quarter of the way to Vegas — while travel,
margin-regression and week-specific stacking add nothing. **Nothing beats the closing line**
(all models ≈ 49.5% against the spread vs a 52.4% break-even), and where our best model
disagrees with the market on the winner, the market is right 58% of the time.

### Layout
```
src/cfbrank/
  data.py        # load & clean games + betting lines (majority-vote spread mapping)
  elo.py         # online Elo family
  rankers.py     # PageRank family, Bradley-Terry, Blade-Chest
  features.py    # leak-free pre-game features
  mlmodels.py    # gradient boosting, MLP
  backtest.py    # walk-forward harness + benchmarks
  metrics.py     # accuracy, log loss, Brier, calibration
scripts/
  run_backtest.py    # runs the whole study -> results/
  make_figures.py    # figures + final rankings
  ats_analysis.py    # market-efficiency tests
results/           # predictions, metrics, tuning tables, figures
```

### Data
Sourced from the public [`cfbfastR-data`](https://github.com/sportsdataverse/cfbfastR-data)
mirror of CollegeFootballData.com. See `src/cfbrank/data.py` for provenance and the note in
`REPORT.md` on why betting lines (not AP/CFP polls) are the benchmark.

### Run
```bash
pip install -r requirements.txt
python -m src.cfbrank.data && python scripts/run_backtest.py && python scripts/make_figures.py
```
