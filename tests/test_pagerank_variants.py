"""Verify the point-flow PageRank variants match their specification exactly.

points_against: team T sends its rank to opponents in proportion to the points
those opponents scored on it. Opponents scoring (3, 10, 7) on T  ->  15/50/35%.

points_keep: same, plus a self-loop weighted by T's own points. If T scored 20
in total (== points allowed here), it keeps 50% and splits the rest 15/50/35.
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.cfbrank.rankers import _result_edges


def _t_outflow(weight):
    # T (home) plays A, B, C; opponents score 3, 10, 7 on T; T scores 5, 8, 7 = 20.
    games = [("T", "A", 5, 3), ("T", "B", 8, 10), ("T", "C", 7, 7)]
    out = defaultdict(float)
    for home, away, hp, ap in games:
        hw = 1 if hp > ap else 0
        for s, d, wt in _result_edges(home, away, hp, ap, hw, 1.0, weight, 28.0):
            if s == "T":
                out[d] += wt
    tot = sum(out.values())
    return {k: round(v / tot * 100, 1) for k, v in out.items()}


def test_points_against():
    assert _t_outflow("points_against") == {"A": 15.0, "B": 50.0, "C": 35.0}


def test_points_keep():
    assert _t_outflow("points_keep") == {"A": 7.5, "B": 25.0, "C": 17.5, "T": 50.0}


if __name__ == "__main__":
    test_points_against()
    test_points_keep()
    print("ok: point-flow variants match spec")
