"""ML models (gradient boosting, neural net) and the stacked ensemble."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

ML_FEATURES = [
    "elo_prob", "elo_diff", "home_ind", "neutral", "conf_game",
    "wp_diff", "margin_diff", "gp_home", "gp_away", "rest_diff",
]


def make_gbm():
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=400,
        l2_regularization=1.0, min_samples_leaf=40, random_state=0,
    )


def make_mlp():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-2, max_iter=400,
                      early_stopping=True, random_state=0),
    )
