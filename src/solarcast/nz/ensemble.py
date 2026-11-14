"""The v4 model: a raw-GHI model and a clear-sky-index model, averaged.

Most of the hour-to-hour change in sunlight is the sun moving, which astronomy knows exactly. So one model predicts the
clear-sky index k = GHI / clear-sky GHI of the target hour (how cloudy it is) and we multiply back by the target hour's
clear-sky GHI; a second model predicts raw GHI directly. The final forecast is the average of the two.

Rules (identical to scripts/train_nz_v4.py):
  * raw model:   next-hour GHI, clipped at 0, exactly 0 when the sun is down (trained on rows with the sun up).
  * index model: k clipped to [0, 1.5] times cs_target where cs_target >= CS_MIN (W/m2); below that (dawn/dusk) k is
                 unstable, so the raw model's value is used instead (trained on rows with cs_target >= CS_MIN).
  * average of the two, 0 when the sun is down.

The class lives in this importable module so that joblib can unpickle the saved model anywhere.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor

from solarcast.nz.features import FEATURES

CS_MIN = 20.0            # W/m2
K_MAX = 1.5
DEFAULT_PARAMS = {"learning_rate": 0.08, "max_iter": 300, "max_leaf_nodes": 31, "min_samples_leaf": 50,
                  "l2_regularization": 1.0, "loss": "squared_error"}


class SolarEnsemble(RegressorMixin, BaseEstimator):
    """Average of a raw-GHI gradient-boosting model and a clear-sky-index gradient-boosting model.

    A scikit-learn style estimator (predict/fit) so that permutation_importance and joblib work with it.
    """

    def __init__(self, features=None, params=None):
        self.features = list(features or FEATURES)
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.raw = None
        self.kidx = None

    def _hgb(self, **over):
        return HistGradientBoostingRegressor(categorical_features=[self.features.index("region_id")], random_state=0,
                                             **{**self.params, **over})

    def fit(self, feat: pd.DataFrame, y=None):
        """Fit both models on a feature frame that has `target_ghi` (night rows are skipped by the sun-up filters)."""
        known = feat["target_ghi"].notna()
        up = feat[(feat["elev_target"] > 0) & known]
        self.raw = self._hgb().fit(up[self.features], up["target_ghi"])
        hi = feat[(feat["cs_target"] >= CS_MIN) & known]
        k = (hi["target_ghi"] / hi["cs_target"]).clip(0, K_MAX)
        self.kidx = self._hgb(loss="squared_error").fit(hi[self.features], k)      # squared error: k is a smooth ratio
        return self

    def predict_parts(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(raw-model GHI, clear-sky-index-model GHI), both in W/m2 and both 0 when the sun is down."""
        up = X["elev_target"].to_numpy() > 0
        raw = np.where(up, np.clip(self.raw.predict(X[self.features]), 0, None), 0.0)
        cs = X["cs_target"].to_numpy()
        kghi = np.clip(self.kidx.predict(X[self.features]), 0, K_MAX) * cs
        kpart = np.where(up, np.where(cs >= CS_MIN, kghi, raw), 0.0)
        return raw, kpart

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Next-hour GHI in W/m2: the average of the two models, never negative, 0 when the sun is down."""
        raw, kpart = self.predict_parts(X)
        return (raw + kpart) / 2
