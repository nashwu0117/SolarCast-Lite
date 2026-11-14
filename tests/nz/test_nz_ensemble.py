import pickle

import numpy as np
import pandas as pd
import pytest

from solarcast.nz import model as nzmodel
from solarcast.nz.ensemble import CS_MIN, SolarEnsemble
from solarcast.nz.features import FEATURES, V1_FEATURES, build_features


@pytest.fixture(scope="module")
def fitted(long_df):
    feat = build_features(long_df)
    tm = feat.index.get_level_values("target_time")
    train, test = feat[tm < "2025-12-01"], feat[tm >= "2025-12-01"]
    return nzmodel.train(train, max_iter=40), train, test


def test_train_returns_ensemble_with_both_models(fitted):
    m, _, _ = fitted
    assert isinstance(m, SolarEnsemble) and m.raw is not None and m.kidx is not None
    assert m.features == FEATURES and m.raw.loss == "absolute_error" and m.kidx.loss == "squared_error"


def test_average_rule_zero_at_night_and_fallback_at_dawn(fitted):
    m, _, test = fitted
    X = test[FEATURES]
    raw, kpart = m.predict_parts(X)
    pred = m.predict(X)
    assert np.allclose(pred, (raw + kpart) / 2) and (pred >= 0).all()
    night = (test["elev_target"] <= 0).to_numpy()
    assert (pred[night] == 0).all() and night.any()
    low = ((test["cs_target"] < CS_MIN) & (test["elev_target"] > 0)).to_numpy()
    assert low.any() and np.allclose(kpart[low], raw[low])                # index model is not used below CS_MIN
    hi = (test["cs_target"] >= CS_MIN).to_numpy()
    k = np.clip(m.kidx.predict(X[hi]), 0, 1.5) * test["cs_target"].to_numpy()[hi]
    assert np.allclose(kpart[hi], k)                                       # k * clear-sky GHI where the sun is high


def test_pickle_roundtrip_and_model_api(fitted):
    m, _, test = fitted
    m2 = pickle.loads(pickle.dumps(m))
    assert np.allclose(m2.predict(test[FEATURES]), m.predict(test[FEATURES]))
    p = nzmodel.predict(m, test)
    assert isinstance(p, pd.Series) and p.index.equals(test.index) and (p[test["elev_target"] <= 0] == 0).all()
    ev = nzmodel.evaluate(test, nzmodel.all_predictions(m, test))
    assert ev["overall"]["model"]["mae_wm2"] < ev["overall"]["persistence"]["mae_wm2"]


def test_forecast_inputs_help_on_synthetic_data(long_df):
    feat = build_features(long_df)
    tm = feat.index.get_level_values("target_time")
    train, test = feat[tm < "2025-12-01"], feat[tm >= "2025-12-01"]
    with_fc = nzmodel.train(train, max_iter=60)
    no_fc = SolarEnsemble(V1_FEATURES, {**nzmodel.V1_PARAMS, "max_iter": 60}).fit(train)       # same data, no fc_* inputs
    def mae(mod, cols):
        p = pd.Series(np.where(test["elev_target"] > 0, mod.predict(test[cols]), 0.0), index=test.index)
        return nzmodel.evaluate(test, {"model": p})["overall"]["model"]["mae_wm2"]

    assert mae(with_fc, FEATURES) < mae(no_fc, V1_FEATURES)
