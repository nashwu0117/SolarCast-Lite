import json

import numpy as np
import pandas as pd
import pytest

from solarcast.nz import config, model as nzmodel
from solarcast.nz.features import FEATURES, build_features


@pytest.fixture(scope="module")
def fitted(long_df):
    feat = build_features(long_df)
    tm = feat.index.get_level_values("target_time")
    train, test = feat[tm < "2025-12-01"], feat[tm >= "2025-12-01"]
    return nzmodel.train(train, max_iter=60), train, test


def test_predictions_nonnegative_and_zero_at_night(fitted):
    m, _, test = fitted
    p = nzmodel.predict(m, test)
    assert (p >= 0).all() and (p[test["elev_target"] <= 0] == 0).all() and p.index.equals(test.index)


def test_model_beats_persistence_on_synthetic(fitted):
    m, _, test = fitted
    ev = nzmodel.evaluate(test, nzmodel.all_predictions(m, test))
    assert ev["overall"]["model"]["mae_wm2"] < ev["overall"]["persistence"]["mae_wm2"]
    assert set(ev["per_region"]) == {"Auckland", "Wellington", "Queenstown"}
    assert {"mae_wm2", "rmse_wm2", "mae_kw", "rmse_kw", "n"} <= set(ev["overall"]["clear_sky"])


def test_baselines(fitted):
    _, _, test = fitted
    assert (nzmodel.persistence(test) == test["ghi_t"].clip(lower=0)).all()
    assert (nzmodel.clear_sky_persistence(test) >= 0).all()


def test_save_load_roundtrip_and_failures(fitted, tmp_path, monkeypatch):
    m, _, test = fitted
    mp, jp = tmp_path / "m.joblib", tmp_path / "m.json"
    nzmodel.save(m, {"x": 1}, model_path=mp, meta_path=jp)
    meta = json.loads(jp.read_text())
    assert meta["features"] == FEATURES and "sklearn_version" in meta and meta["metrics"] == {"x": 1}
    loaded = nzmodel.load(mp, jp)
    assert np.allclose(nzmodel.predict(loaded, test), nzmodel.predict(m, test))
    meta["sklearn_version"] = "0.0.1"
    jp.write_text(json.dumps(meta))
    with pytest.warns(UserWarning):
        assert nzmodel.load(mp, jp) is None                       # version mismatch -> retrain
    mp.write_bytes(b"junk")
    jp.unlink()
    with pytest.warns(UserWarning):
        assert nzmodel.load(mp, jp) is None                       # corrupt file


def test_build_report(fitted):
    m, train, test = fitted
    rep = nzmodel.build_report(m, train.iloc[-3000:], test)
    assert set(rep) == {"periods", "overall", "per_region", "importance", "sample"}
    assert [o["model"] for o in rep["overall"]] == ["SolarCast model", "Persistence", "Clear-sky persistence"]
    assert len(rep["importance"]) == 12 and len(rep["per_region"]) == 9
    s = rep["sample"]
    assert len(s["times"]) == len(s["actual_kw"]) == len(s["predicted_kw"]) == len(s["persistence_kw"]) > 100
    json.dumps(rep)                                               # JSON-safe
