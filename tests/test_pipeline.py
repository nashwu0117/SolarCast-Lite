"""Pipeline test on the synthetic site; skipped until features.py and baselines.py exist."""
import pytest

pytest.importorskip("solarcast.features")
pytest.importorskip("solarcast.baselines")

from solarcast import data, pipeline  # noqa: E402


@pytest.fixture(scope="module")
def results(site_df):
    mp = pytest.MonkeyPatch()
    mp.setattr(data, "load_site", lambda site: site_df)
    yield pipeline.run_experiment("A"), pipeline.run_experiment("B")
    mp.undo()


def test_run_experiment_structure(results):
    r, _ = results
    assert r.site == "A" and set(r.splits) == {"train", "val", "test"}
    assert set(r.models) == {"Model A", "Model B"}
    names = {"Persistence", "Previous-day same hour", "Model A", "Model B"}
    for s in ("val", "test"):
        assert set(r.predictions[s]) == names
        assert set(r.metrics[s].index) == names
        assert r.metrics[s]["mae_kw"].is_monotonic_increasing
        for p in r.predictions[s].values():
            assert p.index.equals(r.splits[s].index)


def test_no_leakage_between_splits(results):
    r, _ = results
    assert r.splits["train"].index.max() < r.splits["val"].index.min()
    assert r.splits["val"].index.max() < r.splits["test"].index.min()


def test_model_b_beats_persistence_on_synthetic(results):
    t = results[0].metrics["test"]
    assert t.loc["Model B", "mae_kw"] < t.loc["Persistence", "mae_kw"]


def test_cross_site(results):
    a, b = results
    t = pipeline.cross_site(a, b)
    assert {"Model A", "Model B", "Persistence"} <= set(t.index)
    assert t.loc["Model B", "n"] == b.metrics["test"].loc["Model B", "n"]
