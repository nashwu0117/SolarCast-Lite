"""End-to-end experiment: load -> features -> split -> train A and B -> score on val and test."""
from dataclasses import dataclass

import pandas as pd

from solarcast import data, evaluate, models
from solarcast.baselines import BASELINES
from solarcast.features import build_features


@dataclass
class ExperimentResult:
    site: str
    df: pd.DataFrame
    feat: pd.DataFrame
    splits: dict
    models: dict
    predictions: dict   # {split_name: {model_name: Series}}
    metrics: dict       # {split_name: DataFrame}


def _predict_all(trained: dict, feat: pd.DataFrame) -> dict:
    """Both baselines plus every trained model, as {name: Series of kW}."""
    preds = {name: fn(feat) for name, fn in BASELINES.items()}
    preds.update({name: models.predict(m, feat) for name, m in trained.items()})
    return preds


def run_experiment(site: str = "A") -> ExperimentResult:
    """Train on train only; val is for choices, test is what we report."""
    df = data.load_site(site)
    feat = build_features(df)
    splits = evaluate.split_by_time(feat)
    trained = {"Model A": models.train_model(splits["train"], "A"),
               "Model B": models.train_model(splits["train"], "B")}
    predictions = {s: _predict_all(trained, splits[s]) for s in ("val", "test")}
    metrics = {s: evaluate.compare(splits[s], predictions[s]) for s in predictions}
    return ExperimentResult(site, df, feat, splits, trained, predictions, metrics)


def cross_site(result_a: ExperimentResult, result_b: ExperimentResult) -> pd.DataFrame:
    """Site A's trained models (and the baselines) scored on Site B's test split."""
    test_b = result_b.splits["test"]
    return evaluate.compare(test_b, _predict_all(result_a.models, test_b))
