"""Gradient-boosting model for one-hour-ahead PV power: build, train, predict, save, explain."""
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from solarcast import config


def _columns(feature_set: str) -> list[str]:
    """Input column names for feature set "A" or "B" (imported lazily from features.py)."""
    from solarcast.features import FEATURE_SETS
    return FEATURE_SETS[feature_set]


def make_model(random_state: int = config.RANDOM_STATE) -> HistGradientBoostingRegressor:
    """An unfitted model. Early stopping is off so results are the same every run."""
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
        early_stopping=False, random_state=random_state,
    )


def train_model(feat_train: pd.DataFrame, feature_set: str):
    """Fit on the chosen feature columns only (never meta_* columns); remember which set."""
    cols = _columns(feature_set)
    model = make_model()
    model.fit(feat_train[cols], feat_train["target"])
    model.feature_set = feature_set
    return model


def predict(model, feat: pd.DataFrame) -> pd.Series:
    """Predict kW for every row and clip to the physical range [0, CAPACITY_KW]."""
    cols = _columns(model.feature_set)
    pred = model.predict(feat[cols]).clip(0, config.CAPACITY_KW)
    return pd.Series(pred, index=feat.index, name="prediction_kw")


def save_model(model, name: str):
    """Save under MODELS_DIR/<name>.joblib and return the path."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def load_model(name: str):
    """Load a model saved with save_model (feature_set attribute is kept)."""
    return joblib.load(config.MODELS_DIR / f"{name}.joblib")


def permutation_importance_table(model, feat: pd.DataFrame, n_repeats: int = 5) -> pd.DataFrame:
    """How much MAE gets worse when one feature is shuffled. Bigger = more important."""
    cols = _columns(model.feature_set)
    res = permutation_importance(
        model, feat[cols], feat["target"], n_repeats=n_repeats,
        scoring="neg_mean_absolute_error", random_state=config.RANDOM_STATE,
    )
    table = pd.DataFrame({"feature": cols, "importance_mean": res.importances_mean,
                          "importance_std": res.importances_std})
    return table.sort_values("importance_mean", ascending=False).reset_index(drop=True)
