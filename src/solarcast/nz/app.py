"""Helpers for the notebooks: load the replay data and the trained model, and build the Service.

The replay file (data/nz/replay.csv.gz) is a small slice of Open-Meteo history, so the dashboard starts in seconds
instead of downloading five years of weather. Live mode needs the internet; if it fails, replay still works.
"""
from pathlib import Path

import pandas as pd

from solarcast.nz import config, data, model as nzmodel
from solarcast.nz.service import Service

REPLAY_PATH = config.ROOT / "data" / "nz" / "replay.csv.gz"


def load_replay(replay_path=REPLAY_PATH) -> pd.DataFrame:
    """Replay CSV -> long DataFrame with the same columns as data.fetch_history (solar position recomputed)."""
    df = pd.read_csv(replay_path)
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    parts = []
    for region, g in df.groupby("region", sort=False):
        g = g.sort_values("time_utc").reset_index(drop=True)
        lat, lon = config.REGIONS[region]
        # radiation stamped T is the mean over [T-1h, T), so the matching sun position is at T - 30 min
        g["solar_elevation_deg"], g["solar_azimuth_deg"] = data.solar_position(g["time_utc"] - pd.Timedelta("30min"), lat, lon)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


class _SafeLiveService(Service):
    """Service whose live mode never takes the dashboard down: when the live download fails, say so (HTTP 502)."""

    def _live_state(self):
        try:
            return super()._live_state()
        except Exception as exc:                                   # network down, rate limit, API change ...
            raise RuntimeError(f"Live data is not available right now ({type(exc).__name__}). Replay mode still works.") from exc


def load_service(replay_path=REPLAY_PATH, model_path=None, meta_path=None) -> Service:
    """Load the model and the replay data and return a Service.

    Raises a clear error if the model cannot be loaded (for example a different scikit-learn version is installed:
    install the version named in models/nz_model.json first, see the notebook setup cell).
    """
    model = nzmodel.load(model_path, meta_path)
    if model is None:
        import sklearn
        want = nzmodel.load_meta(meta_path).get("sklearn_version", "?")
        raise RuntimeError(f"Could not load the model (installed scikit-learn {sklearn.__version__}, "
                           f"model saved with {want}). Run: pip install scikit-learn=={want}  and restart the runtime.")
    ahead, info = load_ahead_models()
    return _SafeLiveService(model, load_replay(replay_path), meta=nzmodel.load_meta(meta_path),
                            ahead_models=ahead, ahead_info=info, recent_replay=True)


def load_ahead_models() -> tuple[dict, dict]:
    """The 2- and 3-hours-ahead models (scripts/train_horizons.py) for "Predict next"; missing ones are skipped."""
    import json
    import joblib
    import sklearn
    info_path = config.ROOT / "models" / "nz_horizons.json"
    try:
        info = json.loads(info_path.read_text())
    except (OSError, ValueError):
        return {}, {}
    if info.get("sklearn_version") != sklearn.__version__:
        return {}, info
    models = {}
    for h in (2, 3):
        try:
            models[h] = joblib.load(config.ROOT / "models" / f"nz_model_h{h}.joblib")
        except Exception:                                          # noqa: BLE001  that horizon is simply not offered
            pass
    return models, info
