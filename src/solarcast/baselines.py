"""Simple forecasts that every model must beat. Each takes the build_features table, returns kW."""
import pandas as pd


def persistence(feat: pd.DataFrame) -> pd.Series:
    """Next hour will equal the hour that just ended."""
    return feat["power_lag1"]


def previous_day_same_hour(feat: pd.DataFrame) -> pd.Series:
    """Next hour will equal the same hour yesterday."""
    return feat["power_same_hour_yesterday"]


BASELINES = {"Persistence": persistence, "Previous-day same hour": previous_day_same_hour}
