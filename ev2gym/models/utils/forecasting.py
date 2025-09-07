import pandas as pd
import numpy as np
from perlin_noise import PerlinNoise
from typing import List, Optional
import pytz

def generate_smooth_noise(timesteps: int, octaves: int, seed: Optional[int] = None) -> List[float]:
    """
    Generates a smoothly varying noise series using Perlin noise.

    Args:
        timesteps (int): The number of timesteps to generate noise for.
        octaves (int): The number of octaves for the Perlin noise generator.
        seed (Optional[int]): The random seed for the noise generator.

    Returns:
        List[float]: A list of noise values ranging from -1 to 1.
    """
    noise = PerlinNoise(octaves=octaves, seed=seed)
    return [noise([i / timesteps]) for i in range(timesteps)]

_clf_dbg_printed = False

def create_lookahead_forecast(
    data: pd.Series,
    start_time: pd.Timestamp,
    forecast_horizon_hours: int = 24,
    noise_level: float = 0.1,
    noise_octaves: int = 4,
    noise_seed: Optional[int] = None,
) -> np.ndarray:
    """
    Creates a lookahead forecast by taking future values from a time series and adding smooth noise.

    Args:
        data (pd.Series): The time series data with a DatetimeIndex.
        start_time (pd.Timestamp): The current time from which to start the forecast.
        forecast_horizon_hours (int): The number of hours to forecast ahead.
        noise_level (float): The magnitude of the noise to add to the forecast.
        noise_octaves (int): The complexity of the Perlin noise.
        noise_seed (Optional[int]): Seed for the noise generator for reproducibility.

    Returns:
        np.ndarray: A numpy array containing the 24-hour forecast.
    """
    # Ensure timezone is fixed +10 (no DST) and index is tz-aware
    fixed_tz = pytz.FixedOffset(600)  # +10 hours
    series = data.copy()
    # Force numeric and handle NaNs progressively
    try:
        series = pd.to_numeric(series, errors='coerce')
    except Exception:
        pass
    if series.index.tz is None:
        series.index = series.index.tz_localize(fixed_tz)
    else:
        series.index = series.index.tz_convert(fixed_tz)
    s = pd.Timestamp(start_time)
    if s.tzinfo is None:
        s = s.tz_localize(series.index.tz)
    else:
        s = s.tz_convert(series.index.tz)

    # Build target timestamps at exact hour offsets and sample nearest values from the original series
    targets = pd.DatetimeIndex([s + pd.Timedelta(hours=h) for h in range(1, forecast_horizon_hours + 1)])
    # Use indexer with nearest; robust fallbacks for missing values
    idxpos = series.index.get_indexer(targets, method='nearest')
    vals: list[float] = []
    s_ff = series.ffill().bfill()
    for pos, t in zip(idxpos, targets):
        v = None
        if pos != -1:
            v = series.iloc[pos]
        if v is None or pd.isna(v):
            try:
                v = s_ff.asof(t)
            except Exception:
                v = None
        if v is None or pd.isna(v):
            v = 0.0
        vals.append(float(v))
    base = np.asarray(vals, dtype=float)

    # Add smooth multiplicative noise around 1.0
    if noise_level and noise_level > 0:
        noise = np.array(generate_smooth_noise(len(base), noise_octaves, noise_seed))
        base = base * (1.0 + noise_level * noise)
    # One-time debug dump to help diagnose zero forecasts
    global _clf_dbg_printed
    if not _clf_dbg_printed:
        try:
            preview = ", ".join(f"{x:.4f}" for x in base[:6])
            print(f"[DBG clf] horizon={forecast_horizon_hours} start={s} series_len={len(series)} preview=[{preview}] min={base.min():.4f} max={base.max():.4f}")
        except Exception:
            pass
        _clf_dbg_printed = True
    return base.astype(float)
