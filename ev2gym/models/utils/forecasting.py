import pandas as pd
import numpy as np
from perlin_noise import PerlinNoise
from typing import List, Optional

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
    # Define the forecast period
    end_time = start_time + pd.Timedelta(hours=forecast_horizon_hours)

    # Extract the actual future data for the forecast horizon
    future_data = data[start_time:end_time]

    # Resample to hourly if not already
    if pd.infer_freq(future_data.index) != 'H':
        # Ensure we have enough data to resample, forward fill if necessary
        future_data = future_data.resample('H').mean().ffill()

    # Ensure the forecast has exactly the desired number of steps
    if len(future_data) > forecast_horizon_hours:
        future_data = future_data.iloc[:forecast_horizon_hours]
    elif len(future_data) < forecast_horizon_hours:
        # If not enough data, pad with the last known value
        padding_needed = forecast_horizon_hours - len(future_data)
        last_value = future_data.iloc[-1] if not future_data.empty else 0
        padding_index = pd.date_range(start=future_data.index[-1] + pd.Timedelta(hours=1), periods=padding_needed, freq='H')
        padding_series = pd.Series([last_value] * padding_needed, index=padding_index)
        future_data = pd.concat([future_data, padding_series])

    # Generate smooth noise
    smooth_noise = generate_smooth_noise(
        timesteps=forecast_horizon_hours,
        octaves=noise_octaves,
        seed=noise_seed
    )

    # Apply noise to the forecast
    forecast_values = future_data.values
    noise_to_apply = np.array(smooth_noise) * noise_level * np.mean(np.abs(forecast_values))
    
    forecast = forecast_values + noise_to_apply

    return forecast.astype(np.float32)
