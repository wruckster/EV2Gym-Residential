#!/usr/bin/env python3
"""
Plot signals from EV2Gym data sources, including:
- price (spot + forecast horizons)
- grid demand (from external features if available)
- household demand (from NSW household CSVs)
- PV generation (from NSW household CSVs or external features)

For price: uses the same loader path as the environment so spot and forecast
match training. Forecast matrices are expected in `env.price_forecast` with
shape (simulation_length, H), and spot prices in `env.spot_price`.

Usage examples:
  # Price with forecast overlay
  python scripts/plot_price_forecasts.py \
    --target price \
    --config ev2gym/example_config_files/residential_v2g.yaml \
    --horizons 1 6 12 24 \
    --start-step 0 --window-steps 288 \
    --save-path results/price_forecast_check.png

  # Household demand (actual series only)
  python scripts/plot_price_forecasts.py \
    --target household_demand \
    --config ev2gym/example_config_files/residential_v2g.yaml

  # PV (actual series only)
  python scripts/plot_price_forecasts.py \
    --target pv_generation \
    --config ev2gym/example_config_files/residential_v2g.yaml

  # Grid demand from external features (auto-detected column or override)
  python scripts/plot_price_forecasts.py \
    --target grid_demand \
    --config ev2gym/example_config_files/residential_v2g.yaml \
    --column total_demand
"""
from __future__ import annotations

import argparse
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.utilities.loaders import _load_household_profiles, _load_external_features
from ev2gym.models.utils.forecasting import create_lookahead_forecast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot selected EV2Gym signals (price, grid demand, household demand, PV generation).")
    parser.add_argument(
        "--target",
        type=str,
        choices=["price", "grid_demand", "household_demand", "pv_generation"],
        default="price",
        help="Which signal to plot.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="ev2gym/example_config_files/residential_v2g.yaml",
        help="Path to EV2Gym YAML config used to load external features.",
    )
    parser.add_argument(
        "--column",
        type=str,
        default=None,
        help="Optional column name to use for non-price targets when multiple candidates exist (case-insensitive).",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 6, 12, 24],
        help="Forecast horizons (in hours) to plot, e.g., 1 6 12 24",
    )
    parser.add_argument(
        "--show-forecast",
        action="store_true",
        help="For household_demand or pv_generation: overlay H-hour lookahead forecasts aligned to target times.",
    )
    parser.add_argument(
        "--align",
        type=str,
        choices=["now", "future"],
        default="now",
        help="Alignment of forecast curves: 'now' overlays forecasts at current timestamp (default). 'future' plots at t+h.",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=0,
        help="Start step within the simulation to plot from.",
    )
    parser.add_argument(
        "--window-steps",
        type=int,
        default=288,  # one day at 5-min steps
        help="Number of steps to include in the plot window.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Optional path to save the plot image. If not given, just shows the plot.",
    )
    return parser.parse_args()


def build_env(config_path: str) -> EV2Gym:
    env = EV2Gym(
        config_file=config_path,
        save_replay=False,
        verbose=False,
        render_mode=None,
    )
    return env


def align_forecast_to_actual(
    spot: np.ndarray,
    forecast_matrix: np.ndarray,
    horizon_hours: int,
    steps_per_hour: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align the forecast-at-time-t-for-t+H with the actual at t+H by shifting
    the spot price backward by H hours (or, equivalently, shifting the forecast
    forward). We trim to the overlapping window.
    """
    h_steps = horizon_hours * steps_per_hour
    if h_steps < 0:
        raise ValueError("horizon_hours must be non-negative")

    # Forecast is generated at each t for t+H; compare to actual at t+H.
    fc = forecast_matrix[:, horizon_hours - 1]
    # Valid indices where both are defined: t in [0, N-h_steps)
    n = len(spot)
    end = max(0, n - h_steps)
    aligned_fc = fc[:end]
    aligned_actual = spot[h_steps: h_steps + end]
    return aligned_fc, aligned_actual


def pick_column(df, candidates: List[str], override: Optional[str] = None) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    if override:
        key = override.lower()
        return lower_map.get(key)
    for key in candidates:
        if key in lower_map:
            return lower_map[key]
    # fallback: substring search
    for key in candidates:
        for lc, orig in lower_map.items():
            if key in lc:
                return orig
    return None


def main() -> None:
    args = parse_args()

    env = build_env(args.config)

    # Derive steps per hour from env.timescale (minutes per step)
    if env.timescale <= 0:
        raise ValueError("env.timescale must be > 0")
    steps_per_hour = int(round(60 / env.timescale))

    # Compute window indices generically once we know the series length
    def _plot_series(series: np.ndarray, label: str, ylabel: str):
        start = max(0, args.start_step)
        end = min(len(series), start + max(1, args.window_steps))
        t = np.arange(start, end)
        plt.figure(figsize=(12, 6))
        plt.plot(t, series[start:end], label=label, color="black", linewidth=2)
        plt.title(f"{label}")
        plt.xlabel("Step")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if args.save_path:
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            plt.savefig(args.save_path, dpi=160)
            print(f"Saved plot to {args.save_path}")
        else:
            plt.show()

    def _overlay_forecast_matrix(base_series_idx: np.ndarray, full_indexed_series, colors, label_prefix: str):
        """
        Build and overlay an H-hour lookahead forecast for each step in the window.
        - base_series_idx: array of integer timesteps for plotting x-axis
        - full_indexed_series: pandas Series with DatetimeIndex covering full history
        - colors: iterable of colors for horizons
        - label_prefix: label prefix for legend
        """
        # Determine the start timestamp for each step in window (env.sim_date is a datetime)
        base_time = pd.Timestamp(env.sim_date) + pd.Timedelta(minutes=int(args.start_step * env.timescale))
        # Construct timestamps for each step in plotting window
        window_steps = len(base_series_idx)
        step_times = [base_time + pd.Timedelta(minutes=int(i * env.timescale)) for i in range(window_steps)]

        # For each horizon, compute aligned forecast values and plot
        steps_per_hour = max(1, int(round(60 / env.timescale)))
        for c, h in zip(colors, args.horizons):
            if h <= 0:
                continue
            # For each step time, compute forecast for t+H using lookahead on the full series
            aligned_values = []
            for ts in step_times:
                params = dict(env.forecasting_config.get('params', {}))
                # Remove any conflicting keys; we pass explicit values below
                params.pop('forecast_horizon_hours', None)
                params.pop('start_time', None)
                params.pop('data', None)
                fc = create_lookahead_forecast(
                    data=full_indexed_series,
                    start_time=pd.Timestamp(ts),
                    forecast_horizon_hours=max(args.horizons),
                    **params,
                )
                # fc is hourly horizons length H; take (h-1)
                if len(fc) >= h:
                    aligned_values.append(fc[h - 1])
                else:
                    aligned_values.append(np.nan)
            aligned_values = np.asarray(aligned_values, dtype=float)
            h_steps = h * steps_per_hour
            idx = base_series_idx
            plot_idx = idx if args.align == "now" else (idx + h_steps)
            plt.plot(plot_idx, aligned_values, label=f"{label_prefix} H={h}h", color=c, alpha=0.6)

    if args.target == "price":
        # Validate presence of spot and forecast data
        if not hasattr(env, "spot_price") or env.spot_price is None:
            raise RuntimeError(
                "env.spot_price is missing. Ensure your external features contain a price column (e.g., RRP)"
            )
        if not hasattr(env, "price_forecast") or env.price_forecast is None:
            raise RuntimeError(
                "env.price_forecast is missing. Ensure your external features include forecast columns like rrp_h01..rrp_h24"
            )

        spot = np.asarray(env.spot_price)
        fc_matrix = np.asarray(env.price_forecast)

        start = max(0, args.start_step)
        end = min(len(spot), start + max(1, args.window_steps))
        t = np.arange(start, end)

        plt.figure(figsize=(12, 6))
        # Plot actual spot price in the window
        plt.plot(t, spot[start:end], label="Actual spot", color="black", linewidth=0.7)

        # Overlay forecasts for requested horizons (aligned to compare to the same timestamps)
        colors = plt.cm.viridis(np.linspace(0.15, 0.95, len(args.horizons)))
        for c, h in zip(colors, args.horizons):
            if h <= 0 or h - 1 >= fc_matrix.shape[1]:
                print(f"Skipping horizon {h}h: outside available forecast columns (H={fc_matrix.shape[1]})")
                continue
            fc_aligned, _ = align_forecast_to_actual(
                spot=spot, forecast_matrix=fc_matrix, horizon_hours=h, steps_per_hour=steps_per_hour
            )
            # Clip to plotting window indices
            # aligned series correspond to times t in [0, N-h_steps). We need those within [start, end)
            h_steps = h * steps_per_hour
            plot_start = start
            plot_end = min(end, len(fc_aligned))
            if plot_end <= plot_start:
                print(f"Horizon {h}h has no overlapping window to plot; skipping.")
                continue
            idx = np.arange(plot_start, plot_end)
            plt.plot(idx + h_steps, fc_aligned[plot_start:plot_end], label=f"Forecast H={h}h", color=c, alpha=0.8)

        plt.title("Spot price vs Forecast prices (aligned to target time)")
        plt.xlabel("Step")
        plt.ylabel("Price [$ / kWh]")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if args.save_path:
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            plt.savefig(args.save_path, dpi=160)
            print(f"Saved plot to {args.save_path}")
        else:
            plt.show()

    elif args.target == "household_demand":
        hh = _load_household_profiles(env)
        if hh is None or 'demand' not in hh.columns:
            raise RuntimeError("Household profiles not available or missing 'demand' column. Check YAML inflexible_loads.data_files.")
        series = np.asarray(hh['demand'].values, dtype=float)
        start = max(0, args.start_step)
        end = min(len(series), start + max(1, args.window_steps))
        t = np.arange(start, end)
        plt.figure(figsize=(12, 6))
        plt.plot(t, series[start:end], label="Household demand", color="black", linewidth=2)
        if args.show_forecast:
            # Build overlay using full resampled history with DatetimeIndex
            hh_full = _load_household_profiles(env, ignore_date_filter=True)
            if not isinstance(hh_full.index, pd.DatetimeIndex):
                raise RuntimeError("Full household profiles missing DatetimeIndex; cannot build forecast overlay.")
            colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(args.horizons)))
            _overlay_forecast_matrix(t, hh_full['demand'], colors, label_prefix="Forecast")
        plt.title("Household demand (with forecasts)" if args.show_forecast else "Household demand")
        plt.xlabel("Step")
        plt.ylabel("kW")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if args.save_path:
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            plt.savefig(args.save_path, dpi=160)
            print(f"Saved plot to {args.save_path}")
        else:
            plt.show()

    elif args.target == "pv_generation":
        hh = _load_household_profiles(env)
        if hh is not None and 'solar' in hh.columns:
            series = np.asarray(hh['solar'].values, dtype=float)
            start = max(0, args.start_step)
            end = min(len(series), start + max(1, args.window_steps))
            t = np.arange(start, end)
            plt.figure(figsize=(12, 6))
            plt.plot(t, series[start:end], label="PV generation", color="black", linewidth=1)
            if args.show_forecast:
                hh_full = _load_household_profiles(env, ignore_date_filter=True)
                if not isinstance(hh_full.index, pd.DatetimeIndex):
                    raise RuntimeError("Full household profiles missing DatetimeIndex; cannot build forecast overlay.")
                colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(args.horizons)))
                _overlay_forecast_matrix(t, hh_full['solar'], colors, label_prefix="Forecast")
            plt.title("PV generation (with forecasts)" if args.show_forecast else "PV generation")
            plt.xlabel("Step")
            plt.ylabel("kW")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            if args.save_path:
                os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
                plt.savefig(args.save_path, dpi=160)
                print(f"Saved plot to {args.save_path}")
            else:
                plt.show()
        else:
            ext = _load_external_features(env)
            if ext is None:
                raise RuntimeError("No PV series found in household profiles and no external features available.")
            col = pick_column(ext, candidates=["solar", "pv", "pv_generation"], override=args.column)
            if col is None:
                raise RuntimeError("Could not find a PV column in external features. Try --column <name>.")
            series = np.asarray(ext[col].values, dtype=float)
            _plot_series(series, label=f"{col}", ylabel="kW")

    elif args.target == "grid_demand":
        ext = _load_external_features(env)
        if ext is None:
            raise RuntimeError("External features not available; cannot plot grid demand.")
        col = pick_column(
            ext,
            candidates=["grid_demand", "total_demand", "operational_demand", "demand", "load"],
            override=args.column,
        )
        if col is None:
            raise RuntimeError("Could not find a demand column in external features. Try --column <name>.")
        series = np.asarray(ext[col].values, dtype=float)
        _plot_series(series, label=f"{col}", ylabel="kW")


if __name__ == "__main__":
    main()
