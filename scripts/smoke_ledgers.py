from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

# Ensure project root is on path when run directly
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ev2gym.models.ev2gym_env import EV2Gym  # noqa: E402
import tempfile
import shutil
import pandas as pd
from ev2gym.utilities.loaders import _load_household_profiles


def _make_env(config_rel: str) -> EV2Gym:
    config_path = os.path.join(PROJECT_ROOT, config_rel)
    assert os.path.exists(config_path), f"Config not found: {config_path}"
    env = EV2Gym(
        config_file=config_path,
        save_replay=False,
        save_plots=False,
        lightweight_plots=True,  # speed
        verbose=False,
    )
    return env


def smoke_init_ledgers() -> None:
    env = _make_env("ev2gym/example_config_files/residential_v2g.yaml")
    assert env.global_buffers is not None, "global_buffers not initialized"
    assert isinstance(env.account_buffers, dict) and env.account_buffers, "account_buffers missing"
    # Expect one account per CS for now
    assert len(env.account_buffers) == len(env.charging_stations), "account_buffers size != number of CS"
    # Check columns contain core fields
    gcols = set(env.global_buffers.columns)
    expected = {
        "timestamp", "step", "total_power_usage_kw", "power_setpoint_kw",
        "ev_power_kw", "inflexible_load_kw", "solar_production_kw", "evs_parked",
    }
    assert expected.issubset(gcols), f"Missing global columns: {expected - gcols}"
    print("[OK] smoke_init_ledgers")


def smoke_step_writes(steps: int = 3) -> None:
    env = _make_env("ev2gym/example_config_files/residential_v2g.yaml")
    obs, _ = env.reset()
    # Step a few times with zero actions
    n_ports = env.number_of_ports
    for _ in range(steps):
        env.step(actions=np.zeros(n_ports))
    # Validate rows written (t=0..steps-1)
    gdf = env.global_buffers.to_pandas()
    assert len(gdf) >= steps, "Global ledger length < steps"
    # Power setpoint present
    assert np.isfinite(gdf.loc[0, "power_setpoint_kw"]) or np.isnan(gdf.loc[0, "power_setpoint_kw"]) == False, "power_setpoint_kw missing at t=0"
    # Per-account exists and has per-port fields
    for cs in env.charging_stations:
        buf = env.account_buffers.get(cs.id)
        assert buf is not None, f"account buffer missing for cs {cs.id}"
        adf = buf.to_pandas()
        assert len(adf) >= steps, f"account ledger too short for cs {cs.id}"
        # at least port0 columns exist
        assert f"port0_amps" in adf.columns and f"port0_soc" in adf.columns, f"port columns missing for cs {cs.id}"
    print("[OK] smoke_step_writes")


def smoke_parity() -> None:
    env = _make_env("ev2gym/example_config_files/residential_v2g.yaml")
    obs, _ = env.reset()
    n_ports = env.number_of_ports
    env.step(actions=np.zeros(n_ports))
    t = 0
    # Parity: evs_parked field equals env.current_evs_parked at the same step
    gdf = env.global_buffers.to_pandas()
    assert int(gdf.loc[t, "evs_parked"]) == int(env.current_evs_parked), "evs_parked mismatch"
    print("[OK] smoke_parity")


def smoke_reset_reinit() -> None:
    env = _make_env("ev2gym/example_config_files/residential_v2g.yaml")
    obs, _ = env.reset()
    # Step some timesteps
    n_ports = env.number_of_ports
    for _ in range(2):
        env.step(actions=np.zeros(n_ports))
    # Capture buffer id then reset
    gb_id_before = id(env.global_buffers)
    obs, _ = env.reset()
    gb_id_after = id(env.global_buffers)
    # Expectations: current_step reset, new buffers object, shapes align
    assert env.current_step == 0, "current_step not reset to 0"
    assert gb_id_before != gb_id_after, "global_buffers not re-initialized on reset()"
    assert env.global_buffers is not None and len(env.global_buffers.timestamps) == env.simulation_length, "timestamps length mismatch after reset"
    print("[OK] smoke_reset_reinit")


def smoke_time_features() -> None:
    env = _make_env("ev2gym/example_config_files/residential_v2g.yaml")
    obs, _ = env.reset()
    # After reset, before any step, we expect row t=0 to have time features matching env.sim_date
    t = 0
    gdf = env.global_buffers.to_pandas()
    for col in ("step_ratio", "dow", "hour", "minute"):
        assert col in gdf.columns, f"Missing time feature column: {col}"
    expected_dow = env.sim_date.weekday()
    expected_hour = env.sim_date.hour
    expected_min = env.sim_date.minute
    assert int(gdf.loc[t, "dow"]) == expected_dow, "dow mismatch at t=0"
    assert int(gdf.loc[t, "hour"]) == expected_hour, "hour mismatch at t=0"
    assert int(gdf.loc[t, "minute"]) == expected_min, "minute mismatch at t=0"
    # step_ratio in [0,1]
    sr = float(gdf.loc[t, "step_ratio"])
    assert 0.0 <= sr <= 1.0, "step_ratio out of range"
    print("[OK] smoke_time_features")


def smoke_price_forecast_columns() -> None:
    env = _make_env("ev2gym/example_config_files/residential_v2g.yaml")
    obs, _ = env.reset()
    t = 0
    pf = getattr(env, "price_forecast", None)
    assert pf is not None, "env.price_forecast missing"
    pf_row = np.asarray(pf[t]) if t < len(pf) else None
    assert pf_row is not None and pf_row.shape[0] >= 24, "price_forecast row has < 24 horizons"
    gdf = env.global_buffers.to_pandas()
    # Check columns exist and values match
    for h in range(24):
        col = f"price_fc_h{h+1:02d}"
        assert col in gdf.columns, f"Missing column {col}"
        v = float(gdf.loc[t, col])
        assert np.isfinite(v), f"Non-finite forecast at {col}"
        # Allow tiny numeric differences
        assert abs(v - float(pf_row[h])) < 1e-6, f"Forecast mismatch at {col}: {v} != {pf_row[h]}"
    print("[OK] smoke_price_forecast_columns")


def smoke_actions_limits_flags() -> None:
    env = _make_env("ev2gym/example_config_files/residential_v2g.yaml")
    obs, _ = env.reset()
    n_ports = env.number_of_ports
    # Take one no-op step to ensure charger currents and signals are set this episode
    env.step(actions=np.zeros(n_ports))
    t = 0
    eps = 1e-6
    for cs in env.charging_stations:
        buf = env.account_buffers.get(cs.id)
        assert buf is not None, f"account buffer missing for cs {cs.id}"
        adf = buf.to_pandas()
        # charger-level limit present
        assert "cs_kw_limit" in adf.columns, "cs_kw_limit missing"
        for p in range(cs.n_ports):
            # Columns exist
            for col in (f"port{p}_action_norm", f"port{p}_amps_limit", f"port{p}_connected", f"port{p}_time_to_departure", f"port{p}_time_since_arrival"):
                assert col in adf.columns, f"Missing column {col} for cs {cs.id}"
            # Bounds checks
            amps = float(adf.loc[t, f"port{p}_amps"]) if f"port{p}_amps" in adf.columns else np.nan
            limit = float(adf.loc[t, f"port{p}_amps_limit"]) if f"port{p}_amps_limit" in adf.columns else np.nan
            if np.isfinite(amps) and np.isfinite(limit):
                assert abs(amps) <= limit + 1e-3, f"amps exceed limit at cs {cs.id} p{p}: {amps}>{limit}"
            a_norm = float(adf.loc[t, f"port{p}_action_norm"]) if f"port{p}_action_norm" in adf.columns else np.nan
            if np.isfinite(a_norm):
                assert -1.001 <= a_norm <= 1.001, f"action_norm out of range at cs {cs.id} p{p}: {a_norm}"
            # Connected flag is 0/1
            conn = adf.loc[t, f"port{p}_connected"]
            assert int(conn) in (0, 1), f"connected not in {0,1} at cs {cs.id} p{p}: {conn}"
            if int(conn) == 1:
                ttd = adf.loc[t, f"port{p}_time_to_departure"]
                tsa = adf.loc[t, f"port{p}_time_since_arrival"]
                assert (np.isnan(ttd) or ttd >= -eps) and (np.isnan(tsa) or tsa >= -eps), "negative time features"
    print("[OK] smoke_actions_limits_flags")

def smoke_reward_tracking() -> None:
    env = _make_env("ev2gym/example_config_files/residential_v2g.yaml")
    obs, _ = env.reset()
    n_ports = env.number_of_ports
    steps = 5
    for _ in range(steps):
        env.step(actions=np.zeros(n_ports))
    gdf = env.global_buffers.to_pandas()
    # Check columns exist
    for col in ("reward_step", "reward_cumsum", "tracking_error", "invalid_action_punishment"):
        assert col in gdf.columns, f"Missing column {col}"
    # Validate cumsum correctness for first few steps (rows 0..steps-1)
    rewards = [float(gdf.loc[t, "reward_step"]) for t in range(steps)]
    for t in range(steps):
        rc = float(gdf.loc[t, "reward_cumsum"]) if np.isfinite(gdf.loc[t, "reward_cumsum"]) else np.nan
        if np.isfinite(rc):
            assert abs(rc - sum(rewards[: t + 1])) < 1e-5, f"reward_cumsum mismatch at t={t}"
        te = gdf.loc[t, "tracking_error"]
        if np.isfinite(te):
            assert te >= -1e-8, f"negative tracking_error at t={t}"
    print("[OK] smoke_reward_tracking")

def smoke_export_parquet() -> None:
    env = _make_env("ev2gym/example_config_files/residential_v2g.yaml")
    obs, _ = env.reset()
    # Step a few times
    n_ports = env.number_of_ports
    for _ in range(3):
        env.step(actions=np.zeros(n_ports))
    tmpdir = tempfile.mkdtemp(prefix="ev2gym_ledgers_")
    try:
        paths = env.save_ledgers_parquet(tmpdir)
        # Load back
        gdf = pd.read_parquet(paths["global"])  # type: ignore
        assert len(gdf) == env.simulation_length, "global parquet row count mismatch"
        # Check one account file exists and loads
        any_acc = next(iter(paths["accounts"].values())) if paths["accounts"] else None
        assert any_acc is not None, "no account parquet written"
        adf = pd.read_parquet(any_acc)  # type: ignore
        assert len(adf) == env.simulation_length, "account parquet row count mismatch"
        # Spot check that time features are present
        for col in ("step_ratio", "dow", "hour", "minute"):
            assert col in gdf.columns, f"missing {col} in global parquet"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print("[OK] smoke_export_parquet")

def smoke_weather_dr_forecasts() -> None:
    env = _make_env("ev2gym/example_config_files/residential_v2g.yaml")
    # Ensure forecasting enabled with targets by config; test assumes loader populates full_timeseries_data
    obs, _ = env.reset()
    gdf = env.global_buffers.to_pandas()
    t = 0
    # Columns should exist for weather forecasts; DR is optional
    for prefix in ("temp_fc_h", "wind_fc_h"):
        for h in range(1, 25):
            assert f"{prefix}{h:02d}" in gdf.columns, f"Missing {prefix}{h:02d}"
    # If forecasts are present in env, the first few should be finite
    tf = getattr(env, "temperature_forecast", None)
    wf = getattr(env, "wind_forecast", None)
    drf = getattr(env, "dr_event_forecast", None)
    if tf is not None:
        assert len(tf) >= 24, "temperature_forecast length < 24"
        vals = [gdf.loc[t, f"temp_fc_h{h:02d}"] for h in range(1, 25)]
        assert np.isfinite(vals).all(), "Non-finite temp forecast values"
    if wf is not None:
        assert len(wf) >= 24, "wind_forecast length < 24"
        vals = [gdf.loc[t, f"wind_fc_h{h:02d}"] for h in range(1, 25)]
        assert np.isfinite(vals).all(), "Non-finite wind forecast values"
    # DR columns are optional; validate only if present
    dr_cols_present = all((f"dr_fc_h{h:02d}" in gdf.columns) for h in range(1, 25))
    if dr_cols_present and drf is not None:
        assert len(drf) >= 24, "dr_event_forecast length < 24"
        vals = [gdf.loc[t, f"dr_fc_h{h:02d}"] for h in range(1, 25)]
        # Values are 0/1
        assert all(int(v) in (0, 1) for v in vals), "DR forecast not binary"
    print("[OK] smoke_weather_dr_forecasts")

def smoke_household_parquet_loader() -> None:
    """Validate parquet + household_ids path in YAML for household profiles.
    Checks both filtered (windowed) and ignore_date_filter behaviors.
    """
    env = _make_env("ev2gym/example_config_files/residential_v2g.yaml")
    # Windowed path
    df_win = _load_household_profiles(env, ignore_date_filter=False)
    assert df_win is not None, "_load_household_profiles returned None for windowed path"
    # Expect RangeIndex-like frame with timestamp column and exact simulation_length rows
    assert hasattr(df_win, 'shape') and len(df_win) == env.simulation_length, "windowed length mismatch"
    for c in ("timestamp", "demand", "solar"):
        assert c in df_win.columns, f"missing column {c} in windowed"
    # Ignore-date-filter path: DatetimeIndex, not padded/truncated
    df_full = _load_household_profiles(env, ignore_date_filter=True)
    assert df_full is not None, "_load_household_profiles returned None for full path"
    # Must be a resampled time series with DatetimeIndex
    import pandas as _pd
    assert isinstance(df_full.index, _pd.DatetimeIndex), "full path must preserve DatetimeIndex"
    for c in ("demand", "solar"):
        assert c in df_full.columns, f"missing column {c} in full"
    print("[OK] smoke_household_parquet_loader")

if __name__ == "__main__":
    smoke_init_ledgers()
    smoke_step_writes(steps=3)
    smoke_parity()
    smoke_reset_reinit()
    smoke_time_features()
    smoke_price_forecast_columns()
    smoke_actions_limits_flags()
    smoke_reward_tracking()
    smoke_export_parquet()
    smoke_weather_dr_forecasts()
    smoke_household_parquet_loader()
    print("All smoke tests passed.")
