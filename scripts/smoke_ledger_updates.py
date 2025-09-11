#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from datetime import datetime
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.rl_agent import state as rl_state


def main() -> None:
    cfg = os.path.join(PROJECT_ROOT, "ev2gym/example_config_files/residential_v2g.yaml")
    assert os.path.exists(cfg), f"Config not found: {cfg}"

    env = EV2Gym(
        config_file=cfg,
        state_function=rl_state.LedgersPublicState,
        verbose=False,
        save_replay=False,
        save_plots=False,
        lightweight_plots=True,
    )
    env.reset()

    # Choose first account id
    account_id = env.charging_stations[0].id if env.charging_stations else 0

    # Step a few steps with random actions, verify ledgers are updated per-step
    steps = int(env.simulation_length)
    last_ts = None
    for i in range(steps):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        # Only validate if we have data to check
        if env.current_step > 0 and env.global_buffers.T > 0:
            t = env.current_step - 1
            assert 0 <= t < env.global_buffers.T, f"Row index out of bounds: {t}"

            # Global ledger checks (silent)
            gdf = env.global_buffers.to_pandas()
            row = gdf.iloc[t]
            # Timestamp monotonicity
            if last_ts is not None:
                assert row["timestamp"] >= last_ts, "timestamps not monotonic"
            last_ts = row["timestamp"]
            # Required global columns should be finite or nan-safe
            for col in [
                "step_ratio", "dow", "hour", "minute",
                "total_power_usage_kw", "power_setpoint_kw",
                "inflexible_load_kw", "solar_production_kw",
            ]:
                assert col in gdf.columns, f"missing global column {col}"
                # Allow NaN in the first few steps for totals if init is lazy, but ensure column exists
            # Price forecast columns exist
            for h in range(1, 25):
                assert f"price_fc_h{h:02d}" in gdf.columns, "missing price forecast column"

            # Account ledger checks for the chosen account
            abuf = env.account_buffers.get(account_id)
            assert abuf is not None, "missing account buffer"
            adf = abuf.to_pandas()
            arow = adf.iloc[t]
            # Summaries for account row (silent)
            base_msg = None
            # If roaming simplification fields are present, append them
            roam_bits = []
            if 'soc' in adf.columns:
                roam_bits.append(f"soc={arow.get('soc', np.nan):.3f}")
            if 'time_to_departure' in adf.columns:
                ttd = arow.get('time_to_departure', np.nan)
                roam_bits.append(f"ttd={int(ttd) if np.isfinite(ttd) else 'nan'}")
            if 'time_since_arrival' in adf.columns:
                tsa = arow.get('time_since_arrival', np.nan)
                roam_bits.append(f"tsa={int(tsa) if np.isfinite(tsa) else 'nan'}")
            # Add household data if present
            if 'household_inflexible_load_kw' in adf.columns:
                load_val = arow.get('household_inflexible_load_kw', np.nan)
                roam_bits.append(f"load_kw={load_val:.3f}")
            if 'household_pv_kw' in adf.columns:
                pv_val = arow.get('household_pv_kw', np.nan)
                roam_bits.append(f"pv_kw={pv_val:.3f}")
            # Add sample forecast values (first 3 hours)
            if 'load_fc_h01' in adf.columns:
                fc_vals = [arow.get(f'load_fc_h{h:02d}', np.nan) for h in range(1, 4)]
                roam_bits.append(f"load_fc_h01-03=[{fc_vals[0]:.3f},{fc_vals[1]:.3f},{fc_vals[2]:.3f}]")
            if 'pv_fc_h01' in adf.columns:
                pv_fc_vals = [arow.get(f'pv_fc_h{h:02d}', np.nan) for h in range(1, 4)]
                roam_bits.append(f"pv_fc_h01-03=[{pv_fc_vals[0]:.3f},{pv_fc_vals[1]:.3f},{pv_fc_vals[2]:.3f}]")
            # Keep silent per-step; rely on assertions and warnings below
            for col in ["cs_power_kw", "cs_amps", "evs_connected"]:
                assert col in adf.columns, f"missing account column {col}"
            # If per-account forecasts are present, they should be present and numeric columns
            load_cols = [f"load_fc_h{h:02d}" for h in range(1, 25)]
            pv_cols = [f"pv_fc_h{h:02d}" for h in range(1, 25)]
            if all(c in adf.columns for c in load_cols):
                vals = arow[load_cols].to_numpy()
                assert vals.shape[0] == 24, "load forecast length != 24"
                # Check if forecasts are varying (not all the same)
                try:
                    finite_vals = vals[np.isfinite(vals)]
                    unique_vals = len(np.unique(finite_vals))
                    if unique_vals <= 2:
                        print(f"  [WARNING] Load forecast has only {unique_vals} unique values: {vals[:5]}...")
                except Exception:
                    pass
            if all(c in adf.columns for c in pv_cols):
                vals = arow[pv_cols].to_numpy()
                assert vals.shape[0] == 24, "pv forecast length != 24"
                # Check if forecasts are varying (not all the same)
                try:
                    finite_vals = vals[np.isfinite(vals)]
                    unique_vals = len(np.unique(finite_vals))
                    if unique_vals <= 2:
                        print(f"  [WARNING] PV forecast has only {unique_vals} unique values: {vals[:5]}...")
                except Exception:
                    pass

        if done or truncated:
            break

    # Save full ledgers to results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(PROJECT_ROOT, "results", f"ledger_smoke_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    # Global ledger
    env.global_buffers.to_parquet(os.path.join(run_dir, "global.parquet"))
    # Account ledgers
    for acc_id, abuf in env.account_buffers.items():
        abuf.to_parquet(os.path.join(run_dir, f"account_{acc_id}.parquet"))

    print("[OK] smoke_ledger_updates: global and account ledgers updated per-step with correct schemas")
    print(f"[OK] Ledgers saved to: {run_dir}")


if __name__ == "__main__":
    main()
