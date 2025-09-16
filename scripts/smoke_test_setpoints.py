#!/usr/bin/env python3
"""
Smoke test for forecast-aware residential setpoints.

- Instantiates EV2Gym with a given config
- Optionally forces first EV to arrive at t=1 with sufficient stay and energy
- Regenerates setpoints and prints a concise summary

This script does not modify library behavior. It is safe to run locally.
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from ev2gym.mpc.residential_mpc import RuleBasedController

from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.utilities.utils import generate_power_setpoints


def force_first_ev_arrival(env: EV2Gym) -> None:
    """Mutate the first EV profile to ensure allocations can occur.

    - arrival at t=1
    - departure >= arrival + 20 (or up to simulation end)
    - ensure required energy > 0 (at least 5 kWh)
    """
    if not getattr(env, 'EVs_profiles', None):
        return
    ev = env.EVs_profiles[0]
    if ev.time_of_arrival == 0:
        ev.time_of_arrival = 1
    if ev.time_of_departure <= ev.time_of_arrival + 20:
        ev.time_of_departure = min(env.simulation_length, ev.time_of_arrival + 20)
    if ev.battery_capacity_at_arrival >= ev.battery_capacity:
        # ensure at least ~5 kWh needed
        ev.battery_capacity_at_arrival = max(0.0, ev.battery_capacity - 5.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test setpoints")
    parser.add_argument(
        "--config",
        default="ev2gym/example_config_files/residential_v2g.yaml",
        help="Path to EV2Gym YAML config",
    )
    parser.add_argument(
        "--force-arrival",
        action="store_true",
        help="Force first EV arrival at t=1 with sufficient stay/energy",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot raw vs filtered setpoints",
    )
    args = parser.parse_args()

    # Output directory under results/
    outdir = os.path.join("results", "smoke_test_setpoints")
    os.makedirs(outdir, exist_ok=True)

    # Ensure replays and any artifacts are written to results/
    env = EV2Gym(
        config_file=args.config,
        verbose=False,
        save_replay=True,
        save_plots=False,
        replay_save_path=outdir,
    )

    print(f"sim_length={env.simulation_length} timescale={env.timescale} cs={env.cs}")
    pf = getattr(env, 'price_forecast', None)
    if pf is not None:
        pfa = np.asarray(pf)
        print(f"price_forecast_shape={pfa.shape}")

    print(f"profiles={len(getattr(env, 'EVs_profiles', []) or [])}")

    if args.force_arrival:
        force_first_ev_arrival(env)

    # Baseline setpoints (filters default-off per YAML)
    ps_base = generate_power_setpoints(env)
    nz = int(np.count_nonzero(ps_base))
    first = int(np.argmax(ps_base != 0)) if nz > 0 else -1
    last = int(len(ps_base) - 1 - np.argmax(ps_base[::-1] != 0)) if nz > 0 else -1
    print(f"base nonzero={nz} span=[{first},{last}] preview20={np.round(ps_base[:20], 3)}")

    # Enable filters via env.config and regenerate to validate effects
    cfg = getattr(env, 'config', {})
    if isinstance(cfg, dict):
        cfg.setdefault('setpoint_filters', {})
        spf = cfg['setpoint_filters']
        spf['smoothing'] = {'enabled': True, 'ema_alpha': 0.3}
        spf['ramp_limit'] = {'enabled': True, 'max_ramp_kw_per_step': 5.0}
        cfg['pv_preference'] = {'enabled': True, 'weight': 1.0}
        env.config = cfg

    ps_filt = generate_power_setpoints(env)
    nz_f = int(np.count_nonzero(ps_filt))
    first_f = int(np.argmax(ps_filt != 0)) if nz_f > 0 else -1
    last_f = int(len(ps_filt) - 1 - np.argmax(ps_filt[::-1] != 0)) if nz_f > 0 else -1
    print(f"filt nonzero={nz_f} span=[{first_f},{last_f}] preview20={np.round(ps_filt[:20], 3)}")

    # Check ramp deltas
    diffs = np.diff(ps_filt)
    if diffs.size > 0:
        max_up = float(np.max(diffs))
        max_dn = float(np.min(diffs))
        print(f"ramp_check max_up={max_up:.3f} max_dn={max_dn:.3f}")

        # Assert ramp limit respected
        limit = float(cfg['setpoint_filters']['ramp_limit'].get('max_ramp_kw_per_step', 5.0))
        tol = 1e-6
        assert max_up <= limit + tol, f"Ramp up exceeded: {max_up} > {limit}"
        assert abs(max_dn) <= limit + tol, f"Ramp down exceeded: {max_dn} < -{limit}"

        # Assert smoothing reduced ramp volatility vs baseline
        diffs_base = np.diff(ps_base)
        if diffs_base.size > 0:
            sd_base = float(np.std(diffs_base))
            sd_filt = float(np.std(diffs))
            print(f"ramp_volatility sd_base={sd_base:.3f} sd_filt={sd_filt:.3f}")
            assert sd_filt <= sd_base + 1e-9, "EMA smoothing did not reduce ramp volatility"

    # Optional plot
    if args.plot:
        plt.figure(figsize=(10, 4))
        plt.plot(ps_base, label='raw')
        plt.plot(ps_filt, label='filtered')
        plt.title('Aggregate setpoints: raw vs filtered')
        plt.xlabel('timestep')
        plt.ylabel('kW')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # PV-preference behavior check using RuleBasedController on synthetic inputs
    # We expect: with PV preference enabled, charging power under PV surplus >= without preference.
    rb_base = RuleBasedController()
    rb_pref = RuleBasedController()
    rb_base.set_config({'pv_preference': {'enabled': False, 'weight': 0.0}})
    rb_pref.set_config({'pv_preference': {'enabled': True, 'weight': 1.0}})

    # Shared parameters
    dt_hours = env.timescale / 60.0
    battery_capacity_kWh = 10.0
    max_charge_rate_kW = 3.6
    max_discharge_rate_kW = 3.6
    min_soc_kWh = 1.0
    max_soc_kWh = 9.0
    # Synthetic PV surplus scenarios
    scenarios = [
        {'load': 0.5, 'pv': 2.0, 'soc': 5.0},
        {'load': 0.2, 'pv': 1.5, 'soc': 8.0},
        {'load': 0.0, 'pv': 3.0, 'soc': 2.0},
    ]
    for sc in scenarios:
        a0 = rb_base.compute_battery_action(
            current_time=None,
            load_power_kW=sc['load'],
            pv_power_kW=sc['pv'],
            current_soc_kWh=sc['soc'],
            battery_capacity_kWh=battery_capacity_kWh,
            max_charge_rate_kW=max_charge_rate_kW,
            max_discharge_rate_kW=max_discharge_rate_kW,
            min_soc_kWh=min_soc_kWh,
            max_soc_kWh=max_soc_kWh,
            import_price_per_kWh=0.3,
            export_price_per_kWh=0.08,
            tou_prices_per_kWh=None,
            dt_hours=dt_hours,
        )
        a1 = rb_pref.compute_battery_action(
            current_time=None,
            load_power_kW=sc['load'],
            pv_power_kW=sc['pv'],
            current_soc_kWh=sc['soc'],
            battery_capacity_kWh=battery_capacity_kWh,
            max_charge_rate_kW=max_charge_rate_kW,
            max_discharge_rate_kW=max_discharge_rate_kW,
            min_soc_kWh=min_soc_kWh,
            max_soc_kWh=max_soc_kWh,
            import_price_per_kWh=0.3,
            export_price_per_kWh=0.08,
            tou_prices_per_kWh=None,
            dt_hours=dt_hours,
        )
        # Charging is positive (kW). With PV preference, charge should be >= baseline under surplus.
        assert a1 >= a0 - 1e-9, f"PV preference not increasing charge: pref={a1}, base={a0}, scenario={sc}"

    # quick step to confirm env still runs
    state, _ = env.reset()
    for _ in range(min(5, env.simulation_length - 1)):
        a = np.zeros(env.number_of_ports)
        env.step(a)
    print("step_ok")

    # Persist a replay and ledgers to results/
    try:
        replay_path = env._save_sim_replay()
        print(f"[smoke_test_setpoints] Replay saved to {replay_path}")
    except Exception as e:
        print(f"[smoke_test_setpoints] Failed to save replay: {e}")
    try:
        ledger_dir = os.path.join(outdir, "ledgers")
        os.makedirs(ledger_dir, exist_ok=True)
        env.save_ledgers_parquet(ledger_dir)
        print(f"[smoke_test_setpoints] Ledgers saved under {ledger_dir}")
    except Exception as e:
        print(f"[smoke_test_setpoints] Failed to save ledgers: {e}")


if __name__ == "__main__":
    main()
