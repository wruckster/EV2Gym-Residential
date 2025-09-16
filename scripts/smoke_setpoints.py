#!/usr/bin/env python3
"""
Smoke test for setpoints (no CLI):
  - Runs a short residential simulation
  - Enables per-account online setpoints in-memory
  - Saves a replay and ledgers
  - Generates a plot using evaluator_plot.plot_from_replay (main plot)

Edit the constants below to change behavior.
"""
from __future__ import annotations
import os
import sys

# Ensure project root is importable when run as a script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from ev2gym.visuals.evaluator_plot import plot_from_replay
from ev2gym.models.ev2gym_env import EV2Gym


# -------------------------------
# Editable constants (no CLI)
# -------------------------------
CONFIG_PATH = os.path.join("ev2gym", "example_config_files", "residential_v2g.yaml")
STEPS = 1000  # ~1 day at 5-min resolution
OUTDIR = os.path.join("results", "smoke_setpoints")
PLOT_FILENAME = "setpoints_smoke.png"


def ensure_online_setpoints_enabled(env: EV2Gym) -> None:
    """Enable per-account online setpoints with safe defaults if accounts block exists.
    This does not modify the YAML file; it's in-memory only for the smoke.
    """
    acc = env.config.setdefault("accounts", {}) if isinstance(env.config, dict) else {}
    os_cfg = acc.setdefault("online_setpoints", {}) if isinstance(acc, dict) else {}
    os_cfg.setdefault("enabled", True)
    os_cfg.setdefault("v2g_allowed", env.config.get("v2g_enabled", True))
    os_cfg.setdefault("grid_export_allowed", True)
    os_cfg.setdefault("peak_price_threshold", 0.35)  # $/kWh
    # Also set the new unified path for compatibility with refactor
    sp = env.config.setdefault("setpoints", {}) if isinstance(env.config, dict) else {}
    acc_online = sp.setdefault("account_online", {}) if isinstance(sp, dict) else {}
    acc_online.setdefault("enabled", True)
    acc_online.setdefault("v2g_allowed", env.config.get("v2g_enabled", True))
    acc_online.setdefault("grid_export_allowed", True)
    acc_online.setdefault("peak_price_threshold", 0.35)


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)

    env = EV2Gym(
        config_file=CONFIG_PATH,
        save_replay=True,
        save_plots=False,
        replay_save_path=OUTDIR,
    )

    # Light-touch: enable account online setpoints at runtime
    try:
        ensure_online_setpoints_enabled(env)
    except Exception:
        pass

    obs, _ = env.reset()

    # Build a no-op action vector (zeros) sized to total ports
    action_dim = env.number_of_ports
    actions = np.zeros(action_dim, dtype=float)

    # Run limited steps for a quick smoke
    max_steps = max(1, min(int(STEPS), int(env.simulation_length)))

    done = False
    steps = 0
    while not done and steps < max_steps:
        obs, reward, done, truncated, info = env.step(actions)
        steps += 1

    # Save replay and ledgers
    try:
        replay_path = env._save_sim_replay()
    except Exception:
        # Fallback: write replay to outdir if method is not available
        replay_path = os.path.join(OUTDIR, "replay_smoke.pkl")
        try:
            import pickle
            with open(replay_path, "wb") as f:
                pickle.dump(env, f)
        except Exception as e:
            print(f"[smoke_setpoints] Failed to save replay: {e}")

    try:
        # Write ledgers as parquet under OUTDIR/ledgers
        ledger_dir = os.path.join(OUTDIR, "ledgers")
        ledger_paths = env.save_ledgers_parquet(ledger_dir)
        # Also dump CSV copies into OUTDIR for quick inspection
        try:
            gparq = ledger_paths.get("global") if isinstance(ledger_paths, dict) else None
            if gparq and os.path.isfile(gparq):
                gdf = pd.read_parquet(gparq)
                gcsv = os.path.join(OUTDIR, "global.csv")
                gdf.to_csv(gcsv, index=False)
                print(f"[smoke_setpoints] Wrote global ledger CSV -> {gcsv}")
            accts = ledger_paths.get("accounts") if isinstance(ledger_paths, dict) else {}
            if isinstance(accts, dict):
                for acc_id, apath in accts.items():
                    try:
                        if apath and os.path.isfile(apath):
                            adf = pd.read_parquet(apath)
                            acsv = os.path.join(OUTDIR, f"account_{acc_id}.csv")
                            adf.to_csv(acsv, index=False)
                            print(f"[smoke_setpoints] Wrote account {acc_id} CSV -> {acsv}")
                    except Exception as ie:
                        print(f"[smoke_setpoints] Failed CSV export for account {acc_id}: {ie}")
        except Exception as ie:
            print(f"[smoke_setpoints] Failed to write CSV copies of ledgers: {ie}")
    except Exception as e:
        print(f"[smoke_setpoints] Failed to save ledgers: {e}")

    # Generate plot from replay
    try:
        plot_path = os.path.join(OUTDIR, PLOT_FILENAME)
        plot_from_replay(replay_path, save_path=plot_path, plot_type="main")
        print(f"[smoke_setpoints] Plot saved to {plot_path}")
    except Exception as e:
        print(f"[smoke_setpoints] Failed to plot from replay: {e}")


if __name__ == "__main__":
    main()
