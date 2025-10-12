#!/usr/bin/env python3
"""
Run a single-episode baseline using a heuristic policy and generate the same
replay plots as the PPO evaluation pipeline.

This module is intended to be imported and called from Python, not used via CLI.

API:
  run_heuristic_baseline(
      config: dict | str,
      heuristic: str = "LedgerAwareSetpointFollower",
      seed: int = 42,
  ) -> dict

Returns a dict with keys: {"run_dir", "replay_dir", "total_reward", "steps"}
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from typing import Any, Optional, Callable

import numpy as np

from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.visuals import evaluator_plot
from ev2gym.mpc import heuristics as mpc_heuristics

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


def load_yaml_config(config_path: str) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load configuration files.")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_run_dirs(exp_cfg: dict) -> tuple[str, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = exp_cfg.get("run_name", "heuristic_baseline")
    results_dir = exp_cfg.get("results_dir", "results")
    run_dir = os.path.join(results_dir, f"{run_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    replay_dir = os.path.join(run_dir, "replay_files")
    os.makedirs(replay_dir, exist_ok=True)
    return run_dir, replay_dir


def instantiate_heuristic(name: str, env: EV2Gym):
    """Create a heuristic instance by class name from ev2gym.mpc.heuristics.
    Tries ctor(env=env) then falls back to default ctor().
    """
    try:
        cls = getattr(mpc_heuristics, name)
    except AttributeError as e:  # pragma: no cover
        raise ValueError(f"Heuristic '{name}' not found in ev2gym.mpc.heuristics") from e
    try:
        return cls(env=env)  # type: ignore[arg-type]
    except TypeError:
        return cls()  # type: ignore[call-arg]


def run_episode(env: EV2Gym, heuristic_name: str, seed: int) -> tuple[float, int]:
    # For LedgerAwareSetpointFollower, use custom aggressive V2G settings
    if heuristic_name == "LedgerAwareSetpointFollower":
        # Use a negative buffer to allow discharging below the reserve for aggressive V2G
        # -5.0 means we can discharge up to 5 kWh below the reserve level
        heuristic = mpc_heuristics.LedgerAwareSetpointFollower(
            use_account_online=True, 
            verbose=True,
            aggressive_v2g_buffer=-5.0  # Allow discharging 5 kWh below reserve
        )
    else:
        # For other heuristics, use the standard instantiation
        heuristic = instantiate_heuristic(heuristic_name, env)
        
    obs, info = env.reset(seed=seed)
    done, truncated = False, False
    total_reward = 0.0
    steps = 0
    while not done and not truncated:
        action = heuristic.get_action(env)
        obs, r, done, truncated, info = env.step(action)
        try:
            total_reward += float(r)
        except Exception:
            pass
        steps += 1
    return total_reward, steps


def generate_plots_from_latest_replay(replay_dir: str, run_dir: str, label: str) -> None:
    replays = []
    for root, _, files in os.walk(replay_dir):
        for f in files:
            if f.endswith(".pkl"):
                replays.append(os.path.join(root, f))
    if not replays:
        logging.warning("No replay files found for plotting.")
        return
    replays.sort(key=os.path.getmtime)
    latest = replays[-1]
    # Main plot
    evaluator_plot.plot_from_replay(
        [latest],
        save_path=os.path.join(run_dir, "evaluation_plots.png"),
        labels=[label],
        plot_type="main",
    )
    # Replays plot
    evaluator_plot.plot_from_replay(
        [latest],
        save_path=os.path.join(run_dir, "evaluation_replays.png"),
        labels=[label],
        plot_type="replays",
    )


def run_heuristic_baseline(
    config: dict | str,
    heuristic: str = "LedgerAwareSetpointFollower",
    seed: int = 42,
) -> dict:
    """Run a single-episode heuristic and generate plots.

    Parameters
    ----------
    config: dict | str
        Either a loaded YAML dict or a path to a YAML config file.
    heuristic: str
        Class name in `ev2gym.mpc.heuristics`.
    seed: int
        Random seed for env.reset().

    Returns
    -------
    dict
        {"run_dir", "replay_dir", "total_reward", "steps"}
    """
    # Load config if a path is provided
    cfg = load_yaml_config(config) if isinstance(config, str) else config

    # Accept two shapes:
    # 1) Training YAML with keys: experiment, environment (environment.config_file)
    # 2) Direct EV2Gym env config: {"config_file": "/path/to/env.yaml"} or a string path to env.yaml
    exp_cfg = {}
    env_cfg = {}
    rl_cfg = {}
    env_config_path: Optional[str] = None

    if isinstance(cfg, dict) and ("experiment" in cfg or "environment" in cfg):
        exp_cfg = cfg.get("experiment", {})
        env_cfg = cfg.get("environment", {})
        rl_cfg = cfg.get("rl", {})
        env_config_path = env_cfg.get("config_file")
    elif isinstance(cfg, dict) and ("config_file" in cfg):
        env_config_path = cfg.get("config_file")
    elif isinstance(config, str):
        # If the provided path looks like an EV2Gym env YAML (no 'environment' section), use directly
        env_config_path = config
    else:
        raise ValueError(
            "Unsupported config format. Provide either the training YAML (with 'environment.config_file') "
            "or a direct dict/path containing 'config_file'."
        )

    run_dir, replay_dir = build_run_dirs(exp_cfg)

    # Configure logging to file and console in the run directory
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(run_dir, "baseline.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info("Starting heuristic baseline run")
    logging.info(f"Heuristic: {heuristic}")
    if env_config_path:
        logging.info(f"Config file: {os.path.abspath(env_config_path)}")

    # Load reward and state functions if specified in RL config
    from ev2gym.rl_agent import reward as reward_module, state as state_module
    
    reward_fn = None
    state_fn = None
    
    if rl_cfg:
        try:
            reward_fn_name = rl_cfg.get('reward_function')
            state_fn_name = rl_cfg.get('state_function')
            if reward_fn_name:
                reward_fn = getattr(reward_module, reward_fn_name, None)
                if reward_fn:
                    logging.info(f"Using reward function: {reward_fn_name}")
                else:
                    logging.warning(f"Reward function '{reward_fn_name}' not found")
            if state_fn_name:
                state_fn = getattr(state_module, state_fn_name, None)
                if state_fn:
                    logging.info(f"Using state function: {state_fn_name}")
                else:
                    logging.warning(f"State function '{state_fn_name}' not found")
        except Exception as e:
            logging.warning(f"Could not load RL functions from config: {e}")
    
    # Provide default state function if none specified
    if state_fn is None:
        state_fn = state_module.V2G_profit_max_enhanced
        logging.info("Using default state function: V2G_profit_max_enhanced")
    
    # Provide default reward function if none specified
    if reward_fn is None:
        reward_fn = reward_module.solar_profit_reward_balanced
        logging.info("Using default reward function: solar_profit_reward_balanced")

    # Create environment with replay saving enabled
    env = EV2Gym(
        config_file=env_config_path,
        reward_function=reward_fn,
        state_function=state_fn,
        save_replay=True,
        replay_save_path=replay_dir,
        verbose=True,  # Enable verbose mode to see debug output
        save_plots=False,
        lightweight_plots=False,
    )

    try:
        total_reward, steps = run_episode(env, heuristic, exp_cfg.get("seed", seed))
        logging.info(
            f"Heuristic episode finished in {steps} steps. Total reward (if defined): {total_reward}"
        )
        # Ensure a replay file exists; force save if none created
        pre_files = [f for f in os.listdir(replay_dir) if f.endswith('.pkl')]
        if not pre_files and hasattr(env, "_save_sim_replay"):
            env._save_sim_replay()  # type: ignore[attr-defined]
        generate_plots_from_latest_replay(replay_dir, run_dir, heuristic)

        # Export ledgers (global + per-account) to Parquet into the run directory
        try:
            ledgers_dir = os.path.join(run_dir, "ledgers")
            os.makedirs(ledgers_dir, exist_ok=True)
            paths = env.save_ledgers_parquet(ledgers_dir)  # type: ignore[attr-defined]
            # Summarize what we wrote
            global_path = paths.get("global") if isinstance(paths, dict) else None
            n_accounts = len(paths.get("accounts", {})) if isinstance(paths, dict) else 0
            logging.info(f"Saved ledgers parquet to: {ledgers_dir} (global: {global_path}, accounts: {n_accounts})")
            print(f"[baseline] ledgers saved to: {ledgers_dir}")
        except Exception as e:
            logging.warning(f"Could not export ledgers to parquet: {e}")

        # Print run_dir for discoverability in non-CLI usage
        print(f"[baseline] results saved to: {run_dir}")
        return {
            "run_dir": run_dir,
            "replay_dir": replay_dir,
            "total_reward": total_reward,
            "steps": steps,
        }
    finally:
        try:
            env.close()
        except Exception:
            pass

if __name__ == "__main__":
    config_file = os.path.join(os.path.dirname(__file__), "..", "ev2gym", "example_config_files", "residential_v2g.yaml")
    print(config_file)
    run_heuristic_baseline(config_file)