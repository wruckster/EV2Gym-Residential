# ev2gym/rl_agent/monitor_wrappers.py
from __future__ import annotations
from typing import Any, Dict, Tuple, Optional, List
import logging
import gymnasium as gym
import numpy as np
import csv
import os

class ActionMonitor(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        log_every_n_steps: int = 0,
        write_csv_path: Optional[str] = None,
        state_label_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__(env)
        self.log_every_n_steps = log_every_n_steps
        self.write_csv_path = write_csv_path
        self.state_label_names = state_label_names
        self._labels_built: bool = state_label_names is not None
        self._obs_warned: bool = False

        self.step_idx: int = 0
        self.episode_idx: int = 0

        # Time-series buffers
        self.neg_frac_series: list[float] = []
        self.act_mean_series: list[float] = []
        self.act_min_series: list[float] = []
        self.act_max_series: list[float] = []

        if self.write_csv_path:
            os.makedirs(os.path.dirname(self.write_csv_path), exist_ok=True)
            with open(self.write_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "step", "neg_frac", "act_mean", "act_min", "act_max"])

    def reset(self, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        if self.step_idx > 0:
            self.episode_idx += 1
            self.step_idx = 0
            self._obs_warned = False
        out = self.env.reset(**kwargs)
        # Normalize to Gymnasium-style (obs, info)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            # Classic Gym compatibility: only obs returned
            obs, info = out, {}
        # Build labels once if not provided, but only when env is verbose
        if not self._labels_built and getattr(self.env, "verbose", False):
            self._auto_build_labels()
        # Sanitize observations
        obs = self._sanitize_obs(obs)
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        # Handle both scalar and vector actions
        a = np.array(action, dtype=float).ravel()
        neg_frac = float((a < 0).mean()) if a.size > 0 else float(a < 0)

        self.neg_frac_series.append(neg_frac)
        self.act_mean_series.append(float(a.mean()) if a.size > 0 else float(a))
        self.act_min_series.append(float(a.min()) if a.size > 0 else float(a))
        self.act_max_series.append(float(a.max()) if a.size > 0 else float(a))

        if self.write_csv_path:
            with open(self.write_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.episode_idx, self.step_idx,
                    self.neg_frac_series[-1], self.act_mean_series[-1],
                    self.act_min_series[-1], self.act_max_series[-1]
                ])

        # Increment step index before calling env so modulus aligns with printed timestep after transition
        self.step_idx += 1
        step_out = self.env.step(action)
        # Normalize to Gymnasium-style (obs, reward, terminated, truncated, info)
        if isinstance(step_out, tuple) and len(step_out) == 5:
            obs, rew, terminated, truncated, info = step_out
        elif isinstance(step_out, tuple) and len(step_out) == 4:
            obs, rew, done, info = step_out
            terminated, truncated = bool(done), False
        else:
            raise RuntimeError("Unexpected env.step() return format: expected 4 or 5-tuple")
        # Build labels if still missing (e.g., if reset didn't run in vector env context),
        # but only when env is verbose
        if not self._labels_built and getattr(self.env, "verbose", False):
            self._auto_build_labels()
        # Sanitize observations before returning to policy
        obs = self._sanitize_obs(obs)

        # Attach per-step action stats for downstream logging/collector (optional)
        info = dict(info)
        info["neg_action_frac"] = neg_frac
        info["action_mean"] = self.act_mean_series[-1]
        info["action_min"] = self.act_min_series[-1]
        info["action_max"] = self.act_max_series[-1]

        # Periodic detailed debug print of timestep, actions, and labelled state
        if self.step_idx % 10 == 0:
            # Actions (original as passed in)
            a_str = np.array2string(np.array(action, dtype=float), precision=3, floatmode='fixed')
            # State with labels if provided
            obs_flat = np.array(obs).ravel()
            if self.state_label_names and len(self.state_label_names) == obs_flat.size:
                state_lines = [f"    {name}: {float(val):.4f}" for name, val in zip(self.state_label_names, obs_flat)]
            else:
                state_lines = [f"    s[{i}]: {float(val):.4f}" for i, val in enumerate(obs_flat)]

            print(
                "\n".join(
                    [
                        f"[ActionMonitor] ep={self.episode_idx} step={self.step_idx}",
                        f"  actions: {a_str}",
                        f"  neg_frac={neg_frac:.3f} mean={self.act_mean_series[-1]:.3f} "
                        f"min={self.act_min_series[-1]:.3f} max={self.act_max_series[-1]:.3f}",
                        "  state:",
                        *state_lines,
                    ]
                )
            )

        return obs, rew, terminated, truncated, info

    def get_action_stats(self) -> Dict[str, np.ndarray]:
        return {
            "neg_frac": np.array(self.neg_frac_series, dtype=float),
            "mean": np.array(self.act_mean_series, dtype=float),
            "min": np.array(self.act_min_series, dtype=float),
            "max": np.array(self.act_max_series, dtype=float),
        }

    # --- Internal helpers ---
    def _auto_build_labels(self) -> None:
        """Attempt to infer state labels for V2G_profit_max automatically.
        If inference fails, leave labels as None and rely on indexed printing.
        """
        try:
            labels: List[str] = []
            # 1) Step and previous power usage
            labels.append("t")
            labels.append("current_power_usage_prev")
            # 2) Price horizon: infer horizon length if possible, default to 20
            horizon = 20
            try:
                # If charge_prices is a 2D array [1, T], use min(20, remaining)
                if hasattr(self.env, "charge_prices") and hasattr(self.env, "current_step"):
                    total_remaining = int(self.env.charge_prices.shape[-1] - self.env.current_step)
                    horizon = min(20, max(0, total_remaining)) or 20
            except Exception:
                horizon = 20
            labels.extend([f"price_h+{k}" for k in range(horizon)])

            # 3) Per transformer / CS / port labels for soc and time_to_departure
            if hasattr(self.env, "transformers") and hasattr(self.env, "charging_stations"):
                for tr in getattr(self.env, "transformers", []):
                    for cs_index, cs in enumerate(getattr(self.env, "charging_stations", [])):
                        try:
                            if getattr(cs, "connected_transformer", None) == getattr(tr, "id", None):
                                n_ports = int(getattr(cs, "n_ports", 0))
                                for p in range(n_ports):
                                    labels.append(f"soc_tr{getattr(tr,'id','?')}_cs{cs_index}_port{p}")
                                    labels.append(f"time_to_departure_tr{getattr(tr,'id','?')}_cs{cs_index}_port{p}")
                        except Exception:
                            # Skip malformed CS entries
                            continue

            # Only set if non-empty; otherwise keep None
            if labels:
                self.state_label_names = labels
                self._labels_built = True
        except Exception:
            # Leave labels unset on any failure
            self._labels_built = False

    def _sanitize_obs(self, obs: Any) -> Any:
        """Replace NaN/Inf in observation with finite values and clip extremes.
        Works for numpy arrays and lists. Logs a one-time warning per episode.
        """
        try:
            arr = np.asarray(obs, dtype=float)
            if not np.isfinite(arr).all():
                if not self._obs_warned:
                    logging.warning("[ActionMonitor] Non-finite observation detected; applying nan_to_num.")
                    self._obs_warned = True
                arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
            # Optional: clip extreme magnitudes to keep policy stable
            arr = np.clip(arr, -1e6, 1e6)
            return arr
        except Exception:
            return obs