"""
Centralized experiment logger for managing training runs, metrics, and artifacts.
"""
import os
import logging
import torch
import numpy as np
from typing import Any, Dict, List, Callable

from tianshou.env import DummyVectorEnv
from tianshou.utils import TensorboardLogger

from ev2gym.utilities.metrics import MetricsTracker
from ev2gym.visuals.evaluator_plot import plot_from_replay, plot_action_analysis, plot_learning_curves, plot_ev_metrics


class ExperimentLogger:
    """Manages logging, metrics, and artifacts for an RL experiment."""

    def __init__(self, log_dir: str, run_name: str, use_tensorboard: bool = True):
        """
        Initializes the logger and sets up directories.

        Args:
            log_dir (str): The base directory for all experiment results.
            run_name (str): The unique name for this specific run.
            use_tensorboard (bool): Whether to enable TensorBoard logging.
        """
        self.run_dir = os.path.join(log_dir, run_name)
        self.replay_dir = os.path.join(self.run_dir, "replay_files")
        os.makedirs(self.replay_dir, exist_ok=True)

        self.metrics_tracker = MetricsTracker(self.run_dir, use_tensorboard)
        self.action_stats: Dict[str, List] = {'actions': [], 'means': [], 'stds': []}
        self.ev_metrics: Dict[str, Dict] = {'soc': {}, 'charging_power': {}}

        logging.info(f"Experiment results will be saved to: {self.run_dir}")

    def get_tianshou_logger(self):
        """Returns a Tianshou-compatible logger instance."""
        return TensorboardLogger(self.metrics_tracker.writer)

    def get_tianshou_test_fn(self, policy: Any) -> Callable[[int, int], None]:
        """
        Returns a callback function for Tianshou's `test_fn` argument.
        This function collects detailed metrics during evaluation phases.

        Args:
            policy: The Tianshou policy object being trained.

        Returns:
            A callback function.
        """
        def collect_metrics_callback(epoch: int, env_step: int) -> None:
            # Collect action distribution stats from the policy's last batch
            if hasattr(policy, 'last_batch') and policy.last_batch is not None:
                batch = policy.last_batch
                if hasattr(batch, 'act'):
                    self.action_stats['actions'] = batch.act.cpu().numpy()
                if hasattr(policy, 'dist'):
                    try:
                        dist = policy.dist(batch.obs)
                        if hasattr(dist, 'mean'):
                            self.action_stats['means'] = dist.mean.detach().cpu().numpy()
                        if hasattr(dist, 'stddev'):
                            self.action_stats['stds'] = dist.stddev.detach().cpu().numpy()
                    except Exception as e:
                        logging.warning(f"Could not collect action distribution stats: {e}")
            
            # Log the collected action stats
            if self.action_stats['actions']:
                self.metrics_tracker.log_action_stats(
                    actions=np.array(self.action_stats['actions']),
                    action_means=np.array(self.action_stats['means']) if self.action_stats['means'] else None,
                    action_stds=np.array(self.action_stats['stds']) if self.action_stats['stds'] else None,
                    step=epoch
                )
                self.action_stats = {k: [] for k in self.action_stats}

        return collect_metrics_callback

    def get_save_best_fn(self) -> Callable[[Any], None]:
        """Returns a callback to save the best performing policy."""
        def save_best_policy(policy):
            torch.save(policy.state_dict(), os.path.join(self.run_dir, "best_policy.pth"))
        return save_best_policy

    def post_training_analysis(self, eval_collector: Any, lightweight_plots: bool):
        """
        Performs post-training evaluation, plotting, and artifact saving.

        Args:
            eval_collector: The Tianshou collector used for final evaluation.
            lightweight_plots (bool): Flag to generate only essential plots.
        """
        logging.info("Starting post-training evaluation and plotting...")
        try:
            # Load best policy for final evaluation
            best_policy_path = os.path.join(self.run_dir, "best_policy.pth")
            if os.path.exists(best_policy_path):
                eval_collector.policy.load_state_dict(torch.load(best_policy_path))
            else:
                logging.warning("No best policy found to load for evaluation.")

            # Run final evaluation and save replay
            eval_collector.collect(n_episode=1, render=0.0)

            # Generate plots from replay files
            plot_from_replay(
                self.replay_dir,
                plot_type="city",
            )

            # Generate plots from tracked metrics
            plot_action_analysis(self.metrics_tracker.metrics, self.run_dir)
            plot_learning_curves(self.metrics_tracker.metrics, self.run_dir)
            
            if any(self.metrics_tracker.metrics.get(f'ev_traces/ev/soc/ev_cs0_port{i}', {}) for i in range(2)):
                 plot_ev_metrics(self.metrics_tracker.metrics, self.run_dir)

        except Exception as e:
            logging.error(f"Error during post-training analysis: {e}", exc_info=True)
        finally:
            self.close()

    def close(self):
        """Saves final metrics and closes any open writers."""
        logging.info("Saving final metrics and closing logger.")
        self.metrics_tracker.save_metrics()
        self.metrics_tracker.close()
