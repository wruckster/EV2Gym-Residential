"""
YAML-driven Tianshou RL Training Script for EV2Gym

This script acts as a client to the EV2Gym library, using its utilities
for configuration, environment creation, and modular component selection.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Callable, Optional, Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
import yaml
from tianshou.env import DummyVectorEnv
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger

# --- EV2Gym Core Imports ---
from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.rl_agent import state, reward, cost, action_wrappers, noise_wrappers, monitor_wrappers
from ev2gym.rl_agent.networks import PolicyNet
from ev2gym.visuals import evaluator_plot
from ev2gym.rl_agent.monitor_wrappers import ActionMonitor

# --- Utility Functions ---
def get_component(module: Any, component_name: str) -> Optional[Callable]:
    """Dynamically get a component (function or class) from a module."""
    if not component_name:
        return None
    try:
        return getattr(module, component_name)
    except AttributeError:
        logging.error(f"Component '{component_name}' not found in '{module.__name__}'.")
        raise

def load_yaml_config(config_path: str) -> dict:
    """Loads a YAML config file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main(config_path: str):
    """Main training function driven by a YAML config."""
    # Load configuration
    config = load_yaml_config(config_path)

    # Check if we're in evaluation-only mode
    if config.get('experiment', {}).get('evaluation', {}).get('enabled', False):
        eval_config = config['experiment']['evaluation']
        replay_dir = eval_config.get('replay_dir', 'results')
        plot_save_name = eval_config.get('plot_save_name', 'evaluation_plots.png')
        
        # Find the most recent replay file
        try:
            replay_files = []
            for root, _, files in os.walk(replay_dir):
                for file in files:
                    if file.endswith('.pkl'):
                        replay_files.append(os.path.join(root, file))
            
            if not replay_files:
                logging.error(f"No replay files found in {replay_dir}")
                return
                
            # Sort by modification time (newest first)
            replay_files.sort(key=os.path.getmtime, reverse=True)
            latest_replay = replay_files[0]
            
            logging.info(f"Running in evaluation-only mode")
            logging.info(f"Using replay file: {latest_replay}")
            
            # Generate and save plots
            plot_save_path = os.path.join(os.path.dirname(latest_replay), plot_save_name)
            evaluator_plot.plot_from_replay(
                [latest_replay],
                save_path=plot_save_path,
                labels=["Evaluation"],
                plot_type="main",
            )

            # # Prices plot
            # evaluator_plot.plot_from_replay(
            #     [latest_replay],
            #     save_path=os.path.join(os.path.dirname(plot_save_path), "evaluation_prices.png"),
            #     plot_type="prices",
            # )

            # # Solar plot
            # evaluator_plot.plot_from_replay(
            #     [latest_replay],
            #     save_path=os.path.join(os.path.dirname(plot_save_path), "evaluation_solar.png"),
            #     plot_type="solar",
            # )

            # # Details plot
            # evaluator_plot.plot_from_replay(
            #     [latest_replay],
            #     save_path=os.path.join(os.path.dirname(plot_save_path), "evaluation_details.png"),
            #     plot_type="details",
            # )

            # Replays plot
            evaluator_plot.plot_from_replay(
                [latest_replay],
                save_path=os.path.join(os.path.dirname(plot_save_path), "evaluation_replays.png"),
                labels=["Evaluation"],
                plot_type="replays",
            )

            return
            
        except Exception as e:
            logging.error(f"Error during evaluation: {str(e)}", exc_info=True)
            return

    # --- 1. Load Configuration ---
    exp_params = config.get('experiment', {})
    env_params = config.get('environment', {})
    rl_params = config.get('rl', {})
    ppo_params = rl_params.get('ppo', {})

    # --- 2. Setup Logging and Results Directory ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = exp_params.get('run_name', 'ppo_run')
    run_dir = os.path.join(exp_params.get('results_dir', 'results'), f"{run_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    log_file = os.path.join(run_dir, "training.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )
    logging.info(f"Starting run: {run_name}_{timestamp}")
    logging.info(f"Configuration loaded from: {os.path.abspath(config_path)}")
    logging.info(f"Results will be saved to: {os.path.abspath(run_dir)}")

    # --- 3. Select RL Components Dynamically ---
    logging.info("Loading RL components from configuration...")
    state_fn = get_component(state, rl_params.get('state_function'))
    reward_fn = get_component(reward, rl_params.get('reward_function'))
    cost_fn = get_component(cost, rl_params.get('cost_function'))
    action_wrapper_cls = get_component(action_wrappers, env_params.get('action_wrapper'))
    noise_wrapper_cls = get_component(noise_wrappers, env_params.get('noise_wrapper'))
    verbosity = env_params['is_verbose']
    logging.info(f"-> State: {rl_params.get('state_function', 'Default')}, Reward: {rl_params.get('reward_function', 'Default')}, Action Wrapper: {env_params.get('action_wrapper', 'None')}")

    # --- 3.5 Configure Logging Handlers (file + console) and capture warnings ---
    try:
        import warnings as _warnings
        # Ensure we have a file handler to run_dir/training.log
        root_logger = logging.getLogger()
        has_file = any(isinstance(h, logging.FileHandler) for h in root_logger.handlers)
        if not has_file:
            log_path = os.path.join(run_dir, "training.log")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            fh = logging.FileHandler(log_path)
            fh.setLevel(logging.DEBUG)
            fmt = logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            fh.setFormatter(fmt)
            root_logger.addHandler(fh)
        # Ensure a console/stream handler exists (useful if not redirected)
        has_stream = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root_logger.handlers)
        if not has_stream:
            sh = logging.StreamHandler()
            sh.setLevel(logging.INFO)
            sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            root_logger.addHandler(sh)
        root_logger.setLevel(logging.INFO)
        # Route Python warnings (including numpy RuntimeWarning) into logging
        logging.captureWarnings(True)
        _warnings.filterwarnings("default")
        logging.info(f"Logging configured. File: {os.path.join(run_dir, 'training.log')}")
    except Exception:
        pass

    # Monkey-patch PPOPolicy at class level to ensure our inspection runs even if trainer bypasses instance attributes
    try:
        if not hasattr(PPOPolicy, "_class_update_wrapped"):
            PPOPolicy._orig_update = PPOPolicy.update  # type: ignore[attr-defined]

            def _class_wrapped_update(self, *args, **kwargs):  # type: ignore[no-redef]
                batch = None
                if len(args) >= 2:
                    batch = args[1]
                elif "batch" in kwargs:
                    batch = kwargs["batch"]
                logging.info("[PPO-Batch] (class) update() called. Has batch: %s", str(batch is not None))
                if batch is not None:
                    _inspect_batch_tensors("batch", batch)
                return PPOPolicy._orig_update(self, *args, **kwargs)  # type: ignore[attr-defined]

            PPOPolicy.update = _class_wrapped_update  # type: ignore[method-assign]
            PPOPolicy._class_update_wrapped = True  # type: ignore[attr-defined]
            logging.info("[PPO-Batch] Class-level update() inspection ENABLED.")

        if not hasattr(PPOPolicy, "_class_process_wrapped"):
            PPOPolicy._orig_process = PPOPolicy.process_fn  # type: ignore[attr-defined]

            def _class_wrapped_process(self, batch, buffer, indices):  # type: ignore[no-redef]
                try:
                    logging.info("[PPO-Batch] (class) process_fn() inspecting batch...")
                    _inspect_batch_tensors("batch", batch)
                except Exception as e:
                    logging.warning(f"[PPO-Batch] (class) process_fn inspect failed: {e}")
                return PPOPolicy._orig_process(self, batch, buffer, indices)  # type: ignore[attr-defined]

            PPOPolicy.process_fn = _class_wrapped_process  # type: ignore[method-assign]
            PPOPolicy._class_process_wrapped = True  # type: ignore[attr-defined]
            logging.info("[PPO-Batch] Class-level process_fn() inspection ENABLED.")
    except Exception:
        logging.warning("[PPO-Batch] Failed to enable class-level monkey patches.")

    # --- 4. Create Vectorized Environments ---
    logging.info("Creating vectorized environments...")
    
    # Create replay directory for saving replay files
    replay_dir = os.path.join(run_dir, "replay_files")
    os.makedirs(replay_dir, exist_ok=True)
    
    def make_env(seed_offset: int = 0, role: str = "train"):
        def _init():
            env = EV2Gym(
                config_file=env_params.get('config_file'),
                state_function=state_fn,
                reward_function=reward_fn,
                cost_function=cost_fn,
                verbose=verbosity,
                save_plots=False,
                replay_save_path=replay_dir,
            )
            # Ensure debug_setpoints follows the environment YAML flag, not global verbosity
            try:
                dbg = bool(getattr(env, "debug_setpoints", False))
                setattr(env, "debug_setpoints", dbg)
            except Exception:
                pass
            if action_wrapper_cls:
                env = action_wrapper_cls(env)
            if noise_wrapper_cls:
                env = noise_wrapper_cls(env)

            # Ensure wrappers and the base env both see the same debug_setpoints from YAML
            try:
                dbg = bool(getattr(env, "debug_setpoints", False))
                setattr(env, "debug_setpoints", dbg)
                if hasattr(env, "env"):
                    setattr(env.env, "debug_setpoints", dbg)
            except Exception:
                pass

            # Attach ActionMonitor
            if role == "train":
                # Don’t write CSV for every worker; keep it lightweight
                env = ActionMonitor(env, log_every_n_steps=0, write_csv_path=None)
            else:
                # Write a single CSV for eval to the run directory
                eval_csv = os.path.join(run_dir, "eval_actions.csv")
                env = ActionMonitor(env, log_every_n_steps=50, write_csv_path=eval_csv)

            env.reset(seed=exp_params.get('seed', 42) + seed_offset)
            return env
        return _init

    training_num = rl_params.get('training_num', 4)
    test_num = rl_params.get('test_num', 1)
    train_envs = DummyVectorEnv([make_env(seed_offset=i, role="train") for i in range(training_num)])
    test_envs = DummyVectorEnv([make_env(seed_offset=i + training_num, role="test") for i in range(test_num)])
    logging.info(f"-> {training_num} training environments and {test_num} testing environments created.")

    # --- 5. Instantiate Policy and Network ---
    env = make_env()()
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    device = exp_params.get('device', 'auto')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.info(f"Device: {device}, State Shape: {state_shape}, Action Shape: {action_shape}")

    net = PolicyNet(state_shape, action_shape, device=device).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=ppo_params.get('lr', 3e-4))

    dist_fn = None
    if isinstance(env.action_space, gym.spaces.Box):
        def dist_fn_wrapper(x):
            """Return an Independent Normal so that log_prob is scalar per sample."""
            mean, std = x  # unpack first
            # Ensure leading batch dimension exists when batch size == 1
            if mean.dim() == 1:
                mean = mean.unsqueeze(0)
                std = std.unsqueeze(0)
            
            # Replace NaNs/Infs to keep distribution valid
            if torch.isnan(mean).any() or torch.isinf(mean).any():
                logging.warning("NaNs or Infs detected in action mean – replacing with zeros.")
                mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
            
            if torch.isnan(std).any() or torch.isinf(std).any() or (std <= 0).any():
                logging.warning("Invalid std detected (NaN/Inf/<=0) – applying fallback clamp.")
                std = torch.nan_to_num(std, nan=1.0, posinf=7.0, neginf=0.05)
                std = torch.clamp(std, min=0.05, max=7.0)
            normal = torch.distributions.Normal(mean, std)
            return torch.distributions.Independent(normal, 1)
        
        dist_fn = dist_fn_wrapper

    # Get policy args from config
    policy_args = ppo_params.get('policy_args', {}).copy()

    policy = PPOPolicy(
        actor=net.actor,
        critic=net.critic,
        optim=optim,
        dist_fn=dist_fn,
        action_space=env.action_space,
        # All PPO parameters come from config
        **policy_args
    )
    logging.info("PPO policy created successfully.")

    # --- 5.1 Instrument PPO update with pre-batch finiteness checks ---
    def _inspect_batch_tensors(name_prefix: str, obj: Any) -> None:
        """Recursively inspect a tianshou Batch/dict/list/tuple for non-finite tensors."""
        # try:
        #     import torch
        #     import numpy as np
        # except Exception:
        #     return

        def _log_stats(label: str, arr) -> None:
            try:
                if isinstance(arr, np.ndarray):
                    a = arr
                elif isinstance(arr, torch.Tensor):
                    a = arr.detach().cpu().numpy()
                else:
                    return
                if a.size == 0:
                    return
                finite = np.isfinite(a)
                if not finite.all():
                    n_total = a.size
                    n_bad = int((~finite).sum())
                    max_abs = float(np.nanmax(np.abs(a))) if n_bad < n_total else float('nan')
                    logging.warning(f"[PPO-Batch] Non-finite values in {label}: bad={n_bad}/{n_total}, max_abs={max_abs:.3e}")
            except Exception:
                pass

        if isinstance(obj, dict):
            for k, v in obj.items():
                _inspect_batch_tensors(f"{name_prefix}.{k}" if name_prefix else str(k), v)
        else:
            # Try tianshou Batch API
            try:
                keys = list(obj.keys())  # type: ignore[attr-defined]
                for k in keys:
                    v = obj[k]
                    _inspect_batch_tensors(f"{name_prefix}.{k}" if name_prefix else str(k), v)
                return
            except Exception:
                pass

            if isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    _inspect_batch_tensors(f"{name_prefix}[{i}]", v)
            else:
                _log_stats(name_prefix or "batch", obj)

    if not hasattr(policy, "_update_wrapped"):
        _orig_update = policy.update

        def _wrapped_update(*args, **kwargs):
            # Tianshou calls update(self, sample_size, batch)
            batch = None
            if len(args) >= 2:
                batch = args[1]
            elif "batch" in kwargs:
                batch = kwargs["batch"]
            logging.info("[PPO-Batch] update() called. Has batch: %s", str(batch is not None))
            if batch is not None:
                logging.info("[PPO-Batch] Inspecting batch before update...")
                _inspect_batch_tensors("batch", batch)
            return _orig_update(*args, **kwargs)

        policy.update = _wrapped_update  # type: ignore[assignment]
        policy._update_wrapped = True  # type: ignore[attr-defined]
        logging.info("[PPO-Batch] Pre-update batch inspection ENABLED.")

    # Additionally, wrap process_fn to reliably access the training batch
    if not hasattr(policy, "_process_wrapped") and hasattr(policy, "process_fn"):
        _orig_process = policy.process_fn

        def _wrapped_process(batch, buffer, indices):  # type: ignore[override]
            try:
                logging.info("[PPO-Batch] process_fn() inspecting batch...")
                _inspect_batch_tensors("batch", batch)
            except Exception as e:
                logging.warning(f"[PPO-Batch] process_fn inspect failed: {e}")

            processed = _orig_process(batch, buffer, indices)

            # Sanitize advantages/returns to avoid propagating non-finite values into update
            try:
                # 1) Robust advantage normalization using only finite entries
                if hasattr(processed, "adv") and isinstance(processed.adv, torch.Tensor):
                    adv = processed.adv
                    finite_mask = torch.isfinite(adv)
                    n_total = adv.numel()
                    n_finite = int(finite_mask.sum().item())
                    if n_finite > 0:
                        mean = adv[finite_mask].mean()
                        std = adv[finite_mask].std(unbiased=False)
                        eps = torch.tensor(1e-8, dtype=adv.dtype, device=adv.device)
                        norm = (adv - mean) / torch.clamp(std, min=eps)
                        # Replace non-finite results of normalization with 0
                        norm = torch.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)
                        adv = torch.clamp(norm, -1e6, 1e6)
                        processed.adv = adv
                        if n_finite < n_total:
                            logging.warning(
                                f"[PPO-Batch] adv normalized with finite subset: {n_finite}/{n_total} finite"
                            )
                    else:
                        logging.warning("[PPO-Batch] adv has 0 finite entries; zeroing out adv")
                        processed.adv = torch.zeros_like(adv)

                # 2) Ensure returns are finite and clamped
                if hasattr(processed, "returns") and isinstance(processed.returns, torch.Tensor):
                    ret = processed.returns
                    if torch.logical_not(torch.isfinite(ret)).any():
                        n_bad = int((~torch.isfinite(ret)).sum().item())
                        logging.warning(
                            f"[PPO-Batch] Sanitizing non-finite returns in processed batch: {n_bad} bad values"
                        )
                        ret = torch.nan_to_num(ret, nan=0.0, posinf=1e6, neginf=-1e6)
                    processed.returns = torch.clamp(ret, -1e6, 1e6)
            except Exception as e:
                logging.warning(f"[PPO-Batch] Failed to sanitize adv/returns: {e}")

            return processed

        policy.process_fn = _wrapped_process  # type: ignore[assignment]
        policy._process_wrapped = True  # type: ignore[attr-defined]
        logging.info("[PPO-Batch] process_fn batch inspection ENABLED.")

    # --- 6. Setup Collectors ---
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(rl_params.get('buffer_size', 20000), len(train_envs)),
    )
    test_collector = Collector(policy, test_envs)
    logging.info("Train and test collectors created.")

    # --- 7. Setup Logger and Trainer ---
    writer = SummaryWriter(os.path.join(run_dir, "tensorboard"))
    ts_logger = TensorboardLogger(writer)
    logging.info(f"TensorBoard logs will be saved to: {os.path.join(run_dir, 'tensorboard')}")

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(run_dir, "best_policy.pth"))

    trainer = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        logger=ts_logger,
        save_best_fn=save_best_fn,
        **rl_params.get('trainer_args', {})
    )

    # --- 8. Run Training ---
    try:
        logging.info("Starting training...")
        result = trainer.run()
        logging.info(f"\n\n!!!CELEBRATE!!!!\n\nFinished training: {result}\n\n")

    except KeyboardInterrupt:
        logging.warning("Training interrupted by user.")
    except Exception as e:
        logging.error("An error occurred during training:", exc_info=True)
    finally:
        # --- 9. Save Final Policy and Clean Up ---
        final_policy_path = os.path.join(run_dir, "final_policy.pth")
        torch.save(policy.state_dict(), final_policy_path)
        logging.info(f"Final policy saved to {final_policy_path}")

        # --- 10. Post-Training Evaluation and Plotting ---
        logging.info("Starting post-training evaluation and plotting...")
        try:
            # Create a dedicated environment for evaluation with replay saving enabled
            eval_replay_path = os.path.join(run_dir, "replay_files")
            os.makedirs(eval_replay_path, exist_ok=True)

            eval_env = EV2Gym(
                config_file=env_params.get('config_file'),
                state_function=state_fn,
                reward_function=reward_fn,
                cost_function=cost_fn,
                save_replay=True,
                replay_save_path=eval_replay_path,
                verbose=False,
                save_plots=False,
                lightweight_plots=False  # Ensure detailed statistics are collected
            )
            # Propagate env YAML debug_setpoints for evaluation as well
            try:
                dbg = bool(getattr(eval_env, "debug_setpoints", False))
                setattr(eval_env, "debug_setpoints", dbg)
            except Exception:
                pass
            if action_wrapper_cls:
                eval_env = action_wrapper_cls(eval_env)

            # Ensure wrappers and base env both see the same YAML-driven debug_setpoints in eval
            try:
                dbg = bool(getattr(eval_env, "debug_setpoints", False))
                setattr(eval_env, "debug_setpoints", dbg)
                if hasattr(eval_env, "env"):
                    setattr(eval_env.env, "debug_setpoints", dbg)
            except Exception:
                pass

            # Load the best policy
            best_policy_path = os.path.join(run_dir, "best_policy.pth")
            if os.path.exists(best_policy_path):
                policy.load_state_dict(torch.load(best_policy_path, map_location=device))
                logging.info(f"Loaded best policy from {best_policy_path}")
            else:
                logging.warning("No best policy found to load for evaluation.")

            # Run one full episode manually to control when ledgers are saved
            logging.info("Starting manual evaluation episode...")
            obs, info = eval_env.reset()
            done, truncated = False, False
            total_reward = 0
            while not done and not truncated:
                # Note: Tianshou's Batch object is needed for the policy to process the observation
                from tianshou.data import Batch
                batch = Batch(obs=np.array([obs]), info=info)
                action = policy(batch).act.cpu().numpy().flatten()
                obs, step_reward, done, truncated, info = eval_env.step(action)
                total_reward += step_reward
            logging.info(f"Manual evaluation complete. Total reward: {total_reward}")

            # Parquet ledger export is decommissioned in this script to keep replay-only outputs.

            # Log episode details from the evaluation environment
            try:
                base_env = getattr(eval_env, 'env', eval_env)
                current_step = getattr(base_env, 'current_step', None)
                sim_length = getattr(base_env, 'simulation_length', None)
                evs_spawned = getattr(base_env, 'total_evs_spawned', None)
                is_done = getattr(base_env, 'done', None)
                logging.info(f"Episode completed at step {current_step}/{sim_length}")
                logging.info(f"Total EVs spawned: {evs_spawned}")
                logging.info(f"Episode done: {is_done}")
            except Exception:
                logging.warning("Could not log episode details from evaluation environment.")

            # Ensure a replay file exists; explicitly save if none was created
            try:
                # Count replays before forcing save
                pre_files = [f for f in os.listdir(eval_replay_path) if f.endswith('.pkl')]
                if not pre_files:
                    logging.info("No replay files detected after evaluation; attempting explicit save...")
                    base_env = getattr(eval_env, 'env', eval_env)
                    if hasattr(base_env, '_save_sim_replay'):
                        base_env._save_sim_replay()
                        logging.info("Explicit replay save invoked.")
                    else:
                        logging.warning("Base evaluation env has no _save_sim_replay; skipping explicit save.")
            except Exception:
                logging.warning("Explicit replay save attempt failed.")

            # Close evaluation env
            try:
                eval_env.close()
            except Exception:
                pass

            # Generate plots from the replay files
            if os.path.exists(eval_replay_path):
                replay_files = sorted([f for f in os.listdir(eval_replay_path) if f.endswith('.pkl')], 
                                    key=lambda x: os.path.getmtime(os.path.join(eval_replay_path, x)))
            else:
                replay_files = []

            if replay_files:
                print(f"[Main] Found {len(replay_files)} replay files. Generating plots...")
                # Main plot
                evaluator_plot.plot_from_replay(
                    replay_files=[os.path.join(eval_replay_path, replay_files[-1])],
                    save_path=os.path.join(run_dir, "evaluation_plots.png"),
                    labels=["Evaluation"],
                    plot_type="main",
                )


                evaluator_plot.plot_from_replay(
                    replay_files=[os.path.join(eval_replay_path, replay_files[-1])],
                    save_path=os.path.join(run_dir, "evaluation_replays.png"),
                    labels=["Evaluation"],
                    plot_type="replays",
                )
            else:
                logging.warning(f"No replay files found in {eval_replay_path} after evaluation.")

        except Exception as e:
            logging.error("An error occurred during evaluation and plotting:", exc_info=True)

        train_envs.close()
        test_envs.close()
        writer.close()
        logging.info("Cleaned up resources.")

if __name__ == "__main__":
    # Default config path, can be overridden by command line argument
    config_file = "train_config.yaml"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    
    if not os.path.exists(config_file):
        print(f"Error: Configuration file '{config_file}' not found.")
        sys.exit(1)
        
    main(config_file)
