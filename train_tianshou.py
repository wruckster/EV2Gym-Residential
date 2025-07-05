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
from ev2gym.rl_agent import state, reward, cost, action_wrappers, noise_wrappers
from ev2gym.rl_agent.networks import PolicyNet
from ev2gym.visuals import evaluator_plot

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
                labels=['Evaluation']
            )
            logging.info(f"Evaluation plots saved to {plot_save_path}")
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

    # --- 4. Create Vectorized Environments ---
    logging.info("Creating vectorized environments...")
    
    # Create replay directory for saving replay files
    replay_dir = os.path.join(run_dir, "replay_files")
    os.makedirs(replay_dir, exist_ok=True)
    
    def make_env(seed_offset: int = 0):
        def _init():
            env = EV2Gym(
                config_file=env_params.get('config_file'),
                state_function=state_fn,
                reward_function=reward_fn,
                cost_function=cost_fn,
                verbose=verbosity,
                save_plots=False, # Plots handled by this script
                replay_save_path=replay_dir
            )
            if action_wrapper_cls:
                env = action_wrapper_cls(env)
            if noise_wrapper_cls:
                env = noise_wrapper_cls(env)
            env.reset(seed=exp_params.get('seed', 42) + seed_offset)
            return env
        return _init

    training_num = rl_params.get('training_num', 4)
    test_num = rl_params.get('test_num', 1)
    train_envs = DummyVectorEnv([make_env(seed_offset=i) for i in range(training_num)])
    test_envs = DummyVectorEnv([make_env(seed_offset=i + training_num) for i in range(test_num)])
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
            return torch.distributions.Normal(loc=x[0], scale=x[1])
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
        logging.info(f"Finished training: {result}")

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
            if action_wrapper_cls:
                eval_env = action_wrapper_cls(eval_env)

            # Load the best policy
            best_policy_path = os.path.join(run_dir, "best_policy.pth")
            if os.path.exists(best_policy_path):
                policy.load_state_dict(torch.load(best_policy_path, map_location=device))
                logging.info(f"Loaded best policy from {best_policy_path}")
            else:
                logging.warning("No best policy found to load for evaluation.")

            # Run one full episode
            eval_collector = Collector(policy, eval_env)
            logging.info("Starting evaluation episode...")
            collect_result = eval_collector.collect(n_episode=1, render=0.0, reset_before_collect=True)
            logging.info(f"Evaluation complete: {collect_result}")
            
            # Log episode details
            unwrapped_env = eval_env.unwrapped
            logging.info(f"Episode completed at step {unwrapped_env.current_step}/{unwrapped_env.simulation_length}")
            logging.info(f"Total EVs spawned: {unwrapped_env.total_evs_spawned}")
            logging.info(f"Episode done: {unwrapped_env.done}")

            eval_env.close()  # Ensure replay file is saved

            # Generate and save plots
            # Wait a moment for the file to be written to disk
            import time
            time.sleep(1)
            
            replay_files = sorted([f for f in os.listdir(eval_replay_path) if f.endswith('.pkl')], 
                                key=lambda x: os.path.getmtime(os.path.join(eval_replay_path, x)))
            
            if not replay_files:
                # Try to find replay files in the parent directory as a fallback
                replay_files = sorted([f for f in os.listdir('.') if f.endswith('.pkl')],
                                    key=lambda x: os.path.getmtime(x))
                if replay_files:
                    latest_replay_file = replay_files[-1]  # Most recent file
                    logging.info(f"Found replay file in working directory: {latest_replay_file}")
                else:
                    logging.warning("No replay files found. Checking directory contents:")
                    logging.warning(f"Contents of {eval_replay_path}: {os.listdir(eval_replay_path) if os.path.exists(eval_replay_path) else 'Directory not found'}")
                    logging.warning(f"Current working directory contents: {os.listdir('.')}")
            else:
                latest_replay_file = os.path.join(eval_replay_path, replay_files[-1])

            if replay_files:
                if not latest_replay_file.startswith(eval_replay_path):
                    # If we found the file in the working directory, update the path
                    latest_replay_file = os.path.abspath(latest_replay_file)
                else:
                    latest_replay_file = os.path.join(eval_replay_path, replay_files[-1])
                
                logging.info(f"Generating plots from replay file: {latest_replay_file}")
                plot_save_path = os.path.join(run_dir, "evaluation_plots.png")
                
                try:
                    evaluator_plot.plot_from_replay(
                        [latest_replay_file],
                        save_path=plot_save_path,
                        labels=['PPO']
                    )
                    logging.info(f"Evaluation plots saved to {plot_save_path}")
                except Exception as e:
                    logging.error(f"Error generating plots: {str(e)}", exc_info=True)
                    # Try to load and log the replay file to help with debugging
                    try:
                        with open(latest_replay_file, 'rb') as f:
                            import pickle
                            data = pickle.load(f)
                            logging.info(f"Replay file keys: {list(data.keys()) if hasattr(data, 'keys') else 'Not a dictionary'}")
                    except Exception as load_error:
                        logging.error(f"Failed to load replay file: {str(load_error)}")
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
