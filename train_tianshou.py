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
# from ev2gym.visuals import evaluator_plot # Placeholder for future integration

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
    # --- 1. Load Configuration ---
    config = load_yaml_config(config_path)
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
    logging.info(f"-> State: {rl_params.get('state_function', 'Default')}, Reward: {rl_params.get('reward_function', 'Default')}, Action Wrapper: {env_params.get('action_wrapper', 'None')}")

    # --- 4. Create Vectorized Environments ---
    logging.info("Creating vectorized environments...")
    def make_env(seed_offset: int = 0):
        def _init():
            env = EV2Gym(
                config_file=env_params.get('config_file'),
                state_function=state_fn,
                reward_function=reward_fn,
                cost_function=cost_fn,
                verbose=False,
                save_plots=False # Plots handled by this script
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
        dist_fn = torch.distributions.Normal

    policy = PPOPolicy(
        actor=net.actor,
        critic=net.critic,
        optim=optim,
        dist_fn=dist_fn,
        action_space=env.action_space,
        **ppo_params.get('policy_args', {})
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