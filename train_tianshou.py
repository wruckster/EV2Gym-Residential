"""
YAML-driven Tianshou RL Training Script for EV2Gym

This script acts as a client to the EV2Gym library, using its utilities
for configuration, environment creation, and modular component selection.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer

from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.rl_agent import action_wrappers, cost, reward, state
from ev2gym.rl_agent.networks import Actor_V2G_Continuous, Critic_V2G_Continuous
from ev2gym.utilities.loaders import get_component, load_config
from ev2gym.utilities.experiment_logger import ExperimentLogger

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def main(config_path: str):
    """Main training function."""
    # --- 1. Load Configuration and Setup Run ---
    config = load_config(config_path)
    env_params = config['environment']
    rl_params = config['rl']
    log_params = config['experiment']

    run_name = f"{log_params['run_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logging.info(f"Starting run: {run_name}")

    # --- 2. Initialize Experiment Logger ---
    exp_logger = ExperimentLogger(
        log_dir=log_params['results_dir'],
        run_name=run_name,
        use_tensorboard=log_params.get('tensorboard', True)
    )
    log_dir = exp_logger.run_dir

    # --- 3. Create Environment ---
    def make_env(log_path, seed_offset=0, is_test=False):
        def _init():
            # Configure replay saving for evaluation environments
            replay_save_path = None
            save_replay = False
            
            # For evaluation environments, enable replay saving
            if is_test:
                save_replay = True
                replay_save_path = os.path.join(log_path, "replay_files")
                os.makedirs(replay_save_path, exist_ok=True)
            
            env = EV2Gym(
                config_file=env_params['config_file'],
                save_replay=save_replay,
                replay_save_path=replay_save_path
            )
            
            # Apply action wrapper if specified
            if 'action_wrapper' in env_params:
                action_wrapper_cls = get_component('action_wrappers', env_params['action_wrapper'], env=env)
                env = action_wrapper_cls  # The wrapper is already instantiated with the env
            
            # Apply noise wrapper if specified and not in test mode
            if not is_test and 'noise_wrapper' in env_params and env_params['noise_wrapper']:
                noise_wrapper_cls = get_component('noise_wrappers', env_params['noise_wrapper'])
                env = noise_wrapper_cls(env)
            
            env.reset(seed=log_params.get('seed', 42) + seed_offset)
            return env
        return _init

    # Create a single environment to get observation/action space info
    env = make_env(log_dir)()
    
    # --- 4. Initialize RL Components with Environment ---
    # Get and initialize component functions/classes that require the environment
    state_fn = get_component('state', rl_params.get('state_function'), env=env)
    reward_fn = get_component('reward', rl_params.get('reward_function'), env=env)
    cost_fn = get_component('cost', rl_params.get('cost_function'), env=env) if rl_params.get('cost_function') else None

    # --- 5. Create Vectorized Environments ---
    train_envs = SubprocVectorEnv([make_env(log_dir, i) for i in range(rl_params.get('training_num', 8))])
    test_envs = SubprocVectorEnv([make_env(log_dir, i, is_test=True) for i in range(rl_params.get('test_num', 1))])
    
    # --- 6. Setup Networks and Policy ---
    device = rl_params.get('device', 'cpu')

    actor = Actor_V2G_Continuous(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        **rl_params.get('actor_args', {})
    ).to(device)

    critic = Critic_V2G_Continuous(
        state_dim=env.observation_space.shape[0],
        **rl_params.get('critic_args', {})
    ).to(device)

    optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=rl_params.get('learning_rate', 1e-3))

    def dist_fn(x):
        mean, log_std = torch.split(x, x.shape[-1] // 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        return torch.distributions.Independent(torch.distributions.Normal(loc=mean, scale=torch.exp(log_std)), 1)

    policy = PPOPolicy(
        actor=actor, critic=critic, optim=optim, dist_fn=dist_fn,
        action_space=env.action_space, **rl_params.get('policy_args', {})
    )
    logging.info("PPO policy created successfully.")

    # --- 7. Setup Collectors ---
    train_collector = Collector(policy, train_envs, VectorReplayBuffer(rl_params['buffer_size'], len(train_envs)))
    test_collector = Collector(policy, test_envs)
    logging.info("Train and test collectors created.")

    # --- 8. Setup Trainer ---
    trainer = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        logger=exp_logger.get_tianshou_logger(),
        save_best_fn=exp_logger.get_save_best_fn(),
        test_fn=exp_logger.get_tianshou_test_fn(policy),
        **rl_params.get('trainer_args', {})
    )

    # --- 9. Run Training ---
    try:
        logging.info("Starting training...")
        result = trainer.run()
        logging.info(f"Finished training: {result}")
    except Exception as e:
        logging.error("An error occurred during training:", exc_info=True)
    finally:
        # --- 10. Post-Training Analysis and Cleanup ---
        exp_logger.post_training_analysis(
            eval_collector=test_collector,
            lightweight_plots=log_params.get('lightweight_plots', False)
        )
        train_envs.close()
        test_envs.close()
        logging.info("Cleaned up resources.")


if __name__ == "__main__":
    # The config path is now the single source of truth for configuration.
    # You can change the default to point to your desired config file.
    main(config_path='train_config.yaml')
