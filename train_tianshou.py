"""
Tianshou RL Training Script for EV2Gym
"""
import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.data import Collector, VectorReplayBuffer

from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.rl_agent.action_wrappers import Rescale_RepairLayer
from ev2gym.rl_agent.state import V2G_profit_max
from ev2gym.rl_agent.reward import profit_maximization

def make_env(config_file, seed=None):
    """Create a single environment instance with the given config."""
    def _init():
        env = EV2Gym(
            config_file=config_file,
            verbose=False,
            save_plots=False,
            load_from_replay_path=None,
            save_replay=False
        )
        # Apply action wrapper
        env = Rescale_RepairLayer(env)
        
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init

class PolicyNet(nn.Module):
    """Neural network for the policy."""
    def __init__(self, state_shape, action_shape, device='cpu'):
        super().__init__()
        self.device = device
        self.net = Net(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_sizes=[128, 128],
            device=device,
        )
        
    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        return self.net(obs), state

def train():
    # Configuration
    config_file = "ev2gym/example_config_files/residential_v2g.yaml"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 42
    num_envs = 4  # Number of parallel environments
    buffer_size = 20000
    lr = 3e-4
    gamma = 0.99
    epoch = 100
    step_per_epoch = 1000
    step_per_collect = 1000
    repeat_per_collect = 4
    batch_size = 256
    training_num = min(10, num_envs)
    test_num = min(1, num_envs)
    
    # Setup logger
    log_path = os.path.join('log', 'tianshou', 'residential_v2g')
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)
    
    # Create environments
    train_envs = DummyVectorEnv(
        [make_env(config_file, seed=seed + i) for i in range(training_num)]
    )
    test_envs = DummyVectorEnv(
        [make_env(config_file, seed=seed + 1000 + i) for i in range(test_num)]
    )
    
    # Get environment info
    env = make_env(config_file)()
    state_shape = len(V2G_profit_max(env))
    action_shape = env.action_space.shape or env.action_space.n
    
    # Create policy network
    net = PolicyNet(state_shape, action_shape, device).to(device)
    
    # Create policy
    policy = PPOPolicy(
        actor=net,
        critic=net,
        optim=torch.optim.Adam(net.parameters(), lr=lr),
        dist_fn=torch.distributions.Categorical,
        action_space=env.action_space,
        deterministic_eval=True,
        action_scaling=True,
        action_bound_method='clip',
    )
    
    # Create collectors
    train_collector = Collector(
        policy, 
        train_envs,
        VectorReplayBuffer(buffer_size, len(train_envs)),
        exploration_noise=True
    )
    test_collector = Collector(policy, test_envs)
    
    # Define a test function
    def test_fn(epoch, env_step):
        print(f"Epoch: {epoch}, Env Step: {env_step}")
        policy.eval()
        test_result = test_collector.collect(n_episode=1)
        print(f"Test reward: {test_result['rews'].mean()}")
        policy.train()
    
    # Train the agent
    result = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=epoch,
        step_per_epoch=step_per_epoch,
        repeat_per_collect=repeat_per_collect,
        episode_per_test=test_num,
        batch_size=batch_size,
        step_per_collect=step_per_collect,
        test_fn=test_fn,
        stop_fn=lambda mean_rewards: mean_rewards >= 1000,
        logger=logger,
    ).run()
    
    # Save the final policy
    torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))
    return result

if __name__ == "__main__":
    train()
