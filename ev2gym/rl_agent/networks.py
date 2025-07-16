# /Users/benwruck/Documents/projects/EV2Gym/ev2gym/rl_agent/networks.py
"""
Network architectures for RL agents, designed for Tianshou compatibility.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Union, Optional


class Actor(nn.Module):
    """Actor network for continuous actions with learnable std dev."""

    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.action_dim = np.prod(action_shape)
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        self.output_layer = nn.Linear(128, self.action_dim * 2)

    def forward(self, obs, state=None, info={}):
        """
        Accepts obs, returns a concatenated tensor for mean and log_std,
        and the recurrent state (which is None for this feed-forward network).
        """
        if isinstance(obs, dict):
            obs = obs['obs']
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)

        features = self.model(obs)
        output = self.output_layer(features)

        return output, state


class Critic(nn.Module):
    """Critic network with a Tianshou-compatible forward method."""

    def __init__(self, state_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, obs, **kwargs):
        """Accepts obs and ignores other kwargs to be robust."""
        if isinstance(obs, dict):
            obs = obs['obs']
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)

        return self.model(obs)


class Actor_V2G_Continuous(nn.Module):
    """Actor network for continuous action spaces in V2G environments."""
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Tuple[int, ...] = (256, 256), activation: nn.Module = nn.ReLU(), max_action: float = 1.0):
        super().__init__()
        self.max_action = max_action
        layers = []
        input_dim = state_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(activation)
            input_dim = hidden_size
        self.feature_extractor = nn.Sequential(*layers)

        self.mean_layer = nn.Linear(input_dim, action_dim)
        self.log_std_layer = nn.Linear(input_dim, action_dim)

    def forward(
        self, 
        obs: torch.Tensor, 
        state: Optional[dict] = None, 
        info: dict = {}
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """Forward pass returns a concatenated tensor of mean and log_std, and the state."""
        if isinstance(obs, dict):
            obs = obs['obs']
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
            
        features = self.feature_extractor(obs)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        return torch.cat([mean, log_std], dim=-1), state


class Critic_V2G_Continuous(nn.Module):
    """Critic network for continuous action spaces in V2G environments."""
    def __init__(self, state_dim: int, hidden_sizes: Tuple[int, ...] = (256, 256), activation: nn.Module = nn.ReLU()):
        super().__init__()
        layers = []
        input_dim = state_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(activation)
            input_dim = hidden_size
        self.feature_extractor = nn.Sequential(*layers)
        self.value_head = nn.Linear(input_dim, 1)

    def forward(self, obs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass returns the state value."""
        if isinstance(obs, dict):
            obs = obs['obs']
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.value_head.weight.device)
        features = self.feature_extractor(obs)
        return self.value_head(features)


class PolicyNet(nn.Module):
    """
    A container for an actor and a critic network.
    This structure is compatible with Tianshou's PPOPolicy, which expects
    separate actor and critic modules.
    """

    def __init__(self, state_shape, action_shape, device='cpu'):
        super().__init__()
        self.device = device
        if isinstance(state_shape, int):
            state_shape = (state_shape,)

        self.actor = Actor(state_shape, action_shape)
        self.critic = Critic(state_shape)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights with orthogonal initialization."""
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, obs, state=None, info={}, mode: str = 'actor'):
        """
        Forward pass for the policy network.

        :param obs: Observation from the environment.
        :param state: Recurrent state (if any).
        :param info: Additional info from the environment.
        :param mode: 'actor' or 'critic'. Determines the output.
                     - 'actor': returns (mean, std), state
                     - 'critic': returns value
        """
        if isinstance(obs, dict):
            obs = obs['obs']
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        if torch.isnan(obs).any():
            print("NaN detected in observation:", obs)
            # You might want to raise an error or handle it in some way
            # For now, let's replace NaNs with zeros to prevent a crash
            obs = torch.nan_to_num(obs)

        if mode == 'actor':
            return self.actor(obs, state=state, info=info)
        elif mode == 'critic':
            return self.critic(obs)

        # Default behavior if mode is not specified (though PPO will use modes)
        actor_output, state = self.actor(obs, state=state, info=info)
        critic_output = self.critic(obs)
        return actor_output, critic_output
