# /Users/benwruck/Documents/projects/EV2Gym/ev2gym/rl_agent/networks.py
"""
Network architectures for RL agents, designed for Tianshou compatibility.
"""

import numpy as np
import torch
import torch.nn as nn


class Actor(nn.Module):
    """Actor network with a Tianshou-compatible forward method."""

    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_shape)),
        )

    def forward(self, obs, state=None, info={}):
        """Accepts obs, state, and info, returns logits and state."""
        if isinstance(obs, dict):
            obs = obs['obs']
        if not isinstance(obs, torch.Tensor):
            # Tianshou's batching should handle device placement
            obs = torch.tensor(obs, dtype=torch.float32)

        logits = self.model(obs)
        return logits, state


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

    def forward(self, obs, state=None, info={}):
        """
        This forward pass is not directly used by PPOPolicy when actor and
        critic are passed separately, but it's good practice to have it for
        testing or for use with other policy types.
        """
        logits, state = self.actor(obs, state=state, info=info)
        value = self.critic(obs).flatten()
        return (logits, value), state