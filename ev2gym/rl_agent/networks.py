# /Users/benwruck/Documents/projects/EV2Gym/ev2gym/rl_agent/networks.py
"""
Network architectures for RL agents, designed for Tianshou compatibility.
"""

import numpy as np
import logging
import torch
import torch.nn as nn


class Actor(nn.Module):
    """Actor network for continuous actions with learnable std dev."""

    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.action_dim = np.prod(action_shape)
        # Required by PPOPolicy for action scaling
        self.max_action = 1.0
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        self.mean_layer = nn.Linear(128, self.action_dim)
        self.log_std_layer = nn.Linear(128, self.action_dim)
        # Logging rate-limit helpers
        self._warn_count = 0
        self._warn_interval = 1000  # log every N occurrences
        self._obs_indices_logged = False

    def forward(self, obs, state=None, info={}):
        """
        Accepts obs, returns a tuple of (mean, std) for the action distribution
        and the recurrent state (which is None for this feed-forward network).
        """
        if isinstance(obs, dict):
            obs = obs['obs']
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)

        # Ensure obs has batch dimension
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)

        # Sanitize observations to avoid propagating NaN/Inf into the network
        if torch.logical_not(torch.isfinite(obs)).any():
            self._warn_count += 1
            if not self._obs_indices_logged:
                # Log indices of non-finite in first sample only (diagnostic)
                try:
                    sample0 = obs[0]
                    bad_mask = torch.logical_not(torch.isfinite(sample0))
                    idx = torch.nonzero(bad_mask, as_tuple=False).flatten().tolist()
                    logging.warning(f"[Actor] Non-finite observation detected; first-sample bad indices: {idx}")
                except Exception:
                    pass
                self._obs_indices_logged = True
            elif self._warn_count % self._warn_interval == 0:
                logging.warning("[Actor] Non-finite observation detected (rate-limited).")
            obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        # Clip extreme magnitudes
        obs = torch.clamp(obs, -1e6, 1e6)

        features = self.model(obs)
        # Sanitize features to avoid propagating NaN/Inf
        if torch.logical_not(torch.isfinite(features)).any():
            self._warn_count += 1
            if self._warn_count % self._warn_interval == 0:
                logging.warning("[Actor] Non-finite features detected (rate-limited); applying nan_to_num.")
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        mean = self.mean_layer(features)
        # Sanitize mean
        if torch.logical_not(torch.isfinite(mean)).any():
            self._warn_count += 1
            if self._warn_count % self._warn_interval == 0:
                logging.warning("[Actor] Non-finite action mean detected (rate-limited); applying nan_to_num.")
            mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        # Clamp mean to a reasonable range to prevent blow-ups
        mean = torch.clamp(mean, -10.0, 10.0)

        # We learn the log of the standard deviation for stability
        log_std = self.log_std_layer(features)
        # Sanitize log_std
        if torch.logical_not(torch.isfinite(log_std)).any():
            self._warn_count += 1
            if self._warn_count % self._warn_interval == 0:
                logging.warning("[Actor] Non-finite log_std detected (rate-limited); applying nan_to_num.")
            log_std = torch.nan_to_num(log_std, nan=0.0, posinf=0.0, neginf=0.0)
        # Clamp the log_std to prevent it from becoming too large or too small
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        # Always keep the first (batch) dimension, even if batch size is 1.
        # Removing the batch dimension causes shape mismatches during log_prob
        # calculation inside standard Tianshou PPOPolicy. Therefore, we no
        # longer squeeze singleton batch dimensions here.

        return (mean, std), state


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
        # Sanitize observations similar to Actor
        if torch.logical_not(torch.isfinite(obs)).any():
            logging.warning("[Critic] Non-finite observation detected; applying nan_to_num.")
            obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        obs = torch.clamp(obs, -1e6, 1e6)

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
