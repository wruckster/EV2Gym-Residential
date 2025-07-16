"""Utility functions for tracking and logging training metrics."""
from typing import Dict, List, Optional, Union, Any
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import json
import time


class MetricsTracker:
    """Class for tracking and logging training metrics."""
    
    def __init__(self, log_dir: str, use_tensorboard: bool = True):
        """Initialize metrics tracker.
        
        Args:
            log_dir: Directory to save logs and plots
            use_tensorboard: Whether to use TensorBoard for logging
        """
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard
        self.metrics = defaultdict(list)
        self.start_time = time.time()
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize TensorBoard writer if enabled
        self.writer = SummaryWriter(log_dir) if use_tensorboard else None
    
    def log_metrics(self, 
                   metrics_dict: Dict[str, Union[float, int, np.ndarray]], 
                   step: Optional[int] = None) -> None:
        """Log metrics to memory and TensorBoard.
        
        Args:
            metrics_dict: Dictionary of metric names and values
            step: Current training step (epoch/update)
        """
        if step is None:
            step = len(self.metrics.get('train/returns', [])) - 1
        
        for name, value in metrics_dict.items():
            # Convert numpy arrays and tensors to Python scalars
            if isinstance(value, (np.ndarray, torch.Tensor)):
                if value.size == 1:
                    value = value.item()
                else:
                    # For arrays, store summary statistics
                    self.metrics[f'{name}/mean'].append(float(value.mean()))
                    self.metrics[f'{name}/std'].append(float(value.std()))
                    self.metrics[f'{name}/min'].append(float(value.min()))
                    self.metrics[f'{name}/max'].append(float(value.max()))
                    
                    # Log to TensorBoard
                    if self.writer is not None:
                        self.writer.add_scalar(f'{name}/mean', value.mean(), step)
                        self.writer.add_scalar(f'{name}/std', value.std(), step)
                        self.writer.add_histogram(name, value, step)
                    continue
            
            # Store the metric
            self.metrics[name].append(value)
            
            # Log to TensorBoard
            if self.writer is not None and isinstance(value, (int, float)):
                self.writer.add_scalar(name, value, step)
    
    def log_ev_metrics(self, 
                      ev_metrics: Dict[str, Dict[str, np.ndarray]], 
                      step: Optional[int] = None) -> None:
        """Log EV-specific metrics.
        
        Args:
            ev_metrics: Nested dictionary of EV metrics
                        {metric_name: {ev_id: values}}
            step: Current training step
        """
        if step is None:
            step = len(self.metrics.get('train/returns', [])) - 1
        
        for metric_name, ev_data in ev_metrics.items():
            for ev_id, values in ev_data.items():
                tag = f'ev/{metric_name}/ev_{ev_id}'
                if isinstance(values, (list, np.ndarray)) and len(values) > 0:
                    # Log statistics for the episode
                    self.log_metrics({
                        f'{tag}/mean': np.mean(values),
                        f'{tag}/max': np.max(values),
                        f'{tag}/min': np.min(values)
                    }, step)
                    
                    # Store the full trace for later plotting
                    if f'ev_traces/{tag}' not in self.metrics:
                        self.metrics[f'ev_traces/{tag}'] = {}
                    self.metrics[f'ev_traces/{tag}'][step] = values
    
    def log_action_stats(self, 
                        actions: np.ndarray, 
                        action_means: Optional[np.ndarray] = None,
                        action_stds: Optional[np.ndarray] = None,
                        step: Optional[int] = None) -> None:
        """Log statistics about actions taken by the policy.
        
        Args:
            actions: Array of actions taken
            action_means: Array of action means from policy
            action_stds: Array of action standard deviations from policy
            step: Current training step
        """
        if step is None:
            step = len(self.metrics.get('train/returns', [])) - 1
        
        # Log action statistics
        if actions is not None and len(actions) > 0:
            actions = np.asarray(actions)
            if len(actions.shape) == 1:
                actions = actions[:, np.newaxis]
                
            for i in range(actions.shape[1]):
                self.log_metrics({
                    f'actions/dim_{i}/mean': actions[:, i].mean(),
                    f'actions/dim_{i}/std': actions[:, i].std(),
                    f'actions/dim_{i}/min': actions[:, i].min(),
                    f'actions/dim_{i}/max': actions[:, i].max()
                }, step)
        
        # Log action distribution parameters if available
        if action_means is not None and len(action_means) > 0:
            action_means = np.asarray(action_means)
            if len(action_means.shape) == 1:
                action_means = action_means[:, np.newaxis]
                
            for i in range(action_means.shape[1]):
                self.log_metrics({
                    f'policy/action_mean/dim_{i}': action_means[:, i].mean(),
                    f'policy/action_std/dim_{i}': action_means[:, i].std()
                }, step)
        
        if action_stds is not None and len(action_stds) > 0:
            action_stds = np.asarray(action_stds)
            if len(action_stds.shape) == 1:
                action_stds = action_stds[:, np.newaxis]
                
            for i in range(action_stds.shape[1]):
                self.log_metrics({
                    f'policy/action_std/dim_{i}': action_stds[:, i].mean(),
                    f'policy/action_std_std/dim_{i}': action_stds[:, i].std()
                }, step)
    
    def save_metrics(self, filename: str = 'metrics.json') -> None:
        """Save all metrics to a JSON file.
        
        Args:
            filename: Name of the file to save metrics to
        """
        # Convert numpy arrays to lists for JSON serialization
        metrics_dict = {}
        for k, v in self.metrics.items():
            if isinstance(v, (np.ndarray, torch.Tensor)):
                metrics_dict[k] = v.tolist()
            elif isinstance(v, dict):
                metrics_dict[k] = {
                    k2: v2.tolist() if hasattr(v2, 'tolist') else v2 
                    for k2, v2 in v.items()
                }
            else:
                metrics_dict[k] = v
        
        # Save to file
        with open(os.path.join(self.log_dir, filename), 'w') as f:
            json.dump(metrics_dict, f, indent=2)
    
    def close(self) -> None:
        """Close any open resources (e.g., TensorBoard writer)."""
        if self.writer is not None:
            self.writer.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def early_stopping(returns: List[float], 
                 patience: int = 10, 
                 min_delta: float = 0.0) -> bool:
    """Check if training should stop early based on validation returns.
    
    Args:
        returns: List of validation returns
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        
    Returns:
        bool: True if training should stop, False otherwise
    """
    if len(returns) < patience + 1:
        return False
    
    best_return = max(returns)
    current_return = returns[-1]
    
    # Check if current return is not within min_delta of the best return
    if current_return + min_delta < best_return:
        # Check if we've waited for 'patience' epochs
        best_idx = returns.index(best_return)
        if len(returns) - best_idx > patience:
            return True
    
    return False
