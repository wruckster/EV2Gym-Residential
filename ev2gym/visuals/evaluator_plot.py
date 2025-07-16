import os
import pickle
from typing import List, Union, Optional
import matplotlib.pyplot as plt
import matplotlib.ticker as _mticker
import numpy as np
import pandas as pd
import datetime as _dt


# -------------------------------------------------------------
# Generic helper ------------------------------------------------
# -------------------------------------------------------------

def _load_replay(file_path: str):
    """Load a pickle replay file and return the stored object.

    The original replay is produced by `EvCityReplay` and should therefore
    be a Python *object* with attributes rather than a plain dict.  We keep
    the loader very generic so that it also works if the replay has been
    serialised as a dictionary.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Replay file not found: {file_path}")

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    # In historic versions the object is stored under the attribute
    # `replay` – handle that gracefully.
    if hasattr(data, "replay"):
        return data.replay
    return data


def _extract_series(
    replay_obj: Union[object, dict], attr_name: str
) -> Optional[np.ndarray]:
    """Safely extract a time series from a replay object or dict."""
    # Handle object-based replays first
    if hasattr(replay_obj, attr_name):
        arr = getattr(replay_obj, attr_name)
        # Some attributes are saved as list; normalise to ndarray
        return np.asarray(arr)

    # Then, handle older dict-based replays for backward compatibility
    if isinstance(replay_obj, dict) and attr_name in replay_obj:
        return np.asarray(replay_obj[attr_name])

    return None


# -------------------------------------------------------------
# Main plotting API -------------------------------------------
# -------------------------------------------------------------

def plot_from_replay(replay_dir: str, plot_type: str = "city", save_path: str = None):
    """
    Generates and saves plots from a replay file or a directory containing replay files.

    Args:
        replay_dir (str): Path to a replay .pkl file or a directory containing them.
        plot_type (str): The type of plot to generate (e.g., 'city').
        save_path (str): Optional path to save the plot. If None, saves in replay_dir.
    """
    replay_file_path = None
    if os.path.isdir(replay_dir):
        # Find the first .pkl file in the directory
        try:
            replay_file_path = next(
                os.path.join(replay_dir, f) for f in os.listdir(replay_dir) if f.endswith('.pkl')
            )
        except StopIteration:
            print(f"[evaluator_plot] Error: No .pkl replay files found in directory {replay_dir}")
            return
    elif os.path.isfile(replay_dir) and replay_dir.endswith('.pkl'):
        replay_file_path = replay_dir
    else:
        print(f"[evaluator_plot] Error: Provided path is not a valid replay file or directory: {replay_dir}")
        return

    if save_path is None:
        save_path = os.path.join(os.path.dirname(replay_file_path), "evaluation_plots.png")

    try:
        with open(replay_file_path, 'rb') as f:
            replay_data = pickle.load(f)
    except FileNotFoundError:
        print(f"[evaluator_plot] Error: Replay file not found at {replay_file_path}")
        return
    except Exception as e:
        print(f"[evaluator_plot] Error loading replay file: {e}")
        return

    # --- Extract Data ---
    current_power_usage = _extract_series(replay_data, 'current_power_usage')
    tracking_error = _extract_series(replay_data, 'tracking_error')
    ev_count = _extract_series(replay_data, 'ev_count')
    cumulative_cost = _extract_series(replay_data, 'cumulative_cost')

    # Convert to numpy arrays for safe operations
    current_power_usage = np.asarray(current_power_usage) if current_power_usage is not None else np.asarray([])
    tracking_error = np.asarray(tracking_error) if tracking_error is not None else np.asarray([])
    ev_count = np.asarray(ev_count) if ev_count is not None else np.asarray([])
    cumulative_cost = np.asarray(cumulative_cost) if cumulative_cost is not None else np.asarray([])

    # If key series are missing, try sensible fall-backs so that plotting still proceeds.
    n_steps = len(current_power_usage)

    # 1) Tracking error: |setpoint − actual| if not provided
    if tracking_error.size == 0 and n_steps > 0:
        setpoints = _extract_series(replay_data, 'power_setpoints')
        if setpoints is not None and len(setpoints) == n_steps:
            tracking_error = np.abs(np.asarray(setpoints) - current_power_usage)
        else:
            # default to zeros
            tracking_error = np.zeros(n_steps)

    # 2) EV count: derive from total_evs_parked if missing
    if ev_count.size == 0:
        ev_count_series = _extract_series(replay_data, 'total_evs_parked')
        if ev_count_series is not None:
            ev_count = np.asarray(ev_count_series)
        elif n_steps > 0:
            ev_count = np.zeros(n_steps)

    # 3) Cumulative cost / reward: cumulative sum of per-step rewards if missing
    if cumulative_cost.size == 0:
        rewards = _extract_series(replay_data, 'reward_history')
        if rewards is not None:
            cumulative_cost = np.cumsum(np.asarray(rewards))
        elif n_steps > 0:
            cumulative_cost = np.zeros(n_steps)

    # After fall-backs, ensure all required arrays have data; if not, abort.
    if current_power_usage.size == 0 or tracking_error.size == 0:
        print("[evaluator_plot] Error: Essential series (power usage / tracking error) still missing. Skipping plot generation.")
        return

    # --- Create Plot ---
    fig, axs = plt.subplots(4, 1, figsize=(15, 20), sharex=True)
    fig.suptitle('Evaluation Results from Replay', fontsize=16)

    # 1. Power Usage
    axs[0].plot(current_power_usage, label='Total Power Usage (kW)', color='blue')
    axs[0].set_ylabel('Power (kW)')
    axs[0].set_title('Power Usage vs. Time')
    axs[0].legend()
    axs[0].grid(True)

    # 2. Tracking Error
    axs[1].plot(tracking_error, label='Tracking Error (kW)', color='red')
    axs[1].set_ylabel('Error (kW)')
    axs[1].set_title('Tracking Error vs. Time')
    axs[1].legend()
    axs[1].grid(True)

    # 3. Number of EVs
    axs[2].plot(ev_count, label='Number of EVs Connected', color='green', linestyle='--')
    axs[2].set_ylabel('Count')
    axs[2].set_title('Number of Connected EVs vs. Time')
    axs[2].legend()
    axs[2].grid(True)

    # 4. Cumulative Cost/Reward
    axs[3].plot(cumulative_cost, label='Cumulative Cost/Reward', color='purple')
    axs[3].set_xlabel('Time Step')
    axs[3].set_ylabel('Cost/Reward ($)')
    axs[3].set_title('Cumulative Cost/Reward vs. Time')
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(save_path)
    plt.close()
    print(f"[evaluator_plot] Evaluation plot saved to {save_path}")


def _plot_main(replays, labels, save_path):
    """Generate the main 6-panel evaluation plot."""
    plt.close("all")
    plt.style.use("seaborn-v0_8")
    fig = plt.figure(figsize=(15, 12))

    # ------------------------------------------------------------------
    # 3. Extract data series from replays
    # ------------------------------------------------------------------
    power_data = []
    setpoint_data = []
    ev_count_data = []
    reward_data = []
    solar_data = []  # aggregated solar power per replay
    demand_data = []  # aggregated residential inflexible load per replay

    for rep in replays:
        power = _extract_series(rep, "current_power_usage")
        setpoint = _extract_series(rep, "power_setpoints")
        ev_count = _extract_series(rep, "total_evs_parked")
        rewards = _extract_series(rep, "reward_history")
        demand = _extract_series(rep, "tr_inflexible_loads")
        solar = _extract_series(rep, "tr_solar_power")

        if power is not None:
            power_data.append(power)
        if setpoint is not None:
            setpoint_data.append(setpoint)
        if ev_count is not None:
            ev_count_data.append(ev_count)
        if rewards is not None:
            reward_data.append(rewards)
        if demand is not None:
            demand_tot = demand.sum(axis=0) if isinstance(demand, np.ndarray) else np.sum(demand, axis=0)
            demand_data.append(demand_tot)
        if solar is not None:
            # Sum across transformers to get total solar generation
            solar_tot = solar.sum(axis=0) if isinstance(solar, np.ndarray) else np.sum(solar, axis=0)
            solar_data.append(solar_tot)

    # ------------------------------------------------------------------
    # 4. Create subplots
    # ------------------------------------------------------------------
    grid = plt.GridSpec(3, 2, figure=fig)

    # 4.1 Total power usage (actual vs setpoint)
    ax1 = fig.add_subplot(grid[0, 0])
    _plot_power_usage(ax1, power_data, setpoint_data, demand_data, solar_data, labels)

    # 4.2 Power tracking error
    ax2 = fig.add_subplot(grid[0, 1])
    _plot_tracking_error(ax2, power_data, setpoint_data, labels)

    # 4.3 Total EVs parked
    ax3 = fig.add_subplot(grid[1, 0])
    _plot_ev_count(ax3, ev_count_data, labels)

    # 4.4 Cumulative reward
    ax4 = fig.add_subplot(grid[1, 1])
    _plot_cumulative_reward(ax4, reward_data, labels)

    # 4.5 Per-step reward
    ax5 = fig.add_subplot(grid[2, 0])
    _plot_per_step_reward(ax5, reward_data, labels)

    # 4.6 Info panel
    ax6 = fig.add_subplot(grid[2, 1])
    _add_info_panel(ax6, replays, labels)

    # Apply step + datetime formatter to all time-series axes
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        _apply_time_formatter(ax, replays[0])

    # ------------------------------------------------------------------
    # 5. Finalize and save
    # ------------------------------------------------------------------
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=120)
        print(f"[evaluator_plot] Saved evaluation plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def _plot_power_usage(ax, power_data, setpoint_data, demand_data, solar_data, labels):
    for i, power in enumerate(power_data):
        ax.plot(power, label=f"{labels[i]} – actual")
    for i, setpoint in enumerate(setpoint_data):
        ax.plot(setpoint, "--", label=f"{labels[i]} – set-point")
    for i, demand in enumerate(demand_data):
        ax.plot(demand, label=f"{labels[i]} – demand", linestyle=":", alpha=0.8)
    for i, solar in enumerate(solar_data):
        ax.plot(solar, label=f"{labels[i]} – solar", alpha=0.7)
    ax.set_title("Total Power / Demand vs PV Generation [kW]")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("kW")
    ax.legend()
    ax.grid(True, which="both", ls=":", lw=0.5)


def _plot_tracking_error(ax, power_data, setpoint_data, labels):
    for i, (power, setpoint) in enumerate(zip(power_data, setpoint_data)):
        error = power - setpoint
        ax.plot(error, label=labels[i])
    ax.set_title("Power Tracking Error (Actual - Setpoint)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("kW")
    ax.legend()
    ax.grid(True, which="both", ls=":", lw=0.5)


def _plot_ev_count(ax, ev_count_data, labels):
    for i, ev_count in enumerate(ev_count_data):
        ax.plot(ev_count, label=labels[i])
    ax.set_title("Total EVs Parked")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("# EVs")
    ax.legend()
    ax.grid(True, which="both", ls=":", lw=0.5)


def _plot_cumulative_reward(ax, reward_data, labels):
    for i, rewards in enumerate(reward_data):
        ax.plot(np.cumsum(rewards), label=labels[i])
    ax.set_title("Cumulative Episode Reward")
    ax.set_xlabel("Timestep")
    ax.legend()
    ax.grid(True, which="both", ls=":", lw=0.5)


def _plot_per_step_reward(ax, reward_data, labels):
    for i, rewards in enumerate(reward_data):
        ax.plot(rewards, label=labels[i])
    ax.set_title("Per-Step Reward")
    ax.set_xlabel("Timestep")
    ax.legend()
    ax.grid(True, which="both", ls=":", lw=0.5)


def _add_info_panel(ax, replays, labels):
    ax.axis("off")
    meta_txt = "\n".join([f"{label}: {replay.sim_length} steps" for label, replay in zip(labels, replays)])
    ax.text(0.02, 0.98, meta_txt, va="top", ha="left", fontsize=10)


def _plot_details(replay, save_path):
    """Generate detailed plots similar to ev_city_plot from a replay file."""
    plt.close("all")
    plt.style.use("seaborn-v0_8")
    fig = plt.figure(figsize=(20, 25))
    grid = plt.GridSpec(3, 1, figure=fig, hspace=0.4)

    # Plot 1: Transformer Power
    ax1 = fig.add_subplot(grid[0, 0])
    _plot_transformer_power(ax1, replay)

    # Plot 2: Charging Station Currents
    ax2 = fig.add_subplot(grid[1, 0])
    _plot_cs_currents(ax2, replay)

    # Plot 3: EV Energy Levels
    ax3 = fig.add_subplot(grid[2, 0])
    _plot_ev_energy(ax3, replay)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"[evaluator_plot] Saved detailed plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def _plot_transformer_power(ax, replay):
    """Plot transformer power usage from a replay."""
    tr_power = _extract_series(replay, "tr_power")
    if tr_power is None:
        ax.text(0.5, 0.5, "No transformer power data", ha="center", va="center")
        return

    for i, p in enumerate(tr_power):
        ax.plot(p, label=f"Transformer {i}")
    ax.set_title("Transformer Power Usage [kW]")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("kW")
    ax.legend()
    ax.grid(True, which="both", ls=":", lw=0.5)
    _apply_time_formatter(ax, replay)


def _plot_cs_currents(ax, replay):
    """Plot charging station currents from a replay."""
    cs_currents = _extract_series(replay, "cs_current")
    if cs_currents is None:
        ax.text(0.5, 0.5, "No CS current data", ha="center", va="center")
        return

    for i, c in enumerate(cs_currents):
        ax.plot(c, label=f"CS {i}")
    ax.set_title("Charging Station Current [A]")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Amperes")
    ax.legend()
    ax.grid(True, which="both", ls=":", lw=0.5)
    _apply_time_formatter(ax, replay)


def _plot_ev_energy(ax, replay):
    """Plot EV energy levels from a replay."""
    ev_energy = _extract_series(replay, "port_energy_level")
    if ev_energy is None:
        ax.text(0.5, 0.5, "No EV energy data", ha="center", va="center")
        return

    # ev_energy is (n_ports, n_cs, n_steps)
    n_cs = ev_energy.shape[1]
    for cs_idx in range(n_cs):
        for port_idx in range(ev_energy.shape[0]):
            ax.plot(ev_energy[port_idx, cs_idx, :], label=f"CS {cs_idx}, Port {port_idx}")

    ax.set_title("EV Energy Level [kWh]")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("kWh")
    if n_cs * ev_energy.shape[0] < 15:  # Avoid crowded legend
        ax.legend()
    ax.grid(True, which="both", ls=":", lw=0.5)
    _apply_time_formatter(ax, replay)


def _plot_prices(replay, save_path):
    """Generate a plot of electricity prices."""
    charge_p = _extract_series(replay, "charge_prices")
    discharge_p = _extract_series(replay, "discharge_prices")

    if charge_p is None and discharge_p is None:
        print("[evaluator_plot] No price data found in replay.")
        return

    n_steps = len(charge_p[0]) if charge_p is not None else len(discharge_p[0])
    timesteps = np.arange(n_steps)

    plt.close("all")
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(12, 6))

    # Prices can be per-CS; plot the first as a representative sample.
    if charge_p is not None:
        ax.plot(timesteps, charge_p[0], label="Charge Price")
    if discharge_p is not None:
        ax.plot(timesteps, discharge_p[0], label="Discharge Price")

    ax.set_title("Electricity Prices [€/kWh]")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Price [€/kWh]")
    ax.legend()
    ax.grid(True, which="both", ls=":", lw=0.5)
    ax.axhline(0, color="black", lw=0.5, ls="--")
    _apply_time_formatter(ax, replay)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=120)
        print(f"[evaluator_plot] Saved prices plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def _plot_solar(replay, save_path):
    """Generate a plot of solar power generation."""
    solar_power = _extract_series(replay, "tr_solar_power")

    if solar_power is None:
        print("[evaluator_plot] No solar power data found in replay.")
        return

    n_steps = len(solar_power[0])
    timesteps = np.arange(n_steps)

    plt.close("all")
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(12, 6))

    # solar_power is (n_transformers, n_steps)
    for i, tr_solar in enumerate(solar_power):
        ax.plot(timesteps, tr_solar, label=f"Transformer {i}")

    ax.set_title("Solar Power Generation [kW]")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("kW")
    ax.legend()
    ax.grid(True, which="both", ls=":", lw=0.5)
    _apply_time_formatter(ax, replay)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=120)
        print(f"[evaluator_plot] Saved solar plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def _plot_ev_city(replay, save_path):
    """Generate EV-city plots (multiple PNGs) using the legacy ev_city_plot()."""
    import datetime as _dt
    try:
        from ev2gym.visuals.plots import ev_city_plot as _ev_city_plot
    except ImportError:
        print("[evaluator_plot] Could not import ev_city_plot – falling back to details plot.")
        _plot_details(replay, save_path)
        return

    # Extract simulation metadata
    sim_len = getattr(replay, "sim_length", None)
    if sim_len is None:
        print("[evaluator_plot] Could not extract simulation length from replay.")
        return

    start_dt = getattr(replay, "sim_date", None)
    if start_dt is None:
        # Fallback: today at midnight
        start_dt = _dt.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    timescale = getattr(replay, "timescale", 15)
    end_dt = start_dt + _dt.timedelta(minutes=timescale * (sim_len - 1))

    # Generate plots
    for i in range(sim_len):
        dt = start_dt + _dt.timedelta(minutes=timescale * i)
        _ev_city_plot(replay, dt, save_path=f"{save_path}_{i:03d}.png", lightweight_plots=False)


# ----------------------------------------------------------------------
# X-axis step + datetime formatter --------------------------------------
# ----------------------------------------------------------------------

def _apply_time_formatter(ax, replay):
    """Format *ax*'s x-axis ticks as "step: N YY-MM-DD HH:MM".

    If *replay* lacks ``sim_date`` the call is a no-op.
    """
    start_dt = getattr(replay, "sim_date", None)
    timescale = getattr(replay, "timescale", 15)
    if start_dt is None:
        return  # nothing to do

    def _fmt(x, _):  # x is the tick position (timestep index)
        step = int(x)
        dt = start_dt + _dt.timedelta(minutes=timescale * step)
        return f"step: {step}\n{dt.strftime('%y-%m-%d %H:%M')}"

    ax.xaxis.set_major_formatter(_mticker.FuncFormatter(_fmt))

    # Use at most ~6 ticks for readability
    ax.xaxis.set_major_locator(_mticker.MaxNLocator(nbins=6, integer=True, prune=None))


# ----------------------------------------------------------------------
# Backwards-compat convenience wrappers --------------------------------
# ----------------------------------------------------------------------
# The evaluator script imports a handful of individual plotting functions.
# We keep stubs here to avoid breaking that workflow.  They all delegate
# to `plot_from_replay` until more fine-grained plots are re-implemented.


def plot_total_power(*args, **kwargs):
    """Deprecated wrapper – uses `plot_from_replay`."""
    print("[evaluator_plot] plot_total_power() is deprecated – use plot_from_replay().")
    return plot_from_replay(*args, **kwargs)


def plot_comparable_EV_SoC(*args, **kwargs):
    print("[evaluator_plot] plot_comparable_EV_SoC() is deprecated – use plot_from_replay().")
    return plot_from_replay(*args, **kwargs)


def plot_total_power_V2G(*args, **kwargs):
    print("[evaluator_plot] plot_total_power_V2G() is deprecated – use plot_from_replay().")
    return plot_from_replay(*args, **kwargs)


def plot_actual_power_vs_setpoint(*args, **kwargs):
    print("[evaluator_plot] plot_actual_power_vs_setpoint() is deprecated – use plot_from_replay().")
    return plot_from_replay(*args, **kwargs)


def plot_comparable_EV_SoC_single(*args, **kwargs):
    print("[evaluator_plot] plot_comparable_EV_SoC_single() is deprecated – use plot_from_replay().")
    return plot_from_replay(*args, **kwargs)


def plot_prices(*args, **kwargs):
    print("[evaluator_plot] plot_prices() is deprecated – use plot_from_replay().")
    return plot_from_replay(*args, **kwargs)


def plot_action_analysis(action_data: dict, save_path: str) -> None:
    """Plot action distribution and standard deviations over time.
    
    Args:
        action_data: Dictionary containing action data with keys:
            - 'stds': Array of action standard deviations over time
            - 'means': Array of action means over time
            - 'actions': Array of actual actions taken
        save_path: Directory to save the plots
    """
    import seaborn as sns
    import pandas as pd
    
    # Plot action standard deviations over time
    if 'stds' in action_data and len(action_data['stds']) > 0:
        plt.figure(figsize=(12, 6))
        stds = np.array(action_data['stds'])
        for i in range(stds.shape[1]):
            plt.plot(stds[:, i], label=f'Action Dim {i}')
        plt.title('Action Standard Deviations Over Time')
        plt.xlabel('Timestep')
        plt.ylabel('Std Dev')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'action_stds.png'))
        plt.close()
    
    # Plot action distributions
    if 'actions' in action_data and len(action_data['actions']) > 0:
        actions = np.array(action_data['actions'])
        n_dims = actions.shape[1] if len(actions.shape) > 1 else 1
        
        # Create subplots for each action dimension
        fig, axes = plt.subplots(n_dims, 2, figsize=(15, 4 * n_dims))
        if n_dims == 1:
            axes = np.array([axes])
        
        for i in range(n_dims):
            dim_actions = actions[:, i] if n_dims > 1 else actions
            
            # Time series of actions
            axes[i, 0].plot(dim_actions, alpha=0.6)
            axes[i, 0].set_title(f'Action Dim {i} Over Time')
            axes[i, 0].set_xlabel('Timestep')
            axes[i, 0].set_ylabel('Action Value')
            axes[i, 0].grid(True)
            
            # Histogram of action distribution
            sns.histplot(dim_actions, kde=True, ax=axes[i, 1])
            axes[i, 1].set_title(f'Action Dim {i} Distribution')
            axes[i, 1].set_xlabel('Action Value')
            axes[i, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'action_distributions.png'))
        plt.close()

def plot_learning_curves(metrics: dict, save_path: str) -> None:
    """Plot learning curves from training metrics.
    
    Args:
        metrics: Dictionary containing training metrics
        save_path: Directory to save the plots
    """
    if not metrics:
        return
    
    # Plot returns
    if 'returns' in metrics and metrics['returns']:
        plt.figure(figsize=(12, 6))
        plt.plot(metrics['returns'], label='Returns')
        if 'test_returns' in metrics and metrics['test_returns']:
            plt.plot(metrics['test_returns'], label='Test Returns')
        plt.title('Episode Returns Over Training')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'returns.png'))
        plt.close()
    
    # Plot losses
    loss_metrics = {k: v for k, v in metrics.items() if 'loss' in k.lower()}
    if loss_metrics:
        plt.figure(figsize=(12, 6))
        for name, values in loss_metrics.items():
            if values:  # Only plot if we have values
                plt.plot(values, label=name.replace('_', ' ').title())
        plt.title('Training Losses Over Time')
        plt.xlabel('Update Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'losses.png'))
        plt.close()
    
    # Plot policy metrics
    policy_metrics = {k: v for k, v in metrics.items() 
                     if any(m in k.lower() for m in ['entropy', 'kl', 'clipfrac'])}
    if policy_metrics:
        plt.figure(figsize=(12, 6))
        for name, values in policy_metrics.items():
            if values:  # Only plot if we have values
                plt.plot(values, label=name.replace('_', ' ').title())
        plt.title('Policy Metrics Over Time')
        plt.xlabel('Update Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'policy_metrics.png'))
        plt.close()

def plot_ev_metrics(ev_data: dict, save_path: str) -> None:
    """Plot EV-specific metrics.
    
    Args:
        ev_data: Dictionary containing EV metrics with keys like 'soc', 'charging_power', etc.
        save_path: Directory to save the plots
    """
    if not ev_data:
        return
    
    # Plot State of Charge over time
    if 'soc' in ev_data and len(ev_data['soc']) > 0:
        plt.figure(figsize=(12, 6))
        for ev_id, soc_values in ev_data['soc'].items():
            plt.plot(soc_values, label=f'EV {ev_id}')
        plt.title('State of Charge Over Time')
        plt.xlabel('Timestep')
        plt.ylabel('State of Charge')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'ev_soc.png'))
        plt.close()
    
    # Plot charging power over time
    if 'charging_power' in ev_data and len(ev_data['charging_power']) > 0:
        plt.figure(figsize=(12, 6))
        for ev_id, power_values in ev_data['charging_power'].items():
            plt.plot(power_values, label=f'EV {ev_id}')
        plt.title('Charging Power Over Time')
        plt.xlabel('Timestep')
        plt.ylabel('Power (kW)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'ev_charging_power.png'))
        plt.close()
