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

def plot_from_replay(
    replay_files: Union[str, List[str]],
    save_path: str = "evaluation_plots.png",
    labels: Optional[List[str]] = None,
    plot_type: str = "main",
):
    """Create evaluation figures from one or more replay files.

    Parameters
    ----------
    replay_files
        A single path or a list of paths pointing to replay ``.pkl`` files.
    save_path
        Where to write the PNG.  When *None*, the plot is only shown.
    labels
        Optional legend labels – one entry per replay.  If omitted, file
        basenames are used instead.
    plot_type
        Type of plot to generate. Options:
        - "main": Standard 6-panel evaluation plot (default)
        - "replays": 4-panel residential simulation plot
        - "prices": Plot of electricity prices
        - "solar": Plot of solar power generation
        - "details": Detailed plots of transformer, CS, and EV states
        - "ev_status": Plot of EV plug-in status (home/work/away)
    """

    # ------------------------------------------------------------------
    # 0. Normalise input
    # ------------------------------------------------------------------
    if isinstance(replay_files, str):
        replay_files = [replay_files]

    if labels is None:
        labels = [os.path.basename(f) for f in replay_files]

    # Ensure we have the same number of labels as replay files
    if len(labels) != len(replay_files):
        labels = [f"Replay {i}" for i in range(len(replay_files))]

    # ------------------------------------------------------------------
    # 1. Load replay data
    # ------------------------------------------------------------------
    replays = []
    for replay_file in replay_files:
        try:
            rep = _load_replay(replay_file)
            replays.append(rep)
        except Exception as e:
            print(f"[evaluator_plot] Error loading replay {replay_file}: {e}")
            continue

    if not replays:
        print("[evaluator_plot] No valid replay files loaded.")
        return

    # ------------------------------------------------------------------
    # 2. Generate the requested plot type
    # ------------------------------------------------------------------
    if plot_type == "main":
        _plot_main(replays, labels, save_path)
    elif plot_type == "replays":
        _plot_replays(replays, labels, save_path)
    elif plot_type == "prices":
        # For simplicity, plot prices from the first replay only
        _plot_prices(replays[0], save_path)
    elif plot_type == "solar":
        _plot_solar(replays[0], save_path)
    elif plot_type == "details":
        _plot_details(replays[0], save_path)
    elif plot_type == "ev_status":
        _plot_ev_status(replays[0], save_path)
    else:  # Default to main plot
        _plot_main(replays, labels, save_path)


def _plot_main(replays, labels, save_path):
    """Generate the main 6-panel evaluation plot."""
    plt.close("all")
    plt.style.use("seaborn-v0_8")
    fig = plt.figure(figsize=(15, 15))  

    # ------------------------------------------------------------------
    # 3. Extract data series from replays
    # ------------------------------------------------------------------
    power_data = []
    setpoint_data = []
    ev_count_data = []
    reward_data = []
    solar_data = []  
    demand_data = []  
    ev_locations = []  

    for rep in replays:
        power = _extract_series(rep, "current_power_usage")
        setpoint = _extract_series(rep, "power_setpoints")
        ev_count = _extract_series(rep, "total_evs_parked")
        rewards = _extract_series(rep, "reward_history")
        demand = _extract_series(rep, "tr_inflexible_loads")
        solar = _extract_series(rep, "tr_solar_power")
        
        # Extract EV metadata if available (from enhanced replay)
        ev_meta = _extract_ev_location_data(rep)
        if ev_meta is not None:
            ev_locations.append(ev_meta)

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
    grid = plt.GridSpec(4, 2, figure=fig)  

    # 4.1 Total power usage (actual vs setpoint)
    ax1 = fig.add_subplot(grid[0, 0])
    _plot_power_usage(ax1, power_data, setpoint_data, demand_data, solar_data, labels)

    # 4.2 EV Trajectory
    ax2 = fig.add_subplot(grid[0, 1])
    _plot_ev_trajectory(ax2, replays[0], np.arange(len(replays[0].reward_history)))

    # 4.3 Total EVs parked
    ax3 = fig.add_subplot(grid[1, 0])
    _plot_ev_count(ax3, ev_count_data, labels)

    # 4.4 Cumulative reward
    ax4 = fig.add_subplot(grid[1, 1])
    _plot_cumulative_reward(ax4, reward_data, labels)

    # 4.5 Per-step reward
    ax5 = fig.add_subplot(grid[2, 0])
    _plot_per_step_reward(ax5, reward_data, labels)
    
    # 4.6 EV plug-in status (NEW)
    ax6 = fig.add_subplot(grid[2, 1])
    _plot_ev_plug_status(ax6, ev_locations, labels, replays[0])

    # 4.7 EV Trajectory
    ax7 = fig.add_subplot(grid[3, 0:])
    plot_ev_trajectories(replays[0], ax7)

    # Apply step + datetime formatter to all time-series axes
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
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


def _plot_replays(replays, labels, save_path):
    """Generate the 4-panel residential simulation plot."""
    plt.close("all")
    plt.style.use("seaborn-v0_8")
    fig = plt.figure(figsize=(12, 12))  

    # ------------------------------------------------------------------
    # 3. Extract data series from replays
    # ------------------------------------------------------------------
    power_data = []
    setpoint_data = []
    ev_count_data = []
    reward_data = []
    solar_data = []  
    demand_data = []  
    ev_locations = []  

    for rep in replays:
        power = _extract_series(rep, "current_power_usage")
        setpoint = _extract_series(rep, "power_setpoints")
        ev_count = _extract_series(rep, "total_evs_parked")
        rewards = _extract_series(rep, "reward_history")
        demand = _extract_series(rep, "tr_inflexible_loads")
        solar = _extract_series(rep, "tr_solar_power")
        
        # Extract EV metadata if available (from enhanced replay)
        ev_meta = _extract_ev_location_data(rep)
        if ev_meta is not None:
            ev_locations.append(ev_meta)

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
    grid = plt.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    # Extract a common time_steps array for plotting
    time_steps = np.arange(len(replays[0].reward_history))

    # 4.1 Energy Flow Breakdown
    ax1 = fig.add_subplot(grid[0, 0])
    _plot_energy_flow_breakdown(ax1, replays[0], time_steps)

    # 4.2 EV SoC vs. Price
    ax2 = fig.add_subplot(grid[0, 1])
    _plot_ev_soc_vs_price(ax2, replays[0], time_steps)

    # 4.3 EV Trajectory (SoC and Location)
    ax3 = fig.add_subplot(grid[1, 0])
    _plot_ev_trajectory(ax3, replays[0], time_steps)

    # 4.4 Cumulative Reward/Cost
    ax4 = fig.add_subplot(grid[1, 1])
    _plot_cumulative_reward_cost(ax4, replays[0], time_steps)

    # Apply step + datetime formatter to all time-series axes
    for ax in [ax1, ax2, ax3, ax4]:
        _apply_time_formatter(ax, replays[0])

    # ------------------------------------------------------------------
    # 5. Finalize and save
    # ------------------------------------------------------------------
    fig.suptitle('Residential Simulation Evaluation', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
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
    if n_cs * ev_energy.shape[0] < 15:  
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
        return  

    def _fmt(x, _):  
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


def _extract_ev_location_data(replay_obj):
    """Extract EV location and plug-in status data from replay.
    
    Returns a dictionary with timesteps as keys and lists of EVs with their
    locations and plug-in status as values.
    """
    # Try to extract EV metadata if it exists in the replay
    if hasattr(replay_obj, "ev_location_data"):
        return replay_obj.ev_location_data
    
    # For backward compatibility with older replays
    return None


def _plot_ev_plug_status(ax, ev_location_data_list, labels, replay):
    """Plot EV plug-in status over time (home/work/away)."""
    if not ev_location_data_list:
        ax.text(0.5, 0.5, "No EV plug-in data available", ha="center", va="center")
        ax.set_title("EV Plug-in Status")
        return
    
    timesteps = None
    colors = {'home': 'green', 'work': 'blue', 'away': 'red'}
    markers = {'home': 'o', 'work': 's', 'away': 'x'}
    
    for i, ev_data in enumerate(ev_location_data_list):
        # Extract data for plotting
        home_count = []
        work_count = []
        away_count = []
        
        if isinstance(ev_data, dict):
            timesteps = sorted(ev_data.keys())
            for t in timesteps:
                loc_counts = {"home": 0, "work": 0, "away": 0}
                for ev_info in ev_data[t]:
                    if "location_type" in ev_info:
                        loc_type = ev_info["location_type"]
                        if loc_type in loc_counts:
                            loc_counts[loc_type] += 1
                
                home_count.append(loc_counts["home"])
                work_count.append(loc_counts["work"])
                away_count.append(loc_counts["away"])
        
        # If no valid data, try to estimate from simulation length
        if not timesteps and hasattr(replay, "sim_length"):
            timesteps = list(range(replay.sim_length))
            home_count = [0] * len(timesteps)
            work_count = [0] * len(timesteps)
            away_count = [0] * len(timesteps)
        
        if timesteps:
            label_base = labels[i] if i < len(labels) else f"Replay {i}"
            ax.plot(timesteps, home_count, '-', color=colors['home'], 
                   marker=markers['home'], markevery=max(1, len(timesteps)//20),
                   label=f"{label_base} - Home")
            ax.plot(timesteps, work_count, '-', color=colors['work'], 
                   marker=markers['work'], markevery=max(1, len(timesteps)//20),
                   label=f"{label_base} - Work")
            ax.plot(timesteps, away_count, '-', color=colors['away'], 
                   marker=markers['away'], markevery=max(1, len(timesteps)//20),
                   label=f"{label_base} - Away")
    
    ax.set_title("EV Plug-in Status")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Number of EVs")
    ax.legend()
    ax.grid(True, which="both", ls=":", lw=0.5)


def _plot_ev_status(replay, save_path):
    """Generate detailed EV plug-in status plots."""
    plt.close("all")
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract EV metadata if available
    ev_data = _extract_ev_location_data(replay)
    
    if ev_data:
        _plot_ev_plug_status(ax, [ev_data], ["EV Location"], replay)
    else:
        ax.text(0.5, 0.5, "No EV plug-in data available", ha="center", va="center")
        ax.set_title("EV Plug-in Status")
    
    _apply_time_formatter(ax, replay)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=120)
        print(f"[evaluator_plot] Saved EV status plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_ev_trajectories(replay_data, ax=None):
    """Plot EV location states over time.
    
    Args:
        replay_data: Loaded replay data dictionary
        ax: Optional matplotlib axis to plot on
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
    
    ev_data = replay_data.ev_location_data
    states = replay_data.location_states
    timesteps = np.arange(ev_data.shape[2])
    
    # Create a colormap for states
    colors = {
        -1: 'lightgray',  # No EV
        0: 'green',      # Home
        1: 'blue',       # Work
        2: 'orange'      # Commuting
    }
    
    # Plot each EV's trajectory
    for port in range(ev_data.shape[0]):
        for cs in range(ev_data.shape[1]):
            trajectory = ev_data[port, cs, :]
            # Only plot if this port/cs combination has an EV at some point
            if not np.all(trajectory == -1):
                # Create segments for each state
                current_state = trajectory[0]
                start_idx = 0
                for i in range(1, len(trajectory)):
                    if trajectory[i] != current_state:
                        # Plot the segment
                        ax.plot(timesteps[start_idx:i], 
                               [port + cs * ev_data.shape[0]] * (i - start_idx),
                               color=colors[current_state],
                               linewidth=3,
                               label=states[current_state] if start_idx == 0 else "")
                        start_idx = i
                        current_state = trajectory[i]
                # Plot the final segment
                ax.plot(timesteps[start_idx:], 
                       [port + cs * ev_data.shape[0]] * (len(trajectory) - start_idx),
                       color=colors[current_state],
                       linewidth=3,
                       label=states[current_state] if start_idx == 0 else "")
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('EV Port')
    ax.set_title('EV Location States Over Time')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    return ax


def _plot_ev_trajectory(ax, replay, time_steps):
    """Plot EV SoC and location trajectory.
    
    Priority of data sources:
    1. `ev_soc` and `ev_locations` arrays if they exist (legacy / explicit tracking)
    2. Derive from `port_energy_level` and `ev_location_data` which are always
       present in residential simulations. We use port 0 of charging station 0
       as a proxy for the "first EV". This keeps the plot informative even if
       detailed per-EV arrays were not saved.
    """
    sim_steps = len(time_steps)

    if hasattr(replay, 'ev_soc') and hasattr(replay, 'ev_locations') and replay.ev_soc.shape[1] > 0:
        ev_soc = replay.ev_soc[:sim_steps, 0] * 100  # percentage
        locations = replay.ev_locations[:sim_steps, 0]
    elif hasattr(replay, 'port_energy_level') and hasattr(replay, 'ev_location_data'):
        # Use port 0 @ CS 0 as representative trajectory
        soc_raw = replay.port_energy_level[0, 0, :sim_steps]
        max_cap = np.nanmax(soc_raw) if np.nanmax(soc_raw) > 0 else 1.0
        ev_soc = (soc_raw / max_cap) * 100  # normalise to percentage
        locations = replay.ev_location_data[0, 0, :sim_steps]
    else:
        ax.text(0.5, 0.5, 'EV Trajectory data not available in replay.',
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title("EV Trajectory: SoC & Location")
        return
 
    # Plot SoC line
    ax.plot(time_steps, ev_soc, label='EV SoC', color='#007ACC', zorder=10)
    ax.set_ylabel('State of Charge (%)', color='#007ACC')
    ax.tick_params(axis='y', labelcolor='#007ACC')
    ax.set_ylim(-5, 105)
    ax.set_title("EV Trajectory: SoC & Location (EV 0)")
    ax.grid(True, which="both", ls=":", lw=0.5)

    # Create a secondary axis for location bars
    ax_loc = ax.twinx()
    ax_loc.set_ylim(0, 1)
    ax_loc.set_yticks([]) # Hide y-ticks for the location axis

    # Define location colors and labels
    location_map = {
        0: {'label': 'Home', 'color': '#5CB85C'},
        1: {'label': 'Work', 'color': '#F0AD4E'},
        2: {'label': 'Driving', 'color': '#D9534F'}
    }

    # Plot location bars
    for loc_id, props in location_map.items():
        ax_loc.fill_between(time_steps, 0, 1, where=(locations == loc_id),
                             color=props['color'], alpha=0.6, label=props['label'], step='mid')

    # Create a single legend for both axes
    lines, labels = ax.get_legend_handles_labels()
    patches, patch_labels = ax_loc.get_legend_handles_labels()
    ax.legend(lines + patches, labels + patch_labels, loc='upper left')

def _plot_energy_flow_breakdown(ax, replay, time_steps):
    # Plot energy flow breakdown
    if hasattr(replay, 'energy_flow_breakdown'):
        ax.plot(time_steps, replay.energy_flow_breakdown['grid_draw'][:len(time_steps)], label='Grid')
        ax.plot(time_steps, replay.energy_flow_breakdown['solar_production'][:len(time_steps)], label='Solar')
        ax.plot(time_steps, replay.energy_flow_breakdown['battery_discharge'][:len(time_steps)], label='Battery')
        ax.plot(time_steps, replay.energy_flow_breakdown['ev_charge_demand'][:len(time_steps)], label='EV Charging')
        ax.set_title("Energy Flow Breakdown")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Energy [kWh]")
        ax.legend()
        ax.grid(True, which="both", ls=":", lw=0.5)
    else:
        ax.text(0.5, 0.5, 'Energy flow data not available in replay.', 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title("Energy Flow Breakdown")

def _plot_ev_soc_vs_price(ax, replay, time_steps):
    # Plot EV SoC vs. price
    # Check if we have port_energy_level data (for SoC)
    if hasattr(replay, 'port_energy_level') and replay.port_energy_level.shape[0] > 0:
        # Get the first EV's energy level (first port, first CS)
        ev_energy = replay.port_energy_level[0, 0, :len(time_steps)]
        
        # Calculate SoC as a percentage of max capacity (assuming 100 kWh as max if not available)
        max_capacity = 100  # Default max capacity in kWh
        if hasattr(replay, 'EVs') and len(replay.EVs) > 0:
            max_capacity = replay.EVs[0].battery_capacity
        
        ev_soc = (ev_energy / max_capacity) * 100
        ax.plot(time_steps, ev_soc, label='EV SoC')
        
        # Plot charge prices if available
        if hasattr(replay, 'charge_prices'):
            # Use the first charging station's prices
            prices = replay.charge_prices[0, :len(time_steps)]
            ax.plot(time_steps, prices, label='Charge Price')
        
        ax.set_title("EV SoC vs. Charge Price")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("SoC [%] / Price [€/kWh]")
        ax.legend()
        ax.grid(True, which="both", ls=":", lw=0.5)
    else:
        ax.text(0.5, 0.5, 'EV SoC data not available in replay.', 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title("EV SoC vs. Charge Price")

def _plot_cumulative_reward_cost(ax, replay, time_steps):
    # Plot cumulative reward and cost
    sim_steps = len(time_steps)
    
    # Check if we have reward history
    if hasattr(replay, 'reward_history') and len(replay.reward_history) > 0:
        rewards = replay.reward_history[:sim_steps]
        ax.plot(time_steps, np.cumsum(rewards), label='Cumulative Reward')
    
    # Check if we have cost history
    if hasattr(replay, 'cost_history') and len(replay.cost_history) > 0:
        costs = replay.cost_history[:sim_steps]
        ax.plot(time_steps, np.cumsum(costs), label='Cumulative Cost')
    
    ax.set_title("Cumulative Reward and Cost")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Reward / Cost [€]")
    ax.legend()
    ax.grid(True, which="both", ls=":", lw=0.5)
