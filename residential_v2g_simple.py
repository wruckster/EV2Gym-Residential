"""
Residential V2G Simulation with Enhanced Tracking
"""
from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.mpc.empc_v2 import eMPC_V2G_v2
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from ev2gym.scripts.create_eval_replays import evalreplay
import numpy as np
import os

def setup_output_directory():
    """Create a timestamped output directory for results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/v2g_simulation_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def log_step(step_data, output_dir):
    """Log step data to a CSV file"""
    log_file = f"{output_dir}/simulation_log.csv"
    df = pd.DataFrame([step_data])
    
    if not os.path.exists(log_file):
        df.to_csv(log_file, index=False, mode='w', header=True)
    else:
        df.to_csv(log_file, index=False, mode='a', header=False)

def run_residential_v2g(save_replay=True):
    # Setup
    output_dir = setup_output_directory()
    config_file = "ev2gym/example_config_files/residential_v2g.yaml"
    timestep_minutes = 15  # Should match your config file

    # Create replay if needed
    if save_replay:
        print("Creating evaluation replay...")
        replay_path = evalreplay(
            config_file=config_file,
            save_opt_trajectories=True,
            save_replay=True
        )
        print(f"Replay saved to: {replay_path}")
    else:
        replay_path = None

    print("Setup complete")
    
    # Initialize environment and agent
    # Initialize environment with replay if available
    env = EV2Gym(
        config_file=config_file,
        verbose=True,
        save_plots=True,
        load_from_replay_path=replay_path if replay_path else None,
        save_replay=save_replay
    )
    agent = eMPC_V2G_v2(env, verbose=True)

    print("Environment and agent initialized")
    
    # Track metrics
    metrics = {
        'timestep': [],
        'datetime': [],
        'soc': [],
        'power_kw': [],
        'price': [],
        'action': [],
        'reward': [],
        'total_energy_charged_kwh': [],
        'total_energy_discharged_kwh': [],
        'current_cost': [],
        'cumulative_cost': 0
    }
    
    # Run simulation
    state, _ = env.reset()
    done = False
    start_time = env.sim_date  # Use the simulation date from the environment
    
    print("\n=== Starting V2G Simulation ===")
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"Simulation Length: {env.simulation_length} steps")
    print(f"Output Directory: {output_dir}\n")
    
    while not done:
        # Get current timestep info
        current_time = start_time + timedelta(minutes=env.current_step * timestep_minutes)
        
        # Get actions from agent
        actions = agent.get_action(env)
        
        # Step environment
        state, reward, done, _, info = env.step(actions)
        
        # Get EV and charger states
        ev = env.charging_stations[0].evs_connected[0] if env.charging_stations[0].evs_connected[0] else None
        charger = env.charging_stations[0]
        
        # Calculate metrics
        current_price = env.price_import[env.current_step-1] if hasattr(env, 'price_import') else 0
        power_kw = charger.current_power_output if hasattr(charger, 'current_power_output') else 0
        soc = ev.get_soc() if ev else 0
        
        # Update metrics
        metrics['timestep'].append(env.current_step)
        metrics['datetime'].append(current_time)
        metrics['soc'].append(soc)
        metrics['power_kw'].append(power_kw)
        metrics['price'].append(current_price)
        metrics['action'].append(actions[0] if actions else 0)
        metrics['reward'].append(reward)
        
        if hasattr(charger, 'total_energy_charged'):
            metrics['total_energy_charged_kwh'].append(charger.total_energy_charged)
            metrics['total_energy_discharged_kwh'].append(charger.total_energy_discharged)
        
        # Log progress
        if env.current_step % 4 == 0:  # Every hour
            print(f"\n=== Timestep {env.current_step} ===")
            print(f"Time: {current_time.strftime('%Y-%m-%d %H:%M')}")
            print(f"EV SoC: {soc*100:.1f}%")
            print(f"Power: {power_kw:.2f} kW")
            print(f"Price: {current_price:.2f} $/kWh")
            print(f"Reward: {reward:.2f}")
            
            # Log to file
            step_data = {
                'timestep': env.current_step,
                'datetime': current_time,
                'soc': soc,
                'power_kw': power_kw,
                'price': current_price,
                'action': actions[0] if actions else 0,
                'reward': reward
            }
            log_step(step_data, output_dir)
    
    # Generate and save final plots
    env.generate_summary()
    plt.savefig(f"{output_dir}/summary_plot.png")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(f"{output_dir}/simulation_metrics.csv", index=False)
    
    print("\n=== Simulation Complete ===")
    print(f"Final SoC: {metrics['soc'][-1]*100:.1f}%")
    print(f"Total Energy Charged: {metrics['total_energy_charged_kwh'][-1]:.2f} kWh")
    print(f"Total Energy Discharged: {metrics['total_energy_discharged_kwh'][-1]:.2f} kWh")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    run_residential_v2g(save_replay=True)  # Set to False to disable replay saving