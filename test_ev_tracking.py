import numpy as np
from ev2gym.models import EV2Gym
from ev2gym.visuals.evaluator_plot import plot_from_replay
import os
import matplotlib.pyplot as plt
import pickle

def test_ev_tracking():
    # Initialize environment with workplace scenario and replay saving
    env = EV2Gym(
        config_file='ev2gym/example_config_files/residential_v2g.yaml',
        save_replay=True,
        replay_save_path='results/ev_tracking_test/',
        lightweight_plots=False,  # Enable full data recording
        verbose=True
    )
    
    # Reset environment
    obs = env.reset()
    print(f"\nInitial state:")
    print(f"- Active EVs: {len(env.EVs)}")
    print(f"- Charging stations: {len(env.charging_stations)}")
    for cs in env.charging_stations:
        connected_evs = [ev for ev in cs.evs_connected if ev is not None]  # Filter out None values
        if connected_evs:  # Check if there are any non-None connected EVs
            socs = [ev.get_soc() * 100 for ev in connected_evs]
            avg_soc = sum(socs) / len(socs)
            print(f"  * CS {cs.id}: {len(connected_evs)}/{cs.n_ports} EVs connected at avg SoC {avg_soc:.1f}%")
        else:
            print(f"  * CS {cs.id}: 0/{cs.n_ports} EVs connected")
    
    # Track EV states over time
    metrics = {
        'timestep': [],
        'datetime': [],
        'demand_response': [],
        'total_pv_generation': [],
        'total_power_usage': [],
        'evs': {},
        'charging_stations': {}
    }
    
    # Initialize EV tracking
    for ev in env.EVs:
        metrics['evs'][ev.id] = {
            'soc': [],
            'location': [],
            'charging_power': [],
            'discharging_power': []
        }
    
    # Initialize charging station tracking
    for cs in env.charging_stations:
        metrics['charging_stations'][cs.id] = {
            'total_charging_power': [],
            'total_discharging_power': [],
            'connected_evs': []
        }
    
    # Run for 96 timesteps (one day)
    rewards = []
    for step in range(env.simulation_length - 1):
        # Take random action (scaled to be mostly charging)
        action = env.action_space.sample() * 0.8 + 0.2  # Bias towards charging
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        
        # Store metrics for this timestep
        metrics['timestep'].append(step)
        metrics['datetime'].append(step)  # Store step as datetime for now
        metrics['demand_response'].append(info.get('demand_response', 0))
        metrics['total_pv_generation'].append(info.get('pv_generation', 0))
        metrics['total_power_usage'].append(info.get('total_power_usage', 0))
        
        # Track metrics for each EV
        for ev_id, ev in enumerate(env.EVs):
            # Check if EV is connected to any charging station
            is_connected = any(ev in cs.evs_connected for cs in env.charging_stations if cs.evs_connected is not None)
            
            if is_connected:
                # Get the charging station this EV is connected to
                cs = next((cs for cs in env.charging_stations if ev in cs.evs_connected), None)
                current_energy = ev.current_energy if hasattr(ev, 'current_energy') else 0
                
                # Update EV metrics
                if ev_id not in metrics['evs']:
                    metrics['evs'][ev_id] = {
                        'soc': [],
                        'location': [],
                        'charging_power': [],
                        'discharging_power': []
                    }
                
                metrics['evs'][ev_id]['soc'].append(ev.get_soc() * 100)  # Convert to percentage
                metrics['evs'][ev_id]['location'].append(ev.location if hasattr(ev, 'location') else 'unknown')
                metrics['evs'][ev_id]['charging_power'].append(current_energy)
                metrics['evs'][ev_id]['discharging_power'].append(0)  # Add discharging power
                
            else:
                # Update EV metrics for disconnected EVs
                if ev_id not in metrics['evs']:
                    metrics['evs'][ev_id] = {
                        'soc': [],
                        'location': [],
                        'charging_power': [],
                        'discharging_power': []
                    }
                
                metrics['evs'][ev_id]['soc'].append(ev.get_soc() * 100)  # Convert to percentage
                metrics['evs'][ev_id]['location'].append(ev.location if hasattr(ev, 'location') else 'unknown')
                metrics['evs'][ev_id]['charging_power'].append(0)
                metrics['evs'][ev_id]['discharging_power'].append(0)
        
        # Track charging station metrics
        for cs in env.charging_stations:
            total_charging = 0
            total_discharging = 0
            connected_evs = []
            
            for ev in cs.evs_connected:
                if ev is not None:
                    connected_evs.append(ev.id)
                    current_energy = ev.current_energy if hasattr(ev, 'current_energy') else 0
                    if current_energy > 0:
                        total_charging += current_energy
                    else:
                        total_discharging += abs(current_energy)
            
            metrics['charging_stations'][cs.id]['total_charging_power'].append(total_charging)
            metrics['charging_stations'][cs.id]['total_discharging_power'].append(total_discharging)
            metrics['charging_stations'][cs.id]['connected_evs'].append(connected_evs)
        
        # Print status every 12 steps (every 3 hours)
        if step % 12 == 0 or step == 0:
            # Convert step to hours and minutes for display
            hours = step // 4
            minutes = (step % 4) * 15
            time_str = f"{hours:02d}:{minutes:02d}"
            
            print(f"\n--- Step {step} ({time_str}) ---")
            print(f"Demand Response: {metrics['demand_response'][-1]:.2f} kW")
            print(f"PV Generation: {metrics['total_pv_generation'][-1]:.2f} kW")
            print(f"Total Power Usage: {metrics['total_power_usage'][-1]:.2f} kW")
            
            # Print EV status
            print("\nEV Status:")
            for ev_id, ev_data in metrics['evs'].items():
                print(f"  EV {ev_id}:")
                print(f"    - SoC: {ev_data['soc'][-1]:.1f}%")
                print(f"    - Location: {ev_data['location'][-1]}")
                print(f"    - Charging: {ev_data['charging_power'][-1]:.2f} kW")
                print(f"    - Discharging: {ev_data['discharging_power'][-1]:.2f} kW")
            
            # Print charging station status
            print("\nCharging Stations:")
            for cs_id, cs_data in metrics['charging_stations'].items():
                print(f"  CS {cs_id}:")
                print(f"    - Connected EVs: {len(cs_data['connected_evs'][-1])}")
                print(f"    - Total Charging: {cs_data['total_charging_power'][-1]:.2f} kW")
                print(f"    - Total Discharging: {cs_data['total_discharging_power'][-1]:.2f} kW")
    
    print("\nSimulation complete!")
    print(f"Average reward: {np.mean(rewards):.2f}")
    
    # Save metrics to file for further analysis
    with open('results/ev_tracking_test/simulation_metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    
    print("\nSimulation complete. Metrics saved to results/ev_tracking_test/simulation_metrics.pkl")
    
    # Save replay file and get path
    replay_path = env._save_sim_replay()
    print(f"\nReplay saved to: {replay_path}")
    
    # Load replay from file
    with open(replay_path, 'rb') as f:
        replay = pickle.load(f)
    
    # Print power usage debug info
    print("\nPower Usage Summary:")
    print("-------------------")
    print(f"Total power usage shape: {replay.current_power_usage.shape}")
    print(f"Non-zero power steps: {np.count_nonzero(replay.current_power_usage)}")
    if hasattr(replay, 'cs_power'):
        print("\nCharging Station Power:")
        cs_power = replay.cs_power
        for cs_idx in range(cs_power.shape[0]):
            nonzero_steps = np.nonzero(cs_power[cs_idx])[0]
            if len(nonzero_steps) > 0:
                print(f"CS {cs_idx}: {len(nonzero_steps)} active steps, last active at step {nonzero_steps[-1]}")
            else:
                print(f"CS {cs_idx}: No power usage recorded")
    
    plot_path = os.path.join('results/ev_tracking_test', 'ev_trajectory.png')
    
    # Create figure with larger size and higher DPI
    plt.figure(figsize=(15, 20))
    plot_from_replay(replay_path, plot_type='replays', save_path=plot_path)
    print(f"\nVisualization saved to {plot_path}")

    return metrics

if __name__ == '__main__':
    # Create results directory if it doesn't exist
    os.makedirs('results/ev_tracking_test', exist_ok=True)
    test_ev_tracking()
