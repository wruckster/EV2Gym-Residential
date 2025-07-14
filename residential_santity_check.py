import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.utilities.loaders import _load_household_profiles, _load_external_features, generate_residential_inflexible_loads, generate_pv_generation

def test_nsw_profiles(config_path="ev2gym/example_config_files/residential_v2g.yaml"):
    # Load the config to check the data files
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=== Testing NSW Profile Loading with Date Filtering ===")
    print(f"Config file: {os.path.abspath(config_path)}")
    
    # Display date filtering parameters from config
    year = config.get('year', 2019)
    month = config.get('month', 1)
    day = config.get('day', 1)
    timescale = config.get('timescale', 15)
    simulation_length = config.get('simulation_length', 96)
    
    print(f"\nDate Filtering Parameters:")
    print(f"  - Start Date: {year}-{month:02d}-{day:02d}")
    print(f"  - Timescale: {timescale} minutes")
    print(f"  - Simulation Length: {simulation_length} steps")
    print(f"  - Total Duration: {timescale * simulation_length / 60:.1f} hours")
    
    print(f"\nData Sources:")
    print(f"  - Household data files: {config.get('inflexible_loads', {}).get('data_files', [])}")
    print(f"  - Use household CSV for solar: {config.get('solar_power', {}).get('data_from_household_csv', False)}")
    
    # Initialize the environment
    print("\nInitializing environment...")
    env = EV2Gym(
        config_file=config_path,
        save_replay=False,
        save_plots=False,
    )
    
    # Debug: First directly load the CSV data using pandas to verify contents
    print("\n=== Directly Loading CSV ===")
    csv_path = env.config['inflexible_loads']['data_files'][0]
    raw_csv = pd.read_csv(csv_path, parse_dates=['interval_start'])
    
    # Extract time-filtered data to match our simulation window
    year = env.config.get('year', 2019)
    month = env.config.get('month', 1)
    day = env.config.get('day', 1)
    hour = env.config.get('hour', 0)
    minute = env.config.get('minute', 0)
    
    # Create start date with all components including hour and minute
    start_date = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute)
    
    # Calculate end date based on simulation_length and timescale
    minutes_to_add = env.timescale * env.simulation_length
    end_date = start_date + pd.Timedelta(minutes=minutes_to_add)
    
    print(f"Filtering data from {start_date} to {end_date}")
    
    # Find rows in raw CSV that match our window
    filtered_csv = raw_csv[(raw_csv['interval_start'] >= start_date) & 
                           (raw_csv['interval_start'] <= end_date)]
    
    if not filtered_csv.empty:
        print(f"Found {len(filtered_csv)} matching rows in raw CSV")
        print("First 5 rows of raw CSV data:")
        print(filtered_csv[['interval_start', 'demand', 'solar']].head())
    else:
        print("WARNING: No matching rows found in raw CSV!")
    
    # Now use the proper loader function to get the processed data
    print("\n=== Using _load_household_profiles ===")
    household_df = _load_household_profiles(env)
    if household_df is not None:
        print(f"Household DataFrame loaded successfully. Shape: {household_df.shape}")
        print("First 5 rows of household_df:")
        print(household_df.head())
        
        # Generate the loads that would be passed to transformers
        print("\n=== Generating Transformer Data ===")
        inflexible_loads = generate_residential_inflexible_loads(env)
        solar_power = generate_pv_generation(env)
        
        print(f"Inflexible loads shape: {inflexible_loads.shape}")
        print(f"Solar power shape: {solar_power.shape}")
        
        print("\n=== Direct Comparison ===")
        print("Timestamp, CSV Demand, Transformer Load, Ratio, CSV Solar, Transformer Solar, Ratio")
        for i in range(min(10, len(household_df))):
            csv_demand = household_df['demand'].iloc[i]
            sim_demand = inflexible_loads[0, i]
            demand_ratio = sim_demand / csv_demand if csv_demand != 0 else "N/A"
            
            csv_solar = household_df['solar'].iloc[i]
            sim_solar = solar_power[0, i]
            solar_ratio = sim_solar / csv_solar if csv_solar != 0 else "N/A"
            
            print(f"{i}, {csv_demand:.4f}, {sim_demand:.4f}, {demand_ratio}, {csv_solar:.4f}, {sim_solar:.4f}, {solar_ratio}")
    else:
        print("No household data loaded!")
    
    # Reset the environment
    print("\n=== Starting Simulation ===")
    obs = env.reset()
    
    # Display actual simulation date
    print(f"\nActual Simulation Date: {env.sim_date}")
    
    # Check if household profiles are loaded correctly
    print("\nChecking household profiles...")
    household_df = _load_household_profiles(env)
    if household_df is not None:
        print(f"  - Loaded household data with {len(household_df)} rows")
        print(f"  - Columns: {household_df.columns.tolist()}")
        print(f"  - Demand range: {household_df['demand'].min():.2f} to {household_df['demand'].max():.2f}")
        if 'solar' in household_df.columns:
            print(f"  - Solar range: {household_df['solar'].min():.2f} to {household_df['solar'].max():.2f}")
    else:
        print("  - No household data available")
    
    # Check if external features are loaded correctly
    print("\nChecking external features...")
    external_df = _load_external_features(env)
    if external_df is not None:
        print(f"  - Loaded external features with {len(external_df)} rows")
        print(f"  - Columns: {external_df.columns.tolist()}")
    else:
        print("  - No external features available")
    
    # Debug: Print scaling factors from config
    print("\nDEBUG - Scaling factors from config:")
    demand_scale = env.config.get('inflexible_loads', {}).get('scale_mean', 1.0)
    demand_multiplier = env.config.get('inflexible_loads', {}).get('inflexible_loads_capacity_multiplier_mean', 1.0)
    solar_scale = env.config.get('solar_power', {}).get('solar_power_capacity_multiplier_mean', 1.0)
    print(f"  - Demand scale_mean: {demand_scale}")
    print(f"  - Demand capacity multiplier: {demand_multiplier}")
    print(f"  - Solar capacity multiplier: {solar_scale}")
    
    # Debug: Check if there are any other multipliers in the transformer code
    print("\nDEBUG - Checking for additional multipliers in transformer initialization:")
    print(f"  - Transformer random seed: {env.tr_seed}")
    
    # Debug: Generate inflexible loads and solar power directly
    print("\nDEBUG - Direct generation of inflexible loads and solar power:")
    inflexible_loads = generate_residential_inflexible_loads(env)
    solar_power = generate_pv_generation(env)
    
    # Print the first few values
    print("  - Inflexible loads (first 5 steps):")
    for i in range(min(5, inflexible_loads.shape[1])):
        print(f"    Step {i}: {inflexible_loads[0, i]:.6f} kW")
    
    print("  - Solar power (first 5 steps):")
    for i in range(min(5, solar_power.shape[1])):
        print(f"    Step {i}: {solar_power[0, i]:.6f} kW")
    
    # Run a few steps
    print("\nRunning 10 steps...")
    for i in range(10):
        action = env.action_space.sample()  # Random action
        obs, reward, done, truncated, info = env.step(action)
        
        # Print some info
        print(f"\nStep {i+1}:")
        print(f"  - Sim datetime: {env.sim_date}")
        print(f"  - Inflexible load: {env.transformers[0].inflexible_load[i]:.2f} kW")
        print(f"  - Solar power: {env.transformers[0].solar_power[i]:.2f} kW")
        print(f"  - Reward: {reward:.4f}")
        print(f"  - Done: {done}, Truncated: {truncated}")
        
        if done or truncated:
            print("Episode finished early!")
            break
    
    # After running steps, let's plot the loaded profiles
    print("\nPlotting loaded profiles...")
    
    # Create time index based on simulation start time and timescale
    timescale_minutes = env.config.get('timescale', 15)
    
    # Use the environment's simulation date directly
    time_index = pd.date_range(
        start=env.sim_date, 
        periods=100, 
        freq=f'{timescale_minutes}min'
    )
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # Plot inflexible loads
    ax = axes[0]
    ax.plot(time_index, env.transformers[0].inflexible_load[:100], label='Inflexible Load')
    ax.set_title(f'First 100 Steps of Inflexible Load (Starting {time_index[0].strftime("%Y-%m-%d %H:%M")})')
    ax.set_ylabel('Power (kW)')
    ax.legend()
    ax.grid(True)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot solar power
    ax = axes[1]
    ax.plot(time_index, env.transformers[0].solar_power[:100], label='Solar Power')
    ax.set_title(f'First 100 Steps of Solar Power (Starting {time_index[0].strftime("%Y-%m-%d %H:%M")})')
    ax.set_ylabel('Power (kW)')
    ax.legend()
    ax.grid(True)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot electricity prices if available
    ax = axes[2]
    if hasattr(env, 'charge_prices') and hasattr(env, 'discharge_prices'):
        ax.plot(time_index, -env.charge_prices[0, :100], label='Charge Price')
        ax.plot(time_index, env.discharge_prices[0, :100], label='Discharge Price')
        ax.set_title(f'First 100 Steps of Electricity Prices (Starting {time_index[0].strftime("%Y-%m-%d %H:%M")})')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price (EUR/kWh)')
        ax.legend()
        ax.grid(True)
    else:
        ax.plot(time_index, np.zeros(100), label='No Price Data')
        ax.set_title('Electricity Prices (Not Available)')
        ax.set_xlabel('Time')
        ax.legend()
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    
    # Save the plot
    output_file = 'residential_sanity_check_results.png'
    plt.savefig(output_file, bbox_inches='tight')
    print(f"\nPlot saved as '{output_file}'")
    
    print("\n=== Test Complete ===")
    
    return env

if __name__ == "__main__":
    # You can specify a different config path if needed
    test_nsw_profiles()