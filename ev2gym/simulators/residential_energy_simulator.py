"""
Residential Energy Simulator for EV2Gym.

This module provides the ResidentialEnergySimulator class for simulating residential energy systems
with support for various control strategies.
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yaml
import os

from ev2gym.models.energy.config import ResidentialEnergyConfig
from ev2gym.mpc.residential_mpc import RuleBasedController

@dataclass
class TimeSeriesData:
    """Class for storing time series data with timestamps."""
    timestamps: List[datetime]
    values: np.ndarray

    def __post_init__(self):
        if len(self.timestamps) != len(self.values):
            raise ValueError("Timestamps and values must have the same length")

class ResidentialEnergySimulator:
    """Simulator for residential energy systems with configurable control strategies."""
    
    def __init__(self, config_path: str):
        """Initialize the simulator with the given configuration file.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config = ResidentialEnergyConfig(config_path)
        self.controller = RuleBasedController()  # Default mode, can be overridden by config
        
        # Pass controller-specific configurations from the main YAML
        if self.config.simulation.controller_config:
            self.controller.set_config(self.config.simulation.controller_config)
        if self.config.simulation.controller_mode:  # Explicit mode from simulation config overrides
            self.controller.mode = self.config.simulation.controller_mode
            
        # Pass TOU rates to controller if available
        if hasattr(self.config.pricing, 'parsed_tou_rates'):
            self.controller.tou_rates = self.config.pricing.parsed_tou_rates
            
        self._setup_simulation()
        
    def _setup_simulation(self):
        """Setup the simulation environment based on the configuration."""
        # Create time index
        self.timestep = timedelta(minutes=self.config.simulation.timescale)
        total_steps = int(24 * 60 / self.config.simulation.timescale * 
                         self.config.simulation.simulation_days)
        
        start_time = datetime.strptime(self.config.simulation.start_date, "%Y-%m-%d")
        self.timestamps = [start_time + i * self.timestep 
                          for i in range(total_steps)]
        
        # Initialize data storage
        self.load = TimeSeriesData(self.timestamps, np.zeros(total_steps))
        self.pv_generation = TimeSeriesData(self.timestamps, np.zeros(total_steps))
        self.battery_soc = TimeSeriesData(self.timestamps, np.zeros(total_steps))
        self.grid_import = TimeSeriesData(self.timestamps, np.zeros(total_steps))
        self.grid_export = TimeSeriesData(self.timestamps, np.zeros(total_steps))
        self.battery_power = TimeSeriesData(self.timestamps, np.zeros(total_steps))
        
        # Initialize battery SoC in kWh
        self.battery_soc.values[0] = self.config.battery.initial_soc * self.config.battery.capacity
        
        # Generate load and PV profiles
        self._generate_load_profile()
        self._generate_pv_profile()
        
    def _generate_load_profile(self):
        """Generate a synthetic load profile based on configuration"""
        daily_kwh = self.config.home.average_daily_consumption
        base_load = np.array([
            0.3, 0.25, 0.2, 0.18, 0.15, 0.18, 0.25, 0.4,  # 00:00-07:45
            0.5, 0.6, 0.65, 0.7, 0.7, 0.68, 0.65, 0.7,    # 08:00-15:45
            0.8, 1.0, 1.2, 1.1, 0.9, 0.7, 0.5, 0.3         # 16:00-23:45
        ])
        
        # Scale to match daily consumption
        scale = daily_kwh / (base_load.sum() * (self.config.simulation.timescale / 60))
        base_load = base_load * scale
        
        # Repeat for each day
        steps_per_day = int(24 * 60 / self.config.simulation.timescale)
        for day in range(self.config.simulation.simulation_days):
            start_idx = day * steps_per_day
            end_idx = (day + 1) * steps_per_day
            self.load.values[start_idx:end_idx] = base_load[:steps_per_day]
            
    def _generate_pv_profile(self):
        """Generate PV generation profile based on configuration"""
        # Simplified PV generation model
        hours = np.array([t.hour + t.minute/60 for t in self.timestamps])
        
        for i, (timestamp, hour) in enumerate(zip(self.timestamps, hours)):
            # Simple sinusoidal model - max at solar noon
            solar_noon = 12  # Simplified
            hour_angle = (hour - solar_noon) * 15  # degrees
            hour_angle = np.radians(hour_angle)
            
            # Simple clear-sky model
            solar_elevation = (90 - abs(hour_angle))  # Simplified
            if solar_elevation > 0:
                # Basic clear-sky model
                generation = (self.config.pv.capacity * 
                            np.sin(np.radians(solar_elevation)) * 
                            self.config.pv.efficiency)
                # Add some randomness
                generation *= np.random.uniform(0.8, 1.0)
                self.pv_generation.values[i] = max(0, generation)
                
    def _update_battery(self, power: float, timestep: int) -> float:
        """Update battery state of charge and return actual power used"""
        if power == 0:
            return 0
            
        # Calculate energy in kWh for this timestep (power in kW * time in hours)
        energy_kwh = power * (self.config.simulation.timescale / 60)
        
        if power > 0:  # Charging
            # Calculate maximum charge power considering efficiency
            max_charge_energy = (self.config.battery.capacity - 
                              self.battery_soc.values[timestep-1]) / self.config.battery.charging_efficiency
            actual_energy = min(energy_kwh, max_charge_energy)
            self.battery_soc.values[timestep] = (self.battery_soc.values[timestep-1] + 
                                              actual_energy * self.config.battery.charging_efficiency)
            return actual_energy * (60 / self.config.simulation.timescale)  # Convert back to power
            
        else:  # Discharging
            # Calculate maximum discharge power considering efficiency
            max_discharge_energy = (self.battery_soc.values[timestep-1] * 
                                 self.config.battery.discharging_efficiency)
            actual_energy = max(energy_kwh, -max_discharge_energy)
            self.battery_soc.values[timestep] = (self.battery_soc.values[timestep-1] + 
                                              actual_energy / self.config.battery.discharging_efficiency)
            return actual_energy * (60 / self.config.simulation.timescale)  # Convert back to power
    
    def run_simulation(self):
        """Run the simulation for the configured period"""
        for t in range(1, len(self.timestamps)):
            # Get current state
            current_time = self.timestamps[t]
            load = self.load.values[t]
            pv = self.pv_generation.values[t]
            soc = self.battery_soc.values[t-1] / self.config.battery.capacity  # Normalized SOC
            
            # Get control action from controller
            battery_power = self.controller.compute_action(
                current_time=current_time,
                load=load,
                pv_generation=pv,
                battery_soc=soc,
                battery_capacity=self.config.battery.capacity,
                battery_power_rating=self.config.battery.power_rating
            )
            
            # Update battery state
            actual_power = self._update_battery(battery_power, t)
            self.battery_power.values[t] = actual_power
            
            # Calculate grid flows
            net_load = load - pv - actual_power
            
            if net_load > 0:
                self.grid_import.values[t] = net_load
                self.grid_export.values[t] = 0
            else:
                self.grid_import.values[t] = 0
                self.grid_export.values[t] = -net_load
    
    def save_results(self, output_dir: str):
        """Save simulation results to CSV files"""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create DataFrame with all results
        results = pd.DataFrame({
            'timestamp': self.timestamps,
            'load_kw': self.load.values,
            'pv_generation_kw': self.pv_generation.values,
            'battery_soc_kwh': self.battery_soc.values,
            'battery_power_kw': self.battery_power.values,
            'grid_import_kw': self.grid_import.values,
            'grid_export_kw': self.grid_export.values
        })
        
        # Save to CSV
        results.to_csv(os.path.join(output_dir, 'simulation_results.csv'), index=False)
        
        # Save configuration
        config_dict = {
            'simulation': self.config.simulation.__dict__,
            'battery': self.config.battery.__dict__,
            'pv': self.config.pv.__dict__,
            'home': self.config.home.__dict__,
            'pricing': self.config.pricing.__dict__
        }
        
        with open(os.path.join(output_dir, 'simulation_config.yaml'), 'w') as f:
            yaml.dump(config_dict, f, sort_keys=False)
    
    def plot_results(self, output_dir: str = None):
        """Plot simulation results"""
        plt.figure(figsize=(12, 8))
        
        # Plot load, PV, and battery
        plt.subplot(3, 1, 1)
        plt.plot(self.timestamps, self.load.values, label='Load (kW)', color='blue')
        plt.plot(self.timestamps, self.pv_generation.values, label='PV Generation (kW)', color='orange')
        plt.plot(self.timestamps, self.battery_power.values, label='Battery Power (kW)', color='green')
        plt.legend()
        plt.title('Load, PV Generation, and Battery Power')
        plt.grid(True)
        
        # Plot battery SOC
        plt.subplot(3, 1, 2)
        plt.plot(self.timestamps, self.battery_soc.values, label='Battery SOC (kWh)', color='purple')
        plt.legend()
        plt.title('Battery State of Charge')
        plt.grid(True)
        
        # Plot grid flows
        plt.subplot(3, 1, 3)
        plt.plot(self.timestamps, self.grid_import.values, label='Grid Import (kW)', color='red')
        plt.plot(self.timestamps, -self.grid_export.values, label='Grid Export (kW)', color='green')
        plt.legend()
        plt.title('Grid Flows')
        plt.grid(True)
        
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'simulation_results.png'))
        else:
            plt.show()
        
        plt.close()
