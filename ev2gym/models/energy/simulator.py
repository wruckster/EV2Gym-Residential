# File: models/energy/simulator.py
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from ev2gym.models.energy.config import ResidentialEnergyConfig
from ev2gym.mpc.residential_mpc import RuleBasedController

@dataclass
class TimeSeriesData:
    timestamps: List[datetime]
    values: np.ndarray

    def __post_init__(self):
        if len(self.timestamps) != len(self.values):
            raise ValueError("Timestamps and values must have the same length")

class ResidentialEnergySimulator:
    def __init__(self, config_path: str):
        self.config = ResidentialEnergyConfig(config_path)
        self.controller = RuleBasedController() # Default mode, can be overridden by config
        # Pass controller-specific configurations from the main YAML
        if self.config.simulation.controller_config:
            self.controller.set_config(self.config.simulation.controller_config)
        if self.config.simulation.controller_mode: # Explicit mode from simulation config overrides
            self.controller.mode = self.config.simulation.controller_mode
        self._setup_simulation()
        
    def _setup_simulation(self):
        """Initialize simulation parameters and data structures"""
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
        
        # Initialize battery SoC in kWh
        self.battery_soc.values[0] = self.config.battery.initial_soc * self.config.battery.capacity
        
        # Generate load and PV profiles
        self._generate_load_profile()
        self._generate_pv_profile()
        
    def _generate_load_profile(self):
        """Generate a synthetic load profile based on configuration"""
        # This is a simplified version - in practice, you'd use real data or more sophisticated models
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
        for day in range(self.config.simulation.simulation_days):
            start_idx = day * len(base_load)
            end_idx = (day + 1) * len(base_load)
            self.load.values[start_idx:end_idx] = base_load
            
    def _generate_pv_profile(self):
        """Generate PV generation profile based on configuration"""
        # Simplified PV generation model - in practice, use pvlib or similar
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
        battery = self.config.battery
        dt_hours = self.config.simulation.timescale / 60  # Convert minutes to hours
        
        current_soc = self.battery_soc.values[timestep]
        
        if power > 0:  # Charging
            # Maximum power limited by charge rate and available capacity
            max_charge_power = min(
                battery.max_charge_rate,
                (battery.max_soc * battery.capacity - current_soc) / 
                (dt_hours * battery.efficiency_charge)
            )
            actual_power = min(power, max_charge_power)
            energy_stored = actual_power * dt_hours * battery.efficiency_charge
            new_soc = current_soc + energy_stored
            
        elif power < 0:  # Discharging
            # Maximum power limited by discharge rate and available energy
            max_discharge_power = min(
                battery.max_discharge_rate,
                (current_soc - battery.min_soc * battery.capacity) * 
                battery.efficiency_discharge / dt_hours
            )
            actual_power = max(power, -max_discharge_power)
            energy_released = -actual_power * dt_hours / battery.efficiency_discharge
            new_soc = current_soc - energy_released
            
        else:  # No power
            actual_power = 0
            new_soc = current_soc
            
        # Update next timestep's SOC
        if timestep + 1 < len(self.battery_soc.values):
            self.battery_soc.values[timestep + 1] = new_soc
            
        return actual_power
        
    def run_simulation(self):
        """Run the energy simulation"""
        dt_hours = self.config.simulation.timescale / 60.0
        min_soc_kWh = self.config.battery.min_soc * self.config.battery.capacity
        max_soc_kWh = self.config.battery.max_soc * self.config.battery.capacity

        for i in range(len(self.timestamps) - 1):
            current_time = self.timestamps[i]
            load_kW = self.load.values[i]
            pv_kW = self.pv_generation.values[i]
            current_soc_kWh = self.battery_soc.values[i]

            battery_target_power_kW = self.controller.compute_battery_action(
                current_time=current_time,
                load_power_kW=load_kW,
                pv_power_kW=pv_kW,
                current_soc_kWh=current_soc_kWh,
                battery_capacity_kWh=self.config.battery.capacity,
                max_charge_rate_kW=self.config.battery.max_charge_rate,
                max_discharge_rate_kW=self.config.battery.max_discharge_rate,
                min_soc_kWh=min_soc_kWh,
                max_soc_kWh=max_soc_kWh,
                import_price_per_kWh=self.config.pricing.import_rate,
                export_price_per_kWh=self.config.pricing.export_rate,
                tou_prices_per_kWh=self.config.pricing.parsed_tou_rates,
                dt_hours=dt_hours
            )
            
            # Update battery and get actual power used
            actual_battery_power = self._update_battery(battery_target_power_kW, i)
            
            # Calculate grid import/export
            # net_load was load - pv. actual_battery_power is positive for charge, negative for discharge.
            # So, grid_net = (load - pv) - (-actual_battery_power_if_discharging) = load - pv + discharge_power
            # grid_net = (load - pv) - (actual_battery_power_if_charging) = load - pv - charge_power
            # This means if battery charges, it adds to the load from the grid's perspective.
            # If battery discharges, it subtracts from the load from the grid's perspective.
            grid_net = (self.load.values[i] - self.pv_generation.values[i]) + actual_battery_power
            if grid_net > 0:
                self.grid_import.values[i] = grid_net
                self.grid_export.values[i] = 0
            else:
                self.grid_import.values[i] = 0
                self.grid_export.values[i] = -grid_net
                
    def calculate_costs(self) -> Dict[str, float]:
        """Calculate total costs and savings"""
        total_import = self.grid_import.values.sum() * (self.config.simulation.timescale / 60)
        total_export = self.grid_export.values.sum() * (self.config.simulation.timescale / 60)
        
        return {
            'import_cost': total_import * self.config.pricing.import_rate,
            'export_income': total_export * self.config.pricing.export_rate,
            'net_cost': (total_import * self.config.pricing.import_rate - 
                        total_export * self.config.pricing.export_rate)
        }
        
    def plot_results(self, save_path: Optional[str] = None):
        """Plot simulation results"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        
        # Format x-axis
        locator = mdates.AutoDateLocator()
        formatter = mdates.DateFormatter('%H:%M\n%Y-%m-%d')
        
        # Plot 1: Generation and Consumption
        ax1.set_title('Energy Generation and Consumption')
        ax1.plot(self.timestamps, self.load.values, 'r-', label='Load')
        ax1.plot(self.timestamps, self.pv_generation.values, 'y-', label='PV Generation')
        ax1.fill_between(self.timestamps, 0, self.load.values, color='red', alpha=0.2)
        ax1.fill_between(self.timestamps, 0, self.pv_generation.values, color='yellow', alpha=0.2)
        ax1.set_ylabel('Power (kW)')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Battery SOC
        ax2.set_title('Battery State of Charge')
        ax2.plot(self.timestamps, 
                self.battery_soc.values / self.config.battery.capacity * 100, 
                'g-')
        ax2.set_ylabel('State of Charge (%)')
        ax2.set_ylim(0, 100)
        ax2.grid(True)
        
        # Plot 3: Grid Import/Export
        ax3.set_title('Grid Interaction')
        ax3.plot(self.timestamps, self.grid_import.values, 'b-', label='Grid Import')
        ax3.plot(self.timestamps, -self.grid_export.values, 'g-', label='Grid Export')
        ax3.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax3.set_ylabel('Power (kW)')
        ax3.set_xlabel('Time')
        ax3.legend()
        ax3.grid(True)
        
        # Format x-axis
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
    def save_results(self, output_dir: str):
        """Save simulation results to CSV"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create DataFrame with all results
        df = pd.DataFrame({
            'timestamp': self.timestamps,
            'load_kw': self.load.values,
            'pv_generation_kw': self.pv_generation.values,
            'battery_soc_kwh': self.battery_soc.values,
            'grid_import_kw': self.grid_import.values,
            'grid_export_kw': self.grid_export.values
        })
        
        # Save to CSV
        csv_path = output_dir / 'simulation_results.csv'
        df.to_csv(csv_path, index=False)
        
        # Save summary
        summary = {
            'total_load_kwh': self.load.values.sum() * (self.config.simulation.timescale / 60),
            'total_pv_generation_kwh': self.pv_generation.values.sum() * (self.config.simulation.timescale / 60),
            'self_consumption_ratio': (self.pv_generation.values.sum() - self.grid_export.values.sum()) / 
                                    self.pv_generation.values.sum() if self.pv_generation.values.sum() > 0 else 0,
            **self.calculate_costs()
        }
        
        with open(output_dir / 'summary.json', 'w') as f:
            import json
            json.dump(summary, f, indent=2)
            
        # Save plot
        self.plot_results(str(output_dir / 'simulation_plot.png'))
        
        return csv_path