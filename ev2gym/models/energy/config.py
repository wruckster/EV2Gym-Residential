# File: models/energy/config.py
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Tuple # Add Tuple
import numpy as np

@dataclass
class SimulationConfig:
    timescale: int  # minutes per timestep
    simulation_days: int
    start_date: str
    controller_mode: Optional[str] = None
    controller_config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HomeConfig:
    base_load_profile: str
    average_daily_consumption: float  # kWh/day
    voltage: float
    phases: int

@dataclass 
class PVConfig:
    capacity: float  # kWp
    efficiency: float
    tilt: float  # degrees
    azimuth: float  # degrees

@dataclass
class BatteryConfig:
    capacity: float  # kWh
    max_charge_rate: float  # kW
    max_discharge_rate: float  # kW
    min_soc: float  # 0-1
    max_soc: float  # 0-1
    efficiency_charge: float  # 0-1
    efficiency_discharge: float  # 0-1
    initial_soc: float  # 0-1

@dataclass
class PricingConfig:
    import_rate: float  # €/kWh
    export_rate: float  # €/kWh
    # Parsed TOU rates: {("HH:MM", "HH:MM"): rate}
    parsed_tou_rates: Optional[Dict[Tuple[str, str], float]] = None 
    raw_tou_config: Optional[Dict[str, Any]] = None # Keep raw for other uses if needed

@dataclass
class LoggingConfig:
    level: str
    output_dir: str
    save_plots: bool

class ResidentialEnergyConfig:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.raw_config = self._load_config()
        self.simulation = self._init_simulation_config()
        self.home = self._init_home_config()
        self.pv = self._init_pv_config()
        self.battery = self._init_battery_config()
        self.pricing = self._init_pricing_config()
        self.logging = self._init_logging_config()

    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _init_simulation_config(self) -> SimulationConfig:
        sim = self.raw_config['simulation']
        return SimulationConfig(
            timescale=sim['timescale'],
            simulation_days=sim['simulation_days'],
            start_date=sim['start_date'],
            controller_mode=sim.get('controller_mode'),
            controller_config=sim.get('controller_config', {})
        )

    def _init_home_config(self) -> HomeConfig:
        home = self.raw_config['home']
        return HomeConfig(
            base_load_profile=home['base_load_profile'],
            average_daily_consumption=home['average_daily_consumption'],
            voltage=home['voltage'],
            phases=home['phases']
        )

    def _init_pv_config(self) -> PVConfig:
        pv = self.raw_config['pv']
        return PVConfig(
            capacity=pv['capacity'],
            efficiency=pv['efficiency'],
            tilt=pv['tilt'],
            azimuth=pv['azimuth']
        )

    def _init_battery_config(self) -> BatteryConfig:
        bat = self.raw_config['home_battery']
        return BatteryConfig(
            capacity=bat['capacity'],
            max_charge_rate=bat['max_charge_rate'],
            max_discharge_rate=bat['max_discharge_rate'],
            min_soc=bat['min_soc'],
            max_soc=bat['max_soc'],
            efficiency_charge=bat['efficiency_charge'],
            efficiency_discharge=bat['efficiency_discharge'],
            initial_soc=bat['initial_soc']
        )

    def _parse_tou_rates(self, tou_config: Optional[Dict[str, Any]]) -> Optional[Dict[Tuple[str, str], float]]:
        if not tou_config:
            return None
        parsed_rates = {}
        # Expected format in YAML: {"HH:MM-HH:MM": rate, ...} or {"period_name": {"time_windows": [["HH:MM", "HH:MM"]], "rate": X}, ...}
        # This parser handles the simple format: {"HH:MM-HH:MM": rate}
        for time_range_str, rate in tou_config.items():
            if isinstance(rate, (int, float)) and isinstance(time_range_str, str) and '-' in time_range_str:
                try:
                    start_str, end_str = time_range_str.split('-')
                    # Basic validation for HH:MM format
                    if len(start_str) == 5 and start_str[2] == ':' and len(end_str) == 5 and end_str[2] == ':':
                         # Further validation (e.g. valid hours/minutes) could be added here
                        parsed_rates[(start_str.strip(), end_str.strip())] = float(rate)
                    # else: print(f"Warning: Invalid time range format {time_range_str}")
                except ValueError:
                    # print(f"Warning: Could not parse time range {time_range_str}")
                    continue
            # else: print(f"Warning: Skipping invalid TOU entry {time_range_str}: {rate}")
        return parsed_rates if parsed_rates else None

    def _init_pricing_config(self) -> PricingConfig:
        pricing_data = self.raw_config['pricing']
        raw_tou = pricing_data.get('time_of_use')
        parsed_tou = self._parse_tou_rates(raw_tou)
        return PricingConfig(
            import_rate=pricing_data['import_rate'],
            export_rate=pricing_data['export_rate'],
            parsed_tou_rates=parsed_tou,
            raw_tou_config=raw_tou
        )

    def _init_logging_config(self) -> LoggingConfig:
        log = self.raw_config['logging']
        return LoggingConfig(
            level=log['level'],
            output_dir=log['output_dir'],
            save_plots=log['save_plots']
        )