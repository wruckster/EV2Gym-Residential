# File: ev2gym/mpc/residential_mpc.py
from dataclasses import dataclass
from typing import Optional, Literal, Dict, Tuple
import numpy as np
from datetime import datetime, time

@dataclass
class RuleBasedController:
    mode: Literal['self_consumption', 'cost_optimization', 'peak_shaving'] = 'self_consumption'
    peak_shaving_threshold_kW: float = 2.0 # Default, can be overridden by config

    def compute_battery_action(
        self,
        current_time: datetime,
        load_power_kW: float,
        pv_power_kW: float,
        current_soc_kWh: float,
        battery_capacity_kWh: float,
        max_charge_rate_kW: float,
        max_discharge_rate_kW: float,
        min_soc_kWh: float, # Min energy in kWh
        max_soc_kWh: float, # Max energy in kWh
        import_price_per_kWh: float,
        export_price_per_kWh: float,
        tou_prices_per_kWh: Optional[Dict[Tuple[str, str], float]] = None, # Parsed: {("HH:MM", "HH:MM"): price}
        dt_hours: float = 0.25
    ) -> float:
        """
        Compute the optimal battery charge/discharge power setpoint (in kW) for the current timestep.
        
        Selects the control strategy based on the configured mode (self_consumption, cost_optimization, peak_shaving),
        applies the corresponding strategy, and ensures the resulting action is feasible given battery SoC and physical limits.
        
        Args:
            current_time (datetime): Current simulation time.
            load_power_kW (float): House load in kW (positive = consumption).
            pv_power_kW (float): PV generation in kW (positive = production).
            current_soc_kWh (float): Current battery state of charge in kWh.
            battery_capacity_kWh (float): Total battery capacity in kWh.
            max_charge_rate_kW (float): Maximum charging rate in kW.
            max_discharge_rate_kW (float): Maximum discharging rate in kW.
            min_soc_kWh (float): Minimum allowed battery SoC in kWh.
            max_soc_kWh (float): Maximum allowed battery SoC in kWh.
            import_price_per_kWh (float): Grid import price per kWh.
            export_price_per_kWh (float): Grid export price per kWh.
            tou_prices_per_kWh (Optional[Dict[Tuple[str, str], float]]): Optional TOU rates by time range.
            dt_hours (float): Length of timestep in hours.
        Returns:
            float: Battery power setpoint (kW), positive for charging, negative for discharging.
        """
        net_load_kW = load_power_kW - pv_power_kW # Positive if house needs power
        battery_power_target_kW = 0.0

        if self.mode == 'self_consumption':
            battery_power_target_kW = self._self_consumption_strategy(
                net_load_kW, current_soc_kWh, battery_capacity_kWh,
                max_charge_rate_kW, max_discharge_rate_kW, min_soc_kWh, max_soc_kWh, dt_hours
            )
        elif self.mode == 'cost_optimization':
            battery_power_target_kW = self._cost_optimization_strategy(
                current_time, net_load_kW, current_soc_kWh, battery_capacity_kWh,
                max_charge_rate_kW, max_discharge_rate_kW, min_soc_kWh, max_soc_kWh,
                import_price_per_kWh, export_price_per_kWh, tou_prices_per_kWh, dt_hours
            )
        elif self.mode == 'peak_shaving':
            battery_power_target_kW = self._peak_shaving_strategy(
                net_load_kW, current_soc_kWh, battery_capacity_kWh,
                max_charge_rate_kW, max_discharge_rate_kW, min_soc_kWh, max_soc_kWh,
                self.peak_shaving_threshold_kW, dt_hours
            )
        
        # Ensure action is within battery's true capability based on SoC and physical limits
        actual_battery_power_kW = 0.0
        if battery_power_target_kW > 0: # Attempting to charge
            max_power_to_fill_kW = (max_soc_kWh - current_soc_kWh) / dt_hours if dt_hours > 0 else float('inf')
            actual_battery_power_kW = min(battery_power_target_kW, max_charge_rate_kW, max_power_to_fill_kW)
        elif battery_power_target_kW < 0: # Attempting to discharge
            max_power_to_empty_kW = (current_soc_kWh - min_soc_kWh) / dt_hours if dt_hours > 0 else float('inf')
            actual_battery_power_kW = max(battery_power_target_kW, -max_discharge_rate_kW, -max_power_to_empty_kW)
        
        return actual_battery_power_kW

    def _self_consumption_strategy(self, net_load_kW, current_soc_kWh, battery_capacity_kWh,
                                 max_charge_rate_kW, max_discharge_rate_kW, 
                                 min_soc_kWh, max_soc_kWh, dt_hours):
        """
        Strategy to maximize self-consumption of PV generation.
        Discharges battery to cover load when net load is positive, or charges battery with PV surplus when net load is negative.
        
        Args:
            net_load_kW (float): Net house load (load - PV) in kW.
            current_soc_kWh (float): Current battery state of charge in kWh.
            battery_capacity_kWh (float): Total battery capacity in kWh.
            max_charge_rate_kW (float): Maximum charging rate in kW.
            max_discharge_rate_kW (float): Maximum discharging rate in kW.
            min_soc_kWh (float): Minimum allowed battery SoC in kWh.
            max_soc_kWh (float): Maximum allowed battery SoC in kWh.
            dt_hours (float): Length of timestep in hours.
        Returns:
            float: Battery power setpoint (kW), positive for charging, negative for discharging.
        """
        if net_load_kW > 0:
            target_discharge_kW = min(net_load_kW, max_discharge_rate_kW)
            return -target_discharge_kW
        else:
            surplus_pv_kW = -net_load_kW 
            target_charge_kW = min(surplus_pv_kW, max_charge_rate_kW)
            return target_charge_kW

    def _cost_optimization_strategy(self, current_time_dt, net_load_kW, current_soc_kWh, battery_capacity_kWh,
                                  max_charge_rate_kW, max_discharge_rate_kW, 
                                  min_soc_kWh, max_soc_kWh,
                                  default_import_rate, default_export_rate, 
                                  tou_prices_per_kWh, dt_hours):
        """
        Strategy to minimize electricity cost by charging when prices are low and discharging when prices are high.
        Uses TOU price information if available, otherwise defaults to import price.
        
        Args:
            current_time_dt (datetime): Current simulation time.
            net_load_kW (float): Net house load (load - PV) in kW.
            current_soc_kWh (float): Current battery state of charge in kWh.
            battery_capacity_kWh (float): Total battery capacity in kWh.
            max_charge_rate_kW (float): Maximum charging rate in kW.
            max_discharge_rate_kW (float): Maximum discharging rate in kW.
            min_soc_kWh (float): Minimum allowed battery SoC in kWh.
            max_soc_kWh (float): Maximum allowed battery SoC in kWh.
            default_import_rate (float): Default grid import price per kWh.
            default_export_rate (float): Default grid export price per kWh.
            tou_prices_per_kWh (Optional[Dict[Tuple[str, str], float]]): Optional TOU rates by time range.
            dt_hours (float): Length of timestep in hours.
        Returns:
            float: Battery power setpoint (kW), positive for charging, negative for discharging.
        """
        current_import_price = self._get_current_price(current_time_dt, default_import_rate, tou_prices_per_kWh)
        
        if current_import_price > default_import_rate * 1.2 and net_load_kW > 0:
             target_discharge_kW = min(net_load_kW, max_discharge_rate_kW)
             return -target_discharge_kW
        elif current_import_price < default_import_rate * 0.8: # Arbitrary low threshold
            # If price is low, charge from grid up to max_charge_rate_kW,
            # irrespective of net_load (could be charging from grid even if load is low)
            return max_charge_rate_kW 
        return self._self_consumption_strategy(net_load_kW, current_soc_kWh, battery_capacity_kWh,
                                 max_charge_rate_kW, max_discharge_rate_kW, 
                                 min_soc_kWh, max_soc_kWh, dt_hours)

    def _peak_shaving_strategy(self, net_load_kW, current_soc_kWh, battery_capacity_kWh,
                             max_charge_rate_kW, max_discharge_rate_kW, 
                             min_soc_kWh, max_soc_kWh, peak_threshold_kW, dt_hours):
        """
        Strategy to limit household peak power demand by discharging the battery when net load exceeds a threshold.
        
        Args:
            net_load_kW (float): Net house load (load - PV) in kW.
            current_soc_kWh (float): Current battery state of charge in kWh.
            battery_capacity_kWh (float): Total battery capacity in kWh.
            max_charge_rate_kW (float): Maximum charging rate in kW.
            max_discharge_rate_kW (float): Maximum discharging rate in kW.
            min_soc_kWh (float): Minimum allowed battery SoC in kWh.
            max_soc_kWh (float): Maximum allowed battery SoC in kWh.
            peak_threshold_kW (float): Peak power threshold in kW.
            dt_hours (float): Length of timestep in hours.
        Returns:
            float: Battery power setpoint (kW), positive for charging, negative for discharging.
        """
        if net_load_kW > peak_threshold_kW: 
            power_to_shave_kW = net_load_kW - peak_threshold_kW
            target_discharge_kW = min(power_to_shave_kW, max_discharge_rate_kW)
            return -target_discharge_kW
        elif net_load_kW < 0: 
            surplus_pv_kW = -net_load_kW
            target_charge_kW = min(surplus_pv_kW, max_charge_rate_kW)
            return target_charge_kW
        return 0.0

    def _get_current_price(self, current_time_dt: datetime, default_price: float, 
                           tou_prices_per_kWh: Optional[Dict[Tuple[str, str], float]]) -> float:
        """
        Returns the applicable import price for the given time, using TOU rates if provided, otherwise the default price.
        Args:
            current_time_dt (datetime): Current simulation time.
            default_price (float): Default import price per kWh.
            tou_prices_per_kWh (Optional[Dict[Tuple[str, str], float]]): Optional TOU rates by time range.
        Returns:
            float: Import price per kWh for the current time.
        """

        if not tou_prices_per_kWh:
            return default_price
        
        current_t = current_time_dt.time()
        for (start_str, end_str), rate_val in tou_prices_per_kWh.items():
            try:
                start_h, start_m = map(int, start_str.split(':'))
                end_h, end_m = map(int, end_str.split(':'))
                start_time = time(start_h, start_m)
                end_time = time(end_h, end_m)
                if self._is_time_in_range(start_time, end_time, current_t):
                    return rate_val
            except ValueError:
                # Log error or handle malformed time string in tou_rates
                # print(f"Warning: Malformed time string in TOU rates: {start_str}-{end_str}")
                continue 
        return default_price

    def _is_time_in_range(self, start: time, end: time, current: time) -> bool:
        if start == end: # Handles full day rate for a specific period definition, or error in definition
            return True # Or False, depending on desired behavior for exact match to start/end of a 0-duration period
        if start < end: 
            return start <= current < end
        else: # Period spans midnight (e.g., 22:00-06:00)
            return start <= current or current < end

    def set_config(self, controller_config: dict):
        """Allow controller parameters to be set from a config dict."""
        if "mode" in controller_config:
            self.mode = controller_config["mode"]
        if "peak_shaving_threshold_kW" in controller_config:
            self.peak_shaving_threshold_kW = float(controller_config["peak_shaving_threshold_kW"])
