'''This file contains various example reward functions for the RL agent. Users can create their own reward function here or in their own file using the same structure as below
'''

import math
import warnings
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np

try:
    from typing import TYPE_CHECKING
except ImportError:  # pragma: no cover
    TYPE_CHECKING = False  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from ev2gym.models.ev2gym_env import EV2Gym


@dataclass(frozen=True)
class SolarProfitRewardParams:
    w_profit: float = 1.0
    w_soc: float = 0.5
    w_wear: float = 0.05
    w_solar: float = 0.2
    w_setpoint: float = 0.3  # Weight for setpoint tracking component
    w_ev_solar_charge: float = 0.8
    alpha: float = 0.1
    solar_norm_kwh: float = 1.0
    wear_cost_per_kwh: float = 1.0
    # grid_import_penalty_per_kwh: float = 0.0
    # invalid_action_penalty: float = 0.0
    # Normalization constants (auto-computed from env config in from_env(); these are fallbacks only)
    max_soc_deficit_kwh: float = 51.2  # Fallback: 80% of typical 64 kWh battery
    max_throughput_kwh: float = 168.0  # Fallback: 7 kW * 24 hours
    typical_profit_range: float = 50.0  # Fallback: typical $50 per day
    max_setpoint_error_kw: float = 7.0  # Fallback: typical max charge power
    normalize_components: bool = True

    @classmethod
    def from_env(
        cls,
        env: "EV2Gym",
        base: Optional["SolarProfitRewardParams"] = None,
    ) -> "SolarProfitRewardParams":
        reward_cfg: Dict[str, float] = {}
        if hasattr(env, "config") and isinstance(env.config, dict):
            reward_cfg = env.config.get("reward", {}) or {}

        preset_name = reward_cfg.get("preset") or reward_cfg.get("preset_name")
        # Only use config preset if no explicit base was provided
        if base is not None:
            base_params = base
        elif isinstance(preset_name, str):
            preset_key = preset_name.lower()
            base_params = SOLAR_REWARD_PRESETS.get(preset_key, DEFAULT_SOLAR_REWARD_PARAMS)
        else:
            base_params = DEFAULT_SOLAR_REWARD_PARAMS

        weights_cfg: Dict[str, float] = reward_cfg.get("weights", {}) or {}

        overrides: Dict[str, float] = {}
        for key in asdict(base_params).keys():
            if key == "normalize_components":
                continue

            value = weights_cfg.get(key)
            if value is None:
                value = reward_cfg.get(key)
            # Legacy fallback: only use soc_deficit_cost_per_kwh if no explicit base was provided
            if value is None and key == "w_soc" and base is None:
                value = reward_cfg.get("soc_deficit_cost_per_kwh")
            if value is None:
                continue
            try:
                overrides[key] = float(value)
            except (TypeError, ValueError):
                continue

        # Compute normalization constants from environment config
        norm_constants = cls._compute_normalization_constants(env)
        overrides.update(norm_constants)

        overrides["normalize_components"] = True

        return base_params.clone_with_overrides(**overrides)

    @staticmethod
    def _compute_normalization_constants(env: "EV2Gym") -> Dict[str, float]:
        """Compute max plausible values for each reward component from environment config."""
        config = getattr(env, "config", {}) or {}
        ev_cfg = config.get("ev", {}) or {}
        
        # Max SoC deficit: desired_capacity * battery_capacity
        battery_capacity = float(ev_cfg.get("battery_capacity", 64.0))
        desired_capacity_ratio = float(ev_cfg.get("desired_capacity", 0.8))
        max_soc_deficit_kwh = battery_capacity * desired_capacity_ratio
        
        # Max throughput: max_charge_power * forecast_horizon_hours
        max_charge_power = float(ev_cfg.get("max_ac_charge_power", 7.0))
        forecast_horizon_hours = 24.0
        forecasting_cfg = config.get("forecasting", {}) or {}
        params_cfg = forecasting_cfg.get("params", {}) or {}
        if "forecast_horizon_hours" in params_cfg:
            forecast_horizon_hours = float(params_cfg["forecast_horizon_hours"])
        max_throughput_kwh = max_charge_power * forecast_horizon_hours
        
        # Typical profit range: Use median price * max_throughput as baseline
        # This represents the daily cost/revenue from fully cycling the battery at median prices
        charge_prices = getattr(env, "charge_prices", None)
        median_price = 0.25  # $/kWh fallback
        if charge_prices is not None and hasattr(charge_prices, "flatten"):
            prices_flat = charge_prices.flatten()
            prices_finite = prices_flat[np.isfinite(prices_flat)]
            if prices_finite.size > 0:
                median_price = float(np.median(prices_finite))
        
        # Typical daily profit range: median_price * max_throughput (full battery cycle)
        typical_profit_range = median_price * max_throughput_kwh
        
        # Max setpoint error: use max charge power as typical max tracking error
        max_setpoint_error_kw = max_charge_power
        
        return {
            "max_soc_deficit_kwh": max_soc_deficit_kwh,
            "max_throughput_kwh": max_throughput_kwh,
            "typical_profit_range": typical_profit_range,
            "max_setpoint_error_kw": max_setpoint_error_kw,
            # Note: solar_norm_kwh is NOT overridden here to preserve user-specified values
        }

    def clone_with_overrides(self, **overrides: float) -> "SolarProfitRewardParams":
        payload = asdict(self)
        payload.update(overrides)
        return SolarProfitRewardParams(**payload)

DEFAULT_SOLAR_REWARD_PARAMS = SolarProfitRewardParams()

SOLAR_REWARD_PRESETS: Dict[str, SolarProfitRewardParams] = {
    "profit_first": DEFAULT_SOLAR_REWARD_PARAMS.clone_with_overrides(
        w_profit=1.0,
        w_soc=0.5,
        w_wear=0.05,
        w_solar=0.2,
        w_setpoint=0.2,  # Light setpoint tracking for profit-first strategy
        # alpha=0.1,
    ),
    "balanced": DEFAULT_SOLAR_REWARD_PARAMS.clone_with_overrides(
        w_profit=0.9,
        w_soc=0.9,
        w_wear=0.9,
        w_solar=1.0,
        w_setpoint=0.8,  # Moderate setpoint tracking for balanced strategy
        w_ev_solar_charge=1,
        # alpha=0.5,
    ),
    "renewable_first": DEFAULT_SOLAR_REWARD_PARAMS.clone_with_overrides(
        w_profit=0.6,
        w_soc=0.6,
        w_wear=0.02,
        w_solar=1.0,
        w_setpoint=0.4,  # Higher setpoint tracking for renewable-first strategy
        w_ev_solar_charge=1.0,
        # alpha=1.0,
    ),
    "user_priority": DEFAULT_SOLAR_REWARD_PARAMS.clone_with_overrides(
        w_profit=0.8,
        w_soc=2.0,
        w_wear=0.05,
        w_solar=0.2,
        w_setpoint=0.5,  # Higher setpoint tracking for user-priority strategy
        w_ev_solar_charge=0.2,
        # alpha=0.3,
    ),
    "setpoint_tracking": DEFAULT_SOLAR_REWARD_PARAMS.clone_with_overrides(
        w_profit=0.1,
        w_soc=0.1,
        w_wear=0.3,
        w_solar=0.2,
        w_setpoint=1.0,  # Higher setpoint tracking for user-priority strategy
        w_ev_solar_charge=0.7,
        # alpha=0.3,
    ),
}


def _safe_array_value(array: Optional[np.ndarray], idx: int, default: float = 0.0) -> float:
    if array is None:
        return default
    if not isinstance(array, np.ndarray):
        return default
    if idx < 0 or idx >= array.shape[-1]:
        return default
    value = array[..., idx]
    if isinstance(value, np.ndarray):
        value = float(np.mean(value))
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def _mean_price(matrix: Optional[np.ndarray], idx: int) -> float:
    if matrix is None:
        return 0.0
    if not isinstance(matrix, np.ndarray):
        return 0.0
    if matrix.ndim == 1:
        if idx < 0 or idx >= matrix.shape[0]:
            return 0.0
        value = float(matrix[idx])
        return value if math.isfinite(value) else 0.0
    if matrix.ndim >= 2:
        if idx < 0 or idx >= matrix.shape[1]:
            return 0.0
        column = matrix[:, idx]
        finite = column[np.isfinite(column)]
        if finite.size == 0:
            return 0.0
        return float(np.mean(finite))
    return 0.0


def _timescale_hours(env: "EV2Gym") -> float:
    timescale = getattr(env, "timescale", 60)
    try:
        value = float(timescale) / 60.0
    except (TypeError, ValueError):
        value = 1.0
    if value <= 0:
        return 1.0
    return value


def _soc_deficit_kwh(env: "EV2Gym") -> float:
    total = 0.0
    for ev in getattr(env, "EVs", []):
        desired = float(getattr(ev, "desired_capacity", getattr(ev, "battery_capacity", 0.0)))
        current = float(getattr(ev, "current_capacity", 0.0))
        deficit = desired - current
        if deficit > 0 and math.isfinite(deficit):
            total += deficit
    return total


def _battery_throughput_kwh(env: "EV2Gym") -> float:
    """Calculate total battery throughput for the current timestep (not cumulative).
    
    Returns the sum of absolute energy exchanged by all EVs in the current step,
    converted from power (kW) to energy (kWh) using the environment's timescale.
    """
    total = 0.0
    dt_hours = _timescale_hours(env)
    for ev in getattr(env, "EVs", []):
        # current_energy is in kW (positive for charging, negative for discharging)
        power_kw = float(getattr(ev, "current_energy", 0.0))
        energy_kwh = abs(power_kw) * dt_hours
        if math.isfinite(energy_kwh):
            total += energy_kwh
    return total


def solar_profit_reward(
    env: "EV2Gym",
    total_costs: float,
    user_satisfaction_list: Optional[List[float]],
    invalid_action_punishment: float,
    params: Optional[SolarProfitRewardParams] = None,
    *args,
) -> float:
    params = SolarProfitRewardParams.from_env(env, base=params or DEFAULT_SOLAR_REWARD_PARAMS)

    idx = int(max(0, min(getattr(env, "current_step", 0), getattr(env, "simulation_length", 1) - 1)))
    dt_hours = _timescale_hours(env)

    breakdown = getattr(env, "energy_flow_breakdown", None)
    breakdown_dict = breakdown if isinstance(breakdown, dict) else {}

    grid_kw = _safe_array_value(breakdown_dict.get("grid_draw"), idx)
    ev_kw = _safe_array_value(breakdown_dict.get("ev_power"), idx)
    solar_kw = _safe_array_value(breakdown_dict.get("solar_production"), idx)
    load_kw = _safe_array_value(breakdown_dict.get("inflexible_load"), idx)

    grid_energy_kwh = grid_kw * dt_hours
    ev_energy_kwh = ev_kw * dt_hours
    solar_gen_kwh = max(0.0, -solar_kw * dt_hours)
    load_energy_kwh = max(0.0, load_kw * dt_hours)

    e_buy = max(0.0, grid_energy_kwh)
    e_sell = max(0.0, -grid_energy_kwh)

    charge_price = _mean_price(getattr(env, "charge_prices", None), idx)
    discharge_price = _mean_price(getattr(env, "discharge_prices", None), idx)

    revenue = discharge_price * e_sell
    cost_import = charge_price * e_buy
    profit_term = revenue - cost_import
    if not math.isfinite(profit_term):
        profit_term = 0.0
    elif abs(profit_term) < 1e-9 and math.isfinite(total_costs):
        profit_term = -float(total_costs)

    local_consumption_kwh = load_energy_kwh + max(0.0, ev_energy_kwh)
    solar_used_kwh = min(solar_gen_kwh, local_consumption_kwh)

    # Portion of solar generation that feeds EV charging specifically
    ev_consumption_kwh = max(0.0, ev_energy_kwh)
    ev_solar_charge_kwh = min(ev_consumption_kwh, solar_gen_kwh)

    # solar_norm = params.solar_norm_kwh if params.solar_norm_kwh > 0 else 1.0

    soc_penalty_kwh = _soc_deficit_kwh(env)
    wear_kwh = _battery_throughput_kwh(env)

    # wear_penalty = wear_kwh * params.wear_cost_per_kwh
    # grid_penalty = params.grid_import_penalty_per_kwh * e_buy
    # invalid_penalty = params.invalid_action_penalty * float(invalid_action_punishment or 0.0)
    
    # ========== Setpoint Tracking Component ==========
    setpoint_contrib = 0.0
    
    # Check if account-online mode is enabled
    account_online_enabled = False
    try:
        config = getattr(env, 'config', {}) or {}
        setpoints_cfg = config.get('setpoints', {}) or {}
        account_cfg = setpoints_cfg.get('account_online', {}) or {}
        account_online_enabled = bool(account_cfg.get('enabled', False))
    except Exception:
        pass
    
    if account_online_enabled and hasattr(env, 'account_buffers') and env.account_buffers:
        # Account-online mode: aggregate tracking error across all accounts
        try:
            t = int(max(0, min(env.current_step, env.global_buffers.T - 1)))
            prev_t = max(0, t - 1)
            
            total_error = 0.0
            num_accounts = 0
            
            for account_id, buf in env.account_buffers.items():
                if buf is None:
                    continue
                
                setpoint_arr = buf._data.get('account_power_setpoint_kw')
                actual_arr = buf._data.get('account_actual_power_kw')
                
                if setpoint_arr is not None and actual_arr is not None:
                    power_setpoint = float(setpoint_arr[prev_t])
                    current_power = float(actual_arr[prev_t])
                    
                    if math.isfinite(power_setpoint) and math.isfinite(current_power):
                        total_error += abs(power_setpoint - current_power)
                        num_accounts += 1
            
            if num_accounts > 0:
                avg_error_kw = total_error / num_accounts
                tracking_error_norm = avg_error_kw / params.max_setpoint_error_kw if params.max_setpoint_error_kw > 0 else avg_error_kw
                setpoint_contrib = -params.w_setpoint * tracking_error_norm
        except Exception:
            pass
    elif hasattr(env, 'global_buffers') and env.global_buffers is not None:
        # Global mode: use global setpoint and power usage
        try:
            t = int(max(0, min(env.current_step, env.global_buffers.T - 1)))
            prev_t = max(0, t - 1)
            
            setpoint_arr = env.global_buffers._data.get('power_setpoint_kw')
            power_arr = env.global_buffers._data.get('total_power_usage_kw')
            
            if setpoint_arr is not None and power_arr is not None:
                power_setpoint = float(setpoint_arr[prev_t])
                current_power = float(power_arr[prev_t])
                
                if math.isfinite(power_setpoint):
                    tracking_error_kw = abs(power_setpoint - current_power)
                    tracking_error_norm = tracking_error_kw / params.max_setpoint_error_kw if params.max_setpoint_error_kw > 0 else tracking_error_kw
                    setpoint_contrib = -params.w_setpoint * tracking_error_norm
        except Exception:
            pass

    # Normalize components to [0, 1] range if enabled
    # Normalize profit to typical timestep range (daily range / steps per day)
    timescale_minutes = float(getattr(env, "timescale", 5))
    steps_per_day = int(24 * 60 / timescale_minutes)
    typical_profit_per_step = params.typical_profit_range / steps_per_day if steps_per_day > 0 else params.typical_profit_range
    profit_norm = profit_term / typical_profit_per_step if typical_profit_per_step > 0 else profit_term

    # Normalize SoC deficit to max plausible deficit per timestep
    # (assume max deficit accumulation rate = max_charge_power * timescale_hours)
    dt_hours = _timescale_hours(env)
    max_soc_deficit_per_step = max(params.max_soc_deficit_kwh, 1.0)  # Use full deficit as max
    soc_deficit_norm = soc_penalty_kwh / max_soc_deficit_per_step if max_soc_deficit_per_step > 0 else soc_penalty_kwh

    # Normalize wear to max plausible throughput per timestep
    max_wear_per_step = (params.max_throughput_kwh / steps_per_day) if steps_per_day > 0 else params.max_throughput_kwh
    wear_norm = wear_kwh / max_wear_per_step if max_wear_per_step > 0 else wear_kwh
    wear_penalty_norm = wear_norm * params.wear_cost_per_kwh

    # Solar bonus: normalize solar_used to [0,1], then directly use as bonus (without tiny alpha multiplier)
    # This puts solar contribution on similar scale as other normalized components
    solar_used_norm = solar_used_kwh / params.solar_norm_kwh if params.solar_norm_kwh > 0 else solar_used_kwh
    solar_bonus_norm = params.w_solar * solar_used_norm  # Remove alpha for normalized mode
    
    # EV solar charging bonus: reward proportional to solar→EV power (kW)
    # Convert kWh back to kW to make it comparable to profit/kW scale
    ev_solar_charge_kw = (ev_solar_charge_kwh / dt_hours) if dt_hours > 0 else 0.0
    ev_solar_charge_contrib = params.w_ev_solar_charge * ev_solar_charge_kw

    # Build normalized reward (all components in roughly [-1, 1] range before weighting)
    reward = params.w_profit * profit_norm
    reward -= params.w_soc * soc_deficit_norm
    reward -= params.w_wear * wear_penalty_norm
    # reward -= grid_penalty / typical_profit_per_step if typical_profit_per_step > 0 else grid_penalty
    reward += solar_bonus_norm
    reward += ev_solar_charge_contrib
    reward += setpoint_contrib  # Add setpoint tracking component
    # reward -= invalid_penalty / typical_profit_per_step if typical_profit_per_step > 0 else invalid_penalty

    for score in (user_satisfaction_list or []):
        if score is None:
            continue
        try:
            value = float(score)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        if value < 1.0:
            reward -= params.w_soc * max(0.0, 1.0 - value)

    if not math.isfinite(reward):
        warnings.warn("Non-finite reward detected in solar_profit_reward; replacing with 0.0")
        reward = 0.0

    return float(reward)


def solar_profit_reward_profit_first(env, total_costs, user_satisfaction_list, invalid_action_punishment, *args):
    return solar_profit_reward(
        env,
        total_costs,
        user_satisfaction_list,
        invalid_action_punishment,
        SOLAR_REWARD_PRESETS["profit_first"],
        *args,
    )


def solar_profit_reward_balanced(env, total_costs, user_satisfaction_list, invalid_action_punishment, *args):
    return solar_profit_reward(
        env,
        total_costs,
        user_satisfaction_list,
        invalid_action_punishment,
        SOLAR_REWARD_PRESETS["balanced"],
        *args,
    )


def solar_profit_reward_renewable_first(env, total_costs, user_satisfaction_list, invalid_action_punishment, *args):
    return solar_profit_reward(
        env,
        total_costs,
        user_satisfaction_list,
        invalid_action_punishment,
        SOLAR_REWARD_PRESETS["renewable_first"],
        *args,
    )


def solar_profit_reward_user_priority(env, total_costs, user_satisfaction_list, invalid_action_punishment, *args):
    return solar_profit_reward(
        env,
        total_costs,
        user_satisfaction_list,
        invalid_action_punishment,
        SOLAR_REWARD_PRESETS["user_priority"],
        *args,
    )

def solar_profit_reward_setpoint_tracking(env, total_costs, user_satisfaction_list, invalid_action_punishment, *args):
    return solar_profit_reward(
        env,
        total_costs,
        user_satisfaction_list,
        invalid_action_punishment,
        SOLAR_REWARD_PRESETS["setpoint_tracking"],
        *args,
    )


def SquaredTrackingErrorReward(env,*args):
    '''This reward function is the squared tracking error that uses the minimum of the power setpoints and the charge power potential
    The reward is negative'''
    
    # Use ledger data instead of legacy arrays
    if hasattr(env, 'global_buffers') and env.global_buffers is not None:
        t = int(max(0, min(env.current_step, env.global_buffers.T - 1)))
        prev_t = max(0, t - 1)
        
        # Get power setpoint from global ledger
        setpoint_arr = env.global_buffers._data.get('power_setpoint_kw')
        power_setpoint = float(setpoint_arr[prev_t]) if setpoint_arr is not None else 0.0
        
        # Get EV power as proxy for charge power potential
        ev_power_arr = env.global_buffers._data.get('ev_power_kw')
        charge_potential = float(ev_power_arr[prev_t]) if ev_power_arr is not None else 0.0
        
        # Get total power usage
        power_arr = env.global_buffers._data.get('total_power_usage_kw')
        current_power = float(power_arr[prev_t]) if power_arr is not None else 0.0
        
        reward = - (min(power_setpoint, charge_potential) - current_power)**2
    else:
        # Fallback to legacy arrays
        reward = - (min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1]) -
            env.current_power_usage[env.current_step-1])**2
        
    return reward

def SqTrError_TrPenalty_UserIncentives(env, _, user_satisfaction_list, *args):
    ''' This reward function is the squared tracking error that uses the minimum of the power setpoints and the charge power potential
    It penalizes transofrmers that are overloaded    
    The reward is negative'''
    
    # Use ledger data instead of legacy arrays
    if hasattr(env, 'global_buffers') and env.global_buffers is not None:
        t = int(max(0, min(env.current_step, env.global_buffers.T - 1)))
        prev_t = max(0, t - 1)
        
        # Get power setpoint from global ledger
        setpoint_arr = env.global_buffers._data.get('power_setpoint_kw')
        power_setpoint = float(setpoint_arr[prev_t]) if setpoint_arr is not None else 0.0
        
        # Get EV power as proxy for charge power potential
        ev_power_arr = env.global_buffers._data.get('ev_power_kw')
        charge_potential = float(ev_power_arr[prev_t]) if ev_power_arr is not None else 0.0
        
        # Get total power usage
        power_arr = env.global_buffers._data.get('total_power_usage_kw')
        current_power = float(power_arr[prev_t]) if power_arr is not None else 0.0
        
        # Get transformer max limit (fallback to transformer object)
        tr_max_limit = env.transformers[0].max_power[env.current_step-1]
        
        reward = - (min(power_setpoint, charge_potential, tr_max_limit) - current_power)**2
    else:
        # Fallback to legacy arrays
        tr_max_limit = env.transformers[0].max_power[env.current_step-1]
        
        reward = - (min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1],tr_max_limit) -
            env.current_power_usage[env.current_step-1])**2
            
    for tr in env.transformers:
        reward -= 100 * tr.get_how_overloaded()
        
    for score in user_satisfaction_list:
        reward -= 1000 * (1 - score)
                    
    return reward

def ProfitMax_TrPenalty_UserIncentives(env, total_costs, user_satisfaction_list, *args):
    
    reward = total_costs
    
    for tr in env.transformers:
        reward -= 100 * tr.get_how_overloaded()                        
    
    for score in user_satisfaction_list:        
        reward -= 100 * math.exp(-10*score)
        
    return reward

def SquaredTrackingErrorRewardWithPenalty(env,*args):
    ''' This reward function is the squared tracking error that uses the minimum of the power setpoints and the charge power potential
    The reward is negative
    If the EV is not charging, the reward is penalized
    '''
    # Use ledger data instead of legacy arrays
    if hasattr(env, 'global_buffers') and env.global_buffers is not None:
        t = int(max(0, min(env.current_step, env.global_buffers.T - 1)))
        prev_t = max(0, t - 1)
        prev_prev_t = max(0, t - 2)
        
        # Get power values from ledger
        power_arr = env.global_buffers._data.get('total_power_usage_kw')
        current_power = float(power_arr[prev_t]) if power_arr is not None else 0.0
        
        ev_power_arr = env.global_buffers._data.get('ev_power_kw')
        charge_potential_prev = float(ev_power_arr[prev_prev_t]) if ev_power_arr is not None else 0.0
        charge_potential = float(ev_power_arr[prev_t]) if ev_power_arr is not None else 0.0
        
        setpoint_arr = env.global_buffers._data.get('power_setpoint_kw')
        power_setpoint = float(setpoint_arr[prev_t]) if setpoint_arr is not None else 0.0
        
        if current_power == 0 and charge_potential_prev != 0:
            reward = - (min(power_setpoint, charge_potential) - current_power)**2 - 1000
        else:
            reward = - (min(power_setpoint, charge_potential) - current_power)**2
    else:
        # Fallback to legacy arrays
        if env.current_power_usage[env.current_step-1] == 0 and env.charge_power_potential[env.current_step-2] != 0:
            reward = - (min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1]) -
            env.current_power_usage[env.current_step-1])**2 - 1000
        else:
            reward = - (min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1]) -
            env.current_power_usage[env.current_step-1])**2
    
    return reward

def SimpleReward(env,*args):
    '''This reward function does not consider the charge power potential'''
    
    # Check if account-online mode is enabled
    account_online_enabled = False
    try:
        config = getattr(env, 'config', {}) or {}
        setpoints_cfg = config.get('setpoints', {}) or {}
        account_cfg = setpoints_cfg.get('account_online', {}) or {}
        account_online_enabled = bool(account_cfg.get('enabled', False))
    except Exception:
        pass
    
    reward = 0.0
    
    if account_online_enabled and hasattr(env, 'account_buffers') and env.account_buffers:
        # Account-online mode: aggregate tracking error across all accounts
        try:
            t = int(max(0, min(env.current_step, env.global_buffers.T - 1)))
            prev_t = max(0, t - 1)
            
            total_error = 0.0
            num_accounts = 0
            
            for account_id, buf in env.account_buffers.items():
                if buf is None:
                    continue
                
                setpoint_arr = buf._data.get('account_power_setpoint_kw')
                actual_arr = buf._data.get('account_actual_power_kw')
                
                if setpoint_arr is not None and actual_arr is not None:
                    power_setpoint = float(setpoint_arr[prev_t])
                    current_power = float(actual_arr[prev_t])
                    
                    if math.isfinite(power_setpoint) and math.isfinite(current_power):
                        total_error += (power_setpoint - current_power) ** 2
                        num_accounts += 1
            
            if num_accounts > 0:
                reward = -(total_error / num_accounts)
        except Exception:
            pass
    elif hasattr(env, 'global_buffers') and env.global_buffers is not None:
        # Global mode: use global setpoint and power usage
        try:
            t = int(max(0, min(env.current_step, env.global_buffers.T - 1)))
            prev_t = max(0, t - 1)
            
            setpoint_arr = env.global_buffers._data.get('power_setpoint_kw')
            power_setpoint = float(setpoint_arr[prev_t]) if setpoint_arr is not None else 0.0
            
            power_arr = env.global_buffers._data.get('total_power_usage_kw')
            current_power = float(power_arr[prev_t]) if power_arr is not None else 0.0
            
            reward = -(power_setpoint - current_power) ** 2
        except Exception:
            pass
    else:
        # Fallback to legacy arrays
        try:
            idx = env.current_step - 1
            reward = -(env.power_setpoints[idx] - env.current_power_usage[idx]) ** 2
        except Exception:
            pass
    
    return reward

def MinimizeTrackerSurplusWithChargeRewards(env,*args):
    ''' This reward function minimizes the tracker surplus and gives a reward for charging '''
    
    # Check if account-online mode is enabled
    account_online_enabled = False
    try:
        config = getattr(env, 'config', {}) or {}
        setpoints_cfg = config.get('setpoints', {}) or {}
        account_cfg = setpoints_cfg.get('account_online', {}) or {}
        account_online_enabled = bool(account_cfg.get('enabled', False))
    except Exception:
        pass
    
    reward = 0.0
    
    if account_online_enabled and hasattr(env, 'account_buffers') and env.account_buffers:
        # Account-online mode: aggregate across all accounts
        try:
            t = int(max(0, min(env.current_step, env.global_buffers.T - 1)))
            prev_t = max(0, t - 1)
            
            total_surplus_penalty = 0.0
            total_charge_reward = 0.0
            
            for account_id, buf in env.account_buffers.items():
                if buf is None:
                    continue
                
                setpoint_arr = buf._data.get('account_power_setpoint_kw')
                actual_arr = buf._data.get('account_actual_power_kw')
                
                if setpoint_arr is not None and actual_arr is not None:
                    power_setpoint = float(setpoint_arr[prev_t])
                    current_power = float(actual_arr[prev_t])
                    
                    if math.isfinite(power_setpoint) and math.isfinite(current_power):
                        # Penalize surplus (when current power exceeds setpoint)
                        if power_setpoint < current_power:
                            total_surplus_penalty += (current_power - power_setpoint) ** 2
                        
                        # Reward for charging (positive power usage)
                        total_charge_reward += current_power
            
            reward = -total_surplus_penalty + total_charge_reward
        except Exception:
            pass
    elif hasattr(env, 'global_buffers') and env.global_buffers is not None:
        # Global mode: use global setpoint and power usage
        try:
            t = int(max(0, min(env.current_step, env.global_buffers.T - 1)))
            prev_t = max(0, t - 1)
            
            setpoint_arr = env.global_buffers._data.get('power_setpoint_kw')
            power_setpoint = float(setpoint_arr[prev_t]) if setpoint_arr is not None else 0.0
            
            power_arr = env.global_buffers._data.get('total_power_usage_kw')
            current_power = float(power_arr[prev_t]) if power_arr is not None else 0.0
            
            if power_setpoint < current_power:
                reward -= (current_power - power_setpoint) ** 2
            
            reward += current_power
        except Exception:
            pass
    else:
        # Fallback to legacy arrays
        try:
            idx = env.current_step - 1
            if env.power_setpoints[idx] < env.current_power_usage[idx]:
                reward -= (env.current_power_usage[idx] - env.power_setpoints[idx]) ** 2
            
            reward += env.current_power_usage[idx]
        except Exception:
            pass
    
    return reward

def profit_maximization(env, total_costs, user_satisfaction_list, *args):
    ''' This reward function is used for the profit maximization case '''
    # Prefer net grid cost at the meter as the objective signal.
    # env.cost_history includes: grid import cost + charging cost - discharge credit.
    # We maximize profit by minimizing cost: reward = - net_cost.
    try:
        idx = max(0, env.current_step - 1)
        net_cost = float(env.cost_history[idx])
        reward = -net_cost
    except Exception:
        # Fallback to legacy total_costs if cost_history is unavailable
        reward = float(total_costs)

    # Per-step penalty when EV is below its emergency SoC threshold
    try:
        coeff = float(getattr(env, 'emergency_soc_penalty_kwh', 2.0))  # reward units per kWh deficit
        below_penalty = 0.0
        for cs in getattr(env, 'charging_stations', []):
            for ev in getattr(cs, 'evs_connected', []):
                if ev is None:
                    continue
                deficit = ev.min_emergency_battery_capacity - ev.current_capacity
                if deficit > 1e-9:
                    below_penalty += deficit * coeff
        reward -= below_penalty
    except Exception:
        # Be robust: ignore penalty calculation errors
        pass

    for score in user_satisfaction_list:
        # reward -= 100 * (1 - score)
        scale = 2
        reward -= scale * math.exp(-10*score)

    # Numerical safety: ensure finite reward
    if not math.isfinite(reward):
        warnings.warn("Non-finite reward detected in profit_maximization; replacing with 0.0")
        reward = 0.0
    return reward


@dataclass(frozen=True)
class NormalizedProfitRewardParams:
    """Parameters for normalized profit maximization reward with setpoint awareness."""
    w_profit: float = 0.8
    w_setpoint: float = 2
    w_soc: float = 0.5
    w_user_satisfaction: float = 0.8
    # Normalization constants (auto-computed from env config)
    max_cost_per_step: float = 10.0  # Fallback: typical max cost per timestep
    max_setpoint_error_kw: float = 20.0  # Fallback: typical max tracking error
    max_soc_deficit_kwh: float = 51.2  # Fallback: 80% of typical 64 kWh battery
    
    @classmethod
    def from_env(
        cls,
        env: "EV2Gym",
        base: Optional["NormalizedProfitRewardParams"] = None,
    ) -> "NormalizedProfitRewardParams":
        """Create parameters from environment configuration."""
        reward_cfg: Dict[str, float] = {}
        if hasattr(env, "config") and isinstance(env.config, dict):
            reward_cfg = env.config.get("reward", {}) or {}
        
        base_params = base or DEFAULT_NORMALIZED_PROFIT_PARAMS
        weights_cfg: Dict[str, float] = reward_cfg.get("weights", {}) or {}
        
        overrides: Dict[str, float] = {}
        for key in asdict(base_params).keys():
            value = weights_cfg.get(key)
            if value is None:
                value = reward_cfg.get(key)
            if value is None:
                continue
            try:
                overrides[key] = float(value)
            except (TypeError, ValueError):
                continue
        
        # Compute normalization constants from environment config
        norm_constants = cls._compute_normalization_constants(env)
        overrides.update(norm_constants)
        
        return base_params.clone_with_overrides(**overrides)
    
    @staticmethod
    def _compute_normalization_constants(env: "EV2Gym") -> Dict[str, float]:
        """Compute max plausible values for each reward component from environment config."""
        config = getattr(env, "config", {}) or {}
        ev_cfg = config.get("ev", {}) or {}
        
        # Max cost per step: use median price * max charge power * timescale
        max_charge_power = float(ev_cfg.get("max_ac_charge_power", 7.0))
        timescale_minutes = float(getattr(env, "timescale", 5))
        dt_hours = timescale_minutes / 60.0
        
        charge_prices = getattr(env, "charge_prices", None)
        median_price = 0.25  # $/kWh fallback
        if charge_prices is not None and hasattr(charge_prices, "flatten"):
            prices_flat = charge_prices.flatten()
            prices_finite = prices_flat[np.isfinite(prices_flat)]
            if prices_finite.size > 0:
                median_price = float(np.median(prices_finite))
        
        max_cost_per_step = median_price * max_charge_power * dt_hours
        
        # Max setpoint error: use max charge power as proxy
        max_setpoint_error_kw = max_charge_power
        
        # Max SoC deficit: desired_capacity * battery_capacity
        battery_capacity = float(ev_cfg.get("battery_capacity", 64.0))
        desired_capacity_ratio = float(ev_cfg.get("desired_capacity", 0.8))
        max_soc_deficit_kwh = battery_capacity * desired_capacity_ratio
        
        return {
            "max_cost_per_step": max_cost_per_step,
            "max_setpoint_error_kw": max_setpoint_error_kw,
            "max_soc_deficit_kwh": max_soc_deficit_kwh,
        }
    
    def clone_with_overrides(self, **overrides: float) -> "NormalizedProfitRewardParams":
        payload = asdict(self)
        payload.update(overrides)
        return NormalizedProfitRewardParams(**payload)


DEFAULT_NORMALIZED_PROFIT_PARAMS = NormalizedProfitRewardParams()


def profit_maximization_normalized(
    env: "EV2Gym",
    total_costs: float,
    user_satisfaction_list: Optional[List[float]],
    invalid_action_punishment: float,
    params: Optional[NormalizedProfitRewardParams] = None,
    *args,
) -> float:
    """
    Normalized profit maximization reward with setpoint tracking awareness.
    
    This reward function combines:
    1. Profit maximization (minimizing grid costs)
    2. Setpoint tracking (following power setpoints when available)
    3. SoC management (ensuring EVs meet their desired capacity)
    4. User satisfaction (penalizing low satisfaction scores)
    
    All components are normalized to similar scales for balanced learning.
    
    Args:
        env: EV2Gym environment instance
        total_costs: Total costs from the environment (fallback)
        user_satisfaction_list: List of user satisfaction scores [0, 1]
        invalid_action_punishment: Penalty for invalid actions
        params: Reward parameters (auto-computed from env if None)
    
    Returns:
        Normalized reward value
    """
    params = NormalizedProfitRewardParams.from_env(env, base=params or DEFAULT_NORMALIZED_PROFIT_PARAMS)
    
    idx = max(0, env.current_step - 1)
    
    # ========== Component 1: Profit (negative cost) ==========
    try:
        net_cost = float(env.cost_history[idx])
        profit_raw = -net_cost
    except Exception:
        profit_raw = float(total_costs)
    
    # Normalize profit to [-1, 1] range based on typical cost per step
    profit_norm = profit_raw / params.max_cost_per_step if params.max_cost_per_step > 0 else profit_raw
    profit_contrib = params.w_profit * profit_norm
    
    # ========== Component 2: Setpoint Tracking ==========
    setpoint_contrib = 0.0
    
    # Check if account-online mode is enabled
    account_online_enabled = False
    try:
        config = getattr(env, 'config', {}) or {}
        setpoints_cfg = config.get('setpoints', {}) or {}
        account_cfg = setpoints_cfg.get('account_online', {}) or {}
        account_online_enabled = bool(account_cfg.get('enabled', False))
    except Exception:
        pass
    
    if account_online_enabled and hasattr(env, 'account_buffers') and env.account_buffers:
        # Account-online mode: aggregate tracking error across all accounts
        try:
            t = int(max(0, min(env.current_step, env.global_buffers.T - 1)))
            prev_t = max(0, t - 1)
            
            total_error = 0.0
            num_accounts = 0
            
            for account_id, buf in env.account_buffers.items():
                if buf is None:
                    continue
                
                setpoint_arr = buf._data.get('account_power_setpoint_kw')
                actual_arr = buf._data.get('account_actual_power_kw')
                
                if setpoint_arr is not None and actual_arr is not None:
                    power_setpoint = float(setpoint_arr[prev_t])
                    current_power = float(actual_arr[prev_t])
                    
                    # Only apply setpoint tracking if setpoint is valid (not NaN)
                    if math.isfinite(power_setpoint) and math.isfinite(current_power):
                        total_error += abs(power_setpoint - current_power)
                        num_accounts += 1
            
            if num_accounts > 0:
                avg_error_kw = total_error / num_accounts
                tracking_error_norm = avg_error_kw / params.max_setpoint_error_kw if params.max_setpoint_error_kw > 0 else avg_error_kw
                setpoint_contrib = -params.w_setpoint * tracking_error_norm
        except Exception:
            pass
    elif hasattr(env, 'global_buffers') and env.global_buffers is not None:
        # Global mode: use global setpoint and power usage
        try:
            t = int(max(0, min(env.current_step, env.global_buffers.T - 1)))
            prev_t = max(0, t - 1)
            
            # Get power setpoint and actual power from global ledger
            setpoint_arr = env.global_buffers._data.get('power_setpoint_kw')
            power_arr = env.global_buffers._data.get('total_power_usage_kw')
            
            if setpoint_arr is not None and power_arr is not None:
                power_setpoint = float(setpoint_arr[prev_t])
                current_power = float(power_arr[prev_t])
                
                # Only apply setpoint tracking if setpoint is valid (not NaN)
                if math.isfinite(power_setpoint):
                    tracking_error_kw = abs(power_setpoint - current_power)
                    # Normalize to [0, 1] range, then negate (lower error = higher reward)
                    tracking_error_norm = tracking_error_kw / params.max_setpoint_error_kw if params.max_setpoint_error_kw > 0 else tracking_error_kw
                    setpoint_contrib = -params.w_setpoint * tracking_error_norm
        except Exception:
            pass
    elif hasattr(env, 'power_setpoints') and env.power_setpoints is not None:
        # Fallback to legacy arrays
        try:
            power_setpoint = float(env.power_setpoints[idx])
            current_power = float(env.current_power_usage[idx])
            
            if math.isfinite(power_setpoint):
                tracking_error_kw = abs(power_setpoint - current_power)
                tracking_error_norm = tracking_error_kw / params.max_setpoint_error_kw if params.max_setpoint_error_kw > 0 else tracking_error_kw
                setpoint_contrib = -params.w_setpoint * tracking_error_norm
        except Exception:
            pass
    
    # ========== Component 3: SoC Management ==========
    soc_penalty_kwh = _soc_deficit_kwh(env)
    soc_deficit_norm = soc_penalty_kwh / params.max_soc_deficit_kwh if params.max_soc_deficit_kwh > 0 else soc_penalty_kwh
    soc_contrib = -params.w_soc * soc_deficit_norm
    
    # ========== Component 4: User Satisfaction ==========
    user_contrib = 0.0
    for score in (user_satisfaction_list or []):
        if score is None:
            continue
        try:
            value = float(score)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        # Exponential penalty for low satisfaction (normalized to [0, 1])
        user_contrib -= params.w_user_satisfaction * math.exp(-10 * value)
    
    # ========== Combine Components ==========
    reward = profit_contrib + setpoint_contrib + soc_contrib + user_contrib
    
    # Numerical safety: ensure finite reward
    if not math.isfinite(reward):
        warnings.warn("Non-finite reward detected in profit_maximization_normalized; replacing with 0.0")
        reward = 0.0
    
    return float(reward)


# Previous reward functions for testing
#############################################################################################################
        # reward = total_costs  # - 0.5
        # print(f'total_costs: {total_costs}')
        # print(f'user_satisfaction_list: {user_satisfaction_list}')
        # for score in user_satisfaction_list:
        #     reward -= 100 * (1 - score)

        # Punish invalid actions (actions that try to charge or discharge when there is no EV connected)
        # reward -= 2 * (invalid_action_punishment/self.number_of_ports)

        # reward = min(2, 1 * 4 * self.cs / (0.00001 + (
        #     self.power_setpoints[self.current_step-1] - self.current_power_usage[self.current_step-1])**2))

        # this is the new reward function
        # reward = min(2, 1/((min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #           self.current_power_usage[self.current_step-1])**2 + 0.000001))

        # new_*10*charging
        # if self.power_setpoints[self.current_step-1] < self.current_power_usage[self.current_step-1]:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #             self.current_power_usage[self.current_step-1])
        # else:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #             self.current_power_usage[self.current_step-1])*10

        # new_1_equal
        # if self.power_setpoints[self.current_step-1] < self.current_power_usage[self.current_step-1]:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #             self.current_power_usage[self.current_step-1])
        # else:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #             self.current_power_usage[self.current_step-1])

        # new_0.1
        # if self.power_setpoints[self.current_step-1] < self.current_power_usage[self.current_step-1]:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #             self.current_power_usage[self.current_step-1])**2
        # else:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #             self.current_power_usage[self.current_step-1])*0.1

        # new_reward squared
        # if self.power_setpoints[self.current_step-1] < self.current_power_usage[self.current_step-1]:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #           self.current_power_usage[self.current_step-1])**2
        # else:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #           self.current_power_usage[self.current_step-1])

        # for score in user_satisfaction_list:
        #     reward -= 100 * (1 - score)

        # for tr in self.transformers:
        #     if tr.current_amps > tr.max_current:
        #         reward -= 1000 * abs(tr.current_amps - tr.max_current)
        #     elif tr.current_amps < tr.min_current:
        #         reward -= 1000 * abs(tr.current_amps - tr.min_current)

        # reward -= 100 * (tr.current_amps < tr.min_amps)
        #######################################################################################################
        # squared tracking error
        # reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #            self.current_power_usage[self.current_step-1])**2

        # best reward so far
        ############################################################################################################
        # if self.power_setpoints[self.current_step-1] < self.current_power_usage[self.current_step-1]:
        #     reward -= (self.current_power_usage[self.current_step-1]-self.power_setpoints[self.current_step-1])

        # reward += self.current_power_usage[self.current_step-1]/75
        ############################################################################################################
        # normalize reward to -1 1
        # reward = reward/1000
        # reward = (100 +reward) / 1000
        # print(f'reward: {reward}')

        # reward -= 2 * (invalid_action_punishment/self.number_of_ports)
        # reward /= 100
        # reward = (100 +reward) / 1000
        # print(f'current_power_usage: {self.current_power_usage[self.current_step-1]}')

        # return reward