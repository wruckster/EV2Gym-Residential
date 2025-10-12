'''  This file contains various example state functions for the RL agent '''
import math
import numpy as np
from typing import Dict, List, Tuple


def PublicPST(env, *args):
    '''This state function is the public power setpoints
    The state is the public power setpoints
    The state is a vector '''

    state = [
        (env.current_step/env.simulation_length),
        # env.sim_date.weekday() / 7,
        # turn hour and minutes in sin and cos
        # math.sin(env.sim_date.hour/24*2*math.pi),
        # math.cos(env.sim_date.hour/24*2*math.pi),
    ]

    # Use ledger data instead of legacy arrays
    t = int(max(0, min(env.current_step, env.global_buffers.T - 1)))
    
    # Get power setpoint from global ledger
    if hasattr(env, 'global_buffers') and env.global_buffers is not None:
        setpoint_arr = env.global_buffers._data.get('power_setpoint_kw')
        setpoint = float(setpoint_arr[t]) if setpoint_arr is not None else 0.0
    else:
        # Fallback to legacy if ledgers not available
        if env.current_step < env.simulation_length:  
            setpoint = env.power_setpoints[env.current_step]
        else:
            setpoint = np.zeros((1))
        
    state.append(setpoint)
    
    # Get current power usage from global ledger (previous step)
    if hasattr(env, 'global_buffers') and env.global_buffers is not None:
        power_arr = env.global_buffers._data.get('total_power_usage_kw')
        prev_t = max(0, t - 1)
        current_power = float(power_arr[prev_t]) if power_arr is not None else 0.0
    else:
        # Fallback to legacy
        current_power = env.current_power_usage[env.current_step-1]
    
    state.append(current_power)

    # For every transformer
    for tr in env.transformers:
        # For every charging station connected to the transformer
        for cs in env.charging_stations:
            if cs.connected_transformer == tr.id:
                # For every EV connected to the charging station
                for EV in cs.evs_connected:
                    # If there is an EV connected
                    if EV is not None:
                        state.append([
                            1 if EV.get_soc() == 1 else 0.5,  # we know if the EV is full
                            EV.total_energy_exchanged,
                            # EV.max_ac_charge_power*1000 /
                            # (cs.voltage*math.sqrt(cs.phases))/100,
                            # EV.min_ac_charge_power*1000 /
                            # (cs.voltage*math.sqrt(cs.phases))/100,
                            (env.current_step-EV.time_of_arrival)
                            ])

                    # else if there is no EV connected put zeros
                    else:
                        state.append(np.zeros(3))

    state = np.array(np.hstack(state))

    np.set_printoptions(suppress=True)

    return state

## Ledger-based observation API

def _ledger_column_plan(env, account_id: int) -> tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Return ordered column lists from global and account ledgers with caching."""
    if getattr(env, "global_buffers", None) is None:
        return tuple(), tuple()

    cache: Dict[int, Tuple[Tuple[Tuple[str, ...], Tuple[str, ...]], Tuple[str, ...], Tuple[str, ...]]] | None = getattr(env, "_ledger_plan_cache", None)  # type: ignore[assignment]
    if cache is None:
        cache = {}
        setattr(env, "_ledger_plan_cache", cache)

    global_cols_tuple = tuple(env.global_buffers.columns)

    abuf = env.account_buffers.get(account_id) if getattr(env, "account_buffers", None) else None
    account_cols_tuple = tuple(abuf.columns) if abuf is not None else tuple()
    signature = (global_cols_tuple, account_cols_tuple)

    cached_entry = cache.get(account_id)
    if cached_entry is not None and cached_entry[0] == signature:
        return cached_entry[1], cached_entry[2]

    global_col_set = set(global_cols_tuple)
    g_pref = (
        "step_ratio",
        "dow",
        "hour",
        "minute",
        "charge_price",
        "discharge_price",
        "power_setpoint_kw",
        "total_power_usage_kw",
        "ev_power_kw",
        "inflexible_load_kw",
        "solar_production_kw",
        "evs_parked",
        *[f"price_fc_h{h:02d}" for h in range(1, 25)],
    )
    gcols = tuple(c for c in g_pref if c in global_col_set)

    if abuf is None:
        cache[account_id] = (signature, gcols, tuple())
        return gcols, tuple()

    port_indices: list[int] = []
    for name in account_cols_tuple:
        if name.startswith("port") and name.endswith("_soc"):
            try:
                idx = int(name.split("port")[1].split("_")[0])
                port_indices.append(idx)
            except Exception:
                continue
    port_indices = sorted(set(port_indices))

    a_pref: list[str] = [
        "account_power_setpoint_kw",
        "account_actual_power_kw",
        "cs_power_kw",
        "cs_kw_limit",
        "cs_amps",
        "household_inflexible_load_kw",
        "household_pv_kw",
        "evs_connected",
    ]
    for p in port_indices:
        a_pref.append(f"port{p}_soc")
        a_pref.append(f"port{p}_amps")
        a_pref.append(f"port{p}_amps_limit")
        a_pref.append(f"port{p}_connected")
        a_pref.append(f"port{p}_time_to_departure")
        a_pref.append(f"port{p}_time_since_arrival")

    load_fc_cols = [f"load_fc_h{h:02d}" for h in range(1, 25)]
    pv_fc_cols = [f"pv_fc_h{h:02d}" for h in range(1, 25)]
    a_pref.extend([c for c in load_fc_cols if c in account_cols_tuple])
    a_pref.extend([c for c in pv_fc_cols if c in account_cols_tuple])

    acols = tuple(c for c in a_pref if c in account_cols_tuple)
    cache[account_id] = (signature, gcols, acols)
    return gcols, acols


def _resolve_ledger_value_cache(
    env,
    account_id: int,
    gcols: tuple[str, ...],
    acols: tuple[str, ...],
) -> tuple[tuple[np.ndarray | None, ...], tuple[np.ndarray | None, ...]]:
    """Return cached column arrays for global and account ledgers."""

    global_buffers = getattr(env, "global_buffers", None)
    if global_buffers is None:
        return tuple(), tuple()

    cache = getattr(env, "_ledger_value_cache", None)
    global_buffer_id = id(global_buffers)
    if not isinstance(cache, dict) or cache.get("global_buffer_id") != global_buffer_id:
        cache = {
            "global_buffer_id": global_buffer_id,
            "global_plan_cache": {},
            "account_plan_cache": {},
        }

    global_plan_cache: dict[tuple[str, ...], tuple[np.ndarray | None, ...]] = cache["global_plan_cache"]  # type: ignore[assignment]
    g_arrays = global_plan_cache.get(gcols)
    if g_arrays is None:
        g_arrays = tuple(global_buffers._data.get(col) for col in gcols)  # type: ignore[attr-defined]
        global_plan_cache[gcols] = g_arrays

    accounts_cache: dict[tuple[int, int | None, tuple[str, ...]], tuple[np.ndarray | None, ...]] = cache["account_plan_cache"]  # type: ignore[assignment]
    abuf = env.account_buffers.get(account_id) if getattr(env, "account_buffers", None) else None
    account_buffer_id = id(abuf) if abuf is not None else None
    account_key = (account_id, account_buffer_id, acols)

    a_arrays = accounts_cache.get(account_key)
    if a_arrays is None:
        if abuf is None:
            a_arrays = tuple(None for _ in acols)
        else:
            a_arrays = tuple(abuf._data.get(col) for col in acols)  # type: ignore[attr-defined]
        accounts_cache[account_key] = a_arrays

    setattr(env, "_ledger_value_cache", cache)

    return g_arrays, a_arrays


def LedgersPublicState(env, account_id: int | None = None, *args) -> np.ndarray:
    """Build observation purely from ledgers for the given account at current step.

    If account_id is None, defaults to the first charging station's id.
    """
    assert getattr(env, "global_buffers", None) is not None, "global_buffers not initialized"
    # Choose default account
    if account_id is None:
        account_id = env.charging_stations[0].id if getattr(env, "charging_stations", None) else 0

    gcols, acols = _ledger_column_plan(env, account_id)
    t = int(max(0, min(env.current_step, env.global_buffers.T - 1)))

    step_cache = getattr(env, "_ledger_state_step_cache", None)
    if not isinstance(step_cache, dict) or step_cache.get("step") != t:
        step_cache = {"step": t, "values": {}}
        setattr(env, "_ledger_state_step_cache", step_cache)

    cached_value = step_cache["values"].get(account_id)
    if cached_value is not None:
        return cached_value

    g_arrays, a_arrays = _resolve_ledger_value_cache(env, account_id, gcols, acols)

    gvals: list[float] = []
    for arr in g_arrays:
        if arr is not None and 0 <= t < len(arr):
            try:
                gvals.append(float(arr[t]))
            except Exception:
                gvals.append(np.nan)
        else:
            gvals.append(np.nan)

    avals: list[float] = []
    for arr in a_arrays:
        if arr is not None and 0 <= t < len(arr):
            try:
                avals.append(float(arr[t]))
            except Exception:
                avals.append(np.nan)
        else:
            avals.append(np.nan)

    obs = np.nan_to_num(
        np.array(gvals + avals, dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    step_cache["values"][account_id] = obs
    return obs

def V2G_profit_max(env, *args):
    '''
    This is the state function for the V2GProfitMax scenario.
    '''
    
    state = [
        (env.current_step),        
    ]

    # Use ledger data instead of legacy arrays
    if hasattr(env, 'global_buffers') and env.global_buffers is not None:
        t = int(max(0, min(env.current_step, env.global_buffers.T - 1)))
        power_arr = env.global_buffers._data.get('total_power_usage_kw')
        prev_t = max(0, t - 1)
        current_power = float(power_arr[prev_t]) if power_arr is not None else 0.0
    else:
        # Fallback to legacy
        current_power = env.current_power_usage[env.current_step-1]
    
    state.append(current_power)

    horizon = 20
    # Prefer rrp_hxx forecast if available
    if hasattr(env, 'price_forecast') and env.price_forecast is not None:
        # price_forecast shape: (simulation_length, H)
        fc_slice = env.price_forecast[env.current_step] if env.current_step < len(env.price_forecast) else np.array([])
        charge_prices = np.array(fc_slice[:horizon])
    else:
        charge_prices = abs(env.charge_prices[0, env.current_step: env.current_step + horizon])

    if len(charge_prices) < horizon:
        charge_prices = np.append(charge_prices, np.zeros(horizon - len(charge_prices)))

    state.append(charge_prices)
    
    # For every transformer
    for tr in env.transformers:

        # For every charging station connected to the transformer
        for cs in env.charging_stations:
            if cs.connected_transformer == tr.id:

                # For every EV connected to the charging station
                for EV in cs.evs_connected:
                    # If there is an EV connected
                    if EV is not None:
                        state.append([
                            EV.get_soc(),
                            EV.time_of_departure - env.current_step,
                            ])

                    # else if there is no EV connected put zeros
                    else:
                        state.append(np.zeros(2))

    state = np.array(np.hstack(state))

    return state

def V2G_profit_max_loads(env, *args):
    '''
    This is the state function for the V2GProfitMax scenario with loads
    '''
    
    state = [
        (env.current_step),        
    ]

    # Use ledger data instead of legacy arrays
    if hasattr(env, 'global_buffers') and env.global_buffers is not None:
        t = int(max(0, min(env.current_step, env.global_buffers.T - 1)))
        power_arr = env.global_buffers._data.get('total_power_usage_kw')
        prev_t = max(0, t - 1)
        current_power = float(power_arr[prev_t]) if power_arr is not None else 0.0
    else:
        # Fallback to legacy
        current_power = env.current_power_usage[env.current_step-1]
    
    state.append(current_power)

    charge_prices = abs(env.charge_prices[0, env.current_step:
        env.current_step+20])
    
    if len(charge_prices) < 20:
        charge_prices = np.append(charge_prices, np.zeros(20-len(charge_prices)))
    
    state.append(charge_prices)
    
    # For every transformer
    for tr in env.transformers:
        loads, pv = tr.get_load_pv_forecast(step = env.current_step,
                                            horizon = 20)
        power_limits = tr.get_power_limits(step = env.current_step,
                                           horizon = 20)
        state.append(loads-pv)
        state.append(power_limits)
        
        # For every charging station connected to the transformer
        for cs in env.charging_stations:
            if cs.connected_transformer == tr.id:

                # For every EV connected to the charging station
                for EV in cs.evs_connected:
                    # If there is an EV connected
                    if EV is not None:
                        state.append([
                            EV.get_soc(),
                            EV.time_of_departure - env.current_step,
                            ])

                    # else if there is no EV connected put zeros
                    else:
                        state.append(np.zeros(2))

    state = np.array(np.hstack(state))

    return state
    


def V2G_profit_max_enhanced(env, *args):
    '''
    Enhanced state function for V2G profit maximization with:
    1. Normalized price signals (relative to episode min/max)
    2. Price percentile ranking (is current price high or low?)
    3. Net load (demand - solar) to encourage charging from surplus PV
    4. Time-to-departure urgency for each EV
    5. Price trend information
    '''
    
    state = []
    
    # Normalized time in episode [0, 1]
    if env.simulation_length <= 0:
        time_norm = 0.0
    else:
        time_norm = min(1.0, env.current_step / env.simulation_length)
    state.append(time_norm)
    
    # Get current prices
    t = env.current_step
    current_price = float(env.charge_prices[0, t]) if t < env.charge_prices.shape[1] else 0.0
    
    # Calculate price statistics over the entire episode for normalization
    all_prices = env.charge_prices[0, :].copy()
    
    # Ensure price array is valid
    if not np.isfinite(all_prices).all():
        all_prices = np.nan_to_num(all_prices, nan=0.0, posinf=1.0, neginf=0.0)
    
    price_mean = float(np.mean(all_prices))
    price_std = float(np.std(all_prices)) + 1e-8
    price_min = float(np.min(all_prices))
    price_max = float(np.max(all_prices))
    
    # Normalized current price (z-score)
    if not np.isfinite(current_price):
        current_price = 0.0
    price_zscore = (current_price - price_mean) / price_std
    state.append(price_zscore)
    
    # Price percentile (0-1): tells agent if current price is in top/bottom quantile
    if len(all_prices) > 0:
        price_percentile = float(np.searchsorted(np.sort(all_prices), current_price) / len(all_prices))
    else:
        price_percentile = 0.5
    state.append(price_percentile)
    
    # Binary signals: is price in top/bottom 20%?
    is_high_price = float(price_percentile > 0.8)
    is_low_price = float(price_percentile < 0.2)
    state.extend([is_high_price, is_low_price])
    
    # Price forecast (next 24 hours normalized)
    horizon = 24
    if hasattr(env, 'price_forecast') and env.price_forecast is not None:
        fc_slice = env.price_forecast[t] if t < len(env.price_forecast) else np.zeros(horizon)
        forecast = np.array(fc_slice[:horizon])
    else:
        forecast = env.charge_prices[0, t:t + horizon].copy()
    
    if len(forecast) < horizon:
        forecast = np.append(forecast, np.full(horizon - len(forecast), current_price))
    
    # Normalize forecast relative to current episode
    forecast_norm = (forecast - price_mean) / price_std
    state.extend(forecast_norm.tolist())
    
    # Net load information (demand - solar generation)
    # Check if account_online mode is enabled
    try:
        sp_cfg = env._sp_cfg()
        account_online_enabled = bool(sp_cfg.get('account_online', {}).get('enabled', False))
    except Exception:
        account_online_enabled = False
    
    current_load = 0.0
    current_solar = 0.0
    
    if account_online_enabled and hasattr(env, 'account_buffers'):
        # Aggregate from account buffers
        t_safe = int(max(0, min(t, env.simulation_length - 1)))
        for acct_id, buf in env.account_buffers.items():
            if buf is not None:
                t_buf_safe = int(max(0, min(t_safe, buf.T - 1)))
                load_val = buf._data.get('household_inflexible_load_kw')
                solar_val = buf._data.get('household_pv_kw')
                
                if load_val is not None and t_buf_safe < len(load_val):
                    val = float(load_val[t_buf_safe])
                    if np.isfinite(val):
                        current_load += val
                
                if solar_val is not None and t_buf_safe < len(solar_val):
                    val = float(solar_val[t_buf_safe])
                    if np.isfinite(val):
                        current_solar += val
    elif hasattr(env, 'global_buffers') and env.global_buffers is not None:
        # Use global buffers
        t_safe = int(max(0, min(t, env.global_buffers.T - 1)))
        
        load_arr = env.global_buffers._data.get('inflexible_load_kw')
        solar_arr = env.global_buffers._data.get('solar_production_kw')
        
        if load_arr is not None and t_safe < len(load_arr):
            val = float(load_arr[t_safe])
            if np.isfinite(val):
                current_load = val
        
        if solar_arr is not None and t_safe < len(solar_arr):
            val = float(solar_arr[t_safe])
            if np.isfinite(val):
                current_solar = val
    
    # Calculate net load (solar is already negative in the data, so add it)
    net_load = current_load + current_solar
    
    # Excess solar available for charging (negative net load)
    excess_solar = max(0.0, -net_load)
    state.append(excess_solar / 10.0)  # Scale to reasonable range
    
    # Net load (positive means consuming from grid)
    state.append(net_load / 10.0)
    
    # EV information for each charging station
    for tr in env.transformers:
        for cs in env.charging_stations:
            if cs.connected_transformer == tr.id:
                for EV in cs.evs_connected:
                    if EV is not None:
                        soc = EV.get_soc()
                        
                        # Protect against NaN/Inf in SOC
                        if not np.isfinite(soc):
                            soc = 0.0
                        
                        time_remaining = max(0, EV.time_of_departure - env.current_step)
                        
                        # Normalized time remaining [0, 1]
                        if env.simulation_length > 0:
                            time_remaining_norm = min(1.0, time_remaining / env.simulation_length)
                        else:
                            time_remaining_norm = 0.0
                        
                        # Urgency: does EV need charging? (below desired SOC with little time)
                        desired_soc = getattr(EV, 'desired_capacity_ratio', 0.8)
                        soc_deficit = max(0.0, desired_soc - soc)
                        
                        # Emergency flag: below minimum SOC
                        # Protect against division by zero
                        if EV.battery_capacity > 1e-6:
                            min_soc = EV.min_emergency_battery_capacity / EV.battery_capacity
                        else:
                            min_soc = 0.2  # Default fallback
                        
                        if not np.isfinite(min_soc):
                            min_soc = 0.2
                            
                        is_emergency = float(soc < min_soc)
                        
                        # Available capacity for discharge (SOC above minimum)
                        available_for_discharge = max(0.0, soc - min_soc)
                        
                        state.extend([
                            soc,
                            time_remaining_norm,
                            soc_deficit,
                            is_emergency,
                            available_for_discharge
                        ])
                    else:
                        # No EV connected
                        state.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Final sanity check: replace any NaN/Inf values that slipped through
    state_arr = np.array(state, dtype=np.float32)
    if not np.isfinite(state_arr).all():
        import logging
        bad_indices = np.where(~np.isfinite(state_arr))[0]
        logging.warning(f"[V2G_profit_max_enhanced] Non-finite values detected at indices {bad_indices.tolist()}: {state_arr[bad_indices]}")
        state_arr = np.nan_to_num(state_arr, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return state_arr


def BusinessPSTwithMoreKnowledge(env, *args):
    '''
    This state function is used for the business case scenario that requires more knowledge such as SoC and time of departure for each EV present.
    '''

    state = [
        (env.current_step) / env.simulation_length,
        #env.sim_date.weekday() / 5,
        # turn hour and minutes in sin and cos
        #math.sin(env.sim_date.hour/12*2*math.pi),
        #math.cos(env.sim_date.hour/12*2*math.pi),
    ]

    # Use ledger data instead of legacy arrays
    if hasattr(env, 'global_buffers') and env.global_buffers is not None:
        t = int(max(0, min(env.current_step, env.global_buffers.T - 1)))
        
        # Get power setpoint from global ledger
        setpoint_arr = env.global_buffers._data.get('power_setpoint_kw')
        setpoint = float(setpoint_arr[t]) if setpoint_arr is not None else 0.0
        state.append(setpoint)
        
        # For charge power potential, we'll use EV power from ledger as proxy
        ev_power_arr = env.global_buffers._data.get('ev_power_kw')
        charge_potential = float(ev_power_arr[t]) if ev_power_arr is not None else 0.0
        state.append(charge_potential)
    else:
        # Fallback to legacy arrays
        if env.current_step < env.simulation_length:
            state.append(env.power_setpoints[env.current_step]) #/100
            state.append(env.charge_power_potential[env.current_step]) #/100
        else:
            state.append(env.power_setpoints[env.current_step-1]) #/100
            state.append(env.charge_power_potential[env.current_step-1]) #/100   

    for tr in env.transformers:
        state.append(tr.max_current/100)
        for cs in env.charging_stations:
            if cs.connected_transformer == tr.id:
                for EV in cs.evs_connected:
                    if EV is not None:
                        state.append([#EV.total_energy_exchanged / EV.battery_capacity, #how much soc we charge
                                      #EV.max_ac_charge_power*1000 /            same EVs, no need right now
                                      #(cs.voltage*math.sqrt(cs.phases)),
                                      #EV.min_ac_charge_power*1000 /
                                      #(cs.voltage*math.sqrt(cs.phases)),
                                      EV.time_of_arrival / env.simulation_length,  # time of arrival
                                      EV.etime_of_departure / env.simulation_length,  # time of departure
                                      EV.get_soc(),  # soc
                                      #(EV.etime_of_departure - env.current_step) \
                                      #  / env.simulation_length, #remaining time
                                      #(env.current_step-EV.time_of_arrival) \
                                      #  / env.simulation_length,  # time stayed
                                      #(EV.etime_of_departure - \
                                      # EV.time_of_arrival) / env.simulation_length, # total staying time
                                      #(((EV.battery_capacity - EV.battery_capacity_at_arrival) /
                                      #  (EV.etime_of_departure - EV.time_of_arrival)) / EV.max_ac_charge_power),  # average charging speed
                                      #(((EV.battery_capacity - EV.battery_capacity_at_arrival) / EV.battery_capacity)) \
                                      #  / ((EV.etime_of_departure - env.current_step + 1) / env.simulation_length),   #charging priority
                                      #EV.required_power / EV.battery_capacity,  # required energy
                                      ])
                    else:
                        state.append(np.zeros(3))

    state = np.array(np.hstack(state))

    np.set_printoptions(suppress=True)

    return state