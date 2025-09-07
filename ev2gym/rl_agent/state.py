'''  This file contains various example state functions for the RL agent '''
import math
import numpy as np
from typing import List, Tuple


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

    # the final state of each simulation
    # if env.current_step < env.simulation_length:        
    #     setpoint = min(env.power_setpoints[env.current_step], env.charge_power_potential[env.current_step])        
    # else:
    #     setpoint = 0       
    if env.current_step < env.simulation_length:  
        # setpoint = env.power_setpoints[env.current_step:env.current_step+10]
        setpoint = env.power_setpoints[env.current_step]
    else:
        setpoint = np.zeros((1))
        
    # if len(setpoint) < 10:
    #     setpoint = np.append(setpoint, np.zeros(10-len(setpoint)))
    
    state.append(setpoint)
    state.append(env.current_power_usage[env.current_step-1])

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

def _ledger_column_plan(env, account_id: int) -> tuple[list[str], list[str]]:
    """Return ordered column lists from global and account ledgers.
    Conservative, stable selection for observations.
    """
    # Global: time features, 24h price forecast, and key totals
    g_pref = [
        "step_ratio", "dow", "hour", "minute",
        *[f"price_fc_h{h:02d}" for h in range(1, 25)],
        "power_setpoint_kw", "total_power_usage_kw", "ev_power_kw",
        "inflexible_load_kw", "solar_production_kw", "evs_parked",
    ]
    gcols = [c for c in g_pref if getattr(env, "global_buffers", None) is not None and c in env.global_buffers.columns]

    # Account: charger-level and per-port core values, plus optional per-account forecasts
    abuf = env.account_buffers.get(account_id) if getattr(env, "account_buffers", None) else None
    if abuf is None:
        return gcols, []
    # Discover port indices present
    port_indices: list[int] = []
    for name in abuf.columns:
        if name.startswith("port") and name.endswith("_soc"):
            try:
                idx = int(name.split("port")[1].split("_")[0])
                port_indices.append(idx)
            except Exception:
                continue
    port_indices = sorted(set(port_indices))

    a_pref = ["cs_power_kw", "cs_amps", "evs_connected"]
    for p in port_indices:
        a_pref.append(f"port{p}_soc")
        a_pref.append(f"port{p}_action_norm")
    # Add per-account 24h forecasts if present
    load_fc_cols = [f"load_fc_h{h:02d}" for h in range(1, 25)]
    pv_fc_cols = [f"pv_fc_h{h:02d}" for h in range(1, 25)]
    a_pref.extend([c for c in load_fc_cols if c in abuf.columns])
    a_pref.extend([c for c in pv_fc_cols if c in abuf.columns])
    acols = [c for c in a_pref if c in abuf.columns]
    return gcols, acols


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

    gvals: list[float] = []
    for c in gcols:
        arr = env.global_buffers._data.get(c)  # type: ignore[attr-defined]
        gvals.append(float(arr[t]) if arr is not None else np.nan)

    avals: list[float] = []
    abuf = env.account_buffers.get(account_id)
    if abuf is not None:
        for c in acols:
            arr = abuf._data.get(c)  # type: ignore[attr-defined]
            avals.append(float(arr[t]) if arr is not None else np.nan)

    obs = np.array(gvals + avals, dtype=float)
    return obs

def V2G_profit_max(env, *args):
    '''
    This is the state function for the V2GProfitMax scenario.
    '''
    
    state = [
        (env.current_step),        
    ]

    state.append(env.current_power_usage[env.current_step-1])

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

    state.append(env.current_power_usage[env.current_step-1])

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

    # the final state of each simulation
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