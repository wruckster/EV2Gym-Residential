'''
This file contains the loaders for the EV City environment.
'''

import os
import numpy as np
import pandas as pd
import math
import datetime
import importlib.resources
import pathlib
from pathlib import Path
from functools import lru_cache
from importlib.resources.abc import Traversable
import pytz

def get_resource_path(package, resource):
    """Helper function to get resource path using importlib.resources"""
    # All data files are in ev2gym.data, ignore the subpackage structure
    with importlib.resources.path('ev2gym.data', resource) as p:
        return str(p)
import json
from typing import Dict, List, Optional, Tuple

from ev2gym.models.ev.charger import EV_Charger
from ev2gym.models.ev.vehicle import EV
from ev2gym.models.transformer import Transformer

from ev2gym.profiles.schedule_generator import generate_ev_profiles
from ev2gym.utilities.utils import EV_spawner, generate_power_setpoints, EV_spawner_GF


_HOUSEHOLD_PROFILE_CACHE: Dict[Tuple, pd.DataFrame] = {}
_FALLBACK_HOUSEHOLD_DATA_CACHE: Dict[Tuple, np.ndarray] = {}
_FALLBACK_PV_DATA_CACHE: Dict[Tuple, np.ndarray] = {}
_EXTERNAL_FEATURES_CACHE: Dict[Tuple, pd.DataFrame] = {}
_WEATHER_CACHE: Dict[Tuple, pd.DataFrame] = {}


@lru_cache(maxsize=16)
def _read_household_parquet(data_file: str, household_ids_key: Optional[Tuple[str, ...]]) -> pd.DataFrame:
    columns = ["timestamp", "household_id", "demand", "solar"]
    df = pd.read_parquet(data_file, columns=columns)
    required_cols = set(columns)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Parquet missing columns {missing} in {data_file}")

    if household_ids_key:
        df = df[df["household_id"].isin(household_ids_key)].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    # Aggregate across households per timestamp (sum over selected ids)
    return df.groupby(level=0)[["demand", "solar"]].sum()


@lru_cache(maxsize=64)
def _read_household_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        parse_dates=["interval_start"],
        usecols=["interval_start", "demand", "solar"],
    )


@lru_cache(maxsize=8)
def _read_external_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def _ensure_filter_config_defaults(env) -> None:
    """Ensure env.config has default blocks for filter-related options.

    Adds the following keys with safe defaults if missing:
      - setpoint_filters: controls aggregate setpoint smoothing/ramp limiting
      - control_filters: controls controller post-processing filters
      - pv_preference:   soft bias for using PV surplus in rule-based controller

    Defaults keep behavior identical to previous versions (all disabled).
    """
    cfg = getattr(env, 'config', None)
    if not isinstance(cfg, dict):
        return

    # Aggregate setpoint filters (used by generate_power_setpoints)
    # Support both unified setpoints.filters and legacy setpoint_filters
    setpoints = cfg.setdefault('setpoints', {})
    spf_unified = setpoints.setdefault('filters', {})
    spf_legacy = cfg.setdefault('setpoint_filters', {})
    
    # Ensure unified structure exists with defaults
    sm = spf_unified.setdefault('smoothing', {})
    sm.setdefault('enabled', False)
    sm.setdefault('ema_alpha', 0.3)
    rl = spf_unified.setdefault('ramp_limit', {})
    rl.setdefault('enabled', False)
    rl.setdefault('max_ramp_kw_per_step', 5.0)
    
    # Also ensure legacy structure for backward compatibility
    sm_legacy = spf_legacy.setdefault('smoothing', {})
    sm_legacy.setdefault('enabled', False)
    sm_legacy.setdefault('ema_alpha', 0.3)
    rl_legacy = spf_legacy.setdefault('ramp_limit', {})
    rl_legacy.setdefault('enabled', False)
    rl_legacy.setdefault('max_ramp_kw_per_step', 5.0)

    # Controller post-filters (used by MPC and rule-based controller if they opt-in)
    cf = cfg.setdefault('control_filters', {})
    c_sm = cf.setdefault('smoothing', {})
    c_sm.setdefault('enabled', False)
    c_sm.setdefault('ema_alpha', 0.3)
    c_rl = cf.setdefault('ramp_limit', {})
    c_rl.setdefault('enabled', False)
    c_rl.setdefault('max_ramp_kw_per_step', 5.0)

    # PV surplus preference for rule-based controller
    pvp = cfg.setdefault('pv_preference', {})
    pvp.setdefault('enabled', False)
    pvp.setdefault('weight', 0.0)


def load_ev_spawn_scenarios(env) -> None:
    '''Loads the EV spawn scenarios of the simulation'''

    # Load the EV specs
    if env.config['heterogeneous_ev_specs']:
        
        if "ev_specs_file" in env.config:
            ev_specs_file = env.config['ev_specs_file']
        else:            
            ev_specs_file = get_resource_path('ev2gym.data', 'ev_specs.json')
        
        with open(ev_specs_file) as f:
            env.ev_specs = json.load(f)

        registrations = np.zeros(len(env.ev_specs.keys()))
        for i, ev_name in enumerate(env.ev_specs.keys()):
            # sum the total number of registrations
            registrations[i] = env.ev_specs[ev_name]['number_of_registrations']

        env.normalized_ev_registrations = registrations/registrations.sum()

    if env.scenario == 'GF':
        env.df_arrival = np.load('./GF_data/time_of_arrival.npy')  # weekdays
        env.time_of_connection_vs_hour_weekday = np.load(
            './GF_data/weekday_time_of_stay.npy')
        env.time_of_connection_vs_hour_weekend = np.load(
            './GF_data/weekend_time_of_stay.npy')
        env.df_req_energy_weekday = np.load('./GF_data/weekday_volumeKWh.npy')
        env.df_req_energy_weekend = np.load('./GF_data/weekend_volumeKWh.npy')

        return

    df_arrival_week_file = get_resource_path('ev2gym.data', 'distribution-of-arrival.csv')
    df_arrival_weekend_file = get_resource_path('ev2gym.data', 'distribution-of-arrival-weekend.csv')
    df_connection_time_file = get_resource_path('ev2gym.data', 'distribution-of-connection-time.csv')
    df_energy_demand_file = get_resource_path('ev2gym.data', 'distribution-of-energy-demand.csv')
    time_of_connection_vs_hour_file = get_resource_path('ev2gym.data', 'time_of_connection_vs_hour.npy')
    df_req_energy_file = get_resource_path('ev2gym.data', 'mean-demand-per-arrival.csv')
    df_time_of_stay_vs_arrival_file = get_resource_path('ev2gym.data', 'mean-session-length-per.csv')

    env.df_arrival_week = pd.read_csv(df_arrival_week_file)  # weekdays
    env.df_arrival_weekend = pd.read_csv(df_arrival_weekend_file)  # weekends
    env.df_connection_time = pd.read_csv(
        df_connection_time_file)  # connection time
    env.df_energy_demand = pd.read_csv(df_energy_demand_file)  # energy demand
    env.time_of_connection_vs_hour = np.load(
        time_of_connection_vs_hour_file)  # time of connection vs hour

    env.df_req_energy = pd.read_csv(
        df_req_energy_file)  # energy demand per arrival
    # replace column work with workplace
    env.df_req_energy = env.df_req_energy.rename(columns={'work': 'workplace',
                                                          'home': 'private'})
    env.df_req_energy = env.df_req_energy.fillna(0)

    env.df_time_of_stay_vs_arrival = pd.read_csv(
        df_time_of_stay_vs_arrival_file)  # time of stay vs arrival
    env.df_time_of_stay_vs_arrival = env.df_time_of_stay_vs_arrival.fillna(0)
    env.df_time_of_stay_vs_arrival = env.df_time_of_stay_vs_arrival.rename(columns={'work': 'workplace',
                                                                                    'home': 'private'})


def load_power_setpoints(env) -> np.ndarray:
    '''
    Loads the power setpoints of the simulation based on the day-ahead prices
    '''

    if env.load_from_replay_path:
        return env.replay.power_setpoints
    else:
        # Ensure config blocks exist so utils can read them safely
        _ensure_filter_config_defaults(env)
        return generate_power_setpoints(env)


def generate_residential_inflexible_loads(env) -> np.ndarray:
    '''
    This function loads the inflexible loads of each transformer
    in the simulation.
    '''

    cached_env_arr = getattr(env, "_cached_inflexible_loads", None)
    if cached_env_arr is not None:
        return cached_env_arr

    # Load the data
    # --- Use NSW household profiles if provided ---
    household_df = _load_household_profiles(env)
    if household_df is not None:
        scale = env.config['inflexible_loads'].get('scale_mean', 1.0)
        demand_series = household_df['demand'].to_numpy(dtype=np.float32, copy=False)
        base = demand_series * scale
        factors = env.tr_rng.uniform(0.9, 1.1, size=(env.number_of_transformers, 1))
        arr = factors * base[np.newaxis, :]
        try:
            if not hasattr(env, '_dbg_inflex_once'):
                env._dbg_inflex_once = True
                src = env.config.get('inflexible_loads', {}).get('data_file') or env.config.get('inflexible_loads', {}).get('data_files')
                if np.allclose(arr, 0.0):
                    print("[WARN inflex] Generated inflexible loads are all zeros from household path. Check scale_mean and input files/date filter.")
        except Exception:
            pass
        env._cached_inflexible_loads = arr
        return arr

    # If no NSW data available, fall back to the default dataset
    data_path = get_resource_path('ev2gym.data', 'residential_loads.csv')
    cache_key = (
        data_path,
        env.timescale,
        env.simulation_length,
    )
    base_matrix = _FALLBACK_HOUSEHOLD_DATA_CACHE.get(cache_key)
    if base_matrix is None:
        data = pd.read_csv(data_path, header=None)

        desired_timescale = env.timescale
        simulation_length = env.simulation_length
        simulation_date = env.sim_starting_date.strftime('%Y-%m-%d %H:%M:%S')

        dataset_timescale = 15
        dataset_starting_date = '2022-01-01 00:00:00'

        if desired_timescale > dataset_timescale:
            data = data.groupby(
                data.index // (desired_timescale/dataset_timescale)).max()
        elif desired_timescale < dataset_timescale:
            data = data.loc[data.index.repeat(
                dataset_timescale/desired_timescale)].reset_index(drop=True)

        data = pd.concat([data, data], ignore_index=True)
        data['date'] = pd.date_range(
            start=dataset_starting_date, periods=data.shape[0], freq=f'{desired_timescale}min')

        year = int(dataset_starting_date.split('-')[0])
        simulation_date = f'{year}-{simulation_date.split("-")[1]}-{simulation_date.split("-")[2]}'
        simulation_index = data[data['date'] == simulation_date].index[0]
        data = data[simulation_index:simulation_index+simulation_length]
        data = data.drop(columns=['date'])
        base_matrix = data.to_numpy(dtype=np.float32)
        _FALLBACK_HOUSEHOLD_DATA_CACHE[cache_key] = base_matrix

    number_of_transformers = env.number_of_transformers
    k = min(10, base_matrix.shape[1])
    arr = np.empty((number_of_transformers, base_matrix.shape[0]), dtype=np.float32)
    for i in range(number_of_transformers):
        idx = env.tr_rng.integers(base_matrix.shape[1], size=k)
        arr[i] = base_matrix[:, idx].sum(axis=1)
    try:
        if not hasattr(env, '_dbg_inflex_once'):
            env._dbg_inflex_once = True
            if np.allclose(arr, 0.0):
                print("[WARN inflex] Generated inflexible loads are all zeros from fallback CSV. Check data file and scaling.")
    except Exception:
        pass
    env._cached_inflexible_loads = arr
    return arr


def generate_pv_generation(env) -> np.ndarray:
    '''
    This function loads the PV generation of each transformer by loading the data from a file
    and then adding minor variations to the data
    '''

    cached_env_arr = getattr(env, "_cached_pv_generation", None)
    if cached_env_arr is not None:
        return cached_env_arr

    # --- Load PV from household file (Parquet or CSVs) when requested ---
    sp_cfg = env.config.get('solar_power', {})
    use_household_file = bool(sp_cfg.get('data_from_household_file', False) or sp_cfg.get('data_from_household_csv', False))
    if use_household_file:
        household_df = _load_household_profiles(env)
        if household_df is None:
            raise ValueError("solar_power requested from household file but no household data available (check inflexible_loads.data_file or data_files)")

        # Check if solar column exists
        if 'solar' not in household_df.columns:
            household_df['solar'] = 0.0
        solar_series = household_df['solar'].to_numpy(dtype=np.float32, copy=False)
        factors = env.tr_rng.uniform(0.9, 1.1, size=(env.number_of_transformers, 1))
        arr = factors * solar_series[np.newaxis, :]
        env._cached_pv_generation = arr
        return arr

    # If no household solar data, check for external features with solar data
    external_df = _load_external_features(env)
    if external_df is not None and 'solar' in external_df.columns:
        solar_series = external_df['solar'].to_numpy(dtype=np.float32, copy=False)
        factors = env.tr_rng.uniform(0.9, 1.1, size=(env.number_of_transformers, 1))
        arr = factors * solar_series[np.newaxis, :]
        env._cached_pv_generation = arr
        return arr

    # If no NSW data available, fall back to the default dataset
    data_path = get_resource_path('ev2gym.data', 'pv_netherlands.csv')
    cache_key = (
        data_path,
        env.timescale,
        env.simulation_length,
    )
    base_series = _FALLBACK_PV_DATA_CACHE.get(cache_key)
    if base_series is None:
        data = pd.read_csv(data_path, sep=',', header=0)
        data.drop(['time', 'local_time'], inplace=True, axis=1)

        desired_timescale = env.timescale
        simulation_length = env.simulation_length
        simulation_date = env.sim_starting_date.strftime('%Y-%m-%d %H:%M:%S')

        dataset_timescale = 60
        dataset_starting_date = '2019-01-01 00:00:00'

        if desired_timescale > dataset_timescale:
            data = data.groupby(
                data.index // (desired_timescale/dataset_timescale)).max()
        elif desired_timescale < dataset_timescale:
            data = data.loc[data.index.repeat(
                dataset_timescale/desired_timescale)].reset_index(drop=True)

        window = max(1, 60//desired_timescale)
        data['electricity'] = data['electricity'].rolling(
            window=window, min_periods=1).mean()
        data['electricity'] = data['electricity'].ewm(
            span=window, adjust=True).mean()

        data = pd.concat([data, data], ignore_index=True)
        data['date'] = pd.date_range(
            start=dataset_starting_date, periods=data.shape[0], freq=f'{desired_timescale}min')

        year = int(dataset_starting_date.split('-')[0])
        simulation_date = f'{year}-{simulation_date.split("-")[1]}-{simulation_date.split("-")[2]}'
        simulation_index = data[data['date'] == simulation_date].index[0]
        data = data[simulation_index:simulation_index+simulation_length]
        base_series = data['electricity'].to_numpy(dtype=np.float32)
        _FALLBACK_PV_DATA_CACHE[cache_key] = base_series

    factors = env.tr_rng.uniform(0.9, 1.1, size=(env.number_of_transformers, 1))
    arr = factors * base_series[np.newaxis, :]
    env._cached_pv_generation = arr
    return arr


def load_transformers(env) -> List[Transformer]:
    '''Loads the transformers of the simulation
    If load_from_replay_path is None, then the transformers are created randomly

    Returns:
        - transformers: a list of transformer objects
    '''

    if env.load_from_replay_path is not None:
        return env.replay.transformers

    transformers = []

    if env.config['inflexible_loads']['include']:

        if env.scenario == 'private':
            inflexible_loads = generate_residential_inflexible_loads(env)

        # TODO add inflexible loads for public and workplace scenarios
        else:
            inflexible_loads = generate_residential_inflexible_loads(env)

    else:
        inflexible_loads = np.zeros((env.number_of_transformers,
                                    env.simulation_length))

    if env.config['solar_power']['include']:
        solar_power = generate_pv_generation(env)
    else:
        solar_power = np.zeros((env.number_of_transformers,
                                env.simulation_length))

    if env.charging_network_topology:
        # parse the topology file and create the transformers
        cs_counter = 0
        for i, tr in enumerate(env.charging_network_topology):
            cs_ids = []
            for cs in env.charging_network_topology[tr]['charging_stations']:
                cs_ids.append(cs_counter)
                cs_counter += 1
            transformer = Transformer(id=i,
                                      env=env,
                                      cs_ids=cs_ids,
                                      max_power=env.charging_network_topology[tr]['max_power'],
                                      inflexible_load=inflexible_loads[i, :],
                                      solar_power=solar_power[i, :],
                                      simulation_length=env.simulation_length
                                      )

            transformers.append(transformer)

    else:
        if env.number_of_transformers > env.cs:
            raise ValueError(
                'The number of transformers cannot be greater than the number of charging stations')
        for i in range(env.number_of_transformers):
            # get indexes where the transformer is connected
            transformer = Transformer(id=i,
                                      env=env,
                                      cs_ids=np.where(
                                          np.array(env.cs_transformers) == i)[0],
                                      max_power=env.config['transformer']['max_power'],
                                      inflexible_load=inflexible_loads[i, :],
                                      solar_power=solar_power[i, :],
                                      simulation_length=env.simulation_length
                                      )

            transformers.append(transformer)
    env.n_transformers = len(transformers)
    return transformers


def load_ev_charger_profiles(env) -> List[EV_Charger]:
    '''Loads the EV charger profiles of the simulation
    If load_from_replay_path is None, then the EV charger profiles are created randomly

    Returns:
        - ev_charger_profiles: a list of ev_charger_profile objects'''

    charging_stations = []
    if env.load_from_replay_path is not None:
        return env.replay.charging_stations

    v2g_enabled = env.config['v2g_enabled']

    if env.charging_network_topology:
        # parse the topology file and create the charging stations
        cs_counter = 0
        for i, tr in enumerate(env.charging_network_topology):
            for cs in env.charging_network_topology[tr]['charging_stations']:
                ev_charger = EV_Charger(id=cs_counter,
                                        connected_bus=0,
                                        connected_transformer=i,
                                        min_charge_current=env.charging_network_topology[tr][
                                            'charging_stations'][cs]['min_charge_current'],
                                        max_charge_current=env.charging_network_topology[tr][
                                            'charging_stations'][cs]['max_charge_current'],
                                        min_discharge_current=env.charging_network_topology[tr][
                                            'charging_stations'][cs]['min_discharge_current'],
                                        max_discharge_current=env.charging_network_topology[tr][
                                            'charging_stations'][cs]['max_discharge_current'],
                                        voltage=env.charging_network_topology[tr][
                                            'charging_stations'][cs]['voltage'],
                                        n_ports=env.charging_network_topology[tr][
                                            'charging_stations'][cs]['n_ports'],
                                        charger_type=env.charging_network_topology[tr][
                                            'charging_stations'][cs]['charger_type'],
                                        phases=env.charging_network_topology[tr]['charging_stations'][cs]['phases'],
                                        timescale=env.timescale,
                                        verbose=env.verbose,)
                cs_counter += 1
                charging_stations.append(ev_charger)
        env.cs = len(charging_stations)
        return charging_stations

    else:
        if v2g_enabled:
            max_discharge_current = env.config['charging_station']['max_discharge_current']
            min_discharge_current = env.config['charging_station']['min_discharge_current']
        else:
            max_discharge_current = 0
            min_discharge_current = 0

        for i in range(env.cs):
            ev_charger = EV_Charger(id=i,
                                    connected_bus=0,  # env.cs_buses[i],
                                    connected_transformer=env.cs_transformers[i],
                                    n_ports=env.number_of_ports_per_cs,
                                    max_charge_current=env.config['charging_station']['max_charge_current'],
                                    min_charge_current=env.config['charging_station']['min_charge_current'],
                                    max_discharge_current=max_discharge_current,
                                    min_discharge_current=min_discharge_current,
                                    phases=env.config['charging_station']['phases'],
                                    voltage=env.config['charging_station']['voltage'],
                                    timescale=env.timescale,
                                    verbose=env.verbose,)

            charging_stations.append(ev_charger)
        return charging_stations


def load_ev_profiles(env) -> List[EV]:
    '''Loads the EV profiles of the simulation
    If load_from_replay_path is None, then the EV profiles are created randomly

    Returns:
        - ev_profiles: a list of ev_profile objects'''

    # 1. If replay provided, just use it.
    if env.load_from_replay_path is not None:
        return env.replay.EVs

    # 2. Check for user-defined profiles (new schedule system)
    ev_profiles = generate_ev_profiles(env)
    if len(ev_profiles) > 0:
        return ev_profiles

    # 3. Fallback to legacy random spawners
    if env.scenario == 'GF':
        ev_profiles = EV_spawner_GF(env)
        while len(ev_profiles) == 0:
            ev_profiles = EV_spawner_GF(env)
        return ev_profiles

    ev_profiles = EV_spawner(env)
    while len(ev_profiles) == 0:
        ev_profiles = EV_spawner(env)

    return ev_profiles


def load_electricity_prices(env) -> Tuple[np.ndarray, np.ndarray]:
    '''Loads the electricity prices of the simulation
    If load_from_replay_path is None, then the electricity prices are created randomly

    Returns:
        - charge_prices: a matrix of size (number of charging stations, simulation length) with the charge prices
        - discharge_prices: a matrix of size (number of charging stations, simulation length) with the discharge prices'''

    if env.load_from_replay_path is not None:
        return env.replay.charge_prices, env.replay.discharge_prices

    # Try to load prices from external features first
    external_features = _load_external_features(env)
    price_col = None
    forecast_cols = []
    if external_features is not None:
        # Look for 'rrp' (for AUD) or any 'price' column as spot price
        if 'rrp' in [c.lower() for c in external_features.columns]:
            price_col = 'RRP' if 'RRP' in external_features.columns else 'rrp'
        else:
            for col in external_features.columns:
                if 'price' in col.lower():
                    price_col = col
                    break

        # Detect forecast columns like rrp_h01, rrp_h02, ... (case-insensitive)
        lower_cols = {c.lower(): c for c in external_features.columns}
        # Collect keys that match pattern rrp_h\d+
        import re
        pattern = re.compile(r"^rrp_h(\d+)$")
        numbered = []
        for lc, orig in lower_cols.items():
            m = pattern.match(lc)
            if m:
                numbered.append((int(m.group(1)), orig))
        numbered.sort(key=lambda t: t[0])
        forecast_cols = [orig for _, orig in numbered]

    if price_col:
        # Prices are assumed to be in $/MWh, converting to $/kWh
        prices = external_features[price_col].values / 1000
        charge_prices = np.tile(prices, (env.cs, 1))
        discharge_prices = np.tile(prices, (env.cs, 1))

        # Attach forecast matrix to env if available: shape (simulation_length, H)
        if forecast_cols:
            try:
                forecast_df = external_features[forecast_cols]
                # Convert $/MWh to $/kWh
                forecast_matrix = forecast_df.values / 1000
                # Ensure length matches simulation_length; truncate or pad if needed
                if len(forecast_matrix) >= env.simulation_length:
                    env.price_forecast = forecast_matrix[:env.simulation_length, :]
                else:
                    reps = math.ceil(env.simulation_length / len(forecast_matrix)) + 1
                    tiled = np.vstack([forecast_matrix] * reps)
                    env.price_forecast = tiled[:env.simulation_length, :]
                # Provide spot price vector for convenience
                env.spot_price = prices[:env.simulation_length]
            except Exception as e:
                print(f"Warning: failed to build price forecast from external features: {e}")
    else:
        # Fallback to original method if external features don't contain prices
        if env.price_data is None:
            # else load historical prices
            file_path = get_resource_path('ev2gym.data', 'Netherlands_day-ahead-2015-2024.csv')
            env.price_data = pd.read_csv(file_path, sep=',', header=0)
            # Standardize column name to 'price'
            if 'Price (EUR/MWhe)' in env.price_data.columns:
                env.price_data.rename(columns={'Price (EUR/MWhe)': 'price'}, inplace=True)
            elif 'Price ($/MWhe)' in env.price_data.columns:
                env.price_data.rename(columns={'Price ($/MWhe)': 'price'}, inplace=True)

            drop_columns = ['Country', 'Datetime (Local)']
            env.price_data.drop(drop_columns, inplace=True, axis=1, errors='ignore')

            env.price_data['year'] = pd.DatetimeIndex(env.price_data['Datetime (UTC)']).year
            env.price_data['month'] = pd.DatetimeIndex(env.price_data['Datetime (UTC)']).month
            env.price_data['day'] = pd.DatetimeIndex(env.price_data['Datetime (UTC)']).day
            env.price_data['hour'] = pd.DatetimeIndex(env.price_data['Datetime (UTC)']).hour

        # assume charge and discharge prices are the same
        # assume prices are the same for all charging stations

        data = env.price_data
        charge_prices = np.zeros((env.cs, env.simulation_length))
        discharge_prices = np.zeros((env.cs, env.simulation_length))
        # for every simulation step, take the price of the corresponding hour
        sim_temp_date = env.sim_date
        for i in range(env.simulation_length):

            year = sim_temp_date.year
            month = sim_temp_date.month
            day = sim_temp_date.day
            hour = sim_temp_date.hour
            # find the corresponding price
            try:
                price_value = data.loc[(data['year'] == year) & (data['month'] == month) & (data['day'] == day) & (data['hour'] == hour),
                                       'price'].iloc[0] / 1000  # price/kWh
                charge_prices[:, i] = price_value
                discharge_prices[:, i] = price_value
            except IndexError:
                print(
                    'Error: no price found for the given date and hour. Using 2022 prices instead.')

                year = 2022
                if day > 28:
                    day -= 1
                price_value = data.loc[(data['year'] == year) & (data['month'] == month) & (data['day'] == day) & (data['hour'] == hour),
                                       'price'].iloc[0] / 1000  # price/kWh
                charge_prices[:, i] = price_value
                discharge_prices[:, i] = price_value

            # step to next
            sim_temp_date = sim_temp_date + \
                datetime.timedelta(minutes=env.timescale)

    # Normalize discharge prices to POSITIVE credits (€/kWh revenue for export)
    # This aligns with cost calculation: net_cost = ... - discharging_kwh * discharge_price
    factor = env.config.get('discharge_price_factor', 1.0)
    factor = abs(factor)  # Force positive magnitude
    discharge_prices = discharge_prices * factor
    return charge_prices, discharge_prices


# ------------------------------------------------------------------------
# NSW household demand & solar helper
# ------------------------------------------------------------------------
def _load_household_profiles(env, ignore_date_filter: bool = False):
    """
    Load NSW household CSV traces specified in the YAML config
    (`inflexible_loads.data_files`) and resample them to the environment's
    timestep. Returns a DataFrame with at least the columns
    ['demand', 'solar'] **or** None if the config does not provide any files.

    If ignore_date_filter is False (default):
      - Filter by the simulation window derived from env.year/month/day/hour/minute
      - Resample to env.timescale
      - Ensure length == env.simulation_length by repeating/truncating
      - Reset index to a simple RangeIndex (backward compatible with existing usage)

    If ignore_date_filter is True (for forecasting use-cases):
      - Do NOT filter by date; use the entire dataset
      - Resample to env.timescale
      - Preserve DatetimeIndex and DO NOT truncate/pad to simulation_length
        (needed so forecasting can slice by actual timestamps and look ahead)
    """
    cfg = env.config.get('inflexible_loads', {})
    # New optional path: rolled-up Parquet with household_ids selection
    data_file = cfg.get('data_file')  # may be str or [str]
    household_ids = cfg.get('household_ids')
    if data_file is not None:
        # Normalize to string
        if isinstance(data_file, list):
            if len(data_file) == 0:
                return None
            data_file = data_file[0]
        if not os.path.exists(data_file):
            print(f"Error: Parquet file not found: {data_file}")
            return None

        household_ids_key: Optional[Tuple] = None
        if isinstance(household_ids, (list, tuple)) and len(household_ids) > 0:
            household_ids_key = tuple(sorted(household_ids))
        else:
            household_ids = []

        # Construct cache key including simulation window information
        cache_key = (
            "parquet",
            data_file,
            household_ids_key,
            env.timescale,
            ignore_date_filter,
            env.config.get('year'),
            env.config.get('month'),
            env.config.get('day'),
            env.config.get('hour'),
            env.config.get('minute'),
            env.simulation_length,
        )
        cached = _HOUSEHOLD_PROFILE_CACHE.get(cache_key)
        if cached is not None:
            return cached.copy()

        try:
            df_agg = _read_household_parquet(data_file, household_ids_key)
        except Exception as e:
            print(f"Error reading Parquet {data_file}: {e}")
            return None

        # Construct start/end
        year = env.config.get('year', 2019)
        month = env.config.get('month', 1)
        day = env.config.get('day', 1)
        hour = env.config.get('hour', 0)
        minute = env.config.get('minute', 0)
        start_date = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute)
        end_date = start_date + pd.Timedelta(minutes=env.timescale * env.simulation_length)

        # Enforce constant GMT+10 (no DST) for the entire dataset
        fixed_tz = pytz.FixedOffset(600)  # +10 hours
        if df_agg.index.tz is None:
            df_agg.index = df_agg.index.tz_localize(fixed_tz)
        else:
            # Convert any incoming tz (e.g., Australia/Sydney) to fixed +10
            df_agg.index = df_agg.index.tz_convert(fixed_tz)

        # Resample to env timescale
        df_rs = df_agg.resample(f"{env.timescale}min").mean().interpolate(method='time')
        if not ignore_date_filter:
            # Align boundaries to fixed +10 tz
            idx_tz = df_rs.index.tz
            if start_date.tzinfo is None:
                start_date = start_date.tz_localize(idx_tz)
                end_date = end_date.tz_localize(idx_tz)
            # Slice to [start_date, end_date] and then ensure exact length by pad/trunc
            df_rs = df_rs[(df_rs.index >= start_date) & (df_rs.index <= end_date)]
            # If empty, fallback to full resampled series
            if df_rs.empty:
                print(f"Warning: No data in {data_file} for date range {start_date} to {end_date}. Using full dataset.")
                df_use = df_rs
                df_use = df_agg.resample(f"{env.timescale}min").mean().interpolate(method='time')
            else:
                df_use = df_rs
            # Align length
            # If too short, repeat last value; if too long, truncate
            desired = int(env.simulation_length)
            if len(df_use) < desired and not df_use.empty:
                last_val = df_use.iloc[-1]
                need = desired - len(df_use)
                pad_index = pd.date_range(start=df_use.index[-1] + pd.Timedelta(minutes=env.timescale), periods=need, freq=f"{env.timescale}min")
                pad_df = pd.DataFrame([last_val.values] * need, index=pad_index, columns=df_use.columns)
                df_use = pd.concat([df_use, pad_df])
            if len(df_use) > desired:
                df_use = df_use.iloc[:desired]
            # Backward compatibility: return RangeIndex and only the needed columns
            df_out = df_use.reset_index(drop=False)
            df_out = df_out[["timestamp", "demand", "solar"]] if "timestamp" in df_out.columns else df_out.rename_axis("timestamp").reset_index()[["timestamp", "demand", "solar"]]
            _HOUSEHOLD_PROFILE_CACHE[cache_key] = df_out
            return df_out.copy()
        else:
            # Forecasting path: preserve DatetimeIndex, no truncation/padding
            _HOUSEHOLD_PROFILE_CACHE[cache_key] = df_rs
            return df_rs.copy()

    # Legacy path: list of CSVs in data_files
    file_paths = cfg.get('data_files', [])
    if not file_paths:
        return None

    # Construct start_date from year, month, day, hour, and minute
    year = env.config.get('year', 2019)
    month = env.config.get('month', 1)
    day = env.config.get('day', 1)
    hour = env.config.get('hour', 0)
    minute = env.config.get('minute', 0)

    # Create start and end dates for filtering (used when ignore_date_filter=False)
    start_date = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute)
    minutes_to_add = env.timescale * env.simulation_length
    end_date = start_date + pd.Timedelta(minutes=minutes_to_add)

    dfs = []
    for p in file_paths:
        try:
            df = _read_household_csv(p)
            # Expect at least 'demand' and 'solar' columns
            if not {'demand', 'solar'}.issubset(df.columns):
                raise ValueError(f"{p} must contain 'demand' and 'solar' columns")

            # Prepare index and optionally filter by date range
            df = df.set_index('interval_start')
            if not ignore_date_filter:
                filtered_df = df[(df.index >= start_date) & (df.index <= end_date)]
                # If filtered data is empty, use the original data with a warning
                if filtered_df.empty:
                    print(f"Warning: No data in {p} for date range {start_date} to {end_date}. Using full dataset.")
                    df = df
                else:
                    df = filtered_df
            # Resample to the simulation time-step and fill gaps by interpolation
            df = df.resample(f"{env.timescale}min").mean().interpolate(method='time')
            if not ignore_date_filter:
                # Backward compatibility: return RangeIndex for consumers that expect it
                df = df.reset_index(drop=False)
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {p}: {e}")
            continue

    if not dfs:
        raise ValueError("No valid household data files could be loaded")

    # Average multiple households if more than one file supplied
    data = pd.concat(dfs).groupby(level=0).mean()

    cache_key = (
        "csv",
        tuple(sorted(file_paths)),
        env.timescale,
        ignore_date_filter,
        env.config.get('year'),
        env.config.get('month'),
        env.config.get('day'),
        env.config.get('hour'),
        env.config.get('minute'),
        env.simulation_length,
    )
    cached = _HOUSEHOLD_PROFILE_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy()

    if ignore_date_filter:
        # DatetimeIndex preserved when ignore_date_filter=True
        _HOUSEHOLD_PROFILE_CACHE[cache_key] = data
        return data.copy()

    # Ensure we have at least env.simulation_length rows
    reps = math.ceil(env.simulation_length / len(data)) + 1
    data = pd.concat([data] * reps).iloc[:env.simulation_length]
    data.reset_index(drop=True, inplace=True)
    assert len(data) == env.simulation_length, \
        f"Household profile data length ({len(data)}) does not match simulation length ({env.simulation_length})"

    _HOUSEHOLD_PROFILE_CACHE[cache_key] = data
    return data.copy()


def _load_external_features(env):
    """
    Load external features dataset (weather, prices, etc.) and filter by date range.
    Returns a DataFrame with external features or None if the file doesn't exist.

    Uses the environment's year, month, day, timescale, and simulation_length
    to filter and resample the data.
    """
    if 'data_path' not in env.config:
        return None

    external_file = env.config['data_path']
    if not os.path.exists(external_file):
        fallback = os.path.join(env.config['data_path'], 'nsw_dataset', 'external_features', 'external_dataset.parquet')
        if not os.path.exists(fallback):
            return None
        external_file = fallback

    resolved_path = Path(external_file).resolve()
    try:
        stat = resolved_path.stat()
        file_sig = (str(resolved_path), stat.st_mtime_ns, stat.st_size)
    except FileNotFoundError:
        return None

    cache_key = (
        file_sig,
        env.timescale,
        env.simulation_length,
        env.config.get('year', 2019),
        env.config.get('month', 1),
        env.config.get('day', 1),
        env.config.get('hour', 0),
        env.config.get('minute', 0),
    )
    cached = _EXTERNAL_FEATURES_CACHE.get(cache_key)
    if cached is not None:
        setattr(env, "_external_features_cache_key", cache_key)
        return cached.copy(deep=True)

    try:
        df = _read_external_parquet(str(resolved_path))

        if not isinstance(df.index, pd.DatetimeIndex):
            datetime_candidates = [
                'interval_start',
                'timestamp',
                'Datetime (UTC)',
                'DatetimeUTC',
                'datetime_utc',
                'datetime',
            ]
            dt_col = None
            for candidate in datetime_candidates:
                if candidate in df.columns:
                    dt_col = candidate
                    break
            if dt_col is None:
                raise ValueError("External features file must have a DatetimeIndex or a recognizable datetime column")
            df[dt_col] = pd.to_datetime(df[dt_col], errors='coerce', utc=True)
            df = df.set_index(dt_col)

        if df.index.tz is not None:
            df.index = df.index.tz_convert('UTC').tz_localize(None)

        start_date = pd.Timestamp(
            year=env.config.get('year', 2019),
            month=env.config.get('month', 1),
            day=env.config.get('day', 1),
            hour=env.config.get('hour', 0),
            minute=env.config.get('minute', 0),
        )
        minutes_to_add = env.timescale * env.simulation_length
        end_date = start_date + pd.Timedelta(minutes=minutes_to_add)

        filtered_df = df[(df.index >= start_date) & (df.index <= end_date)]
        if filtered_df.empty:
            print(
                f"Warning: No data in external features for date range {start_date} to {end_date}. Using full dataset."
            )
            filtered_df = df

        filtered_df = filtered_df.resample(f"{env.timescale}min").mean().interpolate(method='time')

        if len(filtered_df) < env.simulation_length:
            print(
                f"Warning: Not enough external data rows ({len(filtered_df)}) for simulation length ({env.simulation_length}). Repeating data."
            )
            reps = math.ceil(env.simulation_length / len(filtered_df)) + 1
            filtered_df = pd.concat([filtered_df] * reps).iloc[:env.simulation_length]
        else:
            filtered_df = filtered_df.iloc[:env.simulation_length]

        _EXTERNAL_FEATURES_CACHE[cache_key] = filtered_df
        setattr(env, "_external_features_cache_key", cache_key)
        return filtered_df.copy(deep=True)

    except Exception as e:
        print(f"Error loading external features: {e}")
        return None


def load_weather_data(env) -> pd.DataFrame | None:
    """
    Extract temperature and wind features from the external dataset, aligned to the
    current simulation window and timescale.

    Returns a DataFrame indexed by time with available columns among:
    - 'temperature' (or closest match)
    - 'wind_speed' (or closest match)

    Also stores the result on `env.weather_data` for convenience.
    """
    existing_weather = getattr(env, "weather_data", None)
    if isinstance(existing_weather, pd.DataFrame) and not existing_weather.empty:
        return existing_weather

    external = _load_external_features(env)
    if external is None:
        env.weather_data = None
        return None

    # Case-insensitive column lookup helpers
    def pick_col(cands):
        lower_map = {c.lower(): c for c in external.columns}
        for key in cands:
            if key in lower_map:
                return lower_map[key]
        # fallback: substring search
        for lc, orig in lower_map.items():
            for key in cands:
                if key in lc:
                    return orig
        return None

    temp_col = pick_col(['temperature', 'temp', 't2m'])
    wind_col = pick_col(['wind_speed', 'windspeed', 'wind', 'ws'])

    cols: list[str] = []
    rename: Dict[str, str] = {}
    if temp_col:
        cols.append(temp_col)
        rename[temp_col] = 'temperature'
    if wind_col:
        cols.append(wind_col)
        rename[wind_col] = 'wind_speed'

    if not cols:
        print("Warning: No temperature/wind columns found in external features.")
        env.weather_data = None
        return None

    cache_key = getattr(env, "_external_features_cache_key", None)
    weather_key = None
    if cache_key is not None:
        weather_key = (
            cache_key,
            env.timescale,
            env.simulation_length,
            tuple(cols),
            tuple(sorted(rename.items())),
        )
        cached_weather = _WEATHER_CACHE.get(weather_key)
        if cached_weather is not None:
            env.weather_data = cached_weather
            return cached_weather

    weather = external[cols].rename(columns=rename)
    if weather_key is not None:
        _WEATHER_CACHE[weather_key] = weather
    env.weather_data = weather
    return weather