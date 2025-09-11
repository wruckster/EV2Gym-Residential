'''
This file contains the EVCity class, which is used to represent the environment of the city.
The environment is a gym environment and can be also used with the OpenAI gym standards and baselines.
The environment an also be used for standalone simulations without the gym environment.
'''

import gymnasium as gym
import pandas as pd
from gymnasium import spaces
import numpy as np
import datetime
import pickle
import os
import random
from copy import deepcopy
import yaml
import json

# from .grid import Grid
from ev2gym.models.utils.replay import EvCityReplay
from ev2gym.visuals.plots import ev_city_plot, visualize_step
from ev2gym.utilities.utils import get_statistics, print_statistics, calculate_charge_power_potential
from ev2gym.utilities.loaders import load_ev_spawn_scenarios, load_power_setpoints, load_transformers, load_ev_charger_profiles, load_ev_profiles, load_electricity_prices, _load_household_profiles
from ev2gym.models.utils.forecasting import create_lookahead_forecast
from ev2gym.visuals.render import Renderer

from ev2gym.rl_agent.reward import SquaredTrackingErrorReward
from ev2gym.rl_agent.state import PublicPST, LedgersPublicState
from ev2gym.rl_agent.ledgers import (
    GlobalLedgerBuffers,
    AccountLedgerBuffers,
    ColumnSpec,
)


class EV2Gym(gym.Env):

    def __init__(self,
                 config_file=None,
                 load_from_replay_path=None,  # path of replay file to load
                 replay_save_path='./replay/',  # where to save the replay file
                 generate_rnd_game=True,  # generate a random game without terminating conditions
                 seed=None,
                 save_replay=False,
                 save_plots=False,
                 state_function=LedgersPublicState,
                 reward_function=SquaredTrackingErrorReward,
                 cost_function=None,  # cost function to use in the simulation
                 eval_mode="Normal",  # eval mode can be "Normal", "Unstirred" or "Optimal" in order to save the correct statistics in the replay file
                 lightweight_plots=False,
                 # whether to empty the ports at the end of the simulation or not
                 empty_ports_at_end_of_simulation=True,
                 extra_sim_name=None,
                 verbose=False,
                 render_mode=None,
                 ):

        super(EV2Gym, self).__init__()
        # Initialize replay attribute immediately to prevent any and all race conditions.
        self.replay = None
        # Initialize stats dictionary immediately to prevent any and all race conditions.
        self.stats = {}
        # Initialize EV location data dictionary for tracking plug-in status
        self.ev_location_data = {}

        if verbose:
            print(f'Initializing EV2Gym environment...')

        # read yaml config file (ensure file is closed and use safe loader)
        assert config_file is not None, "Please provide a config file!!!"
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Debug flag: enable extra logging for setpoints/actions when True
        self.debug_setpoints = bool(self.config.get('debug_setpoints', False))

        self.forecasting_config = self.config.get('forecasting', {'enabled': False})
        self.demand_forecast = None
        self.solar_forecast = None
        self.temperature_forecast = None
        self.wind_forecast = None
        self.dr_event_forecast = None
        # Optional per-account series and forecasts (residential 1:1 household mapping)
        self.account_load_series = {}   # cs.id -> pd.Series at env timescale
        self.account_pv_series = {}     # cs.id -> pd.Series at env timescale
        self.account_load_forecast = {} # cs.id -> np.ndarray length 24
        self.account_pv_forecast = {}   # cs.id -> np.ndarray length 24

        self.ev_parameters = self.config.get('ev', {})
        # Accounts mode: roaming means a single account aggregates multiple CS/ports
        self.roaming_accounts: bool = bool(self.config.get('accounts', {}).get('roaming', False))

        self.generate_rnd_game = generate_rnd_game
        self.load_from_replay_path = load_from_replay_path
        self.empty_ports_at_end_of_simulation = empty_ports_at_end_of_simulation
        self.save_replay = save_replay
        self.save_plots = save_plots
        self.lightweight_plots = lightweight_plots
        self.eval_mode = eval_mode
        self.verbose = verbose  # Whether to print the simulation progress or not
        # Whether to render the simulation in real-time or not
        self.render_mode = render_mode

        self.reward_history = []
        self.total_evs_parked = []

        self.simulation_length = self.config['simulation_length']

        self.timestamps = []

        # Ledger scaffolding (initialized later in _init_ledgers)
        self.global_buffers = None
        self.account_buffers = {}

        # for backward compatibility
        self.replay_path = replay_save_path

        cs = self.config['number_of_charging_stations']

        self.reward_function = reward_function
        self.state_function = state_function
        self.cost_function = cost_function

        if seed is None:
            self.seed = np.random.randint(0, 1000000)
            # print(f"Random seed: {self.seed}")
        else:
            self.seed = seed
        # set random seed
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.tr_seed = self.config['tr_seed']
        if self.tr_seed == -1:
            self.tr_seed = self.seed
        self.tr_rng = np.random.default_rng(seed=self.tr_seed)

        if load_from_replay_path is not None:
            with open(load_from_replay_path, 'rb') as file:
                self.replay = pickle.load(file)

            sim_name = self.replay.replay_path.split(
                'replay_')[-1].split('.')[0]
            self.sim_name = sim_name + '_replay'
            self.sim_date = self.replay.sim_date
            self.timescale = self.replay.timescale
            self.cs = self.replay.n_cs
            self.number_of_transformers = self.replay.n_transformers
            self.number_of_ports_per_cs = self.replay.max_n_ports
            self.scenario = self.replay.scenario
            self.heterogeneous_specs = self.replay.heterogeneous_specs
            self.charging_stations = load_ev_charger_profiles(self)

        else:
            assert cs is not None, "Please provide the number of charging stations"
            self.cs = cs  # Number of charging stations
            # Threshold for the user satisfaction score

            self.number_of_ports_per_cs = self.config['number_of_ports_per_cs']
            self.number_of_transformers = self.config['number_of_transformers']
            self.timescale = self.config['timescale']
            self.scenario = self.config['scenario']
            self.simulation_length = int(self.config['simulation_length'])
            # Simulation time

            # household_df = _load_household_profiles(self)
            # if household_df is not None:
            #     # If the index is datetime, use it; otherwise, use the 'interval_start' column
            #     if isinstance(household_df.index, pd.DatetimeIndex):
            #         self.simulation_datetimes = household_df.index.to_list()
            #     elif 'interval_start' in household_df.columns:
            #         self.simulation_datetimes = pd.to_datetime(household_df['interval_start']).to_list()
            # else:
            #     self.simulation_datetimes = None

            # Initialize time-series arrays for plotting and analysis
            self.port_energy_level = np.zeros(
                (self.number_of_ports_per_cs, self.cs, self.simulation_length)
            )

            if self.config['random_day']:
                if "random_hour" in self.config:
                    if self.config["random_hour"]:
                        self.config['hour'] = random.randint(5, 15)

                self.sim_date = datetime.datetime(2022,
                                                  1,
                                                  1,
                                                  self.config['hour'],
                                                  self.config['minute'],
                                                  ) + datetime.timedelta(days=random.randint(0, int(1.5*365)))

                if self.scenario == 'workplace':
                    # dont simulate weekends
                    while self.sim_date.weekday() > 4:
                        self.sim_date += datetime.timedelta(days=1)

                if self.config['simulation_days'] == "weekdays":
                    # dont simulate weekends
                    while self.sim_date.weekday() > 4:
                        self.sim_date += datetime.timedelta(days=1)
                elif self.config['simulation_days'] == "weekends" and self.scenario != 'workplace':
                    # simulate only weekends
                    while self.sim_date.weekday() < 5:
                        self.sim_date += datetime.timedelta(days=1)
            else:

                self.sim_date = datetime.datetime(self.config['year'],
                                                  self.config['month'],
                                                  self.config['day'],
                                                  self.config['hour'],
                                                  self.config['minute'])
            self.sim_starting_date = self.sim_date
            self.sim_name = f'sim_' + \
                f'{datetime.datetime.now().strftime("%Y_%m_%d_%f")}'

            self.heterogeneous_specs = self.config['heterogeneous_ev_specs']

        # Whether to simulate the grid or not (Future feature...)
        self.simulate_grid = False

        # Set the simulation starting date
        self.sim_starting_date = self.sim_date

        # Read the config.charging_network_topology json file and read the topology
        try:
            with open(self.config['charging_network_topology']) as json_file:
                self.charging_network_topology = json.load(json_file)

        except FileNotFoundError:
            if not self.config['charging_network_topology'] == 'None':
                print(
                    f'Did not find file {self.config["charging_network_topology"]}')
            self.charging_network_topology = None

        self.sim_name = extra_sim_name + \
            self.sim_name if extra_sim_name is not None else self.sim_name

        # Simulate grid
        if self.simulate_grid:
            pass
            # self.grid = Grid(charging_stations=self.cs, case=case)
            # self.cs_buses = self.grid.get_charging_stations_buses()
            # self.cs_transformers = self.grid.get_bus_transformers()
        else:
            # self.cs_buses = [None] * self.cs
            if self.charging_network_topology is None:
                self.cs_transformers = [
                    *np.arange(self.number_of_transformers)] * (self.cs // self.number_of_transformers)
                self.cs_transformers += random.sample(
                    [*np.arange(self.number_of_transformers)], self.cs % self.number_of_transformers)
                random.shuffle(self.cs_transformers)

        # Instantiate Transformers
        self.transformers = load_transformers(self)
        for tr in self.transformers:
            tr.reset(step=0)

        # Instantiate Charging Stations
        if load_from_replay_path is None:
            self.charging_stations = load_ev_charger_profiles(self)
        for cs in self.charging_stations:
            cs.reset()

        # Initialize replay-related attributes
        self.replay_save_path = replay_save_path
        max_ports = max(cs.n_ports for cs in self.charging_stations)
        self.tr_solar_power = np.zeros((len(self.transformers), self.simulation_length))
        self.port_energy_level = np.zeros((max_ports, len(self.charging_stations), self.simulation_length))

        # Calculate the total number of ports in the simulation
        self.number_of_ports = np.array(
            [cs.n_ports for cs in self.charging_stations]).sum()

        # Load EV spawn scenarios
        if self.load_from_replay_path is None:
            load_ev_spawn_scenarios(self)

        # Spawn EVs
        self.EVs_profiles = load_ev_profiles(self)
        self.EVs = []

        # Load Electricity prices for every charging station
        self.price_data = None
        self.charge_prices, self.discharge_prices = load_electricity_prices(
            self)

        # Initialize global/account ledgers after topology and data are available
        self._init_ledgers()

        # Prepare arrays; setpoints will be generated after forecasts are updated
        self.current_power_usage = np.zeros(self.simulation_length)
        self.charge_power_potential = np.zeros(self.simulation_length)

        # Store full series for forecasting if enabled
        if self.forecasting_config.get('enabled'):
            # This assumes _load_household_profiles returns a DataFrame with all data
            # which can be used for lookaheads.
            self.full_timeseries_data = _load_household_profiles(self, ignore_date_filter=True)

        # Initialize per-account household series aligned to env timeline
        try:
            self._init_account_series()
        except Exception:
            # Non-fatal; account-level columns will be NaN if data is missing
            pass

        # --- FINAL INITIALIZATION ---
        # Initialize statistics now that all dependencies are loaded
        self.init_statistic_variables()

        # Variable showing whether the simulation is done or not
        self.done = False

        # Make folders for results
        if self.save_replay:
            os.makedirs(self.replay_save_path, exist_ok=True)

        if self.render_mode:
            # Initialize the rendering of the simulation
            self.renderer = Renderer(self)

        if self.save_plots:
            os.makedirs("./results", exist_ok=True)
            print(f"Creating directory: ./results/{self.sim_name}")
            os.makedirs(f"./results/{self.sim_name}", exist_ok=True)

        # Action space: is a vector of size "Sum of all ports of all charging stations"
        high = np.ones([self.number_of_ports])
        if self.config['v2g_enabled']:
            lows = -1 * np.ones([self.number_of_ports])
        else:
            lows = np.zeros([self.number_of_ports])
        self.action_space = spaces.Box(low=lows, high=high, dtype=np.float64)

        # Ensure forecasts are up-to-date BEFORE using them for setpoints or building observation space
        self._update_forecasts()

        # Now generate power setpoints (can use forecasts if enabled)
        self.power_setpoints = load_power_setpoints(self)

        # Create replay only after power_setpoints (and other fields) exist
        if self.save_replay and self.replay is None:
            self.replay = EvCityReplay(self)

        # Observation space: vector length of current observation (which may include forecasts)
        obs_dim = len(self._get_observation())

        high = np.inf*np.ones([obs_dim])
        self.observation_space = spaces.Box(
            low=-high, high=high, dtype=np.float64)

        # Observation mask: is a vector of size ("Sum of all ports of all charging stations") showing in which ports an EV is connected
        self.observation_mask = np.zeros(self.number_of_ports)

        self._log_initial_state()

    def reset(self, seed=None, options=None, **kwargs):
        '''Resets the environment to its initial state'''

        if seed is None:
            self.seed = np.random.randint(0, 1000000)
        else:
            self.seed = seed

        # set random seed
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.tr_seed == -1:
            self.tr_seed = self.seed
        self.tr_rng = np.random.default_rng(seed=self.tr_seed)

        self.current_step = 0
        self.stats = None

        # Reset all charging stations to ensure a clean state before spawning EVs
        for cs in self.charging_stations:
            cs.reset()

        for tr in self.transformers:
            tr.reset(step=self.current_step)

        if self.load_from_replay_path is None or not self.config['random_day']:
            self.sim_date = self.sim_starting_date
        else:
            # select random date in range

            if "random_hour" in self.config:
                if self.config["random_hour"]:
                    self.config['hour'] = random.randint(5, 15)

            self.sim_date = datetime.datetime(2022,
                                              1,
                                              1,
                                              self.config['hour'],
                                              self.config['minute'],
                                              ) + datetime.timedelta(days=random.randint(0, int(1.5*365)))

            if self.scenario == 'workplace':
                # dont simulate weekends
                while self.sim_date.weekday() > 4:
                    self.sim_date += datetime.timedelta(days=1)

            if self.config['simulation_days'] == "weekdays":
                # dont simulate weekends
                while self.sim_date.weekday() > 4:
                    self.sim_date += datetime.timedelta(days=1)
            elif self.config['simulation_days'] == "weekends" and self.scenario != 'workplace':
                # simulate only weekends
                while self.sim_date.weekday() < 5:
                    self.sim_date += datetime.timedelta(days=1)

        self.sim_starting_date = self.sim_date
        self.EVs_profiles = load_ev_profiles(self)
        self.power_setpoints = load_power_setpoints(self)
        self.EVs = []

        # Re-initialize ledgers for the new episode (timestamps realign, schemas consistent)
        self._init_ledgers()

        # Populate initial row (t=0) with time features and any known values
        # so consumers can access time columns immediately after reset.
        self.current_step = 0
        try:
            self._update_global_ledger_row()
            self._update_account_ledger_row()
        except Exception:
            # Keep reset robust; ledger rows will be populated on first step
            pass

        # Optional concise debug: summarize EV profiles and power setpoints
        if getattr(self, 'debug_setpoints', False):
            try:
                ev_count = len(self.EVs_profiles) if self.EVs_profiles is not None else 0
                arr_times = [ev.time_of_arrival for ev in self.EVs_profiles] if ev_count > 0 else []
                dep_times = [ev.time_of_departure for ev in self.EVs_profiles] if ev_count > 0 else []
                nonzero = int(np.count_nonzero(self.power_setpoints)) if self.power_setpoints is not None else 0
                max_price = float(np.max(np.abs(self.charge_prices[0]))) if hasattr(self, 'charge_prices') else float('nan')
                preview = np.round(self.power_setpoints[:20], 3) if self.power_setpoints is not None else []
                print(f"[DBG reset] EVs={ev_count} arrivals=[{min(arr_times) if arr_times else 'NA'},{max(arr_times) if arr_times else 'NA'}] "
                      f"deps=[{min(dep_times) if dep_times else 'NA'},{max(dep_times) if dep_times else 'NA'}] "
                      f"nonzero_setpoints={nonzero} max_price={max_price:.4f} preview(20)={preview}")
            except Exception as e:
                print(f"[DBG reset] summary error: {e}")

        # print(f'Simulation starting date: {self.sim_date}')

        # self.sim_name = f'ev_city_{self.simulation_length}_' + \
        # f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}'

        # Spawn EVs with arrival time 0
        self._spawn_evs_at_current_step()

        self.init_statistic_variables()

        self._log_initial_state()
        self._update_forecasts()

        return self._get_observation(), {}

    def init_statistic_variables(self):
        '''
        Initializes the variables used for keeping simulation statistics
        '''
        self.current_step = 0
        self.total_evs_spawned = 0
        self.total_reward = 0

        self.current_ev_departed = 0
        self.current_ev_arrived = 0
        self.current_evs_parked = 0
        
        # Initialize/reset EV location data for tracking
        self.ev_location_data = np.zeros(
            (self.number_of_ports_per_cs, self.cs, self.simulation_length),
            dtype=np.int8
        )

        self.previous_power_usage = self.current_power_usage
        self.current_power_usage = np.zeros(self.simulation_length)

        # self.transformer_amps = np.zeros([self.number_of_transformers,
        #                                   self.simulation_length])

        self.cs_power = np.zeros([self.cs, self.simulation_length])
        self.cs_current = np.zeros([self.cs, self.simulation_length])

        self.tr_overload = np.zeros(
            [self.number_of_transformers, self.simulation_length])

        self.tr_inflexible_loads = np.zeros(
            [self.number_of_transformers, self.simulation_length])

        self.tr_solar_power = np.zeros(
            [self.number_of_transformers, self.simulation_length])

        # New structures for detailed plotting
        self.energy_flow_breakdown = {
            'grid_draw': np.zeros(self.simulation_length, dtype=np.float32),
            'solar_production': np.zeros(self.simulation_length, dtype=np.float32),
            'ev_power': np.zeros(self.simulation_length, dtype=np.float32),
            'inflexible_load': np.zeros(self.simulation_length, dtype=np.float32), 

        }
        self.cost_history = np.zeros(self.simulation_length, dtype=np.float32)

        # self.port_power = np.zeros([self.number_of_ports,
        #                             self.cs,
        #                             self.simulation_length],
        #                            dtype=np.float16)
        if not self.lightweight_plots:
            self.port_current = np.zeros([self.number_of_ports,
                                          self.cs,
                                          self.simulation_length],
                                         dtype=np.float16,
                                         )
            self.port_current_signal = np.zeros([self.number_of_ports,
                                                 self.cs,
                                                 self.simulation_length],
                                                dtype=np.float16,
                                                )

            self.port_energy_level = np.zeros([self.number_of_ports,
                                               self.cs,
                                               self.simulation_length],
                                              dtype=np.float16)
            # self.port_charging_cycles = np.zeros([self.number_of_ports,
            #                                       self.cs,
            #                                       self.simulation_length],
            #                                      dtype=np.float16)
            self.port_arrival = dict({f'{j}.{i}': []
                                      for i in range(self.number_of_ports)
                                      for j in range(self.cs)})

        self.done = False

    def step(self, actions, visualize=False):
        ''''
        Takes an action as input and returns the next state, reward, and whether the episode is done
        Inputs:
            - actions: is a vector of size "Sum of all ports of all charging stations taking values in [-1,1]"
        Returns:
            - observation: is a matrix with the complete observation space
            - reward: is a scalar value representing the reward of the current step
            - done: is a boolean value indicating whether the episode is done or not
        '''
        assert not self.done, "Episode is done, please reset the environment"

        if self.verbose:
            print(f"Step: {self.current_step}/{self.simulation_length}")

        # Spawn EVs with arrival time equal to the current step
        self._spawn_evs_at_current_step()

        # Reset the current number of EVs departed and arrived
        self.current_ev_departed = 0
        self.current_ev_arrived = 0

        # Update EV states and drain battery for commuting EVs
        for cs in self.charging_stations:
            for ev in cs.evs_connected:
                if ev is not None:
                    ev.update_location_state(self.current_step)
                    if ev.location_state == 2:  # If commuting
                        ev.drain_commuting_battery(distance_km=1)  # Assume 1km per step

        # Reset power usage for this timestep to zero before processing charging stations
        self.current_power_usage[self.current_step] = 0.0

        # Add inflexible loads and solar power from transformers to the current power usage
        for tr in self.transformers:
            # Reset sets the current_power to inflexible_load + solar_power for the current step
            tr.reset(step=self.current_step)
            self.current_power_usage[self.current_step] += tr.current_power

        total_costs = 0
        total_invalid_action_punishment = 0
        user_satisfaction_list = []
        self.departing_evs = []

        port_counter = 0

        # Call step for each charging station and spawn EVs where necessary
        for i, cs in enumerate(self.charging_stations):
            n_ports = cs.n_ports
            costs, user_satisfaction, invalid_action_punishment, ev = cs.step(
                actions[port_counter:port_counter + n_ports],
                self.charge_prices[cs.id, self.current_step],
                self.discharge_prices[cs.id, self.current_step])

            # Store departing EV for logging / statistics
            self.departing_evs.append(ev)

            for u in user_satisfaction:
                user_satisfaction_list.append(u)

            self.current_power_usage[self.current_step] += cs.current_power_output

            # Update transformer variables for this timestep
            self.transformers[cs.connected_transformer].step(
                cs.current_total_amps, cs.current_power_output)

            total_costs += costs
            total_invalid_action_punishment += invalid_action_punishment
            self.current_ev_departed += len(user_satisfaction)

            port_counter += n_ports

        # Spawn EVs
        counter = self.total_evs_spawned
        
        for i, ev in enumerate(self.EVs_profiles[counter:]):
            if ev.time_of_arrival == self.current_step:
                ev = deepcopy(ev)
                ev.reset()
                ev.simulation_length = self.simulation_length
                
                # Spawn the new EV at its designated charging station
                cs_index = ev.location
                if 0 <= cs_index < len(self.charging_stations):     # verifies that the EV is assigned to a valid charging station
                    target_cs = self.charging_stations[cs_index]
                    # Only connect if EV isn't already connected somewhere else
                    if (target_cs.n_evs_connected < target_cs.n_ports and 
                        ev.id not in [ev.id for ev in self.EVs]):
                        spawn_result = target_cs.spawn_ev(ev)
                        if spawn_result is not None:  # Spawn successful
                            self.EVs.append(ev)
                            self.total_evs_spawned += 1
                            self.current_ev_arrived += 1
                        else:
                            if self.verbose:
                                print(f"⚠️ EV {ev.id} could not spawn at station {cs_index}: charger is full.")
                    else:
                        if self.verbose:
                            print(f"⚠️ EV {ev.id} could not spawn at station {cs_index}: charger is full.")
                else:
                    print(f"Warning: EV {ev.id} has invalid location {cs_index} and was not spawned.")

            elif ev.time_of_arrival > self.current_step:
                break

        # Update EV location states based on schedule transitions
        for ev in self.EVs:
            ev.update_location_state(self.current_step)

        # Update power stats arrays for this step
        self._update_power_statistics(self.departing_evs)

        # Stash invalid action punishment for this step so the global ledger can include it
        self._pending_invalid_action_punishment = float(total_invalid_action_punishment)

        # Refresh forecasts/series so current-step weather and lookaheads are up-to-date
        try:
            self._update_forecasts()
        except Exception:
            pass

        # Ledger writes for current step
        self._update_global_ledger_row()
        self._update_account_ledger_row()

        # Track EVs parked count
        self.current_evs_parked += self.current_ev_arrived - self.current_ev_departed

        # Compute reward
        if self.simulate_grid:
            # TODO: transform actions -> grid_actions
            raise NotImplementedError
        else:
            reward = self._calculate_reward(total_costs,
                                            user_satisfaction_list,
                                            total_invalid_action_punishment)

        # Optional cost
        if self.cost_function is not None:
            cost = self.cost_function(self,
                                      total_costs,
                                      user_satisfaction_list,
                                      total_invalid_action_punishment)
        else:
            cost = None

        info = {
            'cost': cost,
            'total_power_usage': self.current_power_usage[self.current_step - 1],
            'power_setpoint': self.power_setpoints[self.current_step - 1] if self.power_setpoints is not None else 0,
            'pv_generation': np.sum([tr.solar_power[self.current_step - 1] for tr in self.transformers]),
            'ev_soc': np.mean(self.port_energy_level[:, :, self.current_step - 1][self.port_energy_level[:, :, self.current_step - 1] > 0]) if np.any(self.port_energy_level[:, :, self.current_step - 1] > 0) else 0,
            'num_evs_parked': self.current_evs_parked,
        }

        self.reward_history.append(reward)
        # Finalize reward-dependent fields for row t = current_step - 1
        try:
            self._finalize_reward_for_row(self.current_step - 1, float(reward))
        except Exception:
            pass
        self.total_evs_parked.append(len(self.EVs))

        if visualize:
            visualize_step(self)

        self.render()

        # Record solar power for the current step if within simulation bounds
        if self.current_step < self.simulation_length:
            for i, tr in enumerate(self.transformers):
                self.tr_solar_power[i, self.current_step] = tr.solar_power[self.current_step]

        # Record port energy levels for the current step for plotting
        if self.current_step < self.simulation_length:
            for i, cs in enumerate(self.charging_stations):
                for j in range(self.number_of_ports_per_cs):
                    if j < cs.n_ports and cs.evs_connected[j] is not None:
                        self.port_energy_level[j, i, self.current_step] = cs.evs_connected[j].get_soc()
                    else:
                        self.port_energy_level[j, i, self.current_step] = 0

        # Track EV locations and plug-in status for this timestep
        self._update_ev_location_data()

        # Check termination conditions and return the appropriate values
        obs = self._get_observation()
        
        return self._check_termination(reward, info)

    def render(self):
        '''Renders the simulation'''
        if self.render_mode:
            self.renderer.render()

    def save_ledgers_parquet(self, dir_path: str) -> dict:
        """Export the global and per-account ledgers to Parquet files.

        Files written:
          - global.parquet
          - account_{id}.parquet for each charging station/account

        Returns a dict with keys 'global' and 'accounts' mapping to file paths.
        """
        assert self.global_buffers is not None, "global_buffers not initialized"
        os.makedirs(dir_path, exist_ok=True)
        out: dict = {"global": "", "accounts": {}}
        global_path = os.path.join(dir_path, "global.parquet")
        self.global_buffers.to_parquet(global_path)
        out["global"] = global_path
        # accounts
        for cs in self.charging_stations:
            buf = self.account_buffers.get(cs.id)
            if buf is None:
                continue
            path = os.path.join(dir_path, f"account_{cs.id}.parquet")
            buf.to_parquet(path)
            out["accounts"][cs.id] = path
        return out

    def _save_sim_replay(self):
        '''Saves the simulation data in a pickle file'''
        replay = EvCityReplay(self)
        print(f"Saving replay file at {replay.replay_path}")
        with open(replay.replay_path, 'wb') as f:
            pickle.dump(replay, f)

        return replay.replay_path

    def _init_ledgers(self) -> None:
        """Initialize global and per-account ledgers with timestamp plus basic columns.
        Columns are derived from current topology (chargers/ports).
        """
        try:
            # Build a deterministic timestamp array across the full simulation
            # Use pandas for convenience then convert to numpy datetime64[ns]
            freq = pd.Timedelta(minutes=int(self.timescale))
            ts = pd.date_range(start=self.sim_starting_date,
                               periods=int(self.simulation_length),
                               freq=freq)
            timestamps_np = ts.to_numpy(dtype='datetime64[ns]')
        except Exception:
            # Fallback: compute via numpy from start date
            start = np.datetime64(self.sim_starting_date, 'ns')
            step = np.timedelta64(int(self.timescale), 'm')
            timestamps_np = start + np.arange(int(self.simulation_length)) * step

        # Build Global schema
        global_cols: list[ColumnSpec] = []
        # meta
        global_cols.append(ColumnSpec("step", np.dtype(np.int32)))
        global_cols.append(ColumnSpec("step_ratio", np.dtype(np.float32)))
        global_cols.append(ColumnSpec("dow", np.dtype(np.int8)))
        global_cols.append(ColumnSpec("hour", np.dtype(np.int8)))
        global_cols.append(ColumnSpec("minute", np.dtype(np.int8)))
        # forecasts (H=24) – price forecast horizon columns
        for h in range(24):
            global_cols.append(ColumnSpec(f"price_fc_h{h+1:02d}", np.dtype(np.float32)))
        # weather/DR forecasts (H=24)
        for h in range(24):
            global_cols.append(ColumnSpec(f"temp_fc_h{h+1:02d}", np.dtype(np.float32)))
        for h in range(24):
            global_cols.append(ColumnSpec(f"wind_fc_h{h+1:02d}", np.dtype(np.float32)))
        # totals
        global_cols.append(ColumnSpec("total_power_usage_kw", np.dtype(np.float32)))
        global_cols.append(ColumnSpec("power_setpoint_kw", np.dtype(np.float32)))
        global_cols.append(ColumnSpec("ev_power_kw", np.dtype(np.float32)))
        global_cols.append(ColumnSpec("inflexible_load_kw", np.dtype(np.float32)))
        global_cols.append(ColumnSpec("solar_production_kw", np.dtype(np.float32)))
        # current prices (averaged across CS where applicable)
        global_cols.append(ColumnSpec("charge_price", np.dtype(np.float32)))
        global_cols.append(ColumnSpec("discharge_price", np.dtype(np.float32)))
        # current weather observations (aligned to env timeline when available)
        global_cols.append(ColumnSpec("temp_c", np.dtype(np.float32)))
        global_cols.append(ColumnSpec("wind_speed", np.dtype(np.float32)))
        global_cols.append(ColumnSpec("evs_parked", np.dtype(np.int16)))
        # tracking/cost/reward
        global_cols.append(ColumnSpec("tracking_error", np.dtype(np.float32)))
        global_cols.append(ColumnSpec("invalid_action_punishment", np.dtype(np.float32)))
        global_cols.append(ColumnSpec("reward_step", np.dtype(np.float32)))
        global_cols.append(ColumnSpec("reward_cumsum", np.dtype(np.float32)))
        # No per-CS fields in global ledger to avoid mirroring; these live in account buffers.

        self.global_buffers = GlobalLedgerBuffers(
            timestamps=timestamps_np,
            columns=global_cols,
        )

        # Account buffers: assume one account per charging station (subject to change later)
        # Account schema (supports roaming single-account mode)
        if self.roaming_accounts:
            account_cols: list[ColumnSpec] = []
            account_cols.append(ColumnSpec("step", np.dtype(np.int32)))
            account_cols.append(ColumnSpec("cs_power_kw", np.dtype(np.float32)))  # aggregated over all CS
            account_cols.append(ColumnSpec("cs_amps", np.dtype(np.float32)))      # aggregated over all CS
            account_cols.append(ColumnSpec("account_power_setpoint_kw", np.dtype(np.float32)))
            account_cols.append(ColumnSpec("household_inflexible_load_kw", np.dtype(np.float32)))
            account_cols.append(ColumnSpec("household_pv_kw", np.dtype(np.float32)))
            account_cols.append(ColumnSpec("tracking_error_account", np.dtype(np.float32)))
            account_cols.append(ColumnSpec("evs_connected", np.dtype(np.int16)))      # total across CS
            account_cols.append(ColumnSpec("cs_kw_limit", np.dtype(np.float32)))      # total/aggregate
            # simplified per-account EV status (single EV in residential roaming)
            account_cols.append(ColumnSpec("soc", np.dtype(np.float32)))
            account_cols.append(ColumnSpec("time_to_departure", np.dtype(np.float32)))
            account_cols.append(ColumnSpec("time_since_arrival", np.dtype(np.float32)))
            # per-account forecasts H=24
            for h in range(24):
                account_cols.append(ColumnSpec(f"load_fc_h{h+1:02d}", np.dtype(np.float32)))
            for h in range(24):
                account_cols.append(ColumnSpec(f"pv_fc_h{h+1:02d}", np.dtype(np.float32)))

            self.account_buffers = {0: AccountLedgerBuffers(
                account_id=0,
                timestamps=timestamps_np,
                columns=account_cols,
            )}
        else:
            for cs in self.charging_stations:
                account_cols: list[ColumnSpec] = []
                account_cols.append(ColumnSpec("step", np.dtype(np.int32)))
                account_cols.append(ColumnSpec("cs_power_kw", np.dtype(np.float32)))
                account_cols.append(ColumnSpec("cs_amps", np.dtype(np.float32)))
                account_cols.append(ColumnSpec("account_power_setpoint_kw", np.dtype(np.float32)))
                account_cols.append(ColumnSpec("household_inflexible_load_kw", np.dtype(np.float32)))
                account_cols.append(ColumnSpec("household_pv_kw", np.dtype(np.float32)))
                account_cols.append(ColumnSpec("tracking_error_account", np.dtype(np.float32)))
                account_cols.append(ColumnSpec("evs_connected", np.dtype(np.int16)))
                account_cols.append(ColumnSpec("cs_kw_limit", np.dtype(np.float32)))
                for p in range(cs.n_ports):
                    account_cols.append(ColumnSpec(f"port{p}_amps", np.dtype(np.float32)))
                    account_cols.append(ColumnSpec(f"port{p}_soc", np.dtype(np.float32)))
                    account_cols.append(ColumnSpec(f"port{p}_action_norm", np.dtype(np.float32)))
                    account_cols.append(ColumnSpec(f"port{p}_amps_limit", np.dtype(np.float32)))
                    account_cols.append(ColumnSpec(f"port{p}_connected", np.dtype(np.int8)))
                    account_cols.append(ColumnSpec(f"port{p}_time_to_departure", np.dtype(np.float32)))
                    account_cols.append(ColumnSpec(f"port{p}_time_since_arrival", np.dtype(np.float32)))
                    account_cols.append(ColumnSpec(f"port{p}_action", np.dtype(np.float32)))
                    account_cols.append(ColumnSpec(f"port{p}_is_charging", np.dtype(np.int8)))
                    account_cols.append(ColumnSpec(f"port{p}_is_discharging", np.dtype(np.int8)))
                for h in range(24):
                    account_cols.append(ColumnSpec(f"load_fc_h{h+1:02d}", np.dtype(np.float32)))
                for h in range(24):
                    account_cols.append(ColumnSpec(f"pv_fc_h{h+1:02d}", np.dtype(np.float32)))
                self.account_buffers[cs.id] = AccountLedgerBuffers(
                    account_id=cs.id,
                    timestamps=timestamps_np,
                    columns=account_cols,
                )

    def _update_global_ledger_row(self) -> None:
        """Write current-step global values into global_buffers."""
        t = self.current_step
        if self.global_buffers is None:
            return
        # time features for step t (before increment): use current sim_date
        try:
            dow = int(self.sim_date.weekday())
            hour = int(self.sim_date.hour)
            minute = int(self.sim_date.minute)
        except Exception:
            dow = -1
            hour = -1
            minute = -1
        denom = max(1, int(self.simulation_length) - 1)
        step_ratio = float(t / denom)
        values = {
            "step": int(t),
            "step_ratio": step_ratio,
            "dow": np.int8(dow),
            "hour": np.int8(hour),
            "minute": np.int8(minute),
            "total_power_usage_kw": float(self.current_power_usage[t]) if t < self.simulation_length else np.nan,
            "power_setpoint_kw": float(self.power_setpoints[t]) if self.power_setpoints is not None and t < len(self.power_setpoints) else np.nan,
            "ev_power_kw": float(self.energy_flow_breakdown['ev_power'][t]) if 'ev_power' in self.energy_flow_breakdown else float(np.sum(self.cs_power[:, t])) if t < self.simulation_length else np.nan,
            "inflexible_load_kw": float(self.energy_flow_breakdown['inflexible_load'][t]) if 'inflexible_load' in self.energy_flow_breakdown else np.nan,
            "solar_production_kw": float(self.energy_flow_breakdown['solar_production'][t]) if 'solar_production' in self.energy_flow_breakdown else np.nan,
            # current prices averaged across CS
            "charge_price": (float(np.mean(self.charge_prices[:, t])) if hasattr(self, 'charge_prices') and t < getattr(self.charge_prices, 'shape', [0, 0])[1] else np.nan),
            "discharge_price": (float(np.mean(self.discharge_prices[:, t])) if hasattr(self, 'discharge_prices') and t < getattr(self.discharge_prices, 'shape', [0, 0])[1] else np.nan),
            # current weather observations if available
            "temp_c": (float(self.temperature_series[t]) if hasattr(self, 'temperature_series') and isinstance(getattr(self, 'temperature_series'), (list, np.ndarray)) and t < len(self.temperature_series) else np.nan),
            "wind_speed": (float(self.wind_speed_series[t]) if hasattr(self, 'wind_speed_series') and isinstance(getattr(self, 'wind_speed_series'), (list, np.ndarray)) and t < len(self.wind_speed_series) else np.nan),
            "evs_parked": int(self.current_evs_parked),
            # Defaults for tracking/cost/reward; reward fields will be finalized after reward calculation
            "tracking_error": np.nan,
            "invalid_action_punishment": float(getattr(self, '_pending_invalid_action_punishment', np.nan)),
            "reward_step": np.nan,
            "reward_cumsum": np.nan,
        }
        # price forecast horizon values
        try:
            pf = getattr(self, 'price_forecast', None)
            row = pf[t] if pf is not None and t < len(pf) else None
        except Exception:
            row = None
        for h in range(24):
            key = f"price_fc_h{h+1:02d}"
            if row is not None and h < len(row):
                try:
                    values[key] = float(row[h])
                except Exception:
                    values[key] = np.nan
            else:
                values[key] = np.nan
        # Temperature forecast (float32)
        try:
            tfc = getattr(self, 'temp_forecast', None)
        except Exception:
            tfc = None
        for h in range(24):
            key = f"temp_fc_h{h+1:02d}"
            if tfc is not None and h < len(tfc):
                try:
                    values[key] = np.float32(float(tfc[h]))
                except Exception:
                    values[key] = np.nan
            else:
                values[key] = np.nan
        # Wind forecast (float32)
        try:
            wfc = getattr(self, 'wind_forecast', None)
        except Exception:
            wfc = None
        for h in range(24):
            key = f"wind_fc_h{h+1:02d}"
            if wfc is not None and h < len(wfc):
                try:
                    values[key] = np.float32(float(wfc[h]))
                except Exception:
                    values[key] = np.nan
            else:
                values[key] = np.nan
        # DR forecast (binary/int8)
        try:
            dfc = getattr(self, 'dr_event_forecast', None)
        except Exception:
            dfc = None
        for h in range(24):
            key = f"dr_fc_h{h+1:02d}"
            if dfc is not None and h < len(dfc):
                try:
                    values[key] = np.int8(int(dfc[h] > 0.5))
                except Exception:
                    values[key] = np.int8(0)
            else:
                values[key] = np.int8(0)
        # No per-CS fields written into the global ledger.
        self.global_buffers.set_row(t, values)

    def _finalize_reward_for_row(self, t: int, reward_value: float) -> None:
        """Fill in reward_step, reward_cumsum, and tracking_error for row t after reward is computed."""
        if self.global_buffers is None or t < 0 or t >= int(self.simulation_length):
            return
        # compute tracking error at t using available series
        try:
            setpt = float(self.power_setpoints[t]) if self.power_setpoints is not None else np.nan
            cpp = float(self.charge_power_potential[t]) if t < len(self.charge_power_potential) else np.nan
            usage = float(self.current_power_usage[t]) if t < len(self.current_power_usage) else np.nan
            if np.isfinite(setpt) and np.isfinite(cpp) and np.isfinite(usage):
                target = min(setpt, cpp)
                trk_err = float((target - usage) ** 2)
            else:
                trk_err = np.nan
        except Exception:
            trk_err = np.nan
        # reward cumsum
        try:
            rc = float(np.nansum(self.reward_history[: t + 1])) if hasattr(self, 'reward_history') else reward_value
        except Exception:
            rc = reward_value
        self.global_buffers.update_values(t, {
            "tracking_error": trk_err,
            "reward_step": float(reward_value),
            "reward_cumsum": float(rc),
        })

    def _init_account_series(self) -> None:
        """Populate per-account household demand/PV series aligned to env timeline.

        Reads the filtered, resampled household profiles and stores:
          - self.account_load_series[account_id] -> pd.Series (kW)
          - self.account_pv_series[account_id] -> pd.Series (kW)

        In roaming mode, uses account_id=0. Otherwise assigns per CS id.
        """
        try:
            df = _load_household_profiles(self, ignore_date_filter=False)
        except Exception:
            return
        if not isinstance(df, pd.DataFrame) or df.empty:
            return
        # Build a time-indexed frame from loader output. It may return a RangeIndex with a 'timestamp' column.
        if 'timestamp' in df.columns:
            dt_index = pd.to_datetime(df['timestamp'])
            # Enforce fixed +10 tz to match loader behavior
            if dt_index.tz is None:
                try:
                    import pytz
                    dt_index = dt_index.tz_localize(pytz.FixedOffset(600))
                except Exception:
                    pass
            ts_df = df[['demand', 'solar']].copy()
            ts_df.index = dt_index
        else:
            # Already time-indexed
            ts_df = df[['demand', 'solar']].copy()
            dt_index = ts_df.index
        # Determine target timeline aligned with env start and tz
        start_date = pd.Timestamp(year=self.config.get('year', 2019),
                                  month=self.config.get('month', 1),
                                  day=self.config.get('day', 1),
                                  hour=self.config.get('hour', 0),
                                  minute=self.config.get('minute', 0))
        if getattr(dt_index, 'tz', None) is not None and start_date.tzinfo is None:
            start_date = start_date.tz_localize(dt_index.tz)
        idx = pd.date_range(start=start_date,
                            periods=int(self.simulation_length),
                            freq=f"{self.timescale}min",
                            tz=(dt_index.tz if getattr(dt_index, 'tz', None) is not None else None))
        # Reindex to env timeline
        load_series = ts_df['demand'].reindex(idx, method='nearest') if 'demand' in ts_df.columns else None
        pv_series = ts_df['solar'].reindex(idx, method='nearest') if 'solar' in ts_df.columns else None

        # Assign to accounts
        if self.roaming_accounts:
            if load_series is not None:
                self.account_load_series[0] = load_series
            if pv_series is not None:
                self.account_pv_series[0] = pv_series
        else:
            for cs in self.charging_stations:
                if load_series is not None:
                    self.account_load_series[cs.id] = load_series
                if pv_series is not None:
                    self.account_pv_series[cs.id] = pv_series

    def _update_account_ledger_row(self) -> None:
        """Write current-step per-account values into account_buffers.
        In roaming mode, a single account (id=0) aggregates over all charging stations/ports.
        """
        t = self.current_step
        if self.roaming_accounts:
            buf = self.account_buffers.get(0)
            if buf is None:
                return
            # Default per-account setpoint split (can be overridden by user logic)
            def _get_account_setpoint(step: int) -> float:
                try:
                    gsp = float(self.power_setpoints[step]) if self.power_setpoints is not None else np.nan
                    return gsp if np.isfinite(gsp) else np.nan
                except Exception:
                    return np.nan
            # Aggregate CS-levels
            total_power_kw = float(np.nansum(self.cs_power[:, t]))
            total_amps = float(np.nansum(self.cs_current[:, t]))
            total_evs = int(np.nansum([cs.n_evs_connected for cs in self.charging_stations]))
            agg_kw_limit = float(np.nansum([getattr(cs, 'get_max_power', lambda: np.nan)() if hasattr(cs, 'get_max_power') else np.nan for cs in self.charging_stations]))
            values = {
                "step": int(t),
                "cs_power_kw": total_power_kw,
                "cs_amps": total_amps,
                "account_power_setpoint_kw": float(_get_account_setpoint(t)),
                "household_inflexible_load_kw": np.nan,
                "household_pv_kw": np.nan,
                "tracking_error_account": np.nan,
                "evs_connected": total_evs,
                "cs_kw_limit": agg_kw_limit,
            }
            # Household series (shared) and forecasts (aggregate sum baseline for roaming)
            try:
                # Use aggregate sum across transformers as the roaming baseline
                if hasattr(self, 'tr_inflexible_loads') and self.tr_inflexible_loads is not None:
                    total_load = float(np.nansum(self.tr_inflexible_loads[:, t]))
                    values["household_inflexible_load_kw"] = total_load
                if hasattr(self, 'tr_solar_power') and self.tr_solar_power is not None:
                    total_solar = float(np.nansum(self.tr_solar_power[:, t]))
                    values["household_pv_kw"] = total_solar
                lfc = getattr(self, 'account_load_forecast', {}).get(0)
                if lfc is None and getattr(self, 'account_load_forecast', {}):
                    lfc = next(iter(self.account_load_forecast.values()))
                pfc = getattr(self, 'account_pv_forecast', {}).get(0)
                if pfc is None and getattr(self, 'account_pv_forecast', {}):
                    pfc = next(iter(self.account_pv_forecast.values()))
                # Removed verbose debug comparison prints used during ledger troubleshooting
                for h in range(24):
                    values[f"load_fc_h{h+1:02d}"] = float(lfc[h]) if lfc is not None and h < len(lfc) else np.nan
                for h in range(24):
                    values[f"pv_fc_h{h+1:02d}"] = float(pfc[h]) if pfc is not None and h < len(pfc) else np.nan
            except Exception:
                pass
            # Simplified EV status (pick the first connected EV across all CS)
            ev_found = None
            for cs in self.charging_stations:
                for ev in cs.evs_connected:
                    if ev is not None:
                        ev_found = ev
                        break
                if ev_found is not None:
                    break
            if ev_found is not None:
                try:
                    values["soc"] = float(ev_found.get_soc())
                except Exception:
                    values["soc"] = np.nan
                try:
                    ttd = max(0, int(getattr(ev_found, 'time_of_departure', t) - t))
                    tsa = max(0, int(t - getattr(ev_found, 'time_of_arrival', t)))
                except Exception:
                    ttd, tsa = np.nan, np.nan
                values["time_to_departure"] = float(ttd) if np.isfinite(ttd) else np.nan
                values["time_since_arrival"] = float(tsa) if np.isfinite(tsa) else np.nan
            else:
                values["soc"] = np.nan
                values["time_to_departure"] = np.nan
                values["time_since_arrival"] = np.nan
            # Account-level tracking error (aggregate)
            try:
                setpt = float(values.get("account_power_setpoint_kw", np.nan))
                ev_kw = float(values.get("cs_power_kw", np.nan))
                load_kw = float(values.get("household_inflexible_load_kw", np.nan))
                pv_kw = float(values.get("household_pv_kw", np.nan))
                if np.isfinite(setpt) and np.isfinite(ev_kw) and np.isfinite(load_kw) and np.isfinite(pv_kw):
                    net_meter = load_kw - pv_kw + ev_kw
                    values["tracking_error_account"] = float((setpt - net_meter) ** 2)
            except Exception:
                pass
            buf.set_row(t, values)
            return

        # Non-roaming per-CS path
        for cs in self.charging_stations:
            buf = self.account_buffers.get(cs.id)
            if buf is None:
                continue
            def _get_account_setpoint(cs_id: int, step: int) -> float:
                try:
                    gsp = float(self.power_setpoints[step]) if self.power_setpoints is not None else np.nan
                    n_acc = max(1, len(self.charging_stations))
                    return gsp / n_acc if np.isfinite(gsp) else np.nan
                except Exception:
                    return np.nan
            values = {
                "step": int(t),
                "cs_power_kw": float(self.cs_power[cs.id, t]),
                "cs_amps": float(self.cs_current[cs.id, t]),
                "account_power_setpoint_kw": float(_get_account_setpoint(cs.id, t)),
                "household_inflexible_load_kw": np.nan,
                "household_pv_kw": np.nan,
                "tracking_error_account": np.nan,
                "evs_connected": int(cs.n_evs_connected),
                "cs_kw_limit": float(getattr(cs, 'get_max_power', lambda: np.nan)() if hasattr(cs, 'get_max_power') else np.nan),
            }
            # Optional per-account household signals if provided
            try:
                # Use transformer data for this CS's connected transformers
                if hasattr(self, 'tr_inflexible_loads') and self.tr_inflexible_loads is not None:
                    # Find transformers connected to this CS
                    tr_ids = []
                    for tr in self.transformers:
                        if cs.id in tr.cs_ids:
                            tr_ids.append(tr.id)
                    if tr_ids:
                        cs_load = float(np.nansum(self.tr_inflexible_loads[tr_ids, t]))
                        values["household_inflexible_load_kw"] = cs_load
                if hasattr(self, 'tr_solar_power') and self.tr_solar_power is not None:
                    # Find transformers connected to this CS
                    tr_ids = []
                    for tr in self.transformers:
                        if cs.id in tr.cs_ids:
                            tr_ids.append(tr.id)
                    if tr_ids:
                        cs_solar = float(np.nansum(self.tr_solar_power[tr_ids, t]))
                        values["household_pv_kw"] = cs_solar
            except Exception:
                pass
            # per-port amps and soc
            for p in range(cs.n_ports):
                # current signal if available
                if not self.lightweight_plots:
                    try:
                        values[f"port{p}_amps"] = float(self.port_current_signal[p, cs.id, t])
                    except Exception:
                        values[f"port{p}_amps"] = np.nan
                    try:
                        values[f"port{p}_soc"] = float(self.port_energy_level[p, cs.id, t])
                    except Exception:
                        values[f"port{p}_soc"] = np.nan
                else:
                    # In lightweight mode, approximate from cs.current_signal and ev.get_soc if available
                    values[f"port{p}_amps"] = float(cs.current_signal[p]) if p < len(cs.current_signal) else np.nan
                    if p < len(cs.evs_connected) and cs.evs_connected[p] is not None:
                        values[f"port{p}_soc"] = float(cs.evs_connected[p].get_soc())
                    else:
                        values[f"port{p}_soc"] = np.nan

                # Action (normalized) derived from amps and limits
                try:
                    amps_val = float(values.get(f"port{p}_amps", np.nan))
                except Exception:
                    amps_val = np.nan
                max_ch = float(getattr(cs, 'max_charge_current', np.nan))
                max_dis = float(abs(getattr(cs, 'max_discharge_current', 0)))
                amp_limit = float(max(max_ch if np.isfinite(max_ch) else 0, max_dis if np.isfinite(max_dis) else 0))
                values[f"port{p}_amps_limit"] = amp_limit if amp_limit > 0 else np.nan
                if np.isfinite(amps_val) and amp_limit > 0:
                    # Map to [-1,1] using appropriate side
                    if amps_val >= 0 and np.isfinite(max_ch) and max_ch > 0:
                        values[f"port{p}_action_norm"] = float(amps_val / max_ch)
                    elif amps_val < 0 and np.isfinite(max_dis) and max_dis > 0:
                        values[f"port{p}_action_norm"] = float(amps_val / max_dis)
                    else:
                        values[f"port{p}_action_norm"] = 0.0
                else:
                    values[f"port{p}_action_norm"] = np.nan

                # Connection flag and timing features
                connected = 1 if (p < len(cs.evs_connected) and cs.evs_connected[p] is not None) else 0
                values[f"port{p}_connected"] = np.int8(connected)
                if connected:
                    ev = cs.evs_connected[p]
                    try:
                        ttd = max(0, int(getattr(ev, 'time_of_departure', t) - t))
                        tsa = max(0, int(t - getattr(ev, 'time_of_arrival', t)))
                    except Exception:
                        ttd, tsa = np.nan, np.nan
                else:
                    ttd, tsa = np.nan, np.nan
                values[f"port{p}_time_to_departure"] = float(ttd) if np.isfinite(ttd) else np.nan
                values[f"port{p}_time_since_arrival"] = float(tsa) if np.isfinite(tsa) else np.nan
            # Per-account forecasts (if present)
            try:
                lfc = getattr(self, 'account_load_forecast', {}).get(cs.id)
                pfc = getattr(self, 'account_pv_forecast', {}).get(cs.id)
                for h in range(24):
                    values[f"load_fc_h{h+1:02d}"] = float(lfc[h]) if lfc is not None and h < len(lfc) else np.nan
                for h in range(24):
                    values[f"pv_fc_h{h+1:02d}"] = float(pfc[h]) if pfc is not None and h < len(pfc) else np.nan
            except Exception:
                pass
            # Account-level tracking error if we have sufficient pieces
            try:
                setpt = float(values.get("account_power_setpoint_kw", np.nan))
                ev_kw = float(values.get("cs_power_kw", np.nan))
                load_kw = float(values.get("household_inflexible_load_kw", np.nan))
                pv_kw = float(values.get("household_pv_kw", np.nan))
                if np.isfinite(setpt) and np.isfinite(ev_kw) and np.isfinite(load_kw) and np.isfinite(pv_kw):
                    net_meter = load_kw - pv_kw + ev_kw
                    values["tracking_error_account"] = float((setpt - net_meter) ** 2)
            except Exception:
                pass
            buf.set_row(t, values)

    def set_save_plots(self, save_plots):
        if save_plots:
            os.makedirs("./results", exist_ok=True)
            print(f"Creating directory: ./results/{self.sim_name}")
            os.makedirs(f"./results/{self.sim_name}", exist_ok=True)

        self.save_plots = save_plots

    def _update_power_statistics(self, departing_evs):
        '''Updates the power statistics of the simulation'''

        # if not self.lightweight_plots:
        for tr in self.transformers:
            # self.transformer_amps[tr.id, self.current_step] = tr.current_amps
            self.tr_overload[tr.id,
                             self.current_step] = tr.get_how_overloaded()
            self.tr_inflexible_loads[tr.id,
                                     self.current_step] = tr.inflexible_load[self.current_step]
            self.tr_solar_power[tr.id,
                                self.current_step] = tr.solar_power[self.current_step]
        
            # Update energy flow breakdown for solar production
            self.energy_flow_breakdown['solar_production'][self.current_step] += tr.solar_power[self.current_step]

        # Calculate total EV power (positive for charging, negative for discharging)
        total_ev_power = 0
        
        for i, cs in enumerate(self.charging_stations):
            self.cs_power[cs.id, self.current_step] = cs.current_power_output
            self.cs_current[cs.id, self.current_step] = cs.current_total_amps
            
            # Accumulate EV power directly
            total_ev_power += cs.current_power_output

            for j in range(self.number_of_ports_per_cs):
                if j < len(cs.evs_connected) and cs.evs_connected[j] is not None:
                    ev = cs.evs_connected[j]
                else:
                    continue

                if not self.lightweight_plots:
                    self.port_current_signal[j, cs.id,
                                             self.current_step] = cs.current_signal[j]

                self.port_energy_level[j, cs.id,
                                       self.current_step] = ev.get_soc()

        # Departed EVs are no longer connected; their port data has already been
        # set to 0 above. We keep them only for high-level metrics, so skip
        # port-level array updates that rely on integer indices.

        # Update the energy flow breakdown for the current step
        # Sum inflexible loads for this timestep
        self.energy_flow_breakdown['inflexible_load'][self.current_step] = sum(
            tr.inflexible_load[self.current_step] for tr in self.transformers
        )
        self.energy_flow_breakdown['grid_draw'][self.current_step] = self.current_power_usage[self.current_step]
        self.energy_flow_breakdown['ev_power'][self.current_step] = total_ev_power

        # Always calculate cost for the current step (for visualization and analysis)
        timestep_hours = self.timescale / 60.0
        
        # Positive power is grid draw, negative is grid injection (from solar/battery)
        grid_energy_kwh = self.energy_flow_breakdown['grid_draw'][self.current_step] * timestep_hours
        
        # For cost calculation, split ev_power into charging (positive) and discharging (negative)
        ev_power = self.energy_flow_breakdown['ev_power'][self.current_step]
        charging_kwh = max(0, ev_power) * timestep_hours
        discharging_kwh = max(0, -ev_power) * timestep_hours
        
        grid_price = np.mean([self.charge_prices[cs.id, self.current_step] for cs in self.charging_stations])
        discharge_price = np.mean([self.discharge_prices[cs.id, self.current_step] for cs in self.charging_stations])
        
        # Calculate cost: pay for grid energy and charging, get credit for discharging
        self.cost_history[self.current_step] = (
            grid_energy_kwh * grid_price +
            charging_kwh * grid_price -
            discharging_kwh * discharge_price
        )

    def _step_date(self):
        '''Steps the simulation date by one timestep'''
        self.sim_date = self.sim_date + \
            datetime.timedelta(minutes=self.timescale)

    def _update_forecasts(self):
        """Generates forecasts for all configured targets."""
        if not self.forecasting_config.get('enabled'):
            return

        targets = self.forecasting_config.get('targets', [])
        params = self.forecasting_config.get('params', {})

        # Build an aligned datetime index for the current simulation window
        try:
            import pytz
            fixed_tz = pytz.FixedOffset(600)
        except Exception:
            fixed_tz = None
        start_idx = pd.Timestamp(year=self.config.get('year', 2019),
                                 month=self.config.get('month', 1),
                                 day=self.config.get('day', 1),
                                 hour=self.config.get('hour', 0),
                                 minute=self.config.get('minute', 0))
        if fixed_tz is not None and start_idx.tzinfo is None:
            start_idx = start_idx.tz_localize(fixed_tz)
        sim_index = pd.date_range(start=start_idx,
                                  periods=int(self.simulation_length),
                                  freq=f"{self.timescale}min",
                                  tz=start_idx.tz)

        # Determine availability of transformer arrays up-front
        has_tr_load = hasattr(self, 'tr_inflexible_loads') and self.tr_inflexible_loads is not None and getattr(self, 'tr_inflexible_loads').shape[0] > 0
        has_tr_pv = hasattr(self, 'tr_solar_power') and self.tr_solar_power is not None and getattr(self, 'tr_solar_power').shape[0] > 0

        for target in targets:
            if target == 'household_demand':
                # Use aggregate sum across transformers as the forecast baseline for roaming account
                if has_tr_load and hasattr(self, 'transformers') and self.transformers:
                    # Use full transformer series (not step-filled arrays)
                    try:
                        agg_load = np.nansum([tr.inflexible_load for tr in self.transformers], axis=0)
                    except Exception:
                        # Fallback to step-filled arrays if needed
                        agg_load = np.nansum(self.tr_inflexible_loads, axis=0)
                    step_per_hour = max(1, int(60 // self.timescale))
                    t0 = int(self.current_step)
                    horizon = int(params.get('forecast_horizon_hours', 24))
                    fc = []
                    for h in range(1, horizon + 1):
                        idx = t0 + h * step_per_hour
                        fc.append(float(agg_load[idx]) if idx < len(agg_load) else float(agg_load[-1]))
                    self.demand_forecast = np.asarray(fc, dtype=float)
                    # Removed per-step forecast debug prints
                # Primary fallback: use full_timeseries_data which contains the entire horizon
                elif hasattr(self, 'full_timeseries_data') and 'demand' in getattr(self, 'full_timeseries_data', pd.DataFrame()).columns:
                    series = self.full_timeseries_data['demand']
                    self.demand_forecast = create_lookahead_forecast(
                        data=series,
                        start_time=self.sim_date,
                        **params
                    )
                # Otherwise skip
                else:
                    # Data not ready; skip to avoid building zero forecasts. Will try again next tick.
                    self.demand_forecast = None
            elif target == 'temperature':
                # Load weather data and extract temperature forecasts and current series
                from ev2gym.utilities.loaders import load_weather_data
                weather_df = load_weather_data(self)
                if weather_df is not None and 'temperature' in weather_df.columns:
                    # Persist current series aligned to simulation timeline
                    try:
                        aligned = weather_df.reindex(sim_index, method='nearest')
                        self.temperature_series = aligned['temperature'].to_numpy(dtype=float)[: int(self.simulation_length)]
                    except Exception:
                        self.temperature_series = None
                    # Build lookahead forecast from current step
                    temp_series = aligned['temperature'] if 'aligned' in locals() else weather_df['temperature']
                    step_per_hour = max(1, int(60 // self.timescale))
                    t0 = int(self.current_step)
                    horizon = int(params.get('forecast_horizon_hours', 24))
                    fc = []
                    for h in range(1, horizon + 1):
                        idx = t0 + h * step_per_hour
                        val = temp_series.iloc[idx] if idx < len(temp_series) else temp_series.iloc[-1]
                        fc.append(float(val))
                    self.temp_forecast = np.asarray(fc, dtype=float)
                else:
                    self.temp_forecast = None
            elif target == 'wind':
                # Load weather data and extract wind forecasts and current series
                from ev2gym.utilities.loaders import load_weather_data
                weather_df = load_weather_data(self)
                if weather_df is not None and 'wind_speed' in weather_df.columns:
                    # Persist current series aligned to simulation timeline
                    try:
                        aligned = weather_df.reindex(sim_index, method='nearest')
                        self.wind_speed_series = aligned['wind_speed'].to_numpy(dtype=float)[: int(self.simulation_length)]
                    except Exception:
                        self.wind_speed_series = None
                    # Build lookahead forecast from current step
                    wind_series = aligned['wind_speed'] if 'aligned' in locals() else weather_df['wind_speed']
                    step_per_hour = max(1, int(60 // self.timescale))
                    t0 = int(self.current_step)
                    horizon = int(params.get('forecast_horizon_hours', 24))
                    fc = []
                    for h in range(1, horizon + 1):
                        idx = t0 + h * step_per_hour
                        val = wind_series.iloc[idx] if idx < len(wind_series) else wind_series.iloc[-1]
                        fc.append(float(val))
                    self.wind_forecast = np.asarray(fc, dtype=float)
                else:
                    self.wind_forecast = None
            elif target == 'solar_production':
                if has_tr_pv and hasattr(self, 'transformers') and self.transformers:
                    try:
                        agg_pv = np.nansum([tr.solar_power for tr in self.transformers], axis=0)
                    except Exception:
                        agg_pv = np.nansum(self.tr_solar_power, axis=0)
                    step_per_hour = max(1, int(60 // self.timescale))
                    t0 = int(self.current_step)
                    horizon = int(params.get('forecast_horizon_hours', 24))
                    fc = []
                    for h in range(1, horizon + 1):
                        idx = t0 + h * step_per_hour
                        fc.append(float(agg_pv[idx]) if idx < len(agg_pv) else float(agg_pv[-1]))
                    self.solar_forecast = np.asarray(fc, dtype=float)
                elif hasattr(self, 'full_timeseries_data') and 'solar' in getattr(self, 'full_timeseries_data', pd.DataFrame()).columns:
                    series = self.full_timeseries_data['solar']
                    self.solar_forecast = create_lookahead_forecast(
                        data=series,
                        start_time=self.sim_date,
                        **params
                    )
                elif has_tr_pv:
                    agg_pv = np.nansum(self.tr_solar_power, axis=0)
                    step_per_hour = max(1, int(60 // self.timescale))
                    t0 = int(self.current_step)
                    horizon = int(params.get('forecast_horizon_hours', 24))
                    fc = []
                    for h in range(1, horizon + 1):
                        idx = t0 + h * step_per_hour
                        fc.append(float(agg_pv[idx]) if idx < len(agg_pv) else float(agg_pv[-1]))
                    self.solar_forecast = np.asarray(fc, dtype=float)
            elif target in ('temperature', 'temp'):
                # try common column names
                for cname in ['temperature', 'temp', 'air_temp']:
                    if hasattr(self, 'full_timeseries_data') and cname in getattr(self, 'full_timeseries_data', pd.DataFrame()).columns:
                        self.temperature_forecast = create_lookahead_forecast(
                            data=self.full_timeseries_data[cname],
                            start_time=self.sim_date,
                            **params
                        )
                        break
            elif target in ('wind_speed', 'wind'):
                for cname in ['wind_speed', 'wind', 'wind_spd']:
                    if hasattr(self, 'full_timeseries_data') and cname in getattr(self, 'full_timeseries_data', pd.DataFrame()).columns:
                        self.wind_forecast = create_lookahead_forecast(
                            data=self.full_timeseries_data[cname],
                            start_time=self.sim_date,
                            **params
                        )
                        break
            elif target in ('dr_event_active', 'dr_event', 'dr'):
                # Treat as binary series; forecast returns floats we will threshold when writing
                for cname in ['dr_event_active', 'dr_event', 'dr']:
                    if hasattr(self, 'full_timeseries_data') and cname in getattr(self, 'full_timeseries_data', pd.DataFrame()).columns:
                        self.dr_event_forecast = create_lookahead_forecast(
                            data=self.full_timeseries_data[cname].astype(float),
                            start_time=self.sim_date,
                            **params
                        )
                        break

        # Removed concise forecast availability debug prints

        # Mirror forecasts to per-account dicts for ledger writing (silently)
        try:
            if self.roaming_accounts:
                if getattr(self, 'demand_forecast', None) is not None:
                    self.account_load_forecast[0] = self.demand_forecast
                if getattr(self, 'solar_forecast', None) is not None:
                    self.account_pv_forecast[0] = self.solar_forecast
            else:
                for cs in self.charging_stations:
                    if getattr(self, 'demand_forecast', None) is not None:
                        self.account_load_forecast[cs.id] = self.demand_forecast
                    if getattr(self, 'solar_forecast', None) is not None:
                        self.account_pv_forecast[cs.id] = self.solar_forecast
        except Exception:
            pass

    def _get_observation(self):
        obs = self.state_function(self)

        # Append forecasts to the observation if they exist
        if self.demand_forecast is not None:
            obs = np.concatenate([obs, self.demand_forecast])
        if self.solar_forecast is not None:
            obs = np.concatenate([obs, self.solar_forecast])

        return obs

    def _calculate_reward(self, total_costs, user_satisfaction_list, invalid_action_punishment):
        '''Calculates the reward for the current step'''

        reward = self.reward_function(
            self, total_costs, user_satisfaction_list, invalid_action_punishment)
        self.total_reward += reward

        return reward

    def _check_termination(self, reward, info):
        '''Checks if the episode is done or any constraint is violated'''
        truncated = False
        action_mask = np.zeros(self.number_of_ports)
        # action mask is 1 if an EV is connected to the port
        for i, cs in enumerate(self.charging_stations):
            for j in range(cs.n_ports):
                if cs.evs_connected[j] is not None:
                    action_mask[i*cs.n_ports + j] = 1

        # Check if the episode is done or any constraint is violated
        if self.current_step >= self.simulation_length - 1 or \
            (any(tr.is_overloaded() > 0 for tr in self.transformers)
             and not self.generate_rnd_game):
            """Terminate if:
                - The simulation length is reached
                - Any user satisfaction score is below the threshold
                - Any charging station is overloaded
                Dont terminate when overloading if :
                - generate_rnd_game is True
                Carefull: if generate_rnd_game is True,
                the simulation might end up in infeasible problem
                """

            self.done = True
            self.stats = get_statistics(self)

            self.stats['action_mask'] = action_mask
            self.cost = info['cost']
            self.stats.update(info)

            if self.verbose:
                print_statistics(self)

                if any(tr.is_overloaded() for tr in self.transformers):
                    print(
                        f"Transformer overloaded, {self.current_step} timesteps\n")
                else:
                    print(
                        f"Episode finished after {self.current_step} timesteps\n")

            if self.save_replay:
                self._save_sim_replay()

            if self.save_plots:
                # save the env as a pickle file
                with open(f"./results/{self.sim_name}/env.pkl", 'wb') as f:
                    self.renderer = None
                    pickle.dump(self, f)
                ev_city_plot(self)

            if self.cost_function is not None:
                return self._get_observation(), reward, True, truncated, self.stats
            else:
                return self._get_observation(), reward, True, truncated, self.stats
        else:
            stats = {
                'action_mask': action_mask,
            }
            stats.update(info)

            if self.cost_function is not None:
                return self._get_observation(), reward, False, truncated, stats
            else:
                return self._get_observation(), reward, False, truncated, stats

    def set_cost_function(self, cost_function):
        '''
        This function sets the cost function of the environment
        '''
        self.cost_function = cost_function

    def set_reward_function(self, reward_function):
        '''
        This function sets the reward function of the environment
        '''
        self.reward_function = reward_function

    def _update_ev_location_data(self):
        """Update EV location data for the current timestep."""
        # For each charging station and port
        for cs_idx, cs in enumerate(self.charging_stations):
            for port_idx in range(self.number_of_ports_per_cs):
                if port_idx < len(cs.evs_connected) and cs.evs_connected[port_idx] is not None:
                    ev = cs.evs_connected[port_idx]
                    # Update location state (0=home, 1=work, 2=commuting)
                    self.ev_location_data[port_idx, cs_idx, self.current_step] = ev.location_state  # Track location state
                else:
                    # No EV in this port
                    self.ev_location_data[port_idx, cs_idx, self.current_step] = -1  # -1 indicates no EV

    def _log_initial_state(self):
        """Logs the initial state of EVs in the environment if verbose is True."""
        if self.verbose:
            initial_evs_connected = sum(1 for cs in self.charging_stations for ev in cs.evs_connected if ev is not None)
            print(f"[EV2Gym] Environment initialized. Total EV profiles: {len(self.EVs_profiles)}")
            print(f"[EV2Gym] Initial EVs connected at step 0: {initial_evs_connected}")
            for cs in self.charging_stations:
                for ev in cs.evs_connected:
                    if ev is not None:
                        print(f"[EV2Gym] - EV {ev.id} is at station {cs.id} (Location: {ev.location}) at start.")

    def _spawn_evs_at_current_step(self):
        """Connects EVs that are scheduled to arrive at the current timestep."""
        # Track which EVs have already been connected to avoid duplicates
        connected_evs = set()
        
        # Get IDs of EVs already in the system to avoid duplicates
        existing_ev_ids = {ev.id for ev in self.EVs}
        
        for ev_profile in self.EVs_profiles:
            # Check if this EV should spawn at current step
            if ev_profile.time_of_arrival == self.current_step:
                # Skip if this EV is already spawned
                if ev_profile.id in existing_ev_ids:
                    continue
                    
                # Use deepcopy to create a new EV instance from the profile
                new_ev = deepcopy(ev_profile)
                new_ev.reset()
                new_ev.simulation_length = self.simulation_length
                
                # Spawn the new EV at its designated charging station
                cs_index = new_ev.location
                if 0 <= cs_index < len(self.charging_stations):     # verifies that the EV is assigned to a valid charging station
                    target_cs = self.charging_stations[cs_index]
                    # Only connect if EV isn't already connected somewhere else
                    if (target_cs.n_evs_connected < target_cs.n_ports and 
                        new_ev.id not in connected_evs):
                        spawn_result = target_cs.spawn_ev(new_ev)
                        if spawn_result is not None:  # Spawn successful
                            self.EVs.append(new_ev)
                            connected_evs.add(new_ev.id)
                            existing_ev_ids.add(new_ev.id)  # Update the set
                        else:
                            if self.verbose:
                                print(f"⚠️ EV {new_ev.id} could not spawn at station {cs_index}: charger is full.")
                    else:
                        if self.verbose:
                            print(f"⚠️ EV {new_ev.id} could not spawn at station {cs_index}: charger is full.")
                else:
                    print(f"Warning: EV {new_ev.id} has invalid location {cs_index} and was not spawned.")
