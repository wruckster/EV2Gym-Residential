# This file contains support functions for the EV City environment.

import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
from typing import List, Dict

from ev2gym.models.ev.vehicle import EV


def get_statistics(env) -> Dict:
    total_ev_served = np.array(
        [cs.total_evs_served for cs in env.charging_stations]).sum()
    total_profits = np.array(
        [cs.total_profits for cs in env.charging_stations]).sum()
    total_energy_charged = np.array(
        [cs.total_energy_charged for cs in env.charging_stations]).sum()
    total_energy_discharged = np.array(
        [cs.total_energy_discharged for cs in env.charging_stations]).sum()
    _aus_list = [
        cs.get_avg_user_satisfaction() for cs in env.charging_stations
        if cs.total_evs_served > 0
    ]
    if len(_aus_list) == 0:
        average_user_satisfaction = 0.0
    else:
        average_user_satisfaction = float(np.mean(np.asarray(_aus_list, dtype=float)))

    # get transformer overload from env.tr_overload
    total_transformer_overload = np.array(env.tr_overload).sum()

    tracking_error = 0
    energy_tracking_error = 0
    power_tracker_violation = 0

    for t in range(env.simulation_length):
        # tracking_error += (min(env.power_setpoints[t], env.charge_power_potential[t]) -
        #                    env.current_power_usage[t])**2
        # energy_tracking_error += abs(min(env.power_setpoints[t], env.charge_power_potential[t]) -
        #                              env.current_power_usage[t])

        tracking_error += (env.power_setpoints[t] -
                           env.current_power_usage[t])**2
        energy_tracking_error += abs(env.power_setpoints[t] -
                                     env.current_power_usage[t])

        if env.current_power_usage[t] > env.power_setpoints[t]:
            power_tracker_violation += env.current_power_usage[t] - \
                env.power_setpoints[t]

    energy_tracking_error *= env.timescale / 60

    # calculate total batery degradation
    battery_degradation = np.array(
        [np.array(ev.get_battery_degradation()).reshape(-1) for ev in env.EVs])
    if len(battery_degradation) == 0:
        battery_degradation = np.zeros((1, 2))
    battery_degradation_calendar = battery_degradation[:, 0].sum()
    battery_degradation_cycling = battery_degradation[:, 1].sum()
    battery_degradation = battery_degradation.sum()

    total_steps_min_emergency_battery_capacity_violation = 0
    energy_user_satisfaction = np.zeros((len(env.EVs)))
    for i, ev in enumerate(env.EVs):
        e_actual = ev.current_capacity
        e_max = ev.max_energy_AFAP
        energy_user_satisfaction[i] = (e_actual / e_max) * 100
        total_steps_min_emergency_battery_capacity_violation += ev.min_emergency_battery_capacity_metric

    # Add safety checks for empty arrays
    if len(energy_user_satisfaction) == 0:
        mean_energy_user_satisfaction = 0.0
        std_energy_user_satisfaction = 0.0
        min_energy_user_satisfaction = 0.0
    else:
        mean_energy_user_satisfaction = np.mean(energy_user_satisfaction)
        std_energy_user_satisfaction = np.std(energy_user_satisfaction)
        min_energy_user_satisfaction = np.min(energy_user_satisfaction)

    stats = {'total_ev_served': total_ev_served,
             'total_profits': total_profits,
             'total_energy_charged': total_energy_charged,
             'total_energy_discharged': total_energy_discharged,
             'average_user_satisfaction': average_user_satisfaction,
             'power_tracker_violation': power_tracker_violation,
             'tracking_error': tracking_error,
             'energy_tracking_error': energy_tracking_error,
             'energy_user_satisfaction': mean_energy_user_satisfaction,
             'std_energy_user_satisfaction': std_energy_user_satisfaction,
             'min_energy_user_satisfaction': min_energy_user_satisfaction,
             'total_steps_min_emergency_battery_capacity_violation': total_steps_min_emergency_battery_capacity_violation,
             'total_transformer_overload': total_transformer_overload,
             'battery_degradation': battery_degradation,
             'battery_degradation_calendar': battery_degradation_calendar,
             'battery_degradation_cycling': battery_degradation_cycling,
             'total_reward': env.total_reward,
             }

    if env.eval_mode != "optimal" and env.replay is not None:
        if env.replay.optimal_stats is not None:
            stats['opt_profits'] = env.replay.optimal_stats["total_profits"]
            stats['opt_tracking_error'] = env.replay.optimal_stats["tracking_error"]
            stats['opt_actual_tracking_error'] = env.replay.optimal_stats["energy_tracking_error"]
            stats['opt_power_tracker_violation'] = env.replay.optimal_stats["power_tracker_violation"]
            stats['opt_energy_user_satisfaction'] = env.replay.optimal_stats["energy_user_satisfaction"]
            stats['opt_total_energy_charged'] = env.replay.optimal_stats["total_energy_charged"]

    return stats


def print_statistics(env) -> None:

    assert env.stats is not None, "No statistics available. Run the simulation first!"

    stats = env.stats

    total_ev_served = stats['total_ev_served']
    total_profits = stats['total_profits']
    total_energy_charged = stats['total_energy_charged']
    total_energy_discharged = stats['total_energy_discharged']
    average_user_satisfaction = stats['average_user_satisfaction']
    total_transformer_overload = stats['total_transformer_overload']
    tracking_error = stats['tracking_error']
    energy_tracking_error = stats['energy_tracking_error']
    power_tracker_violation = stats['power_tracker_violation']
    energy_user_satisfaction = stats['energy_user_satisfaction']
    std_energy_user_satisfaction = stats['std_energy_user_satisfaction']
    min_energy_user_satisfaction = stats['min_energy_user_satisfaction']

    total_transformer_overload = stats['total_transformer_overload']
    battery_degradation = stats['battery_degradation']
    battery_degradation_calendar = stats['battery_degradation_calendar']
    battery_degradation_cycling = stats['battery_degradation_cycling']

    print("\n\n==============================================================")
    print("Simulation statistics:")
    for cs in env.charging_stations:
        print(cs)
    print(
        f'  - Total EVs spawned: {env.total_evs_spawned} |  served: {total_ev_served}')
    print(f'  - Total profits: {total_profits:.2f} €')
    print(
        f'  - Average user satisfaction: {average_user_satisfaction*100:.2f} %')

    print(
        f'  - Total energy charged: {total_energy_charged:.1f} | discharged: {total_energy_discharged:.1f} kWh')
    print(
        f'  - Power Tracking squared error: {tracking_error:.2f}, Power Violation: {power_tracker_violation:.2f} kW')
    print(f' - Actual Energy Tracking error: {energy_tracking_error:.2f} kW')
    print(
        f'  - Mean energy user satisfaction: {energy_user_satisfaction:.2f} % | Min: {min_energy_user_satisfaction:.2f} %')
    print(
        f'  - Std Energy user satisfaction: {std_energy_user_satisfaction:.2f} %')
    print(
        f'  - Total Battery degradation: {battery_degradation:.5f}% | Calendar: {battery_degradation_calendar:.5f}%, Cycling: {battery_degradation_cycling:.5f}%')
    print(
        f'  - Total transformer overload: {total_transformer_overload:.2f} kWh \n')

    print("==============================================================\n\n")


def spawn_single_EV(env,
                    scenario,
                    cs_id,
                    port,
                    hour,
                    minute,
                    step,
                    min_time_of_stay_steps
                    ) -> EV:
    '''
    This function spawns a single EV and returns it
    '''

    # required energy independent of time of arrival
    # required_energy = env.df_energy_demand[scenario].iloc[np.random.randint(
    #     0, 100, size=1)].values[0]  # kWh

    # roound minute to 30 or 0
    if minute < 30:
        minute = 0
    else:
        minute = 30

    # required energy dependent on time of arrival
    arrival_time = f'{hour:02d}:{minute:02d}'

    required_energy_mean = env.df_req_energy[
        (env.df_req_energy['Arrival Time'] == arrival_time)
    ][scenario].values[0]

    required_energy = np.random.normal(
        required_energy_mean, 0.5*required_energy_mean)  # kWh

    if required_energy < 5:
        required_energy = np.random.randint(5, 10)

    if env.heterogeneous_specs:
        sampled_ev = np.random.choice(
            list(env.ev_specs.keys()), p=env.normalized_ev_registrations)
        battery_capacity = env.ev_specs[sampled_ev]["battery_capacity"]
    else:
        battery_capacity = env.config["ev"]["battery_capacity"]

    if battery_capacity < required_energy:
        initial_battery_capacity = np.random.randint(1, battery_capacity)
    else:
        initial_battery_capacity = battery_capacity - required_energy

    if initial_battery_capacity > env.config["ev"]['desired_capacity']:
        initial_battery_capacity = np.random.randint(1, battery_capacity)

    if initial_battery_capacity < env.config["ev"]['min_battery_capacity'] and battery_capacity > 2*env.config["ev"]['min_battery_capacity']:
        initial_battery_capacity = env.config["ev"]['min_battery_capacity']

    # time of stay dependent on time of arrival
    time_of_stay_mean = env.df_time_of_stay_vs_arrival[(
        env.df_time_of_stay_vs_arrival['Arrival Time'] == arrival_time)
    ][scenario].values[0]

    time_of_stay = np.random.normal(
        time_of_stay_mean, 0.2*time_of_stay_mean)  # hours

    # turn from hours to steps
    time_of_stay = time_of_stay * 60 / env.timescale + 1

    # Alternative method for time of stay based on 10.000 charging sessions
    # time_of_stay = np.random.choice(
    #     np.arange(0, 48, 1), 1, p=env.time_of_connection_vs_hour[hour, :])/2
    # time_of_stay = time_of_stay[0] * 60 / env.timescale + 1

    # Alternative method for time of stay without taking into account the hour
    # time_of_stay = env.df_connection_time[scenario].iloc[np.random.randint(
    #     0, 100, size=1)].values[0] * 60 / env.timescale + 1

    if time_of_stay < min_time_of_stay_steps:
        time_of_stay = min_time_of_stay_steps

    if env.empty_ports_at_end_of_simulation:
        if time_of_stay + step + 4 >= env.simulation_length:
            return None
            time_of_stay = env.simulation_length - step - 4 - 2

    if "transition_soc_multiplier" in env.config["ev"]:
        transition_soc_multiplier = env.config["ev"]["transition_soc_multiplier"]
    else:
        transition_soc_multiplier = 1

    min_emergency_battery_capacity = env.config["ev"]["min_emergency_battery_capacity"]

    if min_emergency_battery_capacity > battery_capacity:
        min_emergency_battery_capacity = 0.7*battery_capacity

    if env.heterogeneous_specs:

        # get charge efficiency from env.ev_specs dict
        # if there is key charge_efficiency_v
        if "3ph_ch_efficiency" in env.ev_specs[sampled_ev]:
            charge_efficiency_v = env.ev_specs[sampled_ev]["3ph_ch_efficiency"]
            current_levels = env.ev_specs[sampled_ev]["ch_current"]
            assert len(charge_efficiency_v) == len(current_levels)
            assert all([0 <= x <= 100 for x in charge_efficiency_v])

            # make a dict with charge leves kay and charge efficiency value
            charge_efficiency = dict(zip(current_levels, charge_efficiency_v))

            for i in range(0, 101):
                if i not in charge_efficiency or charge_efficiency[i] == 0:
                    nonzero_keys = [
                        k for k, v in charge_efficiency.items() if v != 0]
                    if nonzero_keys:
                        closest = min(nonzero_keys, key=lambda x: abs(x - i))
                        charge_efficiency[i] = charge_efficiency[closest]

            discharge_efficiency = charge_efficiency.copy()

        else:
            charge_efficiency = np.round(1 -
                                         (np.random.rand()+0.00001)/20, 3)  # [0.95-1]
            discharge_efficiency = np.round(1 -
                                            (np.random.rand()+0.00001)/20, 3)  # [0.95-1]

        # Build robust parameter set with fallbacks to config when JSON lacks fields
        _spec = env.ev_specs[sampled_ev]
        max_ac_charge_power = _spec.get("max_ac_charge_power", env.ev_parameters.get("max_ac_charge_power", 0))
        min_ac_charge_power = _spec.get("min_ac_charge_power", env.ev_parameters.get("min_ac_charge_power", 0))
        max_dc_charge_power = _spec.get("max_dc_charge_power", env.ev_parameters.get("max_dc_charge_power", 0))
        # Prefer DC discharge power if available, else AC, else fallback
        max_discharge_power = _spec.get(
            "max_dc_discharge_power",
            _spec.get("max_ac_discharge_power", env.ev_parameters.get("max_discharge_power", 0))
        )
        min_discharge_power = _spec.get("min_discharge_power", env.ev_parameters.get("min_discharge_power", 0))

        # desired_capacity in GF path is a fraction times battery_capacity; mirror that behavior here
        desired_capacity_ratio = env.config.get("ev", {}).get('desired_capacity', 1)

        # Whitelist of valid arguments for the EV constructor to prevent TypeErrors
        valid_ev_args = {
            'battery_capacity', 'min_battery_capacity', 'min_emergency_battery_capacity',
            'max_ac_charge_power', 'min_ac_charge_power', 'max_dc_charge_power',
            'max_discharge_power', 'min_discharge_power', 'ev_phases',
            'transition_soc', 'transition_soc_multiplier', 'charge_efficiency',
            'discharge_efficiency', 'timescale', 'metadata', 'location_state',
            'commuting_consumption_kwh_km', 'desired_capacity'
        }

        # Filter the parameters from the config to only include valid EV constructor arguments
        filtered_params = {k: v for k, v in env.ev_parameters.items() if k in valid_ev_args}

        # Remove keys that are explicitly passed to avoid TypeError
        explicit_keys = {
            'battery_capacity', 'max_ac_charge_power', 'min_ac_charge_power',
            'max_dc_charge_power', 'max_discharge_power', 'min_discharge_power',
            'desired_capacity', 'charge_efficiency', 'discharge_efficiency',
            'min_emergency_battery_capacity', 'transition_soc_multiplier'
        }
        for key in explicit_keys:
            filtered_params.pop(key, None)

        return EV(id=port,
                  location=cs_id,
                  battery_capacity_at_arrival=initial_battery_capacity,
                  time_of_arrival=step,
                  time_of_departure=step + int(time_of_stay),
                  battery_capacity=_spec["battery_capacity"],
                  max_ac_charge_power=max_ac_charge_power,
                  min_ac_charge_power=min_ac_charge_power,
                  max_dc_charge_power=max_dc_charge_power,
                  max_discharge_power=max_discharge_power,
                  min_discharge_power=min_discharge_power,
                  desired_capacity=desired_capacity_ratio * _spec["battery_capacity"],
                  charge_efficiency=charge_efficiency,
                  discharge_efficiency=discharge_efficiency,
                  min_emergency_battery_capacity=min_emergency_battery_capacity,
                  transition_soc_multiplier=transition_soc_multiplier,
                  **filtered_params)

    else:
        return EV(id=port,
                  location=cs_id,
                  battery_capacity_at_arrival=initial_battery_capacity,
                  time_of_arrival=step,
                  time_of_departure=step + int(time_of_stay),
                  **env.ev_parameters)


def spawn_single_EV_GF(env,
                       day,
                       cs_id,
                       port,
                       hour,
                       minute,
                       step,
                       min_time_of_stay_steps
                       ) -> EV:
    '''
    This function spawns a single EV and returns it
    '''

    # round minute to 30 or 0
    if minute < 30:
        minute = 0
    else:
        minute = 30

    if day < 5:
        size = env.df_req_energy_weekday.shape[1]
        required_energy = 2 * \
            np.random.choice(size, 1, p=env.df_req_energy_weekday[hour])
        size = env.time_of_connection_vs_hour_weekday.shape[1]
        time_of_stay = 10 * \
            np.random.choice(
                size, 1, p=env.time_of_connection_vs_hour_weekday[hour])
    else:
        size = env.df_req_energy_weekend.shape[1]
        required_energy = 2 * \
            np.random.choice(size, 1, p=env.df_req_energy_weekend[hour])
        size = env.time_of_connection_vs_hour_weekend.shape[1]
        time_of_stay = 10 * \
            np.random.choice(
                size, 1, p=env.time_of_connection_vs_hour_weekend[hour])

    required_energy = float(required_energy)
    time_of_stay = float(time_of_stay)

    if required_energy < 2:
        required_energy = np.random.randint(5, 10)

    if env.heterogeneous_specs:
        sampled_ev = np.random.choice(
            list(env.ev_specs.keys()), p=env.normalized_ev_registrations)
        battery_capacity = env.ev_specs[sampled_ev]["battery_capacity"]
    else:
        battery_capacity = env.config["ev"]["battery_capacity"]

    if battery_capacity < required_energy:
        initial_battery_capacity = np.random.randint(1, battery_capacity)
    else:
        initial_battery_capacity = battery_capacity - required_energy

    if initial_battery_capacity > env.config["ev"]['desired_capacity']:
        initial_battery_capacity = np.random.randint(1, battery_capacity)

    if initial_battery_capacity < env.config["ev"]['min_battery_capacity'] and battery_capacity > 2*env.config["ev"]['min_battery_capacity']:
        initial_battery_capacity = env.config["ev"]['min_battery_capacity']

    # turn from minutes to steps
    time_of_stay = time_of_stay // env.timescale + 1

    if time_of_stay < min_time_of_stay_steps:
        time_of_stay = min_time_of_stay_steps

    if env.empty_ports_at_end_of_simulation:
        if time_of_stay + step + 4 >= env.simulation_length:
            return None
            time_of_stay = env.simulation_length - step - 4 - 2
    if initial_battery_capacity > battery_capacity:
        print(f"Initial battery capacity: {initial_battery_capacity}")
        print(f"Battery capacity: {battery_capacity}")
        raise ValueError(
            "Initial battery capacity cannot be higher than battery capacity!")

    if "transition_soc_multiplier" in env.config["ev"]:
        transition_soc_multiplier = env.config["ev"]["transition_soc_multiplier"]
    else:
        transition_soc_multiplier = 1

    if env.heterogeneous_specs:
        return EV(id=port,
                  location=cs_id,
                  battery_capacity_at_arrival=initial_battery_capacity,
                  max_ac_charge_power=env.ev_specs[sampled_ev]["max_ac_charge_power"],
                  max_dc_charge_power=env.ev_specs[sampled_ev]["max_dc_charge_power"],
                  max_discharge_power=-
                  env.ev_specs[sampled_ev]["max_dc_discharge_power"],
                  discharge_efficiency=np.round(1 -
                                                (np.random.rand()+0.00001)/20, 3),  # [0.95-1]
                  transition_soc=np.round(0.9 -
                                          (np.random.rand()+0.00001)/5, 3),  # [0.7-0.9]
                  transition_soc_multiplier=transition_soc_multiplier,
                  battery_capacity=battery_capacity,
                  desired_capacity=env.config["ev"]['desired_capacity'] *
                  battery_capacity,
                  time_of_arrival=step+1,
                  time_of_departure=int(
                      time_of_stay + step + 3),
                  ev_phases=3,
                  timescale=env.timescale,
                  )
    else:
        return EV(id=port,
                  location=cs_id,
                  battery_capacity_at_arrival=initial_battery_capacity,
                  battery_capacity=battery_capacity,
                  desired_capacity=env.config["ev"]['desired_capacity'] *
                  battery_capacity,
                  max_ac_charge_power=env.config["ev"]['max_ac_charge_power'],
                  min_ac_charge_power=env.config["ev"]['min_ac_charge_power'],
                  max_dc_charge_power=env.config["ev"]['max_dc_charge_power'],
                  max_discharge_power=env.config["ev"]['max_discharge_power'],
                  min_discharge_power=env.config["ev"]['min_discharge_power'],
                  time_of_arrival=step+1,
                  time_of_departure=int(
                      time_of_stay + step + 3),
                  ev_phases=env.config["ev"]['ev_phases'],
                  transition_soc=env.config["ev"]['transition_soc'],
                  transition_soc_multiplier=transition_soc_multiplier,
                  charge_efficiency=env.config["ev"]['charge_efficiency'],
                  discharge_efficiency=env.config["ev"]['discharge_efficiency'],
                  timescale=env.timescale,
                  )


def EV_spawner(env) -> List[EV]:
    '''
    This function spawns all the EVs of the current simulation and returns the list of EVs

    Returns:
        EVs: list of EVs
    '''

    ev_list = []

    occupancy_list = np.zeros((env.number_of_ports, env.simulation_length))

    arrival_probabilities = np.random.rand(env.number_of_ports,
                                           env.simulation_length)

    scenario = env.scenario
    user_spawn_multiplier = env.config["spawn_multiplier"]
    time = env.sim_date

    # Define minimum time of stay duration so that an EV can fully charge
    min_time_of_stay = env.config['ev']["min_time_of_stay"]
    min_time_of_stay_steps = min_time_of_stay // env.timescale

    if env.simulation_length-min_time_of_stay_steps-1 < 0:
        raise ValueError(
            "Simulation length is too short for the minimum time of stay! Increase the simulation length or decrease the minimum time of stay.")

    for t in range(2, env.simulation_length-min_time_of_stay_steps-1):
        day = time.weekday()
        hour = time.hour
        minute = time.minute
        # Divide by 15 because the spawn rate is in 15 minute intervals (in the csv file)
        i = hour*4 + minute//15

        if day < 5:
            if scenario == "workplace" and (hour < 6 or hour > 18):
                time = time + datetime.timedelta(minutes=env.timescale)
                continue
            else:
                tau = env.df_arrival_week[scenario].iloc[i]
                multiplier = 1  # 10
        else:
            if scenario == "workplace":
                time = time + datetime.timedelta(minutes=env.timescale)
                continue
            else:
                tau = env.df_arrival_weekend[scenario].iloc[i]

            if day == 5:
                multiplier = 1  # 8
            else:
                multiplier = 1  # 6

        counter = 0
        for cs in env.charging_stations:
            for port in range(cs.n_ports):
                # if port is empty
                if occupancy_list[counter, t] == 0 and \
                    occupancy_list[counter, t-1] == 0 and \
                        occupancy_list[counter, t-2] == 0:
                    # and there is an EV arriving
                    if arrival_probabilities[counter, t]*100 < tau * multiplier * (env.timescale/60) * user_spawn_multiplier:
                        ev = spawn_single_EV(env=env,
                                             scenario=scenario,
                                             cs_id=cs.id,
                                             port=port,
                                             hour=hour,
                                             minute=minute,
                                             step=t,
                                             min_time_of_stay_steps=min_time_of_stay_steps)

                        if ev is not None:
                            ev_list.append(ev)

                            occupancy_list[counter, t +
                                           1:ev.time_of_departure] = 1
                counter += 1
        # step the time
        time = time + datetime.timedelta(minutes=env.timescale)

    return ev_list


def EV_spawner_GF(env) -> List[EV]:
    '''
    This function spawns all the EVs of the current simulation and returns the list of EVs

    Returns:
        EVs: list of EVs
    '''

    ev_list = []

    occupancy_list = np.zeros((env.number_of_ports, env.simulation_length))

    arrival_probabilities = np.random.rand(env.number_of_ports,
                                           env.simulation_length)

    user_spawn_multiplier = env.config["spawn_multiplier"]
    time = env.sim_date

    # Define minimum time of stay duration so that an EV can fully charge
    min_time_of_stay = env.config['ev']["min_time_of_stay"]
    min_time_of_stay_steps = min_time_of_stay // env.timescale

    if env.simulation_length-min_time_of_stay_steps-1 < 0:
        raise ValueError(
            "Simulation length is too short for the minimum time of stay! Increase the simulation length or decrease the minimum time of stay.")

    for t in range(2, env.simulation_length-min_time_of_stay_steps-1):
        day = time.weekday()
        hour = time.hour
        minute = time.minute
        # Divide by 10 because the spawn rate is in 10 minute intervals
        i = hour*6 + minute//10

        tau = env.df_arrival[day, i]
        multiplier = 0.5

        counter = 0
        for cs in env.charging_stations:
            for port in range(cs.n_ports):
                # if port is empty
                if occupancy_list[counter, t] == 0 and \
                    occupancy_list[counter, t-1] == 0 and \
                        occupancy_list[counter, t-2] == 0:
                    # and there is an EV arriving
                    # if arrival_probabilities[counter, t] < tau * multiplier * (env.timescale/60) * user_spawn_multiplier:
                    if arrival_probabilities[counter, t] < tau * 100 * (env.timescale/60) * user_spawn_multiplier * multiplier:
                        ev = spawn_single_EV_GF(env=env,
                                                day=day,
                                                cs_id=cs.id,
                                                port=port,
                                                hour=hour,
                                                minute=minute,
                                                step=t,
                                                min_time_of_stay_steps=min_time_of_stay_steps)

                        if ev is not None:
                            ev_list.append(ev)

                            occupancy_list[counter, t +
                                           1:ev.time_of_departure] = 1
                counter += 1
        # step the time
        time = time + datetime.timedelta(minutes=env.timescale)

    return ev_list


def smooth_vector(v) -> np.ndarray:
    n = len(v)
    smoothed_v = [0] * n

    # Calculate the sum of the original vector
    total_sum = sum(v)

    for i in range(n):
        # Calculate the range for averaging
        start = max(0, i - 1)
        end = min(n, i + 2)

        # Calculate the average of neighboring elements
        smoothed_v[i] = sum(v[start:end]) / (end - start)

    # Adjust the smoothed vector to maintain the original sum
    smoothed_sum = sum(smoothed_v)
    sum_ratio = total_sum / smoothed_sum

    # Apply the ratio to each element of the smoothed vector
    smoothed_v = [value * sum_ratio for value in smoothed_v]

    return smoothed_v


def median_smoothing(v, window_size) -> np.ndarray:
    smoothed_v = np.zeros_like(v)
    half_window = window_size // 2

    for i in range(len(v)):
        start = max(0, i - half_window)
        end = min(len(v), i + half_window + 1)
        smoothed_v[i] = np.median(v[start:end])

    return smoothed_v


def generate_power_setpoints(env) -> np.ndarray:
    '''
    This function generates the power setpoints for the entire simulation using
    the list of EVs and the charging stations from the environment.

    It considers the ev SoC and teh steps required to fully charge the EVs.

    Returns:
        power_setpoints: np.ndarray

    '''

    # Minimal gate for residential V2G setpoints (default-off)
    cfg = getattr(env, 'config', {}).get('res_v2g_setpoints', {})
    use_res_v2g = bool(cfg.get('enabled', False))

    power_setpoints = np.zeros(env.simulation_length)
    # Derive a 1D spot price vector from env.charge_prices (shape: [cs, T])
    cp = np.asarray(env.charge_prices)
    if cp.ndim == 2:
        # Average across charging stations to get a single spot series
        prices = np.mean(cp, axis=0)
    else:
        prices = cp
    prices = np.abs(prices)
    # Day-ahead normalization: use per-timestep max from env.price_forecast (shape: [T, 24])
    zero_price_flag = 0  # debug: 1 if normalization denominator is zero anywhere
    pf = getattr(env, 'price_forecast', None)
    if pf is not None:
        try:
            pf_arr = np.asarray(pf)
            if pf_arr.ndim == 2 and pf_arr.shape[0] == env.simulation_length and pf_arr.shape[1] == 24:
                denom = np.max(np.abs(pf_arr), axis=1)
                eps = 1e-9
                zero_price_flag = 1 if np.any(denom <= eps) else 0
                denom_safe = np.where(denom > eps, denom, 1.0)
                prices = prices / denom_safe
            else:
                # Fallback to global max normalization if shape unexpected
                price_max = float(np.max(prices)) if np.size(prices) > 0 else 0.0
                zero_price_flag = 1 if price_max == 0 else 0
                prices = prices / price_max if price_max > 0 else prices
        except Exception:
            # Fallback on any error
            price_max = float(np.max(prices)) if np.size(prices) > 0 else 0.0
            zero_price_flag = 1 if price_max == 0 else 0
            prices = prices / price_max if price_max > 0 else prices
    else:
        # No forecast available -> fallback to global max normalization
        price_max = float(np.max(prices)) if np.size(prices) > 0 else 0.0
        zero_price_flag = 1 if price_max == 0 else 0
        prices = prices / price_max if price_max > 0 else prices

    # Optional full-series demand/solar from forecasting pipeline
    demand_series = None
    solar_series = None
    if use_res_v2g:
        full_df = getattr(env, 'full_timeseries_data', None)
        if full_df is not None:
            try:
                if 'demand' in full_df.columns:
                    demand_series = np.asarray(full_df['demand'].values[:env.simulation_length], dtype=float)
                if 'solar' in full_df.columns:
                    solar_series = np.asarray(full_df['solar'].values[:env.simulation_length], dtype=float)
            except Exception as e:
                print(f"[DBG setpts] failed to read full_timeseries_data: {e}")

    # Normalize demand/solar to [0,1] if present
    def _norm(x):
        if x is None:
            return None
        xmax = float(np.max(x)) if np.size(x) > 0 else 0.0
        return x / xmax if xmax > 0 else np.zeros_like(x)

    demand_norm = _norm(demand_series)
    solar_norm = _norm(solar_series)

    # Weight parameters (tunable; keep conservative defaults)
    alpha_price = float(cfg.get('alpha_price', 1.0))     # favor low price
    beta_solar = float(cfg.get('beta_solar', 0.5))       # favor high PV
    beta_load = float(cfg.get('beta_load', 0.2))         # favor low demand

    required_energy_multiplier = 100 + \
        env.config.get("power_setpoint_flexibility", 10) 

    min_cs_power = env.charging_stations[0].get_min_charge_power()
    max_cs_power = env.charging_stations[0].get_max_power()

    # Debug counters
    dbg_enabled = bool(getattr(env, 'debug_setpoints', False))
    total_evs_spawned = 0
    dbg = {
        'profiles': len(getattr(env, 'EVs_profiles', []) or []),
        'allocations': 0,
        'short_stay_skips': 0,
        'limit_inversions': 0,
        'zero_price_denom': zero_price_flag,
    }
    for t in range(env.simulation_length):
        counter = total_evs_spawned
        for _, ev in enumerate(env.EVs_profiles[counter:]):
            if ev.time_of_arrival == t:
                total_evs_spawned += 1

                required_energy = ev.battery_capacity - ev.battery_capacity_at_arrival
                required_energy = required_energy * required_energy_multiplier / 100
                min_power_limit = max(ev.min_ac_charge_power, min_cs_power)
                max_power_limit = min(ev.max_ac_charge_power, max_cs_power)

                if min_power_limit > max_power_limit:
                    dbg['limit_inversions'] += 1
                    continue

                # Spread required energy over the time of stay using weights
                horizon = ev.time_of_departure - (t + 1)
                if horizon <= 0:
                    dbg['short_stay_skips'] += 1
                    continue
                # Base weight from prices (prefer low price)
                price_w = prices[t+1:ev.time_of_departure]

                # Forecast-aware weighting if available and enabled
                use_fc = bool(use_res_v2g and demand_norm is not None and solar_norm is not None)
                if use_fc:
                    d_w = demand_norm[t+1:ev.time_of_departure]
                    s_w = solar_norm[t+1:ev.time_of_departure]
                    # combined score: higher when price low, solar high, demand low
                    combined = alpha_price*(1 - price_w) + beta_solar*s_w + beta_load*(1 - d_w)
                    combined = np.clip(combined, 0, None)
                    if np.all(combined == 0) or float(np.sum(combined)) == 0.0:
                        combined = 1 - price_w  # fallback to price only
                    weights = combined
                else:
                    weights = 1 - price_w

                # Add a small jitter to avoid pathological concentration
                eps = 1e-6
                if np.size(weights) > 0:
                    jitter = np.random.normal(loc=0.0, scale=max(eps, float(np.min(weights + eps))), size=horizon)
                    proto = np.clip(weights + jitter, a_min=0, a_max=None)
                    if float(np.sum(proto)) == 0.0:
                        proto = np.ones_like(proto)
                    shifted_load = proto / np.sum(proto)
                else:
                    shifted_load = np.ones(horizon) / max(1, horizon)

                shifted_load = shifted_load * required_energy * 60 / env.timescale

                # find power lower than min_power_limit and higher than max_power_limit
                step = 0
                while np.min(shifted_load[shifted_load != 0]) < min_power_limit or \
                        np.max(shifted_load) > max_power_limit:

                    if step > 10:
                        break

                    # print(f"Shifted load: {shifted_load}")
                    for i in range(len(shifted_load)):
                        if shifted_load[i] < min_power_limit and shifted_load[i] > 0:
                            load_to_shift = shifted_load[i]
                            shifted_load[i] = 0

                            if i == len(shifted_load) - 1:
                                shifted_load[0] += load_to_shift
                            else:
                                shifted_load[i+1] += load_to_shift

                        elif shifted_load[i] > max_power_limit:
                            load_to_shift = shifted_load[i] - max_power_limit
                            shifted_load[i] = max_power_limit

                            if i == len(shifted_load) - 1:
                                shifted_load[0] += load_to_shift
                            else:
                                shifted_load[i+1] += load_to_shift
                    step += 1

                power_setpoints[t+1:ev.time_of_departure] += shifted_load
                dbg['allocations'] += 1

            elif ev.time_of_arrival > t:
                break

    # return smooth_vector(power_setpoints)

    # if env.timescale < 15:
    #     power_setpoints = median_smoothing(power_setpoints, 5)
    #     # make the setpoint have the same value for 15 minutes
    #     new_setpoints = np.zeros(env.simulation_length)
    #     for t in range(env.simulation_length):
    #         # average of the setpoints for the next 15 minutes
    #         new_setpoints[t] = np.mean(power_setpoints[t:t+15])

    #     return new_setpoints
    multiplier = int(15 / env.timescale)
    if multiplier < 1:
        multiplier = 1
    out = median_smoothing(power_setpoints, 5 * multiplier)

    # Optional aggregate setpoint filters: ramp limiting and EMA smoothing in kW
    sp_filt = getattr(env, 'config', {}).get('setpoint_filters', {})
    sp_ramp = (sp_filt or {}).get('ramp_limit', {})
    sp_smooth = (sp_filt or {}).get('smoothing', {})

    ramp_enabled = bool(sp_ramp.get('enabled', False))
    max_ramp_kw = float(sp_ramp.get('max_ramp_kw_per_step', 5.0))
    smooth_enabled = bool(sp_smooth.get('enabled', False))
    ema_alpha = float(sp_smooth.get('ema_alpha', 0.3))

    if ramp_enabled or smooth_enabled:
        # Work on a copy to avoid mutating out during ramp checks
        filt = np.array(out, dtype=float)
        # Ramp limiting first
        if ramp_enabled and len(filt) > 0:
            prev = filt[0]
            for i in range(1, len(filt)):
                x = filt[i]
                delta = x - prev
                if delta > max_ramp_kw:
                    x = prev + max_ramp_kw
                elif delta < -max_ramp_kw:
                    x = prev - max_ramp_kw
                filt[i] = x
                prev = x
        # EMA smoothing next
        if smooth_enabled and len(filt) > 0:
            a = min(max(ema_alpha, 0.0), 1.0)
            y_prev = filt[0]
            for i in range(1, len(filt)):
                y = a * filt[i] + (1 - a) * y_prev
                filt[i] = y
                y_prev = y
        out = filt

    if dbg_enabled:
        try:
            nz = int(np.count_nonzero(out))
            first_nz = int(np.argmax(out != 0)) if nz > 0 else -1
            last_nz = int(len(out) - 1 - np.argmax(out[::-1] != 0)) if nz > 0 else -1
            print(f"[DBG setpts] profiles={dbg['profiles']} evs_seen={total_evs_spawned} allocs={dbg['allocations']} "
                  f"short_stay={dbg['short_stay_skips']} lim_inv={dbg['limit_inversions']} price_denom_zero={dbg['zero_price_denom']} "
                  f"nonzero={nz} span=[{first_nz},{last_nz}]")
        except Exception as e:
            print(f"[DBG setpts] summary error: {e}")

    return out


def calculate_charge_power_potential(env) -> float:
    '''
    This function calculates the total charge power potential of all currently parked EVs for the current time step     
    '''

    power_potential = 0
    for cs in env.charging_stations:
        cs_power_potential = 0
        for port in range(cs.n_ports):
            ev = cs.evs_connected[port]
            if ev is not None:
                if ev.get_soc() < 1 and ev.time_of_departure > env.current_step:
                    phases = min(cs.phases, ev.ev_phases)
                    ev_current = ev.max_ac_charge_power * \
                        1000/(math.sqrt(phases)*cs.voltage)
                    current = min(cs.max_charge_current, ev_current)
                    cs_power_potential += math.sqrt(phases) * \
                        cs.voltage*current/1000

        max_cs_power = math.sqrt(cs.phases) * \
            cs.voltage*cs.max_charge_current/1000
        min_cs_power = math.sqrt(cs.phases) * \
            cs.voltage*cs.min_charge_current/1000

        if cs_power_potential > max_cs_power:
            power_potential += max_cs_power
        elif cs_power_potential < min_cs_power:
            power_potential += 0
        else:
            power_potential += cs_power_potential

    return power_potential
