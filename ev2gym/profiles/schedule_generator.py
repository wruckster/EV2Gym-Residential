"""Schedule generator for user-defined EV presence profiles.

This module converts the `user_profiles` section of the YAML config into
concrete EV spawn profiles that EV2Gym can understand.

A simplified approach is used:
1. A presence schedule is generated at simulation resolution (`env.timescale`).
2. Each continuous presence interval is mapped to one EV profile
   (arrival at the first step, departure at the last+1 step).
3. Battery capacity on arrival is computed from: previous departure SoC −
   energy consumed on the trip (distance * kWh_per_km).

The implementation targets residential/private scenarios with exactly two
charging stations: home (0) and work (1). It gracefully falls back to the
legacy random spawner if `user_profiles` is absent.

THIS IS AN INITIAL IMPLEMENTATION meant to unblock testing.  Improvements
(support for >2 stations, sophisticated stochastic trip distances, etc.) can
be layered on later.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from ev2gym.models.ev.vehicle import EV

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PresenceBlock:
    start_step: int  # inclusive
    end_step: int    # exclusive
    location: int    # station id
    energy_consumed: float  # kWh consumed while away before this block


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def generate_ev_profiles(env) -> List[EV]:  # noqa: C901 – complexity okay for now
    """Generate a deterministic list of EV profiles from *user_profiles* config.

    Parameters
    ----------
    env : EV2Gym
        The active environment.  Must have ``config`` fully loaded and
        attributes such as ``simulation_length`` and ``timescale`` defined.

    Returns
    -------
    List[EV]
        List sorted by ``time_of_arrival`` as expected by downstream code.
    """
    cfg = env.config.get("user_profiles")
    if cfg is None:
        # Fallback handled by caller.
        return []

    rng = np.random.default_rng(seed=env.seed)
    timestep_minutes = env.timescale
    sim_len = env.simulation_length

    start_datetime = env.sim_starting_date

    all_ev_profiles: List[EV] = []
    ev_id_counter = 0

    for profile_key, p_cfg in cfg.items():
        v_count: int = int(p_cfg.get("vehicle_count", 1))
        home_station = int(p_cfg.get("default_station", 0))
        work_station = int(p_cfg.get("work_station", 1))
        commute_dist = float(p_cfg["commute"].get("distance_km", 20))
        kwh_per_km = float(p_cfg["commute"].get("consumption_kwh_per_km", 0.18))
        short_trip_cfg = p_cfg.get("short_trip", {})

        # Pre-compute energy per one-way commute.
        commute_kwh = commute_dist * kwh_per_km

        # Generate presence blocks day by day.
        presence_blocks: List[PresenceBlock] = []

        current_dt = start_datetime
        step = 0

        # First, collect all non-home blocks for the entire simulation period.
        non_home_blocks: List[Tuple[int, int, int, float]] = []

        while step < sim_len:
            weekday = current_dt.weekday()  # 0=Mon
            weekday_key = "weekday" if weekday < 5 else "weekend"
            day_sched: Dict[str, dict] = p_cfg["schedule"].get(weekday_key, {})

            # Helper to convert HH:MM to step offset within day.
            def _hm_to_offset(hm: str) -> int:
                hour, minute = map(int, hm.split(":"))
                return (hour * 60 + minute) // timestep_minutes

            for blk_name, blk_cfg in day_sched.items():
                start_ofs = _hm_to_offset(blk_cfg["start_time"])
                end_ofs = _hm_to_offset(blk_cfg["end_time"])
                loc_str = blk_cfg["location"]

                energy = 0.0
                if loc_str == "work":
                    loc = work_station
                    energy = commute_kwh
                elif loc_str == "away":
                    # For 'away' blocks, we don't have a fixed location, but we can treat it
                    # as a trip that consumes energy. The vehicle is not available.
                    # We will model this by just having a gap in presence.
                    energy = (short_trip_cfg.get("avg_distance_km", commute_dist / 2)) * kwh_per_km
                    loc = -1 # Sentinel for 'away'
                else: # home
                    continue # Skip home blocks, we will fill them in later

                abs_start = step + start_ofs
                abs_end = step + end_ofs
                if abs_start >= sim_len:
                    break
                abs_end = min(abs_end, sim_len)
                if abs_end > abs_start:
                    non_home_blocks.append((abs_start, abs_end, loc, energy))

            step += (24 * 60) // timestep_minutes
            current_dt += _dt.timedelta(days=1)

        # Sort the blocks by start time to handle them chronologically.
        non_home_blocks.sort(key=lambda x: x[0])

        # Now, create the final presence schedule, filling gaps with 'home' blocks.
        last_end_step = 0
        for start, end, loc, energy in non_home_blocks:
            # If there's a gap before this block, it's a 'home' block.
            if start > last_end_step:
                presence_blocks.append(PresenceBlock(last_end_step, start, home_station, 0.0))
            
            # Add the actual 'work' or 'away' block if it's not 'away'.
            if loc != -1:
                presence_blocks.append(PresenceBlock(start, end, loc, energy))
            
            last_end_step = end

        # Add a final home block if the simulation doesn't end with a non-home block.
        if last_end_step < sim_len:
            presence_blocks.append(PresenceBlock(last_end_step, sim_len, home_station, 0.0))


        # Convert presence blocks to EVs. Now, create `v_count` vehicles.
        for i in range(v_count):
            # Associate a unique EV ID for each vehicle in the profile
            vehicle_ev_id = f"{profile_key}_{i}"
            for blk in presence_blocks:
                # Arrival battery capacity: random 40-80% minus trip energy, but at least min capacity.
                ev_spec = env.config["ev"]
                batt_kwh = float(ev_spec["battery_capacity"])
                min_capacity = float(ev_spec.get("min_battery_capacity", 10))
                soc_arrival = rng.uniform(0.4, 0.8)
                cap_arrival = max(min_capacity, soc_arrival * batt_kwh - blk.energy_consumed)
                cap_arrival = max(min_capacity, cap_arrival)

                desired_capacity = float(ev_spec.get("desired_capacity", 0.8)) * batt_kwh
                ev_profile = EV(
                    id=f"{vehicle_ev_id}_{blk.start_step}", # Unique ID per presence block
                    location=blk.location,
                    battery_capacity_at_arrival=cap_arrival,
                    time_of_arrival=blk.start_step,
                    time_of_departure=blk.end_step,
                    desired_capacity=desired_capacity,
                    battery_capacity=batt_kwh,
                    min_battery_capacity=min_capacity,
                    min_emergency_battery_capacity=float(ev_spec.get("min_emergency_battery_capacity", 15)),
                    max_ac_charge_power=float(ev_spec["max_ac_charge_power"]),
                    min_ac_charge_power=float(ev_spec["min_ac_charge_power"]),
                    max_dc_charge_power=float(ev_spec["max_dc_charge_power"]),
                    max_discharge_power=float(ev_spec["max_discharge_power"]),
                    min_discharge_power=float(ev_spec["min_discharge_power"]),
                    ev_phases=int(ev_spec["ev_phases"]),
                    transition_soc=float(ev_spec["transition_soc"]),
                    charge_efficiency=ev_spec["charge_efficiency"],
                    discharge_efficiency=ev_spec["discharge_efficiency"],
                    timescale=env.timescale,
                )
                all_ev_profiles.append(ev_profile)
                ev_id_counter += 1

    # Sort by arrival time as required by env logic.
    all_ev_profiles.sort(key=lambda ev: ev.time_of_arrival)
    return all_ev_profiles
