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
        while step < sim_len:
            weekday = current_dt.weekday()  # 0=Mon
            weekday_key = "weekday" if weekday < 5 else "weekend"
            day_sched: Dict[str, dict] = p_cfg["schedule"].get(weekday_key, {})

            # Build list of (start,end,location) intervals for this day.
            day_blocks: List[Tuple[int, int, int, float]] = []

            # Helper to convert HH:MM to step offset within day.
            def _hm_to_offset(hm: str) -> int:
                hour, minute = map(int, hm.split(":"))
                return (hour * 60 + minute) // timestep_minutes

            for blk_name, blk_cfg in day_sched.items():
                start_ofs = _hm_to_offset(blk_cfg["start_time"])
                end_ofs = _hm_to_offset(blk_cfg["end_time"])
                loc_str = blk_cfg["location"]
                if loc_str == "home":
                    loc = home_station
                    energy = 0.0  # no driving before this block
                elif loc_str == "work":
                    loc = work_station
                    energy = commute_kwh  # just drove from home → work
                else:  # away (short/random trip)
                    loc = home_station  # treat as returning to home after trip
                    # Random duration already in schedule; compute distance via time * avg speed stub
                    energy = (short_trip_cfg.get("avg_distance_km", commute_dist / 2)) * kwh_per_km

                # Compute absolute sim steps.
                abs_start = step + start_ofs
                abs_end = step + end_ofs
                if abs_start >= sim_len:
                    break
                abs_end = min(abs_end, sim_len)
                if abs_end > abs_start:
                    presence_blocks.append(PresenceBlock(abs_start, abs_end, loc, energy))

            # Advance to next day.
            step += (24 * 60) // timestep_minutes
            current_dt += _dt.timedelta(days=1)

        # Convert presence blocks to EVs.
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
                id=ev_id_counter,
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
            if ev_id_counter >= v_count:
                break  # one vehicle per profile for now

    # Sort by arrival time as required by env logic.
    all_ev_profiles.sort(key=lambda ev: ev.time_of_arrival)
    return all_ev_profiles
```
