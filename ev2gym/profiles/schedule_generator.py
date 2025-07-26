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
from typing import Dict, List, Tuple, Optional
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
    is_plugged_in: bool = True  # whether the EV is plugged in at this location
    location_type: str = "home"  # one of: "home", "work", "away"


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
    print(f"DEBUG: Starting date: {start_datetime}, simulation length: {sim_len}, timestep: {timestep_minutes}min")

    all_ev_profiles: List[EV] = []
    ev_id_counter = 0

    for profile_key, p_cfg in cfg.items():
        print(f"DEBUG: Processing profile: {profile_key}")
        v_count: int = int(p_cfg.get("vehicle_count", 1))
        home_station = int(p_cfg.get("default_station", 0))
        work_station = int(p_cfg.get("work_station", 1))
        commute_dist = float(p_cfg["commute"].get("distance_km", 20))
        commute_min_time = int(p_cfg["commute"].get("min_time_minutes", 30))
        commute_max_time = int(p_cfg["commute"].get("max_time_minutes", 60))
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

            start_hour_offset = env.sim_starting_date.hour * 60 + env.sim_starting_date.minute
            start_offset_steps = start_hour_offset // timestep_minutes
            steps_in_day = (24 * 60) // timestep_minutes

            # Helper to convert HH:MM to step offset from midnight.
            def _hm_to_offset(hm: str) -> int:
                hour, minute = map(int, hm.split(":"))
                return (hour * 60 + minute) // timestep_minutes
            
            # Process main schedule blocks
            regular_blocks = []
            for blk_name, blk_cfg in day_sched.items():
                start_ofs = _hm_to_offset(blk_cfg["start_time"])
                end_ofs = _hm_to_offset(blk_cfg["end_time"])

                # Handle overnight blocks by adding a day to the end time
                if end_ofs < start_ofs:
                    end_ofs += steps_in_day

                # Make offsets relative to simulation start time
                start_ofs -= start_offset_steps
                end_ofs -= start_offset_steps

                loc_str = blk_cfg["location"]
                is_plugged_in = True  # Default is plugged in
                
                if loc_str == "home":
                    loc = home_station
                    loc_type = "home"
                    energy = 0.0  # no driving before this block
                elif loc_str == "work":
                    loc = work_station
                    loc_type = "work"
                    energy = commute_kwh  # just drove from home → work
                else:  # away (short/random trip)
                    loc = -1  # Not at any charging station
                    loc_type = "away"
                    is_plugged_in = False
                    # Random duration already in schedule; compute distance via time * avg speed stub
                    energy = (short_trip_cfg.get("avg_distance_km", commute_dist / 2)) * kwh_per_km

                # Compute absolute sim steps.
                abs_start = step + start_ofs
                abs_end = step + end_ofs
                if abs_start >= sim_len:
                    break
                abs_end = min(abs_end, sim_len)
                if abs_end > abs_start:
                    regular_blocks.append((abs_start, abs_end, loc, loc_type, energy, is_plugged_in))
            
            # Process short trips (random trips during blocks)
            final_blocks = []
            for start, end, loc, loc_type, energy, is_plugged_in in regular_blocks:
                # If we're at home or work, check for random trips
                if loc_type in ["home", "work"]:
                    # Find original block name by matching start time
                    # Reverse the offset calculation to find the original offset from midnight
                    original_start_ofs = (start - step) + start_offset_steps

                    blk_name = next((name for name, cfg in day_sched.items() 
                                     if _hm_to_offset(cfg["start_time"]) == original_start_ofs), None)
                    
                    if blk_name:
                        blk_cfg = day_sched[blk_name]
                        prob_short_trip = float(blk_cfg.get("prob_short_trip", 0.0))
                        
                        # Check if we should generate a short trip
                        if prob_short_trip > 0 and rng.random() < prob_short_trip:
                            # Get trip duration parameters
                            trip_dur_range = blk_cfg.get("short_trip_duration", [15, 120])
                            min_dur, max_dur = trip_dur_range
                            
                            # Ensure duration is within block bounds
                            max_possible_dur = (end - start) * timestep_minutes // 2  # Max half the block
                            actual_max_dur = min(max_dur, max_possible_dur)
                            
                            if actual_max_dur > min_dur:
                                # Generate random trip duration
                                trip_dur_min = rng.integers(min_dur, actual_max_dur)
                                trip_dur_steps = trip_dur_min // timestep_minutes
                                
                                # Calculate trip start time (avoid very start/end of block)
                                earliest_start = start + 1
                                latest_start = end - trip_dur_steps - 1
                                
                                if latest_start > earliest_start:
                                    trip_start = rng.integers(earliest_start, latest_start)
                                    trip_end = trip_start + trip_dur_steps
                                    
                                    # Split the block into: before trip, trip, after trip
                                    if trip_start > start:
                                        final_blocks.append(PresenceBlock(
                                            start_step=start,
                                            end_step=trip_start,
                                            location=loc,
                                            energy_consumed=energy,
                                            is_plugged_in=is_plugged_in,
                                            location_type=loc_type
                                        ))
                                        
                                    # Add trip block (unplugged)
                                    trip_distance = (short_trip_cfg.get("avg_distance_km", commute_dist / 3)) * kwh_per_km
                                    final_blocks.append(PresenceBlock(
                                        start_step=trip_start,
                                        end_step=trip_end,
                                        location=-1,  # Away
                                        energy_consumed=trip_distance,
                                        is_plugged_in=False,
                                        location_type="away"
                                    ))
                                    
                                    if trip_end < end:
                                        final_blocks.append(PresenceBlock(
                                            start_step=trip_end,
                                            end_step=end,
                                            location=loc,
                                            energy_consumed=trip_distance,
                                            is_plugged_in=is_plugged_in,
                                            location_type=loc_type
                                        ))
                                    continue
                
                # If no trip was generated, add the original block
                final_blocks.append(PresenceBlock(
                    start_step=start,
                    end_step=end,
                    location=loc,
                    energy_consumed=energy,
                    is_plugged_in=is_plugged_in,
                    location_type=loc_type
                ))
                        
            # Add commutes between blocks if needed
            presence_blocks.extend(final_blocks)
            
            # Debug the blocks we've generated so far
            print(f"DEBUG: Day {current_dt.strftime('%Y-%m-%d')}, added {len(final_blocks)} blocks, total now: {len(presence_blocks)}")
                
            # Advance to next day.
            step += (24 * 60) // timestep_minutes
            current_dt += _dt.timedelta(days=1)

        # Convert presence blocks to EVs.
        print(f"DEBUG: Generated {len(presence_blocks)} presence blocks for profile {profile_key}")
        
        # Debug the first few blocks to understand their properties
        for i, blk in enumerate(presence_blocks[:5]):
            print(f"DEBUG: Block {i}: location={blk.location}, is_plugged_in={blk.is_plugged_in}, location_type={blk.location_type}, start={blk.start_step}, end={blk.end_step}")
            
        # Collect all plugged-in blocks (home/work) for this profile
        plugged_blocks = [b for b in presence_blocks if b.is_plugged_in]

        if not plugged_blocks:
            continue  # nothing to generate

        # Create one EV profile per vehicle count, each with its own schedule.
        for _ in range(v_count):
            ev_id_counter += 1

            first_block = plugged_blocks[0]
            last_block = plugged_blocks[-1]

            # If the first block starts before the simulation, set start time to 0.
            if first_block.start_step < 0:
                first_block.start_step = 0

            ev_spec = env.config["ev"]
            batt_kwh = float(ev_spec["battery_capacity"])
            min_capacity = float(ev_spec.get("min_battery_capacity", 10))
            soc_arrival = rng.uniform(0.4, 0.8)
            cap_arrival = max(min_capacity, soc_arrival * batt_kwh - first_block.energy_consumed)
            cap_arrival = max(min_capacity, cap_arrival)

            desired_capacity = float(ev_spec.get("desired_capacity", 0.8)) * batt_kwh

            ev_profile = EV(
                id=f"ev_{profile_key}_{ev_id_counter}",
                location=first_block.location,
                battery_capacity_at_arrival=cap_arrival,
                time_of_arrival=first_block.start_step,
                time_of_departure=first_block.end_step,
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
                transition_soc_multiplier=float(ev_spec.get("transition_soc_multiplier", 1.0)),
                charge_efficiency=ev_spec["charge_efficiency"],
                discharge_efficiency=ev_spec["discharge_efficiency"],
                timescale=timestep_minutes,
                metadata={"presence_blocks": [(b.start_step, b.end_step, b.location) for b in plugged_blocks]},
                location_state=0 if first_block.location_type == "home" else 1,
            )

            # Build full transition list: for each consecutive block pair, add depart & arrive.
            for idx, blk in enumerate(plugged_blocks[:-1]):
                next_blk = plugged_blocks[idx + 1]
                # end of current block -> commuting
                ev_profile.add_schedule_transition(blk.end_step, 2)
                # start of next block -> new location state (home=0, work=1)
                new_state = 0 if next_blk.location_type == "home" else 1
                ev_profile.add_schedule_transition(next_blk.start_step, new_state)

            # Extend overall availability so EV object exists until after final block
            ev_profile.time_of_departure = last_block.end_step

            all_ev_profiles.append(ev_profile)

    # Sort by arrival time as required by env logic.
    all_ev_profiles.sort(key=lambda ev: ev.time_of_arrival)
    print(f"DEBUG: Total EV profiles created: {len(all_ev_profiles)}")
    return all_ev_profiles
