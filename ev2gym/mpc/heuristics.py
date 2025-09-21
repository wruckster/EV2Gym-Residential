# this class contains heurisyic algorithms for the power setpoint tracking problem
import math
import numpy as np
from typing import List


class RoundRobin():
    '''
    This is a class that contains the Round Robin heuristic algorithm for the power setpoint tracking problem.
    It does not consider multiple transfomer constraints. 
    And it assumes all chargers have the same number of ports
    '''
    algo_name = "Round Robin"

    def __init__(self, env, verbose=False,  **kwargs):

        self.verbose = verbose
        self.env = env
        # find average charging power of the simulation
        self.average_power = 0
        for cs in env.charging_stations:
            self.average_power += cs.max_charge_current * \
                cs.voltage * math.sqrt(cs.phases) / cs.n_ports
        self.average_power /= len(env.charging_stations)

        self.number_of_ports_per_cs = env.number_of_ports_per_cs
        # list with the ids of EVs that were already served in this round
        self.ev_buffer = []

    def get_env(self):
        return self.env

    def update_ev_buffer(self, env) -> None:
        '''
        This function updates the EV buffer list with the EVs that are currently parked by adding or removing them.
        '''
        counter = 0
        # iterate over all ports
        for cs in env.charging_stations:
            for port in range(cs.n_ports):
                if cs.evs_connected[port] is not None:
                    if cs.evs_connected[port].get_soc() < 1:

                        if counter not in self.ev_buffer:
                            self.ev_buffer.insert(0, counter)
                    else:
                        if counter in self.ev_buffer:
                            self.ev_buffer.remove(counter)
                else:
                    if counter in self.ev_buffer:
                        self.ev_buffer.remove(counter)
                counter += 1

    def get_action(self, env) -> np.ndarray:

        # this function returns the action list based on the round robin algorithm

        total_power = env.power_setpoints[env.current_step] * 1000  # in W        

        number_of_EVs_to_charge = total_power / self.average_power

        if self.verbose:
            print("-------------------Round Robin-------------------")
            print(
                f'Number of EVs to charge: {number_of_EVs_to_charge:.2f},\
                total power: {total_power:.2f}, average power: {self.average_power:.2f}')

        # get currently parked EVs
        self.update_ev_buffer(env)
        max_number_of_EVs_to_charge = len(self.ev_buffer)

        if self.verbose:
            print(f'EV buffer: {self.ev_buffer}')

        # get the EVs to charge in this round
        evs_to_charge = self.ev_buffer[:min(
            int(np.ceil(number_of_EVs_to_charge)), max_number_of_EVs_to_charge)]
        self.ev_buffer = self.ev_buffer[min(
            int(np.ceil(number_of_EVs_to_charge)), max_number_of_EVs_to_charge):]

        self.ev_buffer.extend(evs_to_charge)

        # create action list
        action_list = np.zeros(env.number_of_ports)

        # set the action for the EVs to charge
        for i, ev in enumerate(evs_to_charge):
            action_list[ev] = 1 / env.number_of_ports_per_cs
            if i == len(evs_to_charge) - 1 and number_of_EVs_to_charge < len(evs_to_charge):
                action_list[ev] = (number_of_EVs_to_charge - i)

        if self.verbose:
            print(f'Evs to charge: {evs_to_charge}')

        return action_list


class ChargeAsLateAsPossible():
    '''
    This is a class that contains the Charge As Late As Possible heuristic algorithm.
    '''
    algo_name = "Charge As Late As Possible"

    def __init__(self, verbose=False, **kwargs):

        self.verbose = verbose

    def update_ev_buffer(self, env) -> List[int]:
        '''
        This function updates the EV buffer list with the EVs that are currently parked by adding or removing them.
        '''
        ev_buffer = []
        counter = 0
        # iterate over all ports
        for cs in env.charging_stations:
            for port in range(cs.n_ports):
                if cs.evs_connected[port] is not None:

                    cs_max_power = cs.max_charge_current * \
                        cs.voltage * math.sqrt(cs.phases) / 1000
                    # find minimum steps required to charge the EV
                    min_steps = math.ceil((1 - cs.evs_connected[port].get_soc())
                                          / (cs_max_power * env.timescale/60 / cs.evs_connected[port].battery_capacity))

                    start_of_charging_step = cs.evs_connected[port].time_of_departure - min_steps
                    if cs.evs_connected[port].get_soc() < 1 and start_of_charging_step <= env.current_step:
                        ev_buffer.append(counter)

                counter += 1

        return ev_buffer

    def get_action(self, env) -> np.ndarray:

        # this function returns the action list based on the round robin algorithm

        ev_buffer = self.update_ev_buffer(env)
        # create action list
        action_list = np.zeros(env.number_of_ports)

        # set the action for the EVs to charge
        for i, ev in enumerate(ev_buffer):
            action_list[ev] = 1

        return action_list


class ChargeAsFastAsPossible():
    '''
    This class contains the Charge As Fast As Possible heuristic algorithm.
    '''
    algo_name = "Charge As Fast As Possible"
    
    def __init__(self, verbose=False, **kwargs):
        self.verbose = verbose

    def get_action(self, env) -> np.ndarray:
        '''
        This function returns the action list based on the charge as fast as possible algorithm.
        '''
        action_list = np.ones(env.number_of_ports)
        return action_list

class ChargeAsFastAsPossibleWithPowerLimit():
    '''
    This class contains the Charge As Fast As Possible heuristic algorithm with power limit capacity.
    '''
    
    algo_name = "Charge As Fast As Possible"
    
    def __init__(self, env, power_limit, verbose=False, **kwargs):
        self.verbose = verbose
        
        self.average_power = 0
        for cs in env.charging_stations:
            self.average_power += cs.max_charge_current * \
                cs.voltage * math.sqrt(cs.phases) / cs.n_ports
        self.average_power /= len(env.charging_stations)
        print(f'Average power: {self.average_power}')
        self.power_limit = power_limit * 1000  # in W
        self.ev_buffer = []
        
    def update_ev_buffer(self, env) -> None:
        '''
        This function updates the EV buffer list with the EVs that are currently parked by adding or removing them.
        '''
        counter = 0
        # iterate over all ports
        for cs in env.charging_stations:
            for port in range(cs.n_ports):
                if cs.evs_connected[port] is not None:
                    if cs.evs_connected[port].get_soc() < 1:

                        if counter not in self.ev_buffer:
                            self.ev_buffer.insert(0, counter)
                    else:
                        if counter in self.ev_buffer:
                            self.ev_buffer.remove(counter)
                else:
                    if counter in self.ev_buffer:
                        self.ev_buffer.remove(counter)
                counter += 1

    def get_action(self, env) -> np.ndarray:
        '''
        This function returns the action list based on the charge as fast as possible algorithm.
        '''
        
        number_of_EVs_to_charge = self.power_limit / self.average_power
        # print(f'Number of EVs to charge: {number_of_EVs_to_charge}')
        self.update_ev_buffer(env)
        # print(f'EV buffer: {self.ev_buffer}')
        
        evs_to_charge = np.random.choice(self.ev_buffer, min(int(np.ceil(number_of_EVs_to_charge)), len(self.ev_buffer)), replace=False)
        # print(f'Evs to charge: {evs_to_charge}')
        # print(f'Number of EVs to charge: {len(evs_to_charge)}')
                
        action_list = np.zeros(env.number_of_ports)
        # set the action for the EVs to charge
        for i, ev in enumerate(evs_to_charge):
            action_list[ev] = 1 / env.number_of_ports_per_cs
            if i == len(evs_to_charge) - 1 and number_of_EVs_to_charge < len(evs_to_charge) :
                action_list[ev] = (number_of_EVs_to_charge - i)
        # print(action_list)
        return action_list

class ChargeAsFastAsPossibleToDesiredCapacity():
    '''
    This class contains the Charge As Fast As Possible heuristic algorithm.
    '''
    algo_name = "Charge As Fast As Possible To Desired Capacity"

    def __init__(self, verbose=False, **kwargs):
        self.verbose = verbose

    def get_action(self, env) -> np.ndarray:
        '''
        This function returns the action list based on the charge as fast as possible algorithm.
        It charges the EVs up to their desired capacity.
        Desired capacity can be different than the maximum capacity of the battery.
        '''
        action_list = np.zeros(env.number_of_ports)

        counter = 0
        for i, cs in enumerate(env.charging_stations):
            for port in range(cs.n_ports):
                if cs.evs_connected[port] is not None:
                    max_power_to_charge_cs = cs.get_max_power()
                    max_power_to_charge = min(
                        max_power_to_charge_cs, cs.evs_connected[port].max_ac_charge_power)

                    max_energy_to_charge = max_power_to_charge * env.timescale/60
                    if cs.evs_connected[port].current_capacity + max_energy_to_charge < cs.evs_connected[port].desired_capacity:
                        action_list[counter] = 1
                    else:
                        action_list[counter] = ((cs.evs_connected[port].desired_capacity -
                                                cs.evs_connected[port].current_capacity) *
                                                60 / env.timescale) / max_power_to_charge_cs
                        if action_list[counter] < 0:
                            action_list[counter] = 0

                counter += 1

        return action_list

class RoundRobin_GF():
    '''
    This is a class that contains the Round Robin heuristic algorithm for the power setpoint tracking problem. 
    And it assumes all chargers have the same number of ports
    '''
    algo_name = "Round Robin GF"

    def __init__(self, env, verbose=False,  **kwargs):

        self.verbose = verbose
        self.env = env
        # find average charging power of the simulation
        epsilon = 1e-4
        self.max_cs_power = np.zeros(env.action_space.shape)
        
        for i, cs in enumerate(env.charging_stations):
            self.min_action = cs.min_charge_current / cs.max_charge_current + epsilon
            self.max_cs_power[i] = cs.get_max_power()

        self.number_of_ports_per_cs = env.number_of_ports_per_cs
        # list with the ids of EVs that were already served in this round
        self.ev_buffer = []
        self.min_power = []
        self.max_power = []

    def get_env(self):
        return self.env

    def update_ev_buffer(self, env) -> None:
        '''
        This function updates the EV buffer list with the EVs that are currently parked by adding or removing them.
        '''
        counter = 0
        # iterate over all ports
        for cs in env.charging_stations:
            for port in range(cs.n_ports):
                if cs.evs_connected[port] is not None:
                    if cs.evs_connected[port].get_soc() < 1:

                        if counter not in self.ev_buffer:
                            self.ev_buffer.insert(0, counter)
                            min_power = max(cs.get_min_charge_power(), cs.evs_connected[port].min_ac_charge_power)
                            self.min_power.insert(0, min_power)
                            max_power = min(cs.get_max_power(), cs.evs_connected[port].max_ac_charge_power)
                            self.max_power.insert(0, max_power)
                    else:
                        if counter in self.ev_buffer:
                            # find index of the EV in the buffer
                            index = self.ev_buffer.index(counter)
                            self.ev_buffer.remove(counter)
                            self.min_power.pop(index)
                            self.max_power.pop(index)

                else:
                    if counter in self.ev_buffer:
                        index = self.ev_buffer.index(counter)
                        self.ev_buffer.remove(counter)
                        self.min_power.pop(index)
                        self.max_power.pop(index)
                counter += 1

    def get_action(self, env) -> np.ndarray:

        # this function returns the action list based on the round robin algorithm

        power_setpoint = env.power_setpoints[env.current_step]  # in W        


        if self.verbose:
            print("-------------------Round Robin-------------------")
            print(f'Power setpoint: {power_setpoint:.2f}')

        # get currently parked EVs
        self.update_ev_buffer(env)
        # max_number_of_EVs_to_charge = len(self.ev_buffer)

        if self.verbose:
            print(f'EV buffer: {self.ev_buffer}')
            print(f'Min power: {self.min_power}')
            print(f'Max power: {self.max_power}')
            
        
        total_power_potential = sum(self.min_power)        
        
        evs_to_charge = []
        temp_ev_buffer = self.ev_buffer.copy()
        counter = 0
        for EV in temp_ev_buffer:
            next_power = self.max_power[temp_ev_buffer.index(EV)] - \
                                self.min_power[temp_ev_buffer.index(EV)]
            
            if total_power_potential > power_setpoint:
                break
            total_power_potential += next_power
            counter += 1

        # get the EVs to charge in this round
        evs_to_charge = self.ev_buffer[:counter]
        min_power = self.min_power[:counter]
        max_power = self.max_power[:counter]
        
        self.ev_buffer = self.ev_buffer[counter:]
        self.min_power = self.min_power[counter:]
        self.max_power = self.max_power[counter:]
        
        self.ev_buffer.extend(evs_to_charge)
        self.min_power.extend(min_power)
        self.max_power.extend(max_power)

        # create action list
        
        if self.verbose:
            print(f'Final power used: {total_power_potential}')
        
        action_list = np.ones(env.number_of_ports) * self.min_action

        # set the action for the EVs to charge
        for i, ev in enumerate(evs_to_charge):            
            
            if i == len(evs_to_charge) - 1 and total_power_potential >= power_setpoint:
                if total_power_potential - power_setpoint < 0:
                    break
                action_list[ev] = 1 - (total_power_potential - power_setpoint) / self.max_cs_power[ev]
            else:
                action_list[ev] = 1
                
        if self.verbose:
            print(f'Evs to charge: {evs_to_charge}')      
            print(f'Action list: {action_list}')  
        return action_list
    
        
class RoundRobin_GF_off_allowed():
    '''
    This is a class that contains the Round Robin heuristic algorithm for the power setpoint tracking problem.
    It does not consider multiple transfomer constraints. 
    And it assumes all chargers have the same number of ports
    '''
    algo_name = "Round Robin on/off"
    
    def __init__(self, env, verbose=False,  **kwargs):

        self.verbose = verbose
        self.env = env
        # find average charging power of the simulation
        epsilon = 1e-4
        self.max_cs_power = np.zeros(env.action_space.shape)
        
        for i, cs in enumerate(env.charging_stations):
            self.min_action = cs.min_charge_current / cs.max_charge_current + epsilon
            self.max_cs_power[i] = cs.get_max_power()
        
        self.number_of_ports_per_cs = env.number_of_ports_per_cs
        # list with the ids of EVs that were already served in this round
        self.ev_buffer = []
        self.min_power = []
        self.max_power = []

    def get_env(self):
        return self.env

    def update_ev_buffer(self, env) -> None:
        '''
        This function updates the EV buffer list with the EVs that are currently parked by adding or removing them.
        '''
        counter = 0
        # iterate over all ports
        for cs in env.charging_stations:
            for port in range(cs.n_ports):
                if cs.evs_connected[port] is not None:
                    if cs.evs_connected[port].get_soc() < 1:

                        if counter not in self.ev_buffer:
                            self.ev_buffer.insert(0, counter)
                            min_power = max(cs.get_min_charge_power(), cs.evs_connected[port].min_ac_charge_power)
                            self.min_power.insert(0, min_power)
                            max_power = min(cs.get_max_power(), cs.evs_connected[port].max_ac_charge_power)
                            self.max_power.insert(0, max_power)
                    else:
                        if counter in self.ev_buffer:
                            # find index of the EV in the buffer
                            index = self.ev_buffer.index(counter)
                            self.ev_buffer.remove(counter)
                            self.min_power.pop(index)
                            self.max_power.pop(index)

                else:
                    if counter in self.ev_buffer:
                        index = self.ev_buffer.index(counter)
                        self.ev_buffer.remove(counter)
                        self.min_power.pop(index)
                        self.max_power.pop(index)
                counter += 1

    def get_action(self, env) -> np.ndarray:

        # this function returns the action list based on the round robin algorithm

        power_setpoint = env.power_setpoints[env.current_step]  # in W        

        if self.verbose:
            print("-------------------Round Robin-------------------")
            print(f'Power setpoint: {power_setpoint:.2f}')

        # get currently parked EVs
        self.update_ev_buffer(env)
        # max_number_of_EVs_to_charge = len(self.ev_buffer)

        if self.verbose:
            print(f'EV buffer: {self.ev_buffer}')
            print(f'Min power: {self.min_power}')
            print(f'Max power: {self.max_power}')
            
        
        total_power_potential = 0
        
        evs_to_charge = []
        temp_ev_buffer = self.ev_buffer.copy()
        counter = 0
        for EV in temp_ev_buffer:
            next_power = self.max_power[temp_ev_buffer.index(EV)]
                        
            if total_power_potential > power_setpoint:
                break
            total_power_potential += next_power
            counter += 1

        # # get the EVs to charge in this round
        evs_to_charge = self.ev_buffer[:counter]
        min_power = self.min_power[:counter]
        max_power = self.max_power[:counter]
        
        self.ev_buffer = self.ev_buffer[counter:]
        self.min_power = self.min_power[counter:]
        self.max_power = self.max_power[counter:]
        
        self.ev_buffer.extend(evs_to_charge)
        self.min_power.extend(min_power)
        self.max_power.extend(max_power)

        # create action list
        
        if self.verbose:
            print(f'Final power used: {total_power_potential}')
        
        action_list = np.zeros(env.number_of_ports)

        # set the action for the EVs to charge
        for i, ev in enumerate(evs_to_charge):            
            
            if i == len(evs_to_charge) - 1 and total_power_potential > power_setpoint:
                if total_power_potential - power_setpoint <= 0:
                    break
                action_list[ev] = 1 - (total_power_potential - power_setpoint) / self.max_cs_power[ev]
            else:
                action_list[ev] = 1
        
        if self.verbose:
            print(f'Evs to charge: {evs_to_charge}')

        return action_list
    
    
class DoNothing():

    algo_name = "DO NOTHING"

    def __init__(self, verbose=False, **kwargs):
        self.verbose = verbose

    def get_action(self, env) -> np.ndarray:

        action_list = np.zeros(env.number_of_ports)
        return action_list


class RandomAgent():

    algo_name = "Random Actions"

    def __init__(self, verbose=False, **kwargs):
        self.verbose = verbose

    def get_action(self, env) -> np.ndarray:
        min_action = env.action_space.low
        max_action = env.action_space.high
        action_list = np.random.uniform(
            min_action, max_action, env.number_of_ports)
        return action_list

import numpy as np
import math

class LedgerAwareSetpointFollower:
    """
    Follows power setpoints by allocating power across connected EVs, respecting
    their constraints. 

    The strategy involves three main steps:
      1. Determine the current power target (from ledgers, online calculations, or global setpoints).
      2. Gather the capabilities of all connected EVs (min/max charge, max discharge).
      3. Allocate the target power, prioritising V2G for negative targets or to
         offset household load, otherwise allocating charging power greedily.
    """
    algo_name = "Setpoint Follower"

    def __init__(self, use_account_online=True, verbose=False, aggressive_v2g_buffer=0.0):
        self.use_account_online = use_account_online
        self.verbose = verbose
        # Buffer above reserve capacity required for aggressive V2G (kWh)
        # Negative values allow discharging below the reserve
        self.aggressive_v2g_buffer = aggressive_v2g_buffer
        # Flag to completely bypass reserve check for ultra-aggressive V2G
        self._ultra_aggressive_v2g = False

    def _log(self, t: int, message: str):
        """Helper for conditional logging."""
        if self.verbose and (t < 20 or t % 100 == 0):
            print(f"[SetpointFollower] t={t}: {message}")

    def _get_power_target(self, env) -> float:
        """Determines the power target by checking sources in order of priority."""
        t = env.current_step

        # 1. Try to get setpoint from account ledgers.
        try:
            acc_bufs = getattr(env, 'account_buffers', {})
            if acc_bufs:
                setpoints = []
                for cs in getattr(env, 'charging_stations', []):
                    buf = acc_bufs.get(cs.id if not getattr(env, 'roaming_accounts', False) else 0)
                    if buf and 'account_power_setpoint_kw' in getattr(buf, '_data', {}):
                        val = float(buf._data['account_power_setpoint_kw'][t])
                        if np.isfinite(val):
                            setpoints.append(val)
                    if getattr(env, 'roaming_accounts', False):
                        break # Only need the first for roaming.
                if setpoints:
                    self._log(t, f"Using ledger setpoint: {sum(setpoints):.3f} kW")
                    return sum(setpoints)
        except Exception as e:
            self._log(t, f"Ledger read failed: {e}")

        # 2. Try to compute online setpoint if enabled.
        if self.use_account_online:
            try:
                # This part can be further simplified if you abstract away the direct access
                # to env internals, but for now we keep the logic.
                total_setpoint = 0.0
                has_online_setpoint = False
                for cs in getattr(env, 'charging_stations', []):
                    aid = cs.id if not getattr(env, 'roaming_accounts', False) else 0
                    load = float(env.account_load_series.get(aid, {}).get(t, 0.0))
                    pv = float(env.account_pv_series.get(aid, {}).get(t, 0.0))
                    setpoint = float(env._compute_account_online_setpoint(t, load, pv, cs=cs))
                    if np.isfinite(setpoint):
                        total_setpoint += setpoint
                        has_online_setpoint = True
                    if getattr(env, 'roaming_accounts', False):
                        break
                if has_online_setpoint:
                    self._log(t, f"Using online setpoint: {total_setpoint:.3f} kW")
                    return total_setpoint
            except Exception as e:
                self._log(t, f"Online setpoint calculation failed: {e}")

        # 3. Fallback to global setpoint.
        try:
            global_setpoint = float(env.power_setpoints[t])
            if np.isfinite(global_setpoint):
                self._log(t, f"Using global setpoint: {global_setpoint:.3f} kW")
                return global_setpoint
        except Exception as e:
            self._log(t, f"Global setpoint read failed: {e}")

        return 0.0

    def _get_port_specs(self, env) -> list:
        """Gathers charging and discharging specifications for all connected EVs."""
        port_specs = []
        gidx = 0
        for cs in env.charging_stations:
            cs_max_power_kw = float(cs.get_max_power())
            # Calculate max discharge power from max_discharge_current
            try:
                cs_max_dis_kw = float(math.sqrt(getattr(cs, 'phases', 1)) * getattr(cs, 'voltage', 230.0) * abs(getattr(cs, 'max_discharge_current', 0.0)) / 1000.0)
            except Exception:
                cs_max_dis_kw = 0.0

            for p in range(cs.n_ports):
                if (ev := cs.evs_connected[p]) is None:
                    gidx += 1
                    continue

                # Charging specs
                ev_max_kw = float(getattr(ev, 'max_ac_charge_power', cs_max_power_kw))
                max_charge_kw = max(0.0, min(cs_max_power_kw, ev_max_kw))

                # Discharging specs
                can_discharge = False
                reserve_kwh = max(float(getattr(ev, 'min_emergency_battery_capacity', 0.0)),
                                  float(getattr(ev, 'min_battery_capacity', 0.0)))
                current_kwh = float(getattr(ev, 'current_capacity', 0.0))
                
                # V2G is possible if enabled and battery is above reserve + buffer,
                # or if ultra-aggressive V2G is enabled (which bypasses reserve checks)
                if env.config.get('v2g_enabled', False) and (current_kwh > reserve_kwh + self.aggressive_v2g_buffer or getattr(self, '_ultra_aggressive_v2g', False)):
                    ev_max_dis_kw = abs(float(getattr(ev, 'max_discharge_power', 0.0)))
                    max_discharge_kw = max(0.0, min(cs_max_dis_kw, ev_max_dis_kw))
                    if max_discharge_kw > 0:
                        can_discharge = True
                
                port_specs.append({
                    'gidx': gidx, 'cs': cs, 'p': p, 'ev': ev,
                    'max_charge_kw': max_charge_kw,
                    'can_discharge': can_discharge,
                    'max_discharge_kw': max_discharge_kw if can_discharge else 0.0,
                    'cs_max_power_kw': cs_max_power_kw, # For normalisation
                    'cs_max_dis_kw': cs_max_dis_kw,     # For normalisation
                })
                gidx += 1
        return port_specs

    def _allocate_power(self, target_kw: float, port_specs: list, env_num_ports: int) -> np.ndarray:
        """Allocates a positive (charging) or negative (discharging) power target."""
        # Use the environment's number_of_ports to ensure correct array size
        actions = np.zeros(env_num_ports, dtype=float)

        # --- Discharging Logic ---
        if target_kw < -0.01: # Target is to discharge (export power).
            discharge_ports = [p for p in port_specs if p['can_discharge']]
            total_dis_cap = sum(p['max_discharge_kw'] for p in discharge_ports)

            if total_dis_cap < 1e-6:
                return actions # No capacity to meet discharge target.

            desired_dis_kw = min(abs(target_kw), total_dis_cap)
            for port in discharge_ports:
                share = port['max_discharge_kw'] / total_dis_cap if total_dis_cap > 0 else 0.0
                dis_kw = desired_dis_kw * share
                denom = port['cs_max_dis_kw'] if port['cs_max_dis_kw'] > 1e-6 else 1.0
                actions[port['gidx']] = -float(np.clip(dis_kw / denom, 0.0, 1.0))
            return actions

        # --- Charging Logic ---
        # Sort by max power to prioritise higher capacity EVs
        charge_ports = sorted(port_specs, key=lambda x: x['max_charge_kw'], reverse=True)
        residual_kw = max(0.0, target_kw)

        for port in charge_ports:
            alloc_kw = min(port['max_charge_kw'], residual_kw)
            denom = port['cs_max_power_kw'] if port['cs_max_power_kw'] > 1e-6 else 1.0
            actions[port['gidx']] = float(np.clip(alloc_kw / denom, 0.0, 1.0))
            residual_kw -= alloc_kw
            if residual_kw < 1e-6:
                break
        
        return actions

    def get_action(self, env) -> np.ndarray:
        """Computes the charging/discharging action for each port."""
        t = env.current_step
        num_ports = env.number_of_ports
        actions = np.zeros(num_ports, dtype=float)

        port_specs = self._get_port_specs(env)
        if not port_specs:
            return actions

        site_target_kw = self._get_power_target(env)
        final_target_kw = site_target_kw

        # Check for aggressive V2G to offset household net grid draw using ledger series.
        # Convention: PV generation is negative, so net_grid = load + pv.
        if env.config.get('v2g_enabled', False) and site_target_kw >= 0:
            # This logic assumes a single, aggregated household load for simplicity.
            try:
                buf = getattr(env, 'account_buffers', {}).get(0)
                if buf and 'household_inflexible_load_kw' in buf._data:
                    load = float(buf._data['household_inflexible_load_kw'][t])
                    pv = float(buf._data['household_pv_kw'][t])
                    net_grid_kw = load + pv
                    if net_grid_kw > 0.01:
                        # Net import: discharge to offset grid draw
                        self._log(t, f"Aggressive V2G: offsetting net import of {net_grid_kw:.3f} kW")
                        self._log(t, f"Ultra-aggressive V2G mode: ignoring reserve limits")
                        final_target_kw = -net_grid_kw  # discharge
                        self._ultra_aggressive_v2g = True
                    elif net_grid_kw < -0.01:
                        # Net export (surplus PV): charge to absorb surplus
                        surplus_kw = abs(net_grid_kw)
                        self._log(t, f"Aggressive PV absorb: charging to use surplus of {surplus_kw:.3f} kW")
                        final_target_kw = surplus_kw  # charge
                        self._ultra_aggressive_v2g = False
                    else:
                        self._ultra_aggressive_v2g = False
                else:
                    self._ultra_aggressive_v2g = False
            except Exception:
                self._ultra_aggressive_v2g = False
                pass  # Fail silently if ledger data is not available.
        else:
            self._ultra_aggressive_v2g = False

        self._log(t, f"Site Target: {site_target_kw:.3f} kW, Final Target: {final_target_kw:.3f} kW")

        return self._allocate_power(final_target_kw, port_specs, num_ports)

class SimpleSOCMaintainer:
    '''
    A simple heuristic that ignores setpoints and just maintains EV SOC above a minimum level.
    Useful for testing if the charging mechanics work at all.
    '''
    algo_name = "Simple SOC Maintainer"

    def __init__(self, target_soc: float = 0.8, min_soc: float = 0.2, verbose: bool = True, **kwargs):
        self.target_soc = target_soc
        self.min_soc = min_soc
        self.verbose = verbose

    def get_action(self, env) -> np.ndarray:
        t = env.current_step
        actions = np.zeros(env.number_of_ports, dtype=float)
        
        gidx = 0
        for cs in env.charging_stations:
            cs_max_power_kw = float(cs.get_max_power())
            for p in range(cs.n_ports):
                if cs.evs_connected[p] is None:
                    gidx += 1
                    continue
                
                ev = cs.evs_connected[p]
                try:
                    current_soc = ev.get_soc()
                except:
                    current_soc = 0.5  # fallback
                
                # Simple logic: charge at max rate if SOC < target, otherwise no charging
                if current_soc < self.target_soc:
                    actions[gidx] = 1.0  # Max charging rate
                else:
                    actions[gidx] = 0.0  # No charging
                
                if self.verbose and t % 500 == 0:
                    print(f"[SOCMaintainer] t={t} Port {gidx}: SOC={current_soc:.3f}, action={actions[gidx]:.3f}")
                
                gidx += 1
        
        return actions


class ChargeAsLateAsPossibleToDesiredCapacity():
    '''
    This class contains the Charge As Late As Possible heuristic algorithm.
    '''
    algo_name = "Charge As Late As Possible To Desired Capacity"

    def __init__(self, verbose=False, **kwargs):

        self.verbose = verbose

    def update_ev_buffer(self, env) -> List[int]:
        '''
        This function updates the EV buffer list with the EVs that are currently parked by adding or removing them.
        '''
        ev_buffer = []
        ev_action_list = []
        counter = 0
        # iterate over all ports
        for cs in env.charging_stations:
            for port in range(cs.n_ports):
                if cs.evs_connected[port] is not None:

                    desired_soc = cs.evs_connected[port].desired_capacity / \
                        cs.evs_connected[port].battery_capacity
                    cs_max_power = cs.max_charge_current * \
                        cs.voltage * math.sqrt(cs.phases) / 1000
                    # find minimum steps required to charge the EV
                    min_steps = math.ceil((desired_soc - cs.evs_connected[port].get_soc())
                                          / (cs_max_power * env.timescale/60 / cs.evs_connected[port].battery_capacity))

                    start_of_charging_step = cs.evs_connected[port].time_of_departure - min_steps
                    if cs.evs_connected[port].get_soc() < desired_soc and start_of_charging_step <= env.current_step:
                        ev_buffer.append(counter)

                        min_step = (
                            desired_soc - cs.evs_connected[port].get_soc()) / (cs_max_power * env.timescale/60 / cs.evs_connected[port].battery_capacity)
                        if min_step < 1:
                            ev_action_list.append(min_step)
                        else:
                            ev_action_list.append(1)

                counter += 1

        return ev_buffer, ev_action_list

    def get_action(self, env) -> np.ndarray:

        # this function returns the action list based on the round robin algorithm

        ev_buffer, ev_action_list = self.update_ev_buffer(env)
        # create action list
        action_list = np.zeros(env.number_of_ports)

        # set the action for the EVs to charge
        for i, ev in enumerate(ev_buffer):
            action_list[ev] = ev_action_list[i]

        return action_list