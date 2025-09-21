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
    

class LedgerAwareSetpointFollower:
    '''
    A baseline heuristic that follows the environment's global or per-account
    setpoints and allocates power across connected EVs respecting charger and EV
    constraints. It leverages:
      - Global setpoints: env.power_setpoints[t]
      - Optional per-account online setpoints via env._compute_account_online_setpoint()
      - Charger max/min power and EV min/max AC power

    Action semantics:
      - Returns a normalized action vector in [-1, 1] per port, consistent with env.action_space
        (discharging not used unless env.config['v2g_enabled'] is True and target < 0).
    '''
    algo_name = "Ledger-Aware Setpoint Follower"

    def __init__(self, use_account_online: bool = True, verbose: bool = True, aggressive_v2g_buffer: float = 0.0, **kwargs):
        self.verbose = verbose
        self.use_account_online = use_account_online
        # Buffer above reserve capacity required for aggressive V2G (kWh)
        self.aggressive_v2g_buffer = aggressive_v2g_buffer

    def _current_target_kw(self, env) -> float:
        t = env.current_step
        # Preferred: read per-account setpoints directly from account ledgers
        try:
            acc_bufs = getattr(env, 'account_buffers', {})
            if acc_bufs:
                if getattr(env, 'roaming_accounts', False):
                    buf = acc_bufs.get(0)
                    if buf is not None and 'account_power_setpoint_kw' in getattr(buf, '_data', {}):
                        val = float(buf._data['account_power_setpoint_kw'][t])
                        if self.verbose and t % 500 == 0:
                            print(f"[LedgerAware] t={t} ledger setpoint (roaming, aid=0): {val:.3f}")
                        if np.isfinite(val):
                            return val
                else:
                    total = 0.0
                    have_any = False
                    for cs in getattr(env, 'charging_stations', []):
                        buf = acc_bufs.get(cs.id)
                        if buf is not None and 'account_power_setpoint_kw' in getattr(buf, '_data', {}):
                            v = float(buf._data['account_power_setpoint_kw'][t])
                            if np.isfinite(v):
                                total += v
                                have_any = True
                    if have_any:
                        if self.verbose and t % 500 == 0:
                            print(f"[LedgerAware] t={t} ledger setpoint (sum over accounts): {total:.3f}")
                        return float(total)
        except Exception as e:
            if self.verbose and t % 500 == 0:
                print(f"[LedgerAware] t={t} ledger read failed: {e}")
        if self.use_account_online:
            # If online account setpoints are enabled, aggregate per-account targets
            try:
                sp_cfg = env._sp_cfg()
                online_enabled = bool(sp_cfg.get('account_online', {}).get('enabled', False))
                if self.verbose and t % 500 == 0:
                    print(f"[LedgerAware] t={t} sp_cfg: {sp_cfg}")
                    print(f"[LedgerAware] t={t} online_enabled: {online_enabled}")
                    print(f"[LedgerAware] t={t} roaming_accounts: {getattr(env, 'roaming_accounts', False)}")
                
                if online_enabled:
                    if getattr(env, 'roaming_accounts', False):
                        # Check if account series are populated
                        has_load_series = 0 in env.account_load_series
                        has_pv_series = 0 in env.account_pv_series
                        if self.verbose and t % 500 == 0:
                            print(f"[LedgerAware] t={t} account_load_series keys: {list(env.account_load_series.keys())}")
                            print(f"[LedgerAware] t={t} account_pv_series keys: {list(env.account_pv_series.keys())}")
                            print(f"[LedgerAware] t={t} has_load_series: {has_load_series}, has_pv_series: {has_pv_series}")
                        
                        load0 = float(env.account_load_series.get(0).iloc[t]) if has_load_series else 0.0
                        pv0 = float(env.account_pv_series.get(0).iloc[t]) if has_pv_series else 0.0
                        setpoint = float(env._compute_account_online_setpoint(t, load0, pv0, cs=None))
                        if self.verbose and t % 500 == 0:
                            print(f"[LedgerAware] t={t} roaming mode: load0={load0:.3f}, pv0={pv0:.3f}, setpoint={setpoint:.3f}")
                        return setpoint
                    else:
                        s = 0.0
                        for cs in env.charging_stations:
                            aid = cs.id
                            load_j = float(env.account_load_series.get(aid).iloc[t]) if aid in env.account_load_series else 0.0
                            pv_j = float(env.account_pv_series.get(aid).iloc[t]) if aid in env.account_pv_series else 0.0
                            setpoint_j = float(env._compute_account_online_setpoint(t, load_j, pv_j, cs=cs))
                            s += setpoint_j
                            if self.verbose and t % 500 == 0:
                                print(f"[LedgerAware] t={t} CS {aid}: load={load_j:.3f}, pv={pv_j:.3f}, setpoint={setpoint_j:.3f}")
                        if self.verbose and t % 500 == 0:
                            print(f"[LedgerAware] t={t} multi-account total setpoint: {s:.3f}")
                        return float(s)
            except Exception as e:
                if self.verbose and t % 500 == 0:
                    print(f"[LedgerAware] t={t} Exception in online setpoint: {e}")
                pass
        # Fallback to global setpoint
        try:
            global_setpoint = float(env.power_setpoints[t])
            if self.verbose and t % 500 == 0:
                print(f"[LedgerAware] t={t} Using global setpoint: {global_setpoint:.3f}")
            return global_setpoint
        except Exception as e:
            if self.verbose and t % 500 == 0:
                print(f"[LedgerAware] t={t} Exception in global setpoint: {e}")
            return 0.0

    def get_action(self, env) -> np.ndarray:
        t = env.current_step
        target_kw = self._current_target_kw(env)
        
        # For aggressive V2G, check if we have net positive usage that could be offset
        aggressive_v2g = env.config.get('v2g_enabled', False)
        net_positive_usage = False
        net_usage_kw = 0.0
        
        if aggressive_v2g:
            try:
                # Get household load and PV from account buffers
                acc_bufs = getattr(env, 'account_buffers', {})
                if acc_bufs:
                    if getattr(env, 'roaming_accounts', False):
                        buf = acc_bufs.get(0)
                        if buf is not None and hasattr(buf, '_data'):
                            ld = buf._data.get('household_inflexible_load_kw')
                            pv = buf._data.get('household_pv_kw')
                            if ld is not None and pv is not None:
                                ldv = float(ld[t]) if np.isfinite(ld[t]) else 0.0
                                pvv = float(pv[t]) if np.isfinite(pv[t]) else 0.0
                                net_usage_kw = ldv - pvv
                                net_positive_usage = net_usage_kw > 0.01  # Small threshold to avoid noise
                                if net_positive_usage and self.verbose and t % 100 == 0:
                                    print(f"[LedgerAware] t={t} Aggressive V2G: net_usage={net_usage_kw:.3f}kW")
            except Exception as e:
                if self.verbose and t % 500 == 0:
                    print(f"[LedgerAware] t={t} Exception in aggressive V2G check: {e}")

        # Debug: print target and ledger-derived inputs frequently at start, then every 100 steps
        if self.verbose and (t < 20 or t % 100 == 0):
            try:
                acc_bufs = getattr(env, 'account_buffers', {})
                if acc_bufs:
                    if getattr(env, 'roaming_accounts', False):
                        buf = acc_bufs.get(0)
                        if buf is not None and hasattr(buf, '_data'):
                            sp = buf._data.get('account_power_setpoint_kw')
                            ld = buf._data.get('household_inflexible_load_kw')
                            pv = buf._data.get('household_pv_kw')
                            spv = float(sp[t]) if sp is not None else float('nan')
                            ldv = float(ld[t]) if ld is not None else float('nan')
                            pvv = float(pv[t]) if pv is not None else float('nan')
                            print(f"[LedgerAware][dbg] t={t} ledger_setpoint={spv:.3f} load={ldv:.3f} pv={pvv:.3f}")
            except Exception:
                pass
            try:
                connected_ports = sum(1 for cs in env.charging_stations for ev in cs.evs_connected if ev is not None)
            except Exception:
                connected_ports = -1
            print(f"[LedgerAware][dbg] t={t} target_kw={target_kw:.3f} v2g={env.config.get('v2g_enabled', False)} connected_ports={connected_ports}")

        # Clamp target by global charge power potential if available
        try:
            cpp = float(env.charge_power_potential[t])
            if np.isfinite(cpp):
                target_kw = np.sign(target_kw) * min(abs(target_kw), cpp)
        except Exception:
            pass

        # Gather connected EV ports and their min/max normalized actions
        port_specs = []  # (global_port_idx, cs, port_local, min_norm, max_norm, max_power_kw)
        gidx = 0
        for cs in env.charging_stations:
            cs_max_power_kw = float(cs.get_max_power())
            cs_min_norm = float(cs.min_charge_current / cs.max_charge_current) if getattr(cs, 'max_charge_current', 0) else 0.0
            for p in range(cs.n_ports):
                if cs.evs_connected[p] is None:
                    gidx += 1
                    continue
                ev = cs.evs_connected[p]
                ev_min_kw = float(getattr(ev, 'min_ac_charge_power', 0.0))
                ev_max_kw = float(getattr(ev, 'max_ac_charge_power', cs_max_power_kw))
                min_kw = max(0.0, min(cs.get_max_power(), ev_min_kw))
                max_kw = max(0.0, min(cs.get_max_power(), ev_max_kw))
                # Normalize using charger maximum; avoid division by zero
                denom = cs_max_power_kw if cs_max_power_kw > 1e-6 else 1.0
                min_norm = min(1.0, max(0.0, min_kw / denom))
                max_norm = min(1.0, max(0.0, max_kw / denom))
                port_specs.append((gidx, cs, p, min_norm, max_norm, max_kw))
                gidx += 1

        actions = np.zeros(env.number_of_ports, dtype=float)
        if not port_specs:
            return actions

        # V2G discharging path: allocate negative actions to meet negative target
        # or for aggressive V2G when there's net positive usage
        if target_kw < 0 or (aggressive_v2g and net_positive_usage):
            if not env.config.get('v2g_enabled', False):
                # V2G disabled: fall back to minimal/no charging behavior
                if self.verbose and t % 500 == 0:
                    print(f"[LedgerAware] t={t} target_kw={target_kw:.3f} < 0 but v2g disabled; no discharging")
                return actions  # keep zeros

            # Compute per-port max discharge capability (kW)
            discharge_specs = []  # (gidx, cs, p, cs_max_dis_kw, port_max_dis_kw)
            total_dis_cap = 0.0
            gidx2 = 0
            for cs in env.charging_stations:
                # Charger max discharge in kW
                try:
                    # Take absolute value of max_discharge_current since it's negative in the model
                    cs_max_dis_kw = float(math.sqrt(getattr(cs, 'phases', 1)) * getattr(cs, 'voltage', 230.0) * abs(getattr(cs, 'max_discharge_current', 0.0)) / 1000.0)
                except Exception:
                    cs_max_dis_kw = 0.0
                for p in range(cs.n_ports):
                    if cs.evs_connected[p] is None:
                        gidx2 += 1
                        continue
                    ev = cs.evs_connected[p]
                    
                    # Skip if EV is at or below emergency reserve plus buffer
                    try:
                        reserve_kwh = max(float(getattr(ev, 'min_emergency_battery_capacity', 0.0)),
                                          float(getattr(ev, 'min_battery_capacity', 0.0)))
                        current_kwh = float(getattr(ev, 'current_capacity', 0.0))
                        # Use the configurable buffer for aggressive V2G
                        if current_kwh <= reserve_kwh + self.aggressive_v2g_buffer:
                            gidx2 += 1
                            continue
                    except Exception:
                        pass
                        
                    # EV discharge power is negative in the model, so we need to take the absolute value
                    ev_max_dis_kw = abs(float(getattr(ev, 'max_discharge_power', 0.0)))
                    port_max_dis_kw = max(0.0, min(cs_max_dis_kw, ev_max_dis_kw))
                    if port_max_dis_kw > 0:
                        discharge_specs.append((gidx2, cs, p, cs_max_dis_kw, port_max_dis_kw))
                        total_dis_cap += port_max_dis_kw
                    gidx2 += 1

            if total_dis_cap <= 1e-6:
                if self.verbose:
                    print(f"[LedgerAware] t={t} V2G enabled but no discharge capability detected")
                    # Debug why no discharge capability
                    for cs in env.charging_stations:
                        try:
                            # Take absolute value for calculation and display
                            max_discharge_current = getattr(cs, 'max_discharge_current', 0.0)
                            cs_max_dis_kw = float(math.sqrt(getattr(cs, 'phases', 1)) * getattr(cs, 'voltage', 230.0) * abs(max_discharge_current) / 1000.0)
                            print(f"  CS {cs.id}: max_discharge_current={max_discharge_current}, phases={getattr(cs, 'phases', 1)}, voltage={getattr(cs, 'voltage', 230.0)}, cs_max_dis_kw={cs_max_dis_kw}")
                            
                            for p in range(cs.n_ports):
                                if cs.evs_connected[p] is not None:
                                    ev = cs.evs_connected[p]
                                    reserve_kwh = max(float(getattr(ev, 'min_emergency_battery_capacity', 0.0)), float(getattr(ev, 'min_battery_capacity', 0.0)))
                                    current_kwh = float(getattr(ev, 'current_capacity', 0.0))
                                    ev_max_dis_kw = abs(float(getattr(ev, 'max_discharge_power', 0.0)))
                                    print(f"    Port {p}: EV connected={ev is not None}, current_capacity={current_kwh:.1f}, reserve={reserve_kwh:.1f}, buffer={self.aggressive_v2g_buffer}, max_discharge_power={getattr(ev, 'max_discharge_power', 0.0)}")
                        except Exception as e:
                            print(f"  Error inspecting CS {cs.id}: {e}")
                return actions

            # For aggressive V2G, use net_usage_kw as the discharge target if target_kw >= 0
            if target_kw >= 0 and aggressive_v2g and net_positive_usage:
                desired_dis_kw = min(net_usage_kw, total_dis_cap)
                if self.verbose and t % 100 == 0:
                    print(f"[LedgerAware] t={t} Aggressive V2G: discharging up to {desired_dis_kw:.3f}kW to offset load")
            else:
                desired_dis_kw = min(abs(target_kw), total_dis_cap)
            # Allocate proportionally by each port's capacity
            for (gidxp, cs, p, cs_max_dis_kw, port_max_dis_kw) in discharge_specs:
                share = port_max_dis_kw / total_dis_cap if total_dis_cap > 0 else 0.0
                dis_kw = desired_dis_kw * share
                denom = cs_max_dis_kw if cs_max_dis_kw > 1e-6 else 1.0
                act_norm = - float(np.clip(dis_kw / denom, 0.0, 1.0))
                actions[gidxp] = act_norm

            if self.verbose and t % 100 == 0:
                nz = np.count_nonzero(actions < 0)
                if target_kw < 0:
                    print(f"[LedgerAware] t={t} V2G discharge: target={target_kw:.3f}kW, ports_neg={nz}")
                else:
                    print(f"[LedgerAware] t={t} Aggressive V2G discharge: net_usage={net_usage_kw:.3f}kW, ports_neg={nz}")
            return actions

        # Greedy allocation: start by assigning minimums, then distribute residual up to max
        # Compute how much power minimums would consume
        base_kw = sum(cs.get_max_power() * mn for (_, cs, _, mn, _, _) in port_specs)
        residual_kw = target_kw - base_kw

        # If even the minimum charging exceeds the target and V2G is enabled,
        # allocate discharging to reduce grid draw down to the target without exporting.
        if residual_kw < 0 and env.config.get('v2g_enabled', False):
            need_discharge_kw = min(abs(residual_kw), abs(float(env.config.get('setpoints', {}).get('generation', {}).get('export_limit_kw_per_site', float('inf')))))
            # Build discharge capabilities per port
            discharge_specs = []  # (gidx, cs, p, cs_max_dis_kw, port_max_dis_kw)
            total_dis_cap = 0.0
            gidx3 = 0
            for cs in env.charging_stations:
                try:
                    # Take absolute value of max_discharge_current since it's negative in the model
                    cs_max_dis_kw = float(math.sqrt(getattr(cs, 'phases', 1)) * getattr(cs, 'voltage', 230.0) * abs(getattr(cs, 'max_discharge_current', 0.0)) / 1000.0)
                except Exception:
                    cs_max_dis_kw = 0.0
                for p in range(cs.n_ports):
                    if cs.evs_connected[p] is None:
                        gidx3 += 1
                        continue
                    ev = cs.evs_connected[p]
                    
                    # Skip if EV is at or below emergency reserve plus buffer
                    try:
                        reserve_kwh = max(float(getattr(ev, 'min_emergency_battery_capacity', 0.0)),
                                          float(getattr(ev, 'min_battery_capacity', 0.0)))
                        current_kwh = float(getattr(ev, 'current_capacity', 0.0))
                        # Use the configurable buffer for aggressive V2G
                        if current_kwh <= reserve_kwh + self.aggressive_v2g_buffer:
                            gidx3 += 1
                            continue
                    except Exception:
                        pass
                        
                    # EV discharge power is negative in the model, so we need to take the absolute value
                    ev_max_dis_kw = abs(float(getattr(ev, 'max_discharge_power', 0.0)))
                    port_max_dis_kw = max(0.0, min(cs_max_dis_kw, ev_max_dis_kw))
                    if port_max_dis_kw > 0:
                        discharge_specs.append((gidx3, cs, p, cs_max_dis_kw, port_max_dis_kw))
                        total_dis_cap += port_max_dis_kw
                    gidx3 += 1

            if total_dis_cap > 1e-6:
                to_allocate = min(need_discharge_kw, total_dis_cap)
                for (gidxp, cs, p, cs_max_dis_kw, port_max_dis_kw) in discharge_specs:
                    share = port_max_dis_kw / total_dis_cap
                    dis_kw = to_allocate * share
                    denom = cs_max_dis_kw if cs_max_dis_kw > 1e-6 else 1.0
                    act_norm = - float(np.clip(dis_kw / denom, 0.0, 1.0))
                    # Combine with minimum charging: convert min_norm to positive action, then add negative discharge
                    # Start from min_norm action for charging
                    # Compute min_norm for this port to preserve minimum if desired
                    try:
                        # find corresponding min_norm from port_specs by matching (cs,p)
                        for (gidx_ps, cs_ps, p_ps, min_norm_ps, max_norm_ps, _) in port_specs:
                            if cs_ps is cs and p_ps == p:
                                actions[gidx_ps] = float(np.clip(min_norm_ps + act_norm, -1.0, 1.0))
                                break
                    except Exception:
                        actions[gidxp] = act_norm
                if self.verbose and t % 100 == 0:
                    print(f"[LedgerAware] t={t} Mixed mode: min>{target_kw:.3f}kW, discharging {to_allocate:.3f}kW to meet target")
                return actions
            # If no discharge capacity, fall through to charging allocation with residual>=0 clamp
            residual_kw = max(0.0, residual_kw)
        else:
            residual_kw = max(0.0, residual_kw)

        # Sort ports by slack (max - min) descending for allocation fairness
        port_specs.sort(key=lambda x: (x[4] - x[3]) if isinstance(x[3], (int, float)) else 0.0, reverse=True)

        for (gidx, cs, p, min_norm, max_norm, max_kw) in port_specs:
            # Allocate additional up to the port's remaining capacity
            cs_max_kw = float(cs.get_max_power())
            denom = cs_max_kw if cs_max_kw > 1e-6 else 1.0
            min_kw = min_norm * cs_max_kw
            extra_cap_kw = max(0.0, (max_norm - min_norm) * cs_max_kw)
            alloc_kw = min(extra_cap_kw, residual_kw)
            act_norm = min_norm + (alloc_kw / denom)
            actions[gidx] = float(np.clip(act_norm, -1.0 if env.config.get('v2g_enabled', False) else 0.0, 1.0))
            residual_kw -= alloc_kw
            if residual_kw <= 1e-6:
                # Assign min to remaining
                break

        # Ensure all unspecified connected ports at least get their minimums
        for (gidx, cs, p, min_norm, max_norm, _) in port_specs:
            if actions[gidx] == 0.0 and min_norm > 0.0:
                actions[gidx] = float(min_norm)

        if self.verbose and t % 100 == 0:  # Print every 100 steps to avoid spam
            connected_evs = len(port_specs)
            print(f"[LedgerAware] t={t} target_kw={target_kw:.3f} connected_evs={connected_evs} actions_nonzero={np.count_nonzero(actions)}")
            if connected_evs > 0:
                print(f"  -> actions: {actions[actions != 0]}")

        return actions

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