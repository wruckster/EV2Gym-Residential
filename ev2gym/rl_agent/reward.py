'''This file contains various example reward functions for the RL agent. Users can create their own reward function here or in their own file using the same structure as below
'''

import math
import warnings

def SquaredTrackingErrorReward(env,*args):
    '''This reward function is the squared tracking error that uses the minimum of the power setpoints and the charge power potential
    The reward is negative'''
    
    # Use ledger data instead of legacy arrays
    if hasattr(env, 'global_buffers') and env.global_buffers is not None:
        t = int(max(0, min(env.current_step, env.global_buffers.T - 1)))
        prev_t = max(0, t - 1)
        
        # Get power setpoint from global ledger
        setpoint_arr = env.global_buffers._data.get('power_setpoint_kw')
        power_setpoint = float(setpoint_arr[prev_t]) if setpoint_arr is not None else 0.0
        
        # Get EV power as proxy for charge power potential
        ev_power_arr = env.global_buffers._data.get('ev_power_kw')
        charge_potential = float(ev_power_arr[prev_t]) if ev_power_arr is not None else 0.0
        
        # Get total power usage
        power_arr = env.global_buffers._data.get('total_power_usage_kw')
        current_power = float(power_arr[prev_t]) if power_arr is not None else 0.0
        
        reward = - (min(power_setpoint, charge_potential) - current_power)**2
    else:
        # Fallback to legacy arrays
        reward = - (min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1]) -
            env.current_power_usage[env.current_step-1])**2
        
    return reward

def SqTrError_TrPenalty_UserIncentives(env, _, user_satisfaction_list, *args):
    ''' This reward function is the squared tracking error that uses the minimum of the power setpoints and the charge power potential
    It penalizes transofrmers that are overloaded    
    The reward is negative'''
    
    # Use ledger data instead of legacy arrays
    if hasattr(env, 'global_buffers') and env.global_buffers is not None:
        t = int(max(0, min(env.current_step, env.global_buffers.T - 1)))
        prev_t = max(0, t - 1)
        
        # Get power setpoint from global ledger
        setpoint_arr = env.global_buffers._data.get('power_setpoint_kw')
        power_setpoint = float(setpoint_arr[prev_t]) if setpoint_arr is not None else 0.0
        
        # Get EV power as proxy for charge power potential
        ev_power_arr = env.global_buffers._data.get('ev_power_kw')
        charge_potential = float(ev_power_arr[prev_t]) if ev_power_arr is not None else 0.0
        
        # Get total power usage
        power_arr = env.global_buffers._data.get('total_power_usage_kw')
        current_power = float(power_arr[prev_t]) if power_arr is not None else 0.0
        
        # Get transformer max limit (fallback to transformer object)
        tr_max_limit = env.transformers[0].max_power[env.current_step-1]
        
        reward = - (min(power_setpoint, charge_potential, tr_max_limit) - current_power)**2
    else:
        # Fallback to legacy arrays
        tr_max_limit = env.transformers[0].max_power[env.current_step-1]
        
        reward = - (min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1],tr_max_limit) -
            env.current_power_usage[env.current_step-1])**2
            
    for tr in env.transformers:
        reward -= 100 * tr.get_how_overloaded()
        
    for score in user_satisfaction_list:
        reward -= 1000 * (1 - score)
                    
    return reward

def ProfitMax_TrPenalty_UserIncentives(env, total_costs, user_satisfaction_list, *args):
    
    reward = total_costs
    
    for tr in env.transformers:
        reward -= 100 * tr.get_how_overloaded()                        
    
    for score in user_satisfaction_list:        
        reward -= 100 * math.exp(-10*score)
        
    return reward

def SquaredTrackingErrorRewardWithPenalty(env,*args):
    ''' This reward function is the squared tracking error that uses the minimum of the power setpoints and the charge power potential
    The reward is negative
    If the EV is not charging, the reward is penalized
    '''
    # Use ledger data instead of legacy arrays
    if hasattr(env, 'global_buffers') and env.global_buffers is not None:
        t = int(max(0, min(env.current_step, env.global_buffers.T - 1)))
        prev_t = max(0, t - 1)
        prev_prev_t = max(0, t - 2)
        
        # Get power values from ledger
        power_arr = env.global_buffers._data.get('total_power_usage_kw')
        current_power = float(power_arr[prev_t]) if power_arr is not None else 0.0
        
        ev_power_arr = env.global_buffers._data.get('ev_power_kw')
        charge_potential_prev = float(ev_power_arr[prev_prev_t]) if ev_power_arr is not None else 0.0
        charge_potential = float(ev_power_arr[prev_t]) if ev_power_arr is not None else 0.0
        
        setpoint_arr = env.global_buffers._data.get('power_setpoint_kw')
        power_setpoint = float(setpoint_arr[prev_t]) if setpoint_arr is not None else 0.0
        
        if current_power == 0 and charge_potential_prev != 0:
            reward = - (min(power_setpoint, charge_potential) - current_power)**2 - 1000
        else:
            reward = - (min(power_setpoint, charge_potential) - current_power)**2
    else:
        # Fallback to legacy arrays
        if env.current_power_usage[env.current_step-1] == 0 and env.charge_power_potential[env.current_step-2] != 0:
            reward = - (min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1]) -
            env.current_power_usage[env.current_step-1])**2 - 1000
        else:
            reward = - (min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1]) -
            env.current_power_usage[env.current_step-1])**2
    
    return reward

def SimpleReward(env,*args):
    '''This reward function does not consider the charge power potential'''
    
    # Use ledger data instead of legacy arrays
    if hasattr(env, 'global_buffers') and env.global_buffers is not None:
        t = int(max(0, min(env.current_step, env.global_buffers.T - 1)))
        prev_t = max(0, t - 1)
        
        setpoint_arr = env.global_buffers._data.get('power_setpoint_kw')
        power_setpoint = float(setpoint_arr[prev_t]) if setpoint_arr is not None else 0.0
        
        power_arr = env.global_buffers._data.get('total_power_usage_kw')
        current_power = float(power_arr[prev_t]) if power_arr is not None else 0.0
        
        reward = - (power_setpoint - current_power)**2
    else:
        # Fallback to legacy arrays
        reward = - (env.power_setpoints[env.current_step-1] - env.current_power_usage[env.current_step-1])**2
    
    return reward

def MinimizeTrackerSurplusWithChargeRewards(env,*args):
    ''' This reward function minimizes the tracker surplus and gives a reward for charging '''
    
    # Use ledger data instead of legacy arrays
    if hasattr(env, 'global_buffers') and env.global_buffers is not None:
        t = int(max(0, min(env.current_step, env.global_buffers.T - 1)))
        prev_t = max(0, t - 1)
        
        setpoint_arr = env.global_buffers._data.get('power_setpoint_kw')
        power_setpoint = float(setpoint_arr[prev_t]) if setpoint_arr is not None else 0.0
        
        power_arr = env.global_buffers._data.get('total_power_usage_kw')
        current_power = float(power_arr[prev_t]) if power_arr is not None else 0.0
        
        reward = 0
        if power_setpoint < current_power:
            reward -= (current_power - power_setpoint)**2
        
        reward += current_power #/75
    else:
        # Fallback to legacy arrays
        reward = 0
        if env.power_setpoints[env.current_step-1] < env.current_power_usage[env.current_step-1]:
                reward -= (env.current_power_usage[env.current_step-1]-env.power_setpoints[env.current_step-1])**2

        reward += env.current_power_usage[env.current_step-1] #/75
    
    return reward

def profit_maximization(env, total_costs, user_satisfaction_list, *args):
    ''' This reward function is used for the profit maximization case '''
    # Prefer net grid cost at the meter as the objective signal.
    # env.cost_history includes: grid import cost + charging cost - discharge credit.
    # We maximize profit by minimizing cost: reward = - net_cost.
    try:
        idx = max(0, env.current_step - 1)
        net_cost = float(env.cost_history[idx])
        reward = -net_cost
    except Exception:
        # Fallback to legacy total_costs if cost_history is unavailable
        reward = float(total_costs)

    # Per-step penalty when EV is below its emergency SoC threshold
    try:
        coeff = float(getattr(env, 'emergency_soc_penalty_kwh', 2.0))  # reward units per kWh deficit
        below_penalty = 0.0
        for cs in getattr(env, 'charging_stations', []):
            for ev in getattr(cs, 'evs_connected', []):
                if ev is None:
                    continue
                deficit = ev.min_emergency_battery_capacity - ev.current_capacity
                if deficit > 1e-9:
                    below_penalty += deficit * coeff
        reward -= below_penalty
    except Exception:
        # Be robust: ignore penalty calculation errors
        pass

    for score in user_satisfaction_list:
        # reward -= 100 * (1 - score)
        reward -= 100 * math.exp(-10*score)

    # Numerical safety: ensure finite reward
    if not math.isfinite(reward):
        warnings.warn("Non-finite reward detected in profit_maximization; replacing with 0.0")
        reward = 0.0
    return reward



# Previous reward functions for testing
#############################################################################################################
        # reward = total_costs  # - 0.5
        # print(f'total_costs: {total_costs}')
        # print(f'user_satisfaction_list: {user_satisfaction_list}')
        # for score in user_satisfaction_list:
        #     reward -= 100 * (1 - score)

        # Punish invalid actions (actions that try to charge or discharge when there is no EV connected)
        # reward -= 2 * (invalid_action_punishment/self.number_of_ports)

        # reward = min(2, 1 * 4 * self.cs / (0.00001 + (
        #     self.power_setpoints[self.current_step-1] - self.current_power_usage[self.current_step-1])**2))

        # this is the new reward function
        # reward = min(2, 1/((min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #           self.current_power_usage[self.current_step-1])**2 + 0.000001))

        # new_*10*charging
        # if self.power_setpoints[self.current_step-1] < self.current_power_usage[self.current_step-1]:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #             self.current_power_usage[self.current_step-1])
        # else:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #             self.current_power_usage[self.current_step-1])*10

        # new_1_equal
        # if self.power_setpoints[self.current_step-1] < self.current_power_usage[self.current_step-1]:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #             self.current_power_usage[self.current_step-1])
        # else:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #             self.current_power_usage[self.current_step-1])

        # new_0.1
        # if self.power_setpoints[self.current_step-1] < self.current_power_usage[self.current_step-1]:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #             self.current_power_usage[self.current_step-1])**2
        # else:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #             self.current_power_usage[self.current_step-1])*0.1

        # new_reward squared
        # if self.power_setpoints[self.current_step-1] < self.current_power_usage[self.current_step-1]:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #           self.current_power_usage[self.current_step-1])**2
        # else:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #           self.current_power_usage[self.current_step-1])

        # for score in user_satisfaction_list:
        #     reward -= 100 * (1 - score)

        # for tr in self.transformers:
        #     if tr.current_amps > tr.max_current:
        #         reward -= 1000 * abs(tr.current_amps - tr.max_current)
        #     elif tr.current_amps < tr.min_current:
        #         reward -= 1000 * abs(tr.current_amps - tr.min_current)

        # reward -= 100 * (tr.current_amps < tr.min_amps)
        #######################################################################################################
        # squared tracking error
        # reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #            self.current_power_usage[self.current_step-1])**2

        # best reward so far
        ############################################################################################################
        # if self.power_setpoints[self.current_step-1] < self.current_power_usage[self.current_step-1]:
        #     reward -= (self.current_power_usage[self.current_step-1]-self.power_setpoints[self.current_step-1])

        # reward += self.current_power_usage[self.current_step-1]/75
        ############################################################################################################
        # normalize reward to -1 1
        # reward = reward/1000
        # reward = (100 +reward) / 1000
        # print(f'reward: {reward}')

        # reward -= 2 * (invalid_action_punishment/self.number_of_ports)
        # reward /= 100
        # reward = (100 +reward) / 1000
        # print(f'current_power_usage: {self.current_power_usage[self.current_step-1]}')

        # return reward