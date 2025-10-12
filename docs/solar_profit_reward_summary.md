# Solar Profit Reward Overview

This document summarizes how the solar profit reward family is implemented in `ev2gym/rl_agent/reward.py`.

## Parameter Dataclass: `SolarProfitRewardParams`
- **Weights** control component importance: `w_profit`, `w_soc`, `w_wear`, `w_solar`, `w_setpoint`.
- **Additional scalars**: `alpha`, `solar_norm_kwh`, `wear_cost_per_kwh`, `grid_import_penalty_per_kwh`, `invalid_action_penalty`.
- **Normalization constants** (auto-computed): `max_soc_deficit_kwh`, `max_throughput_kwh`, `typical_profit_range`, `max_setpoint_error_kw`.
- `normalize_components` toggles normalized combination (always `True` after `from_env()`).

## Loading Parameters: `SolarProfitRewardParams.from_env()`
1. Reads `env.config['reward']` to detect a preset (`preset` / `preset_name`).
2. Applies overrides from preset `SolarProfitRewardParams` stored in `SOLAR_REWARD_PRESETS`.
3. Applies explicit overrides from `reward.weights` and direct keys (with compatibility for `soc_deficit_cost_per_kwh`).
4. Computes normalization constants using EV and forecasting settings (battery capacity, max AC charge power, forecast horizon, median charge price).
5. Returns a fully populated, normalized parameter set.

## Preset Definitions (`SOLAR_REWARD_PRESETS`)
- **`profit_first`**: Prioritizes export profit, light setpoint tracking.
- **`balanced`**: Weight 1.0 on profit/solar, heavier SoC and wear penalties, strong tracking (`w_setpoint=0.8`).
- **`renewable_first`**: High solar weight with moderate tracking, lighter wear cost.
- **`user_priority`**: Heavy SoC penalty to protect users, moderate setpoint tracking.

Presets can be selected via `reward.preset` in the YAML config or overridden per-weight via `reward.weights`.

## Reward Composition (`solar_profit_reward()`)
1. **Extract environment signals** from `energy_flow_breakdown` (grid, EV, solar, inflexible load) and ledger-derived arrays.
2. **Compute base quantities**:
   - Energy flows per timestep (`grid_energy_kwh`, `ev_energy_kwh`, `solar_gen_kwh`, `load_energy_kwh`).
   - Import/export split (`e_buy`, `e_sell`).
   - Prices (`charge_price`, `discharge_price`).
3. **Profit term**: `(discharge_price * e_sell) - (charge_price * e_buy)` with fallback to `total_costs` when needed.
4. **Solar self-consumption bonus**: scaled by `solar_used_kwh` and `w_solar`.
5. **Penalty terms**:
   - SoC deficit via `_soc_deficit_kwh()` and `w_soc`.
   - Battery wear via `_battery_throughput_kwh()` and `w_wear`.
   - Grid import penalty and invalid action penalty (if configured).
6. **Setpoint tracking component**:
   - Uses account ledgers (`account_power_setpoint_kw`, `account_actual_power_kw`) when account-online mode is enabled.
   - Falls back to global ledgers (`power_setpoint_kw`, `total_power_usage_kw`) or legacy arrays.
   - Applies average absolute tracking error scaled by `w_setpoint`.
7. **Normalization**: Profit, SoC deficit, wear, solar bonus, and tracking error are normalized using precomputed constants to keep components roughly within [-1, 1].
8. **User satisfaction penalties**: deducts from reward when `user_satisfaction_list` contains sub-1.0 scores.
9. **Final combination**: Weighted sum of normalized components, with protections against non-finite values.

## Data Sources and Fallbacks
- Prefers ledger-backed arrays for consistency with account-online mode.
- Maintains compatibility with legacy arrays (`power_setpoints`, `current_power_usage`) if ledgers are unavailable.

## Usage Notes
- Configure presets/weights in `ev2gym/example_config_files/residential_v2g.yaml` under `reward`.
- When account-online mode is enabled via config (`setpoints.account_online.enabled`), the reward tracks setpoint error per account.
- The reward integrates seamlessly with `train_tianshou.py`, which only supplies the callable; all parameterization flows from the environment config.
