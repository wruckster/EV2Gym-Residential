# EV2Gym Ledger Plan: Global + Account Ledgers

This document outlines the design and rollout plan for introducing in-memory ledgers to standardize observation building, logging, and exports.

## Goals

- Replace ad-hoc state gathering with consistent, typed, NumPy-backed ledgers.
- Provide global and per-account (residential household/charger) views.
- Make the observation builder read exclusively from ledgers.
- Enable reliable exports to Pandas/Polars/Parquet for analysis.

## Components

- `ev2gym/rl_agent/ledgers.py`
  - `ColumnSpec`: name + dtype.
  - `GlobalLedgerBuffers`: NumPy preallocated time series columns for global signals.
  - `AccountLedgerBuffers`: per-account time series columns.
  - Export helpers: `to_pandas()`, `to_polars()`, `to_parquet(path)`.

- `ev2gym/models/ev2gym_env.py`
  - New attributes: `self.global_buffers`, `self.account_buffers: dict[int, AccountLedgerBuffers]`.
  - `_init_ledgers()`: builds timestamps and schemas after topology + data load.
  - `_update_global_ledger_row()`: writes current-step global values.
  - `_update_account_ledger_row()`: writes current-step per-account values.
  - Writes happen in `step()` after `_update_power_statistics()` and before `current_step += 1`.

## Current Ledgers (Implemented)

- Global ledger columns
  - Meta/time: `timestamp`, `step`, `step_ratio`, `dow`, `hour`, `minute`
  - Forecasts H=24:
    - Price: `price_fc_h01..price_fc_h24`
    - Weather: `temp_fc_h01..temp_fc_h24`, `wind_fc_h01..wind_fc_h24`
    - Note: DR is planned as global/state-wide; columns currently omitted
  - Totals: `total_power_usage_kw`, `power_setpoint_kw`, `ev_power_kw`, `inflexible_load_kw`, `solar_production_kw`, `evs_parked`
  - Tracking/cost: `tracking_error`, `invalid_action_punishment`, `reward_step`, `reward_cumsum`
  - No per-CS fields in global (removed: `charge_price_cs{id}`, `discharge_price_cs{id}`, `cs{id}_power_kw`, `cs{id}_amps`)

- Account ledger columns (one account per CS for residential)
  - Meta: `timestamp`, `step`
  - Charger-level: `cs_power_kw`, `cs_amps`, `charge_price`, `discharge_price`, `evs_connected`, `cs_kw_limit`
  - Residential/account fields: `account_power_setpoint_kw`, `household_inflexible_load_kw`, `household_pv_kw`, `tracking_error_account`
  - Per-port: `port{p}_amps`, `port{p}_soc`, `port{p}_action_norm`, `port{p}_amps_limit`, `port{p}_connected`, `port{p}_time_to_departure`, `port{p}_time_since_arrival`, `port{p}_action`, `port{p}_is_charging`, `port{p}_is_discharging`
  - Per-account forecasts H=24: `load_fc_h01..load_fc_h24`, `pv_fc_h01..pv_fc_h24`
  - No mirroring from global (removed: `global_power_setpoint_kw`)

Dtypes:
- float32 for continuous values; int16 for small counters; int32 for step; bool/int8 for flags as needed.

Indexing:
- Store `timestamp` as a dedicated column (`datetime64[ns]`). No reliance on DataFrame index.

## Remaining TODOs / Next Pass

* __Global DR forecast__: Add state-wide DR H=24 columns when source is finalized.
* __Observation builder__: Optionally include per-account `load_fc_hxx` and `pv_fc_hxx` in the default observation for the active account.
* __Topology mapping (account)__: consider adding constant `transformer_id` per account.
* __Parity smokes (optional)__: If/when globals are defined as sum of accounts, validate sums (household load/PV).

## Observation Builder (Planned)

- `ev2gym/rl_agent/state.py`
  - `build_observation(env, account_id: int, step: int) -> np.ndarray`
  - Concatenates global + account rows (no overlapping fields), with forecast slices already populated in ledgers.
  - Optionally include per-account H=24 load/PV forecast vectors.

## Reset Integration

- Re-run `_init_ledgers()` in `reset()` after `self.sim_starting_date` is set to realign timestamps.

## Export API (Planned)

- `env.save_ledgers_parquet(dir_path)`
  - Writes `global.parquet` and `account_{id}.parquet` per account.
  - Optional in-memory `to_pandas()` / `to_polars()` accessors for analysis.

## Acceptance Criteria

- A consistent observation builder using only ledger data.
- Global and per-account ledgers populated at every step with aligned timestamps.
- Forecast horizon H=24 included for prices (and optionally load/PV).
- Minimal overhead in the step loop (NumPy writes only).
- Parquet exports loadable by Pandas/Polars.

## Implementation Notes

- Attribute correctness in writers:
  - Use `cs.evs_connected[p]` (not `cs.ports[p].ev`).
  - Use `ev.get_soc()` or `ev.current_capacity` (no `energy_level` attribute).
- Bounds: write at `t = current_step` and avoid out-of-bounds by respecting `simulation_length`.
- Initialization order: `_init_ledgers()` runs after chargers, transformers, and prices are loaded.

## Open Questions

- Account mapping: one account per charging station is assumed for residential. If different, supply a mapping from chargers/ports to account IDs.
- Which forecasts to include beyond price (load, PV) in the immediate next pass?
