# Residential V2G Power Setpoint Integration Plan

This plan describes how to integrate a practical residential V2G setpoint policy into EV2Gym without breaking existing examples or training scripts.

- Scope files:
  - `ev2gym/utilities/utils.py` (function: `generate_power_setpoints(env)`)
  - `ev2gym/utilities/loaders.py` (price/load/PV loaders)
  - `ev2gym/models/ev2gym_env.py` (initialization, debug summaries)
  - `ev2gym/example_config_files/residential_v2g.yaml` (reuse existing config; add only minimal new flags)
  - Keep backward compatibility for `example.py` and `train_stable_baselines.py`.

---

## Objectives
- Minimize cost while meeting departure SoC targets.
- Respect per-port/CS/transformer capacity and residential export limits.
- Prefer charging in price valleys and PV surplus; discharge in evening peaks.
- Constrain cycling: reserve SoC, throughput limits, smoothing and ramping.
 - Recompute setpoints every timestep and record them for plotting/replay.
 - Residential PoV: shape EV setpoints relative to household net demand (demand − PV).

---

## Step 1 — Reuse Existing Config, Add Minimal Gate (default-off)
Prefer existing parameters in `ev2gym/example_config_files/residential_v2g.yaml`. Add only a tiny gate and optional tuning fields. If the block is missing or disabled, behavior is unchanged.

```yaml
# Residential V2G setpoint policy (default: disabled)
res_v2g_setpoints:
  enabled: false              # Gate to avoid breaking existing runs
  # Optional tuning (all optional):
  export_limit_kw_per_site: 5.0   # If omitted, derive from `transformer.max_power` and station limits
  ramp_kw_per_step: 1.0           # If omitted, derive from EV/CS power × a small fraction
  smoothing_window_steps: 4       # If omitted, reuse existing median_smoothing window logic
  # Weighting automatically reuses available signals (prices/PV/net load) if present
```

Reused fields from existing YAML (no changes required):
- Target departure SoC: `ev.desired_capacity` (0–1)
- Reserve energy: `ev.min_battery_capacity` (kWh) or `ev.min_emergency_battery_capacity` → convert to SoC with `ev.battery_capacity`
- EV power limits: `ev.max_ac_charge_power`, `ev.min_ac_charge_power`, `ev.max_discharge_power`, `ev.min_discharge_power`
- CS electrical limits: `charging_station.(min|max)_(dis)charge_current`, `charging_station.voltage`, `charging_station.phases`
- Transformer cap: `transformer.max_power`
- Prices: `discharge_price_factor` together with loaded price arrays
- Forecast control: `forecasting.enabled` and dataset at `data_path`
- Scenario/spawn/time controls as already defined

---

## Step 2 — Access Forecasts/Signals
Reuse existing plumbing:
- If `forecasting.enabled: true`, use forecast arrays already produced (prices, household_demand, solar_production) from `data_path`.
- Prices: `env.charge_prices[0]` and `env.discharge_prices[0]` (scaled by `discharge_price_factor` where applicable).
- PV/load signals: use available forecasts; if missing, derive proxies from transformer traces during planning.
- If any signal is unavailable, default gracefully (e.g., weights based on prices only).

Implementation notes:
- Ensure `ev2gym/models/ev2gym_env.py` exposes or allows access to arrays already created by the forecast path. Otherwise, derive simple proxies from existing runtime data.
 - Define per-step household net demand: `net_demand[t] = household_demand[t] - solar_production[t]` (0 if missing). This drives a residential “track-net” term.

---

## Step 3 — Extend `generate_power_setpoints(env)`
Implement the residential policy behind the config gate:

1) Early exit (preserve legacy):
```python
cfg = env.config.get('res_v2g_setpoints', {})
if not cfg or not cfg.get('enabled', False):
    # existing price-only allocation path (current behavior)
    return existing_output
```

2) Per-step recalculation and per-EV planning window:
- For each `ev` in `env.EVs_profiles`:
  - Window `t in [arrival+1, departure-1]` (respect indexing used elsewhere).
  - Required energy to reach target SoC using existing config:
    - Target departure SoC from `ev.desired_capacity`
    - Reserve SoC from `ev.min_battery_capacity / ev.battery_capacity` (or `min_emergency_battery_capacity` as stricter bound)
 - At every environment step, recompute the plan for remaining window using current forecasts/state (rolling horizon), then take the first-step setpoint. Record it.

3) Weights and residential track-net per timestep `t`:
- Price-based: prefer charging in low `env.charge_prices[0]`; prefer discharging when `env.discharge_prices[0]` (with `discharge_price_factor`) is high.
- PV-based: if `solar_production` forecast exists, favor charging with higher PV; if absent, weight=0.
- Net load: if `household_demand` exists, favor discharging when net load is high; if absent, weight=0.
- Normalize and blend with simple defaults; avoid adding new coefficients unless needed. Reasonable default: price drives most behavior; PV/net-load only nudge.
 - Residential “track-net” term: target EV setpoint to offset a fraction of net demand, e.g., `p_track[t] = -k * net_demand[t]`, with `0 < k ≤ 1`, bounded by EV/CS/transformer limits and SoC feasibility. Default `k=0.5` (implicit, no new config unless needed).

4) Allocate charge first:
- Distribute positive energy to meet required charge, respecting per-step power caps:
  - EV/CS limits: from `ev.max_ac_charge_power` and CS electrical envelope (`current × voltage × phases`).
  - Transformer headroom via `transformer.max_power`.
- Use greedy or proportional fill by descending charge-weight.

5) Allow limited discharge (V2G):
- Permit negative setpoints in evening peak windows if:
  - SoC above reserve derived from `min_(emergency_)battery_capacity`.
  - Expected remaining energy is still sufficient to reach `ev.desired_capacity` at departure.
  - Export not exceeding site/export caps: use optional `export_limit_kw_per_site` if set; else cap by `transformer.max_power` and CS limits.
 - Cap by EV/CS discharge limits from existing EV/CS specs.

6) Aggregate and clip:
- Sum per-step across EVs; clip to transformer headroom/export caps.
- If clipped, re-distribute proportionally to retain relative shapes.

7) Ramps and smoothing:
- If `ramp_kw_per_step` present in the gate, enforce it. Otherwise, derive a conservative ramp from a fraction of `ev.max_ac_charge_power`.
- Apply median/boxcar smoothing. If `smoothing_window_steps` absent, reuse existing `median_smoothing` behavior in `utils.py`.

8) Safety checks:
- Verify each EV can still meet departure target. If infeasible, reduce discharge and/or increase night charging.
- Consider `demand_response` events: during capacity reduction windows, bias against discharging and/or favor pre-charging beforehand using the same weighting scheme.

---

## Per-step feeder-level setpoint logic (what to compute each t)

This summarizes the per-timestep logic that `ev2gym/utilities/utils.py::generate_power_setpoints(env)` should implement when the gate is enabled. Importantly, `power_setpoints[t]` is a feeder-level target used by the reward; the policy still outputs per-port actions.

- __[net demand]__ Define household/feeder net demand: `net_t = Load_t − PV_t` (kW; positive = grid import needed). When forecasts exist, use them causally with simple linear extrapolation; otherwise fall back to runtime transformer traces.
- __[feeder target]__ Choose a feeder target `P_set_t` (e.g., track-net or zero-import). A common choice is `P_set_t = 0` to aim for net self-sufficiency, or a smoothed baseline if desired.
- __[implied EV target]__ Convert feeder target to EV aggregate target: `target_ev_power_t = P_set_t − (Load_t − PV_t) = P_set_t − net_t`. Positive means charge; negative means discharge.
- __[feasibility caps]__ Clip `target_ev_power_t` by:
  - Aggregate power limits available at t: sum of per-EV/per-port limits, charging-station envelopes, and transformer headroom/export rules.
  - Aggregate SOC headroom/tailroom over EVs connected at t, converted to kW via `timestep_hours = env.timescale/60`.
- __[no-EV case]__ If no EVs connected at t, set `P_set_t = net_t` (so the target equals actual uncontrollable feeder usage) to avoid unreachable tracking penalties.
- __[ramp & smoothing]__ Enforce an optional ramp limit per step and apply median/box smoothing (reuse existing `median_smoothing` behavior). Keep causality.

This keeps setpoints feasible and aligned with the feeder view the reward uses, while leaving the final per-port allocation to the RL policy (or a baseline dispatcher when needed).

---

## Adapting setpoints by objective

Yes, different objectives can and should lightly adjust how `P_set_t` is formed (the core mechanics above remain the same):

- __[Profit maximization]__
  - Bias `P_set_t` to charge in expected price valleys and discharge in peaks (use available `env.charge_prices`/`env.discharge_prices`).
  - Keep the SOC feasibility and departure targets hard; only the timing of energy shifts changes with price weighting.

- __[Self-sufficiency (net-zero import)]__
  - Choose `P_set_t ≈ 0` (possibly a small import allowance to respect export caps), effectively setting `target_ev_power_t ≈ −net_t` before clipping.
  - Emphasize midday charging (PV surplus) and evening discharge, bounded by SOC reserve.

- __[Minimize emissions]__
  - Weight `P_set_t` by carbon intensity forecasts if available (or proxies): prefer charging in low-intensity periods and avoid discharging when the grid is clean.
  - If intensity is time-varying, this acts similarly to price weighting but with an emissions signal.

Implementation notes:

- Place objective-specific weights inside `generate_power_setpoints()` behind the same gate, keyed by a simple `objective` string in YAML, e.g. `res_v2g_setpoints.objective: [profit|self_sufficiency|emissions]` (default can be `self_sufficiency`).
- Data dependencies reuse existing loaders: prices from `load_electricity_prices`, PV/load from household/transformer sources; optional carbon-intensity can be added via the existing forecasting hooks without affecting legacy paths.
- Regardless of objective, always apply the feasibility clip and no-EV guard so the RL agent is not penalized for unreachable targets.

---

## Step 3.5 — Integrate Setpoints via RL State (observation)
This enables the agent to be setpoint-aware without changing the action space or trainer wiring. It complements Step 3 (policy generation) and is compatible with both `Rescale_RepairLayer` and `Rescale_RepairLayer_V2G` wrappers.

- File: `ev2gym/rl_agent/state.py`
- Function: `V2G_profit_max(env, *args)` (used by `train_config.yaml -> rl.state_function`)

What to add:
- Append the current and (optionally) a short horizon of future power setpoints to the observation vector.
- Keep per-port continuous action space unchanged (`Box(shape=(num_ports,))`).

Suggested implementation sketch:

```python
# Inside V2G_profit_max(env, *args)
H = 12  # default horizon; can be made configurable via YAML (see below)
cur = env.power_setpoints[env.current_step] if env.current_step < env.simulation_length else 0.0
win = env.power_setpoints[env.current_step: env.current_step + H]
if len(win) < H:
    import numpy as np
    win = np.append(win, np.zeros(H - len(win)))

# Optional normalization to improve conditioning (choose one):
# scale = max(1.0, float(getattr(env.transformers[0], 'max_power', 1.0)))
# cur /= scale; win = win / scale

state.append(cur)
state.append(win)
```

Optional YAML control for the horizon and scaling (default-safe and backward-compatible):

```yaml
# train_config.yaml
rl:
  state_horizon:
    setpoint_steps: 12     # default if missing
    normalize_by: "transformer_max_power"  # [none|transformer_max_power]
```

To access this inside `V2G_profit_max`, read `env.config` safely (fallback to defaults if missing):

```python
cfg = getattr(env, 'config', {})
steps = int(cfg.get('rl', {}).get('state_horizon', {}).get('setpoint_steps', 12))
norm = str(cfg.get('rl', {}).get('state_horizon', {}).get('normalize_by', 'none'))
```

Reward coupling (optional but recommended):
- Add a soft tracking penalty in `ev2gym/rl_agent/reward.py`, e.g., Huber or L2 on `(net_power - setpoint)`.
- Keep cost/profit terms; tune a weight to balance tracking vs economy.
- This leverages the new state features (setpoint/horizon) to make the penalty learnable.

Why this path:
- Treats the setpoint as an exogenous signal (part of the state), preserving a stationary MDP.
- No changes to `train_tianshou.py` are required; PPO continues to use a Gaussian policy over the same action space.
- Compatible with evaluation/replay/plots already implemented.

Notes:
- If strict positive-only adherence is desired, use `Rescale_RepairLayer` (enforces aggregate towards the setpoint); otherwise keep `Rescale_RepairLayer_V2G` to allow discharging with tracking guided by the reward.

---

## Step 4 — Minimal Data Interfaces
Avoid broad refactors by reusing existing structures:
- Prices: continue to use `env.charge_prices[0]`, `env.discharge_prices[0]`.
- PV/load: if forecasts missing, compute simple proxies from `transformers` at runtime (`tr.solar_power` and `tr.inflexible_load`).
- Export cap: if no per-site metadata, apply a single scalar cap across all CS under the same transformer. If `export_limit_kw_per_site` is missing, limit by `transformer.max_power` and CS limits.

---

## Step 5 — Debugging & Telemetry
When `debug_setpoints` is true, extend logs with single-line summary only:
- For the gated policy print:
  - `policy=res_v2g`, number of EVs, allocations, charge/discharge energy totals,
  - `nonzero`, `span=[first,last]`, number of export-clips and ramp-clips.
- Keep the existing `[DBG reset]` preview and `[DBG setpts]` summary.
 - Record per-step EV total setpoint series for plotting/replay. Ensure replay contains the time series needed by `ev2gym/visuals/evaluator_plot.py` for visualization.

---

## Step 6 — Testing Plan
1) Unit/logic tests (lightweight, synthetic):
- 1 EV, flat price, no PV → charge to meet target, no discharge.
- 1 EV, valley/peak prices → charge in valley, small discharge in peak while meeting target.
- PV surplus (positive mid-day) → midday charging increases, evening discharge allowed.
- Export limit enforcement: cap negative power at limit.
- Ramp/smoothing sanity: no large step changes.

2) Integration checks (short sim):
- `simulation_length: 100`, `timescale: 5`.
- Validate replay: departure SoC met; transformer not overloaded; export never > cap.

3) Back-compat:
- With `enabled=false`, bitwise-identical setpoints vs. current implementation on the same seed.
- Ensure `example.py` and `train_stable_baselines.py` run unchanged.

---

## Step 7 — Rollout Strategy
- Commit in a feature branch.
- Default-off gate ensures current benchmarks unaffected.
- Document the new YAML block in `README.md` and add a small tutorial snippet.

---

## Optional Enhancements
- Per-user target SoC and departure-time uncertainty.
- Battery wear model to penalize excessive cycling.
- Feeder-aware coordination across multiple CS under a transformer using proportional fairness when clipping.

---

## Acceptance Criteria
- With `res_v2g_setpoints.enabled=true`, the logs show nonzero setpoints in expected windows, and replay verifies:
  - Departure SoC targets met for all EVs.
  - Export limit respected; no transformer overloads due to setpoints.
  - Training cost lower than price-only baseline on identical seeds.
- With `enabled=false`, outputs match current behavior.
 - When RL state integration is enabled, observation vectors include current setpoint (and horizon if configured), and setpoint-tracking metrics improve in evaluation plots.
