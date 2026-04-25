# OpenBargain Architecture

## Purpose
OpenBargain is structured as a benchmark system, not just an environment. The architecture separates environment logic, metric definitions, rollout evaluation, and training orchestration so each part can evolve independently.

## Module Map
- [open_bargain/config.py](../open_bargain/config.py): central typed configuration for benchmark, environment, reward, utility, and simulation settings.
- [open_bargain/env](../open_bargain/env): core negotiation dynamics, state contracts, reward aggregation, utility modeling, and observations.
- [open_bargain/metrics](../open_bargain/metrics): deterministic per-episode and aggregate metric computation.
- [open_bargain/simulation](../open_bargain/simulation): rollout execution, baseline policies, benchmark artifacts, and policy comparison summaries.
- [open_bargain/training](../open_bargain/training): PPO-oriented orchestration layer with wrappers, evaluation hooks, and checkpointing.

## System Design Principles
- Determinism first: seed propagation is explicit in simulation and training.
- Separation of concerns: environment transitions are isolated from training logic.
- JSON-safe reporting: benchmark artifacts can be archived or submitted directly.
- Pluggable policies: random, fair, greedy, and model-based policies share one simulation interface.
- Extensible training bridge: external PPO stacks can connect without rewriting environment internals.

## End-To-End Data Flow
```text
TrainingConfig / OpenBargainConfig
        |
        v
Environment Factory (make_env)
        |
        v
PPO-Compatible Wrapper
(Observation Flattening + Action Decoding)
        |
        v
OpenBargainEnv
(State + Transition + Reward)
        |
        v
Step Outputs
(observation, reward, terminated, info)
        |
        +--------------------+
        |                    |
        v                    v
Train Loop            Simulation Rollouts
(train_ppo)           (simulate_episodes)
        |                    |
        v                    v
CheckpointManager      MetricsEngine
        |                    |
        +---------+----------+
                  v
       Benchmark Artifact / Comparison Summary
```

## Environment Lifecycle
### Reset
1. Resolve agent IDs from options.
2. Initialize deterministic negotiation state.
3. Generate deterministic hidden preference profiles.
4. Build partial observations and action masks.

### Step
1. Validate action legality against current mask.
2. Apply proposal, accept, reject, or counteroffer transition.
3. Update round, active proposer, and terminal state if needed.
4. Compute rewards and build metrics hook payloads.
5. Emit next observations with deterministic ordering.

### Terminal Handling
- Agreement terminal: negotiated allocation accepted.
- Failure terminal: timeout or invalid continuation path.
- Final info contains outcome and state summary for downstream metrics.

## Metrics Pipeline
```text
Episode Trace Inputs
(agreement, rounds, final allocation)
        |
        v
MetricsEngine.evaluate_episode
        |
        v
EpisodeMetrics
        |
        v
MetricsEngine.aggregate
        |
        v
AggregateMetrics + benchmark_score
```

Key property: the metric layer is deterministic and independent of PPO implementation details.

## Simulation Pipeline
1. Build environment and policy hooks.
2. Run deterministic episodes with full step traces.
3. Compute per-episode and aggregate metrics.
4. Export benchmark artifacts and policy metadata.
5. Optionally build cross-policy comparison summaries.

## Training Pipeline Integration
The training scaffold does not implement PPO directly. It orchestrates:
- environment creation,
- wrapper adaptation,
- external PPO model execution,
- periodic frozen-policy evaluation through simulation,
- structured logging,
- checkpoint persistence and replay metadata.

This keeps benchmark integrity independent from algorithm internals.

## Extension Points
- Replace PPO backend while preserving TrainingConfig and evaluation hooks.
- Add GRPO-specific wrappers and callbacks.
- Add distributed environment factories for RLlib or Ray.
- Add curriculum schedules by modifying environment factory inputs only.
- Extend policy comparison dashboards using exported JSON artifacts.
