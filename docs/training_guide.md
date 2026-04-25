# OpenBargain Training Guide

## Scope
This guide explains how to use the PPO-oriented orchestration layer in [open_bargain/training/train.py](../open_bargain/training/train.py).

The training module is an interface and systems layer. It does not implement PPO from scratch.

## Dependencies
Required for training:
- gymnasium
- numpy
- stable-baselines3

Install example:
```powershell
pip install -r requirements.txt
pip install stable-baselines3
```

## Core Entry Points
- TrainingConfig: typed configuration with validation and reproducibility controls.
- make_env: deterministic environment factory for PPO pipelines.
- train_ppo: main training launcher.
- evaluate_policy: deterministic frozen-policy benchmark evaluation.
- CheckpointManager: checkpoint and metadata persistence.

## PPO Training Flow
```text
TrainingConfig
    |
    v
train_ppo
    |
    +--> make_env / make_env_thunk
    |
    +--> external PPO model init or load
    |
    +--> callback loop
    |      - scalar logging
    |      - periodic evaluate_policy
    |      - periodic and best checkpoint save
    |
    +--> final evaluation + final checkpoint
    |
    v
TrainingRunResult
```

## Minimal Training Run
```powershell
@'
from open_bargain.training import TrainingConfig, train_ppo

config = TrainingConfig(
    total_timesteps=20000,
    rollout_steps=1024,
    batch_size=64,
    learning_rate=3e-4,
    gamma=0.99,
    clip_range=0.2,
    evaluation_frequency=5000,
    checkpoint_frequency=5000,
    logging_frequency=1000,
    seed=0,
)

result = train_ppo(config)
print(result.to_dict()["best_checkpoint"])
'@ | python
```

## Config Usage Guidance
Important knobs:
- total_timesteps: total PPO rollout budget.
- rollout_steps and batch_size: throughput and gradient update granularity.
- learning_rate, gamma, clip_range: PPO behavior controls.
- evaluation_frequency: how often benchmark evaluation runs.
- checkpoint_frequency: save cadence for model persistence.
- seed: reproducibility anchor.

Validation safeguards:
- learning_rate must be > 0.
- batch_size must be > 0.
- gamma must be in (0, 1].

## Evaluation Cycle
The training callback runs deterministic evaluation at configured intervals:
1. Freeze current policy weights.
2. Run simulate_episodes with model policy adapters.
3. Produce aggregate metrics and benchmark artifact.
4. Log agreement, fairness, exploitability, and benchmark score.
5. Use benchmark_score to track best checkpoint.

This keeps benchmark evaluation separate from training rollouts.

## Checkpoint Behavior
Checkpoint types:
- checkpoint: periodic snapshots.
- best: highest benchmark_score seen so far.
- final: end-of-run model.

Each checkpoint has metadata sidecar JSON containing:
- config snapshot,
- seed,
- training step,
- evaluation metrics,
- policy metadata,
- benchmark artifact summary.

## Resuming From Checkpoint
Set resume_from_checkpoint in TrainingConfig to continue from a saved model path. Keep seed and core hyperparameters stable for reproducible continuation behavior.

## Logging
Default logger:
- JsonlTrainingLogger writes deterministic JSONL rows.

Logged examples:
- train/step_reward
- train/episode_reward
- eval/agreement_rate
- eval/fairness_index
- eval/exploitability_score
- eval/benchmark_score

You can inject a custom logger implementing TrainingLogger for TensorBoard or other systems.

## Performance Tips
- Increase rollout_steps for fewer learner sync points.
- Keep evaluation_frequency moderate to avoid evaluation overhead.
- Reuse deterministic config snapshots across runs for clean comparisons.
- Prefer fixed seeds during benchmark reporting.

## Debugging Tips
If training fails early:
- verify gymnasium, numpy, and stable-baselines3 are installed.
- confirm action and observation dimensions match expected wrapper shapes.
- reduce total_timesteps and evaluation_frequency for quick smoke tests.

If benchmark scores are unstable:
- verify seed is fixed and consistently propagated.
- compare exported policy metadata and config snapshots.
- run baseline policies first to confirm environment behavior.

## Extensibility Notes
The training layer is designed for future upgrades:
- add GRPO launcher next to train_ppo.
- add RLlib-specific env adapters using make_env_thunk.
- add distributed rollout managers while preserving evaluation hooks.
- add curriculum schedules by modifying config-driven environment factories.
