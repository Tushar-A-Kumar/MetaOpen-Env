# OpenBargain

OpenBargain is a deterministic multi-agent bargaining benchmark for evaluating strategic negotiation policies under hidden utility preferences.

## Problem Statement
Modern RL benchmarks often under-represent real negotiation behavior: agents have partial information, private incentives, and must decide whether to cooperate or exploit. OpenBargain focuses on that gap.

The benchmark models:
- Multi-agent bargaining under partial observability.
- Hidden utility preferences per agent.
- Strategic turn-based negotiation with proposals and responses.
- The core research tension between fairness and exploitation.

Why this benchmark exists:
- To test policy quality beyond reward maximization.
- To quantify agreement stability, equity, and strategic robustness.
- To provide a reproducible environment for comparing negotiation policies and RL methods.

## Environment Overview
OpenBargain exposes an OpenEnv and Gymnasium-compatible interaction model through [open_bargain/env/env.py](open_bargain/env/env.py).

Core dynamics:
- Negotiation is round-based with an active proposer.
- Legal actions are proposal, accept, reject, and counteroffer.
- Agents receive partial observations with action masks.
- Private utility profiles are hidden from opponents.
- Episodes terminate on agreement or max-round timeout.

Conceptually, each step is:
1. Active agent observes public state plus private context.
2. Policy selects one legal action.
3. Environment transitions state deterministically.
4. Reward and metrics hooks are emitted for evaluation.

## Key Innovations
- Hidden preference utility system for strategic behavior realism.
- Fairness vs exploitation metrics in the core benchmark surface.
- Deterministic seeding for reproducible evaluation artifacts.
- Multi-policy evaluation and policy comparison support.
- PPO-ready training orchestration layer for modern RL stacks.

## Architecture Overview
Primary modules:
- [open_bargain/env](open_bargain/env): environment, state, reward, utility, observation.
- [open_bargain/metrics](open_bargain/metrics): per-episode and aggregate benchmark metrics.
- [open_bargain/simulation](open_bargain/simulation): deterministic rollouts, traces, baseline policies.
- [open_bargain/training](open_bargain/training): PPO-oriented training orchestration and evaluation hooks.

System flow:

```text
Policy -> Encoded Observation -> Action
   |                            |
   v                            v
Simulation/Training Wrapper -> OpenBargainEnv
							   |
							   v
					 State Transition + Rewards
							   |
							   v
					MetricsEngine + Trace Capture
							   |
							   v
			   Benchmark Artifact / Comparison Report
```

For deeper diagrams and lifecycle details, see [docs/architecture.md](docs/architecture.md).

## How To Run
The commands below are PowerShell-friendly and deterministic by default.

### 1. Install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install stable-baselines3
```

### 2. Run deterministic simulation benchmark

```powershell
@'
import json
from open_bargain.config import OpenBargainConfig
from open_bargain.env.env import OpenBargainEnv
from open_bargain.simulation.simulate import RandomNegotiationPolicy, simulate_episodes

config = OpenBargainConfig()
env = OpenBargainEnv(config=config)
result = simulate_episodes(
	env=env,
	config=config,
	policy_hooks=[
		RandomNegotiationPolicy(seed=0),
		RandomNegotiationPolicy(seed=1),
	],
)
print(json.dumps(result.export_benchmark_artifact(), indent=2, sort_keys=True))
env.close()
'@ | python
```

### 3. Run baseline policy comparison

```powershell
@'
import json
from open_bargain.config import OpenBargainConfig
from open_bargain.env.env import OpenBargainEnv
from open_bargain.simulation.simulate import (
	RandomNegotiationPolicy,
	FairNegotiationPolicy,
	GreedyNegotiationPolicy,
	build_policy_comparison_summary,
	simulate_episodes,
)

config = OpenBargainConfig()
results = {}

for name, hooks in {
	"random": [RandomNegotiationPolicy(seed=0), RandomNegotiationPolicy(seed=1)],
	"fair": [FairNegotiationPolicy(seed=0), FairNegotiationPolicy(seed=1)],
	"greedy": [GreedyNegotiationPolicy(seed=0), GreedyNegotiationPolicy(seed=1)],
}.items():
	env = OpenBargainEnv(config=config)
	results[name] = simulate_episodes(env=env, config=config, policy_hooks=hooks)
	env.close()

summary = build_policy_comparison_summary(results)
print(json.dumps(summary, indent=2, sort_keys=True))
'@ | python
```

### 4. Launch PPO training scaffold

```powershell
@'
from open_bargain.training import TrainingConfig, train_ppo

config = TrainingConfig(
	total_timesteps=20000,
	evaluation_frequency=5000,
	checkpoint_frequency=5000,
	logging_frequency=1000,
	seed=0,
)
result = train_ppo(config)
print(result.to_dict()["best_checkpoint"])
'@ | python
```

## Benchmark Evaluation
OpenBargain reports deterministic episode-level and aggregate metrics.

Primary metrics:
- Agreement rate: fraction of episodes ending in agreement.
- Fairness index: closeness to balanced outcomes.
- Exploitability score: one-sidedness of negotiated allocations.
- Efficiency score: how quickly agreement is reached.
- Social welfare score: cooperative quality under resource use and timing.
- Benchmark score: unified leaderboard score in [0, 1].

Interpretation guidance:
- High agreement rate is good for reliability.
- High fairness and low exploitability indicate equitable negotiation.
- High efficiency means less delay and fewer wasted rounds.
- High benchmark score indicates stronger overall policy quality.

Detailed metric intuition is documented in [docs/metrics_explained.md](docs/metrics_explained.md).

## Baseline Policies
OpenBargain includes deterministic benchmark baselines in [open_bargain/simulation/simulate.py](open_bargain/simulation/simulate.py):
- Random policy: legal random actions using action masks.
- Fair policy: prefers balanced proposals and near-equal acceptance.
- Greedy policy: prefers self-favoring offers and utility-favorable acceptance.

Why they matter:
- Random provides a low-skill control baseline.
- Fair represents cooperative negotiation behavior.
- Greedy represents self-interested strategic pressure.

## PPO Training Guide
Training entrypoint: [open_bargain/training/train.py](open_bargain/training/train.py) via train_ppo.

What the scaffold handles:
- Deterministic environment creation through make_env.
- PPO-compatible observation and action adaptation.
- Structured logging and JSON-safe run records.
- Periodic frozen-policy evaluation through simulation utilities.
- Checkpointing for periodic, best, and final models.

For full configuration and troubleshooting details, see [docs/training_guide.md](docs/training_guide.md).

## Results Format
Judge-facing outputs should include:
- Benchmark score per run.
- Per-policy comparison summary with ranks.
- Aggregate metrics payload.
- Episode trace artifacts for auditability.

Typical exported structures:
- Simulation artifact from SimulationResult.export_benchmark_artifact.
- Policy comparison report from build_policy_comparison_summary.
- Training run metadata from TrainingRunResult.to_dict.

## Reproducibility Guarantee
OpenBargain enforces deterministic behavior through:
- Global seed control in simulation and training layers.
- Seed propagation into environment resets and policy resets.
- Deterministic action selection for benchmark baselines.
- Canonical JSON-safe export ordering for benchmark artifacts.

Expected outcome: repeated runs with identical config and seed produce stable benchmark results.

## OpenEnv Compatibility
OpenBargain aligns with OpenEnv and Gymnasium expectations:
- Package entrypoint metadata in [openenv.yaml](openenv.yaml).
- Environment API surface in [open_bargain/env/env.py](open_bargain/env/env.py).
- PPO-ready adaptation layer in [open_bargain/training/train.py](open_bargain/training/train.py).
- Multi-agent negotiation design preserved through simulation and training wrappers.


## Limitations And Future Work
- GRPO support can be added via the existing pluggable training layer.
- Simultaneous-move negotiation variants are not yet implemented.
- Distributed training with RLlib or Ray can extend current hooks.
- Richer communication protocols and negotiation actions are future extensions.

## Additional Documentation
- [docs/architecture.md](docs/architecture.md)
- [docs/metrics_explained.md](docs/metrics_explained.md)
- [docs/training_guide.md](docs/training_guide.md)
