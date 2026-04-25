"""PPO-oriented training orchestration for OpenBargain experiments.

This module intentionally stays thin around the learning algorithm itself.
It provides environment registration, PPO-compatible wrappers, deterministic
evaluation hooks, structured logging, and checkpoint management so external
PPO implementations can train the benchmark without coupling training logic to
environment internals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
import math
import random
from pathlib import Path
from typing import Any, Callable, Protocol, TYPE_CHECKING

try:  # pragma: no cover - optional runtime dependency.
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - import-safe fallback for minimal environments.
    np = None

from open_bargain.config import OpenBargainConfig

try:  # pragma: no cover - optional runtime dependency.
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:  # pragma: no cover - import-safe fallback for minimal environments.
    gym = None
    spaces = None

if TYPE_CHECKING:  # pragma: no cover - typing-only imports.
    from stable_baselines3.common.base_class import BaseAlgorithm


class _GymEnvBase:
    """Fallback base class used when Gymnasium is unavailable at import time."""

    @classmethod
    def __class_getitem__(cls, item: Any) -> type["_GymEnvBase"]:
        return cls


if gym is not None:  # pragma: no cover - runtime path depends on installed dependency.
    _GymEnvBase = gym.Env


class SimulationPolicy(Protocol):
    """Protocol for simulation-time bargaining policies used during evaluation."""

    def reset(self, *, agent_id: str, agent_ids: list[str], seed: int | None = None) -> None:
        """Reset the policy for a new episode."""

    def select_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Return an environment-compatible bargaining action."""


class TrainingLogger(Protocol):
    """Protocol for deterministic structured training logging backends."""

    def log_scalar(self, name: str, value: float, step: int) -> None:
        """Log one scalar metric value."""

    def log_json(self, name: str, payload: dict[str, Any], step: int | None = None) -> None:
        """Log one structured JSON-safe payload."""

    def flush(self) -> None:
        """Flush any buffered log output."""


def _validate_finite(value: float, field_name: str) -> None:
    """Validate a scalar value is finite."""
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite. Got {value}.")


def _canonical_json_value(value: Any) -> Any:
    """Recursively normalize a value into a JSON-safe deterministic structure."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        _validate_finite(value, "json_float")
        return value
    if isinstance(value, dict):
        return {str(key): _canonical_json_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    if hasattr(value, "to_dict"):
        return _canonical_json_value(value.to_dict())
    raise ValueError(f"Unsupported non-JSON-safe value of type {type(value).__name__}.")


def _json_safe_snapshot(payload: Any) -> dict[str, Any]:
    """Convert a dataclass or mapping-like payload into a canonical JSON-safe dictionary."""
    if hasattr(payload, "__dataclass_fields__"):
        return _canonical_json_value(asdict(payload))
    if hasattr(payload, "to_dict"):
        return _canonical_json_value(payload.to_dict())
    if isinstance(payload, dict):
        return _canonical_json_value(payload)
    raise TypeError(f"Unsupported payload type for JSON snapshot: {type(payload).__name__}.")


def _seed_everything(seed: int) -> None:
    """Seed the Python, NumPy, and optional torch RNGs deterministically."""
    if seed < 0:
        raise ValueError(f"seed must be non-negative. Got {seed}.")
    random.seed(seed)
    np_module = _require_numpy()
    np_module.random.seed(seed)
    try:  # pragma: no cover - torch is optional in this workspace.
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ModuleNotFoundError:
        pass


def _require_numpy() -> Any:
    """Return NumPy when available or raise a clear training-time error."""
    if np is None:  # pragma: no cover - depends on local environment.
        raise ImportError("numpy is required to use the OpenBargain training scaffold.")
    return np


def _validate_policy_metadata(policy_metadata: dict[str, Any], agent_ids: list[str]) -> None:
    """Validate policy metadata consistency against the active agent roster."""
    if not isinstance(policy_metadata, dict):
        raise ValueError("policy_metadata must be a dictionary.")
    metadata_agent_ids = policy_metadata.get("agent_ids")
    if metadata_agent_ids is not None and list(metadata_agent_ids) != list(agent_ids):
        raise ValueError(
            "policy_metadata.agent_ids must match the configured agent roster. "
            f"expected={agent_ids}, got={metadata_agent_ids}."
        )
    metadata_policies = policy_metadata.get("policies")
    if metadata_policies is not None and not isinstance(metadata_policies, dict):
        raise ValueError("policy_metadata.policies must be a dictionary when provided.")


def _build_open_bargain_config(config: OpenBargainConfig | "TrainingConfig") -> OpenBargainConfig:
    """Extract the benchmark config from either a training config or raw benchmark config."""
    if isinstance(config, TrainingConfig):
        return config.open_bargain_config
    return config


def _build_agent_ids(config: OpenBargainConfig | "TrainingConfig") -> tuple[str, ...]:
    """Resolve the agent roster used for training and evaluation."""
    if isinstance(config, TrainingConfig):
        return config.agent_ids
    return ("agent_0", "agent_1")


def _build_seed(config: OpenBargainConfig | "TrainingConfig", seed: int) -> int:
    """Resolve the deterministic seed used for env initialization."""
    if isinstance(config, TrainingConfig) and seed is None:
        return config.seed
    return seed


def _build_config_snapshot(config: OpenBargainConfig | "TrainingConfig") -> dict[str, Any]:
    """Create a JSON-safe snapshot of a benchmark or training configuration."""
    return _json_safe_snapshot(config)


@dataclass(slots=True, frozen=True)
class TrainingConfig:
    """Strongly typed PPO training configuration for OpenBargain.

    Fields are intentionally explicit so experiment runs remain reproducible and
    checkpoint metadata can be replayed without inference from external state.
    """

    open_bargain_config: OpenBargainConfig = field(default_factory=OpenBargainConfig)
    learning_rate: float = 3e-4
    gamma: float = 0.99
    clip_range: float = 0.2
    batch_size: int = 64
    rollout_steps: int = 2048
    total_timesteps: int = 100_000
    evaluation_frequency: int = 10_000
    evaluation_episodes: int = 5
    logging_frequency: int = 1_000
    checkpoint_frequency: int = 10_000
    seed: int = 0
    agent_ids: tuple[str, ...] = ("agent_0", "agent_1")
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    render_mode: str | None = None
    resume_from_checkpoint: str | None = None
    policy_name: str = "MultiInputPolicy"
    device: str = "auto"
    policy_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate reproducibility and PPO hyperparameters."""
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0. Got {self.learning_rate}.")
        if self.gamma <= 0.0 or self.gamma > 1.0:
            raise ValueError(f"gamma must be in (0, 1]. Got {self.gamma}.")
        if self.clip_range <= 0.0 or self.clip_range > 1.0:
            raise ValueError(f"clip_range must be in (0, 1]. Got {self.clip_range}.")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0. Got {self.batch_size}.")
        if self.rollout_steps <= 0:
            raise ValueError(f"rollout_steps must be > 0. Got {self.rollout_steps}.")
        if self.total_timesteps <= 0:
            raise ValueError(f"total_timesteps must be > 0. Got {self.total_timesteps}.")
        if self.evaluation_frequency <= 0:
            raise ValueError(f"evaluation_frequency must be > 0. Got {self.evaluation_frequency}.")
        if self.evaluation_episodes <= 0:
            raise ValueError(f"evaluation_episodes must be > 0. Got {self.evaluation_episodes}.")
        if self.logging_frequency <= 0:
            raise ValueError(f"logging_frequency must be > 0. Got {self.logging_frequency}.")
        if self.checkpoint_frequency <= 0:
            raise ValueError(f"checkpoint_frequency must be > 0. Got {self.checkpoint_frequency}.")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative. Got {self.seed}.")
        if len(self.agent_ids) < 2:
            raise ValueError("agent_ids must contain at least two agents.")
        if len(set(self.agent_ids)) != len(self.agent_ids):
            raise ValueError("agent_ids must be unique.")
        for agent_id in self.agent_ids:
            if not agent_id.strip():
                raise ValueError("agent_ids must not contain empty strings.")
        if not self.checkpoint_dir.strip():
            raise ValueError("checkpoint_dir must be non-empty.")
        if not self.log_dir.strip():
            raise ValueError("log_dir must be non-empty.")
        if not self.policy_name.strip():
            raise ValueError("policy_name must be non-empty.")
        if not self.device.strip():
            raise ValueError("device must be non-empty.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the training configuration into a JSON-safe dictionary."""
        return _json_safe_snapshot(self)


@dataclass(slots=True, frozen=True)
class EvaluationSnapshot:
    """Deterministic evaluation payload produced during training."""

    training_step: int
    seed: int
    episode_count: int
    benchmark_artifact: dict[str, Any]
    aggregate_metrics: dict[str, Any]
    reward_summary: dict[str, Any]
    policy_metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate the snapshot payload is JSON-safe and finite."""
        if self.training_step < 0:
            raise ValueError(f"training_step must be non-negative. Got {self.training_step}.")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative. Got {self.seed}.")
        if self.episode_count <= 0:
            raise ValueError(f"episode_count must be > 0. Got {self.episode_count}.")
        _validate_policy_metadata(self.policy_metadata, list(self.policy_metadata.get("agent_ids", [])))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the evaluation snapshot into JSON-safe form."""
        return {
            "training_step": self.training_step,
            "seed": self.seed,
            "episode_count": self.episode_count,
            "benchmark_artifact": _canonical_json_value(self.benchmark_artifact),
            "aggregate_metrics": _canonical_json_value(self.aggregate_metrics),
            "reward_summary": _canonical_json_value(self.reward_summary),
            "policy_metadata": _canonical_json_value(self.policy_metadata),
        }


@dataclass(slots=True, frozen=True)
class CheckpointRecord:
    """Metadata describing one model checkpoint saved during training."""

    kind: str
    training_step: int
    checkpoint_path: str
    metadata_path: str
    seed: int
    config_snapshot: dict[str, Any]
    evaluation_metrics: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate checkpoint record metadata."""
        if not self.kind.strip():
            raise ValueError("kind must be non-empty.")
        if self.training_step < 0:
            raise ValueError(f"training_step must be non-negative. Got {self.training_step}.")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative. Got {self.seed}.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize checkpoint metadata into JSON-safe form."""
        return {
            "kind": self.kind,
            "training_step": self.training_step,
            "checkpoint_path": self.checkpoint_path,
            "metadata_path": self.metadata_path,
            "seed": self.seed,
            "config_snapshot": _canonical_json_value(self.config_snapshot),
            "evaluation_metrics": _canonical_json_value(self.evaluation_metrics),
        }


@dataclass(slots=True)
class TrainingRunResult:
    """Structured result returned by the PPO training entrypoint."""

    config_snapshot: dict[str, Any]
    total_timesteps: int
    evaluation_history: list[EvaluationSnapshot] = field(default_factory=list)
    checkpoint_history: list[CheckpointRecord] = field(default_factory=list)
    best_checkpoint: CheckpointRecord | None = None
    final_checkpoint: CheckpointRecord | None = None
    resumed_from_checkpoint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the training run result into JSON-safe form."""
        return {
            "config_snapshot": _canonical_json_value(self.config_snapshot),
            "total_timesteps": self.total_timesteps,
            "evaluation_history": [snapshot.to_dict() for snapshot in self.evaluation_history],
            "checkpoint_history": [checkpoint.to_dict() for checkpoint in self.checkpoint_history],
            "best_checkpoint": None if self.best_checkpoint is None else self.best_checkpoint.to_dict(),
            "final_checkpoint": None if self.final_checkpoint is None else self.final_checkpoint.to_dict(),
            "resumed_from_checkpoint": self.resumed_from_checkpoint,
        }


@dataclass(slots=True)
class TrainingHooks:
    """Backward-compatible compatibility hooks for older callers.

    The new training entrypoint is `train_ppo`, but this class remains available
    for code that still expects an object carrying logger and registration hints.
    """

    env_registration_id: str
    logger: TrainingLogger | None = None

    def register_environment(self) -> None:
        """Retain the historical no-op registration hook."""


class JsonlTrainingLogger:
    """Simple deterministic JSONL logger suitable for local training runs."""

    def __init__(self, log_path: str | Path) -> None:
        """Open or create the target JSONL file for structured logging."""
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("a", encoding="utf-8")

    def log_scalar(self, name: str, value: float, step: int) -> None:
        """Log one scalar metric record."""
        _validate_finite(value, name)
        self._write({"type": "scalar", "name": name, "value": value, "step": step})

    def log_json(self, name: str, payload: dict[str, Any], step: int | None = None) -> None:
        """Log one structured JSON-safe record."""
        self._write({"type": "json", "name": name, "step": step, "payload": _canonical_json_value(payload)})

    def flush(self) -> None:
        """Flush buffered log output to disk."""
        self._handle.flush()

    def close(self) -> None:
        """Close the underlying log file handle."""
        self._handle.close()

    def _write(self, payload: dict[str, Any]) -> None:
        """Write one canonical JSONL row."""
        self._handle.write(json.dumps(_canonical_json_value(payload), sort_keys=True) + "\n")
        self.flush()


class PPOCompatibleOpenBargainEnv(_GymEnvBase):
    """Gymnasium-compatible wrapper that adapts OpenBargain to PPO-friendly APIs.

    The wrapper does not alter environment logic. It converts the multi-agent,
    turn-based bargaining interaction into a single-agent view suitable for PPO
    pipelines while preserving raw rewards and transition details in `info`.
    """

    metadata = {"render_modes": [None]}

    def __init__(
        self,
        *,
        config: OpenBargainConfig,
        seed: int,
        agent_ids: tuple[str, ...],
        render_mode: str | None = None,
    ) -> None:
        """Create the PPO-facing wrapper around the OpenBargain environment."""
        if gym is None or spaces is None:  # pragma: no cover - depends on optional dependency.
            raise ImportError("gymnasium is required to construct PPOCompatibleOpenBargainEnv.")
        np_module = _require_numpy()
        if len(agent_ids) < 2:
            raise ValueError("agent_ids must contain at least two agents.")
        self.config = config
        self._seed = seed
        self._agent_ids = tuple(agent_ids)
        self._render_mode = render_mode
        self._history_window = config.environment.history_summary_window
        self._action_types = config.environment.action_types
        self._raw_env = self._build_raw_env()
        self._current_active_agent_id = self._agent_ids[0]
        self._episode_step = 0
        self._feature_size = self._compute_feature_size()
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=-np_module.inf,
                    high=np_module.inf,
                    shape=(self._feature_size,),
                    dtype=np_module.float32,
                ),
                "action_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(len(self._action_types),),
                    dtype=np_module.int8,
                ),
            }
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self._action_types) + len(self._agent_ids),),
            dtype=np_module.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset the wrapped bargaining environment and return PPO-friendly observations."""
        reset_seed = self._seed if seed is None else seed
        if reset_seed < 0:
            raise ValueError(f"seed must be non-negative. Got {reset_seed}.")
        reset_options = dict(options or {})
        reset_options["agent_ids"] = list(self._agent_ids)
        observations, info = self._raw_env.reset(seed=reset_seed, options=reset_options)
        self._episode_step = 0
        self._current_active_agent_id = self._extract_active_agent_id(observations)
        encoded = self._encode_observation(observations[self._current_active_agent_id], self._current_active_agent_id)
        return encoded, {
            "seed": reset_seed,
            "agent_ids": list(self._agent_ids),
            "active_agent_id": self._current_active_agent_id,
            "raw_info": _canonical_json_value(info),
        }

    def step(
        self,
        action: np.ndarray | list[float] | tuple[float, ...] | dict[str, Any],
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Apply one encoded PPO action and return a scalar reward plus diagnostics."""
        active_agent_id = self._current_active_agent_id
        active_observation = self._latest_raw_observation(active_agent_id)
        decoded_action = self._decode_action(action, active_agent_id, active_observation)
        next_observations, rewards, terminated, truncated, info = self._raw_env.step(decoded_action)
        self._episode_step += 1
        next_active_agent_id = self._extract_active_agent_id(next_observations)
        scalar_reward = float(rewards.get(active_agent_id, 0.0))
        encoded_next_observation = self._encode_observation(
            next_observations[next_active_agent_id],
            next_active_agent_id,
        )
        self._current_active_agent_id = next_active_agent_id
        return encoded_next_observation, scalar_reward, bool(terminated.get("__all__", False)), bool(truncated.get("__all__", False)), {
            "active_agent_id": active_agent_id,
            "next_active_agent_id": next_active_agent_id,
            "raw_action": _canonical_json_value(decoded_action),
            "raw_rewards": _canonical_json_value(rewards),
            "raw_terminated": _canonical_json_value(terminated),
            "raw_truncated": _canonical_json_value(truncated),
            "raw_info": _canonical_json_value(info),
        }

    def close(self) -> None:
        """Close the wrapped environment."""
        if hasattr(self._raw_env, "close"):
            self._raw_env.close()

    def _build_raw_env(self) -> Any:
        """Instantiate the underlying OpenBargain environment lazily."""
        from open_bargain.env.env import OpenBargainEnv

        return OpenBargainEnv(config=self.config)

    def _compute_feature_size(self) -> int:
        """Compute the flattened observation size deterministically."""
        return 14 + len(self._agent_ids) + len(self._action_types)

    def _latest_raw_observation(self, agent_id: str) -> dict[str, Any]:
        """Fetch the latest raw observation for the requested agent."""
        observation = self._raw_env._build_all_observations()  # noqa: SLF001 - intentional adapter boundary.
        if agent_id not in observation:
            raise ValueError(f"Missing observation for active agent '{agent_id}'.")
        return observation[agent_id]

    def _extract_active_agent_id(self, observations: dict[str, Any]) -> str:
        """Extract the active proposer from raw multi-agent observations."""
        if not observations:
            raise ValueError("observations must not be empty.")
        first_agent_id = self._agent_ids[0]
        first_observation = observations.get(first_agent_id)
        if not isinstance(first_observation, dict):
            raise ValueError("Agent observation must be a dictionary.")
        public = first_observation.get("public")
        if not isinstance(public, dict):
            raise ValueError("Observation must contain a public dictionary.")
        active_agent_id = public.get("active_proposer_id")
        if not isinstance(active_agent_id, str) or not active_agent_id.strip():
            raise ValueError("public.active_proposer_id must be a non-empty string.")
        return active_agent_id

    def _encode_observation(self, raw_observation: dict[str, Any], active_agent_id: str) -> dict[str, np.ndarray]:
        """Flatten the active agent observation into PPO-friendly features."""
        np_module = _require_numpy()
        public = raw_observation.get("public", {})
        private = raw_observation.get("private", {})
        summary = raw_observation.get("summary", {})
        valid_action_mask = raw_observation.get("valid_action_mask", {})
        if not isinstance(public, dict) or not isinstance(private, dict) or not isinstance(summary, dict):
            raise ValueError("Raw observation sections must be dictionaries.")
        if not isinstance(valid_action_mask, dict):
            raise ValueError("Raw observation must provide a valid_action_mask dictionary.")
        total_resource = self.config.environment.total_resource_amount
        max_rounds = self.config.environment.max_negotiation_rounds
        history_window = max(1, self._history_window)
        profile = private.get("private_utility_profile_summary")
        if not isinstance(profile, dict):
            profile = {}
        current_offer = public.get("current_public_offer")
        if not isinstance(current_offer, dict):
            current_offer = {}
        recent_action_summary = summary.get("recent_action_summary")
        if not isinstance(recent_action_summary, dict):
            recent_action_summary = {}
        features: list[float] = [
            float(public.get("current_round", 0)) / max(1.0, float(max_rounds)),
            float(public.get("current_step", 0)) / max(1.0, float(max_rounds * 10)),
            1.0 if public.get("active_proposer_id") == active_agent_id else 0.0,
            float(public.get("remaining_resource", 0.0)) / max(1.0, float(total_resource)),
            float(public.get("remaining_resource_normalized", 0.0)),
            float(public.get("rounds_remaining", 0)) / max(1.0, float(max_rounds)),
            float(public.get("round_progress", 0.0)),
            1.0 if public.get("is_terminal") else 0.0,
            1.0 if current_offer else 0.0,
            float(summary.get("offer_count", 0)) / max(1.0, float(history_window)),
            float(summary.get("rejection_count", 0)) / max(1.0, float(history_window)),
            1.0 if summary.get("acceptance_flag") else 0.0,
            float(profile.get("greed_sensitivity", 0.0)),
            float(profile.get("fairness_sensitivity", 0.0)),
            float(profile.get("urgency_sensitivity", 0.0)),
        ]
        for agent_id in self._agent_ids:
            features.append(float(current_offer.get(agent_id, 0.0)) / max(1.0, float(total_resource)))
        for action_type in self._action_types:
            features.append(float(recent_action_summary.get(action_type, 0)) / max(1.0, float(history_window)))
        return {
            "observation": np_module.asarray(features, dtype=np_module.float32),
            "action_mask": action_mask,
        }

    def _decode_action(
        self,
        action: np.ndarray | list[float] | tuple[float, ...] | dict[str, Any],
        active_agent_id: str,
        raw_observation: dict[str, Any],
    ) -> dict[str, Any]:
        """Decode a PPO action vector into an OpenBargain action dictionary."""
        np_module = _require_numpy()
        valid_action_mask = raw_observation.get("valid_action_mask")
        if not isinstance(valid_action_mask, dict):
            raise ValueError("Raw observation must include a valid_action_mask dictionary.")
        if isinstance(action, dict):
            action_type_index = int(action.get("action_type_index", 0))
            allocation_scores = np_module.asarray(action.get("allocation", []), dtype=np_module.float32)
        else:
            action_array = np_module.asarray(action, dtype=np_module.float32)
            expected_size = len(self._action_types) + len(self._agent_ids)
            if action_array.shape != (expected_size,):
                raise ValueError(
                    f"Action vector must have shape ({expected_size},). Got {action_array.shape}."
                )
            action_type_scores = action_array[: len(self._action_types)]
            action_type_index = self._select_legal_action_index(action_type_scores, valid_action_mask)
            allocation_scores = action_array[len(self._action_types) :]
        action_type = self._action_types[action_type_index]
        payload: dict[str, Any] = {}
        if action_type in ("propose", "counteroffer"):
            remaining_resource = float(raw_observation.get("public", {}).get("remaining_resource", 0.0))
            allocation = self._decode_allocation(allocation_scores, remaining_resource)
            payload["allocation"] = allocation
        return {"agent_id": active_agent_id, "action_type": action_type, "payload": payload}

    def _select_legal_action_index(self, action_type_scores: np.ndarray, valid_action_mask: dict[str, bool]) -> int:
        """Select the highest-scoring legal action index deterministically."""
        legal_indices = [index for index, action_type in enumerate(self._action_types) if bool(valid_action_mask.get(action_type, False))]
        if not legal_indices:
            raise ValueError("No valid actions available for the current state.")
        best_index = legal_indices[0]
        best_score = float(action_type_scores[best_index])
        for index in legal_indices[1:]:
            score = float(action_type_scores[index])
            if score > best_score or (score == best_score and index < best_index):
                best_index = index
                best_score = score
        return best_index

    def _decode_allocation(self, allocation_scores: np.ndarray, remaining_resource: float) -> dict[str, float]:
        """Convert allocation scores into a valid resource split over all agents."""
        np_module = _require_numpy()
        if allocation_scores.shape != (len(self._agent_ids),):
            raise ValueError(
                f"Allocation score vector must have shape ({len(self._agent_ids)},). Got {allocation_scores.shape}."
            )
        weights = np_module.maximum(allocation_scores.astype(np_module.float64), 0.0)
        if not np_module.any(weights):
            weights = np_module.ones(len(self._agent_ids), dtype=np_module.float64)
        weights = weights / float(np.sum(weights))
        allocation_values = weights * float(remaining_resource)
        allocation = {agent_id: float(value) for agent_id, value in zip(self._agent_ids, allocation_values)}
        residual = float(remaining_resource) - sum(allocation.values())
        allocation[self._agent_ids[-1]] += residual
        return allocation


class PPOModelPolicyAdapter:
    """Adapt an external PPO model to the `SimulationPolicy` interface."""

    def __init__(self, model: Any, config: TrainingConfig, deterministic: bool = True) -> None:
        """Store the external PPO model and conversion context."""
        self._model = model
        self._config = config
        self._deterministic = deterministic
        self._agent_id: str | None = None
        self._agent_ids: list[str] = []
        self._seed: int = config.seed

    def reset(self, *, agent_id: str, agent_ids: list[str], seed: int | None = None) -> None:
        """Reset the adapter for a new agent turn."""
        if not agent_id.strip():
            raise ValueError("agent_id must be non-empty.")
        if agent_id not in agent_ids:
            raise ValueError(f"agent_id '{agent_id}' must exist in agent_ids {agent_ids}.")
        self._agent_id = agent_id
        self._agent_ids = list(agent_ids)
        self._seed = self._config.seed if seed is None else seed

    def select_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Map a raw OpenBargain observation to a PPO action and back again."""
        if self._agent_id is None:
            raise RuntimeError("Policy adapter must be reset before select_action is called.")
        encoded = _encode_policy_observation(
            raw_observation=observation,
            config=self._config,
            agent_ids=self._agent_ids,
            active_agent_id=self._agent_id,
        )
        model_action, _ = self._model.predict(encoded, deterministic=self._deterministic)
        return _decode_policy_action(
            model_action,
            raw_observation=observation,
            agent_ids=self._agent_ids,
            active_agent_id=self._agent_id,
            config=self._config,
        )


def _encode_policy_observation(
    *,
    raw_observation: dict[str, Any],
    config: TrainingConfig,
    agent_ids: list[str],
    active_agent_id: str,
) -> dict[str, np.ndarray]:
    """Encode a raw OpenBargain observation into the PPO wrapper representation."""
    np_module = _require_numpy()
    public = raw_observation.get("public", {})
    private = raw_observation.get("private", {})
    summary = raw_observation.get("summary", {})
    valid_action_mask = raw_observation.get("valid_action_mask", {})
    if not isinstance(public, dict) or not isinstance(private, dict) or not isinstance(summary, dict):
        raise ValueError("Raw observation sections must be dictionaries.")
    if not isinstance(valid_action_mask, dict):
        raise ValueError("Raw observation must provide a valid_action_mask dictionary.")
    total_resource = config.open_bargain_config.environment.total_resource_amount
    max_rounds = config.open_bargain_config.environment.max_negotiation_rounds
    history_window = max(1, config.open_bargain_config.environment.history_summary_window)
    action_types = config.open_bargain_config.environment.action_types
    profile = private.get("private_utility_profile_summary")
    if not isinstance(profile, dict):
        profile = {}
    current_offer = public.get("current_public_offer")
    if not isinstance(current_offer, dict):
        current_offer = {}
    recent_action_summary = summary.get("recent_action_summary")
    if not isinstance(recent_action_summary, dict):
        recent_action_summary = {}
    features: list[float] = [
        float(public.get("current_round", 0)) / max(1.0, float(max_rounds)),
        float(public.get("current_step", 0)) / max(1.0, float(max_rounds * 10)),
        1.0 if public.get("active_proposer_id") == active_agent_id else 0.0,
        float(public.get("remaining_resource", 0.0)) / max(1.0, float(total_resource)),
        float(public.get("remaining_resource_normalized", 0.0)),
        float(public.get("rounds_remaining", 0)) / max(1.0, float(max_rounds)),
        float(public.get("round_progress", 0.0)),
        1.0 if public.get("is_terminal") else 0.0,
        1.0 if current_offer else 0.0,
        float(summary.get("offer_count", 0)) / max(1.0, float(history_window)),
        float(summary.get("rejection_count", 0)) / max(1.0, float(history_window)),
        1.0 if summary.get("acceptance_flag") else 0.0,
        float(profile.get("greed_sensitivity", 0.0)),
        float(profile.get("fairness_sensitivity", 0.0)),
        float(profile.get("urgency_sensitivity", 0.0)),
    ]
    for agent_id in agent_ids:
        features.append(float(current_offer.get(agent_id, 0.0)) / max(1.0, float(total_resource)))
    for action_type in action_types:
        features.append(float(recent_action_summary.get(action_type, 0)) / max(1.0, float(history_window)))
    action_mask = np_module.asarray(
        [1 if bool(valid_action_mask.get(action_type, False)) else 0 for action_type in action_types],
        dtype=np_module.int8,
    )
    return {
        "observation": np_module.asarray(features, dtype=np_module.float32),
        "action_mask": action_mask,
    }


def _decode_policy_action(
    model_action: Any,
    *,
    raw_observation: dict[str, Any],
    agent_ids: list[str],
    active_agent_id: str,
    config: TrainingConfig,
) -> dict[str, Any]:
    """Decode a PPO model action into the OpenBargain environment contract."""
    np_module = _require_numpy()
    action_types = config.open_bargain_config.environment.action_types
    valid_action_mask = raw_observation.get("valid_action_mask")
    if not isinstance(valid_action_mask, dict):
        raise ValueError("Raw observation must include a valid_action_mask dictionary.")
    action_array = np_module.asarray(model_action, dtype=np_module.float32).reshape(-1)
    expected_size = len(action_types) + len(agent_ids)
    if action_array.shape != (expected_size,):
        raise ValueError(f"Model action must have shape ({expected_size},). Got {action_array.shape}.")
    action_type_scores = action_array[: len(action_types)]
    action_index = _select_legal_action_index(action_type_scores, action_types, valid_action_mask)
    action_type = action_types[action_index]
    payload: dict[str, Any] = {}
    if action_type in ("propose", "counteroffer"):
        remaining_resource = float(raw_observation.get("public", {}).get("remaining_resource", 0.0))
        allocation_scores = action_array[len(action_types) :]
        payload["allocation"] = _decode_allocation_scores(allocation_scores, agent_ids, remaining_resource)
    return {"agent_id": active_agent_id, "action_type": action_type, "payload": payload}


def _select_legal_action_index(
    action_scores: Any,
    action_types: tuple[str, ...],
    valid_action_mask: dict[str, bool],
) -> int:
    """Pick the highest-scoring action that is legal in the current observation."""
    legal_indices = [index for index, action_type in enumerate(action_types) if bool(valid_action_mask.get(action_type, False))]
    if not legal_indices:
        raise ValueError("No valid actions available for the current state.")
    best_index = legal_indices[0]
    best_score = float(action_scores[best_index])
    for index in legal_indices[1:]:
        score = float(action_scores[index])
        if score > best_score or (score == best_score and index < best_index):
            best_index = index
            best_score = score
    return best_index


def _decode_allocation_scores(
    allocation_scores: Any,
    agent_ids: list[str],
    remaining_resource: float,
) -> dict[str, float]:
    """Convert allocation scores into a valid resource split over all agents."""
    np_module = _require_numpy()
    if allocation_scores.shape != (len(agent_ids),):
        raise ValueError(
            f"Allocation score vector must have shape ({len(agent_ids)},). Got {allocation_scores.shape}."
        )
    weights = np_module.maximum(allocation_scores.astype(np_module.float64), 0.0)
    if not np_module.any(weights):
        weights = np_module.ones(len(agent_ids), dtype=np_module.float64)
    weights = weights / float(np_module.sum(weights))
    allocation_values = weights * float(remaining_resource)
    allocation = {agent_id: float(value) for agent_id, value in zip(agent_ids, allocation_values)}
    residual = float(remaining_resource) - sum(allocation.values())
    allocation[agent_ids[-1]] += residual
    return allocation


@dataclass(slots=True)
class CheckpointManager:
    """Handle checkpoint file creation and deterministic metadata sidecars."""

    checkpoint_dir: Path
    config_snapshot: dict[str, Any]

    def __post_init__(self) -> None:
        """Create the checkpoint directory eagerly."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        *,
        model: Any,
        kind: str,
        training_step: int,
        evaluation_snapshot: EvaluationSnapshot,
        seed: int,
    ) -> CheckpointRecord:
        """Save a model checkpoint and its metadata sidecar."""
        filename = f"{kind}_{training_step:09d}.zip"
        checkpoint_path = self.checkpoint_dir / filename
        metadata_path = checkpoint_path.with_suffix(".json")
        model.save(str(checkpoint_path))
        metadata = {
            "kind": kind,
            "training_step": training_step,
            "seed": seed,
            "config_snapshot": self.config_snapshot,
            "evaluation_metrics": evaluation_snapshot.aggregate_metrics,
            "benchmark_artifact": evaluation_snapshot.benchmark_artifact,
            "reward_summary": evaluation_snapshot.reward_summary,
            "policy_metadata": evaluation_snapshot.policy_metadata,
        }
        metadata_path.write_text(json.dumps(_canonical_json_value(metadata), indent=2, sort_keys=True), encoding="utf-8")
        return CheckpointRecord(
            kind=kind,
            training_step=training_step,
            checkpoint_path=str(checkpoint_path),
            metadata_path=str(metadata_path),
            seed=seed,
            config_snapshot=self.config_snapshot,
            evaluation_metrics=evaluation_snapshot.aggregate_metrics,
        )


def register_training_environment(
    *,
    env_id: str,
    config: TrainingConfig,
) -> None:
    """Register the PPO-compatible OpenBargain environment with Gymnasium."""
    if gym is None:  # pragma: no cover - optional dependency gate.
        raise ImportError("gymnasium is required to register OpenBargain training environments.")
    from gymnasium.envs.registration import register

    def _factory(**kwargs: Any) -> PPOCompatibleOpenBargainEnv:
        return make_env(config, seed=kwargs.get("seed", config.seed), render_mode=kwargs.get("render_mode"))

    try:
        register(id=env_id, entry_point=_factory)
    except Exception as exc:  # pragma: no cover - registration is environment dependent.
        if "already registered" not in str(exc).lower():
            raise


def make_env(
    config: OpenBargainConfig | TrainingConfig,
    seed: int,
    render_mode: str | None = None,
) -> PPOCompatibleOpenBargainEnv:
    """Create the single training environment entry point for PPO pipelines."""
    benchmark_config = _build_open_bargain_config(config)
    agent_ids = _build_agent_ids(config)
    resolved_seed = _build_seed(config, seed)
    return PPOCompatibleOpenBargainEnv(
        config=benchmark_config,
        seed=resolved_seed,
        agent_ids=agent_ids,
        render_mode=render_mode,
    )


def make_env_thunk(
    config: TrainingConfig,
    seed: int,
    render_mode: str | None = None,
) -> Callable[[], PPOCompatibleOpenBargainEnv]:
    """Create a lazy environment constructor for vectorized training stacks."""

    def _thunk() -> PPOCompatibleOpenBargainEnv:
        return make_env(config, seed=seed, render_mode=render_mode)

    return _thunk


def _build_raw_env_for_evaluation(config: TrainingConfig, seed: int) -> Any:
    """Construct an evaluation environment using the native OpenBargain env class."""
    from open_bargain.env.env import OpenBargainEnv

    benchmark_config = replace(
        config.open_bargain_config,
        simulation=replace(config.open_bargain_config.simulation, default_random_seed=seed),
    )
    return OpenBargainEnv(config=benchmark_config)


def _build_simulation_policies(model: Any, config: TrainingConfig) -> list[SimulationPolicy]:
    """Create one adapter policy per agent for deterministic evaluation rollouts."""
    return [PPOModelPolicyAdapter(model=model, config=config, deterministic=True) for _ in config.agent_ids]


def evaluate_policy(
    model: Any,
    config: TrainingConfig,
    *,
    training_step: int,
    logger: TrainingLogger | None = None,
) -> EvaluationSnapshot:
    """Run deterministic benchmark evaluation for a frozen PPO policy."""
    from open_bargain.simulation.simulate import simulate_episodes

    evaluation_seed = config.seed + config.evaluation_frequency + training_step
    evaluation_env = _build_raw_env_for_evaluation(config, evaluation_seed)
    policies = _build_simulation_policies(model, config)
    benchmark_config = replace(
        config.open_bargain_config,
        simulation=replace(config.open_bargain_config.simulation, default_random_seed=evaluation_seed),
    )
    simulation_result = simulate_episodes(
        env=evaluation_env,
        config=benchmark_config,
        policy_hooks=policies,
    )
    evaluation_env.close()
    if simulation_result.aggregate_metrics is None:
        raise RuntimeError("Evaluation simulation did not produce aggregate metrics.")
    benchmark_artifact = simulation_result.export_benchmark_artifact()
    snapshot = EvaluationSnapshot(
        training_step=training_step,
        seed=evaluation_seed,
        episode_count=len(simulation_result.traces),
        benchmark_artifact=benchmark_artifact,
        aggregate_metrics=simulation_result.aggregate_metrics.to_dict(),
        reward_summary=simulation_result.export_reward_summary(),
        policy_metadata=_canonical_json_value(simulation_result.policy_metadata),
    )
    if logger is not None:
        metrics = simulation_result.aggregate_metrics
        logger.log_scalar("eval/agreement_rate", metrics.agreement_rate, training_step)
        logger.log_scalar("eval/fairness_index", metrics.average_fairness_index, training_step)
        logger.log_scalar("eval/exploitability_score", metrics.average_exploitability_score, training_step)
        logger.log_scalar("eval/benchmark_score", metrics.benchmark_score, training_step)
        logger.log_json("eval/snapshot", snapshot.to_dict(), training_step)
    return snapshot


def _load_sb3_dependencies() -> tuple[type[Any], type[Any], type[Any], type[Any], type[Any]]:
    """Import the PPO training stack lazily so the module stays import-safe."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment.
        raise ImportError(
            "stable-baselines3 is required to run train_ppo(). Install it to enable PPO training."
        ) from exc
    return PPO, BaseCallback, DummyVecEnv, VecMonitor


def train_ppo(
    config: TrainingConfig,
    *,
    logger: TrainingLogger | None = None,
) -> TrainingRunResult:
    """Train a PPO agent against OpenBargain using an external PPO implementation."""
    PPO, BaseCallback, DummyVecEnv, VecMonitor = _load_sb3_dependencies()
    _seed_everything(config.seed)
    config_snapshot = _build_config_snapshot(config)
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=Path(config.checkpoint_dir),
        config_snapshot=config_snapshot,
    )
    if logger is None:
        logger = JsonlTrainingLogger(Path(config.log_dir) / f"seed_{config.seed}.jsonl")
    train_env = DummyVecEnv([make_env_thunk(config, seed=config.seed, render_mode=config.render_mode)])
    train_env = VecMonitor(train_env)
    model_kwargs: dict[str, Any] = {
        "policy": config.policy_name,
        "env": train_env,
        "learning_rate": config.learning_rate,
        "gamma": config.gamma,
        "clip_range": config.clip_range,
        "batch_size": config.batch_size,
        "n_steps": config.rollout_steps,
        "device": config.device,
        "verbose": 0,
        "policy_kwargs": dict(config.policy_kwargs),
        "seed": config.seed,
    }
    model = PPO(**model_kwargs)
    resumed_from_checkpoint: str | None = None
    if config.resume_from_checkpoint is not None:
        model = PPO.load(config.resume_from_checkpoint, env=train_env, device=config.device)
        resumed_from_checkpoint = config.resume_from_checkpoint
    evaluation_history: list[EvaluationSnapshot] = []
    checkpoint_history: list[CheckpointRecord] = []
    best_checkpoint: CheckpointRecord | None = None
    best_score = -math.inf

    class TrainingProgressCallback(BaseCallback):
        """SB3 callback that performs logging, evaluation, and checkpointing."""

        def __init__(self) -> None:
            """Initialize the callback with outer-scope training controls."""
            super().__init__()

        def _on_step(self) -> bool:
            """Handle one training step and trigger periodic side effects."""
            nonlocal best_score, best_checkpoint
            step = int(self.num_timesteps)
            infos = self.locals.get("infos", [])
            rewards = self.locals.get("rewards", [])
            if step % config.logging_frequency == 0 and len(rewards) > 0:
                np_module = _require_numpy()
                logger.log_scalar("train/step_reward", float(np_module.mean(rewards)), step)
            for info in infos:
                episode = info.get("episode") if isinstance(info, dict) else None
                if isinstance(episode, dict):
                    logger.log_scalar("train/episode_reward", float(episode.get("r", 0.0)), step)
                    logger.log_scalar("train/episode_length", float(episode.get("l", 0.0)), step)
            if step % config.evaluation_frequency == 0:
                evaluation_snapshot = evaluate_policy(model, config, training_step=step, logger=logger)
                evaluation_history.append(evaluation_snapshot)
                score = float(evaluation_snapshot.aggregate_metrics.get("benchmark_score", 0.0))
                if score >= best_score:
                    best_score = score
                    best_checkpoint = checkpoint_manager.save(
                        model=model,
                        kind="best",
                        training_step=step,
                        evaluation_snapshot=evaluation_snapshot,
                        seed=config.seed,
                    )
                if step % config.checkpoint_frequency == 0:
                    checkpoint_history.append(
                        checkpoint_manager.save(
                            model=model,
                            kind="checkpoint",
                            training_step=step,
                            evaluation_snapshot=evaluation_snapshot,
                            seed=config.seed,
                        )
                    )
                logger.flush()
            return True

    model.learn(total_timesteps=config.total_timesteps, callback=TrainingProgressCallback())
    final_evaluation = evaluate_policy(model, config, training_step=config.total_timesteps, logger=logger)
    evaluation_history.append(final_evaluation)
    final_checkpoint = checkpoint_manager.save(
        model=model,
        kind="final",
        training_step=config.total_timesteps,
        evaluation_snapshot=final_evaluation,
        seed=config.seed,
    )
    checkpoint_history.append(final_checkpoint)
    if best_checkpoint is None:
        best_checkpoint = final_checkpoint
    logger.log_json("train/final_evaluation", final_evaluation.to_dict(), config.total_timesteps)
    logger.flush()
    if hasattr(train_env, "close"):
        train_env.close()
    return TrainingRunResult(
        config_snapshot=config_snapshot,
        total_timesteps=config.total_timesteps,
        evaluation_history=evaluation_history,
        checkpoint_history=checkpoint_history,
        best_checkpoint=best_checkpoint,
        final_checkpoint=final_checkpoint,
        resumed_from_checkpoint=resumed_from_checkpoint,
    )


def run_training(
    config: OpenBargainConfig | TrainingConfig,
    hooks: TrainingHooks | None = None,
) -> TrainingRunResult:
    """Backward-compatible training entry point that delegates to `train_ppo`."""
    if isinstance(config, TrainingConfig):
        training_config = config
    else:
        training_config = TrainingConfig(open_bargain_config=config)
    logger = None if hooks is None else hooks.logger
    return train_ppo(training_config, logger=logger)


__all__ = [
    "CheckpointManager",
    "CheckpointRecord",
    "EvaluationSnapshot",
    "JsonlTrainingLogger",
    "PPOCompatibleOpenBargainEnv",
    "PPOModelPolicyAdapter",
    "SimulationPolicy",
    "TrainingConfig",
    "TrainingHooks",
    "TrainingLogger",
    "TrainingRunResult",
    "evaluate_policy",
    "make_env",
    "make_env_thunk",
    "register_training_environment",
    "run_training",
    "train_ppo",
]
