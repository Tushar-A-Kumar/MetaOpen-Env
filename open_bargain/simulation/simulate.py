"""Simulation runner and rollout utilities for OpenBargain benchmark evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Protocol

from open_bargain.config import OpenBargainConfig
from open_bargain.env.env import OpenBargainEnv
from open_bargain.metrics.metrics import AggregateMetrics, EpisodeMetrics, MetricsEngine


DEBUG_MODE = False


def set_debug_mode(enabled: bool) -> None:
    """Enable or disable verbose simulation debugging output."""
    global DEBUG_MODE
    DEBUG_MODE = bool(enabled)


class SimulationPolicy(Protocol):
    """Protocol for interchangeable simulation-time bargaining policies."""

    def reset(self, *, agent_id: str, agent_ids: list[str], seed: int | None = None) -> None:
        """Reset policy internal state for a new episode."""
        raise NotImplementedError("Policy reset is not implemented.")

    def select_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Select an environment-compatible action from one agent observation."""
        raise NotImplementedError("Policy action selection is not implemented.")


PolicyHook = SimulationPolicy


def _validate_finite(value: float, field_name: str) -> None:
    """Validate a scalar is finite for deterministic benchmark exports."""
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


def _validate_json_safe_mapping(mapping: dict[str, Any], field_name: str) -> None:
    """Validate a mapping can be exported deterministically to JSON."""
    if not isinstance(mapping, dict):
        raise ValueError(f"{field_name} must be a dictionary.")
    _canonical_json_value(mapping)


def _validate_policy_action(
    *,
    action: dict[str, Any],
    active_agent_id: str,
    observation: dict[str, Any],
    agent_ids: list[str],
) -> None:
    """Fail fast for malformed policy actions before environment execution."""
    if not isinstance(action, dict):
        raise ValueError("Policy action must be a dictionary.")
    if action.get("agent_id") != active_agent_id:
        raise ValueError(
            "Policy returned mismatched agent_id. "
            f"expected={active_agent_id}, got={action.get('agent_id')}."
        )
    action_type = action.get("action_type")
    if not isinstance(action_type, str) or not action_type.strip():
        raise ValueError("Policy action_type must be a non-empty string.")
    valid_action_mask = observation.get("valid_action_mask")
    if not isinstance(valid_action_mask, dict):
        raise ValueError("Observation missing valid_action_mask dictionary.")
    if not valid_action_mask.get(action_type, False):
        raise ValueError(
            f"Policy selected illegal action '{action_type}' for agent '{active_agent_id}'."
        )
    payload = action.get("payload", {})
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("Policy action payload must be a dictionary when provided.")
    if action_type in ("propose", "counteroffer"):
        allocation = payload.get("allocation")
        if isinstance(allocation, list):
            raise ValueError("allocation must be dict, not list")
        if not isinstance(allocation, dict):
            raise ValueError(
                "Proposal and counteroffer actions require payload['allocation'] dictionaries."
            )
        public = observation.get("public", {})
        if not isinstance(public, dict):
            raise ValueError("Observation public section must be a dictionary.")
        remaining_resource = float(public.get("remaining_resource", 0.0))
        total_allocated = 0.0
        for agent_id, value in allocation.items():
            if agent_id not in agent_ids:
                raise ValueError(f"Policy allocation contains unknown agent_id '{agent_id}'.")
            parsed_value = float(value)
            _validate_finite(parsed_value, f"allocation[{agent_id}]")
            if parsed_value < 0.0:
                raise ValueError(
                    f"Policy allocation value must be non-negative. agent_id={agent_id}, value={parsed_value}."
                )
            total_allocated += parsed_value
        _validate_finite(total_allocated, "allocation_total")
        if total_allocated > remaining_resource:
            raise ValueError(
                "Policy allocation exceeds remaining_resource. "
                f"total_allocated={total_allocated}, remaining_resource={remaining_resource}."
            )


def _build_failed_episode_trace(
    *,
    episode_index: int,
    seed: int,
    agent_ids: list[str],
    error: Exception,
    metrics_engine: MetricsEngine,
    max_rounds: int,
) -> EpisodeTrace:
    """Construct a JSON-safe failure trace when a rollout cannot complete."""
    if not agent_ids:
        agent_ids = ["agent_0", "agent_1"]
    step_agent_id = agent_ids[0]
    failure_message = f"{type(error).__name__}: {error}"
    step_trace = StepTrace(
        step_index=0,
        acting_agent_id=step_agent_id,
        action={"agent_id": step_agent_id, "action_type": "reject", "payload": {}},
        observations={"failure": failure_message, "episode_index": episode_index, "seed": seed},
        rewards={agent_id: 0.0 for agent_id in agent_ids},
        terminated={agent_id: True for agent_id in agent_ids} | {"__all__": True},
        truncated={agent_id: False for agent_id in agent_ids} | {"__all__": False},
        info={"error": failure_message, "debug_mode": DEBUG_MODE},
    )
    episode_metrics = metrics_engine.evaluate_episode(
        agreement_reached=False,
        rounds_used=0,
        max_rounds=max_rounds,
        final_allocation=None,
    )
    return EpisodeTrace(
        episode_index=episode_index,
        seed=seed,
        agent_ids=agent_ids,
        steps=[step_trace],
        cumulative_rewards={agent_id: 0.0 for agent_id in agent_ids},
        total_rewards={agent_id: 0.0 for agent_id in agent_ids},
        agreement_reached=False,
        rounds_used=0,
        final_allocation=None,
        outcome={"error": failure_message, "termination_reason": "episode_failed"},
        episode_metrics=episode_metrics,
    )


def _policy_metadata_for(policy: SimulationPolicy) -> dict[str, Any]:
    """Build stable JSON-safe metadata for one policy instance."""
    metadata: dict[str, Any] = {"policy_class": policy.__class__.__name__}
    base_seed = getattr(policy, "_base_seed", None)
    if base_seed is not None:
        metadata["base_seed"] = int(base_seed)
    return metadata


def _build_policy_metadata(policies: dict[str, SimulationPolicy]) -> dict[str, Any]:
    """Build deterministic metadata payload for a policy roster."""
    return {
        "agent_ids": sorted(policies.keys()),
        "policies": {agent_id: _policy_metadata_for(policies[agent_id]) for agent_id in sorted(policies)},
    }


def _validate_policy_metadata_consistency(
    *,
    traces: list[EpisodeTrace],
    policy_metadata: dict[str, Any],
) -> None:
    """Validate policy metadata matches the agent roster captured in traces."""
    if not traces:
        return
    expected_agent_ids = list(traces[0].agent_ids)
    for index, trace in enumerate(traces[1:], start=1):
        if list(trace.agent_ids) != expected_agent_ids:
            raise ValueError(
                "SimulationResult traces must share one agent roster. "
                f"trace_index={index}, expected_agent_ids={expected_agent_ids}, got={trace.agent_ids}."
            )
    metadata_agent_ids = policy_metadata.get("agent_ids")
    if metadata_agent_ids is not None and list(metadata_agent_ids) != expected_agent_ids:
        raise ValueError(
            "policy_metadata.agent_ids must match trace agent_ids. "
            f"expected={expected_agent_ids}, got={metadata_agent_ids}."
        )
    metadata_policies = policy_metadata.get("policies")
    if metadata_policies is not None:
        if not isinstance(metadata_policies, dict):
            raise ValueError("policy_metadata.policies must be a dictionary when provided.")
        if set(metadata_policies.keys()) != set(expected_agent_ids):
            raise ValueError(
                "policy_metadata.policies keys must match trace agent_ids. "
                f"expected={sorted(expected_agent_ids)}, got={sorted(metadata_policies.keys())}."
            )


def _distribution_from_equal_split(
    *,
    agent_ids: list[str],
    total_resource: float,
    favored_agent_id: str | None = None,
    favored_share: float | None = None,
) -> dict[str, float]:
    """Create a deterministic allocation that sums to the supplied resource."""
    if not agent_ids:
        raise ValueError("agent_ids must not be empty for allocation generation.")
    if total_resource < 0.0:
        raise ValueError(f"total_resource must be non-negative. Got {total_resource}.")
    if len(agent_ids) == 1:
        return {agent_ids[0]: total_resource}
    equal_share = total_resource / len(agent_ids)
    allocation = {agent_id: equal_share for agent_id in agent_ids}
    if favored_agent_id is not None:
        if favored_agent_id not in allocation:
            raise ValueError(f"favored_agent_id '{favored_agent_id}' is not part of agent_ids.")
        share = equal_share if favored_share is None else max(0.0, min(total_resource, favored_share))
        allocation[favored_agent_id] = share
        remainder = total_resource - share
        other_agent_ids = [agent_id for agent_id in agent_ids if agent_id != favored_agent_id]
        if other_agent_ids:
            other_share = remainder / len(other_agent_ids)
            for agent_id in other_agent_ids:
                allocation[agent_id] = other_share
    residual = total_resource - sum(allocation.values())
    allocation[agent_ids[-1]] += residual
    return allocation


def _equal_share_score(*, offer: dict[str, float], self_agent_id: str, remaining_resource: float) -> float:
    """Compute a deterministic fairness score relative to an equal split."""
    if remaining_resource <= 0.0:
        return 1.0
    if self_agent_id not in offer:
        return 0.0
    equal_share = remaining_resource / len(offer)
    self_share = float(offer[self_agent_id])
    imbalance = abs(self_share - equal_share) / remaining_resource
    return max(0.0, 1.0 - imbalance)


def _self_share_score(*, offer: dict[str, float], self_agent_id: str, remaining_resource: float) -> float:
    """Compute a deterministic self-favoring score for one observed offer."""
    if remaining_resource <= 0.0:
        return 0.0
    if self_agent_id not in offer:
        return 0.0
    self_share = float(offer[self_agent_id])
    return max(0.0, min(1.0, self_share / remaining_resource))


@dataclass(slots=True, frozen=True)
class StepTrace:
    """One simulation step trace item with observations, action, rewards, and flags."""

    step_index: int
    acting_agent_id: str
    action: dict[str, Any]
    observations: dict[str, Any]
    rewards: dict[str, float]
    terminated: dict[str, bool]
    truncated: dict[str, bool]
    info: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate step trace fields for fail-fast benchmark exports."""
        if self.step_index < 0:
            raise ValueError(f"StepTrace.step_index must be non-negative. Got {self.step_index}.")
        if not self.acting_agent_id.strip():
            raise ValueError("StepTrace.acting_agent_id must be a non-empty string.")
        _validate_json_safe_mapping(self.action, "StepTrace.action")
        _validate_json_safe_mapping(self.observations, "StepTrace.observations")
        _validate_json_safe_mapping(self.rewards, "StepTrace.rewards")
        _validate_json_safe_mapping(self.terminated, "StepTrace.terminated")
        _validate_json_safe_mapping(self.truncated, "StepTrace.truncated")
        _validate_json_safe_mapping(self.info, "StepTrace.info")

    def to_dict(self) -> dict[str, Any]:
        """Serialize step trace to deterministic JSON-safe dictionary."""
        return {
            "step_index": self.step_index,
            "acting_agent_id": self.acting_agent_id,
            "action": dict(self.action),
            "observations": dict(self.observations),
            "rewards": dict(self.rewards),
            "terminated": dict(self.terminated),
            "truncated": dict(self.truncated),
            "info": dict(self.info),
        }


@dataclass(slots=True, frozen=True)
class EpisodeTrace:
    """Complete episode rollout trace with rewards, outcome, and computed metrics."""

    episode_index: int
    seed: int
    agent_ids: list[str]
    steps: list[StepTrace]
    cumulative_rewards: dict[str, float]
    total_rewards: dict[str, float]
    agreement_reached: bool
    rounds_used: int
    final_allocation: dict[str, float] | None
    outcome: dict[str, Any] | None
    episode_metrics: EpisodeMetrics

    def __post_init__(self) -> None:
        """Validate episode trace structure and internal consistency."""
        if self.episode_index < 0:
            raise ValueError(f"EpisodeTrace.episode_index must be non-negative. Got {self.episode_index}.")
        if self.seed < 0:
            raise ValueError(f"EpisodeTrace.seed must be non-negative. Got {self.seed}.")
        if not self.agent_ids:
            raise ValueError("EpisodeTrace.agent_ids must not be empty.")
        if len(set(self.agent_ids)) != len(self.agent_ids):
            raise ValueError("EpisodeTrace.agent_ids must be unique.")
        if not self.steps:
            raise ValueError("EpisodeTrace.steps must not be empty.")
        for index, step in enumerate(self.steps):
            if step.step_index != index:
                raise ValueError(
                    "EpisodeTrace.steps must be sequentially ordered. "
                    f"expected_step_index={index}, got={step.step_index}."
                )
        for agent_id, value in self.cumulative_rewards.items():
            _validate_finite(float(value), f"EpisodeTrace.cumulative_rewards[{agent_id}]")
        for agent_id, value in self.total_rewards.items():
            _validate_finite(float(value), f"EpisodeTrace.total_rewards[{agent_id}]")
        _validate_json_safe_mapping(self.cumulative_rewards, "EpisodeTrace.cumulative_rewards")
        _validate_json_safe_mapping(self.total_rewards, "EpisodeTrace.total_rewards")
        if self.final_allocation is not None:
            _validate_json_safe_mapping(self.final_allocation, "EpisodeTrace.final_allocation")
        if self.outcome is not None:
            _validate_json_safe_mapping(self.outcome, "EpisodeTrace.outcome")

    def to_dict(self) -> dict[str, Any]:
        """Serialize episode trace to deterministic JSON-safe dictionary."""
        return {
            "episode_index": self.episode_index,
            "seed": self.seed,
            "agent_ids": list(self.agent_ids),
            "steps": [step.to_dict() for step in self.steps],
            "cumulative_rewards": dict(self.cumulative_rewards),
            "total_rewards": dict(self.total_rewards),
            "agreement_reached": self.agreement_reached,
            "rounds_used": self.rounds_used,
            "final_allocation": None if self.final_allocation is None else dict(self.final_allocation),
            "outcome": None if self.outcome is None else dict(self.outcome),
            "episode_metrics": self.episode_metrics.to_dict(),
        }


@dataclass(slots=True)
class SimulationResult:
    """Simulation output containing episode traces and aggregate benchmark metrics."""

    traces: list[EpisodeTrace] = field(default_factory=list)
    aggregate_metrics: AggregateMetrics | None = None
    report: dict[str, Any] = field(default_factory=dict)
    policy_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate top-level simulation result payloads."""
        _validate_json_safe_mapping(self.report, "SimulationResult.report")
        _validate_json_safe_mapping(self.policy_metadata, "SimulationResult.policy_metadata")
        _validate_policy_metadata_consistency(
            traces=self.traces,
            policy_metadata=self.policy_metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize simulation result for report export and debugging."""
        return {
            "traces": [trace.to_dict() for trace in self.traces],
            "aggregate_metrics": None if self.aggregate_metrics is None else self.aggregate_metrics.to_dict(),
            "report": dict(self.report),
        }

    def export_aggregate_metrics_summary(self) -> dict[str, Any]:
        """Export JSON-safe aggregate benchmark metrics with policy metadata."""
        if self.aggregate_metrics is None:
            raise ValueError("aggregate_metrics must be available for benchmark export.")
        return {
            "aggregate_metrics": _canonical_json_value(self.aggregate_metrics.to_dict()),
            "benchmark_score": self.aggregate_metrics.benchmark_score,
            "policy_metadata": _canonical_json_value(self.policy_metadata),
        }

    def export_episode_traces(self) -> list[dict[str, Any]]:
        """Export episode traces as JSON-safe dictionaries with validation."""
        exported_traces = [_canonical_json_value(trace.to_dict()) for trace in self.traces]
        for index, trace in enumerate(exported_traces):
            if not isinstance(trace, dict):
                raise ValueError(f"Exported trace at index {index} must be a dictionary.")
            _validate_json_safe_mapping(trace, f"exported_trace[{index}]")
        return exported_traces

    def export_reward_summary(self) -> dict[str, Any]:
        """Export deterministic benchmark reward summaries."""
        total_rewards = [sum(trace.total_rewards.values()) for trace in self.traces]
        if any(not math.isfinite(value) for value in total_rewards):
            raise ValueError("Reward summaries must remain finite.")
        if not total_rewards:
            return {
                "episode_count": 0,
                "total_reward": {"mean": 0.0, "min": 0.0, "max": 0.0},
                "policy_metadata": _canonical_json_value(self.policy_metadata),
            }
        return {
            "episode_count": len(total_rewards),
            "total_reward": {
                "mean": sum(total_rewards) / len(total_rewards),
                "min": min(total_rewards),
                "max": max(total_rewards),
            },
            "policy_metadata": _canonical_json_value(self.policy_metadata),
        }

    def export_benchmark_artifact(self) -> dict[str, Any]:
        """Export a complete JSON-safe benchmark artifact payload."""
        aggregate_summary = self.export_aggregate_metrics_summary()
        reward_summary = self.export_reward_summary()
        return {
            "policy_metadata": aggregate_summary["policy_metadata"],
            "aggregate_metrics": aggregate_summary["aggregate_metrics"],
            "benchmark_score": aggregate_summary["benchmark_score"],
            "reward_summary": reward_summary,
            "episode_traces": self.export_episode_traces(),
            "report": _canonical_json_value(self.report),
        }


class _HeuristicNegotiationPolicy:
    """Shared scaffolding for deterministic heuristic baseline policies."""

    def __init__(self, seed: int = 0) -> None:
        if seed < 0:
            raise ValueError(f"seed must be non-negative. Got {seed}.")
        self._base_seed = seed
        self._rng = random.Random(seed)
        self._agent_id: str | None = None
        self._agent_ids: list[str] = []
        self._profile_summary: dict[str, float] = {
            "greed_sensitivity": 0.0,
            "fairness_sensitivity": 0.0,
            "urgency_sensitivity": 0.0,
        }

    def reset(self, *, agent_id: str, agent_ids: list[str], seed: int | None = None) -> None:
        if not agent_id.strip():
            raise ValueError("agent_id must be non-empty.")
        if agent_id not in agent_ids:
            raise ValueError(f"agent_id '{agent_id}' must exist in agent_ids {agent_ids}.")
        effective_seed = self._base_seed if seed is None else seed
        if effective_seed < 0:
            raise ValueError(f"seed must be non-negative. Got {effective_seed}.")
        self._agent_id = agent_id
        self._agent_ids = list(agent_ids)
        self._rng = random.Random(effective_seed)

    def select_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self._agent_id is None:
            raise RuntimeError("Policy must be reset before select_action is called.")
        valid_action_mask = observation.get("valid_action_mask")
        if not isinstance(valid_action_mask, dict):
            raise ValueError("Observation missing valid_action_mask dictionary.")
        public = observation.get("public", {})
        if not isinstance(public, dict):
            raise ValueError("Observation public section must be a dictionary.")
        private = observation.get("private", {})
        if not isinstance(private, dict):
            raise ValueError("Observation private section must be a dictionary.")
        summary = observation.get("summary", {})
        if not isinstance(summary, dict):
            raise ValueError("Observation summary section must be a dictionary.")
        self._profile_summary = self._read_private_profile(private)
        action_type = self._choose_action_type(
            observation=observation,
            valid_action_mask=valid_action_mask,
            public=public,
            private=private,
            summary=summary,
        )
        if action_type not in valid_action_mask or not valid_action_mask[action_type]:
            raise ValueError(f"Heuristic policy selected illegal action '{action_type}'.")
        payload: dict[str, Any] = {}
        if action_type in ("propose", "counteroffer"):
            payload["allocation"] = self._build_allocation(public=public)
        action = {
            "agent_id": self._agent_id,
            "action_type": action_type,
            "payload": payload,
        }
        _validate_policy_action(
            action=action,
            active_agent_id=self._agent_id,
            observation=observation,
            agent_ids=self._agent_ids,
        )
        return action

    def _choose_action_type(
        self,
        *,
        observation: dict[str, Any],
        valid_action_mask: dict[str, bool],
        public: dict[str, Any],
        private: dict[str, Any],
        summary: dict[str, Any],
    ) -> str:
        raise NotImplementedError

    def _build_allocation(self, *, public: dict[str, Any]) -> dict[str, float]:
        raise NotImplementedError

    def _read_private_profile(self, private: dict[str, Any]) -> dict[str, float]:
        profile = private.get("private_utility_profile_summary")
        if not isinstance(profile, dict):
            return {
                "greed_sensitivity": 0.0,
                "fairness_sensitivity": 0.0,
                "urgency_sensitivity": 0.0,
            }
        return {
            "greed_sensitivity": float(profile.get("greed_sensitivity", 0.0)),
            "fairness_sensitivity": float(profile.get("fairness_sensitivity", 0.0)),
            "urgency_sensitivity": float(profile.get("urgency_sensitivity", 0.0)),
        }

    def _current_offer(self, public: dict[str, Any]) -> dict[str, float] | None:
        raw_offer = public.get("current_public_offer")
        if raw_offer is None:
            return None
        if not isinstance(raw_offer, dict):
            raise ValueError("public.current_public_offer must be a dictionary when provided.")
        return {str(agent_id): float(value) for agent_id, value in raw_offer.items()}

    def _remaining_resource(self, public: dict[str, Any]) -> float:
        remaining_resource = float(public.get("remaining_resource", 0.0))
        _validate_finite(remaining_resource, "public.remaining_resource")
        if remaining_resource < 0.0:
            raise ValueError(f"public.remaining_resource must be non-negative. Got {remaining_resource}.")
        return remaining_resource

    def _is_proposer(self, public: dict[str, Any]) -> bool:
        return public.get("active_proposer_id") == self._agent_id

    def _selection_order(self, valid_action_mask: dict[str, bool], preferred_actions: list[str]) -> str:
        for action_type in preferred_actions:
            if valid_action_mask.get(action_type, False):
                return action_type
        raise ValueError("No valid action available for heuristic policy.")

    def _balanced_allocation(self, *, total_resource: float) -> dict[str, float]:
        return _distribution_from_equal_split(
            agent_ids=self._agent_ids,
            total_resource=total_resource,
        )

    def _self_favoring_allocation(self, *, total_resource: float, favored_share_ratio: float) -> dict[str, float]:
        if self._agent_id is None:
            raise RuntimeError("Policy must be reset before allocation construction.")
        favored_share = total_resource * max(0.0, min(1.0, favored_share_ratio))
        return _distribution_from_equal_split(
            agent_ids=self._agent_ids,
            total_resource=total_resource,
            favored_agent_id=self._agent_id,
            favored_share=favored_share,
        )


class FairNegotiationPolicy(_HeuristicNegotiationPolicy):
    """Deterministic baseline that favors balanced bargaining outcomes."""

    def _choose_action_type(
        self,
        *,
        observation: dict[str, Any],
        valid_action_mask: dict[str, bool],
        public: dict[str, Any],
        private: dict[str, Any],
        summary: dict[str, Any],
    ) -> str:
        if self._is_proposer(public):
            return self._selection_order(valid_action_mask, ["propose", "counteroffer"])
        offer = self._current_offer(public)
        if offer is None:
            return self._selection_order(valid_action_mask, ["reject", "accept", "counteroffer"])
        remaining_resource = self._remaining_resource(public)
        fairness_score = _equal_share_score(
            offer=offer,
            self_agent_id=self._agent_id or "",
            remaining_resource=remaining_resource,
        )
        threshold = min(0.95, 0.70 + 0.10 * max(0.0, self._profile_summary["fairness_sensitivity"]))
        if fairness_score >= threshold:
            return self._selection_order(valid_action_mask, ["accept", "counteroffer", "reject"])
        return self._selection_order(valid_action_mask, ["counteroffer", "reject", "accept"])

    def _build_allocation(self, *, public: dict[str, Any]) -> dict[str, float]:
        remaining_resource = self._remaining_resource(public)
        return self._balanced_allocation(total_resource=remaining_resource)


class GreedyNegotiationPolicy(_HeuristicNegotiationPolicy):
    """Deterministic baseline that favors self-interested bargaining outcomes."""

    def _choose_action_type(
        self,
        *,
        observation: dict[str, Any],
        valid_action_mask: dict[str, bool],
        public: dict[str, Any],
        private: dict[str, Any],
        summary: dict[str, Any],
    ) -> str:
        if self._is_proposer(public):
            return self._selection_order(valid_action_mask, ["propose", "counteroffer"])
        offer = self._current_offer(public)
        if offer is None:
            return self._selection_order(valid_action_mask, ["counteroffer", "reject", "accept"])
        remaining_resource = self._remaining_resource(public)
        self_share_ratio = _self_share_score(
            offer=offer,
            self_agent_id=self._agent_id or "",
            remaining_resource=remaining_resource,
        )
        accept_threshold = min(0.95, 0.50 + 0.20 * max(0.0, self._profile_summary["greed_sensitivity"]))
        if self_share_ratio >= accept_threshold:
            return self._selection_order(valid_action_mask, ["accept", "counteroffer", "reject"])
        return self._selection_order(valid_action_mask, ["counteroffer", "reject", "accept"])

    def _build_allocation(self, *, public: dict[str, Any]) -> dict[str, float]:
        remaining_resource = self._remaining_resource(public)
        favored_share_ratio = min(0.95, 0.55 + 0.30 * max(0.0, self._profile_summary["greed_sensitivity"]))
        return self._self_favoring_allocation(
            total_resource=remaining_resource,
            favored_share_ratio=favored_share_ratio,
        )


class RandomNegotiationPolicy:
    """Deterministic random baseline policy using action masks from observations."""

    def __init__(self, seed: int = 0) -> None:
        """Initialize random baseline policy with deterministic seed."""
        if seed < 0:
            raise ValueError(f"seed must be non-negative. Got {seed}.")
        self._base_seed = seed
        self._rng = random.Random(seed)
        self._agent_id: str | None = None
        self._agent_ids: list[str] = []

    def reset(self, *, agent_id: str, agent_ids: list[str], seed: int | None = None) -> None:
        """Reset policy state and deterministic RNG stream for one episode."""
        if not agent_id.strip():
            raise ValueError("agent_id must be non-empty.")
        if agent_id not in agent_ids:
            raise ValueError(
                f"agent_id '{agent_id}' must exist in agent_ids {agent_ids}."
            )
        self._agent_id = agent_id
        self._agent_ids = list(agent_ids)
        effective_seed = self._base_seed if seed is None else seed
        if effective_seed < 0:
            raise ValueError(f"seed must be non-negative. Got {effective_seed}.")
        self._rng = random.Random(effective_seed)

    def select_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Sample a valid action from observation-provided action mask."""
        if self._agent_id is None:
            raise RuntimeError("Policy must be reset before select_action is called.")
        valid_action_mask = observation.get("valid_action_mask")
        if not isinstance(valid_action_mask, dict):
            raise ValueError("Observation missing valid_action_mask dictionary.")
        valid_actions = [name for name, is_valid in valid_action_mask.items() if bool(is_valid)]
        if not valid_actions:
            raise ValueError("No valid actions available for RandomNegotiationPolicy.")
        action_type = self._rng.choice(valid_actions)
        payload: dict[str, Any] = {}
        if action_type in ("propose", "counteroffer"):
            payload["allocation"] = self._sample_allocation(observation)
        return {
            "agent_id": self._agent_id,
            "action_type": action_type,
            "payload": payload,
        }

    def _sample_allocation(self, observation: dict[str, Any]) -> dict[str, float]:
        """Sample deterministic random allocation summing to remaining resource."""
        if not self._agent_ids:
            raise RuntimeError("agent_ids are not initialized for allocation sampling.")
        public = observation.get("public", {})
        if not isinstance(public, dict):
            raise ValueError("Observation public section must be a dictionary.")
        remaining_resource = float(public.get("remaining_resource", 0.0))
        if remaining_resource < 0.0:
            raise ValueError(f"remaining_resource must be non-negative. Got {remaining_resource}.")
        if len(self._agent_ids) == 1:
            return {self._agent_ids[0]: remaining_resource}
        cuts = sorted(self._rng.random() for _ in range(len(self._agent_ids) - 1))
        points = [0.0, *cuts, 1.0]
        allocation: dict[str, float] = {}
        for index, agent_id in enumerate(self._agent_ids):
            share = max(0.0, points[index + 1] - points[index])
            allocation[agent_id] = share * remaining_resource
        # Ensure numerical conservation by assigning residual to the last agent.
        residual = remaining_resource - sum(allocation.values())
        allocation[self._agent_ids[-1]] += residual
        return allocation


def simulate_episode(
    *,
    env: OpenBargainEnv,
    policies: dict[str, SimulationPolicy],
    metrics_engine: MetricsEngine,
    episode_index: int,
    seed: int,
    reset_options: dict[str, Any] | None = None,
) -> EpisodeTrace:
    """Run one deterministic simulation episode and return full trace output."""
    observations, _ = env.reset(seed=seed, options=reset_options)
    agent_ids = sorted(observations.keys())
    if not agent_ids:
        raise RuntimeError("Environment reset returned no agent observations.")
    for index, agent_id in enumerate(agent_ids):
        if agent_id not in policies:
            raise ValueError(f"Missing policy for agent_id '{agent_id}'.")
        policies[agent_id].reset(agent_id=agent_id, agent_ids=agent_ids, seed=seed + index)
    step_traces: list[StepTrace] = []
    cumulative_rewards = {agent_id: 0.0 for agent_id in agent_ids}
    step_index = 0
    while True:
        try:
            active_agent_id = _extract_active_agent_id(observations, agent_ids)
            action = policies[active_agent_id].select_action(observations[active_agent_id])
            _validate_policy_action(
                action=action,
                active_agent_id=active_agent_id,
                observation=observations[active_agent_id],
                agent_ids=agent_ids,
            )
            if DEBUG_MODE:
                print(f"[OpenBargain] episode={episode_index} step={step_index} agent={active_agent_id} action={action.get('action_type')}")
            next_observations, rewards, terminated, truncated, info = env.step(action)
            # Safe extraction of active agent for metrics
            info["active_agent"] = info.get("active_agent") or info.get("state_summary", {}).get("active_proposer_id") or active_agent_id
            for agent_id, value in rewards.items():
                parsed_reward = float(value)
                _validate_finite(parsed_reward, f"reward[{agent_id}]")
                cumulative_rewards[agent_id] = cumulative_rewards.get(agent_id, 0.0) + parsed_reward
            step_traces.append(
                StepTrace(
                    step_index=step_index,
                    acting_agent_id=active_agent_id,
                    action=dict(action),
                    observations=dict(next_observations),
                    rewards=dict(rewards),
                    terminated=dict(terminated),
                    truncated=dict(truncated),
                    info=dict(info),
                )
            )
            observations = next_observations
            step_index += 1
            if bool(terminated.get("__all__", False)) or bool(truncated.get("__all__", False)) or step_index > 200:
                break
        except Exception as e:
            return _build_failed_episode_trace(
                episode_index=episode_index,
                seed=seed,
                agent_ids=agent_ids,
                error=e,
                metrics_engine=metrics_engine,
                max_rounds=env.config.environment.max_negotiation_rounds,
            )
    final_info = step_traces[-1].info if step_traces else {}
    state_summary = final_info.get("state_summary", {})
    if not isinstance(state_summary, dict):
        raise ValueError("state_summary in env info must be a dictionary.")
    outcome = final_info.get("outcome")
    if outcome is not None and not isinstance(outcome, dict):
        raise ValueError("outcome in env info must be a dictionary when present.")
    agreement_reached = bool(state_summary.get("agreement_reached", False))
    rounds_used = int(state_summary.get("round", 0))
    final_allocation: dict[str, float] | None = None
    if isinstance(outcome, dict):
        raw_allocation = outcome.get("agreed_allocation")
        if isinstance(raw_allocation, dict):
            final_allocation = {str(k): float(v) for k, v in raw_allocation.items()}
    episode_metrics = metrics_engine.evaluate_episode(
        agreement_reached=agreement_reached,
        rounds_used=rounds_used,
        max_rounds=env.config.environment.max_negotiation_rounds,
        final_allocation=final_allocation,
    )
    return EpisodeTrace(
        episode_index=episode_index,
        seed=seed,
        agent_ids=agent_ids,
        steps=step_traces,
        cumulative_rewards=dict(cumulative_rewards),
        total_rewards=dict(cumulative_rewards),
        agreement_reached=agreement_reached,
        rounds_used=rounds_used,
        final_allocation=final_allocation,
        outcome=None if outcome is None else dict(outcome),
        episode_metrics=episode_metrics,
    )


def simulate_episodes(
    env: OpenBargainEnv,
    config: OpenBargainConfig | None = None,
    policy_hooks: list[PolicyHook] | None = None,
) -> SimulationResult:
    """Run deterministic multi-episode simulation and return benchmark outputs."""
    if config is None:
        try:
            from open_bargain.config import OpenBargainConfig
            config = OpenBargainConfig.default()
        except ImportError:
            pass
    if policy_hooks is None:
        raise TypeError("simulate_episodes() missing required argument 'policy_hooks'. Provide a list of policies.")

    sim_cfg = config.get("simulation", {}) if isinstance(config, dict) else getattr(config, "simulation", config)

    def cfg(key, default=None):
        if isinstance(sim_cfg, dict):
            return sim_cfg.get(key, default)
        return getattr(sim_cfg, key, default)

    num_episodes = cfg("num_episodes")
    if num_episodes is None or num_episodes <= 0:
        raise ValueError(f"Invalid num_episodes: {num_episodes}")

    if len(policy_hooks) < 2:
        raise ValueError("At least two policy hooks are required for bargaining simulation.")
    
    # We also safely extract total_resource_amount avoiding attribute errors
    env_cfg = config.get("environment", {}) if isinstance(config, dict) else getattr(config, "environment", config)
    total_resource_amount = env_cfg.get("total_resource_amount", 100.0) if isinstance(env_cfg, dict) else getattr(env_cfg, "total_resource_amount", 100.0)

    metrics_engine = MetricsEngine(
        reference_total_resource=total_resource_amount,
    )
    traces: list[EpisodeTrace] = []
    base_seed = cfg("default_random_seed", 0)
    policy_metadata = _build_policy_metadata(
        {f"agent_{index}": policy for index, policy in enumerate(policy_hooks)}
    )
    for episode_index in range(num_episodes):
        episode_seed = base_seed + episode_index
        reset_options = {"agent_ids": list(policy_metadata["agent_ids"])}
        policies_by_agent = {
            f"agent_{index}": policy for index, policy in enumerate(policy_hooks)
        }
        try:
            trace = simulate_episode(
                env=env,
                policies=policies_by_agent,
                metrics_engine=metrics_engine,
                episode_index=episode_index,
                seed=episode_seed,
                reset_options=reset_options,
            )
        except Exception as error:
            if DEBUG_MODE:
                print(f"[OpenBargain] episode={episode_index} failed: {type(error).__name__}: {error}")
            max_rounds = env_cfg.get("max_negotiation_rounds", 10) if isinstance(env_cfg, dict) else getattr(env_cfg, "max_negotiation_rounds", 10)
            trace = _build_failed_episode_trace(
                episode_index=episode_index,
                seed=episode_seed,
                agent_ids=list(policy_metadata["agent_ids"]),
                error=error,
                metrics_engine=metrics_engine,
                max_rounds=max_rounds,
            )
        traces.append(trace)
    aggregate_metrics = metrics_engine.aggregate([trace.episode_metrics for trace in traces])
    report = _build_simulation_report(traces=traces, aggregate_metrics=aggregate_metrics)
    return SimulationResult(
        traces=traces,
        aggregate_metrics=aggregate_metrics,
        report=report,
        policy_metadata=policy_metadata,
    )


def _extract_active_agent_id(observations: dict[str, Any], agent_ids: list[str]) -> str:
    """Extract active proposer id from observations with deterministic validation."""
    if not observations:
        raise ValueError("observations must not be empty.")
    first_agent = agent_ids[0]
    first_obs = observations.get(first_agent)
    if not isinstance(first_obs, dict):
        raise ValueError("Agent observation must be a dictionary.")
    public = first_obs.get("public")
    if not isinstance(public, dict):
        raise ValueError("Observation must contain public dictionary.")
    active_agent_id = public.get("active_proposer_id")
    if not isinstance(active_agent_id, str) or not active_agent_id.strip():
        raise ValueError("public.active_proposer_id must be a non-empty string.")
    if active_agent_id not in observations:
        raise ValueError(
            f"Active proposer '{active_agent_id}' missing from observations keys."
        )
    return active_agent_id


def _build_simulation_report(
    *,
    traces: list[EpisodeTrace],
    aggregate_metrics: AggregateMetrics,
) -> dict[str, Any]:
    """Build deterministic benchmark summary report from traces and aggregate metrics."""
    reward_totals = [sum(trace.total_rewards.values()) for trace in traces]
    mean_total_reward = (sum(reward_totals) / len(reward_totals)) if reward_totals else 0.0
    return {
        "episode_count": len(traces),
        "aggregate_metrics": aggregate_metrics.to_dict(),
        "reward_summary": {
            "mean_total_reward": mean_total_reward,
            "min_total_reward": min(reward_totals) if reward_totals else 0.0,
            "max_total_reward": max(reward_totals) if reward_totals else 0.0,
        },
        "agreement_rate": aggregate_metrics.agreement_rate,
        "benchmark_score": aggregate_metrics.benchmark_score,
    }


def build_policy_comparison_summary(
    results_by_policy: dict[str, SimulationResult],
) -> dict[str, Any]:
    """Build a deterministic multi-policy benchmark comparison summary."""
    if not results_by_policy:
        raise ValueError("results_by_policy must not be empty.")
    rows: list[dict[str, Any]] = []
    for policy_name in sorted(results_by_policy):
        result = results_by_policy[policy_name]
        if result.aggregate_metrics is None:
            raise ValueError(
                f"SimulationResult for policy '{policy_name}' is missing aggregate metrics."
            )
        _validate_policy_metadata_consistency(
            traces=result.traces,
            policy_metadata=result.policy_metadata,
        )
        aggregate = result.aggregate_metrics
        rows.append(
            {
                "policy_name": policy_name,
                "agreement_rate": aggregate.agreement_rate,
                "fairness_score": aggregate.average_fairness_index,
                "efficiency_score": aggregate.average_efficiency_score,
                "benchmark_score": aggregate.benchmark_score,
                "reward_summary": result.export_reward_summary(),
                "policy_metadata": _canonical_json_value(result.policy_metadata),
            }
        )
    ranked_rows = sorted(rows, key=lambda row: (-row["benchmark_score"], row["policy_name"]))
    best_score = ranked_rows[0]["benchmark_score"]
    for rank, row in enumerate(ranked_rows, start=1):
        row["rank"] = rank
        row["score_gap_to_best"] = best_score - row["benchmark_score"]
    policy_count = len(ranked_rows)
    return {
        "policy_count": policy_count,
        "ranked_policies": ranked_rows,
        "top_policy": ranked_rows[0]["policy_name"],
        "summary": {
            "agreement_rate_mean": sum(row["agreement_rate"] for row in ranked_rows) / policy_count,
            "fairness_score_mean": sum(row["fairness_score"] for row in ranked_rows) / policy_count,
            "efficiency_score_mean": sum(row["efficiency_score"] for row in ranked_rows) / policy_count,
            "benchmark_score_mean": sum(row["benchmark_score"] for row in ranked_rows) / policy_count,
        },
    }
