"""Metrics and analytics layer for OpenBargain benchmark evaluation."""

from dataclasses import dataclass
import math
from typing import Any

from open_bargain.env.state import NegotiationOutcome


def _validate_finite(value: float, field_name: str) -> None:
    """Validate metric scalar is finite."""
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite. Got {value}.")


def _validate_unit_interval(value: float, field_name: str) -> None:
    """Validate metric scalar lies in [0, 1]."""
    _validate_finite(value, field_name)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]. Got {value}.")


@dataclass(slots=True, frozen=True)
class EpisodeMetrics:
    """Per-episode metric result for one bargaining trajectory."""

    agreement_reached: bool
    fairness_index: float
    exploitability_score: float
    rounds_used: int
    agreement_efficiency_score: float
    social_welfare_score: float = 0.0

    def __post_init__(self) -> None:
        """Validate episode metric bounds and consistency."""
        _validate_unit_interval(self.fairness_index, "EpisodeMetrics.fairness_index")
        _validate_unit_interval(
            self.exploitability_score,
            "EpisodeMetrics.exploitability_score",
        )
        _validate_unit_interval(
            self.agreement_efficiency_score,
            "EpisodeMetrics.agreement_efficiency_score",
        )
        _validate_unit_interval(self.social_welfare_score, "EpisodeMetrics.social_welfare_score")
        if self.rounds_used < 0:
            raise ValueError(f"EpisodeMetrics.rounds_used must be non-negative. Got {self.rounds_used}.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize episode metrics to deterministic JSON-safe dictionary."""
        return {
            "agreement_reached": self.agreement_reached,
            "fairness_index": self.fairness_index,
            "exploitability_score": self.exploitability_score,
            "rounds_used": self.rounds_used,
            "agreement_efficiency_score": self.agreement_efficiency_score,
            "social_welfare_score": self.social_welfare_score,
        }


@dataclass(slots=True, frozen=True)
class AggregateMetrics:
    """Aggregated benchmark metrics across multiple episodes."""

    agreement_rate: float
    average_fairness_index: float
    average_exploitability_score: float
    average_rounds_to_agreement: float
    average_efficiency_score: float
    episode_count: int
    fairness_std_dev: float = 0.0
    exploitability_std_dev: float = 0.0
    rounds_to_agreement_std_dev: float = 0.0
    efficiency_std_dev: float = 0.0
    average_social_welfare_score: float = 0.0
    benchmark_score: float = 0.0

    def __post_init__(self) -> None:
        """Validate aggregate metric bounds and count values."""
        _validate_unit_interval(self.agreement_rate, "AggregateMetrics.agreement_rate")
        _validate_unit_interval(
            self.average_fairness_index,
            "AggregateMetrics.average_fairness_index",
        )
        _validate_unit_interval(
            self.average_exploitability_score,
            "AggregateMetrics.average_exploitability_score",
        )
        _validate_unit_interval(
            self.average_efficiency_score,
            "AggregateMetrics.average_efficiency_score",
        )
        _validate_unit_interval(
            self.average_social_welfare_score,
            "AggregateMetrics.average_social_welfare_score",
        )
        _validate_unit_interval(self.benchmark_score, "AggregateMetrics.benchmark_score")
        _validate_finite(
            self.average_rounds_to_agreement,
            "AggregateMetrics.average_rounds_to_agreement",
        )
        _validate_finite(self.fairness_std_dev, "AggregateMetrics.fairness_std_dev")
        _validate_finite(self.exploitability_std_dev, "AggregateMetrics.exploitability_std_dev")
        _validate_finite(
            self.rounds_to_agreement_std_dev,
            "AggregateMetrics.rounds_to_agreement_std_dev",
        )
        _validate_finite(self.efficiency_std_dev, "AggregateMetrics.efficiency_std_dev")
        if self.average_rounds_to_agreement < 0:
            raise ValueError(
                "AggregateMetrics.average_rounds_to_agreement must be non-negative."
            )
        if self.fairness_std_dev < 0 or self.exploitability_std_dev < 0:
            raise ValueError("Aggregate fairness/exploitability std dev must be non-negative.")
        if self.rounds_to_agreement_std_dev < 0 or self.efficiency_std_dev < 0:
            raise ValueError("Aggregate rounds/efficiency std dev must be non-negative.")
        if self.episode_count <= 0:
            raise ValueError(f"AggregateMetrics.episode_count must be > 0. Got {self.episode_count}.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize aggregate metrics to deterministic JSON-safe dictionary."""
        return {
            "agreement_rate": self.agreement_rate,
            "average_fairness_index": self.average_fairness_index,
            "average_exploitability_score": self.average_exploitability_score,
            "average_rounds_to_agreement": self.average_rounds_to_agreement,
            "average_efficiency_score": self.average_efficiency_score,
            "fairness_std_dev": self.fairness_std_dev,
            "exploitability_std_dev": self.exploitability_std_dev,
            "rounds_to_agreement_std_dev": self.rounds_to_agreement_std_dev,
            "efficiency_std_dev": self.efficiency_std_dev,
            "average_social_welfare_score": self.average_social_welfare_score,
            "benchmark_score": self.benchmark_score,
            "episode_count": self.episode_count,
        }


@dataclass(slots=True, frozen=True)
class BargainingMetrics:
    """Backward-compatible aggregate benchmark metrics container."""

    agreement_rate: float
    fairness_index: float
    exploitability_score: float
    round_efficiency: float

    def __post_init__(self) -> None:
        """Validate legacy aggregate metric values."""
        _validate_unit_interval(self.agreement_rate, "BargainingMetrics.agreement_rate")
        _validate_unit_interval(self.fairness_index, "BargainingMetrics.fairness_index")
        _validate_unit_interval(
            self.exploitability_score,
            "BargainingMetrics.exploitability_score",
        )
        _validate_unit_interval(self.round_efficiency, "BargainingMetrics.round_efficiency")

    @classmethod
    def from_episode_traces(cls, traces: list[dict[str, object]]) -> "BargainingMetrics":
        """Compute compatibility metrics from episode trace dictionaries."""
        aggregate = MetricsEngine().aggregate_from_episode_traces(traces)
        return cls(
            agreement_rate=aggregate.agreement_rate,
            fairness_index=aggregate.average_fairness_index,
            exploitability_score=aggregate.average_exploitability_score,
            round_efficiency=aggregate.average_efficiency_score,
        )


class MetricsEngine:
    """Computes per-episode and aggregate benchmark metrics deterministically."""

    def __init__(
        self,
        fairness_tolerance: float = 0.0,
        reference_total_resource: float = 100.0,
        benchmark_score_weights: dict[str, float] | None = None,
    ) -> None:
        """Initialize metrics engine with fairness, welfare, and score settings."""
        if fairness_tolerance < 0.0:
            raise ValueError(f"fairness_tolerance must be non-negative. Got {fairness_tolerance}.")
        if reference_total_resource <= 0.0:
            raise ValueError(
                f"reference_total_resource must be > 0. Got {reference_total_resource}."
            )
        self._fairness_tolerance = fairness_tolerance
        self._reference_total_resource = reference_total_resource
        self._benchmark_score_weights = benchmark_score_weights or {
            "agreement_rate": 0.30,
            "fairness_index": 0.25,
            "efficiency_score": 0.25,
            "social_welfare_score": 0.20,
        }
        self._validate_benchmark_score_weights()

    def evaluate_episode(
        self,
        *,
        agreement_reached: bool,
        rounds_used: int,
        max_rounds: int,
        final_allocation: dict[str, float] | None,
        social_welfare_score: float | None = None,
    ) -> EpisodeMetrics:
        """Compute all benchmark metrics for one episode."""
        if max_rounds <= 0:
            raise ValueError(f"max_rounds must be > 0. Got {max_rounds}.")
        if rounds_used < 0:
            raise ValueError(f"rounds_used must be non-negative. Got {rounds_used}.")
        if rounds_used > max_rounds:
            raise ValueError(
                f"rounds_used cannot exceed max_rounds. rounds_used={rounds_used}, max_rounds={max_rounds}."
            )
        fairness = self.compute_fairness_index(final_allocation if agreement_reached else None)
        exploitability = self.compute_exploitability_score(
            final_allocation if agreement_reached else None
        )
        efficiency = self.compute_efficiency_score(
            agreement_reached=agreement_reached,
            rounds_used=rounds_used,
            max_rounds=max_rounds,
        )
        welfare = (
            self.compute_social_welfare_score(
                agreement_reached=agreement_reached,
                final_allocation=final_allocation,
                rounds_used=rounds_used,
                max_rounds=max_rounds,
            )
            if social_welfare_score is None
            else social_welfare_score
        )
        _validate_unit_interval(welfare, "social_welfare_score")
        return EpisodeMetrics(
            agreement_reached=agreement_reached,
            fairness_index=fairness,
            exploitability_score=exploitability,
            rounds_used=rounds_used,
            agreement_efficiency_score=efficiency,
            social_welfare_score=welfare,
        )

    def evaluate_episode_from_outcome(
        self,
        *,
        outcome: NegotiationOutcome,
        max_rounds: int,
    ) -> EpisodeMetrics:
        """Compute per-episode metrics from a NegotiationOutcome object."""
        return self.evaluate_episode(
            agreement_reached=outcome.agreement_reached,
            rounds_used=outcome.total_rounds_used,
            max_rounds=max_rounds,
            final_allocation=outcome.agreed_allocation,
        )

    def aggregate(self, episodes: list[EpisodeMetrics]) -> AggregateMetrics:
        """Aggregate benchmark metrics over evaluated episodes."""
        if not episodes:
            raise ValueError("episodes must contain at least one EpisodeMetrics item.")
        count = len(episodes)
        agreements = sum(1 for episode in episodes if episode.agreement_reached)
        agreement_rate = agreements / count
        average_fairness = sum(item.fairness_index for item in episodes) / count
        average_exploitability = sum(item.exploitability_score for item in episodes) / count
        average_rounds_to_agreement = sum(item.rounds_used for item in episodes) / count
        average_efficiency = sum(item.agreement_efficiency_score for item in episodes) / count
        average_social_welfare = sum(item.social_welfare_score for item in episodes) / count
        fairness_std_dev = self._std_dev([item.fairness_index for item in episodes], average_fairness)
        exploitability_std_dev = self._std_dev(
            [item.exploitability_score for item in episodes],
            average_exploitability,
        )
        rounds_std_dev = self._std_dev(
            [float(item.rounds_used) for item in episodes],
            average_rounds_to_agreement,
        )
        efficiency_std_dev = self._std_dev(
            [item.agreement_efficiency_score for item in episodes],
            average_efficiency,
        )
        benchmark_score = self.compute_benchmark_score(
            agreement_rate=agreement_rate,
            fairness_index=average_fairness,
            efficiency_score=average_efficiency,
            social_welfare_score=average_social_welfare,
        )
        return AggregateMetrics(
            agreement_rate=agreement_rate,
            average_fairness_index=average_fairness,
            average_exploitability_score=average_exploitability,
            average_rounds_to_agreement=average_rounds_to_agreement,
            average_efficiency_score=average_efficiency,
            fairness_std_dev=fairness_std_dev,
            exploitability_std_dev=exploitability_std_dev,
            rounds_to_agreement_std_dev=rounds_std_dev,
            efficiency_std_dev=efficiency_std_dev,
            average_social_welfare_score=average_social_welfare,
            benchmark_score=benchmark_score,
            episode_count=count,
        )

    def aggregate_from_episode_traces(self, traces: list[dict[str, object]]) -> AggregateMetrics:
        """Aggregate metrics directly from episode trace dictionaries."""
        episodes: list[EpisodeMetrics] = []
        for trace in traces:
            episodes.append(self._episode_from_trace(trace))
        return self.aggregate(episodes)

    def compute_agreement_rate(self, episodes: list[EpisodeMetrics]) -> float:
        """Compute agreement rate as agreements / total episodes."""
        if not episodes:
            raise ValueError("episodes must not be empty for agreement rate.")
        value = sum(1 for episode in episodes if episode.agreement_reached) / len(episodes)
        _validate_unit_interval(value, "agreement_rate")
        return value

    def compute_fairness_index(self, allocation: dict[str, float] | None) -> float:
        """Compute normalized fairness index, highest for balanced allocations."""
        if allocation is None:
            return 0.0
        self._validate_allocation(allocation)
        if len(allocation) <= 1:
            return 1.0
        values = list(allocation.values())
        spread = max(values) - min(values)
        total = sum(values)
        if total <= 0.0:
            return 0.0
        imbalance = spread / total
        fairness = max(0.0, 1.0 - max(0.0, imbalance - self._fairness_tolerance))
        _validate_unit_interval(fairness, "fairness_index")
        return fairness

    def compute_exploitability_score(self, allocation: dict[str, float] | None) -> float:
        """Compute normalized exploitability score, highest for one-sided outcomes."""
        if allocation is None:
            return 1.0
        fairness = self.compute_fairness_index(allocation)
        exploitability = 1.0 - fairness
        _validate_unit_interval(exploitability, "exploitability_score")
        return exploitability

    def compute_efficiency_score(
        self,
        *,
        agreement_reached: bool,
        rounds_used: int,
        max_rounds: int,
    ) -> float:
        """Compute normalized agreement efficiency, highest for early agreement."""
        if max_rounds <= 0:
            raise ValueError(f"max_rounds must be > 0. Got {max_rounds}.")
        if rounds_used < 0 or rounds_used > max_rounds:
            raise ValueError(
                f"rounds_used must be in [0, {max_rounds}]. Got {rounds_used}."
            )
        if not agreement_reached:
            return 0.0
        efficiency = max(0.0, 1.0 - (rounds_used / max_rounds))
        _validate_unit_interval(efficiency, "efficiency_score")
        return efficiency

    def compute_social_welfare_score(
        self,
        *,
        agreement_reached: bool,
        final_allocation: dict[str, float] | None,
        rounds_used: int,
        max_rounds: int,
    ) -> float:
        """Compute normalized social welfare score for cooperative utility output."""
        if not agreement_reached or final_allocation is None:
            return 0.0
        self._validate_allocation(final_allocation)
        allocated_total = sum(final_allocation.values())
        resource_utilization = min(allocated_total / self._reference_total_resource, 1.0)
        _validate_unit_interval(resource_utilization, "resource_utilization")
        efficiency = self.compute_efficiency_score(
            agreement_reached=agreement_reached,
            rounds_used=rounds_used,
            max_rounds=max_rounds,
        )
        welfare = resource_utilization * efficiency
        _validate_unit_interval(welfare, "social_welfare_score")
        return welfare

    def compute_benchmark_score(
        self,
        *,
        agreement_rate: float,
        fairness_index: float,
        efficiency_score: float,
        social_welfare_score: float,
    ) -> float:
        """Compute unified benchmark leaderboard score in [0,1]."""
        _validate_unit_interval(agreement_rate, "agreement_rate")
        _validate_unit_interval(fairness_index, "fairness_index")
        _validate_unit_interval(efficiency_score, "efficiency_score")
        _validate_unit_interval(social_welfare_score, "social_welfare_score")
        weights = self._benchmark_score_weights
        numerator = (
            weights["agreement_rate"] * agreement_rate
            + weights["fairness_index"] * fairness_index
            + weights["efficiency_score"] * efficiency_score
            + weights["social_welfare_score"] * social_welfare_score
        )
        denominator = sum(weights.values())
        if denominator <= 0.0:
            raise ValueError("Sum of benchmark score weights must be > 0.")
        score = numerator / denominator
        _validate_unit_interval(score, "benchmark_score")
        return score

    def report(self, episodes: list[EpisodeMetrics]) -> dict[str, Any]:
        """Return deterministic report payload for benchmark dashboards."""
        aggregate = self.aggregate(episodes)
        return {
            "aggregate": aggregate.to_dict(),
            "episodes": [episode.to_dict() for episode in episodes],
        }

    def _episode_from_trace(self, trace: dict[str, object]) -> EpisodeMetrics:
        """Construct EpisodeMetrics from a generic trace dictionary."""
        agreement_reached = bool(trace.get("agreement_reached", False))
        rounds_used = int(trace.get("rounds_used", 0))
        max_rounds = int(trace.get("max_rounds", 1))
        raw_allocation = trace.get("final_allocation")
        allocation: dict[str, float] | None = None
        if raw_allocation is not None:
            if not isinstance(raw_allocation, dict):
                raise ValueError("trace['final_allocation'] must be a dictionary when provided.")
            allocation = {str(k): float(v) for k, v in raw_allocation.items()}
        return self.evaluate_episode(
            agreement_reached=agreement_reached,
            rounds_used=rounds_used,
            max_rounds=max_rounds,
            final_allocation=allocation,
            social_welfare_score=self._read_optional_float(trace.get("social_welfare_score")),
        )

    def _validate_benchmark_score_weights(self) -> None:
        """Validate benchmark score weight configuration."""
        expected = {"agreement_rate", "fairness_index", "efficiency_score", "social_welfare_score"}
        actual = set(self._benchmark_score_weights.keys())
        if actual != expected:
            raise ValueError(
                "benchmark_score_weights must contain exactly keys "
                f"{sorted(expected)}. Got {sorted(actual)}."
            )
        for key, value in self._benchmark_score_weights.items():
            _validate_finite(value, f"benchmark_score_weights[{key}]")
            if value < 0.0:
                raise ValueError(f"benchmark_score_weights[{key}] must be non-negative.")
        if sum(self._benchmark_score_weights.values()) <= 0.0:
            raise ValueError("Sum of benchmark_score_weights must be > 0.")

    @staticmethod
    def _std_dev(values: list[float], mean_value: float) -> float:
        """Compute deterministic population standard deviation."""
        if not values:
            raise ValueError("values must not be empty for std deviation.")
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        std_dev = math.sqrt(variance)
        _validate_finite(std_dev, "std_dev")
        if std_dev < 0.0:
            raise ValueError("std_dev must be non-negative.")
        return std_dev

    @staticmethod
    def _read_optional_float(value: object) -> float | None:
        """Read an optional float value from a trace dictionary."""
        if value is None:
            return None
        parsed = float(value)
        _validate_finite(parsed, "optional_float")
        return parsed

    @staticmethod
    def _validate_allocation(allocation: dict[str, float]) -> None:
        """Validate allocation values for fairness and exploitability metrics."""
        if not allocation:
            raise ValueError("allocation must not be empty.")
        total = 0.0
        for key, value in allocation.items():
            if not key.strip():
                raise ValueError("allocation contains empty agent id.")
            _validate_finite(value, f"allocation[{key}]")
            if value < 0.0:
                raise ValueError(f"allocation[{key}] must be non-negative. Got {value}.")
            total += value
        if total <= 0.0:
            raise ValueError("allocation total must be > 0.")
