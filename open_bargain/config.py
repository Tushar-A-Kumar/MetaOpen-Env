"""Central configuration contracts for the OpenBargain benchmark.

This module defines strongly typed, validated dataclasses that act as the
single source of truth for benchmark configuration across environment,
utility, reward, simulation, and experiment registration concerns.
"""

from dataclasses import dataclass, field


def _validate_non_empty(value: str, field_name: str) -> None:
    """Validate that a string configuration field is not empty."""
    if not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


def _validate_positive(value: int | float, field_name: str) -> None:
    """Validate that a numeric configuration field is strictly positive."""
    if value <= 0:
        raise ValueError(f"{field_name} must be > 0. Got {value}.")


def _validate_non_negative(value: int | float, field_name: str) -> None:
    """Validate that a numeric configuration field is non-negative."""
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0. Got {value}.")


def _validate_closed_unit_interval(value: float, field_name: str) -> None:
    """Validate that a float belongs to the inclusive [0, 1] interval."""
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1 inclusive. Got {value}.")


def _validate_range(value: tuple[float, float], field_name: str) -> None:
    """Validate that a sensitivity range is ordered as (min <= max)."""
    min_value, max_value = value
    if min_value > max_value:
        raise ValueError(
            f"{field_name} must satisfy min <= max. "
            f"Got min={min_value}, max={max_value}."
        )


@dataclass(slots=True, frozen=True)
class BenchmarkMetadata:
    """Benchmark identity and registration metadata.

    Fields:
    - name: Canonical benchmark name used in docs and registry.
    - version: Semantic version for benchmark contract evolution.
    - description: Human-readable benchmark summary.
    - supported_algorithms: Algorithms expected to be supported by training APIs.
    - environment_id: Environment registration identifier for Gym/OpenEnv adapters.
    """

    name: str = "OpenBargain"
    version: str = "0.1.0"
    description: str = "Multi-agent strategic bargaining benchmark under hidden incentives."
    supported_algorithms: tuple[str, ...] = ("ppo",)
    environment_id: str = "OpenBargain-v0"

    def __post_init__(self) -> None:
        """Validate benchmark metadata integrity."""
        _validate_non_empty(self.name, "BenchmarkMetadata.name")
        _validate_non_empty(self.version, "BenchmarkMetadata.version")
        _validate_non_empty(self.description, "BenchmarkMetadata.description")
        _validate_non_empty(self.environment_id, "BenchmarkMetadata.environment_id")
        if not self.supported_algorithms:
            raise ValueError("BenchmarkMetadata.supported_algorithms must not be empty.")
        for algorithm in self.supported_algorithms:
            _validate_non_empty(algorithm, "BenchmarkMetadata.supported_algorithms item")


@dataclass(slots=True, frozen=True)
class EnvironmentConfig:
    """Environment-level configuration for bargaining process constraints.

    Fields:
    - total_resource_amount: Total divisible quantity available for allocation.
    - max_negotiation_rounds: Maximum number of negotiation rounds per episode.
    - minimum_acceptable_allocation: Minimum acceptable allocation threshold.
    - decay_enabled: Toggle for round-based resource decay.
    - resource_decay_rate_per_round: Per-round fractional decay in [0, 1].
    - fairness_bonus_enabled: Toggle for fairness shaping terms.
    - exploitation_penalty_enabled: Toggle for exploitation penalty shaping.
    - delay_penalty_enabled: Toggle for delay-related penalty shaping.
    - observation_normalization_enabled: Enables normalized observation features.
    - history_summary_window: Number of recent actions considered in summary.
    - action_types: Canonical action types for validity-mask placeholders.
    """

    total_resource_amount: float = 100.0
    max_negotiation_rounds: int = 20
    minimum_acceptable_allocation: float = 1.0
    decay_enabled: bool = True
    resource_decay_rate_per_round: float = 0.01
    fairness_bonus_enabled: bool = True
    exploitation_penalty_enabled: bool = True
    delay_penalty_enabled: bool = True
    observation_normalization_enabled: bool = True
    history_summary_window: int = 5
    action_types: tuple[str, ...] = ("propose", "accept", "reject", "counteroffer")

    def __post_init__(self) -> None:
        """Validate environment configuration values."""
        _validate_positive(self.total_resource_amount, "EnvironmentConfig.total_resource_amount")
        if self.max_negotiation_rounds <= 0:
            raise ValueError(
                "EnvironmentConfig.max_negotiation_rounds must be > 0. "
                f"Got {self.max_negotiation_rounds}."
            )
        _validate_positive(
            self.minimum_acceptable_allocation,
            "EnvironmentConfig.minimum_acceptable_allocation",
        )
        if self.minimum_acceptable_allocation > self.total_resource_amount:
            raise ValueError(
                "EnvironmentConfig.minimum_acceptable_allocation cannot exceed "
                "EnvironmentConfig.total_resource_amount."
            )
        _validate_closed_unit_interval(
            self.resource_decay_rate_per_round,
            "EnvironmentConfig.resource_decay_rate_per_round",
        )
        _validate_positive(self.history_summary_window, "EnvironmentConfig.history_summary_window")
        if not self.action_types:
            raise ValueError("EnvironmentConfig.action_types must not be empty.")
        for action_type in self.action_types:
            _validate_non_empty(action_type, "EnvironmentConfig.action_types item")


@dataclass(slots=True, frozen=True)
class RewardConfig:
    """Reward coefficient configuration for future reward composition.

    Fields:
    - utility_weight: Weight for private utility component.
    - fairness_bonus_weight: Weight for fairness reward component.
    - exploitation_penalty_weight: Weight for exploitation penalty component.
    - delay_penalty_weight: Weight for delay penalty component.
    - agreement_reward_weight: Weight for successful agreement reward.
    - failure_penalty_weight: Weight for failed-negotiation penalty.
    - exploitation_tolerance: Allowed imbalance before exploitation penalty grows.
    - reward_clipping_enabled: Enables clipping of scalar rewards.
    - reward_clip_bounds: Lower and upper clipping bounds.
    """

    utility_weight: float = 1.0
    fairness_bonus_weight: float = 1.0
    exploitation_penalty_weight: float = 1.0
    delay_penalty_weight: float = 0.1
    agreement_reward_weight: float = 1.0
    failure_penalty_weight: float = 1.0
    exploitation_tolerance: float = 0.2
    reward_clipping_enabled: bool = True
    reward_clip_bounds: tuple[float, float] = (-2.0, 2.0)

    def __post_init__(self) -> None:
        """Validate reward coefficient values."""
        _validate_non_negative(self.utility_weight, "RewardConfig.utility_weight")
        _validate_non_negative(self.fairness_bonus_weight, "RewardConfig.fairness_bonus_weight")
        _validate_non_negative(
            self.exploitation_penalty_weight,
            "RewardConfig.exploitation_penalty_weight",
        )
        _validate_non_negative(self.delay_penalty_weight, "RewardConfig.delay_penalty_weight")
        _validate_non_negative(
            self.agreement_reward_weight,
            "RewardConfig.agreement_reward_weight",
        )
        _validate_non_negative(self.failure_penalty_weight, "RewardConfig.failure_penalty_weight")
        _validate_closed_unit_interval(
            self.exploitation_tolerance,
            "RewardConfig.exploitation_tolerance",
        )
        _validate_range(self.reward_clip_bounds, "RewardConfig.reward_clip_bounds")


@dataclass(slots=True, frozen=True)
class UtilityConfig:
    """Hidden utility generation controls for private preference sampling.

    Fields:
    - greed_sensitivity_range: Range controlling greed preference sensitivity.
    - fairness_sensitivity_range: Range controlling fairness preference sensitivity.
    - urgency_sensitivity_range: Range controlling urgency preference sensitivity.
    - deterministic_preference_generation: Toggle deterministic utility sampling.
    - utility_normalization_enabled: Enables normalized utility outputs.
    - utility_normalization_bounds: Target bounds for normalized utility.
    - normalize_breakdown_components: Normalize utility component breakdown values.
    """

    greed_sensitivity_range: tuple[float, float] = (0.1, 1.0)
    fairness_sensitivity_range: tuple[float, float] = (0.1, 1.0)
    urgency_sensitivity_range: tuple[float, float] = (0.1, 1.0)
    deterministic_preference_generation: bool = True
    utility_normalization_enabled: bool = False
    utility_normalization_bounds: tuple[float, float] = (-1.0, 1.0)
    normalize_breakdown_components: bool = False

    def __post_init__(self) -> None:
        """Validate utility sensitivity ranges."""
        _validate_range(self.greed_sensitivity_range, "UtilityConfig.greed_sensitivity_range")
        _validate_range(
            self.fairness_sensitivity_range,
            "UtilityConfig.fairness_sensitivity_range",
        )
        _validate_range(self.urgency_sensitivity_range, "UtilityConfig.urgency_sensitivity_range")
        _validate_range(self.utility_normalization_bounds, "UtilityConfig.utility_normalization_bounds")


@dataclass(slots=True, frozen=True)
class SimulationConfig:
    """Simulation and reproducibility defaults for benchmark experiments.

    Fields:
    - default_random_seed: Primary seed used for deterministic runs.
    - logging_enabled: Enables simulation/training logging hooks.
    - metrics_enabled: Enables benchmark metrics collection paths.
    - trace_history_enabled: Enables per-step trace storage for debugging/eval.
    """

    default_random_seed: int = 0
    logging_enabled: bool = True
    metrics_enabled: bool = True
    trace_history_enabled: bool = True

    def __post_init__(self) -> None:
        """Validate simulation defaults."""
        _validate_non_negative(self.default_random_seed, "SimulationConfig.default_random_seed")


@dataclass(slots=True, frozen=True)
class OpenBargainConfig:
    """Master benchmark configuration object for OpenBargain.

    This object aggregates all configuration domains and should be the primary
    import target for the rest of the benchmark codebase.
    """

    benchmark: BenchmarkMetadata = field(default_factory=BenchmarkMetadata)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    utility: UtilityConfig = field(default_factory=UtilityConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)

    @classmethod
    def default(cls) -> "OpenBargainConfig":
        """Return the default benchmark configuration."""
        return cls()


DEFAULT_BENCHMARK_CONFIG = OpenBargainConfig()
