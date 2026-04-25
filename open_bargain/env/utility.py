"""Hidden preference generation and utility evaluation for OpenBargain.

This module owns private agent preference profiles and deterministic utility
scoring contracts. It intentionally excludes reward shaping, observation logic,
and environment transition behavior.
"""

from dataclasses import dataclass
import math
import random
from typing import Any

from open_bargain.config import OpenBargainConfig, UtilityConfig


def _validate_non_empty(value: str, field_name: str) -> None:
    """Validate that a string field is non-empty."""
    if not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


def _validate_range(value: tuple[float, float], field_name: str) -> None:
    """Validate ordered numeric range with finite boundaries."""
    low, high = value
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError(f"{field_name} bounds must be finite numbers.")
    if low > high:
        raise ValueError(f"{field_name} must satisfy min <= max. Got {value}.")


def _validate_finite(value: float, field_name: str) -> None:
    """Validate value is a finite float."""
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite. Got {value}.")


@dataclass(slots=True, frozen=True)
class PreferenceProfile:
    """Hidden utility sensitivities for one bargaining agent.

    Fields:
    - agent_id: Unique agent/profile identifier.
    - greed_sensitivity: Coefficient scaling resource gain utility.
    - fairness_sensitivity: Coefficient scaling fairness adjustment.
    - urgency_sensitivity: Coefficient scaling urgency-related adjustment.
    """

    agent_id: str
    greed_sensitivity: float
    fairness_sensitivity: float
    urgency_sensitivity: float

    def __post_init__(self) -> None:
        """Validate profile values are finite and structurally valid."""
        _validate_non_empty(self.agent_id, "PreferenceProfile.agent_id")
        _validate_finite(self.greed_sensitivity, "PreferenceProfile.greed_sensitivity")
        _validate_finite(self.fairness_sensitivity, "PreferenceProfile.fairness_sensitivity")
        _validate_finite(self.urgency_sensitivity, "PreferenceProfile.urgency_sensitivity")

    def validate_against_config(self, utility_config: UtilityConfig) -> None:
        """Validate profile coefficients lie within configured sensitivity ranges."""
        _validate_range(
            utility_config.greed_sensitivity_range,
            "UtilityConfig.greed_sensitivity_range",
        )
        _validate_range(
            utility_config.fairness_sensitivity_range,
            "UtilityConfig.fairness_sensitivity_range",
        )
        _validate_range(
            utility_config.urgency_sensitivity_range,
            "UtilityConfig.urgency_sensitivity_range",
        )
        greed_min, greed_max = utility_config.greed_sensitivity_range
        fair_min, fair_max = utility_config.fairness_sensitivity_range
        urg_min, urg_max = utility_config.urgency_sensitivity_range
        if not (greed_min <= self.greed_sensitivity <= greed_max):
            raise ValueError(
                "PreferenceProfile.greed_sensitivity is outside configured range. "
                f"value={self.greed_sensitivity}, range={utility_config.greed_sensitivity_range}."
            )
        if not (fair_min <= self.fairness_sensitivity <= fair_max):
            raise ValueError(
                "PreferenceProfile.fairness_sensitivity is outside configured range. "
                f"value={self.fairness_sensitivity}, range={utility_config.fairness_sensitivity_range}."
            )
        if not (urg_min <= self.urgency_sensitivity <= urg_max):
            raise ValueError(
                "PreferenceProfile.urgency_sensitivity is outside configured range. "
                f"value={self.urgency_sensitivity}, range={utility_config.urgency_sensitivity_range}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize profile to deterministic JSON-safe dictionary."""
        return {
            "agent_id": self.agent_id,
            "greed_sensitivity": self.greed_sensitivity,
            "fairness_sensitivity": self.fairness_sensitivity,
            "urgency_sensitivity": self.urgency_sensitivity,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PreferenceProfile":
        """Deserialize profile from a dictionary payload."""
        return cls(
            agent_id=str(payload["agent_id"]),
            greed_sensitivity=float(payload["greed_sensitivity"]),
            fairness_sensitivity=float(payload["fairness_sensitivity"]),
            urgency_sensitivity=float(payload["urgency_sensitivity"]),
        )


@dataclass(slots=True, frozen=True)
class UtilityBreakdown:
    """Decomposed private utility outputs for diagnostics and attribution.

    Fields:
    - resource_gain_utility: Utility attributed to received resource amount.
    - fairness_adjustment: Utility adjustment for distance from equal split.
    - urgency_adjustment: Utility adjustment for round-dependent urgency.
    - total_utility: Aggregate utility scalar.
    """

    resource_gain_utility: float
    fairness_adjustment: float
    urgency_adjustment: float
    total_utility: float

    def __post_init__(self) -> None:
        """Validate decomposition values are finite and internally consistent."""
        _validate_finite(self.resource_gain_utility, "UtilityBreakdown.resource_gain_utility")
        _validate_finite(self.fairness_adjustment, "UtilityBreakdown.fairness_adjustment")
        _validate_finite(self.urgency_adjustment, "UtilityBreakdown.urgency_adjustment")
        _validate_finite(self.total_utility, "UtilityBreakdown.total_utility")

    def to_dict(self) -> dict[str, float]:
        """Serialize utility breakdown to JSON-safe dictionary."""
        return {
            "resource_gain_utility": self.resource_gain_utility,
            "fairness_adjustment": self.fairness_adjustment,
            "urgency_adjustment": self.urgency_adjustment,
            "total_utility": self.total_utility,
        }


class DeterministicPreferenceGenerator:
    """Seeded preference generator for reproducible hidden profile sampling."""

    def __init__(self, config: OpenBargainConfig, seed: int | None = None) -> None:
        """Initialize generator with benchmark config and optional override seed."""
        self._config = config
        self._seed = config.simulation.default_random_seed if seed is None else seed
        if self._seed < 0:
            raise ValueError("seed must be non-negative.")

    def generate_profile(self, agent_id: str, offset: int = 0) -> PreferenceProfile:
        """Generate one deterministic profile for an agent and stream offset."""
        _validate_non_empty(agent_id, "agent_id")
        if offset < 0:
            raise ValueError(f"offset must be non-negative. Got {offset}.")
        rng = random.Random(self._seed + offset)
        utility_cfg = self._config.utility
        profile = PreferenceProfile(
            agent_id=agent_id,
            greed_sensitivity=rng.uniform(*utility_cfg.greed_sensitivity_range),
            fairness_sensitivity=rng.uniform(*utility_cfg.fairness_sensitivity_range),
            urgency_sensitivity=rng.uniform(*utility_cfg.urgency_sensitivity_range),
        )
        profile.validate_against_config(utility_cfg)
        return profile

    def generate(self, agent_ids: list[str]) -> dict[str, PreferenceProfile]:
        """Generate deterministic profiles for multiple agents in one call."""
        profiles: dict[str, PreferenceProfile] = {}
        for index, agent_id in enumerate(agent_ids):
            if agent_id in profiles:
                raise ValueError(f"Duplicate agent_id detected in batch: '{agent_id}'.")
            profiles[agent_id] = self.generate_profile(agent_id=agent_id, offset=index)
        return profiles

    def summarize_profiles(self, profiles: dict[str, PreferenceProfile]) -> dict[str, Any]:
        """Return deterministic aggregate diagnostics for a profile batch."""
        if not profiles:
            return {
                "count": 0,
                "agent_ids": [],
                "greed": {"mean": 0.0, "min": 0.0, "max": 0.0},
                "fairness": {"mean": 0.0, "min": 0.0, "max": 0.0},
                "urgency": {"mean": 0.0, "min": 0.0, "max": 0.0},
            }
        greed_values = [profile.greed_sensitivity for profile in profiles.values()]
        fairness_values = [profile.fairness_sensitivity for profile in profiles.values()]
        urgency_values = [profile.urgency_sensitivity for profile in profiles.values()]
        return {
            "count": len(profiles),
            "agent_ids": sorted(profiles.keys()),
            "greed": self._summarize_values(greed_values),
            "fairness": self._summarize_values(fairness_values),
            "urgency": self._summarize_values(urgency_values),
        }

    @staticmethod
    def _summarize_values(values: list[float]) -> dict[str, float]:
        """Summarize numeric values using deterministic aggregate statistics."""
        count = len(values)
        return {
            "mean": sum(values) / count,
            "min": min(values),
            "max": max(values),
        }


class UtilityEvaluator:
    """Evaluates private utility of allocations for hidden preference profiles."""

    def __init__(self, config: OpenBargainConfig) -> None:
        """Initialize evaluator with benchmark config for validation constants."""
        self._config = config
        self._total_resource = config.environment.total_resource_amount
        self._max_rounds = config.environment.max_negotiation_rounds
        self._normalization_enabled = config.utility.utility_normalization_enabled
        self._normalization_bounds = config.utility.utility_normalization_bounds
        self._normalize_components = config.utility.normalize_breakdown_components
        self._validate_normalization_config()

    def evaluate_offer(
        self,
        *,
        profile: PreferenceProfile,
        allocation: dict[str, float],
        round_index: int,
        normalize: bool | None = None,
    ) -> tuple[float, UtilityBreakdown]:
        """Compute total utility and decomposition for one agent allocation.

        Inputs:
        - profile: Hidden agent preference profile.
        - allocation: Allocation map over all agents.
        - round_index: Zero-based negotiation round index.

        Returns:
        - Tuple of (total_utility, UtilityBreakdown).

        Raises:
        - ValueError for invalid profile, allocation, or round values.
        """
        raw_total, raw_breakdown = self._evaluate_offer_raw(
            profile=profile,
            allocation=allocation,
            round_index=round_index,
        )
        should_normalize = self._normalization_enabled if normalize is None else normalize
        if not should_normalize:
            return raw_total, raw_breakdown
        normalized_total, breakdown = self._normalize_outputs(raw_breakdown)
        return normalized_total, breakdown

    def evaluate_batch(
        self,
        *,
        profiles: list[PreferenceProfile],
        allocations: list[dict[str, float]],
        round_indices: list[int],
        normalize: bool | None = None,
    ) -> tuple[list[float], list[UtilityBreakdown]]:
        """Evaluate utility for batched profile-allocation-round tuples.

        Validates batch shape and each item, then returns aligned lists of
        scalar utilities and utility breakdowns for rollout storage.
        """
        batch_size = len(profiles)
        if batch_size != len(allocations) or batch_size != len(round_indices):
            raise ValueError(
                "Batch length mismatch: profiles, allocations, and round_indices "
                f"must have equal length. Got {batch_size}, {len(allocations)}, "
                f"{len(round_indices)}."
            )
        if batch_size == 0:
            return [], []
        should_normalize = self._normalization_enabled if normalize is None else normalize
        totals: list[float] = []
        breakdowns: list[UtilityBreakdown] = []
        for profile, allocation, round_index in zip(profiles, allocations, round_indices):
            raw_total, raw_breakdown = self._evaluate_offer_raw(
                profile=profile,
                allocation=allocation,
                round_index=round_index,
            )
            if should_normalize:
                normalized_total, normalized_breakdown = self._normalize_outputs(raw_breakdown)
                totals.append(normalized_total)
                breakdowns.append(normalized_breakdown)
            else:
                totals.append(raw_total)
                breakdowns.append(raw_breakdown)
        return totals, breakdowns

    def inspect_profile(self, profile: PreferenceProfile) -> dict[str, Any]:
        """Return deterministic diagnostic summary for a single profile."""
        profile.validate_against_config(self._config.utility)
        return {
            "agent_id": profile.agent_id,
            "greed_sensitivity": profile.greed_sensitivity,
            "fairness_sensitivity": profile.fairness_sensitivity,
            "urgency_sensitivity": profile.urgency_sensitivity,
            "normalization_enabled_default": self._normalization_enabled,
            "normalization_bounds": self._normalization_bounds,
        }

    def inspect_breakdown(
        self,
        *,
        breakdown: UtilityBreakdown,
        raw_breakdown: UtilityBreakdown | None = None,
    ) -> dict[str, Any]:
        """Return deterministic diagnostics for utility decomposition values."""
        output: dict[str, Any] = {
            "resource_gain_utility": breakdown.resource_gain_utility,
            "fairness_adjustment": breakdown.fairness_adjustment,
            "urgency_adjustment": breakdown.urgency_adjustment,
            "total_utility": breakdown.total_utility,
            "normalization_enabled_default": self._normalization_enabled,
        }
        if raw_breakdown is not None:
            output["raw"] = raw_breakdown.to_dict()
            output["normalization_delta_total"] = breakdown.total_utility - raw_breakdown.total_utility
        return output

    def inspect_batch(
        self,
        *,
        totals: list[float],
        breakdowns: list[UtilityBreakdown],
    ) -> dict[str, Any]:
        """Return aggregate diagnostics for batch utility distributions."""
        if len(totals) != len(breakdowns):
            raise ValueError(
                "totals and breakdowns must have equal length. "
                f"Got {len(totals)} and {len(breakdowns)}."
            )
        if not totals:
            return {
                "count": 0,
                "total_utility": {"mean": 0.0, "min": 0.0, "max": 0.0},
                "resource_gain_utility_mean": 0.0,
                "fairness_adjustment_mean": 0.0,
                "urgency_adjustment_mean": 0.0,
            }
        for index, value in enumerate(totals):
            _validate_finite(value, f"totals[{index}]")
        resource_gain_values = [item.resource_gain_utility for item in breakdowns]
        fairness_values = [item.fairness_adjustment for item in breakdowns]
        urgency_values = [item.urgency_adjustment for item in breakdowns]
        return {
            "count": len(totals),
            "total_utility": {
                "mean": sum(totals) / len(totals),
                "min": min(totals),
                "max": max(totals),
            },
            "resource_gain_utility_mean": sum(resource_gain_values) / len(resource_gain_values),
            "fairness_adjustment_mean": sum(fairness_values) / len(fairness_values),
            "urgency_adjustment_mean": sum(urgency_values) / len(urgency_values),
        }

    def _evaluate_offer_raw(
        self,
        *,
        profile: PreferenceProfile,
        allocation: dict[str, float],
        round_index: int,
    ) -> tuple[float, UtilityBreakdown]:
        """Compute raw utility and breakdown without normalization."""
        profile.validate_against_config(self._config.utility)
        self._validate_round_index(round_index)
        self._validate_allocation(allocation)
        agent_resource = float(allocation.get(profile.agent_id, 0.0))
        resource_gain_utility = (agent_resource / self._total_resource) * profile.greed_sensitivity
        equal_share = self._total_resource / len(allocation)
        fairness_distance = abs(agent_resource - equal_share) / self._total_resource
        fairness_adjustment = (1.0 - fairness_distance) * profile.fairness_sensitivity
        progress = round_index / self._max_rounds
        urgency_adjustment = -progress * profile.urgency_sensitivity
        total_utility = resource_gain_utility + fairness_adjustment + urgency_adjustment
        breakdown = UtilityBreakdown(
            resource_gain_utility=resource_gain_utility,
            fairness_adjustment=fairness_adjustment,
            urgency_adjustment=urgency_adjustment,
            total_utility=total_utility,
        )
        return total_utility, breakdown

    def _normalize_outputs(
        self,
        breakdown: UtilityBreakdown,
    ) -> tuple[float, UtilityBreakdown]:
        """Normalize utility outputs into configured bounded interval."""
        normalized_total = self._normalize_scalar(breakdown.total_utility)
        if not self._normalize_components:
            normalized_breakdown = UtilityBreakdown(
                resource_gain_utility=breakdown.resource_gain_utility,
                fairness_adjustment=breakdown.fairness_adjustment,
                urgency_adjustment=breakdown.urgency_adjustment,
                total_utility=normalized_total,
            )
            return normalized_total, normalized_breakdown
        normalized_breakdown = UtilityBreakdown(
            resource_gain_utility=self._normalize_scalar(breakdown.resource_gain_utility),
            fairness_adjustment=self._normalize_scalar(breakdown.fairness_adjustment),
            urgency_adjustment=self._normalize_scalar(breakdown.urgency_adjustment),
            total_utility=normalized_total,
        )
        return normalized_total, normalized_breakdown

    def _normalize_scalar(self, raw_value: float) -> float:
        """Map raw scalar utility to configured bounded normalization interval."""
        _validate_finite(raw_value, "raw_value")
        low, high = self._normalization_bounds
        centered = math.tanh(raw_value)
        normalized = low + ((centered + 1.0) * 0.5 * (high - low))
        _validate_finite(normalized, "normalized")
        return normalized

    def _validate_normalization_config(self) -> None:
        """Validate normalization settings for deterministic safe execution."""
        low, high = self._normalization_bounds
        if not math.isfinite(low) or not math.isfinite(high):
            raise ValueError("utility_normalization_bounds must be finite values.")
        if low >= high:
            raise ValueError(
                "utility_normalization_bounds must satisfy low < high. "
                f"Got {self._normalization_bounds}."
            )

    def _validate_round_index(self, round_index: int) -> None:
        """Validate round index against configured bounds."""
        if round_index < 0:
            raise ValueError(f"round_index must be non-negative. Got {round_index}.")
        if round_index > self._max_rounds:
            raise ValueError(
                "round_index exceeds configured max_negotiation_rounds. "
                f"round_index={round_index}, "
                f"max_negotiation_rounds={self._max_rounds}."
            )

    def _validate_allocation(self, allocation: dict[str, float]) -> None:
        """Validate allocation is non-empty, non-negative, and resource-bounded."""
        if not allocation:
            raise ValueError("allocation must not be empty.")
        total_allocated = 0.0
        for agent_id, amount in allocation.items():
            _validate_non_empty(agent_id, "allocation key")
            if amount < 0:
                raise ValueError(
                    f"allocation values must be non-negative. agent_id={agent_id}, amount={amount}."
                )
            _validate_finite(amount, f"allocation[{agent_id}]")
            total_allocated += amount
        if total_allocated > self._total_resource:
            raise ValueError(
                "Total allocation exceeds configured total resource amount. "
                f"total_allocated={total_allocated}, "
                f"total_resource_amount={self._total_resource}."
            )
