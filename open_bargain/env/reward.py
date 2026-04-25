"""Reward engine for OpenBargain bargaining trajectories.

This module converts utility signals and negotiation outcomes into bounded,
interpretable reward components suitable for PPO/GRPO/self-play training.
It intentionally excludes environment transitions and observation logic.
"""

from dataclasses import dataclass
import math
from typing import Any

from open_bargain.config import OpenBargainConfig, RewardConfig
from open_bargain.env.state import NegotiationState


def _validate_finite(value: float, field_name: str) -> None:
    """Validate numeric value is finite."""
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite. Got {value}.")


def _validate_non_negative(value: float, field_name: str) -> None:
    """Validate numeric value is non-negative."""
    if value < 0.0:
        raise ValueError(f"{field_name} must be >= 0. Got {value}.")


@dataclass(slots=True, frozen=True)
class RewardBreakdown:
    """Decomposed reward components for one agent and one evaluation point.

    Fields:
    - private_utility_reward: Weighted private utility contribution.
    - fairness_bonus: Bonus for balanced resource allocation.
    - delay_penalty: Penalty for prolonged negotiations.
    - exploitation_penalty: Penalty for exploitative allocation imbalance.
    - no_agreement_penalty: Penalty applied when episode terminates without agreement.
    - total_reward: Final bounded reward scalar.
    """

    private_utility_reward: float
    fairness_bonus: float
    delay_penalty: float
    exploitation_penalty: float
    no_agreement_penalty: float
    total_reward: float

    def __post_init__(self) -> None:
        """Validate all component values are finite for stable training."""
        _validate_finite(self.private_utility_reward, "RewardBreakdown.private_utility_reward")
        _validate_finite(self.fairness_bonus, "RewardBreakdown.fairness_bonus")
        _validate_finite(self.delay_penalty, "RewardBreakdown.delay_penalty")
        _validate_finite(self.exploitation_penalty, "RewardBreakdown.exploitation_penalty")
        _validate_finite(self.no_agreement_penalty, "RewardBreakdown.no_agreement_penalty")
        _validate_finite(self.total_reward, "RewardBreakdown.total_reward")

    def to_dict(self) -> dict[str, float]:
        """Serialize reward breakdown to deterministic JSON-safe dictionary."""
        return {
            "private_utility_reward": self.private_utility_reward,
            "fairness_bonus": self.fairness_bonus,
            "delay_penalty": self.delay_penalty,
            "exploitation_penalty": self.exploitation_penalty,
            "no_agreement_penalty": self.no_agreement_penalty,
            "total_reward": self.total_reward,
        }


@dataclass(slots=True, frozen=True)
class RewardWeights:
    """Configurable weighting coefficients for reward composition."""

    utility_weight: float
    fairness_weight: float
    delay_weight: float
    exploitation_weight: float
    failure_weight: float

    def __post_init__(self) -> None:
        """Validate weight coefficients remain non-negative."""
        _validate_non_negative(self.utility_weight, "RewardWeights.utility_weight")
        _validate_non_negative(self.fairness_weight, "RewardWeights.fairness_weight")
        _validate_non_negative(self.delay_weight, "RewardWeights.delay_weight")
        _validate_non_negative(self.exploitation_weight, "RewardWeights.exploitation_weight")
        _validate_non_negative(self.failure_weight, "RewardWeights.failure_weight")

    @classmethod
    def from_config(cls, reward_config: RewardConfig) -> "RewardWeights":
        """Build reward weights from centralized benchmark configuration."""
        return cls(
            utility_weight=reward_config.utility_weight,
            fairness_weight=reward_config.fairness_bonus_weight,
            delay_weight=reward_config.delay_penalty_weight,
            exploitation_weight=reward_config.exploitation_penalty_weight,
            failure_weight=reward_config.failure_penalty_weight,
        )


class RewardEngine:
    """Primary reward aggregation interface for bargaining environments."""

    def __init__(self, config: OpenBargainConfig) -> None:
        """Initialize reward engine from centralized config."""
        self._config = config
        self._weights = RewardWeights.from_config(config.reward)
        self._max_rounds = config.environment.max_negotiation_rounds
        self._clip_enabled = config.reward.reward_clipping_enabled
        self._clip_bounds = config.reward.reward_clip_bounds
        self._exploitation_tolerance = config.reward.exploitation_tolerance
        self._validate_clip_bounds()

    def compute_intermediate_reward(
        self,
        *,
        state: NegotiationState,
        agent_id: str,
        proposed_allocation: dict[str, float] | None = None,
    ) -> tuple[float, RewardBreakdown]:
        """Compute bounded shaped reward for non-terminal negotiation steps."""
        self._validate_agent_id(state, agent_id)
        self._validate_round(state.current_round)
        delay_penalty = self.compute_delay_penalty(rounds_used=state.current_round)
        exploitation_penalty = 0.0
        fairness_bonus = 0.0
        if proposed_allocation is not None:
            self._validate_allocation(proposed_allocation)
            fairness_bonus = self.compute_fairness_bonus(
                allocation=proposed_allocation,
                agent_id=agent_id,
            ) * 0.1
            exploitation_penalty = self.compute_exploitation_penalty(
                allocation=proposed_allocation,
                agent_id=agent_id,
            ) * 0.1
        raw_total = fairness_bonus - delay_penalty - exploitation_penalty
        total_reward = self._clip_reward(raw_total)
        breakdown = RewardBreakdown(
            private_utility_reward=0.0,
            fairness_bonus=fairness_bonus,
            delay_penalty=delay_penalty,
            exploitation_penalty=exploitation_penalty,
            no_agreement_penalty=0.0,
            total_reward=total_reward,
        )
        return total_reward, breakdown

    def compute_terminal_reward(
        self,
        *,
        state: NegotiationState,
        agent_id: str,
        private_utility_value: float,
    ) -> tuple[float, RewardBreakdown]:
        """Compute bounded terminal reward using all benchmark components."""
        self._validate_agent_id(state, agent_id)
        self._validate_round(state.current_round)
        _validate_finite(private_utility_value, "private_utility_value")

        utility_reward = self.compute_private_utility_reward(private_utility_value)
        delay_penalty = self.compute_delay_penalty(rounds_used=state.current_round)
        fairness_bonus = 0.0
        exploitation_penalty = 0.0
        no_agreement_penalty = 0.0

        if state.outcome is not None and state.outcome.agreed_allocation is not None:
            allocation = state.outcome.agreed_allocation
            self._validate_allocation(allocation)
            fairness_bonus = self.compute_fairness_bonus(allocation=allocation, agent_id=agent_id)
            exploitation_penalty = self.compute_exploitation_penalty(
                allocation=allocation,
                agent_id=agent_id,
            )
        elif state.is_terminal and not state.agreement_reached:
            no_agreement_penalty = self.compute_no_agreement_penalty()

        raw_total = (
            utility_reward
            + fairness_bonus
            - delay_penalty
            - exploitation_penalty
            - no_agreement_penalty
        )
        total_reward = self._clip_reward(raw_total)
        breakdown = RewardBreakdown(
            private_utility_reward=utility_reward,
            fairness_bonus=fairness_bonus,
            delay_penalty=delay_penalty,
            exploitation_penalty=exploitation_penalty,
            no_agreement_penalty=no_agreement_penalty,
            total_reward=total_reward,
        )
        return total_reward, breakdown

    def compute_private_utility_reward(self, private_utility_value: float) -> float:
        """Compute weighted private utility reward component."""
        _validate_finite(private_utility_value, "private_utility_value")
        value = self._weights.utility_weight * private_utility_value
        _validate_finite(value, "private_utility_reward")
        return value

    def compute_fairness_bonus(self, *, allocation: dict[str, float], agent_id: str) -> float:
        """Compute bounded fairness bonus using equal-split distance."""
        self._validate_allocation(allocation)
        if agent_id not in allocation:
            return 0.0
        total_resource = self._config.environment.total_resource_amount
        equal_share = total_resource / len(allocation)
        agent_share = allocation[agent_id]
        fairness_distance = abs(agent_share - equal_share) / total_resource
        fairness_score = max(0.0, 1.0 - fairness_distance)
        bonus = fairness_score * self._weights.fairness_weight
        _validate_finite(bonus, "fairness_bonus")
        return bonus

    def compute_delay_penalty(self, *, rounds_used: int) -> float:
        """Compute bounded delay penalty normalized by configured max rounds."""
        self._validate_round(rounds_used)
        delay_ratio = rounds_used / self._max_rounds
        penalty = delay_ratio * self._weights.delay_weight
        _validate_finite(penalty, "delay_penalty")
        return penalty

    def compute_exploitation_penalty(self, *, allocation: dict[str, float], agent_id: str) -> float:
        """Compute exploitation penalty for excessive inequality beyond tolerance."""
        self._validate_allocation(allocation)
        if agent_id not in allocation:
            return 0.0
        total_resource = self._config.environment.total_resource_amount
        equal_share = total_resource / len(allocation)
        imbalance = abs(allocation[agent_id] - equal_share) / total_resource
        excess_imbalance = max(0.0, imbalance - self._exploitation_tolerance)
        normalized_penalty = excess_imbalance / max(1e-9, 1.0 - self._exploitation_tolerance)
        penalty = normalized_penalty * self._weights.exploitation_weight
        _validate_finite(penalty, "exploitation_penalty")
        return penalty

    def compute_no_agreement_penalty(self) -> float:
        """Compute configurable no-agreement terminal penalty."""
        penalty = self._weights.failure_weight
        _validate_finite(penalty, "no_agreement_penalty")
        return penalty

    def aggregate(
        self,
        state: NegotiationState,
        private_utilities: dict[str, float] | None = None,
        proposed_allocation: dict[str, float] | None = None,
    ) -> tuple[dict[str, float], RewardBreakdown]:
        """Aggregate reward for all agents, preserving legacy return signature.

        Returns:
        - per-agent scalar reward dictionary
        - averaged RewardBreakdown across participating agents
        """
        rewards, breakdowns = self.aggregate_with_breakdowns(
            state=state,
            private_utilities=private_utilities,
            proposed_allocation=proposed_allocation,
        )
        average_breakdown = self._average_breakdowns(list(breakdowns.values()))
        return rewards, average_breakdown

    def aggregate_with_breakdowns(
        self,
        *,
        state: NegotiationState,
        private_utilities: dict[str, float] | None = None,
        proposed_allocation: dict[str, float] | None = None,
    ) -> tuple[dict[str, float], dict[str, RewardBreakdown]]:
        """Aggregate rewards and per-agent component attributions."""
        rewards: dict[str, float] = {}
        breakdowns: dict[str, RewardBreakdown] = {}
        utilities = private_utilities or {}
        for agent_id in state.valid_agent_ids:
            private_utility = float(utilities.get(agent_id, 0.0))
            if state.is_terminal:
                total, breakdown = self.compute_terminal_reward(
                    state=state,
                    agent_id=agent_id,
                    private_utility_value=private_utility,
                )
            else:
                total, breakdown = self.compute_intermediate_reward(
                    state=state,
                    agent_id=agent_id,
                    proposed_allocation=proposed_allocation,
                )
            rewards[agent_id] = total
            breakdowns[agent_id] = breakdown
        return rewards, breakdowns

    def summarize_breakdowns(self, breakdowns: dict[str, RewardBreakdown]) -> dict[str, Any]:
        """Create deterministic, JSON-safe diagnostics for reward components."""
        if not breakdowns:
            return {
                "count": 0,
                "total_reward": {"mean": 0.0, "min": 0.0, "max": 0.0},
            }
        values = list(breakdowns.values())
        totals = [item.total_reward for item in values]
        summary = {
            "count": len(values),
            "total_reward": {
                "mean": sum(totals) / len(totals),
                "min": min(totals),
                "max": max(totals),
            },
            "private_utility_reward_mean": self._mean([item.private_utility_reward for item in values]),
            "fairness_bonus_mean": self._mean([item.fairness_bonus for item in values]),
            "delay_penalty_mean": self._mean([item.delay_penalty for item in values]),
            "exploitation_penalty_mean": self._mean([item.exploitation_penalty for item in values]),
            "no_agreement_penalty_mean": self._mean([item.no_agreement_penalty for item in values]),
        }
        return summary

    def _average_breakdowns(self, breakdowns: list[RewardBreakdown]) -> RewardBreakdown:
        """Average reward breakdowns for backward-compatible aggregate output."""
        if not breakdowns:
            return RewardBreakdown(
                private_utility_reward=0.0,
                fairness_bonus=0.0,
                delay_penalty=0.0,
                exploitation_penalty=0.0,
                no_agreement_penalty=0.0,
                total_reward=0.0,
            )
        return RewardBreakdown(
            private_utility_reward=self._mean([item.private_utility_reward for item in breakdowns]),
            fairness_bonus=self._mean([item.fairness_bonus for item in breakdowns]),
            delay_penalty=self._mean([item.delay_penalty for item in breakdowns]),
            exploitation_penalty=self._mean([item.exploitation_penalty for item in breakdowns]),
            no_agreement_penalty=self._mean([item.no_agreement_penalty for item in breakdowns]),
            total_reward=self._mean([item.total_reward for item in breakdowns]),
        )

    @staticmethod
    def _mean(values: list[float]) -> float:
        """Compute mean for a non-empty numeric list."""
        return sum(values) / len(values)

    def _clip_reward(self, reward: float) -> float:
        """Optionally clip scalar reward into configured stability bounds."""
        _validate_finite(reward, "reward")
        if not self._clip_enabled:
            return reward
        lower, upper = self._clip_bounds
        clipped = min(max(reward, lower), upper)
        _validate_finite(clipped, "clipped_reward")
        return clipped

    def _validate_clip_bounds(self) -> None:
        """Validate configured clipping bounds are finite and ordered."""
        lower, upper = self._clip_bounds
        _validate_finite(lower, "reward_clip_bounds.lower")
        _validate_finite(upper, "reward_clip_bounds.upper")
        if lower >= upper:
            raise ValueError(
                "Reward clip bounds must satisfy lower < upper. "
                f"Got {self._clip_bounds}."
            )

    def _validate_agent_id(self, state: NegotiationState, agent_id: str) -> None:
        """Validate provided agent id is part of the current negotiation state."""
        if agent_id not in state.valid_agent_ids:
            raise ValueError(
                f"Unknown agent_id '{agent_id}'. Expected one of {state.valid_agent_ids}."
            )

    def _validate_round(self, round_index: int) -> None:
        """Validate round index against configured environment constraints."""
        if round_index < 0:
            raise ValueError(f"round_index must be non-negative. Got {round_index}.")
        if round_index > self._max_rounds:
            raise ValueError(
                "round_index exceeds configured max rounds. "
                f"round_index={round_index}, max_rounds={self._max_rounds}."
            )

    def _validate_allocation(self, allocation: dict[str, float]) -> None:
        """Validate allocation is finite, non-negative, and resource bounded."""
        if not allocation:
            raise ValueError("allocation must not be empty.")
        total_allocated = 0.0
        for agent_id, value in allocation.items():
            if not agent_id.strip():
                raise ValueError("allocation contains an empty agent id.")
            if value < 0:
                raise ValueError(
                    "allocation values must be non-negative. "
                    f"agent_id={agent_id}, value={value}."
                )
            _validate_finite(value, f"allocation[{agent_id}]")
            total_allocated += value
        if total_allocated > self._config.environment.total_resource_amount:
            raise ValueError(
                "Total allocation exceeds configured total resource amount. "
                f"total_allocated={total_allocated}, "
                f"total_resource_amount={self._config.environment.total_resource_amount}."
            )


class RewardAggregator(RewardEngine):
    """Backward-compatible alias for legacy reward integration points."""

    def fairness_reward(self, state: NegotiationState) -> float:
        """Return average fairness bonus across state outcome allocation."""
        allocation = None if state.outcome is None else state.outcome.agreed_allocation
        if allocation is None:
            return 0.0
        bonuses = [
            self.compute_fairness_bonus(allocation=allocation, agent_id=agent_id)
            for agent_id in state.valid_agent_ids
        ]
        return sum(bonuses) / len(bonuses)

    def exploitation_penalty(self, state: NegotiationState) -> float:
        """Return average exploitation penalty across state outcome allocation."""
        allocation = None if state.outcome is None else state.outcome.agreed_allocation
        if allocation is None:
            return 0.0
        penalties = [
            self.compute_exploitation_penalty(allocation=allocation, agent_id=agent_id)
            for agent_id in state.valid_agent_ids
        ]
        return sum(penalties) / len(penalties)
