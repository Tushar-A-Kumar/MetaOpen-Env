"""Partial-observation construction for OpenBargain agents.

This module converts full negotiation state into agent-local observations with
strict hidden-information masking. It exposes deterministic, fixed-structure
policy inputs suitable for PPO/GRPO/self-play and LLM rollouts.
"""

from dataclasses import dataclass
import math
from typing import Any

from open_bargain.config import OpenBargainConfig
from open_bargain.env.state import ActionRecord, NegotiationState, OfferRecord
from open_bargain.env.utility import PreferenceProfile


def _validate_non_empty(value: str, field_name: str) -> None:
    """Validate string value is non-empty."""
    if not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


def _validate_finite(value: float, field_name: str) -> None:
    """Validate numeric value is finite."""
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite. Got {value}.")


@dataclass(slots=True, frozen=True)
class PublicObservation:
    """Public negotiation features visible to all agents."""

    current_round: int
    current_step: int
    active_proposer_id: str
    remaining_resource: float
    remaining_resource_normalized: float
    current_public_offer: dict[str, float]
    rounds_remaining: int
    round_progress: float
    is_terminal: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize public observation to deterministic JSON-safe dictionary."""
        return {
            "current_round": self.current_round,
            "current_step": self.current_step,
            "active_proposer_id": self.active_proposer_id,
            "remaining_resource": self.remaining_resource,
            "remaining_resource_normalized": self.remaining_resource_normalized,
            "current_public_offer": dict(self.current_public_offer),
            "rounds_remaining": self.rounds_remaining,
            "round_progress": self.round_progress,
            "is_terminal": self.is_terminal,
        }


@dataclass(slots=True, frozen=True)
class PrivateObservation:
    """Agent-private observation slice containing only local hidden context."""

    observing_agent_id: str
    private_utility_profile_summary: dict[str, float]
    local_context: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize private observation to deterministic JSON-safe dictionary."""
        return {
            "observing_agent_id": self.observing_agent_id,
            "private_utility_profile_summary": dict(self.private_utility_profile_summary),
            "local_context": dict(self.local_context),
        }


@dataclass(slots=True, frozen=True)
class ObservationSummary:
    """Compact fixed-size summary of recent negotiation history."""

    last_offer: dict[str, float]
    recent_action_summary: dict[str, int]
    offer_count: int
    rejection_count: int
    acceptance_flag: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize summary to deterministic JSON-safe dictionary."""
        return {
            "last_offer": dict(self.last_offer),
            "recent_action_summary": dict(self.recent_action_summary),
            "offer_count": self.offer_count,
            "rejection_count": self.rejection_count,
            "acceptance_flag": self.acceptance_flag,
        }


@dataclass(slots=True, frozen=True)
class AgentObservation:
    """Canonical agent observation containing all policy-visible sections."""

    public: PublicObservation
    private: PrivateObservation
    summary: ObservationSummary
    valid_action_mask: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        """Serialize canonical observation for logging and trace export."""
        return {
            "public": self.public.to_dict(),
            "private": self.private.to_dict(),
            "summary": self.summary.to_dict(),
            "valid_action_mask": dict(self.valid_action_mask),
        }

    def to_policy_input(self) -> dict[str, Any]:
        """Export policy-ready observation payload in deterministic structure."""
        return self.to_dict()

    def compact_view(self) -> dict[str, Any]:
        """Export compact observation view for lightweight trace logging."""
        return {
            "round": self.public.current_round,
            "step": self.public.current_step,
            "agent_id": self.private.observing_agent_id,
            "round_progress": self.public.round_progress,
            "remaining_resource_normalized": self.public.remaining_resource_normalized,
            "offer_count": self.summary.offer_count,
            "rejection_count": self.summary.rejection_count,
            "acceptance_flag": self.summary.acceptance_flag,
            "is_terminal": self.public.is_terminal,
        }


LocalObservation = AgentObservation


class ObservationBuilder:
    """Constructs deterministic partial observations from negotiation state."""

    def __init__(self, config: OpenBargainConfig) -> None:
        """Initialize builder with centralized benchmark configuration."""
        self._config = config
        self._normalize = config.environment.observation_normalization_enabled
        self._total_resource = config.environment.total_resource_amount
        self._max_rounds = config.environment.max_negotiation_rounds
        self._history_window = config.environment.history_summary_window
        self._action_types = config.environment.action_types

    def build(
        self,
        state: NegotiationState,
        agent_id: str,
        profile: PreferenceProfile | None = None,
    ) -> LocalObservation:
        """Build deterministic agent-local observation under partial observability."""
        self._validate_agent_id(state, agent_id)
        public_obs = self._extract_public_observation(state=state)
        private_obs = self._extract_private_observation(
            state=state,
            observing_agent_id=agent_id,
            profile=profile,
        )
        summary = self._summarize_history(state=state)
        valid_action_mask = self._build_action_mask(state=state, observing_agent_id=agent_id)
        self._validate_action_mask(valid_action_mask)
        return AgentObservation(
            public=public_obs,
            private=private_obs,
            summary=summary,
            valid_action_mask=valid_action_mask,
        )

    def mask_opponent_information(
        self,
        state: NegotiationState,
        observer_agent_id: str,
    ) -> dict[str, Any]:
        """Return explicit mask metadata proving private-opponent data is withheld."""
        self._validate_agent_id(state, observer_agent_id)
        return {
            "observer_agent_id": observer_agent_id,
            "hidden_fields": (
                "opponent_greed_sensitivity",
                "opponent_fairness_sensitivity",
                "opponent_urgency_sensitivity",
                "opponent_private_profile",
            ),
        }

    def _extract_public_observation(self, state: NegotiationState) -> PublicObservation:
        """Extract public state features visible to all agents."""
        current_offer = self._zero_allocation(state.valid_agent_ids)
        if state.current_active_offer is not None:
            current_offer = self._normalize_allocation(state.current_active_offer.proposed_allocation, state.valid_agent_ids)
        rounds_remaining = max(0, self._max_rounds - state.current_round)
        round_progress = self._normalize_ratio(state.current_round, self._max_rounds)
        remaining_resource_normalized = self._normalize_ratio(
            state.remaining_resource,
            self._total_resource,
        )
        public_obs = PublicObservation(
            current_round=state.current_round,
            current_step=state.current_step,
            active_proposer_id=state.active_proposer_id,
            remaining_resource=state.remaining_resource,
            remaining_resource_normalized=remaining_resource_normalized,
            current_public_offer=current_offer,
            rounds_remaining=rounds_remaining,
            round_progress=round_progress,
            is_terminal=state.is_terminal,
        )
        _validate_finite(public_obs.round_progress, "PublicObservation.round_progress")
        _validate_finite(
            public_obs.remaining_resource_normalized,
            "PublicObservation.remaining_resource_normalized",
        )
        return public_obs

    def _extract_private_observation(
        self,
        *,
        state: NegotiationState,
        observing_agent_id: str,
        profile: PreferenceProfile | None,
    ) -> PrivateObservation:
        """Extract observing-agent-only features and preserve hidden-information safety."""
        profile_summary: dict[str, float] = {
            "greed_sensitivity": 0.0,
            "fairness_sensitivity": 0.0,
            "urgency_sensitivity": 0.0,
        }
        if profile is not None:
            if profile.agent_id != observing_agent_id:
                raise ValueError(
                    "PreferenceProfile.agent_id must match observing_agent_id. "
                    f"profile.agent_id={profile.agent_id}, observing_agent_id={observing_agent_id}."
                )
            profile_summary = {
                "greed_sensitivity": profile.greed_sensitivity,
                "fairness_sensitivity": profile.fairness_sensitivity,
                "urgency_sensitivity": profile.urgency_sensitivity,
            }
        local_context = {
            "is_active_proposer": state.active_proposer_id == observing_agent_id,
            "agent_offer_count": self._count_agent_offers(state.offer_history, observing_agent_id),
            "agent_action_count": self._count_agent_actions(state.action_history, observing_agent_id),
        }
        return PrivateObservation(
            observing_agent_id=observing_agent_id,
            private_utility_profile_summary=profile_summary,
            local_context=local_context,
        )

    def _summarize_history(self, state: NegotiationState) -> ObservationSummary:
        """Produce compact fixed-size summary features from trajectory history."""
        last_offer = self._zero_allocation(state.valid_agent_ids)
        if state.offer_history:
            last_offer = self._normalize_allocation(state.offer_history[-1].proposed_allocation, state.valid_agent_ids)
        recent_actions = state.action_history[-self._history_window :]
        recent_action_summary = {action_type: 0 for action_type in self._action_types}
        rejection_count = 0
        for action in recent_actions:
            if action.action_type in recent_action_summary:
                recent_action_summary[action.action_type] += 1
            if action.action_type == "reject":
                rejection_count += 1
        acceptance_flag = any(action.action_type == "accept" for action in recent_actions)
        return ObservationSummary(
            last_offer=last_offer,
            recent_action_summary=recent_action_summary,
            offer_count=len(state.offer_history),
            rejection_count=rejection_count,
            acceptance_flag=acceptance_flag,
        )

    def _build_action_mask(
        self,
        *,
        state: NegotiationState,
        observing_agent_id: str,
    ) -> dict[str, bool]:
        """Build extensible valid-action mask placeholder for constrained actions."""
        if state.is_terminal:
            return {action: False for action in self._action_types}
        is_proposer = state.active_proposer_id == observing_agent_id
        has_offer = state.current_active_offer is not None
        mask = {
            "propose": is_proposer,
            "accept": (not is_proposer) and has_offer,
            "reject": (not is_proposer) and has_offer,
            "counteroffer": (not is_proposer) and has_offer,
        }
        for action_type in self._action_types:
            mask.setdefault(action_type, False)
        return {action_type: mask[action_type] for action_type in self._action_types}

    def _validate_agent_id(self, state: NegotiationState, agent_id: str) -> None:
        """Validate observing agent belongs to state-defined participants."""
        _validate_non_empty(agent_id, "agent_id")
        if agent_id not in state.valid_agent_ids:
            raise ValueError(
                f"Unknown observing agent_id '{agent_id}'. Expected one of {state.valid_agent_ids}."
            )

    def _validate_action_mask(self, action_mask: dict[str, bool]) -> None:
        """Validate action mask keys and values are consistent."""
        expected = set(self._action_types)
        actual = set(action_mask.keys())
        if expected != actual:
            raise ValueError(
                "Action mask keys mismatch configured action types. "
                f"expected={sorted(expected)}, actual={sorted(actual)}."
            )
        for action_name, is_valid in action_mask.items():
            if not isinstance(is_valid, bool):
                raise ValueError(
                    f"Action mask value for '{action_name}' must be bool. Got {type(is_valid).__name__}."
                )

    def _normalize_ratio(self, value: float, denominator: float) -> float:
        """Normalize ratio to [0, 1] when enabled, else return raw ratio."""
        if denominator <= 0:
            raise ValueError(f"denominator must be > 0 for normalization. Got {denominator}.")
        ratio = value / denominator
        _validate_finite(ratio, "ratio")
        if not self._normalize:
            return ratio
        clipped = min(max(ratio, 0.0), 1.0)
        _validate_finite(clipped, "clipped_ratio")
        return clipped

    @staticmethod
    def _count_agent_offers(offers: list[OfferRecord], agent_id: str) -> int:
        """Count offers proposed by one agent for local context."""
        return sum(1 for offer in offers if offer.proposer_agent_id == agent_id)

    @staticmethod
    def _count_agent_actions(actions: list[ActionRecord], agent_id: str) -> int:
        """Count actions emitted by one agent for local context."""
        return sum(1 for action in actions if action.acting_agent_id == agent_id)

    @staticmethod
    def _zero_allocation(agent_ids: tuple[str, ...]) -> dict[str, float]:
        """Create a deterministic zero allocation for all valid agents."""
        return {agent_id: 0.0 for agent_id in agent_ids}

    @staticmethod
    def _normalize_allocation(allocation: dict[str, float], agent_ids: tuple[str, ...]) -> dict[str, float]:
        """Normalize a sparse allocation to include all agents with stable key order."""
        normalized = {agent_id: float(allocation.get(agent_id, 0.0)) for agent_id in agent_ids}
        return normalized
