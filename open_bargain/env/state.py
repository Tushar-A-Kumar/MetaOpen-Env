"""Canonical negotiation state management for OpenBargain.

This module defines deterministic, serializable state contracts for bargaining
episodes. It tracks progression, proposer turns, offer and action histories,
and terminal outcomes without introducing reward, utility, or environment logic.
"""

from dataclasses import dataclass, field
from typing import Any


def _validate_non_negative(value: int | float, field_name: str) -> None:
    """Validate a numeric field is non-negative."""
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative. Got {value}.")


def _validate_positive(value: int | float, field_name: str) -> None:
    """Validate a numeric field is strictly positive."""
    if value <= 0:
        raise ValueError(f"{field_name} must be > 0. Got {value}.")


def _validate_non_empty(value: str, field_name: str) -> None:
    """Validate a string field is non-empty after trimming."""
    if not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


@dataclass(slots=True, frozen=True)
class OfferRecord:
    """Immutable representation of a single negotiation proposal.

    Fields:
    - round_number: Round in which the offer was emitted.
    - proposer_agent_id: Agent that proposed the offer.
    - proposed_allocation: Proposed split/allocation payload.
    - step_index: Global step index when the offer was made.
    - accepted: Whether the offer was accepted in this step.
    - rejected: Whether the offer was rejected in this step.
    """

    round_number: int
    proposer_agent_id: str
    proposed_allocation: dict[str, float]
    step_index: int
    accepted: bool = False
    rejected: bool = False

    def __post_init__(self) -> None:
        """Validate immutable offer record consistency."""
        _validate_non_negative(self.round_number, "OfferRecord.round_number")
        _validate_non_negative(self.step_index, "OfferRecord.step_index")
        _validate_non_empty(self.proposer_agent_id, "OfferRecord.proposer_agent_id")
        if not self.proposed_allocation:
            raise ValueError("OfferRecord.proposed_allocation must not be empty.")
        for key, value in self.proposed_allocation.items():
            _validate_non_empty(key, "OfferRecord.proposed_allocation key")
            _validate_non_negative(value, "OfferRecord.proposed_allocation value")
        if self.accepted and self.rejected:
            raise ValueError("OfferRecord cannot be both accepted and rejected.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize offer record to a JSON-safe dictionary."""
        return {
            "round_number": self.round_number,
            "proposer_agent_id": self.proposer_agent_id,
            "proposed_allocation": dict(self.proposed_allocation),
            "step_index": self.step_index,
            "accepted": self.accepted,
            "rejected": self.rejected,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OfferRecord":
        """Deserialize an offer record from a dictionary payload."""
        return cls(
            round_number=int(payload["round_number"]),
            proposer_agent_id=str(payload["proposer_agent_id"]),
            proposed_allocation={str(k): float(v) for k, v in dict(payload["proposed_allocation"]).items()},
            step_index=int(payload["step_index"]),
            accepted=bool(payload.get("accepted", False)),
            rejected=bool(payload.get("rejected", False)),
        )


@dataclass(slots=True, frozen=True)
class ActionRecord:
    """Immutable representation of one agent action in trajectory history.

    Fields:
    - acting_agent_id: Agent producing the action.
    - action_type: Semantic action label (e.g., propose/accept/reject).
    - action_payload: JSON-safe action payload.
    - round_number: Round where the action occurred.
    - step_index: Global step index for ordering and replay.
    """

    acting_agent_id: str
    action_type: str
    action_payload: dict[str, Any]
    round_number: int
    step_index: int

    def __post_init__(self) -> None:
        """Validate immutable action record consistency."""
        _validate_non_empty(self.acting_agent_id, "ActionRecord.acting_agent_id")
        _validate_non_empty(self.action_type, "ActionRecord.action_type")
        _validate_non_negative(self.round_number, "ActionRecord.round_number")
        _validate_non_negative(self.step_index, "ActionRecord.step_index")

    def to_dict(self) -> dict[str, Any]:
        """Serialize action record to a JSON-safe dictionary."""
        return {
            "acting_agent_id": self.acting_agent_id,
            "action_type": self.action_type,
            "action_payload": dict(self.action_payload),
            "round_number": self.round_number,
            "step_index": self.step_index,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ActionRecord":
        """Deserialize an action record from a dictionary payload."""
        return cls(
            acting_agent_id=str(payload["acting_agent_id"]),
            action_type=str(payload["action_type"]),
            action_payload=dict(payload.get("action_payload", {})),
            round_number=int(payload["round_number"]),
            step_index=int(payload["step_index"]),
        )


@dataclass(slots=True, frozen=True)
class NegotiationOutcome:
    """Terminal outcome metadata for benchmark evaluation and logging.

    Fields:
    - agreement_reached: Whether negotiation ended in agreement.
    - agreed_allocation: Final allocation if agreement exists.
    - total_rounds_used: Number of rounds consumed in the episode.
    - termination_reason: Canonical textual reason for termination.
    - final_remaining_resource: Resource left at termination.
    """

    agreement_reached: bool
    agreed_allocation: dict[str, float] | None
    total_rounds_used: int
    termination_reason: str
    final_remaining_resource: float

    def __post_init__(self) -> None:
        """Validate terminal outcome consistency."""
        _validate_non_negative(self.total_rounds_used, "NegotiationOutcome.total_rounds_used")
        _validate_non_negative(
            self.final_remaining_resource,
            "NegotiationOutcome.final_remaining_resource",
        )
        _validate_non_empty(self.termination_reason, "NegotiationOutcome.termination_reason")
        if self.agreement_reached and self.agreed_allocation is None:
            raise ValueError(
                "NegotiationOutcome.agreed_allocation must be provided when agreement_reached is True."
            )
        if not self.agreement_reached and self.agreed_allocation is not None:
            raise ValueError(
                "NegotiationOutcome.agreed_allocation must be None when agreement_reached is False."
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize outcome to a JSON-safe dictionary."""
        return {
            "agreement_reached": self.agreement_reached,
            "agreed_allocation": None
            if self.agreed_allocation is None
            else dict(self.agreed_allocation),
            "total_rounds_used": self.total_rounds_used,
            "termination_reason": self.termination_reason,
            "final_remaining_resource": self.final_remaining_resource,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NegotiationOutcome":
        """Deserialize outcome from a dictionary payload."""
        agreed_allocation_payload = payload.get("agreed_allocation")
        agreed_allocation = None
        if agreed_allocation_payload is not None:
            agreed_allocation = {
                str(k): float(v) for k, v in dict(agreed_allocation_payload).items()
            }
        return cls(
            agreement_reached=bool(payload["agreement_reached"]),
            agreed_allocation=agreed_allocation,
            total_rounds_used=int(payload["total_rounds_used"]),
            termination_reason=str(payload["termination_reason"]),
            final_remaining_resource=float(payload["final_remaining_resource"]),
        )


@dataclass(slots=True)
class NegotiationState:
    """Canonical mutable state for one bargaining episode lifecycle.

    Fields:
    - episode_id: Unique episode identifier for traceability.
    - valid_agent_ids: Ordered set of legal agent IDs for proposer validation.
    - max_rounds: Maximum rounds allowed by environment config.
    - current_round: Current round number.
    - current_step: Current global step index.
    - active_proposer_id: Agent currently responsible for proposal.
    - remaining_resource: Current remaining resource budget.
    - current_active_offer: Most recent active offer, if any.
    - offer_history: Ordered historical immutable offers.
    - action_history: Ordered historical immutable actions.
    - agreement_reached: Whether agreement has been achieved.
    - is_terminal: Whether state has reached terminal condition.
    - outcome: Terminal outcome metadata, if available.
    """

    episode_id: str
    valid_agent_ids: tuple[str, ...]
    max_rounds: int
    current_round: int
    current_step: int
    active_proposer_id: str
    remaining_resource: float
    current_active_offer: OfferRecord | None = None
    offer_history: list[OfferRecord] = field(default_factory=list)
    action_history: list[ActionRecord] = field(default_factory=list)
    agreement_reached: bool = False
    is_terminal: bool = False
    outcome: NegotiationOutcome | None = None

    def __post_init__(self) -> None:
        """Validate initial state contract invariants."""
        _validate_non_empty(self.episode_id, "NegotiationState.episode_id")
        if not self.valid_agent_ids:
            raise ValueError("NegotiationState.valid_agent_ids must not be empty.")
        for agent_id in self.valid_agent_ids:
            _validate_non_empty(agent_id, "NegotiationState.valid_agent_ids item")
        _validate_positive(self.max_rounds, "NegotiationState.max_rounds")
        _validate_non_negative(self.current_round, "NegotiationState.current_round")
        _validate_non_negative(self.current_step, "NegotiationState.current_step")
        _validate_non_negative(self.remaining_resource, "NegotiationState.remaining_resource")
        self._validate_agent_id(self.active_proposer_id)
        self._validate_history_consistency()
        if self.outcome is not None and not self.is_terminal:
            raise ValueError("NegotiationState.outcome requires is_terminal=True.")

    @classmethod
    def initialize(
        cls,
        *,
        episode_id: str,
        valid_agent_ids: tuple[str, ...],
        max_rounds: int,
        initial_proposer_id: str,
        initial_resource: float,
    ) -> "NegotiationState":
        """Create a fresh deterministic initial state for episode reset boundaries."""
        return cls(
            episode_id=episode_id,
            valid_agent_ids=valid_agent_ids,
            max_rounds=max_rounds,
            current_round=0,
            current_step=0,
            active_proposer_id=initial_proposer_id,
            remaining_resource=initial_resource,
        )

    def increment_step(self) -> None:
        """Increment global step index by one."""
        self._assert_mutable("increment_step")
        self.current_step += 1

    def advance_round(self) -> None:
        """Advance to the next negotiation round with bounds checking."""
        self._assert_mutable("advance_round")
        next_round = self.current_round + 1
        if next_round > self.max_rounds:
            raise ValueError(
                "Cannot advance round beyond max_rounds. "
                f"current_round={self.current_round}, max_rounds={self.max_rounds}."
            )
        self.current_round = next_round

    def switch_proposer(self, next_proposer_id: str) -> None:
        """Switch the active proposer to another valid agent."""
        self._assert_mutable("switch_proposer")
        self._validate_agent_id(next_proposer_id)
        self.active_proposer_id = next_proposer_id

    def update_current_offer(self, offer: OfferRecord | None) -> None:
        """Update the currently active offer pointer."""
        self._assert_mutable("update_current_offer")
        if offer is not None:
            self._validate_offer_record(offer)
        self.current_active_offer = offer

    def append_offer_record(self, offer: OfferRecord) -> None:
        """Append an immutable offer record while preserving deterministic order."""
        self._assert_mutable("append_offer_record")
        self._validate_offer_record(offer)
        self._validate_offer_sequence(offer)
        self.offer_history.append(offer)
        self.current_active_offer = offer

    def append_action_record(self, action: ActionRecord) -> None:
        """Append an immutable action record while preserving deterministic order."""
        self._assert_mutable("append_action_record")
        self._validate_action_record(action)
        self._validate_action_sequence(action)
        self.action_history.append(action)

    def mark_agreement_reached(
        self,
        *,
        agreed_allocation: dict[str, float],
        termination_reason: str,
    ) -> None:
        """Mark state terminal with a successful agreement outcome."""
        self._assert_mutable("mark_agreement_reached")
        if not agreed_allocation:
            raise ValueError("agreed_allocation must not be empty when agreement is reached.")
        self.agreement_reached = True
        self.is_terminal = True
        self.outcome = NegotiationOutcome(
            agreement_reached=True,
            agreed_allocation={str(k): float(v) for k, v in agreed_allocation.items()},
            total_rounds_used=self.current_round,
            termination_reason=termination_reason,
            final_remaining_resource=self.remaining_resource,
        )

    def mark_negotiation_failed(self, *, termination_reason: str) -> None:
        """Mark state terminal with a failed negotiation outcome."""
        self._assert_mutable("mark_negotiation_failed")
        self.agreement_reached = False
        self.is_terminal = True
        self.outcome = NegotiationOutcome(
            agreement_reached=False,
            agreed_allocation=None,
            total_rounds_used=self.current_round,
            termination_reason=termination_reason,
            final_remaining_resource=self.remaining_resource,
        )

    def attach_outcome(self, outcome: NegotiationOutcome) -> None:
        """Attach terminal outcome metadata to an already terminal state."""
        if not self.is_terminal:
            raise ValueError("Cannot attach outcome before state is marked terminal.")
        if self.outcome is not None:
            raise ValueError("Outcome is already attached and cannot be overwritten.")
        self.outcome = outcome
        self.agreement_reached = outcome.agreement_reached

    def set_remaining_resource(self, value: float) -> None:
        """Set remaining resource with non-negative validation."""
        self._assert_mutable("set_remaining_resource")
        _validate_non_negative(value, "NegotiationState.remaining_resource")
        self.remaining_resource = value

    def reset(
        self,
        *,
        episode_id: str,
        initial_proposer_id: str,
        initial_resource: float,
    ) -> None:
        """Reset this state instance to a clean deterministic episode start."""
        _validate_non_empty(episode_id, "NegotiationState.reset.episode_id")
        self._validate_agent_id(initial_proposer_id)
        _validate_non_negative(initial_resource, "NegotiationState.reset.initial_resource")
        self.episode_id = episode_id
        self.current_round = 0
        self.current_step = 0
        self.active_proposer_id = initial_proposer_id
        self.remaining_resource = initial_resource
        self.current_active_offer = None
        self.offer_history.clear()
        self.action_history.clear()
        self.agreement_reached = False
        self.is_terminal = False
        self.outcome = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize state into a deterministic, JSON-safe dictionary."""
        return {
            "episode_id": self.episode_id,
            "valid_agent_ids": list(self.valid_agent_ids),
            "max_rounds": self.max_rounds,
            "current_round": self.current_round,
            "current_step": self.current_step,
            "active_proposer_id": self.active_proposer_id,
            "remaining_resource": self.remaining_resource,
            "current_active_offer": None
            if self.current_active_offer is None
            else self.current_active_offer.to_dict(),
            "offer_history": [record.to_dict() for record in self.offer_history],
            "action_history": [record.to_dict() for record in self.action_history],
            "agreement_reached": self.agreement_reached,
            "is_terminal": self.is_terminal,
            "outcome": None if self.outcome is None else self.outcome.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NegotiationState":
        """Reconstruct state from a dictionary payload."""
        current_offer_payload = payload.get("current_active_offer")
        outcome_payload = payload.get("outcome")
        state = cls(
            episode_id=str(payload["episode_id"]),
            valid_agent_ids=tuple(str(value) for value in list(payload["valid_agent_ids"])),
            max_rounds=int(payload["max_rounds"]),
            current_round=int(payload["current_round"]),
            current_step=int(payload["current_step"]),
            active_proposer_id=str(payload["active_proposer_id"]),
            remaining_resource=float(payload["remaining_resource"]),
            current_active_offer=None
            if current_offer_payload is None
            else OfferRecord.from_dict(dict(current_offer_payload)),
            offer_history=[
                OfferRecord.from_dict(dict(item)) for item in list(payload.get("offer_history", []))
            ],
            action_history=[
                ActionRecord.from_dict(dict(item)) for item in list(payload.get("action_history", []))
            ],
            agreement_reached=bool(payload.get("agreement_reached", False)),
            is_terminal=bool(payload.get("is_terminal", False)),
            outcome=None
            if outcome_payload is None
            else NegotiationOutcome.from_dict(dict(outcome_payload)),
        )
        return state

    def summary_snapshot(self) -> dict[str, Any]:
        """Return compact summary for lightweight logging and debug traces."""
        return {
            "episode_id": self.episode_id,
            "round": self.current_round,
            "step": self.current_step,
            "active_proposer_id": self.active_proposer_id,
            "remaining_resource": self.remaining_resource,
            "is_terminal": self.is_terminal,
            "agreement_reached": self.agreement_reached,
            "offer_count": len(self.offer_history),
            "action_count": len(self.action_history),
            "termination_reason": None if self.outcome is None else self.outcome.termination_reason,
        }

    def _assert_mutable(self, operation_name: str) -> None:
        """Prevent invalid post-terminal state mutations."""
        if self.is_terminal:
            raise RuntimeError(
                f"Cannot execute '{operation_name}' on terminal NegotiationState."
            )

    def _validate_agent_id(self, agent_id: str) -> None:
        """Validate agent id belongs to valid agent set."""
        _validate_non_empty(agent_id, "agent_id")
        if agent_id not in self.valid_agent_ids:
            raise ValueError(
                f"Invalid proposer/agent id '{agent_id}'. "
                f"Expected one of {self.valid_agent_ids}."
            )

    def _validate_offer_record(self, offer: OfferRecord) -> None:
        """Validate offer compatibility with current state."""
        self._validate_agent_id(offer.proposer_agent_id)
        if offer.round_number != self.current_round:
            raise ValueError(
                "OfferRecord.round_number must equal current_round. "
                f"offer.round_number={offer.round_number}, current_round={self.current_round}."
            )
        if offer.step_index != self.current_step:
            raise ValueError(
                "OfferRecord.step_index must equal current_step. "
                f"offer.step_index={offer.step_index}, current_step={self.current_step}."
            )

    def _validate_action_record(self, action: ActionRecord) -> None:
        """Validate action compatibility with current state."""
        self._validate_agent_id(action.acting_agent_id)
        if action.round_number != self.current_round:
            raise ValueError(
                "ActionRecord.round_number must equal current_round. "
                f"action.round_number={action.round_number}, current_round={self.current_round}."
            )
        if action.step_index != self.current_step:
            raise ValueError(
                "ActionRecord.step_index must equal current_step. "
                f"action.step_index={action.step_index}, current_step={self.current_step}."
            )

    def _validate_offer_sequence(self, offer: OfferRecord) -> None:
        """Validate deterministic offer history ordering."""
        if not self.offer_history:
            return
        previous = self.offer_history[-1]
        if offer.step_index < previous.step_index:
            raise ValueError(
                "Offer history order violation: step_index must be non-decreasing. "
                f"previous={previous.step_index}, new={offer.step_index}."
            )

    def _validate_action_sequence(self, action: ActionRecord) -> None:
        """Validate deterministic action history ordering."""
        if not self.action_history:
            return
        previous = self.action_history[-1]
        if action.step_index < previous.step_index:
            raise ValueError(
                "Action history order violation: step_index must be non-decreasing. "
                f"previous={previous.step_index}, new={action.step_index}."
            )

    def _validate_history_consistency(self) -> None:
        """Validate preloaded offer/action histories for ordering and consistency."""
        for index in range(1, len(self.offer_history)):
            if self.offer_history[index].step_index < self.offer_history[index - 1].step_index:
                raise ValueError(
                    "NegotiationState.offer_history must be ordered by non-decreasing step_index."
                )
        for index in range(1, len(self.action_history)):
            if self.action_history[index].step_index < self.action_history[index - 1].step_index:
                raise ValueError(
                    "NegotiationState.action_history must be ordered by non-decreasing step_index."
                )
