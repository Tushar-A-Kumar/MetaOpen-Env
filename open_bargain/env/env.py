"""OpenEnv-compatible core environment for OpenBargain negotiation episodes."""

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
from gymnasium import spaces

from open_bargain.config import OpenBargainConfig
from open_bargain.env.observation import ObservationBuilder
from open_bargain.env.reward import RewardAggregator
from open_bargain.env.state import ActionRecord, NegotiationState, OfferRecord
from open_bargain.env.utility import (
    DeterministicPreferenceGenerator,
    PreferenceProfile,
    UtilityEvaluator,
)


class OpenBargainEnv(gym.Env[dict[str, Any], dict[str, Any]]):
    """Main environment orchestrating state, utility, reward, and observations."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: OpenBargainConfig,
        reward_aggregator: RewardAggregator | None = None,
        observation_builder: ObservationBuilder | None = None,
        utility_evaluator: UtilityEvaluator | None = None,
    ) -> None:
        """Initialize benchmark environment and subsystem dependencies."""
        super().__init__()
        self.config = config
        self.reward_aggregator = reward_aggregator or RewardAggregator(config=config)
        self.observation_builder = observation_builder or ObservationBuilder(config=config)
        self.utility_evaluator = utility_evaluator or UtilityEvaluator(config=config)
        self._state: NegotiationState | None = None
        self._profiles: dict[str, PreferenceProfile] = {}
        self._agent_ids: tuple[str, ...] = ()
        self._episode_counter: int = 0
        self._current_seed: int = config.simulation.default_random_seed
        self._action_type_to_index = {
            action_type: index for index, action_type in enumerate(self.config.environment.action_types)
        }
        self._index_to_action_type = {
            index: action_type for action_type, index in self._action_type_to_index.items()
        }
        self.observation_space_spec = self._build_observation_space_spec()
        self.action_space_spec = self._build_action_space_spec()
        self.observation_space = self._build_observation_space()
        self.action_space = self._build_action_space()

    @property
    def state(self) -> NegotiationState:
        """Return current negotiation state or raise if uninitialized."""
        if self._state is None:
            raise RuntimeError("Environment state is not initialized. Call reset() first.")
        return self._state

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset environment episode and return initial multi-agent observations."""
        options = options or {}
        self._current_seed = self.config.simulation.default_random_seed if seed is None else seed
        if self._current_seed < 0:
            raise ValueError(f"seed must be non-negative. Got {self._current_seed}.")
        self._agent_ids = self._resolve_agent_ids(options=options)
        self._episode_counter += 1
        episode_id = f"episode_{self._episode_counter}"
        self._state = NegotiationState.initialize(
            episode_id=episode_id,
            valid_agent_ids=self._agent_ids,
            max_rounds=self.config.environment.max_negotiation_rounds,
            initial_proposer_id=self._agent_ids[0],
            initial_resource=self.config.environment.total_resource_amount,
        )
        generator = DeterministicPreferenceGenerator(
            config=self.config,
            seed=self._current_seed,
        )
        self._profiles = generator.generate(list(self._agent_ids))
        observations = self._build_all_observations()
        info = self._build_info(
            event="reset",
            rewards={agent_id: 0.0 for agent_id in self._agent_ids},
        )
        self._validate_observation_schema(observations)
        return observations, info

    def step(
        self,
        action: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, float], dict[str, bool], dict[str, bool], dict[str, Any]]:
        """Apply one action transition and return OpenEnv-compatible step outputs."""
        state = self.state
        if state.is_terminal:
            raise RuntimeError("Cannot call step() on terminal episode. Call reset().")
        normalized_action = self._normalize_step_action_input(action=action)
        acting_agent_id = self._extract_action_agent_id(normalized_action)
        action_type = self._extract_action_type(normalized_action)
        action_payload = self._extract_action_payload(normalized_action)
        self._validate_action_legality(
            state=state,
            acting_agent_id=acting_agent_id,
            action_type=action_type,
        )
        self._apply_action(
            state=state,
            acting_agent_id=acting_agent_id,
            action_type=action_type,
            action_payload=action_payload,
        )
        if self._is_timeout(state):
            state.mark_negotiation_failed(termination_reason="max_rounds_exceeded")
        rewards, _ = self._compute_rewards(state=state, action_payload=action_payload)
        observations = self._build_all_observations()
        terminated = {agent_id: state.is_terminal for agent_id in self._agent_ids}
        terminated["__all__"] = state.is_terminal
        truncated = {agent_id: False for agent_id in self._agent_ids}
        truncated["__all__"] = False
        info = self._build_info(event="step", rewards=rewards)
        self._validate_step_outputs(observations, rewards, terminated, truncated)
        return observations, rewards, terminated, truncated, info

    def reset_batch(
        self,
        *,
        batch_size: int,
        seeds: list[int | None] | None = None,
        options_list: list[Mapping[str, Any] | None] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Sequential batched reset hook for vectorized rollout integration."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0. Got {batch_size}.")
        if seeds is not None and len(seeds) != batch_size:
            raise ValueError(
                "seeds length must match batch_size. "
                f"Got len(seeds)={len(seeds)}, batch_size={batch_size}."
            )
        if options_list is not None and len(options_list) != batch_size:
            raise ValueError(
                "options_list length must match batch_size. "
                f"Got len(options_list)={len(options_list)}, batch_size={batch_size}."
            )
        observations_batch: list[dict[str, Any]] = []
        infos_batch: list[dict[str, Any]] = []
        for index in range(batch_size):
            seed = None if seeds is None else seeds[index]
            options = None if options_list is None else options_list[index]
            observations, info = self.reset(seed=seed, options=options)
            observations_batch.append(observations)
            infos_batch.append(info)
        return observations_batch, infos_batch

    def step_batch(
        self,
        actions: list[dict[str, Any]] | dict[str, dict[str, Any]],
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, float]],
        list[dict[str, bool]],
        list[dict[str, bool]],
        list[dict[str, Any]],
    ]:
        """Sequential batched step hook compatible with rollout worker interfaces."""
        normalized_actions: list[dict[str, Any]]
        if isinstance(actions, dict):
            normalized_actions = [value for _, value in sorted(actions.items(), key=lambda item: item[0])]
        elif isinstance(actions, list):
            normalized_actions = actions
        else:
            raise ValueError("actions must be a list or dict for step_batch.")
        if not normalized_actions:
            raise ValueError("step_batch requires at least one action.")
        observations_batch: list[dict[str, Any]] = []
        rewards_batch: list[dict[str, float]] = []
        terminated_batch: list[dict[str, bool]] = []
        truncated_batch: list[dict[str, bool]] = []
        infos_batch: list[dict[str, Any]] = []
        for action in normalized_actions:
            observations, rewards, terminated, truncated, info = self.step(action)
            observations_batch.append(observations)
            rewards_batch.append(rewards)
            terminated_batch.append(terminated)
            truncated_batch.append(truncated)
            infos_batch.append(info)
        return (
            observations_batch,
            rewards_batch,
            terminated_batch,
            truncated_batch,
            infos_batch,
        )

    def _apply_action(
        self,
        *,
        state: NegotiationState,
        acting_agent_id: str,
        action_type: str,
        action_payload: dict[str, Any],
    ) -> None:
        """Apply validated action to state with deterministic transition ordering."""
        state.append_action_record(
            ActionRecord(
                acting_agent_id=acting_agent_id,
                action_type=action_type,
                action_payload=action_payload,
                round_number=state.current_round,
                step_index=state.current_step,
            )
        )
        if action_type in ("propose", "counteroffer"):
            allocation = self._extract_allocation(action_payload)
            state.append_offer_record(
                OfferRecord(
                    round_number=state.current_round,
                    proposer_agent_id=acting_agent_id,
                    proposed_allocation=allocation,
                    step_index=state.current_step,
                    accepted=False,
                    rejected=False,
                )
            )
            state.increment_step()
            self._advance_round_and_switch_proposer(state=state, current_agent_id=acting_agent_id)
            return
        if action_type == "accept":
            if state.current_active_offer is None:
                raise ValueError("Cannot accept without an active offer.")
            state.mark_agreement_reached(
                agreed_allocation=state.current_active_offer.proposed_allocation,
                termination_reason="offer_accepted",
            )
            state.increment_step()
            return
        if action_type == "reject":
            if state.current_active_offer is None:
                raise ValueError("Cannot reject without an active offer.")
            state.increment_step()
            self._advance_round_and_switch_proposer(state=state, current_agent_id=acting_agent_id)
            return
        raise ValueError(f"Unsupported action type '{action_type}'.")

    def _advance_round_and_switch_proposer(
        self,
        *,
        state: NegotiationState,
        current_agent_id: str,
    ) -> None:
        """Advance round and switch proposer deterministically if episode continues."""
        if state.current_round < state.max_rounds:
            state.advance_round()
        next_agent = self._other_agent_id(current_agent_id)
        state.switch_proposer(next_agent)

    def _compute_rewards(
        self,
        *,
        state: NegotiationState,
        action_payload: dict[str, Any],
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Compute per-agent rewards using terminal or intermediate reward paths."""
        if state.is_terminal:
            allocation = None if state.outcome is None else state.outcome.agreed_allocation
            private_utilities = self._compute_private_utilities(
                allocation=allocation,
                round_index=state.current_round,
            )
            rewards, breakdowns = self.reward_aggregator.aggregate_with_breakdowns(
                state=state,
                private_utilities=private_utilities,
                proposed_allocation=allocation,
            )
            return rewards, {
                "private_utilities": dict(private_utilities),
                "reward_breakdowns": {
                    agent_id: breakdown.to_dict() for agent_id, breakdown in breakdowns.items()
                },
            }
        proposed_allocation = action_payload.get("allocation")
        if isinstance(proposed_allocation, dict):
            parsed_allocation = {str(k): float(v) for k, v in proposed_allocation.items()}
        else:
            parsed_allocation = None
        rewards, breakdowns = self.reward_aggregator.aggregate_with_breakdowns(
            state=state,
            private_utilities=None,
            proposed_allocation=parsed_allocation,
        )
        return rewards, {
            "reward_breakdowns": {
                agent_id: breakdown.to_dict() for agent_id, breakdown in breakdowns.items()
            }
        }

    def _compute_private_utilities(
        self,
        *,
        allocation: dict[str, float] | None,
        round_index: int,
    ) -> dict[str, float]:
        """Compute private utilities for all agents when allocation is available."""
        if allocation is None:
            return {agent_id: 0.0 for agent_id in self._agent_ids}
        utilities: dict[str, float] = {}
        for agent_id in self._agent_ids:
            profile = self._profiles[agent_id]
            utility, _ = self.utility_evaluator.evaluate_offer(
                profile=profile,
                allocation=allocation,
                round_index=round_index,
            )
            utilities[agent_id] = utility
        return utilities

    def _build_all_observations(self) -> dict[str, Any]:
        """Build observation dictionary for every active agent."""
        state = self.state
        observations: dict[str, Any] = {}
        for agent_id in self._agent_ids:
            observations[agent_id] = self.observation_builder.build(
                state=state,
                agent_id=agent_id,
                profile=self._profiles.get(agent_id),
            ).to_policy_input()
        return observations

    def _build_info(self, *, event: str, rewards: dict[str, float]) -> dict[str, Any]:
        """Construct deterministic info payload with metrics hooks."""
        state = self.state
        info: dict[str, Any] = {
            "event": event,
            "seed": self._current_seed,
            "agent_ids": list(self._agent_ids),
            "state_summary": state.summary_snapshot(),
            "metrics_hook": {
                "agreement_reached": state.agreement_reached,
                "rounds_used": state.current_round,
                "rewards": dict(rewards),
            },
            "active_agent": state.active_proposer_id,
        }
        if state.outcome is not None:
            info["outcome"] = state.outcome.to_dict()
        return info

    def _validate_action_legality(
        self,
        *,
        state: NegotiationState,
        acting_agent_id: str,
        action_type: str,
    ) -> None:
        """Validate action type and turn legality against current action mask."""
        if acting_agent_id not in self._agent_ids:
            raise ValueError(
                f"Unknown acting_agent_id '{acting_agent_id}'. Expected one of {self._agent_ids}."
            )
        if action_type not in self.config.environment.action_types:
            raise ValueError(
                f"Unsupported action_type '{action_type}'. "
                f"Expected one of {self.config.environment.action_types}."
            )
        action_mask = self.observation_builder.build(
            state=state,
            agent_id=acting_agent_id,
            profile=self._profiles.get(acting_agent_id),
        ).valid_action_mask
        if not action_mask.get(action_type, False):
            raise ValueError(
                f"Illegal action '{action_type}' for acting_agent_id '{acting_agent_id}' "
                "under current negotiation turn constraints."
            )

    def _normalize_step_action_input(self, action: dict[str, Any]) -> dict[str, Any]:
        """Normalize step input supporting both single-action and MARL dict formats."""
        if "agent_id" in action and "action_type" in action:
            return dict(action)
        if not self._agent_ids:
            raise RuntimeError("Environment must be reset before multi-agent dict actions are used.")
        state = self.state
        active_agent_id = state.active_proposer_id
        unknown_keys = [key for key in action.keys() if key not in self._agent_ids]
        if unknown_keys:
            raise ValueError(
                f"Multi-agent action dict contains unknown agent ids: {unknown_keys}. "
                f"Expected subset of {self._agent_ids}."
            )
        if active_agent_id not in action:
            raise ValueError(
                f"Active agent '{active_agent_id}' action missing in multi-agent action input."
            )
        active_action = action[active_agent_id]
        if not isinstance(active_action, dict):
            raise ValueError("Active agent action must be a dictionary.")
        for agent_id, candidate in action.items():
            if agent_id == active_agent_id:
                continue
            if candidate is None:
                continue
            if not isinstance(candidate, dict):
                raise ValueError(
                    f"Inactive agent action for '{agent_id}' must be a dictionary or None."
                )
            if candidate:
                # Inactive placeholders are accepted but ignored for turn-based execution.
                continue
        normalized = dict(active_action)
        if "agent_id" not in normalized:
            normalized["agent_id"] = active_agent_id
        return normalized

    def _extract_action_agent_id(self, action: dict[str, Any]) -> str:
        """Extract and validate acting agent id from action payload."""
        raw_value = action.get("agent_id")
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise ValueError("action['agent_id'] must be a non-empty string.")
        return raw_value

    def _extract_action_type(self, action: dict[str, Any]) -> str:
        """Extract and validate action type from action payload."""
        raw_value = action.get("action_type")
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise ValueError("action['action_type'] must be a non-empty string.")
        return raw_value

    def _extract_action_payload(self, action: dict[str, Any]) -> dict[str, Any]:
        """Extract normalized action payload dictionary."""
        payload = action.get("payload", {})
        if payload is None:
            return {}
        if not isinstance(payload, dict):
            raise ValueError("action['payload'] must be a dictionary when provided.")
        return dict(payload)

    def _extract_allocation(self, action_payload: dict[str, Any]) -> dict[str, float]:
        """Extract and validate allocation dictionary from action payload."""
        allocation_payload = action_payload.get("allocation")
        if isinstance(allocation_payload, list):
            raise ValueError("allocation must be dict, not list")
        if not isinstance(allocation_payload, dict):
            raise ValueError("Proposal action requires payload['allocation'] dictionary.")
        allocation = {str(agent_id): float(value) for agent_id, value in allocation_payload.items()}
        total_allocated = 0.0
        for agent_id, value in allocation.items():
            if agent_id not in self._agent_ids:
                raise ValueError(
                    f"Allocation contains unknown agent_id '{agent_id}'. "
                    f"Expected one of {self._agent_ids}."
                )
            if value < 0:
                raise ValueError(
                    f"Allocation value must be non-negative. agent_id={agent_id}, value={value}."
                )
            total_allocated += value
        if total_allocated > self.config.environment.total_resource_amount:
            raise ValueError(
                "Allocation exceeds total_resource_amount. "
                f"total_allocated={total_allocated}, "
                f"total_resource_amount={self.config.environment.total_resource_amount}."
            )
        return allocation

    def _resolve_agent_ids(self, options: Mapping[str, Any]) -> tuple[str, ...]:
        """Resolve agent ids from options with deterministic fallback."""
        raw_agent_ids = options.get("agent_ids")
        if raw_agent_ids is None:
            return ("agent_0", "agent_1")
        if not isinstance(raw_agent_ids, (list, tuple)):
            raise ValueError("options['agent_ids'] must be a list or tuple of strings.")
        if len(raw_agent_ids) < 2:
            raise ValueError("At least two agent_ids are required.")
        normalized = tuple(str(value) for value in raw_agent_ids)
        if len(set(normalized)) != len(normalized):
            raise ValueError("agent_ids must be unique.")
        for agent_id in normalized:
            if not agent_id.strip():
                raise ValueError("agent_ids must not contain empty strings.")
        self._refresh_spaces(agent_ids=normalized)
        return normalized

    def _other_agent_id(self, current_agent_id: str) -> str:
        """Return next proposer id using deterministic round-robin switching."""
        current_index = self._agent_ids.index(current_agent_id)
        next_index = (current_index + 1) % len(self._agent_ids)
        return self._agent_ids[next_index]

    def _is_timeout(self, state: NegotiationState) -> bool:
        """Return True when max rounds reached without agreement."""
        return (not state.is_terminal) and state.current_round >= state.max_rounds

    def _build_observation_space_spec(self) -> dict[str, Any]:
        """Build deterministic descriptor for policy-facing observation structure."""
        return {
            "public": {
                "current_round": {"type": "int", "min": 0, "max": self.config.environment.max_negotiation_rounds},
                "current_step": {"type": "int", "min": 0},
                "active_proposer_id": {"type": "string"},
                "remaining_resource": {"type": "float", "min": 0.0, "max": self.config.environment.total_resource_amount},
                "remaining_resource_normalized": {"type": "float", "min": 0.0, "max": 1.0},
                "rounds_remaining": {"type": "int", "min": 0, "max": self.config.environment.max_negotiation_rounds},
                "round_progress": {"type": "float", "min": 0.0, "max": 1.0},
                "is_terminal": {"type": "bool"},
            },
            "private": {
                "observing_agent_id": {"type": "string"},
                "private_utility_profile_summary": {
                    "type": "dict",
                    "keys": ["greed_sensitivity", "fairness_sensitivity", "urgency_sensitivity"],
                },
            },
            "summary": {
                "offer_count": {"type": "int", "min": 0},
                "rejection_count": {"type": "int", "min": 0},
                "acceptance_flag": {"type": "bool"},
                "recent_action_summary": {
                    "type": "dict",
                    "keys": list(self.config.environment.action_types),
                },
            },
            "valid_action_mask": {"type": "dict[bool]", "keys": list(self.config.environment.action_types)},
            "flattened_policy_vector_length": 12 + len(self.config.environment.action_types),
        }

    def _build_action_space_spec(self) -> dict[str, Any]:
        """Build deterministic descriptor for action contract and payload requirements."""
        return {
            "single_action": {
                "agent_id": {"type": "string"},
                "action_type": {"type": "enum", "values": list(self.config.environment.action_types)},
                "payload": {
                    "type": "dict",
                    "allocation": {
                        "type": "dict[str,float]",
                        "required_for": ["propose", "counteroffer"],
                        "constraints": {
                            "non_negative": True,
                            "sum_lte_total_resource": True,
                        },
                    },
                },
            },
            "multi_agent_action_dict": {
                "type": "dict[agent_id -> action]",
                "turn_based_rule": "only active agent action is consumed",
            },
        }

    def _build_observation_space(self) -> spaces.Space[Any]:
        """Build Gym-compatible observation space for registration and wrappers."""
        max_rounds = self.config.environment.max_negotiation_rounds
        total_resource = self.config.environment.total_resource_amount
        action_types = self.config.environment.action_types
        return spaces.Dict(
            {
                "public": spaces.Dict(
                    {
                        "current_round": spaces.Box(low=0, high=max_rounds, shape=(), dtype=int),
                        "current_step": spaces.Box(low=0, high=max_rounds * 10, shape=(), dtype=int),
                        "active_proposer_id": spaces.Text(max_length=64),
                        "remaining_resource": spaces.Box(
                            low=0.0,
                            high=total_resource,
                            shape=(),
                            dtype=float,
                        ),
                        "remaining_resource_normalized": spaces.Box(low=0.0, high=1.0, shape=(), dtype=float),
                        "current_public_offer": spaces.Box(
                            low=0.0,
                            high=total_resource,
                            shape=(max(2, len(self._agent_ids) or 2),),
                            dtype=float,
                        ),
                        "rounds_remaining": spaces.Box(low=0, high=max_rounds, shape=(), dtype=int),
                        "round_progress": spaces.Box(low=0.0, high=1.0, shape=(), dtype=float),
                        "is_terminal": spaces.Discrete(2),
                    }
                ),
                "private": spaces.Dict(
                    {
                        "observing_agent_id": spaces.Text(max_length=64),
                        "private_utility_profile_summary": spaces.Box(low=-10.0, high=10.0, shape=(3,), dtype=float),
                    }
                ),
                "summary": spaces.Dict(
                    {
                        "last_offer": spaces.Box(
                            low=0.0,
                            high=total_resource,
                            shape=(max(2, len(self._agent_ids) or 2),),
                            dtype=float,
                        ),
                        "recent_action_summary": spaces.Box(
                            low=0,
                            high=max_rounds,
                            shape=(len(action_types),),
                            dtype=int,
                        ),
                        "offer_count": spaces.Box(low=0, high=max_rounds, shape=(), dtype=int),
                        "rejection_count": spaces.Box(low=0, high=max_rounds, shape=(), dtype=int),
                        "acceptance_flag": spaces.Discrete(2),
                    }
                ),
                "valid_action_mask": spaces.Box(low=0, high=1, shape=(len(action_types),), dtype=int),
            }
        )

    def _build_action_space(self) -> spaces.Space[Any]:
        """Build Gym-compatible action space descriptor for trainer integration."""
        total_resource = self.config.environment.total_resource_amount
        agent_count = max(2, len(self._agent_ids) or 2)
        return spaces.Dict(
            {
                "agent_id": spaces.Text(max_length=64),
                "action_type_index": spaces.Discrete(len(self.config.environment.action_types)),
                "payload": spaces.Dict(
                    {
                        "allocation": spaces.Box(
                            low=0.0,
                            high=total_resource,
                            shape=(agent_count,),
                            dtype=float,
                        )
                    }
                ),
            }
        )

    def _refresh_spaces(self, agent_ids: tuple[str, ...]) -> None:
        """Refresh dynamic spaces/specs when agent roster changes."""
        self._agent_ids = agent_ids
        self.observation_space_spec = self._build_observation_space_spec()
        self.action_space_spec = self._build_action_space_spec()
        self.observation_space = self._build_observation_space()
        self.action_space = self._build_action_space()

    def _validate_observation_schema(self, observations: dict[str, dict[str, Any]]) -> None:
        """Strictly validate schema compliance for environment observations."""
        if set(observations.keys()) != set(self._agent_ids):
            raise ValueError(f"Observations must contain exactly agents {self._agent_ids}. Got {list(observations.keys())}")
        
        for agent_id, obs in observations.items():
            for key in ("public", "private", "summary", "valid_action_mask"):
                if key not in obs:
                    raise ValueError(f"Observation for {agent_id} missing top-level key '{key}'")
            
            public = obs["public"]
            for p_key in ("active_proposer_id", "remaining_resource", "current_public_offer"):
                if p_key not in public:
                    raise ValueError(f"Observation public section for {agent_id} missing '{p_key}'")
            
            active_proposer = public["active_proposer_id"]
            if not isinstance(active_proposer, str) or not active_proposer.strip() or active_proposer not in self._agent_ids:
                raise ValueError(f"Invalid active_proposer_id '{active_proposer}' in observation for {agent_id}")
            
            offer = public["current_public_offer"]
            if offer is not None and not isinstance(offer, dict):
                raise ValueError(f"current_public_offer must be None or dict. Got {type(offer)} in observation for {agent_id}")

            mask = obs["valid_action_mask"]
            for a_key in ("propose", "counteroffer", "accept", "reject"):
                if a_key not in mask or not isinstance(mask[a_key], bool):
                    raise ValueError(f"Action mask for {agent_id} missing or invalid boolean for '{a_key}'")

    def _validate_step_outputs(
        self,
        observations: dict[str, dict[str, Any]],
        rewards: dict[str, float],
        terminated: dict[str, bool],
        truncated: dict[str, bool]
    ) -> None:
        """Validate step outputs have complete schemas."""
        self._validate_observation_schema(observations)
        
        expected_agents = set(self._agent_ids)
        if set(rewards.keys()) != expected_agents:
            raise ValueError(f"Rewards missing agents. Expected {expected_agents}, got {set(rewards.keys())}")
            
        expected_term = expected_agents | {"__all__"}
        if set(terminated.keys()) != expected_term:
            raise ValueError(f"Terminated missing keys. Expected {expected_term}, got {set(terminated.keys())}")
            
        if set(truncated.keys()) != expected_term:
            raise ValueError(f"Truncated missing keys. Expected {expected_term}, got {set(truncated.keys())}")
