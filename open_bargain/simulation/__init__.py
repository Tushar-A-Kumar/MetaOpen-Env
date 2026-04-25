"""Simulation package for episode rollout scaffolding."""

from open_bargain.simulation.simulate import (
    EpisodeTrace,
    FairNegotiationPolicy,
    GreedyNegotiationPolicy,
    PolicyHook,
    RandomNegotiationPolicy,
    SimulationResult,
    StepTrace,
    build_policy_comparison_summary,
    simulate_episode,
    simulate_episodes,
)

__all__ = [
    "EpisodeTrace",
    "FairNegotiationPolicy",
    "GreedyNegotiationPolicy",
    "PolicyHook",
    "RandomNegotiationPolicy",
    "SimulationResult",
    "StepTrace",
    "build_policy_comparison_summary",
    "simulate_episode",
    "simulate_episodes",
]
