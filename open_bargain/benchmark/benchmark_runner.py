"""Benchmark runner for OpenBargain baseline evaluations."""

from typing import Any
import json

from open_bargain.config import OpenBargainConfig
from open_bargain.env.env import OpenBargainEnv
from open_bargain.simulation.simulate import (
    simulate_episodes,
    RandomNegotiationPolicy,
    FairNegotiationPolicy,
    GreedyNegotiationPolicy,
)

BASELINES = [
    ("random_vs_random", RandomNegotiationPolicy, RandomNegotiationPolicy),
    ("fair_vs_fair", FairNegotiationPolicy, FairNegotiationPolicy),
    ("greedy_vs_greedy", GreedyNegotiationPolicy, GreedyNegotiationPolicy),
    ("fair_vs_greedy", FairNegotiationPolicy, GreedyNegotiationPolicy),
    ("greedy_vs_fair", GreedyNegotiationPolicy, FairNegotiationPolicy),
]

def run_benchmark_matchups(config: OpenBargainConfig | None = None) -> list[dict[str, Any]]:
    """Run all standard baseline matchups and return aggregated results."""
    if config is None:
        config = OpenBargainConfig.default()
        
    env = OpenBargainEnv(config=config)
    results = []

    for baseline_name, policy_a_cls, policy_b_cls in BASELINES:
        policy_a = policy_a_cls(seed=config.simulation.default_random_seed)
        policy_b = policy_b_cls(seed=config.simulation.default_random_seed + 1)
        
        sim_result = simulate_episodes(
            env=env,
            config=config,
            policy_hooks=[policy_a, policy_b]
        )
        
        metrics = sim_result.aggregate_metrics
        if metrics is None:
            raise RuntimeError(f"Aggregate metrics missing for {baseline_name}")

        results.append({
            "benchmark_name": baseline_name,
            "agreement_rate": metrics.agreement_rate,
            "fairness_score": metrics.average_fairness_index,
            "efficiency_score": metrics.average_efficiency_score,
            "benchmark_score": metrics.benchmark_score,
        })

    return results

def rank_benchmark_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Sort and rank baseline results by benchmark score descending."""
    ranked = sorted(results, key=lambda x: x["benchmark_score"], reverse=True)
    
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank

    top_baseline = ranked[0]["benchmark_name"] if ranked else None
    top_score = ranked[0]["benchmark_score"] if ranked else 0.0

    return {
        "ranked_results": ranked,
        "top_baseline": top_baseline,
        "top_score": top_score,
    }

def export_benchmark_results(ranked_summary: dict[str, Any]) -> dict[str, Any]:
    """Export deterministic, versioned benchmark artifact."""
    return {
        "benchmark_version": "1.0",
        "environment": "OpenBargain",
        "results": ranked_summary["ranked_results"],
        "top_baseline": ranked_summary["top_baseline"],
        "top_score": ranked_summary["top_score"],
    }

def print_benchmark_summary(ranked_summary: dict[str, Any]) -> None:
    """Print a human-readable console summary for the benchmark."""
    print("\nBenchmark Results:")
    for row in ranked_summary["ranked_results"]:
        name = row["benchmark_name"]
        score = row["benchmark_score"]
        rank = row["rank"]
        print(f"{rank}. {name.ljust(20, '.')} {score:.2f}")

def run_full_benchmark() -> dict[str, Any]:
    """Execute the full benchmark suite and return the deterministic export artifact."""
    results = run_benchmark_matchups()
    ranked_summary = rank_benchmark_results(results)
    print_benchmark_summary(ranked_summary)
    return export_benchmark_results(ranked_summary)
