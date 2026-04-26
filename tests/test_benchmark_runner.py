"""Tests to ensure the benchmark runner executes deterministically and correctly."""

from open_bargain.benchmark.benchmark_runner import run_full_benchmark

def test_benchmark_runner():
    print("Starting full benchmark test...")
    artifact = run_full_benchmark()
    
    assert artifact is not None
    assert artifact.get("benchmark_version") == "1.0"
    assert artifact.get("environment") == "OpenBargain"
    
    results = artifact.get("results")
    assert results is not None
    assert len(results) == 5, "Expected 5 baselines to be run."
    
    # Verify exact baselines exist
    baselines_run = {row["benchmark_name"] for row in results}
    expected_baselines = {
        "random_vs_random",
        "fair_vs_fair",
        "greedy_vs_greedy",
        "fair_vs_greedy",
        "greedy_vs_fair"
    }
    assert baselines_run == expected_baselines
    
    # Verify ranked descending
    scores = [row["benchmark_score"] for row in results]
    assert scores == sorted(scores, reverse=True), "Results are not ranked in descending order."
    
    # Verify top baseline
    top_baseline = artifact.get("top_baseline")
    assert top_baseline is not None
    assert top_baseline == results[0]["benchmark_name"]
    
    # Verify score bounds
    top_score = artifact.get("top_score")
    assert top_score is not None
    assert 0.0 <= top_score <= 1.0, f"Top score out of bounds: {top_score}"

    print("All benchmark assertions passed successfully.")
    
if __name__ == "__main__":
    test_benchmark_runner()
