# OpenBargain Metrics Explained

## Why Metrics Matter Here
OpenBargain is a negotiation benchmark. A policy can get frequent rewards but still behave poorly in bargaining terms. The metric layer focuses on negotiation quality, not only immediate reward.

## Agreement Rate
What it captures:
- How often episodes end with a deal.

How to read it:
- High value: policy is effective at converging to agreements.
- Low value: policy often deadlocks, rejects too aggressively, or fails under uncertainty.

When to be cautious:
- Very high agreement rate with poor fairness can indicate exploitative deals.

## Fairness Index
What it captures:
- How balanced the final allocation is relative to equal split behavior.

How to read it:
- High value: outcomes are more equitable.
- Low value: outcomes are skewed toward one side.

When to be cautious:
- High fairness with low agreement can mean the policy is too strict and misses feasible deals.

## Exploitability Score
What it captures:
- Degree of one-sidedness in outcomes.

How to read it:
- Low value: less exploitative behavior.
- High value: one agent often gains disproportionately.

Relationship to fairness:
- In this benchmark, exploitability and fairness are complementary. Better fairness usually lowers exploitability.

## Efficiency Score
What it captures:
- How quickly agreements are reached relative to max rounds.

How to read it:
- High value: fast convergence and low negotiation delay.
- Low value: slow or stalled negotiations.

Why it matters:
- Real systems care about latency and negotiation cost, not just eventual agreement.

## Social Welfare Score
What it captures:
- Cooperative quality that combines effective resource use with temporal efficiency.

How to read it:
- High value: outcomes are both useful and timely.
- Low value: agreements may be late, inefficient, or absent.

## Benchmark Score
What it captures:
- Unified leaderboard-friendly quality score in [0, 1].

How to read it:
- High value: strong overall policy quality across agreement, fairness, efficiency, and welfare.
- Medium value: acceptable but imbalanced policy behavior.
- Low value: policy underperforms on core bargaining objectives.

Why this is useful:
- Judges and researchers can compare policies quickly without losing access to component metrics.

## Practical Interpretation Patterns
- High agreement + low fairness + high exploitability:
  policy reaches deals but often in unfair ways.
- Medium agreement + high fairness + medium efficiency:
  policy is cooperative but may negotiate too cautiously.
- High agreement + high fairness + high efficiency:
  strong candidate for robust benchmark performance.

## Recommended Reporting Format
For each run, report:
- aggregate metrics payload,
- benchmark score,
- reward summary,
- policy metadata,
- optional per-episode traces for auditability.

Use deterministic seeds and include seed values in all submitted artifacts.
