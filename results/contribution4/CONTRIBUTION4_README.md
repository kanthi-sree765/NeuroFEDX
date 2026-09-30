# Contribution 4 — Computational, Communication, and Latency Evaluation

This contribution evaluates the existing NeuroFEDX system without introducing a new prediction architecture. It measures centralized RF, the existing 10-client/50-round RF tree-pooling global model, Contribution 2 personalization, and Contribution 3 TreeSHAP latency.

## Configuration
- 10 clients
- 50 FL rounds
- 10 trees/client/round
- 100 global trees
- 10 personalization trees
- RF configuration from `config/model_config.py` unchanged
- seed 42

## Commands
```bash
PYTHONPATH=. python -m evaluation.contribution4 diagnosis
PYTHONPATH=. python -m evaluation.contribution4 severity
PYTHONPATH=. python -m evaluation.finalize_contribution4
PYTHONPATH=. python -m pytest -q
```

## Outputs
See `results/contribution4/` for round/client/communication measurements, model sizes, inference latency, TreeSHAP latency, summary tables, configuration snapshots, and plots.

## Interpretation
All timing and payload values are measured in this execution environment. Ratios are configuration-specific measurements, not universal algorithm properties. Centralized communication is N/A, not zero. Network latency/bandwidth are NOT MEASURED because the FL implementation runs in a simulated single-process environment.
