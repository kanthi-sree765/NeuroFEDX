# Contribution 4 Implementation Report

## Scope
Contribution 4 was added as an evaluation/instrumentation layer to the existing NeuroFEDX project. The Base Paper, Contribution 1, Contribution 2, and Contribution 3 code/results were preserved. No RF hyperparameters, client count, round count, preprocessing, labels, C2 partitioning, or C3 TreeSHAP methodology were changed.

## Existing files inspected
- `config/model_config.py`
- `models/random_forest.py`
- `federated/federated_rf.py`
- `federated/personalized_rf.py`
- `evaluation/contribution2.py`
- `explainability/contribution3.py`
- `results/contribution2/*`
- `results/contribution3/*`
- existing Base Paper/C1 evaluation scripts and tests

## Instrumentation
The new `evaluation/contribution4.py` measures client training, actual joblib serialized local-forest payloads, global model serialization, server tree-pool construction, round duration, process timing, RSS snapshots, inference latency, personalization construction, model sizes, and C3 TreeSHAP latency. `time.perf_counter()` is used for durations.

## Communication methodology
Upload values are actual serialized local-forest payload sizes; the server-to-client value is a simulated serialized global-forest broadcast payload size, because the current single-process tree-pooling loop does not actually transmit or reuse that global forest. The implementation is simulated/single-process, so network latency and real network bandwidth are **NOT MEASURED**.

## Centralized baseline
For C4 timing comparison, the centralized RF is an evaluation-only baseline trained on the pooled federation-training rows used by the C2 experiment. This keeps the training-data basis aligned with the global federated shards without replacing any existing Base/C1/C2 result.

## Memory
Process RSS snapshots are measured where supported. It is distinct from serialized model size. Phase-specific personalized peak RSS is not isolated reliably from the shared process baseline and is therefore reported as NOT MEASURED in the final summary rather than fabricated.

## Timing protocol
- `time.perf_counter()` for elapsed durations
- 7 repetitions for centralized training and SHAP timing
- 20 repetitions for inference timing
- one prediction warm-up before inference timing
- no round/client removal or runtime tuning

## Validation
`summary/behavior_preservation_validation.csv` checks final-round predictions/probabilities against the persisted C2 global models and deterministic personalized client-0 predictions against the C3 reconstruction. `summary/measurement_audit.csv` checks 50 rounds, 10 clients, 100 global trees, non-negative measurements, and monotonic cumulative communication.

## Important limitation
These measurements characterize computational and serialized communication behavior of the simulated federated implementation under the experimental environment. They do not establish real-world network latency, bandwidth performance, distributed-device scaling, or multi-device communication latency.
