# Complete-run artifact inventory

**ogbn-arxiv — executed_synthetic_software_example**

Verified all eight stages and 292 sealed files. The directory contains 309 inventoried files (27,118,872 bytes).

This report checks artifact integrity and actual split membership. It does not deserialize checkpoints, replay statistics, authenticate historical execution, or claim agreement with paper measurements.

## Identities

| Identity | SHA-256 |
|---|---|
| source_sha256 | `9e0154c11c0ebc67d533a84b6d6a80c4e77a8d76e6d2cc165bcff5897980b108` |
| config_sha256 | `2f97bcb6667d1a9c246183513885913fa3677cf4a54b9afb94c0f1c46df331e5` |
| unit_sha256 | `97212529605501122bab96f5a6f71580e449955ec68b53142c0f08749c7f2b63` |

## Contents

| Category | Files | Bytes |
|---|---:|---:|
| ancillary records | 11 | 13,194 |
| candidate and member predictions | 160 | 1,069,971 |
| configuration and environment | 3 | 8,152 |
| decision and statistical records | 12 | 70,073 |
| final predictions and statistics | 8 | 25,344 |
| model and recovery checkpoints | 50 | 25,320,098 |
| offline trajectories | 24 | 156,324 |
| prepared data and splits | 6 | 41,823 |
| stage manifests | 8 | 252,459 |
| training and freeze records | 27 | 161,434 |

## Data and seeds

Master/split seed: `314159`. Training seeds: `[1, 2]`. These are this run's recorded seeds; historical manuscript seed identity is not asserted.

Prediction axis: **node**; split axis: **node**; statistical units: **balanced_synthetic_only**.

| Role | Label rows | Split units | Nonempty statistical units |
|---|---:|---:|---:|
| fit | 57 | 57 | 8 |
| fit_train | 52 | 52 | 8 |
| fit_val | 5 | 5 | 4 |
| explore | 19 | 19 | 7 |
| select | 10 | 10 | 5 |
| confirm | 10 | 10 | 6 |
| admit | 32 | 32 | 8 |
| test | 32 | 32 | 8 |

## Models, trajectories, and decisions

The run contains 4 best checkpoints, 12 complete offline trajectories and 72 recorded transitions. Every checkpoint file is present and matches its seal. Recorded backbone/head hashes agree across stages; logical tensor hashes are not independently recomputed by this stdlib tool.

Admission passed: `False`. Exact qC fallback: `True`. Statistical scope: descriptive dependent-core inference. Seed scope: one fixed prefix ensemble; retraining-seed uncertainty is not estimated.

The JSON companion contains every relative path, size, SHA-256, producing stage, numeric-array shape, trajectory identity, and the captured environment. `unsealed_ancillary` records are hashed for this inventory but are not stage-sealed scientific outputs.
