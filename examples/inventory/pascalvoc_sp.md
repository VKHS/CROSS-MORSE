# Complete-run artifact inventory

**PascalVOC-SP — executed_synthetic_software_example**

Verified all eight stages and 294 sealed files. The directory contains 311 inventoried files (27,582,015 bytes).

This report checks artifact integrity and actual split membership. It does not deserialize checkpoints, replay statistics, authenticate historical execution, or claim agreement with paper measurements.

## Identities

| Identity | SHA-256 |
|---|---|
| source_sha256 | `9e0154c11c0ebc67d533a84b6d6a80c4e77a8d76e6d2cc165bcff5897980b108` |
| config_sha256 | `922018f810bb925df20d0e43ee887ae74ac6cc321b45fb3402246d3945c12563` |
| unit_sha256 | `2364a4ac52b0796d16a19b4fedfc2bc8d4fba6d62897ebf28c6f32c4f55ae7d2` |

## Contents

| Category | Files | Bytes |
|---|---:|---:|
| ancillary records | 11 | 45,159 |
| candidate and member predictions | 160 | 1,010,925 |
| configuration and environment | 3 | 8,239 |
| decision and statistical records | 13 | 116,363 |
| final predictions and statistics | 8 | 63,163 |
| model and recovery checkpoints | 50 | 25,627,748 |
| offline trajectories | 24 | 168,225 |
| prepared data and splits | 7 | 99,803 |
| stage manifests | 8 | 254,644 |
| training and freeze records | 27 | 187,746 |

## Data and seeds

Master/split seed: `314159`. Training seeds: `[1, 2]`. These are this run's recorded seeds; historical manuscript seed identity is not asserted.

Prediction axis: **node**; split axis: **graph**; statistical units: **graphs**.

| Role | Label rows | Split units | Nonempty statistical units |
|---|---:|---:|---:|
| fit | 224 | 28 | 28 |
| fit_train | 208 | 26 | 26 |
| fit_val | 16 | 2 | 2 |
| explore | 80 | 10 | 10 |
| select | 40 | 5 | 5 |
| confirm | 40 | 5 | 5 |
| admit | 128 | 16 | 16 |
| test | 128 | 16 | 16 |

## Models, trajectories, and decisions

The run contains 4 best checkpoints, 12 complete offline trajectories and 96 recorded transitions. Every checkpoint file is present and matches its seal. Recorded backbone/head hashes agree across stages; logical tensor hashes are not independently recomputed by this stdlib tool.

Admission passed: `False`. Exact qC fallback: `True`. Statistical scope: paired graph bootstrap conditional on the fixed trained ensemble. Seed scope: one fixed prefix ensemble; retraining-seed uncertainty is not estimated.

The JSON companion contains every relative path, size, SHA-256, producing stage, numeric-array shape, trajectory identity, and the captured environment. `unsealed_ancillary` records are hashed for this inventory but are not stage-sealed scientific outputs.
