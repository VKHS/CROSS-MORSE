# Complete-run artifact inventory

**Peptides-func — executed_synthetic_software_example**

Verified all eight stages and 294 sealed files. The directory contains 311 inventoried files (26,457,634 bytes).

This report checks artifact integrity and actual split membership. It does not deserialize checkpoints, replay statistics, authenticate historical execution, or claim agreement with paper measurements.

## Identities

| Identity | SHA-256 |
|---|---|
| source_sha256 | `9e0154c11c0ebc67d533a84b6d6a80c4e77a8d76e6d2cc165bcff5897980b108` |
| config_sha256 | `286b22b6af7a6faefa617db030931ef522bef77157294b536089634410ea53da` |
| unit_sha256 | `921f7a501d28cfdcbe38e7a08af9574c8492406aeb0e5c6652d2f47bfeb54f3e` |

## Contents

| Category | Files | Bytes |
|---|---:|---:|
| ancillary records | 11 | 44,947 |
| candidate and member predictions | 160 | 293,880 |
| configuration and environment | 3 | 8,233 |
| decision and statistical records | 13 | 86,133 |
| final predictions and statistics | 8 | 17,593 |
| model and recovery checkpoints | 50 | 25,318,710 |
| offline trajectories | 24 | 168,441 |
| prepared data and splits | 7 | 96,093 |
| stage manifests | 8 | 254,267 |
| training and freeze records | 27 | 169,337 |

## Data and seeds

Master/split seed: `314159`. Training seeds: `[1, 2]`. These are this run's recorded seeds; historical manuscript seed identity is not asserted.

Prediction axis: **graph**; split axis: **graph**; statistical units: **graphs**.

| Role | Label rows | Split units | Nonempty statistical units |
|---|---:|---:|---:|
| fit | 28 | 28 | 28 |
| fit_train | 26 | 26 | 26 |
| fit_val | 2 | 2 | 2 |
| explore | 10 | 10 | 10 |
| select | 5 | 5 | 5 |
| confirm | 5 | 5 | 5 |
| admit | 16 | 16 | 16 |
| test | 16 | 16 | 16 |

## Models, trajectories, and decisions

The run contains 4 best checkpoints, 12 complete offline trajectories and 96 recorded transitions. Every checkpoint file is present and matches its seal. Recorded backbone/head hashes agree across stages; logical tensor hashes are not independently recomputed by this stdlib tool.

Admission passed: `False`. Exact qC fallback: `True`. Statistical scope: paired graph bootstrap conditional on the fixed trained ensemble. Seed scope: one fixed prefix ensemble; retraining-seed uncertainty is not estimated.

The JSON companion contains every relative path, size, SHA-256, producing stage, numeric-array shape, trajectory identity, and the captured environment. `unsealed_ancillary` records are hashed for this inventory but are not stage-sealed scientific outputs.
