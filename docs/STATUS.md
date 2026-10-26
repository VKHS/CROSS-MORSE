# Status: what is complete, what is pending, what is open

[Repository home](../README.md) · [Availability](../AVAILABILITY.md) · [Run archives](RUN_ARCHIVES.md)

This page lists what is complete, what is pending and what is open. It is maintained by hand;
the counts in the availability statement are generated from the tree.

## Complete

| Item | State |
|---|---|
| Implementation of the method for all seven datasets, eight sealed stages, nine policy families | released; runtime source identity `9e0154c11c0e…` (`make fingerprint`) |
| Entry points, three configurations per dataset, Slurm launchers, Arrhenius site profile | released |
| Validated CPU environment (lock + observed record) and per-run environment records | released |
| Seven complete synthetic example runs: data, splits, checkpoints, trajectories, predictions, decisions, manifests | released under `examples/runs/`; `make audit-artifacts` and `make reproduce` verify and replay them |
| Registered split/seed manifests of every configuration; per-run manifests of the examples | released under `manifests/` |
| Run-driven table and figure builders; committed example build | released (`paper/build/`, `paper/generated/examples/`) |
| Table S12c: the selection rule of main Eq. (19) under its three equivalence readings, built from every run and checked against the run's sealed selection | released (`paper/build/selection.py`) |
| Measurement-record contract for the experiments outside the eight-stage run: ten record sets, every printed value traced to its raw log, tables printed from the sets | released (`records/`, `scripts/records.py`, `make records-check`) |
| Run-archive contract: pack, verify, deposit, ingest, index | released (`scripts/archive_run.py`, `scripts/ingest_run.py`, `runs/`) |
| Availability statement generated from the tree and checked in CI | released (`AVAILABILITY.md`, `make availability-check`) |
| Manuscript registry with page provenance and 584 arithmetic checks, used for comparison only | released (`paper/reported_results.json`, `make registry-check`); the manuscript itself is identified by SHA-256 in `paper/sources.json` and not distributed |

## Pending: produced on Arrhenius, then deposited

| Item | How it enters the repository |
|---|---|
| Official-data run archives for ogbn-arxiv, ogbn-proteins, ogbn-products, PascalVOC-SP, COCO-SP, Peptides-func (full profile) | `make archive-run` on the cluster → deposit (DOI) → `make ingest-run` |
| Roman-empire ten-split campaign archive | `make archive-campaign` → deposit → `make ingest-run` |
| GPU environment record of the runs (`environment/arrhenius/observed_environment.json`, `requirements-gpu.lock`) | `environment/arrhenius/capture_environment.sh` inside a GPU job, then commit |
| `paper/generated/official/` (tables, figures, evidence from the archives) | `make paper` after ingestion; commit the result |
| Comparison of run-derived values with the manuscript | `make compare-runs`; the report decides which manuscript values change |
| Measurement records of the ten record sets (Supplementary Tables S6c, S15/S15b, S16/S16b, S19, S23, S23b, S25, S26b, S45, S46/S46b) | raw logs, code and `summary.csv` into `records/<set>/` → `make records-index` ([records/README.md](../records/README.md)) |
| Archival DOI of the code | archive on Zenodo the commit whose `make fingerprint` equals the archives' `source_sha256`; set `code_archive_doi` in `release/release_scope.json` and run `make availability` |

## Open: differences between the released code and the manuscript

These are recorded in [`paper/KNOWN_DIFFERENCES.md`](../paper/KNOWN_DIFFERENCES.md). For the
model and execution differences the code follows the manuscript: the settings of the reported
runs are carried into the configurations and the runtime before the official runs, and the
changed runtime is validated with PyTorch (`make test`, `make train-samples`) and receives a new
source identity. The manuscript's final numbers come from archives that `make paper` rebuilds.

| ID | Difference | Resolution path |
|---|---|---|
| ID | Difference | Resolution (code follows the manuscript) |
|---|---|---|
| D1 | plant parameter counts of the released architectures differ from Supplementary Table S5 | `core.py::build_plant` takes the architecture of the reported plants, read from their checkpoint inventories (`train/model_inventory.json` of the paper runs), so that the counts equal Table S5 |
| D6 | the full profiles of proteins and PascalVOC-SP sense the full state; the reported runs used a reduced sensor | the executed `(k, P_s, f_res)` are deposited as the `sensor_parameters` record set and written into `configs/proteins_full.json` and `configs/pascalvoc_sp_full.json` |
| D7 | products runs full-graph propagation instead of the partition-local execution with buffered boundary | the partition-local propagation step of Supplementary Table S5b is implemented in the products adapter |
| D15 | the budget-matched arms (E3–E6), inserted-site attribution (E13), extended ensemble grid (E14), factorials, negative controls, cross-backbone substitutions and the input gate run outside the eight-stage lifecycle | their record sets are deposited under `records/`; `make records-tables` prints their tables |
| D16 | retrained external architectures (AMP, ADMP-GNN, GRAMA, Graph Mamba, Early-Exit GNN) | the `baselines` record set is deposited under `records/` |
| D17 | the modelled tail cost C_0.99 and the cost tail scenarios are analytical | the observed per-input maximum is read from the timing traces that every run records |

D12 (the three readings of the selection rule) is resolved: Table S12c is built from every run.
Until the pending items land, the numbers in the manuscript are the authors' reported
measurements, and the generated availability statement says so in the same words.
