# Reviewer guide

[Repository home](../README.md) · [Availability](../AVAILABILITY.md) · [Status](STATUS.md) ·
[Run archives](RUN_ARCHIVES.md) · [Implementation contract](IMPLEMENTATION_CONTRACT.md) ·
[Validation record](VALIDATION.md)

This guide answers the two review reports point by point and gives an ordered inspection
path. Every command below runs on a CPU with NumPy; nothing downloads data or trains.
Two diagrams summarise what follows: the [evidence chain](assets/evidence_chain.png) (source
identity → Slurm job → sealed run → archive → index → tables) and the
[decision chain](assets/decision_chain.png) (each decision statistic, its run file, its table cell).

## 1. The reviewers' list, item by item

| Reviewer request | Where it is | How to check it |
|---|---|---|
| Experiment code | `cross_morse/` (plants, adapter, registry, sensing, observer, learning, policies, statistics, lifecycle), `run_cross_morse.py` and seven dataset entry points, `configs/` | `make fingerprint` prints the runtime source identity sealed into every run; `make test` runs 345+ tests with PyTorch, `make test-no-torch` the NumPy subset |
| Data | adapters in `cross_morse/datasets.py`; sources, versions and split rules in `data/sources.json`; `make prepare-data` | datasets are fetched from their providers, never redistributed; every run seals the source array hashes in `prepared/metadata.json` |
| Split/seed manifests | `manifests/registered/` (design of every configuration), `manifests/examples/` and `manifests/official/` (what each run used), `prepared/splits.npz` in every run | `make manifests-check` |
| Environment specification | `environment/requirements-cpu.lock` and `observed_environment.json` (validated CPU environment), `environment/arrhenius/` (GPU capture script and its outputs), `environment.json` sealed in every run | `make environment-check` |
| Checkpoints, predictions, trajectories, statistical outputs | seven synthetic example runs under `examples/runs/<stem>/` today; the official-data archives under `runs/` once ingested (`runs/index.json`) | `make audit-artifacts`, `make reproduce`, `make runs-check` |
| Figure/table generators | `paper/build/` and `scripts/build_paper.py`, including Table S12c (the three readings of the selection rule); outputs under `paper/generated/<label>/` | `make example-paper-check` (committed example build equals a fresh one), `make paper` for official archives |
| Measurements of the experiments outside the eight-stage run (S15/S15b, S16/S16b, S19, S23, S23b, S6c, S25, S26b, S45, S46/S46b) | record sets under `records/<set>/` with the raw logs, code and provenance of each | `make records-check` re-reads every printed value from its raw log; `make records-tables` prints the tables |
| Makefile that regenerates tables and figures | `make tables`, `make figures`, `make paper`, `make example-paper` | `make help` |
| Availability statement consistent with the repository | `AVAILABILITY.md`, generated from `release/release_scope.json`, `runs/index.json` and the tree | `make availability-check` fails in CI when the statement claims more than exists |
| Slurm scripts with the cluster details | `run_cross_morse.sh` and wrappers, `configs/arrhenius-site.env.example`, [ARRHENIUS.md](ARRHENIUS.md) | `bash run_cross_morse_arxiv.sh dry-run full` prints the exact job without creating anything |

What is **not** in the repository is listed in [STATUS.md](STATUS.md) and repeated in the
generated availability statement: until the official-data run archives are deposited and
ingested, the numbers in the manuscript are our reported measurements, transcribed in
`paper/reported_results.json` for comparison only.

## 2. Ordered inspection path

1. **Identity.** `make fingerprint` — the runtime source hash; compare with
   `source_sha256` in any run's `run_state.json` and in `docs/VALIDATION.json`.
2. **Package integrity.** `make audit` — required files, syntax, checksums, the private
   -information scan and the sealed chains of the seven examples.
3. **Manuscript registry arithmetic.** `make registry-check` — registered constants and the
   cross-table arithmetic of the transcribed values; this checks the manuscript's internal
   consistency, not its empirical provenance.
4. **Examples.** `make reproduce REPORT_ROOT=../cm_reports` — recompute selection,
   confirmation, admission and the 28 predictor rows from saved predictions and labels.
   Then open `examples/runs/arxiv/provenance/stages/` and follow one artifact through the
   chain.
5. **Tables from runs.** `make example-paper-check`, then read
   `paper/generated/examples/tables/T1.md` and `M1.md`: the same code builds the paper's
   tables from the official archives (`make paper`).
6. **Manifests.** `make manifests-check`; open `manifests/examples/arxiv.json` for the role
   membership hashes, seeds and environment of that run.
7. **Availability.** `make availability-check`; `AVAILABILITY.md` is what the manuscript
   statement is copied from.
7a. **Selection readings.** `paper/generated/examples/tables/S12c.md`: for every run, the three
   equivalence readings of main Eq. (19), the route selected, and the registered rule over all
   configurations checked against the run's sealed selection.
7b. **Measurement records.** `make records-check`, then `records/<set>/summary.csv` next to its
   `raw/` logs; `make records-tables` prints the supplementary tables from the deposited sets.
8. **Official archives** (when present). `make runs-check`, `make paper`,
   `make compare-runs`; read `paper/generated/official/tables/` and the comparison
   report under `REPORT_ROOT/compare/`.
9. **Checkpoint re-execution** (needs PyTorch). `make reexecute RUN_ROOT=examples/runs/arxiv
   REPORT_ROOT=../cm_reports` — recorded-versus-re-executed probabilities.
10. **A fresh end-to-end run** (needs PyTorch). `make train-sample DATASET=Peptides-func
    RUN_ROOT=../cm_runs/peptides` — eight sealed stages in minutes on a CPU, then steps 4–6
    on that run.

## 3. What each evidence source establishes

| Evidence | Establishes | Does not establish |
|---|---|---|
| Sealed stage chain and hashes | the bytes of a run are the bytes it wrote | who ran it, when, or that it is the run behind a table |
| Numerical replay | the decisions follow from the saved predictions and labels | that the predictions are correct for the official data |
| Checkpoint re-execution | the checkpoints produce the saved predictions | the training history |
| Run-driven tables | the tables are functions of run directories | anything about runs that are not present |
| Comparison with the registry | agreement level per quantity at printed precision | statistical replication; the registry is transcribed, not computed |
| Result binding | our declaration of which run produced which value | independent verification of that declaration |

## 4. Differences between the released code and the manuscript's description

[`paper/KNOWN_DIFFERENCES.md`](../paper/KNOWN_DIFFERENCES.md) lists every place where the
release departs from the manuscript (architecture parameter counts, products execution,
selection reading, instruments outside the release). [STATUS.md](STATUS.md) says which of
them we are resolving by re-running and which by revising the manuscript.
