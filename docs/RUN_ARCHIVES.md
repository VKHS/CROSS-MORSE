# Run archives: the contract between a completed run and the paper

A run archive is the unit of evidence behind a reported number. This page defines what a
completed run directory contains, how it is packed, verified, deposited and ingested, and how
the paper's tables and figures are rebuilt from it. The same contract applies to the seven
synthetic examples under `examples/runs/` and to the official-data runs under `runs/`.

## 1. What a completed run contains

Every run is produced by `run_cross_morse.py` in eight sealed stages,
`prepare → train → explore → fit → select → confirm → admit → evaluate`. Each stage writes a
manifest under `provenance/stages/NN_<stage>.json` that records the runtime source identity,
the configuration identity, the unit (split) identity, the hash of every input it read and
every output it wrote, and the hash of the previous stage's manifest.

| Evidence | Files | Role |
|---|---|---|
| Prepared data | `prepared/graph.npz`, `prepared/metadata.json` | features, edges, targets, source array hashes, preparation choices |
| Splits and statistical units | `prepared/official_splits.npz`, `prepared/splits.npz`, `prepared/clusters.npz` (or `graph_splits.npz`) | exact membership of D_fit / D_explore / D_select / D_confirm / D_admit / D_test and of the resampling units |
| Seeds and training | `config.json`, `train/*_history.json`, `train/manifest.json`, `train/model_inventory.json` | partition seed, checkpoint seeds, stopping, checkpoint selection, tensor inventory |
| Environment | `environment.json` | Python, PyTorch, CUDA, GPU, platform, package versions, determinism flags |
| Checkpoints | `train/plant_seed*_best.pt`, `train/q0_seed*_best.pt`, `fit/{lean,full}/controller.pt` | frozen predictors and fitted controller models with hashes in `policy_lock.json` |
| Trajectories | `explore/trajectory_*.{json,npz}`, `explore/manifest.json` | propensity-logged exploration under the registered action set |
| Candidate predictions | `member_predictions/*.npy`, `candidate_predictions/` | every (route, E, profile) candidate's outputs on the selection units |
| Decisions | `select/`, `confirm/`, `admit/`, `policy_lock.json` | candidate ledger, nested max-t selection, Holm family of contrasts, admission arithmetic, locked policy |
| Final predictions and statistics | `final/q{0,N,C,D}_probabilities.npy`, `final/test_predictions.npz`, `final/results_summary.json`, `final/results.csv` | test-stage scores, paired effects with intervals |
| Timing | `stages/<stage>.json`, `select/candidates.json` | seconds per stage, serial wall time and work ledger per candidate |

`scripts/artifact_inventory.py <run> --check` lists every file with its hash and checks the
sealed chain; `scripts/reproduce_results.py <run>` recomputes every decision from the saved
predictions and labels without loading a checkpoint; `scripts/evaluate_checkpoints.py <run>
--trust-local-checkpoints` re-executes the checkpoints and compares their predictions with the
saved ones.

## 2. Packing on the cluster

```bash
make archive-run RUN_ROOT=/abs/outputs/CROSS_MORSE_arxiv_full_<timestamp> OUTPUT=../deposits/arxiv_full.tar.gz
make archive-campaign RUN_ROOT=/abs/roman_campaign OUTPUT=../deposits/roman_empire_full.tar.gz
python -B scripts/archive_run.py --verify ../deposits/arxiv_full.tar.gz
```

`scripts/archive_run.py` verifies the stage chain (every sealed artifact present with its
recorded hash and size, inputs equal to earlier sealed outputs, chain complete through
`evaluate`), then writes a tarball with `ARCHIVE_MANIFEST.json`: dataset, profile, whether the
run is an official-data scientific run, the run's source/config/unit identities, the
environment summary, and one SHA-256 per file. The archive's own SHA-256 is printed; record
it with the deposit. Nothing is unpickled during packing or verification.

Expected sizes for the full profiles are dominated by checkpoints (12 seeds × plant and q_0)
and candidate predictions; the Roman-empire campaign contains ten complete runs.

## 3. Depositing

Deposit each archive in a repository that mints a DOI (Zenodo, or the institutional
repository), one record per dataset or one record for all seven with one file each. Record in
the deposit description: the archive SHA-256, the runtime source identity inside the archive
(`ARCHIVE_MANIFEST.json → source_sha256`) and the commit hash of this repository whose
`make fingerprint` prints that identity.

## 4. Ingesting into this repository

```bash
make ingest-run ARCHIVE=../deposits/arxiv_full.tar.gz DOI=10.5281/zenodo.NNNNNNN
make runs-check
```

`scripts/ingest_run.py` verifies the archive, extracts it to `runs/<stem>/`
(`runs/roman_empire/split_NN/` for the campaign), re-verifies the stage chain from disk and
appends an entry to `runs/index.json` with the archive hash, the DOI, the run's source identity
and the source identity of this checkout. `make ingest-run` then refreshes
`manifests/official/` and the availability statement. `runs/` is git-ignored except for
`index.json` and `README.md`: the archives live at their DOI, the index records which ones were
verified.

## 5. Building the paper from the archives

```bash
make paper            # evidence records, tables and figures under paper/generated/official
make compare-runs     # agreement of run-derived values with paper/reported_results.json
```

`paper/build/evidence.py` reads one run (or campaign) into a flat evidence record;
`paper/build/tables.py` and `paper/build/figures.py` render the tables and figures listed in
`paper/README.md`. Every value is read from a run record. A quantity that the released runner
does not measure (today: the serving-time input-fallback probability) is printed as a dash
and named in the table note; nothing is estimated to fill a cell.

## 6. Relation between an archive and a reported number

Three different statements are kept apart:

1. **The archive verifies** — its files match their manifest and the stage chain is intact.
2. **The archive was produced by this code** — its `source_sha256` equals the fingerprint of a
   commit of this repository (`make fingerprint`).
3. **The archive is the run behind a reported number** — `make compare-runs` reports, per
   quantity, whether the run value equals the manuscript value at printed precision, lies
   inside the reported interval, or differs; a result binding
   (`scripts/compare_with_manuscript.py --emit-binding`, checked by
   `scripts/check_result_binding.py`) records the authors' declaration of that relation.

A numerical match alone establishes (1) and, with the fingerprint, (2); only the binding
states (3).

## Experiments outside the eight-stage run

The budget-matched arms, the inserted-site attribution, the extended ensemble grid, the
factorials, the negative controls, the cross-backbone substitutions, the input gate, the
retrained external baselines, the nominal timing and the executed sensor settings are not
produced by the eight stages. Their measurements enter as record sets under
[`records/`](../records/README.md): the raw logs, the code that produced them, and a
`summary.csv` whose every value `make records-check` re-reads from its raw file.
