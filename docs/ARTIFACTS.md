# Experiment artifacts and reproducibility

Our repository provides three complementary forms of evidence: the measurements
we report in the submitted manuscript (transcribed for comparison), the
executable benchmark configurations, and complete saved software examples for
all seven task adapters. Each has a specific review route. The official-data run
archives behind the reported numbers follow the contract in
[RUN_ARCHIVES.md](RUN_ARCHIVES.md); their deposit status is in [STATUS.md](STATUS.md).

## Included evidence

| Evidence | Location | Scope |
|---|---|---|
| Manuscript and supplement identities | [`paper/sources.json`](../paper/sources.json) (SHA-256, size, pages; the PDFs are not distributed) | Which document version every transcribed value and page reference belongs to |
| Transcribed measurements and provenance | [`paper/reported_results.json`](../paper/reported_results.json), [`paper/sources.json`](../paper/sources.json), [`PAPER_RESULTS.md`](PAPER_RESULTS.md) | Traceable reported values and source tables, used for comparison only; no invented seed observations |
| Official data-source and split recipe catalogue | [`data/sources.json`](../data/sources.json) | Source URLs, adapter identities and executable split rules |
| All configured profiles and seeds | [`experiments/profiles.json`](../experiments/profiles.json), [`manifests/registered/`](../manifests/registered/) | Generated inventory of the 21 actual configuration files; registered split rule, seeds, folds and margins per full/pilot profile |
| Per-run split, seed and environment manifests | [`manifests/examples/`](../manifests/examples/) (`manifests/official/` once archives are ingested) | Role membership hashes, seeds, bootstrap seeds and recorded environment of every completed run (`make manifests`) |
| Seven complete saved runs | [`examples/runs/`](../examples/runs/) | Actual synthetic executions, with data, checkpoints, predictions, trajectories and statistical records |
| Per-run file and numerical-structure inventories | [`examples/inventory/`](../examples/inventory/) | Every recorded path, SHA-256, role count, checkpoint identity and trajectory index |
| Observed execution environment | [`environment/`](../environment/) and each run's `environment.json` | Captured CPU environment and package versions; each run preserves its own environment record; the GPU record is captured on the cluster by `environment/arrhenius/capture_environment.sh` |
| Paper tables and figures built from runs | [`paper/generated/examples/`](../paper/generated/examples/) (`paper/generated/official/` once archives are ingested) | Every table and figure of the paper that the run contract supports, built by `make example-paper` / `make paper` |

The benchmark-scale run archives behind our reported measurements—split/seed
membership, trained checkpoints, per-unit predictions, complete trajectories and
statistical resamples—are produced on Arrhenius, packed with
`scripts/archive_run.py`, deposited with a DOI and ingested into `runs/` with
`scripts/ingest_run.py`; until they are, `runs/index.json` is empty and the
generated [availability statement](../AVAILABILITY.md) says so. Our reported
measurements remain available and are labelled as transcribed. The bundled
synthetic executions provide complete, inspectable software artifacts; their
scores do not stand in for the reported benchmark results. No missing download
URL, checkpoint hash or historical seed list is invented.

## Start with a saved example

From the repository root:

```bash
# Standard library only: verify actual bytes, stages and split membership.
python -B scripts/artifact_inventory.py --check-static
python -B scripts/artifact_inventory.py examples/runs/arxiv \
  --output examples/inventory/arxiv.json --check

# NumPy: recompute all saved statistical decisions and final scores.
python -B scripts/reproduce_results.py examples/runs/arxiv \
  --output /tmp/cross_morse_arxiv_replay.json

# NumPy/Matplotlib: replay all seven and render prediction-derived tables/figures.
python -B scripts/reproduce_examples.py --runs-root examples/runs \
  --output-dir /tmp/cross_morse_examples_review --figures
```

Use new external report destinations for replay. `--check` compares an existing
inventory to recomputed contents and writes nothing. `make audit-artifacts`
checks all seven committed inventories; `make reproduce` invokes the saved-run
replay route. These routes do not download data or train models.

### Checkpoint execution

The full run directories include trained models and their prepared data. To
reexecute trusted local checkpoints and compare their predictions:

```bash
python -B scripts/evaluate_checkpoints.py examples/runs/arxiv \
  --trust-local-checkpoints --device cpu --threads 2 \
  --models q0 qN qC qD --output /tmp/cross_morse_arxiv_checkpoints.json
```

This command loads `.pt` files, so the explicit trust flag belongs to this
execution route. The byte inventory and numeric replay do not deserialize
checkpoints. Prediction agreement is tested at the command's declared numerical
tolerances. It does not rerun training or establish identical speed on a
different machine.

## What a complete run contains

| Question | Artifacts | What is verified or available |
|---|---|---|
| Which code and configuration ran? | `config.json`, `run_state.json`, `provenance/stages/*.json`; runtime source files | Content identities, contiguous parent links and current-source compatibility |
| Which machine and packages were recorded? | `environment.json` | Actual captured environment, device/precision and package metadata |
| Which examples belong to which role? | `prepared/official_splits.npz`, `splits.npz`, `graph_splits.npz` when applicable | Exact IDs; complete disjoint leaf allocation, internal fit union and official split correspondence |
| Which observations are resampled? | `prepared/clusters.npz`, `metadata.json`; unit IDs/counts in stage NPZ files | Core or graph identity and role occupancy |
| Which weights were selected? | `train/q0_seed*_best.pt`, `train/plant_seed*_best.pt`, `train/manifest.json`, `train/model_inventory.json` | Separate baseline/plant checkpoints per seed, selection rule and actual bytes |
| Did the plant/head remain frozen? | `provenance/frozen_models.json`, every later stage's `model_audit` | Checkpoint-byte equality and recorded before/after backbone/head hash linkage; tensor recomputation requires the full provenance/checkpoint route |
| What exploration data trained the controller? | `explore/trajectory_*.npz` and companion `.json`, `explore/manifest.json`, `nominal_reference.npz` | Numeric state transitions, action/propensity/target arrays, seed index, graph identity where applicable and complete action sequence |
| Which learned controller was used? | `fit/{full,lean}/controller.pt`, fit audit/calibration JSON and NPZ files | Controller checkpoint, partitions, predictions, radii and assessment records |
| Which candidates were compared? | `select/candidates.json`, `candidate_ledger.csv`, `unit_scores.npz`, optional `metric_observations.npz`, `decision.json` | Registered candidates, costs, fold IDs, unit scores and nested selection decisions |
| Which confirmation contrasts were tested? | `confirm/contrast_registry.json`, `paired_contrasts.json`, `decision.json` | Declared per-run contrast family, paired observations and Holm decisions |
| Why was the final policy admitted or rejected? | `admit/pre_admission_lock.json`, `paired_units.npz`, `decision.json`, `policy_lock.json` | Locked comparator/candidate, paired admission observations and decision |
| Which predictions produced the final scores? | `predictions/`, `member_predictions/`, `final/q{0,N,C,D}_probabilities.npy`, `final/test_predictions.npz` | Candidate/member predictions, final probability arrays, test labels and prediction-row/unit mapping |
| What numerical results are replayed? | `final/paired_test_units.npz`, `results_summary.json`, `results.csv` | Saved unit losses/metrics, paired intervals, model scores and reporting scope |

Exploration trajectory count is distinct from recorded transition count and
from independent evaluation-unit count. Graph episodes can recur across
trajectories and training seeds. The inventory reports all three concepts
separately instead of interpreting repeated transitions as additional
independent graphs.

## Integrity checks and their limits

`scripts/artifact_inventory.py` uses the standard library. It requires all
eight stage manifests and every sealed file. Each repeated input reference is
checked against the actual file once per invocation; the cache is not reused
between invocations. It verifies parent links, source/config/split identities,
the final evaluator identity and output allowlist, and the frozen-checkpoint
byte references. Completed scientific directories cannot acquire an unsealed
artifact unnoticed.

The tool reads NPY/NPZ headers without pickle and checks their declared shape,
primitive dtype and payload length. Fixed-width strings used for calibration
metadata are supported; Python-object arrays are rejected. It reads actual
integer split vectors to verify role membership, and validates whole-graph
mapping for independent datasets. Full benchmark split checks require memory
proportional to the saved index vectors; checkpoint files are hashed in chunks.
Numeric array values, confidence intervals, metric arithmetic and checkpoint
forward passes are checked by the separate replay commands.

The inventory does **not** recompute logical tensor hashes, authenticate the
historical machine or prove that logs were produced by an independent party.
It crosschecks the recorded tensor hashes and verifies the actual serialized
checkpoint bytes. Source and output hashes are content identities, not digital
signatures or statistical certificates. Ancillary execution records outside
stage seals are identified as `unsealed_ancillary` in the inventory.

The output JSON and Markdown use relative artifact paths and omit their own
generation time, absolute run directory and file modification times. Therefore
`--check` works after the repository or run directory is moved, provided its
contents and corresponding runtime source are unchanged.

## Statistical interpretation

The comparison is paired on the run's recorded graph/core units. `qC` is the
selected open-loop comparator; `qN` is the frozen nominal full-depth plant at
the comparator's ensemble size; `qD` is the deployed choice after admission.
Failed admission uses the saved `qC`
probabilities exactly. The inventory checks their byte identity when the final
record declares exact fallback.

The numerical replay recomputes nonlinear metrics from prediction/label
observations, rather than substituting mean per-node F1/AUC/AP values. Its
intervals are conditional on the fixed trained ensemble. Core-based inference
on a single transductive graph remains descriptive. Synthetic graph fixtures
exercise the independent-graph branch but do not supply an empirical certificate
for the manuscript's reused-validation measurements. See
[`THEORY_AND_SCOPE.md`](THEORY_AND_SCOPE.md) and [`RESULTS.md`](RESULTS.md).

## Add a real completed benchmark archive

The deposit and ingestion route is `make archive-run`, `make ingest-run`,
`make paper` and `make compare-runs` ([RUN_ARCHIVES.md](RUN_ARCHIVES.md)). To
inspect a completed run by hand, keep its original source snapshot and entire run
unchanged and, with the matching runtime source, write a new inventory
**outside** that run:

```bash
python -B scripts/artifact_inventory.py /path/to/completed_run \
  --output /path/to/review/completed_run_inventory.json
python -B scripts/reproduce_results.py /path/to/completed_run \
  --output /path/to/review/completed_run_numeric_replay.json
python -B scripts/evaluate_checkpoints.py /path/to/completed_run \
  --trust-local-checkpoints --device cuda \
  --output /path/to/review/completed_run_checkpoint_replay.json
```

An explicit manuscript binding can reference those already existing identities
and state whether the archive is the historical author run, an independent
rerun, or a numerical comparison. The comparison script does not turn a newly
trained run into the historical experiment or a reported interval into a
replication acceptance band. For a portable analysis-only export, use
`scripts/build_review_bundle.py`; that smaller format deliberately omits
checkpoints and the prepared graph, so it cannot pass this **complete-run**
inventory check. Preserve the full archive separately for model reexecution.
