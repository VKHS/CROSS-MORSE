# Complete executable examples

[Repository home](../README.md) · [Artifact contract](../docs/ARTIFACTS.md) ·
[Recomputed results](generated/metrics.md) · [Environment](../environment/README.md)

This directory contains seven completed **synthetic smoke runs** that we
executed with the released code. Each uses the dataset's task path and writes
the same kinds of artifacts as an official-data run: prepared inputs, exact role
membership, seed records, frozen predictors, fitted controller models,
exploration trajectories, predictions, statistical decisions and parent-linked
stage manifests.

The data are generated fixtures with four classes or labels. These archives
demonstrate execution and review; they are not our paper-producing benchmark
archives, which are deposited and ingested separately
([docs/RUN_ARCHIVES.md](../docs/RUN_ARCHIVES.md), [docs/STATUS.md](../docs/STATUS.md)).
The paper's measured values remain transcribed in [`paper/`](../paper) for
comparison. The per-run split, seed and environment manifests extracted from these
runs are in [`manifests/examples/`](../manifests/examples/), and the paper's tables
and figures built from them are in [`paper/generated/examples/`](../paper/generated/examples/)
(`make example-paper`).

## Contents

| Task path | Complete archive | Portable inventory | Input unit |
|---|---|---|---|
| ogbn-arxiv | [runs/arxiv](runs/arxiv) | [arxiv](inventory/arxiv.md) | 160 nodes in one graph |
| ogbn-proteins | [runs/proteins](runs/proteins) | [proteins](inventory/proteins.md) | 160 nodes in one graph |
| ogbn-products | [runs/products](runs/products) | [products](inventory/products.md) | 160 nodes in one graph |
| Roman-empire | [runs/roman_empire](runs/roman_empire) | [roman_empire](inventory/roman_empire.md) | 160 nodes in one graph |
| PascalVOC-SP | [runs/pascalvoc_sp](runs/pascalvoc_sp) | [pascalvoc_sp](inventory/pascalvoc_sp.md) | 80 complete graphs, eight nodes each |
| COCO-SP | [runs/coco_sp](runs/coco_sp) | [coco_sp](inventory/coco_sp.md) | 80 complete graphs, eight nodes each |
| Peptides-func | [runs/peptides_func](runs/peptides_func) | [peptides_func](inventory/peptides_func.md) | 80 complete graphs, eight nodes each |

The seven archives contain **2,170 files**, **2,051 stage-sealed artifacts**,
**350 `.pt` files** including fitting and resume states, **28 best selected
predictor checkpoints** (14 propagation-free and 14 plant checkpoints), and
**84 exploration trajectories with 576 recorded transitions**. File totals and
individual identities can be independently recomputed with the inventory tool.
Every run trains seeds `[1, 2]` for three epochs and logs 12 trajectories.
The resolved configuration and training manifest record the actual choices.

The `.npz` and `.npy` files include prepared data, concrete role and unit indices,
model predictions, exploration measurements, cross-fit/calibration partitions
and statistical observations. A `.pt` file may contain a selected model,
controller state or optimizer/resume state; it is not necessarily an additional
independent trained predictor. The inventory distinguishes these categories.

## Recompute the reported example table and figure

From the repository root, using the installed environment:

```bash
make audit-artifacts PYTHON=python
make reproduce PYTHON=python REPORT_ROOT=../cm_reports/examples
```

The first target verifies the complete files and recorded data roles without
NumPy, Torch or checkpoint deserialization. The second uses NumPy to recompute
selection, confirmation, admission, all four predictors' test scores and paired
intervals. Matplotlib renders the result figure. It performs no training,
downloads or checkpoint loading.

The generated outputs are:

- `metrics.csv`: 28 rows, one for each task/predictor pair, including metric
  fractions, losses, paired effects and source/configuration/unit identities.
- `metrics.md`: the human-readable seven-task table and its interpretation.
- `replay.json`: recomputed decisions, metrics and verified numerical inputs.
- `example_scores.svg` and `.png`: prediction-derived scores with the synthetic
  scope stated in the figure.
- `reproduction_manifest.json`: output file hashes and source identity.

The committed copies in [`generated/`](generated) were produced by that same
command. Verify their numerical contents without rewriting anything:

```bash
python -B scripts/reproduce_examples.py --runs-root examples/runs \
  --output-dir examples/generated --check
make example-paper-check PYTHON=python     # the committed paper tables built from these runs
make manifests-check PYTHON=python         # the committed per-run manifests
```

All seven included admission decisions return their selected comparator. The
`qD` and `qC` probability files are byte-identical, with zero paired loss
difference. This validates the fallback path. It does not demonstrate that
feedback improves official benchmark performance.

## Reexecute the actual checkpoints

For the checkpoints included in this repository, after reviewing their source
and hashes:

```bash
python -B scripts/validate_run_manifest.py examples/runs/arxiv \
  --require-complete --json

python -B scripts/evaluate_checkpoints.py examples/runs/arxiv \
  --trust-local-checkpoints --device cpu --threads 2 \
  --output ../cm_reports/arxiv_checkpoint_replay.json
```

The second command executes `q0`, `qN`, `qC` and `qD` using their recorded
configuration and compares the resulting probabilities with the saved outputs.
It requires Torch and explicitly enables loading trusted local checkpoint
objects. Numerical replay above does not perform that operation. The recorded
software validation checked all 28 predictor/task combinations; see
[VALIDATION.md](../docs/VALIDATION.md) for the actual environment and results.

Keep the included run directories immutable. Write reports elsewhere. Their
checksums and stage ancestry describe the bytes originally recorded; editing a
decision, prediction, checkpoint or split makes the corresponding checks fail.

## Train fresh examples

```bash
# One task configuration; a new run directory is required.
make train-sample PYTHON=python DATASET=ogbn-arxiv \
  RUN_ROOT=../cm_work/runs/arxiv DATA_ROOT=../cm_work/data

# All seven task configurations; these are synthetic, with no benchmark download.
make train-samples PYTHON=python WORK_ROOT=../cm_fresh
```

These targets create fresh data and execute the complete lifecycle. For resume,
repeat the corresponding dataset launcher command with `--resume`; it checks
the source/configuration/artifact identities before continuing. A fresh run has
its own timestamps, timings and serialized artifacts. Reproducibility of a
decision or numerical output should be assessed using the replay and checkpoint
tools, rather than by expecting an independently trained archive ZIP to have
the same byte hash.

For an official dataset, use its `pilot` or `full` profile and the acquisition
instructions in [`data/README.md`](../data/README.md). A pilot changes the compute
budget, not the official dataset's membership. The Roman synthetic fixture is
not its ten-official-split campaign; the campaign is documented in
[HPC](../docs/HPC.md).
