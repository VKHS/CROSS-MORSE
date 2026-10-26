# Getting started

[Repository home](../README.md) · [HPC and Slurm](HPC.md) · [Arrhenius](ARRHENIUS.md) · [Reviewer guide](REVIEWER_GUIDE.md)

The shared entry point runs the same eight-stage CROSS-MORSE lifecycle for all
seven datasets. Seven complete synthetic example runs are included, so you can
inspect their checkpoints, replay their predictions and regenerate their tables
before training anything. Use a real-data pilot to check the selected benchmark,
model and compute environment. Run commands from the repository root, keeping
new environments, caches, runs and reports outside the source tree.

## 1. Install the runtime

The declared minimum is Python 3.11; our compatibility workflow targets
Python 3.11 and 3.12. The bundled examples use a recorded Linux x86_64 CPU
environment with Python 3.12.14. Its complete observed package versions and an
installer are in [environment/](../environment/README.md). With a Python 3.12
interpreter, create a fresh environment from those pins:

```bash
bash environment/create_cpu_env.sh --python python3.12 --venv ../cm_cpu_env
source ../cm_cpu_env/bin/activate
python -B environment/check_environment.py
```

The installer refuses existing paths and checks the host and Python major/minor
before installing. Add `--dry-run` to inspect the commands without creating an
environment. The version check also reports whether Python's patch version
matches the recorded environment.

For another supported Python environment, the following manual alternative uses
the CPU Torch version in the compatibility workflow and the project's broader
dependency constraints. It may resolve different versions from the recorded lock:

```bash
python3 -m venv ../cm_env
source ../cm_env/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
```

[requirements.txt](../requirements.txt) supplies the numerical and dataset
packages. PyG, `torch-scatter` and RDKit are not required; the peptide reader uses
the distributed tensor features. For GPU execution, select a
host-compatible CUDA-enabled PyTorch environment; the CPU wheel above is only
for the local examples. The [HPC guide](HPC.md) explains interpreter selection,
optional setup and cluster modules.

Running commands from the checkout does not require installing the project
itself. An editable installation is optional for development:

```bash
python -m pip install --no-deps -e .
```

The installed distribution contains the `cross_morse` package. The launchers,
configs, review scripts and manuscript registry are source-tree resources, so
continue running the documented commands from the checkout. No installed console
command is assumed.

To review saved numeric observations without training or checkpoint loading,
use [requirements-review.txt](../requirements-review.txt) in a separate environment
and follow the [reviewer guide](REVIEWER_GUIDE.md). The optional figure scripts
need Matplotlib; `python -m pip install -e '.[plots]'` adds that extra. It is not
required for training, numeric replay or the CLI configuration checks.

## Replay the bundled examples and regenerate views

The [example runs](../examples/README.md) retain the complete eight-stage
payload, including prepared graph/split arrays, predictor and controller
checkpoints, optimizer state, trajectories, candidate observations, final
probabilities and parent-linked manifests. They contain generated synthetic
data and are software examples, not official benchmark results.

Verify their inventories and the committed prediction-derived tables, then
regenerate the tables and figures into an external report directory:

```bash
make audit-artifacts PYTHON=python
make example-views-check PYTHON=python
make reproduce PYTHON=python REPORT_ROOT="../cm_reports/bundled examples"
```

`make reproduce` verifies every sealed file, recomputes selection, confirmation,
admission and test metrics from saved observations, and writes `metrics.csv`,
`metrics.md`, `replay.json`, `example_scores.svg`, `example_scores.png` and
`reproduction_manifest.json`. The CSV has four predictor rows for each of the
seven task configurations. It does not download data, import PyTorch, load model
checkpoints or train. NumPy and Matplotlib are sufficient. The command can
refresh its own unchanged generated output directory; it refuses unrelated files
or externally edited reports. The committed views live in
[examples/generated/](../examples/generated/).

To recompute predictions from a bundled checkpoint instead, explicitly invoke:

```bash
make reexecute PYTHON=python RUN_ROOT=examples/runs/arxiv \
  REPORT_ROOT="../cm_reports/arxiv checkpoint check" DEVICE=cpu THREADS=2
```

This separate check loads the bundled checkpoints, uses PyTorch and creates a
new report. See [Artifacts](ARTIFACTS.md) for the file categories and the
difference between numerical replay and checkpoint reexecution.

The paper's tables and figures are built from completed run directories, never
from the example scores or from the manuscript registry:

```bash
make example-paper PYTHON=python        # from the seven synthetic examples, into paper/generated/examples
make example-paper-check PYTHON=python  # the committed example build equals a fresh one
make paper PYTHON=python                # from the official run archives registered in runs/index.json
make manifests PYTHON=python            # registered and per-run split/seed/environment manifests
```

`make paper` (equivalently `make paper-from-runs`; `make tables` and `make figures`
for one kind) writes evidence records, tables and figures under
`paper/generated/official`; see [paper/README.md](../paper/README.md) and
[RUN_ARCHIVES.md](RUN_ARCHIVES.md). The transcribed manuscript registry has its
own targets: `make registry-views` redraws its views (`REGISTRY_FIGURE_ROOT` and
`DIAGRAM_ROOT` redirect the figure outputs) and `make registry-check` verifies
them; `make diagrams` writes the deterministic SVG diagrams, `make diagrams-png`
also refreshes their PNG copies and `make diagrams-check` checks both without
rendering. `make verify` runs everything CI runs on a fresh clone: the release
audit, the source fingerprint, the registry checks, the example inventories and
replay, the example paper build, the manifest checks and the availability check;
it needs NumPy and performs no downloads or training. `make audit` is the
standard-library release-integrity check. If you deliberately change tracked
source or generated views, regenerate the release manifest only after validating
those changes.

## 2. Select a dataset and profile

Use the exact canonical dataset name with `run_cross_morse.py --dataset`.
Dataset-specific Python and shell wrappers select the same shared implementation.

| Canonical dataset | Config/wrapper stem | Prediction task | Benchmark metric |
|---|---|---|---|
| `ogbn-arxiv` | `arxiv` | Multiclass nodes | Accuracy |
| `ogbn-products` | `products` | Multiclass nodes | Accuracy |
| `ogbn-proteins` | `proteins` | Multilabel nodes, 112 outputs | ROC-AUC |
| `Roman-empire` | `roman_empire` | Multiclass nodes; ten official split masks | Accuracy |
| `PascalVOC-SP` | `pascalvoc_sp` | Multiclass superpixel nodes in independent graphs | Macro-F1 |
| `COCO-SP` | `coco_sp` | Multiclass superpixel nodes in independent graphs | Macro-F1 |
| `Peptides-func` | `peptides_func` | Multilabel graphs, ten outputs | Average precision |

Each stem has three files under [configs](../configs/):

| Profile | Data and purpose |
|---|---|
| `smoke` | Generated task-shaped fixture, FP32, two seeds, three training epochs and 12 exploration trajectories; `dataset=synthetic`, `scientific_run=false`, downloads disabled |
| `pilot` | Actual benchmark data and official splits, with reduced width/depth, two seeds, at most 30 training epochs and 24 exploration trajectories |
| `full` | Dataset-specific full model, optimizer and exploration settings, with 12 training seeds |

The real-data pilot retains the full graph or graph corpus; it is not a sampled
miniature benchmark. It keeps the support threshold `n_min=64` with only 24
exploration trajectories, so its purpose is integration and resource measurement.
A comparator return is a valid outcome. Use the full protocol to assess
qualification under the registered budget.

The [compute guide](COMPUTE_BUDGET.md) separates the profiles' actual resource
drivers from the manuscript's reported timings and gives commands for measuring
a run on your hardware. A pilot still uses the complete official graph or corpus.

Inspect a resolved configuration without importing the numerical runtime,
downloading data or creating a run:

```bash
python -B run_cross_morse.py \
  --dataset Roman-empire --profile full --split-index 3 --print-config
```

`--config PATH` selects a custom JSON file instead of the profile-based path.
If `--dataset` or `--profile` is also supplied, it must agree with that file.
Wrappers reject a conflicting dataset. With no explicit config, `--profile`
defaults to `pilot`.

## 3. Run synthetic checks

Start with a small node-classification fixture:

```bash
python -B run_cross_morse.py \
  --dataset ogbn-arxiv --profile smoke \
  --device cpu --threads 2 \
  --data-root ../cm_data --output-dir ../cm_runs/arxiv_smoke \
  --stage all
```

The transductive fixtures contain 160 generated nodes. The three independent-graph
fixtures contain 80 generated graphs with eight nodes each; Peptides smoke uses
graph labels, while Pascal/COCO smoke use node labels. These are software checks,
not benchmark measurements. Their saved dataset identity remains `synthetic`.

To train all seven task-shaped fixtures afresh, use an external work directory:

```bash
make train-samples PYTHON=python WORK_ROOT="../cm_work/fresh examples" THREADS=2
```

This explicitly trains, explores and fits the controller for each fixture on
CPU; it is distinct from replaying the bundled results. Outputs are
`WORK_ROOT/runs/<stem>`. `make train-sample DATASET=ogbn-arxiv RUN_ROOT=...` trains
one fixture. The aliases `sample`, `samples` and `smoke` also train; their names
are listed with this behavior by `make help`. Use fresh run paths.

Regenerate tables and figures from those newly trained runs by supplying their
run root directly; this checks their own seals rather than the bundled inventories:

```bash
python -B scripts/reproduce_examples.py --runs-root "../cm_work/fresh examples/runs" \
  --output-dir "../cm_reports/fresh examples" --figures
```

After the first example completes, validate the stage/model records and replay
its saved numerical observations:

```bash
python -B scripts/validate_run_manifest.py ../cm_runs/arxiv_smoke \
  --require-complete --json

python -B scripts/reproduce_results.py ../cm_runs/arxiv_smoke \
  --output ../cm_reports/arxiv_smoke_replay.json
```

Numerical replay recomputes metrics and decisions without retraining or loading
checkpoints. It creates JSON and Markdown reports at new paths outside the run.
The [reviewer guide](REVIEWER_GUIDE.md) also covers frozen-checkpoint execution and
portable evidence bundles.

## 4. Execute a real-data pilot or full run

Official datasets are acquired through the adapters using the source locations
and access notes in [data/README.md](../data/README.md) and
[data/sources.json](../data/sources.json). The synthetic examples do not contain
those benchmark datasets. To download missing official files and prepare their
splits without training, explicitly request the preparation target:

```bash
make prepare-data PYTHON=python DATASET=Roman-empire PROFILE=pilot \
  DATA_ROOT="../cm_data" RUN_ROOT="../cm_runs/roman_prepared" DEVICE=cpu THREADS=2
```

This target accepts `PROFILE=pilot` or `PROFILE=full`, rejects `smoke`, and runs
only `--stage prepare`. It may download substantial official data. It never
starts model training. The dataset's complete graph or graph corpus is retained;
preparation still needs the corresponding disk and host-memory capacity.

On the intended GPU host, activate its CUDA-enabled environment and check the
sparse/dense operations:

```bash
python -B run_cross_morse.py --gpu-health-check
```

All our real-data profiles except Roman-empire use BF16. For those profiles,
look for both `"status": "PASS"` and `"bf16": true`; the health check reports BF16
capability separately. Roman uses FP32. Changing `--device` alone does not turn a
BF16 profile into a CPU-compatible configuration.

For example, run a proteins pilot:

```bash
python -B run_cross_morse.py \
  --dataset ogbn-proteins --profile pilot \
  --device cuda --threads 16 \
  --data-root ../cm_data --output-dir ../cm_runs/proteins_pilot \
  --stage all
```

The equivalent dataset-specific entry point is
`python run_cross_morse_proteins.py --profile pilot ...`. Explicit config usage is
also supported, for example `--config configs/proteins_pilot.json`.

The real-data configs enable `data.download=true`. Missing official raw data are
downloaded into `--data-root`; complete raw caches can be reused. The first run
therefore needs network access unless the cache was populated beforehand. An OGB
processed-pickle cache alone does not replace the official raw files. To run
offline, copy the config outside the checkout, set `data.download=false`, and
point `--config` to that copy after preparing the raw cache.

The [dataset adapters](../cross_morse/datasets.py) use these locations relative
to the shared cache root:

| Dataset | Raw cache layout |
|---|---|
| OGB datasets | `ogbn_arxiv/`, `ogbn_products/`, `ogbn_proteins/`, each containing `raw/` and the official `split/` files |
| Roman-empire | `roman_empire/raw/roman_empire.npz` |
| PascalVOC-SP | `pascalvoc-sp/raw/{train,val,test}.pickle` |
| COCO-SP | `coco-sp/raw/{train,val,test}.pickle` |
| Peptides-func | `peptides-func/raw/{train,val,test}.pt` |

### Full-profile settings

The configured full-model values are listed below; these are settings, not
measured performance or parameter-count claims. Every row uses 12 training seeds.

| Dataset | Backbone family | Hidden width | Controlled depth | Precision | Maximum training epochs | Exploration trajectories |
|---|---|---:|---:|---|---:|---:|
| ogbn-arxiv | Shared transport, four-layer encoder | 512 | 256 | BF16 | 500 | 12,800 |
| ogbn-products | Shared transport, three-layer encoder | 512 | 64 | BF16 | 300 | 19,200 |
| ogbn-proteins | Edge-weighted transport, four-layer encoder | 512 | 128 | BF16 | 400 | 14,400 |
| Roman-empire | Heterophily encoder, four layers | 384 | 256 | FP32 | 800 | 9,600 |
| PascalVOC-SP | GPS-style, six native blocks | 128 | 128 | BF16 | 300 | 20,480 |
| COCO-SP | GPS-style, six native blocks | 128 | 128 | BF16 | 300 | 32,000 |
| Peptides-func | Linear GINE-style, eight native blocks | 128 | 64 | BF16 | 250 | 24,000 |

Controlled depth and the number of native backbone blocks are separate settings
for GPS/GINE. Training applies the recorded early-stopping patience. Peptides
selects the internal-fit checkpoint by maximum AP with a BCE tie-break; the
other tasks use minimum validation loss. The full superpixel graph-batch size is
32, and the Peptides graph-batch size is 128.

Products uses exact full-graph sparse propagation with a 65,536-row dense/readout
chunk limit. That limit is not neighbor sampling or a bound on total GPU memory.
Independent graphs are executed as separate episodes. See the
[implementation contract](IMPLEMENTATION_CONTRACT.md) for model and execution
semantics.

After inspecting the pilot and allocating sufficient resources, choose a fresh
output directory for the full run:

```bash
python -B run_cross_morse.py \
  --dataset ogbn-proteins --profile full \
  --device cuda --threads 16 \
  --data-root ../cm_data --output-dir ../cm_runs/proteins_full \
  --stage all
```

Full real-data workloads can require several allocations; [HPC and Slurm](HPC.md)
provides snapshots and resume commands. The manuscript timings in
[COMPUTE_BUDGET.md](COMPUTE_BUDGET.md) are reported observations, not a completion
estimate for this checkout or your hardware. When the eight stages have
completed, compare the run with the manuscript's reported values for its dataset:

```bash
python -B scripts/compare_with_manuscript.py ../cm_runs/proteins_full \
  --output ../cm_reports/proteins_vs_manuscript.json
```

## 5. Run Roman-empire splits

`--split-index` accepts integers 0 through 9 and applies only to Roman-empire.
The chosen index is stored in the effective configuration. Without an override,
the config's `data.split_index` applies; the shipped configs select split 0.

```bash
python -B run_cross_morse_roman_empire.py \
  --profile pilot --split-index 3 --device cuda --threads 16 \
  --data-root ../cm_data --output-dir ../cm_runs/roman_pilot_split_03
```

Use the [ten-split Slurm campaign](HPC.md#6-run-all-ten-roman-empire-splits) for a
common frozen source/config and outputs named `split_00` through `split_09`. Once
all ten runs are complete, summarize them at a new path outside the campaign:

```bash
python -B scripts/summarize_roman_splits.py /absolute/path/to/roman-campaign \
  --output ../cm_reports/roman_split_summary.json
```

The summary checks all ten sealed runs and common non-split settings. Its paired
split-level Student-t intervals describe variation across the ten official masks
on one graph. They do not turn overlapping splits into independent graphs or
replace each split's deployment decision.

## 6. Resume and inspect outputs

Resume the synthetic example with its original config, source and output path:

```bash
python -B run_cross_morse.py \
  --dataset ogbn-arxiv --profile smoke --device cpu --threads 2 \
  --data-root ../cm_data --output-dir ../cm_runs/arxiv_smoke \
  --stage all --resume
```

For a Roman run, retain its split index too. Completed stages are verified and
reused; unfinished work continues from saved checkpoints. Changed computation
source or configuration is rejected. Direct Python runs record source identity
but do not copy the whole checkout; keep that exact source. Slurm suites contain
their own source/config snapshot.

Stages are `prepare`, `train`, `explore`, `fit`, `select`, `confirm`, `admit` and
`evaluate`, in that order. `all` runs the chain. To stop after data preparation,
start with `--stage prepare`, then continue the same output directory using
`--stage all --resume`. A later individual stage requires completed predecessors.
`train` fits the predictor; `fit` fits controller-side models after exploration.

| Location inside a run | Contents |
|---|---|
| `config.json`, `environment.json`, `prepared/` | Effective task/config, environment, graph data, splits and statistical-unit metadata |
| `train/` | Predictor checkpoints, training records and actual model inventory |
| `explore/`, `fit/` | Trajectories, controller-side fits and calibration |
| `select/`, `confirm/`, `admit/`, `policy_lock.json` | Comparator choice, held-out decisions and locked deployment |
| `final/results_summary.json` | Computed task metrics and final deployment outcome |
| `final/test_predictions.npz`, `final/paired_test_units.npz` | Saved observations and paired-unit records for replay |
| `provenance/stages/` | Hash-linked records of the eight executable stages |
| `source/`, `launcher/` in Slurm suites | Frozen source/config/profile, job logs and execution records |

The [validation record](VALIDATION.md) identifies which software, fixture and
real-data checks were actually executed and what remains untested. It is separate
from our measured experiments. [Results and evidence](RESULTS.md)
explains how to connect a reported value to its result-producing run.

`make help` lists every target. `make reproduce` refreshes its own generated
example-report directory after checking ownership and prior file hashes.
Single-run tools such as `replay`, `reexecute`, `review-bundle` and `compare`
require fresh report paths. Training targets require a fresh run, while the
Python CLI's explicit `--resume` continues an existing sealed run. Existing
Slurm entrypoints and external `CM_ENV_FILE` profiles are unchanged; follow the
[HPC guide](HPC.md) for the generic launcher contract and
[ARRHENIUS.md](ARRHENIUS.md) for the cluster we run on, including the site
profile `configs/arrhenius-site.env.example`, submission, resume and archiving.
