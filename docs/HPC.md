# HPC and Slurm

[Repository home](../README.md) · [Getting started](GETTING_STARTED.md) · [Arrhenius](ARRHENIUS.md) · [Reviewer guide](REVIEWER_GUIDE.md)

This page is the generic launcher contract. The cluster we run on is Arrhenius;
its site profile ships as [`configs/arrhenius-site.env.example`](../configs/arrhenius-site.env.example)
(architecture, partition, job shape, storage layout, environment), and
[ARRHENIUS.md](ARRHENIUS.md) walks through setup, submission, resume and
archiving there. The git-ignored `configs/arrhenius.env.example` is our private
copy of that profile with the allocation and storage-project identifiers filled
in; the launcher discovers it automatically.

The [shared launcher](../run_cross_morse.sh) runs one Python process per Slurm
task. Each task requests **one GPU and 16 CPU threads** by default. Dataset
wrappers such as `run_cross_morse_products.sh` call the same implementation.
Every experiment suite receives a source/config snapshot, an effective cluster
profile and append-only launcher logs.

Use a Linux login shell with Bash 4+, Python 3.11+, `readlink`, `flock` and Slurm
commands. Source, data, environment and output paths must be visible from compute
nodes. Keep generated data and suites outside the checkout. Python 3 on the login
node validates configs and creates campaign snapshots using only the standard
library; computation runs in the explicitly selected GPU environment.

## 1. Create or select your cluster profile

From the repository root, copy the public example outside the source tree:

```bash
mkdir -p "$HOME/cross_morse_profiles"
cp configs/public.env.example "$HOME/cross_morse_profiles/cluster.env"
```

Edit that file, replacing the placeholders with your allocation, partition and
absolute paths. An existing site-specific profile can be selected directly.

```bash
CM_ACCOUNT="YOUR_ALLOCATION"
CM_PARTITION="YOUR_GPU_PARTITION"
CM_BASE="/absolute/path/to/project-work"
CM_VENV_DIR="/absolute/path/to/gpu-python-environment"
CM_THREADS=16
CM_BUILD_MODULE=""
CM_PYTHON_MODULE=""
CM_EXPECT_ARCH=""
```

Select it and inspect a proposed run:

```bash
export CM_ENV_FILE="$HOME/cross_morse_profiles/cluster.env"
bash run_cross_morse.sh --dataset ogbn-arxiv dry-run pilot
```

The dry run validates dataset/profile/split routing and prints the account,
paths, modules and quoted Python command. It creates no directories, installs
no packages and submits no job. Profiles are trusted sourced Bash files.

### Profile precedence and lookup

Routing flags (`--dataset`, `--split-index`, `--data-root`, `--config`) override
the corresponding environment values. Exported `CM_*` values override the
selected profile, including explicitly exported empty values. The profile then
supplies values above generic launcher defaults. `CM_ENV_FILE` selects one
complete profile, rather than combining several files.

Without `CM_ENV_FILE`, the launcher checks these directories in order:

1. `CM_PROJECT_DIR`, when supplied.
2. The directory containing the shared launcher.
3. `SLURM_SUBMIT_DIR`, when available.

Within each directory, it tries `configs/active.env.example`, then
`configs/arrhenius.env.example`, then `configs/public.env.example`, and stops at
the first match. The public source includes the generic profile and the Arrhenius
site profile `configs/arrhenius-site.env.example`. A suite generates its own
`active.env.example`; a site profile provides allocation, storage, module and
architecture settings without changing the shared script.

For Arrhenius, copy `configs/arrhenius-site.env.example` to
`configs/arrhenius.env.example` (git-ignored) and fill in `CM_ACCOUNT` and
`CM_STORAGE_PROJECT`; everything else in that profile—architecture check,
partition, storage layout, interpreter—is already set for the cluster. The public
examples deliberately contain no allocation identifiers. `CM_STORAGE_PROJECT`
records your storage project; `CM_ACCOUNT` selects the Slurm allocation. Confirm
that the cache and suite directories lie on the intended project storage through
`dry-run` before submission.

| Setting | Meaning and default |
|---|---|
| `CM_ACCOUNT` | Required for `submit`; passed explicitly as `sbatch --account` |
| `CM_PARTITION` | Submission partition; default `gpu` |
| `CM_BASE` | Work root; default `$HOME/cross_morse_work` |
| `CM_VENV_DIR` | Environment containing `bin/python`; public-profile default `$CM_BASE/code/venv_cross_morse` |
| `CM_DATA_ROOT` | Common dataset-cache root; default `$CM_BASE/datasets/ogb_cache` |
| `CM_PROJECT_DIR` | Optional explicit source directory |
| `CM_DATASET` | Canonical dataset; generic-shell default `ogbn-arxiv`; dataset wrappers select their own |
| `CM_SPLIT_INDEX` | Optional Roman-empire index 0–9; leave unset for other datasets and ten-split campaigns |
| `CM_CONFIG` | Custom JSON for a new run; validated against the selected dataset/profile |
| `CM_THREADS` | Numerical-library/Python threads; default 16; cannot exceed allocated CPUs |
| `CM_BUILD_MODULE`, `CM_PYTHON_MODULE` | Optional module names; public defaults are empty |
| `CM_EXPECT_ARCH` | Optional required architecture such as `aarch64` or `x86_64` |
| `CM_STAGE` | `all`, or an individual lifecycle stage |
| `CM_TIME` | Walltime override for the submission wrapper |
| `CM_TORCH_SPEC`, `CM_TORCH_INDEX` | Optional explicit setup-only PyTorch requirement and wheel index |
| `CM_STORAGE_PROJECT` | Recorded storage metadata; does not create storage or select the Slurm allocation |

All seven readers accept the common dataset-cache root. Quote paths containing
spaces. The generic profile intentionally leaves the allocation and a
host-specific PyTorch installation recipe for the user to supply.

## 2. Select or prepare the Python environment

The launcher executes **`$CM_VENV_DIR/bin/python`**. Activating another environment
in the login shell does not select it. Point `CM_VENV_DIR` at a compute-node
compatible environment with CUDA-enabled PyTorch and
[requirements.txt](../requirements.txt).

If either module setting is nonempty, the launcher requires the module command,
purges modules, then loads the selected build module followed by the Python
module. If both are empty, it keeps the job's existing module environment.
`CM_EXPECT_ARCH` checks both host and interpreter architecture when supplied.

Skip setup when that environment is ready. To install dependencies explicitly
inside a GPU allocation, submit:

```bash
bash run_cross_morse.sh submit setup
```

Setup creates the environment only if missing, installs the requirements, runs
`pip check` and checks CUDA operations. It preserves importable PyTorch. If
PyTorch is absent, set a host-compatible `CM_TORCH_SPEC` and optional
`CM_TORCH_INDEX` first; setup does not guess a wheel. A CPU-only PyTorch build is
preserved and then fails the CUDA check. Wait for setup to succeed before
submitting experiments. Normal run modes never install packages.

Slurm smoke also uses a GPU. Its preflight checks NumPy, SciPy and PyTorch; dataset
packages are loaded by the relevant adapter. For a CPU smoke check, use the
[direct Python commands](GETTING_STARTED.md#3-run-synthetic-checks).

## 3. Submit one dataset and profile

The seven canonical names and wrapper stems are listed in
[Getting started](GETTING_STARTED.md#2-select-a-dataset-and-profile). A wrapper
makes repeated commands shorter:

```bash
bash run_cross_morse_products.sh dry-run smoke
bash run_cross_morse_products.sh submit smoke
```

After checking that job, run the actual products graph with reduced computation:

```bash
bash run_cross_morse.sh --dataset ogbn-products submit pilot
```

Use the pilot's memory and execution records to plan a full run:

```bash
bash run_cross_morse_products.sh submit full
```

| Mode | Work | Default requested walltime |
|---|---|---|
| `setup` | Explicit environment preparation and CUDA check | 1 hour |
| `smoke` | Generated task-shaped fixture | 1 hour |
| `pilot` | Real benchmark graph/corpus, reduced training and exploration | 4 hours |
| `full` | Full configured dataset protocol | 24 hours |
| `resume` | Continue an existing suite | 24 hours |

These limits are scheduling requests, not completion estimates. `submit` without
a mode defaults to `pilot`. Campaign tasks use the time associated with their
selected profile. Override the wrapper's request when appropriate:

```bash
CM_TIME=08:00:00 bash run_cross_morse_proteins.sh submit pilot
```

The wrapper requests one GPU and the selected CPU-thread count; it does not set
a host-memory or GPU-model constraint. Allocation defaults therefore still
apply. If your site requires additional resource flags, use direct `sbatch`
with the same script and external profile. See [compute planning](COMPUTE_BUDGET.md)
for dataset-specific memory and workload considerations.

The real-data profiles use BF16 except Roman-empire, which uses FP32. BF16 profiles
require the CUDA health check to report `"bf16": true`; `"status": "PASS"` alone
only establishes the sparse/dense operation check. Real-data configs enable
missing-data downloads. When compute nodes have no network access, populate the
complete raw cache beforehand.

For a custom experiment, keep the JSON outside the checkout and use `--config`
or `CM_CONFIG`. Both the submitted JSON bytes and the effective configuration
are captured. A requested Roman split is written into the effective config.
Dataset/profile conflicts fail before submission.

## 4. Inspect the suite and logs

Ordinary runs create unique suites under `$CM_BASE/outputs/`. The Slurm output
prints a `[SUITE]` line with the absolute path. Submission logs are under
`$CM_BASE/logs/CROSS_MORSE_TAG_JOBID.out` and `.err`, where `TAG` is the uppercase
config stem, for example `PRODUCTS`, `ROMAN_EMPIRE` or `PEPTIDES_FUNC`.

For an ordinary submission, source and configuration are copied when the job
starts. Keep that checkout and custom config unchanged while the job is queued.
The ten-split campaign instead creates its shared snapshot before submission.
In both cases the saved hashes identify the bytes actually executed.

```bash
CM_SUITE_PATH="/absolute/path/from-the-SUITE-log-line"
tail -f "$CM_SUITE_PATH/launcher/stdout.log"
```

Errors are also appended to `launcher/stderr.log`. `launcher/status_JOBID.json`
records the job's latest exit status. Source hashes, package versions, module
listings and GPU information accompany the run. The snapshot contains:

- `source/configs/submitted.json`: exact input JSON bytes.
- `source/configs/resolved.json`: validated effective configuration, including the split.
- `source/configs/active.env.example`: captured cluster settings and dataset identity.

`CROSS_MORSE_TAG_LIVE.out`, `.err` and `_LIVE_SUITE.txt` under `$CM_BASE/logs/`
identify the most recently started suite for that dataset. Monitor the exact
suite when jobs run concurrently. Profile variables sourced inside the launcher
do not automatically become interactive-shell variables. Scientific output
locations are listed in [Getting started](GETTING_STARTED.md#6-resume-and-inspect-outputs).

## 5. Resume an existing suite

Use its captured launcher and profile:

```bash
CM_SUITE_PATH="/absolute/path/to/existing-suite"
CM_ENV_FILE="$CM_SUITE_PATH/source/configs/active.env.example" \
CM_PROJECT_DIR="$CM_SUITE_PATH/source" \
bash "$CM_SUITE_PATH/source/run_cross_morse.sh" \
  submit resume "$CM_SUITE_PATH"
```

The saved dataset, source, config, split and original cache location are checked.
Other exported `CM_*` variables still override the captured profile, so review
intentional environment changes. A conflicting dataset or Roman split is
rejected. The suite is locked against concurrent writers; completed stages are
verified and reused. Keep the captured source/config unchanged.

Slurm is asked to send `USR1` 180 seconds before the time limit. The launcher
forwards it to Python, which checkpoints at a safe boundary. A long operation may
not reach that boundary before allocation expiry. Checkpointed interruption exits
with code 75. Requeue eligibility is enabled; actual requeue behavior depends on
the scheduler, and explicit resume remains available.

For staged execution, start a new run with `CM_STAGE=prepare`, then resume the
same suite with `CM_STAGE=all`. Stages are `prepare`, `train`, `explore`, `fit`,
`select`, `confirm`, `admit` and `evaluate`. An individual later stage requires its
predecessors. Clear an old single-stage override when continuing the whole chain.

## 6. Run all ten Roman-empire splits

A single-split example is:

```bash
bash run_cross_morse_roman_empire.sh --split-index 3 submit pilot
```

Without an override, the config's split applies. To execute the ten-split
protocol, choose a new absolute campaign directory outside the source tree and
leave `CM_SPLIT_INDEX` unset:

```bash
CM_CAMPAIGN_ROOT="/absolute/path/to/new-roman-campaign"
bash run_cross_morse_roman_empire.sh \
  submit campaign full "$CM_CAMPAIGN_ROOT"
```

The launcher freezes one common source/config/profile **before submission** and
submits `--array=0-9`. Each array task requests one GPU and its own CPU allocation;
tasks may run concurrently according to scheduler limits. Each task writes an
independent suite at `split_00` through `split_09`, with its index recorded in the
resolved configuration. Campaign logs use the array/task IDs. `pilot` and
explicitly synthetic `smoke` campaigns are available for checking the workflow.

Resume the same ten tasks from the common snapshot:

```bash
CM_ENV_FILE="$CM_CAMPAIGN_ROOT/source/configs/active.env.example" \
CM_PROJECT_DIR="$CM_CAMPAIGN_ROOT/source" \
bash "$CM_CAMPAIGN_ROOT/source/run_cross_morse.sh" --dataset Roman-empire \
  submit campaign resume "$CM_CAMPAIGN_ROOT"
```

Completed stages are verified and reused; unfinished split runs continue. An
individual split can also use the ordinary suite-resume command. Creating a new
campaign at an existing destination is refused.

After all ten suites finish, create a summary outside the campaign:

```bash
python -B scripts/summarize_roman_splits.py "$CM_CAMPAIGN_ROOT" \
  --output /absolute/path/to/new-roman-summary.json
```

The summary requires all ten sealed runs with matching source and non-split
settings. It reports descriptive variation across the official split masks and
does not change any per-split policy.

## 7. Direct Slurm submission

With `CM_ENV_FILE` exported, the submission wrapper is optional:

```bash
sbatch --account=YOUR_ALLOCATION --partition=YOUR_GPU_PARTITION \
  --cpus-per-task=16 --time=04:00:00 \
  run_cross_morse.sh --dataset PascalVOC-SP pilot
```

Replace the placeholders with site values and keep allocated CPUs at least equal
to `CM_THREADS`. From another working directory, export `CM_PROJECT_DIR` as the
repository's absolute path. Direct `sbatch` uses the script's 24-hour header
unless overridden; wrapper-specific defaults and `CM_TIME` do not apply. Its
Slurm logs default to the submission directory. Inside an existing GPU
allocation, invoke the mode directly, for example
`bash run_cross_morse_coco_sp.sh pilot`; `submit` rejects calls from inside a job.

The [validation record](VALIDATION.md) states what was actually tested. Launcher
mocks and synthetic checks do not establish real-dataset completion, compatible
site modules or a full-profile runtime. Check the environment and measure a
pilot on the allocated hardware.
