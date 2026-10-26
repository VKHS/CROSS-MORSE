# Running CROSS-MORSE on Arrhenius

[Repository home](../README.md) · [Generic launcher contract](HPC.md) · [Run archives](RUN_ARCHIVES.md)

This is how we run the seven datasets on the NAISS Arrhenius cluster (NVIDIA GH200 nodes,
aarch64). Everything site-specific is in one file, `configs/arrhenius-site.env.example`;
the launchers, configurations and code are the same on every machine.

## 1. The site profile

| Setting | Value in the profile | Why |
|---|---|---|
| `CM_EXPECT_ARCH` | `aarch64` | GH200 nodes are Arm; the launcher refuses an x86_64 interpreter |
| `CM_PARTITION`, job shape | `gpu`; 1 task, 1 GPU, 16 CPU threads (`#SBATCH` header of `run_cross_morse.sh`) | one run per GPU |
| `CM_TIME` | `24:00:00` for `full`; `04:00:00` pilot; `01:00:00` setup/smoke | resume continues sealed suites, so a run never needs more than one wall-time window to make progress |
| `CM_BASE` | `/nobackup/proj/disk/<storage project>/cross_morse` | suites, caches and logs on the project disk, never under `$HOME` |
| `CM_DATA_ROOT` | `$CM_BASE/datasets/ogb_cache` | one shared raw-data cache for all runs |
| `CM_VENV_DIR` | `$CM_BASE/code/venv_arrhenius_gh200` | persistent interpreter created once by `submit setup` |
| `CM_BUILD_MODULE`, `CM_PYTHON_MODULE` | empty | the virtual environment carries its own CUDA PyTorch wheel; no module is loaded |
| `CM_TORCH_SPEC`, `CM_TORCH_INDEX` | set at setup time to the CUDA build matching the node driver | recorded afterwards by the capture script |
| `CM_ACCOUNT`, `CM_STORAGE_PROJECT` | **placeholders** | the GPU allocation and storage project identifiers of our NAISS project; fill them from the project page |

```bash
cp configs/arrhenius-site.env.example configs/arrhenius.env.example   # git-ignored private copy
$EDITOR configs/arrhenius.env.example                                 # CM_ACCOUNT, CM_STORAGE_PROJECT
bash run_cross_morse_arxiv.sh dry-run full                            # prints account, paths, command; creates nothing
```

The launcher discovers `configs/arrhenius.env.example` automatically; `CM_ENV_FILE` selects any
other profile, and exported `CM_*` variables override the file.

## 2. One-time environment

```bash
bash run_cross_morse_arxiv.sh submit setup
```

`setup` runs inside a GPU job: it creates the virtual environment if missing, installs the
PyTorch wheel named by `CM_TORCH_SPEC`/`CM_TORCH_INDEX` only when `import torch` fails,
installs `requirements.txt`, runs `pip check` and `run_cross_morse.py --gpu-health-check`. It
never touches an existing working PyTorch. Afterwards capture the environment that the runs
will record:

```bash
srun --account="$CM_ACCOUNT" --partition=gpu --gpus=1 --cpus-per-task=4 --time=00:10:00 --pty \
  bash environment/arrhenius/capture_environment.sh "$CM_VENV_DIR/bin/python" environment/arrhenius
```

This writes `environment/arrhenius/observed_environment.json`, `requirements-gpu.lock`,
`nvidia-smi.txt` and `modules.txt`. Commit them; they are the GPU counterpart of the CPU lock in
`environment/`. Every run additionally seals its own `environment.json`.

## 3. Data

Prepare official data once per dataset, without training:

```bash
CM_STAGE=prepare bash run_cross_morse_proteins.sh submit pilot      # stops after preparation; no training
```

The adapters download OGB datasets through the `ogb` package, Roman-empire from its NPZ
release and the LRGB datasets from their provider, into `CM_DATA_ROOT`; `data/sources.json`
lists sources and split rules. Products keeps full-graph hidden states; measure its memory with
the pilot profile before the full one.

## 4. Full runs

```bash
for d in arxiv proteins products pascalvoc_sp coco_sp peptides_func; do
  bash "run_cross_morse_${d}.sh" submit full
done
bash run_cross_morse_roman_empire.sh submit campaign full "$CM_BASE/outputs/roman_campaign"   # --array=0-9
```

Each submission snapshots the source and resolved configuration into the suite directory
(`SUITE/source/`), records the effective profile, and runs the eight stages. A job that hits
the wall time is requeued (`--requeue`, `--signal=B:USR1@180`) and resumes from the last sealed
stage; `submit resume /abs/suite` does the same by hand. Logs are under `$CM_BASE/logs/`.

## 5. After a run

```bash
python -B scripts/artifact_inventory.py /abs/suite --output /abs/suite_inventory.json
make archive-run RUN_ROOT=/abs/suite OUTPUT="$CM_BASE/deposits/arxiv_full.tar.gz" PYTHON="$CM_VENV_DIR/bin/python"
make archive-campaign RUN_ROOT="$CM_BASE/outputs/roman_campaign" OUTPUT="$CM_BASE/deposits/roman_empire_full.tar.gz"
```

Transfer the archives, deposit them (DOI), then on a workstation `make ingest-run` and
`make paper`; see [RUN_ARCHIVES.md](RUN_ARCHIVES.md).

## 6. Rules we keep

* Never train on the login node; `python3` there only validates configurations.
* Never edit a suite by hand; stage manifests detect it and the archive will not verify.
* Keep allocation identifiers out of committed files; the release audit scans for them.
* Re-run `make fingerprint` on the commit you deposit from, and record the printed identity
  and the commit hash with the deposit.
