#!/bin/bash
# Capture the GPU execution environment of a CROSS-MORSE run on the Arrhenius cluster.
#
# Run this inside a GPU allocation with the run's interpreter, e.g.
#   srun --account=$CM_ACCOUNT --partition=gpu --gpus=1 --time=00:10:00 --pty \
#       bash environment/arrhenius/capture_environment.sh "$CM_VENV_DIR/bin/python" environment/arrhenius
# It writes observed_environment.json, requirements-gpu.lock, nvidia-smi.txt and
# modules.txt into the output directory. Every completed run also records its own
# environment.json; this script captures the same information outside a run, for the
# lock file that the manuscript's environment specification refers to.
set -euo pipefail
PYTHON="${1:?interpreter path}"
OUT="${2:?output directory}"
mkdir -p "${OUT}"
"${PYTHON}" -m pip freeze --all > "${OUT}/requirements-gpu.lock"
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi > "${OUT}/nvidia-smi.txt"; else echo "nvidia-smi unavailable" > "${OUT}/nvidia-smi.txt"; fi
if command -v module >/dev/null 2>&1; then module list 2> "${OUT}/modules.txt" || true; else echo "module command unavailable" > "${OUT}/modules.txt"; fi
"${PYTHON}" - "${OUT}" <<'PY'
import json, platform, subprocess, sys, datetime
out = sys.argv[1]
record = {"captured_utc": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat(),
          "hostname": platform.node(), "platform": platform.platform(), "machine": platform.machine(),
          "python": platform.python_version(), "python_implementation": platform.python_implementation(),
          "packages": {}}
for line in open(f"{out}/requirements-gpu.lock", encoding="utf-8"):
    if "==" in line:
        name, version = line.strip().split("==", 1)
        record["packages"][name] = version
try:
    import torch
    record.update({"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                   "cudnn": torch.backends.cudnn.version(), "cuda_available": torch.cuda.is_available(),
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   "gpu_count": torch.cuda.device_count()})
except Exception as exc:  # noqa: BLE001
    record["torch_error"] = repr(exc)
try:
    import numpy, scipy
    record.update({"numpy": numpy.__version__, "scipy": scipy.__version__})
except Exception as exc:  # noqa: BLE001
    record["numpy_error"] = repr(exc)
try:
    record["slurm"] = {k: v for k, v in __import__("os").environ.items() if k.startswith("SLURM_") and k in
                       ("SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_CPUS_PER_TASK", "SLURM_GPUS", "SLURM_JOB_NODELIST")}
except Exception:  # noqa: BLE001
    pass
json.dump(record, open(f"{out}/observed_environment.json", "w", encoding="utf-8"), indent=1, sort_keys=True)
print(json.dumps({k: record.get(k) for k in ("hostname", "machine", "python", "torch", "cuda_runtime", "gpu")}, indent=1))
PY
