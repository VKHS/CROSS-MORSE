# Observed CPU environment

[Getting started](../docs/GETTING_STARTED.md) · [Artifacts](../docs/ARTIFACTS.md)

The bundled synthetic runs and their recorded validation use a clean CPython
3.12.14 environment on Linux x86_64 with **PyTorch 2.10.0+cpu**, NumPy 2.3.5,
SciPy 1.17.0 and Matplotlib 3.10.8. These are versions read from the installed
environment, rather than versions inferred from broad dependency requirements.
The full record is [observed_environment.json](observed_environment.json);
[requirements-cpu.lock](requirements-cpu.lock) contains all 53 installed
distributions, captured with `python -m pip freeze --all`.

## Create an environment

From the repository root, choose a Python 3.12 interpreter and a new path outside
the checkout. First inspect the exact commands:

```bash
bash environment/create_cpu_env.sh --python python3.12 \
  --venv "../cm_cpu_env" --dry-run
```

Create it and inspect the installed versions:

```bash
bash environment/create_cpu_env.sh --python python3.12 --venv "../cm_cpu_env"
source ../cm_cpu_env/bin/activate
python -B environment/check_environment.py
make reproduce PYTHON=python REPORT_ROOT="../cm_reports/bundled"
```

The installer rejects an existing environment or a destination inside the
source tree. It checks Python major/minor, operating system and architecture;
installs the recorded CPU Torch wheel from PyTorch's CPU index; applies the
remaining exact pins from PyPI; and runs `pip check` and the version checker.
It does not download datasets, train models, load checkpoints or submit jobs.
The `make env-cpu` target invokes the same installer with `PYTHON` and `VENV_DIR`.

The lock fixes package versions. It is not a wheel-hash lock and does not pin
native libraries, fonts or the operating-system image. The JSON records the
observed libc version; the checker reports whether Python's patch version also
matches. A different patch version is reported without failing the major/minor
compatibility check. Extra installed packages are not treated as mismatches,
but every locked package must have its recorded version. These checks do not
promise bitwise equality on a different host.

## Other environments

Python 3.11 remains supported by the project and covered by our
compatibility workflow. Use [requirements.txt](../requirements.txt) and a
host-compatible Torch build for that environment; the exact observed lock above
targets Python 3.12 on Linux x86_64. CairoSVG in the lock supports optional PNG
diagram generation and requires the host's Cairo shared library. The SVG-only
diagram target does not need CairoSVG.

For numeric replay without plotting or checkpoint execution, the smaller
[requirements-review.txt](../requirements-review.txt) is sufficient. Package
versions and commands recorded here describe the bundled CPU examples, not the
manuscript's GPU experiments or an Arrhenius CUDA environment. The existing
external `CM_ENV_FILE` interface and [HPC instructions](../docs/HPC.md) continue
to select the cluster's own modules, interpreter and virtual environment.

## The GPU environment of our Arrhenius runs

[arrhenius/capture_environment.sh](arrhenius/capture_environment.sh) records
the GPU execution environment on the cluster: it runs `pip freeze` with the run
interpreter, `nvidia-smi` and `module list`, and writes
`arrhenius/observed_environment.json`, `arrhenius/requirements-gpu.lock`,
`arrhenius/nvidia-smi.txt` and `arrhenius/modules.txt`. It is executed inside a
GPU allocation after `submit setup` ([docs/ARRHENIUS.md](../docs/ARRHENIUS.md)).
Those output files are not yet captured and therefore not yet in this directory;
every completed run nevertheless seals its own `environment.json` with the same
information for that run.
