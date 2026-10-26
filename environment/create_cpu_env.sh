#!/usr/bin/env bash
# Create a new external environment; never modify an existing environment.
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: bash environment/create_cpu_env.sh --venv PATH [--python EXECUTABLE] [--dry-run]

Create a fresh CPU environment from the package versions actually recorded in
environment/requirements-cpu.lock. Requires matching Python major/minor, OS and
architecture; see observed_environment.json. PATH must not exist and must be
outside the source tree. No dataset download, model training or cluster action
occurs. --dry-run prints commands and creates no files or environments.
HELP
}

bootstrap=python3
venv=
dry_run=false
while (($#)); do
  case "$1" in
    --python|--venv)
      (($# >= 2)) || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }
      if [[ "$1" == --python ]]; then bootstrap=$2; else venv=$2; fi
      shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "$venv" ]] || { usage >&2; exit 2; }
spec_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source_root=$(cd -- "$spec_dir/.." && pwd -P)
torch_requirement=$("$bootstrap" -B "$spec_dir/check_environment.py" --bootstrap)
target=$("$bootstrap" -B - "$source_root" "$venv" <<'PY'
from pathlib import Path
import sys
source = Path(sys.argv[1]).resolve()
raw = Path(sys.argv[2]).expanduser()
target = raw.resolve()
if raw.is_symlink() or target.exists():
    raise SystemExit('Choose a new environment path; existing paths are preserved.')
if target == source or target.is_relative_to(source) or source.is_relative_to(target):
    raise SystemExit('The environment must be outside the source tree and cannot be its ancestor.')
print(target)
PY
)

run() {
  if "$dry_run"; then
    printf 'DRY-RUN'; printf ' %q' "$@"; printf '\n'
  else
    "$@"
  fi
}
run "$bootstrap" -m venv "$target"
run "$target/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu "$torch_requirement"
run "$target/bin/python" -m pip install --index-url https://pypi.org/simple -r "$spec_dir/requirements-cpu.lock"
run "$target/bin/python" -m pip check
run "$target/bin/python" -B "$spec_dir/check_environment.py"
