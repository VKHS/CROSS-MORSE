#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/CROSS-MORSE [additional apply_repository_upgrade.py options]" >&2
  exit 2
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$1"
shift
exec python3 "$SCRIPT_DIR/apply_repository_upgrade.py" "$SOURCE" --make-zip "$@"
