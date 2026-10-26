#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --job-name=cross_morse_pascalvoc_sp
#SBATCH --output=cross_morse_pascalvoc_sp_%j.out
#SBATCH --error=cross_morse_pascalvoc_sp_%j.err

# Dataset-specific facade for the shared launcher; no duplicate execution logic.
set -Eeuo pipefail
wrapper_directory="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
launcher_directory=""
if [[ -n "${CM_PROJECT_DIR:-}" ]]; then
    [[ -f "${CM_PROJECT_DIR}/run_cross_morse.sh" ]] || { printf '[ERROR] CM_PROJECT_DIR lacks run_cross_morse.sh.\n' >&2; exit 2; }
    launcher_directory="${CM_PROJECT_DIR}"
else
    for candidate in "${wrapper_directory}" "${SLURM_SUBMIT_DIR:-/nonexistent}"; do
        if [[ -f "${candidate}/run_cross_morse.sh" ]]; then launcher_directory="${candidate}"; break; fi
    done
fi
[[ -n "${launcher_directory}" ]] || { printf '[ERROR] Set CM_PROJECT_DIR to the repository when submitting from another directory.\n' >&2; exit 2; }
exec bash "${launcher_directory}/run_cross_morse.sh" --dataset 'PascalVOC-SP' "$@"
