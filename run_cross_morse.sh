#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --job-name=cross_morse
#SBATCH --output=cross_morse_%j.out
#SBATCH --error=cross_morse_%j.err

# Shared GPU/Slurm entry point. Dataset wrappers call this same implementation.
# CM_ENV_FILE selects a complete trusted Bash profile; exported CM_* values win.
set -Eeuo pipefail
umask 002

die() { printf '[ERROR] %s\n' "$*" >&2; exit 2; }
RUNNER="run_cross_morse.py"
SHELL_NAME="run_cross_morse.sh"
RUNTIME_SHELL="$(readlink -f "${BASH_SOURCE[0]}")"
PROFILE_SOURCE=""
CLI_DATASET="" CLI_SPLIT="" CLI_DATA_ROOT="" CLI_CONFIG=""
POSITIONAL=()
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --dataset|--split-index|--data-root|--config)
            [[ "$#" -ge 2 && -n "$2" ]] || die "$1 needs a value."
            case "$1" in
                --dataset)
                    [[ -z "${CLI_DATASET}" || "${CLI_DATASET}" == "$2" ]] || die "Conflicting --dataset arguments."
                    CLI_DATASET="$2" ;;
                --split-index) CLI_SPLIT="$2" ;;
                --data-root) CLI_DATA_ROOT="$2" ;;
                --config) CLI_CONFIG="$2" ;;
            esac
            shift 2 ;;
        --dataset=*)
            value="${1#*=}"
            [[ -n "${value}" && ( -z "${CLI_DATASET}" || "${CLI_DATASET}" == "${value}" ) ]] || die "Empty/conflicting --dataset."
            CLI_DATASET="${value}"; shift ;;
        --split-index=*) CLI_SPLIT="${1#*=}"; [[ -n "${CLI_SPLIT}" ]] || die "Empty --split-index."; shift ;;
        --data-root=*) CLI_DATA_ROOT="${1#*=}"; [[ -n "${CLI_DATA_ROOT}" ]] || die "Empty --data-root."; shift ;;
        --config=*) CLI_CONFIG="${1#*=}"; [[ -n "${CLI_CONFIG}" ]] || die "Empty --config."; shift ;;
        -h|--help) POSITIONAL=(--help); break ;;
        --*) die "Unknown launcher option: $1" ;;
        *) POSITIONAL+=("$1"); shift ;;
    esac
done
set -- "${POSITIONAL[@]}"

# Save caller exports, source exactly one profile, then restore caller choices.
declare -A cm_prior_env=()
while IFS= read -r cm_name; do
    [[ "${cm_name}" == CM_* ]] || continue
    cm_prior_env["${cm_name}"]="${!cm_name}"
done < <(compgen -e)
if [[ -n "${CM_ENV_FILE:-}" ]]; then
    PROFILE_SOURCE="$(readlink -f "${CM_ENV_FILE}")" || die "Cannot resolve CM_ENV_FILE: ${CM_ENV_FILE}"
    [[ -f "${PROFILE_SOURCE}" ]] || die "CM_ENV_FILE does not exist: ${CM_ENV_FILE}"
else
    for cm_candidate in "${CM_PROJECT_DIR:-/nonexistent}" "$(dirname "${RUNTIME_SHELL}")" "${SLURM_SUBMIT_DIR:-/nonexistent}"; do
        for cm_profile in active.env.example arrhenius.env.example public.env.example; do
            if [[ -f "${cm_candidate}/configs/${cm_profile}" ]]; then
                PROFILE_SOURCE="$(readlink -f "${cm_candidate}/configs/${cm_profile}")"
                break 2
            fi
        done
    done
fi
[[ -n "${PROFILE_SOURCE}" ]] || die "No environment profile found; set CM_ENV_FILE or keep configs/ with the launcher."
# shellcheck source=/dev/null
source "${PROFILE_SOURCE}"
for cm_name in "${!cm_prior_env[@]}"; do
    printf -v "${cm_name}" '%s' "${cm_prior_env[${cm_name}]}"
    export "${cm_name}"
done
unset cm_prior_env cm_name cm_candidate cm_profile
[[ -z "${CLI_DATASET}" ]] || CM_DATASET="${CLI_DATASET}"
[[ -z "${CLI_SPLIT}" ]] || CM_SPLIT_INDEX="${CLI_SPLIT}"
[[ -z "${CLI_DATA_ROOT}" ]] || CM_DATA_ROOT="${CLI_DATA_ROOT}"
[[ -z "${CLI_CONFIG}" ]] || CM_CONFIG="${CLI_CONFIG}"

BASE="$(readlink -m "${CM_BASE:-${HOME}/cross_morse_work}")"
CODE_DIR="${BASE}/code"
# Keep the existing default cache address; all dataset readers accept this root.
DATA_ROOT="$(readlink -m "${CM_DATA_ROOT:-${BASE}/datasets/ogb_cache}")"
CACHE_DIR="${BASE}/cache"
LOGS_DIR="${BASE}/logs"
OUTPUTS_DIR="${BASE}/outputs"
VENV_DIR="$(readlink -m "${CM_VENV_DIR:-${CODE_DIR}/${CM_VENV_NAME:-venv_cross_morse}}")"
PYTHON="${VENV_DIR}/bin/python"
BUILD_MODULE="${CM_BUILD_MODULE:-}"
PYTHON_MODULE="${CM_PYTHON_MODULE:-}"
EXPECTED_ARCH="${CM_EXPECT_ARCH:-}"
THREADS="${CM_THREADS:-16}"
STAGE="${CM_STAGE:-all}"
DATASET="${CM_DATASET:-ogbn-arxiv}"
SPLIT_INDEX="${CM_SPLIT_INDEX:-}"
SPLIT_EXPLICIT=0
[[ -z "${SPLIT_INDEX}" ]] || SPLIT_EXPLICIT=1
SUITE="" CHILD_PID=""
SIGNAL_RECEIVED=0 WAIT_INTERRUPTED=0

usage() {
    cat <<'USAGE'
CROSS-MORSE / seven datasets / one GPU Slurm task

  bash run_cross_morse.sh --dataset ogbn-arxiv dry-run pilot
  bash run_cross_morse.sh --dataset ogbn-products submit setup
  bash run_cross_morse.sh --dataset ogbn-proteins submit smoke
  bash run_cross_morse.sh --dataset PascalVOC-SP submit pilot
  bash run_cross_morse.sh --dataset COCO-SP submit full
  bash run_cross_morse.sh --dataset Peptides-func submit resume /absolute/suite
  bash run_cross_morse.sh --dataset Roman-empire --split-index 3 submit full
  bash run_cross_morse_roman_empire.sh submit campaign full /absolute/campaign
  bash run_cross_morse_roman_empire.sh submit campaign resume /absolute/campaign

Dataset-specific .sh wrappers accept the same modes. Canonical names:
ogbn-arxiv, ogbn-products, ogbn-proteins, Roman-empire, PascalVOC-SP, COCO-SP,
Peptides-func. The generic default is ogbn-arxiv. smoke is explicitly synthetic;
pilot/full load real benchmark data. No walltime is a completion estimate.

Options may precede or follow the mode:
  --dataset NAME     Overrides CM_DATASET; must match the configuration/wrapper
  --split-index 0..9 Overrides CM_SPLIT_INDEX; Roman-empire only
  --data-root PATH   Overrides CM_DATA_ROOT
  --config PATH      Overrides CM_CONFIG for a new run

Precedence: command-line routing > exported CM_* > selected profile > defaults.
CM_ENV_FILE is one complete trusted Bash profile. Without it, inspect
CM_PROJECT_DIR, the invoked shell's directory, then SLURM_SUBMIT_DIR; in each,
try configs/active.env.example, arrhenius.env.example, public.env.example.

Environment settings:
  CM_ENV_FILE, CM_PROJECT_DIR, CM_BASE, CM_DATA_ROOT, CM_VENV_DIR
  CM_ACCOUNT (required for submit), CM_PARTITION (default gpu)
  CM_THREADS (default 16), CM_STAGE (default all), CM_TIME (submit walltime)
  CM_DATASET, CM_SPLIT_INDEX, CM_CONFIG
  CM_BUILD_MODULE, CM_PYTHON_MODULE, CM_EXPECT_ARCH (empty accepts host)
  CM_TORCH_SPEC, CM_TORCH_INDEX (explicit optional setup-only installation)

setup creates the selected venv only if missing, preserves importable PyTorch,
and explicitly installs requirements. Other modes never install packages.
The selected interpreter is always CM_VENV_DIR/bin/python, not an implicitly
activated environment. Empty module settings keep the allocated environment.

submit defaults: setup/smoke 01:00:00, pilot 04:00:00, full/resume 24:00:00.
For direct sbatch, supply --account, --partition and desired --time yourself:
  sbatch --account=YOUR_ALLOCATION --partition=YOUR_PARTITION run_cross_morse.sh --dataset ogbn-arxiv pilot
Inside an existing GPU allocation, omit submit. CLI config validation uses
python3 on the login node; it imports no numerical runtime and downloads nothing.

Roman campaign submits --array=0-9 from one frozen common source/config.
Runs live at CAMPAIGN/split_00 through split_09. campaign resume resubmits the
same ten tasks; completed stages are verified and reused. Summarize with:
  python scripts/summarize_roman_splits.py CAMPAIGN --output NEW_REPORT.json
USAGE
}

mode="${1:-pilot}"
case "${mode}" in help|-h|--help) usage; exit 0 ;; esac
case "${DATASET}" in
    ogbn-arxiv) DATASET_STEM=arxiv ;;
    ogbn-products) DATASET_STEM=products ;;
    ogbn-proteins) DATASET_STEM=proteins ;;
    Roman-empire) DATASET_STEM=roman_empire ;;
    PascalVOC-SP) DATASET_STEM=pascalvoc_sp ;;
    COCO-SP) DATASET_STEM=coco_sp ;;
    Peptides-func) DATASET_STEM=peptides_func ;;
    *) die "Unknown canonical dataset: ${DATASET}" ;;
esac
DATASET_LOG="${DATASET_STEM^^}"
if [[ "${DATASET}" == Roman-empire ]]; then
    [[ -z "${SPLIT_INDEX}" || "${SPLIT_INDEX}" =~ ^[0-9]$ ]] || die "Roman-empire split index must be 0 through 9."
else
    [[ -z "${SPLIT_INDEX}" ]] || die "Split indices are supported only for Roman-empire."
fi
[[ "${THREADS}" =~ ^[1-9][0-9]*$ ]] || die "CM_THREADS must be a positive integer."
case "${STAGE}" in prepare|train|explore|fit|select|confirm|admit|evaluate|all) ;; *) die "Invalid CM_STAGE: ${STAGE}" ;; esac

resolve_project() {
    local candidate own_dir
    own_dir="$(dirname "${RUNTIME_SHELL}")"
    if [[ -n "${CM_PROJECT_DIR:-}" ]]; then
        [[ -f "${CM_PROJECT_DIR}/${RUNNER}" && -f "${CM_PROJECT_DIR}/${SHELL_NAME}" ]] || die "CM_PROJECT_DIR lacks the shared Python/shell entry points."
        PROJECT_DIR="$(readlink -f "${CM_PROJECT_DIR}")"
        return
    fi
    for candidate in "${own_dir}" "${SLURM_SUBMIT_DIR:-/nonexistent}" "${CODE_DIR}/CROSS-MORSE" "${CODE_DIR}/cross_morse_arxiv" "${CODE_DIR}"; do
        if [[ -f "${candidate}/${RUNNER}" && -f "${candidate}/${SHELL_NAME}" ]]; then
            PROJECT_DIR="$(readlink -f "${candidate}")"
            return
        fi
    done
    die "Cannot find source; set CM_PROJECT_DIR to the unpacked repository."
}
resolve_project

select_config() {
    PROFILE="$1"
    case "${PROFILE}" in smoke|pilot|full) ;; *) die "Profile must be smoke, pilot or full." ;; esac
    CONFIG_SOURCE="${CM_CONFIG:-${PROJECT_DIR}/configs/${DATASET_STEM}_${PROFILE}.json}"
    [[ -f "${CONFIG_SOURCE}" ]] || die "Config does not exist: ${CONFIG_SOURCE}"
    CONFIG_SOURCE="$(readlink -f "${CONFIG_SOURCE}")"
}
check_config() {
    SPLIT_INDEX="$(python3 -B - "${PROJECT_DIR}" "${CONFIG_SOURCE}" "${DATASET}" "${PROFILE}" "${SPLIT_INDEX}" <<'PYCONFIGCHECK'
import sys
sys.path.insert(0, sys.argv[1])
from run_cross_morse import resolve_configuration
config = resolve_configuration(config_path=sys.argv[2], dataset=sys.argv[3], profile=sys.argv[4],
                               split_index=int(sys.argv[5]) if sys.argv[5] else None)
print(config['data'].get('split_index', ''))
PYCONFIGCHECK
)"
}
export_effective_environment() {
    export CM_BASE="${BASE}" CM_DATA_ROOT="${DATA_ROOT}" CM_VENV_DIR="${VENV_DIR}"
    export CM_ACCOUNT="${SLURM_JOB_ACCOUNT:-${CM_ACCOUNT:-}}" CM_STORAGE_PROJECT="${CM_STORAGE_PROJECT:-}"
    export CM_PARTITION="${CM_PARTITION:-gpu}" CM_THREADS="${THREADS}"
    export CM_BUILD_MODULE="${BUILD_MODULE}" CM_PYTHON_MODULE="${PYTHON_MODULE}" CM_EXPECT_ARCH="${EXPECTED_ARCH}"
    export CM_TORCH_SPEC="${CM_TORCH_SPEC:-}" CM_TORCH_INDEX="${CM_TORCH_INDEX:-}"
    export CM_DATASET="${DATASET}" CM_SPLIT_INDEX="${SPLIT_INDEX}"
}
read_suite_context() {
    local result
    result="$(python3 -B - "${PROJECT_DIR}" "$1" "${DATASET}" "${SPLIT_INDEX}" "${SPLIT_EXPLICIT}" <<'PYCONTEXT'
import sys
sys.path.insert(0, sys.argv[1])
from run_cross_morse import verify_launcher_snapshot
context = verify_launcher_snapshot(sys.argv[2])
if context['dataset'] != sys.argv[3]:
    raise SystemExit('Dataset does not match the saved suite; use its dataset wrapper/profile.')
if sys.argv[5] == '1' and str(context['split_index']) != sys.argv[4]:
    raise SystemExit('Requested split does not match the saved suite; a resume cannot change splits.')
print(context['data_root'])
print(context['profile'])
print('-' if context['split_index'] is None else context['split_index'])
PYCONTEXT
)"
    mapfile -t saved_context <<< "${result}"
    DATA_ROOT="${saved_context[0]}"
    PROFILE="${saved_context[1]}"
    SPLIT_INDEX="${saved_context[2]}"
    [[ "${SPLIT_INDEX}" != - ]] || SPLIT_INDEX=""
}
read_campaign_profile() {
    python3 -B - "${PROJECT_DIR}" "$1" <<'PYCAMPAIGNCHECK'
import sys
sys.path.insert(0, sys.argv[1])
from run_cross_morse import verify_roman_campaign
print(verify_roman_campaign(sys.argv[2])['profile'])
PYCAMPAIGNCHECK
}

if [[ "${mode}" == dry-run ]]; then
    [[ "$#" -le 2 ]] || die "dry-run takes one profile."
    select_config "${2:-pilot}"
    check_config
    printf 'account=%s\npartition=%s\nbase=%s\nproject=%s\ndata_root=%s\npython=%s\nenv_file=%s\n' \
        "${CM_ACCOUNT:-UNSET}" "${CM_PARTITION:-gpu}" "${BASE}" "${PROJECT_DIR}" "${DATA_ROOT}" "${PYTHON}" "${PROFILE_SOURCE}"
    printf 'dataset=%s\nconfig=%s\nsplit_index=%s\nmodules=%s %s\nprofile=%s\nstage=%s\n' \
        "${DATASET}" "${CONFIG_SOURCE}" "${SPLIT_INDEX:--}" "${BUILD_MODULE}" "${PYTHON_MODULE}" "${PROFILE}" "${STAGE}"
    cmd=("${PYTHON}" -u "SUITE/source/${RUNNER}" --dataset "${DATASET}" --config SUITE/source/configs/resolved.json
         --data-root "${DATA_ROOT}" --output-dir SUITE --device cuda --threads "${THREADS}" --stage "${STAGE}")
    [[ -z "${SPLIT_INDEX}" ]] || cmd+=(--split-index "${SPLIT_INDEX}")
    printf 'Python command after source snapshot:'; printf ' %q' "${cmd[@]}"; printf '\n'
    printf 'No directories, packages or jobs were created.\n'
    exit 0
fi

if [[ "${mode}" == submit ]]; then
    shift
    [[ "$#" -gt 0 ]] || set -- pilot
    [[ -z "${SLURM_JOB_ID:-}" ]] || die "submit must run outside a job; invoke the mode directly inside an allocation."
    command -v sbatch >/dev/null 2>&1 || die "sbatch is not available in this shell."
    [[ -n "${CM_ACCOUNT:-}" ]] || die "Set CM_ACCOUNT in the environment or CM_ENV_FILE before submitting."
    submit_mode="$1"
    batch_extra=()
    case "${submit_mode}" in
        setup)
            [[ "$#" == 1 ]] || die "setup takes no positional arguments."
            default_time=01:00:00; run_args=(setup) ;;
        smoke|pilot|full)
            [[ "$#" == 1 ]] || die "${submit_mode} takes no positional arguments."
            select_config "${submit_mode}"; check_config
            case "${submit_mode}" in smoke) default_time=01:00:00 ;; pilot) default_time=04:00:00 ;; full) default_time=24:00:00 ;; esac
            run_args=(--config "${CONFIG_SOURCE}" "${submit_mode}") ;;
        resume)
            [[ "$#" == 2 && "$2" == /* ]] || die "resume requires one absolute suite path."
            read_suite_context "$2"
            default_time=24:00:00; run_args=(resume "$2") ;;
        campaign)
            [[ "${DATASET}" == Roman-empire ]] || die "campaign is available only for Roman-empire."
            [[ "$#" == 3 && "$3" == /* ]] || die "Use submit campaign smoke|pilot|full|resume /absolute/campaign."
            [[ "${SPLIT_EXPLICIT}" == 0 ]] || die "Campaigns select all ten splits; omit CM_SPLIT_INDEX/--split-index."
            CAMPAIGN_ROOT="$3"
            if [[ "$2" == resume ]]; then
                PROFILE="$(read_campaign_profile "${CAMPAIGN_ROOT}")"
            else
                select_config "$2"; check_config
                (
                    export_effective_environment
                    python3 -B - "${PROJECT_DIR}" "${CAMPAIGN_ROOT}" "${CONFIG_SOURCE}" "${PROFILE}" "${DATA_ROOT}" "${RUNTIME_SHELL}" <<'PYCAMPAIGNCREATE'
import sys
sys.path.insert(0, sys.argv[1])
from run_cross_morse import create_roman_campaign
print('[CAMPAIGN]', create_roman_campaign(*sys.argv[1:]))
PYCAMPAIGNCREATE
                )
            fi
            read_suite_context "${CAMPAIGN_ROOT}"
            PROJECT_DIR="${CAMPAIGN_ROOT}/source"
            PROFILE_SOURCE="${PROJECT_DIR}/configs/active.env.example"
            case "${PROFILE}" in smoke) default_time=01:00:00 ;; pilot) default_time=04:00:00 ;; full) default_time=24:00:00 ;; esac
            batch_extra=(--array=0-9)
            run_args=(campaign "${PROFILE}" "${CAMPAIGN_ROOT}") ;;
        *) die "Unknown submit mode: ${submit_mode}" ;;
    esac
    mkdir -p "${LOGS_DIR}" "${OUTPUTS_DIR}" "${CACHE_DIR}/cross_morse"
    export CM_PROJECT_DIR="${PROJECT_DIR}" CM_ENV_FILE="${PROFILE_SOURCE}"
    routing=(--dataset "${DATASET}" --data-root "${DATA_ROOT}")
    if [[ "${submit_mode}" != campaign && -n "${SPLIT_INDEX}" ]]; then routing+=(--split-index "${SPLIT_INDEX}"); fi
    log_pattern="CROSS_MORSE_${DATASET_LOG}_%j"
    [[ "${submit_mode}" != campaign ]] || log_pattern="CROSS_MORSE_${DATASET_LOG}_%A_%a"
    exec sbatch --account="${CM_ACCOUNT}" --partition="${CM_PARTITION:-gpu}" \
        --time="${CM_TIME:-${default_time}}" --cpus-per-task="${THREADS}" "${batch_extra[@]}" \
        --output="${LOGS_DIR}/${log_pattern}.out" --error="${LOGS_DIR}/${log_pattern}.err" \
        "${PROJECT_DIR}/${SHELL_NAME}" "${routing[@]}" "${run_args[@]}"
fi

case "${mode}" in setup|smoke|pilot|full) [[ "$#" -le 1 ]] || die "${mode} takes no positional arguments." ;;
    resume) [[ "$#" == 2 && "$2" == /* ]] || die "resume requires an absolute suite path." ;;
    campaign) [[ "$#" == 3 && "$3" == /* ]] || die "campaign needs profile and absolute campaign path." ;;
    *) die "Unknown mode: ${mode}. Use --help." ;;
esac
[[ -n "${SLURM_JOB_ID:-}" ]] || die "Use submit, sbatch, or run inside an allocated GPU job."
if [[ -n "${SLURM_CPUS_PER_TASK:-}" && "${THREADS}" -gt "${SLURM_CPUS_PER_TASK}" ]]; then
    die "CM_THREADS exceeds allocated --cpus-per-task (${SLURM_CPUS_PER_TASK})."
fi
if [[ -n "${EXPECTED_ARCH}" && "$(uname -m)" != "${EXPECTED_ARCH}" ]]; then die "This profile requires ${EXPECTED_ARCH}; current host is $(uname -m)."; fi
if [[ -n "${BUILD_MODULE}" || -n "${PYTHON_MODULE}" ]]; then
    command -v module >/dev/null 2>&1 || die "Module command unavailable; use a login shell or empty module settings."
    module --force purge
    [[ -z "${BUILD_MODULE}" ]] || module load "${BUILD_MODULE}"
    [[ -z "${PYTHON_MODULE}" ]] || module load "${PYTHON_MODULE}"
fi
mkdir -p "${LOGS_DIR}" "${OUTPUTS_DIR}" "${CACHE_DIR}/cross_morse" "${DATA_ROOT}"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export_effective_environment
unset PYTHONPATH || true
export OMP_NUM_THREADS="${THREADS}" OPENBLAS_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" NUMEXPR_NUM_THREADS="${THREADS}"
export TORCH_HOME="${CACHE_DIR}/torch" XDG_CACHE_HOME="${CACHE_DIR}/xdg" PIP_CACHE_DIR="${CACHE_DIR}/pip" OGB_HOME="${DATA_ROOT}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
mkdir -p "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${PIP_CACHE_DIR}"

if [[ "${mode}" == setup ]]; then
    exec 9>"${CACHE_DIR}/cross_morse/venv_setup.lock"; flock -x 9
    if [[ ! -x "${PYTHON}" ]]; then
        [[ ! -e "${VENV_DIR}" ]] || die "Incomplete venv exists; repair it explicitly or select another CM_VENV_DIR."
        mkdir -p "$(dirname "${VENV_DIR}")"
        python3 -m venv "${VENV_DIR}"
        "${PYTHON}" -m pip install --upgrade pip setuptools wheel
    fi
    if ! "${PYTHON}" -c 'import torch' >/dev/null 2>&1; then
        [[ -n "${CM_TORCH_SPEC:-}" ]] || die "PyTorch missing; select a host-compatible CM_TORCH_SPEC and optional CM_TORCH_INDEX."
        torch_install=("${PYTHON}" -m pip install "${CM_TORCH_SPEC}")
        [[ -z "${CM_TORCH_INDEX:-}" ]] || torch_install+=(--index-url "${CM_TORCH_INDEX}")
        "${torch_install[@]}"
    fi
    "${PYTHON}" -m pip install -r "${PROJECT_DIR}/requirements.txt"
    "${PYTHON}" -m pip check
    "${PYTHON}" "${PROJECT_DIR}/${RUNNER}" --gpu-health-check
    printf '[SETUP COMPLETE] %s\n' "${VENV_DIR}"
    exit 0
fi
[[ -x "${PYTHON}" ]] || die "Missing ${PYTHON}; run explicit setup or select an existing environment."
"${PYTHON}" - <<'PYENV'
import os, platform
import numpy, scipy, torch
expected = os.environ.get('CM_EXPECT_ARCH', '')
if expected and platform.machine() != expected:
    raise SystemExit('Wrong Python interpreter architecture.')
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable; check the GPU allocation, modules and PyTorch build.')
print('[ENV]', platform.machine(), 'torch', torch.__version__, 'CUDA', torch.version.cuda,
      'GPU', torch.cuda.get_device_name(0), 'NumPy', numpy.__version__, 'SciPy', scipy.__version__)
PYENV

JOB_KEY="${SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID:+_${SLURM_ARRAY_TASK_ID}}"
JOB_MAP="${CACHE_DIR}/cross_morse/job_${JOB_KEY}.suite"
RESUME=0
if [[ "${mode}" == campaign ]]; then
    [[ "${DATASET}" == Roman-empire ]] || die "campaign is Roman-empire only."
    [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-9]$ ]] || die "Campaign requires a Slurm array task from 0 through 9."
    CAMPAIGN_ROOT="$3"
    PROFILE="$(read_campaign_profile "${CAMPAIGN_ROOT}")"
    [[ "$2" == "${PROFILE}" ]] || die "Campaign profile disagrees with its frozen record."
    read_suite_context "${CAMPAIGN_ROOT}"
    PROJECT_DIR="${CAMPAIGN_ROOT}/source"
    CONFIG_SOURCE="${PROJECT_DIR}/configs/resolved.json"
    SPLIT_INDEX="${SLURM_ARRAY_TASK_ID}"
    SPLIT_EXPLICIT=1
    printf -v split_dir 'split_%02d' "${SPLIT_INDEX}"
    SUITE="${CAMPAIGN_ROOT}/${split_dir}"
    if [[ -f "${SUITE}/launcher/run_manifest.json" ]]; then
        RESUME=1
    elif [[ -d "${SUITE}/source" ]]; then
        die "Incomplete campaign split snapshot; inspect ${SUITE} before continuing."
    else
        mkdir -p "${SUITE}"
    fi
elif [[ "${mode}" == resume ]]; then
    SUITE="$(readlink -f "$2")"; RESUME=1
elif [[ "${SLURM_RESTART_COUNT:-0}" -gt 0 && -s "${JOB_MAP}" ]]; then
    SUITE="$(cat "${JOB_MAP}")"; RESUME=1
else
    select_config "${mode}"
    SUITE="$(mktemp -d "${OUTPUTS_DIR}/CROSS_MORSE_${DATASET_STEM}_${PROFILE}_$(date -u +%Y%m%dT%H%M%SZ)_J${JOB_KEY}_XXXXXX")"
fi
[[ -d "${SUITE}" ]] || die "Suite directory does not exist: ${SUITE}"
mkdir -p "${SUITE}/launcher"
exec 8>"${SUITE}/launcher/run.lock"; flock -n 8 || die "Another process is already writing to ${SUITE}."
printf '%s\n' "${SUITE}" > "${JOB_MAP}"
exec > >(tee -a "${SUITE}/launcher/stdout.log") 2> >(tee -a "${SUITE}/launcher/stderr.log" >&2)
ln -sfn "${SUITE}/launcher/stdout.log" "${LOGS_DIR}/CROSS_MORSE_${DATASET_LOG}_LIVE.out"
ln -sfn "${SUITE}/launcher/stderr.log" "${LOGS_DIR}/CROSS_MORSE_${DATASET_LOG}_LIVE.err"
printf '%s\n' "${SUITE}" > "${LOGS_DIR}/CROSS_MORSE_${DATASET_LOG}_LIVE_SUITE.txt"
printf '[SUITE] %s\n[DATASET] %s\n[STAGE] %s\n' "${SUITE}" "${DATASET}" "${STAGE}"

record_exit() {
    local rc=$?
    trap - EXIT
    "${PYTHON}" - "${SUITE}" "${rc}" "${JOB_KEY}" "${SIGNAL_RECEIVED}" "${STAGE}" <<'PYSTATUS' || true
import datetime, json, pathlib, sys
root, code, job, signal, stage = sys.argv[1:]
data = {'job_id': job, 'stage': stage, 'exit_code': int(code),
        'execution_status': 'COMPLETED' if code == '0' else ('INTERRUPTED' if signal == '1' else 'FAILED'),
        'checkpoint_signal_received': signal == '1',
        'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'note': 'Process completion alone does not establish statistical qualification.'}
path = pathlib.Path(root) / 'launcher' / ('status_' + job + '.json')
temporary = path.with_suffix('.json.tmp')
temporary.write_text(json.dumps(data, indent=2) + '\n')
temporary.replace(path)
PYSTATUS
    if [[ "${rc}" != 0 ]]; then
        printf '[RESUME] CM_ENV_FILE=%q CM_PROJECT_DIR=%q bash %q submit resume %q\n' \
            "${SUITE}/source/configs/active.env.example" "${SUITE}/source" "${SUITE}/source/${SHELL_NAME}" "${SUITE}" >&2
    fi
    exit "${rc}"
}
trap record_exit EXIT
cp "${RUNTIME_SHELL}" "${SUITE}/launcher/runtime_shell_${JOB_KEY}.sh"
if [[ "${RESUME}" == 0 ]]; then
    export_effective_environment
    "${PYTHON}" - "${PROJECT_DIR}" "${SUITE}" "${CONFIG_SOURCE}" "${PROFILE}" "${DATA_ROOT}" "${RUNTIME_SHELL}" "${DATASET}" "${SPLIT_INDEX}" <<'PYSNAPSHOT'
import sys
sys.path.insert(0, sys.argv[1])
from run_cross_morse import create_launcher_snapshot
source, suite, config, profile, data_root, runtime_shell, dataset, split = sys.argv[1:]
create_launcher_snapshot(source, suite, config, profile, data_root, runtime_shell,
                         dataset=dataset, split_index=int(split) if split else None)
print('[SNAPSHOT] Captured source, effective configuration and environment.')
PYSNAPSHOT
else
    read_suite_context "${SUITE}"
fi
"${PYTHON}" - "${PROJECT_DIR}" "${SUITE}" <<'PYVERIFY'
import sys
sys.path.insert(0, sys.argv[1])
from run_cross_morse import verify_launcher_snapshot
record = verify_launcher_snapshot(sys.argv[2])
print('[SNAPSHOT VERIFIED]', record['dataset'], 'split', record['split_index'])
PYVERIFY

if command -v module >/dev/null 2>&1; then module list > "${SUITE}/launcher/modules_${JOB_KEY}.txt" 2>&1;
else printf 'No environment module system selected.\n' > "${SUITE}/launcher/modules_${JOB_KEY}.txt"; fi
"${PYTHON}" -m pip freeze > "${SUITE}/launcher/packages_${JOB_KEY}.txt"
nvidia-smi > "${SUITE}/launcher/nvidia_smi_${JOB_KEY}.txt" 2>&1 || true
uname -a > "${SUITE}/launcher/uname_${JOB_KEY}.txt"
"${PYTHON}" "${SUITE}/source/${RUNNER}" --gpu-health-check
request_checkpoint() {
    SIGNAL_RECEIVED=1; WAIT_INTERRUPTED=1
    printf '[SIGNAL] Requesting a checkpoint at the next safe Python boundary.\n' >&2
    [[ -z "${CHILD_PID}" ]] || kill -USR1 "${CHILD_PID}" 2>/dev/null || true
}
trap request_checkpoint USR1
cmd=("${PYTHON}" -u "${SUITE}/source/${RUNNER}" --dataset "${DATASET}"
     --config "${SUITE}/source/configs/resolved.json" --data-root "${DATA_ROOT}" --output-dir "${SUITE}"
     --device cuda --threads "${THREADS}" --stage "${STAGE}")
[[ "${RESUME}" == 0 ]] || cmd+=(--resume)
printf '[COMMAND]'; printf ' %q' "${cmd[@]}"; printf '\n'
cd "${SUITE}/source"
srun --ntasks=1 --cpus-per-task="${THREADS}" --unbuffered "${cmd[@]}" &
CHILD_PID=$!
while :; do
    WAIT_INTERRUPTED=0
    if wait "${CHILD_PID}"; then rc=0; else rc=$?; fi
    [[ "${WAIT_INTERRUPTED}" == 1 ]] && continue
    break
done
CHILD_PID=""
printf '[PROCESS EXIT] rc=%s suite=%s\n' "${rc}" "${SUITE}"
exit "${rc}"
