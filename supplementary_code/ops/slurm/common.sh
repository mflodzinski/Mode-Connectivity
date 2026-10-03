#!/bin/bash
#
# Shared helpers for repo-level Slurm launchers.

set -euo pipefail

mc_script_dir() {
  cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
}

mc_repo_root() {
  local script_dir
  script_dir="$(mc_script_dir)"
  cd "${script_dir}/../.." && pwd
}

mc_activate_venv() {
  local activate_path="${VENV_ACTIVATE:-$HOME/venvs/mode-connectivity/bin/activate}"
  if [ -f "${activate_path}" ]; then
    # shellcheck disable=SC1090
    source "${activate_path}"
  fi
}

mc_setup_python_env() {
  export PROJECT_ROOT="${PROJECT_ROOT:-$(mc_repo_root)}"
  mc_activate_venv
  cd "${PROJECT_ROOT}"
  export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-${USER}}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJECT_ROOT}/.mplcache}"
  mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"
}

mc_require_external_file() {
  local relative_path="$1"
  if [ ! -f "${PROJECT_ROOT}/${relative_path}" ]; then
    echo "Missing ${relative_path} in this checkout."
    exit 1
  fi
}

mc_banner() {
  local title="$1"
  echo "========================================"
  echo "${title}"
  echo "========================================"
}

mc_run_module() {
  local module_name="$1"
  shift
  srun python -m "${module_name}" "$@"
}

mc_eval_curve_checkpoint() {
  local checkpoint_path="$1"
  local output_dir="$2"
  local curve_name="$3"
  local num_bends="$4"

  mc_require_external_file "external/dnn-mode-connectivity/eval_curve.py"
  mkdir -p "${output_dir}"

  srun python "${PROJECT_ROOT}/external/dnn-mode-connectivity/eval_curve.py" \
    --dir "${output_dir}" \
    --dataset "${MC_DATASET:-CIFAR10}" \
    --data_path "${MC_DATA_PATH:-./data}" \
    --transform "${MC_TRANSFORM:-VGG}" \
    --model "${MC_MODEL:-VGG16}" \
    --curve "${curve_name}" \
    --num_bends "${num_bends}" \
    --ckpt "${checkpoint_path}" \
    --num_points "${MC_NUM_POINTS:-61}" \
    --use_test
}
