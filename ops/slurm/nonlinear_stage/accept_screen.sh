#!/bin/bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 RESULT_ROOT [extra accept_screen.py arguments]" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
mc_setup_python_env
ROOT="$1"
shift
python -m experiments.nonlinear_stage.accept_screen --root "$ROOT" "$@"
