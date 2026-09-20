#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/xor_train_linear_minimal.sh"
"${SCRIPT_DIR}/xor_permutation_scale_minimal.sh"
"${SCRIPT_DIR}/xor_joint_scale_path_minimal.sh"
"${SCRIPT_DIR}/xor_joint_scale_polygonal_minimal.sh"
