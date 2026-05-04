#!/bin/bash

set -euo pipefail

SCRIPT="scripts/slurm/analysis/run_pytorch_vgg_split_pair_wm.sh"

sbatch "${SCRIPT}" "100/100"
sbatch "${SCRIPT}" "80/120"
sbatch "${SCRIPT}" "30/170"
sbatch "${SCRIPT}" "8/192"
sbatch "${SCRIPT}" "6/194"
sbatch "${SCRIPT}" "0/200"
sbatch "${SCRIPT}" "independent"
