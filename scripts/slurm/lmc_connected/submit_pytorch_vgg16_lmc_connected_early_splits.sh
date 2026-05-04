#!/bin/bash
set -euo pipefail

for config in \
  vgg16_lmc_connected_pair_1split \
  vgg16_lmc_connected_pair_2split \
  vgg16_lmc_connected_pair_3split
do
  sbatch scripts/slurm/lmc_connected/run_pytorch_vgg16_lmc_connected_from_scratch.sh "${config}"
done
