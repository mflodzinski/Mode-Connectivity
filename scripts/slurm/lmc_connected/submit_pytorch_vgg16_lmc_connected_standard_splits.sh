#!/bin/bash
set -euo pipefail

for config in \
  vgg16_lmc_connected_pair_100split \
  vgg16_lmc_connected_pair_0split \
  vgg16_lmc_connected_pair_8split \
  vgg16_lmc_connected_pair_6split \
  vgg16_lmc_connected_pair_30split \
  vgg16_lmc_connected_pair_80split
do
  sbatch scripts/slurm/lmc_connected/run_pytorch_vgg16_lmc_connected_from_scratch.sh "${config}"
done
