#!/bin/bash
set -euo pipefail

for split in 0 25 100 500 1000; do
  sbatch scripts/slurm/lmc_connected/run_open_lth_vgg16_split_pair.sh "$split"
done
