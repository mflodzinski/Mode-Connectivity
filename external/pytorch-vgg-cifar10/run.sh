#!/bin/bash

set -euo pipefail

ARCH="${ARCH:-vgg11}"
SEEDS=(0 1)

for seed in "${SEEDS[@]}"
do
    save_dir="save_${ARCH}_seed${seed}"
    log_file="log_${ARCH}_seed${seed}"
    echo "python main.py --arch=${ARCH} --seed=${seed} --save-dir=${save_dir} 2>&1 | tee -a ${log_file}"
    python main.py --arch="${ARCH}" --seed="${seed}" --save-dir="${save_dir}" 2>&1 | tee -a "${log_file}"
done
