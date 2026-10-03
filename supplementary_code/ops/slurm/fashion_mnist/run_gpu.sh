#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --gres=gpu:a40:1
#SBATCH --signal=USR1@120

set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${PROJECT_ROOT}/ops/slurm/common.sh"
mc_setup_python_env
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
manifest="$1"
operation="$2"
if [ "${operation}" = "train" ]; then
  seed=$(python -c 'import json,os,sys; c=json.load(open(sys.argv[1]))["config"]; print(sum(c["seed_pairs"], [])[int(os.environ["SLURM_ARRAY_TASK_ID"])])' "${manifest}")
  srun python -m experiments.fashion_mnist.run train --manifest "${manifest}" --seed "${seed}"
elif [ "${operation}" = "analyze" ] || [ "${operation}" = "linear" ] || [ "${operation}" = "nonlinear" ]; then
  read -r replicate epoch < <(python -c 'import json,os,sys; c=json.load(open(sys.argv[1]))["config"]; a=[(r,e) for r in range(len(c["seed_pairs"])) for e in c["stages"]]; print(*a[int(os.environ["SLURM_ARRAY_TASK_ID"])])' "${manifest}")
  srun python -m experiments.fashion_mnist.run "${operation}" --manifest "${manifest}" --replicate "${replicate}" --epoch "${epoch}"
else
  echo "Unsupported GPU operation: ${operation}" >&2
  exit 2
fi
