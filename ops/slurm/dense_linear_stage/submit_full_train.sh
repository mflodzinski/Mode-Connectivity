#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
mc_setup_python_env

if [ "$#" -lt 1 ]; then
  echo "usage: $0 RESULT_ROOT [RESULT_ROOT ...]"
  exit 2
fi

account=()
if [ -n "${MC_ACCOUNT:-}" ]; then
  account=(--account="${MC_ACCOUNT}")
fi
previous_job=""
for experiment_root in "$@"; do
  experiment_root="$(python -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${experiment_root}")"
  mkdir -p "${experiment_root}/logs"
  status_json="$(python -m experiments.dense_linear_stage.full_train status "${experiment_root}")"
  missing="$(python -c 'import json,sys; print(",".join(map(str,json.load(sys.stdin)["missing_tasks"])))' <<<"${status_json}")"
  report_complete="$(python -c 'import json,sys; print(str(json.load(sys.stdin)["report_complete"]).lower())' <<<"${status_json}")"
  dependency=()
  if [ -n "${previous_job}" ]; then
    dependency=(--dependency="afterok:${previous_job}")
  fi
  if [ -n "${missing}" ]; then
    evaluation_job="$(sbatch --parsable \
      --partition="${MC_PARTITION:-general}" \
      --qos="${MC_QOS:-short}" \
      --ntasks=1 --cpus-per-task=1 --mem=3GB --time=03:50:00 \
      --gres="${MC_GRES:-gpu:a40:1}" \
      "${account[@]}" "${dependency[@]}" \
      --signal=USR1@60 --mail-type=FAIL \
      --job-name=dense_full_train \
      --output="${experiment_root}/logs/%x_%A_%a.out" \
      --error="${experiment_root}/logs/%x_%A_%a.err" \
      --array="${missing}%${MC_CONCURRENCY:-3}" \
      "${SCRIPT_DIR}/run_full_train.sh" evaluate "${experiment_root}")"
    evaluation_job="${evaluation_job%%;*}"
    dependency=(--dependency="afterok:${evaluation_job}")
    echo "${experiment_root}: evaluation array ${evaluation_job} tasks ${missing}"
  else
    echo "${experiment_root}: all 18 method/pair profiles already complete"
    if [ "${report_complete}" = "true" ]; then
      echo "${experiment_root}: full-train report already complete"
      continue
    fi
  fi
  report_job="$(sbatch --parsable \
    --partition="${MC_PARTITION:-general}" \
    --qos="${MC_QOS:-short}" \
    --ntasks=1 --cpus-per-task=1 --mem=2GB --time=00:20:00 \
    "${account[@]}" \
    --mail-type=FAIL --job-name=dense_full_report \
    --output="${experiment_root}/logs/%x_%j.out" \
    --error="${experiment_root}/logs/%x_%j.err" \
    "${dependency[@]}" \
    "${SCRIPT_DIR}/run_full_train.sh" report "${experiment_root}")"
  report_job="${report_job%%;*}"
  previous_job="${report_job}"
  echo "${experiment_root}: report job ${report_job}"
done
