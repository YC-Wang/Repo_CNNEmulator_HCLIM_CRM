#!/usr/bin/env bash

set -euo pipefail

STAGE="${STAGE:-all}"
PARTITION="${PARTITION:-gpu}"
ACCOUNT="${ACCOUNT:-}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-64G}"
GPUS_PER_JOB="${GPUS_PER_JOB:-1}"
CONDA_ENV="${CONDA_ENV:-high_res_env}"
LOG_DIR="${LOG_DIR:-slurm_logs}"
EXTRA_SBATCH_ARGS="${EXTRA_SBATCH_ARGS:-}"

CONFIGS=(
  "configs/pr01.yaml"
  "configs/pr03.yaml"
  "configs/pr04.yaml"
  "configs/pr05.yaml"
  "configs/pr06.yaml"
  "configs/pr07.yaml"
)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/submit_all_slurm.sh

Environment overrides:
  STAGE=train|infer|all
  PARTITION=gpu
  ACCOUNT=my_project
  TIME_LIMIT=24:00:00
  CPUS_PER_TASK=8
  MEMORY=64G
  GPUS_PER_JOB=1
  CONDA_ENV=high_res_env
  LOG_DIR=slurm_logs
  EXTRA_SBATCH_ARGS="--constraint=a100"

Examples:
  STAGE=train bash scripts/submit_all_slurm.sh
  STAGE=infer PARTITION=gpu TIME_LIMIT=08:00:00 bash scripts/submit_all_slurm.sh
  ACCOUNT=myproj EXTRA_SBATCH_ARGS="--constraint=a100" bash scripts/submit_all_slurm.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -d "configs" || ! -f "scripts/run_pipeline.py" ]]; then
  echo "Run this script from the repository root." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

for config in "${CONFIGS[@]}"; do
  if [[ ! -f "${config}" ]]; then
    echo "Missing config: ${config}" >&2
    exit 1
  fi
done

for config in "${CONFIGS[@]}"; do
  experiment_id="$(basename "${config}" .yaml)"

  sbatch_args=(
    --job-name="${experiment_id}-${STAGE}"
    --partition="${PARTITION}"
    --time="${TIME_LIMIT}"
    --cpus-per-task="${CPUS_PER_TASK}"
    --mem="${MEMORY}"
    --output="${LOG_DIR}/${experiment_id}-%j.out"
  )

  if [[ -n "${ACCOUNT}" ]]; then
    sbatch_args+=(--account="${ACCOUNT}")
  fi

  if [[ "${GPUS_PER_JOB}" != "0" ]]; then
    sbatch_args+=(--gpus="${GPUS_PER_JOB}")
  fi

  if [[ -n "${EXTRA_SBATCH_ARGS}" ]]; then
    # Intentionally split here to allow multiple sbatch flags via one env var.
    # shellcheck disable=SC2206
    extra_args=( ${EXTRA_SBATCH_ARGS} )
    sbatch_args+=("${extra_args[@]}")
  fi

  echo "Submitting ${config} as ${experiment_id}-${STAGE}"

  sbatch "${sbatch_args[@]}" --wrap "
    set -euo pipefail
    source ~/.bashrc
    mamba activate ${CONDA_ENV}
    cd '$(pwd)'
    python scripts/run_pipeline.py '${config}' --stage '${STAGE}'
  "
done
