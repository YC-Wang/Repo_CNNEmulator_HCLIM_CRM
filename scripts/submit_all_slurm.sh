#!/bin/bash

set -euo pipefail

STAGE="${STAGE:-all}"
PARTITION="${PARTITION:-}"
ACCOUNT="${ACCOUNT:-}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-64G}"
GPUS_PER_JOB="${GPUS_PER_JOB:-1}"
CONDA_ENV="${CONDA_ENV:-high_res_env}"
MAMBA_MODULE="${MAMBA_MODULE:-Mambaforge/23.3.1-1-hpc1}"
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
  MAMBA_MODULE=Mambaforge/23.3.1-1-hpc1
  LOG_DIR=slurm_logs
  EXTRA_SBATCH_ARGS="--constraint=a100"

Examples:
  ACCOUNT=my_project STAGE=train bash scripts/submit_all_slurm.sh
  ACCOUNT=my_project STAGE=infer TIME_LIMIT=08:00:00 bash scripts/submit_all_slurm.sh
  ACCOUNT=my_project EXTRA_SBATCH_ARGS="--constraint=a100" bash scripts/submit_all_slurm.sh
  ACCOUNT=my_project PARTITION=gpu bash scripts/submit_all_slurm.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "${ACCOUNT}" ]]; then
  echo "Set ACCOUNT to your Freja project before submitting, for example:" >&2
  echo "  ACCOUNT=my_project bash scripts/submit_all_slurm.sh" >&2
  exit 1
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
  job_script="${LOG_DIR}/${experiment_id}-${STAGE}.sbatch"

  echo "Submitting ${config} as ${experiment_id}-${STAGE}"

  cat > "${job_script}" <<EOF
#!/bin/bash
#SBATCH -N 1
#SBATCH -t ${TIME_LIMIT}
#SBATCH -A ${ACCOUNT}
#SBATCH -J ${experiment_id}-${STAGE}
#SBATCH -o ${LOG_DIR}/${experiment_id}-%j.out
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEMORY}
EOF

  if [[ -n "${PARTITION}" ]]; then
    cat >> "${job_script}" <<EOF
#SBATCH -p ${PARTITION}
EOF
  fi

  if [[ "${GPUS_PER_JOB}" != "0" ]]; then
    cat >> "${job_script}" <<EOF
#SBATCH --gpus=${GPUS_PER_JOB}
EOF
  fi

  if [[ -n "${EXTRA_SBATCH_ARGS}" ]]; then
    for extra_arg in ${EXTRA_SBATCH_ARGS}; do
      cat >> "${job_script}" <<EOF
#SBATCH ${extra_arg}
EOF
    done
  fi

  cat >> "${job_script}" <<EOF

set -euo pipefail

module load ${MAMBA_MODULE}
mamba activate ${CONDA_ENV}

cd '$(pwd)'
python scripts/run_pipeline.py '${config}' --stage '${STAGE}'
EOF

  chmod +x "${job_script}"
  sbatch "${job_script}"
done
