#!/bin/bash

set -euo pipefail

STAGE="${STAGE:-all}"
ACCOUNT="${ACCOUNT:-aspect}"
TIME_LIMIT="${TIME_LIMIT:-4:00:00}"
CONDA_ENV="${CONDA_ENV:-high_res_env}"
MAMBA_MODULE="${MAMBA_MODULE:-Mambaforge/23.3.1-1-hpc1}"
LOG_DIR="${LOG_DIR:-slurm_logs}"

CONFIGS=(
#  "configs/pr01.yaml"
#  "configs/pr03.yaml"
#  "configs/pr04.yaml"
#  "configs/pr05.yaml"
#  "configs/pr06.yaml"
  "configs/pr07.yaml"
)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/submit_all_slurm.sh

Environment overrides:
  STAGE=train|infer|all
  ACCOUNT=aspect
  TIME_LIMIT=4:00:00
  CONDA_ENV=high_res_env
  MAMBA_MODULE=Mambaforge/23.3.1-1-hpc1
  LOG_DIR=slurm_logs

Examples:
  bash scripts/submit_all_slurm.sh
  STAGE=train bash scripts/submit_all_slurm.sh
  STAGE=infer TIME_LIMIT=8:00:00 bash scripts/submit_all_slurm.sh
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

if ! command -v python >/dev/null 2>&1; then
  echo "python is required to read experiment_id from YAML configs." >&2
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
  experiment_id="$(python - <<EOF
import yaml
with open("${config}", "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
print(config["metadata"]["experiment_id"])
EOF
)"
  job_slug="$(printf '%s' "${experiment_id}" | tr ' /' '__' | tr -cd '[:alnum:]_.-')"
  job_script="${LOG_DIR}/${job_slug}-${STAGE}.sbatch"

  echo "Submitting ${config} as ${experiment_id}-${STAGE}"

  cat > "${job_script}" <<EOF
#!/bin/bash
#SBATCH -N 1
#SBATCH -t ${TIME_LIMIT}
#SBATCH -A ${ACCOUNT}
#SBATCH -J ${job_slug}-${STAGE}
#SBATCH -o ${LOG_DIR}/${job_slug}-%j.out
EOF

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
