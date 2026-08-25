#!/bin/bash
#SBATCH -N 1
#SBATCH -t 4:00:00
#SBATCH -A aspect
#SBATCH -J ECE-MC2HI-all
#SBATCH -o slurm_logs/ECE-MC2HI-%j.out

set -euo pipefail

module load Mambaforge/23.3.1-1-hpc1
mamba activate high_res_env

cd '/nobackup/rossby27/users/sm_yicwa/PROJECTS/01-PROJ_emulator/01-rampal2021-unet/Repo_CNNEmulator_HCLIM_CRM_rainfall_multiexp'
python scripts/run_pipeline.py 'configs/pr06.yaml' --stage 'all'
