#!/bin/bash
#SBATCH -N 1
#SBATCH -t 4:00:00
#SBATCH -A aspect

module load Mambaforge/23.3.1-1-hpc1
mamba activate high_res_env
#python -u training_baseline_models.py
#python -u training_ncp_mse.py
#python -u training_ncp_mse_2003to2009.py
python -u scripts/training_ncp_mse.py exp1_normal_withT.yaml
#python -u training_ncp_mse_normal2009_nosm.py
#python -u inference_ncp_mse.py
#python -u inference_ncp_mse_normalized_mrsol_normal2009_nosm.py
