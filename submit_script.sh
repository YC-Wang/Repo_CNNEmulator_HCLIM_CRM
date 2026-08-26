#!/bin/bash
#SBATCH -N 1
#SBATCH -t 4:00:00
#SBATCH -A aspect

module load Mambaforge/23.3.1-1-hpc1
mamba activate high_res_env
#python -u scripts/training_ncp_mse.py exp1_normal_withT.yaml
python scripts/inference_ncp_mse.py exp1_normal_withT.yaml --model-file /nobackup/rossby27/users/sm_yicwa/PROJECTS/01-PROJ_emulator/01-rampal2021-unet/output/models/20260825-1718_cnn_tas_exp1_heatwave_withT.h5
