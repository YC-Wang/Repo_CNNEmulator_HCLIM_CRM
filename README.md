# CNN Emulator for HCLIM Convection-Permitting Dataset

This repository contains a Python-based machine learning pipeline for emulating and downscaling climate variables (such as `tas` and `pr`) using Convolutional Neural Network (CNN) architectures. It bridges high-resolution climate model data (HCLIM-CRM) with computationally efficient ML emulators.

## Key Features & Extensions

* **Base Architecture:** Adapted from the deep learning downscaling approach by Rampal et al. (2022).
* **HCLIM Support:** Extended with dynamic configuration utilities and custom data pipelines tailored for HCLIM dataset ingestion.
* **Research Context:** Provides the CNN emulator implementation for **Wang et al. (2026)**, submitted to *Artificial Intelligence for the Earth Systems* (AIES).

---

## Reference

* Rampal, N., et al. (2022). High-resolution downscaling with interpretable deep learning: Rainfall extremes over New Zealand. *Weather and Climate Extremes*, 38, 100525. [https://doi.org/10.1016/j.wace.2022.100525](https://doi.org/10.1016/j.wace.2022.100525)

* Wang, F., Y.-C. Wang, H. E. Kourabbaslou, K. Krus, A. Aldama-Campino, G. Nikulin, R. Döscher, P. Lind, S. Mirjalili, C. Lennard, and F. Schenk, 2026: Emulating land–atmosphere feedbacks in convection-permitting regional climate models using machine learning. Artif. Intell. Earth Syst., submitted.

---

## Project Structure


```text
/your-project-root/
├── environment.yml       # Conda environment definition
├── README.md             # Project documentation
├── config.yaml           # Primary experiment configuration
├── scripts/              # Python source code
│   └── train.py          # Main training script
└── outputs/              # Large artifacts (Excluded from Git)
    ├── log_dir/          # Timestamped experiment logs & config backups
    └── models/           # Trained .h5 model weights


## Configuration Management
This project uses a YAML-centric workflow. All hyperparameters, file paths, and metadata are handled in config.yaml. This ensures that experiments are 100% reproducible without modifying the Python source code.

Example Configuration (config.yaml)
YAML
metadata:
  dataset_version: "v1.2_2026_revised"
  notes: "Testing SELU activation on temperature downscaling"

experiment:
  variable: "tas"
  dates:
    train: ["2000-01-01", "2007-12-31"]

model_setup:
  model_type: "cnn"
  layer_filters: [16, 32, 64]
  dropout: 0.6

training:
  learning_rate: 0.0001
  batch_size: 64
  experiment_tag: "baseline_run"

## Setup & Usage
###1. Installation
Create the isolated environment using Conda:

conda env create -f environment.yml
conda activate climate_emulator
###2. Running an Experiment
The script is designed to be run from the root directory. It accepts an optional configuration file argument.

# Run using the default config.yaml in root
python scripts/train.py

# Run using a specific versioned configuration
python scripts/train.py configs/test_v2.yaml
📊 Outputs & Reproducibility
Automated Experiment Tracking
Every execution generates a unique Experiment ID (YYYYMMDD-HHMM_model_var_tag_jobID).

Logs: TensorBoard-compatible logs are saved to outputs/log_dir/{model_type}/{exp_id}/.

Config Backup: A copy of the .yaml file used for the run is saved directly into the log folder.

Model Weights: Trained weights are exported to outputs/models/{exp_id}.h5.

## Monitoring Results
To visualize loss curves and metrics in real-time, point TensorBoard to your output directory:

tensorboard --logdir outputs/log_dir/

### Model Options
Simple Conv (CNN): A convolutional neural network optimized for spatial climate data featuring selu activation and configurable kernel sizes.
