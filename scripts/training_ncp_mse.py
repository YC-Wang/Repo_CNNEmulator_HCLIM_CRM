#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import xarray as xr
import xarray as xr
import os
import numpy as np
import sys
import tensorflow as tf
import tensorflow.keras.backend as K
import cartopy.crs as ccrs
from dask.diagnostics import ProgressBar

# set up code directory
sys.path.append(r'../src/')
os.chdir(r'./')
# read in subroutines
from models import train_model, simple_conv, predict, simple_dense
from losses import gamma_loss_1d, gamma_mse_metric
from prepare_data import format_features, prepare_training_dataset, create_test_train_split


tf.random.set_seed(2)

# ----------------------------
# READ IN CONFIG FILES
# ----------------------------
import yaml
import shutil  # No pip install needed!
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(description="Run ML Experiment with YAML config")
parser.add_argument(
    "config_file", 
    type=str, 
    help="Path to the yaml configuration file",
    default="config.yaml",
    nargs="?" # This makes it optional; defaults to config.yaml if not provided
)
args = parser.parse_args()

try:
    with open(args.config_file, "r") as f:
        raw_cfg = yaml.safe_load(f)
    print(f"--- Loaded Configuration: {args.config_file} ---")
except FileNotFoundError:
    print(f"Error: The file {args.config_file} was not found.")
    sys.exit(1)


# --- Path & Data Reorganization ---
wrkdir = raw_cfg['paths']['work_dir']
data_train_dir = raw_cfg['paths']['data_train_dir']
data_infer_dir = raw_cfg['paths']['data_infer_dir']
variable = raw_cfg['experiment']['variable']

y_file = raw_cfg['experiment']['y_filename_template'].format(variable=variable)
x_file = raw_cfg['experiment']['x_filename']

exp_config = {
    "y": os.path.join(data_infer_dir, y_file),
    "X": os.path.join(wrkdir, data_train_dir, x_file),
    "train_start": raw_cfg['experiment']['dates']['train'][0],
    "train_end":   raw_cfg['experiment']['dates']['train'][1],
    "val_start":   raw_cfg['experiment']['dates']['val'][0],
    "val_end":     raw_cfg['experiment']['dates']['val'][1],
    "test_start":  raw_cfg['experiment']['dates']['test'][0],
    "test_end":    raw_cfg['experiment']['dates']['test'][1],
    "output_var": [variable],
    "downscale_variables": raw_cfg['experiment']['downscale_variables']
}

#print(exp_config)

# --- Individual Model & Training Variables ---
initial_learning_rate = raw_cfg['training']['learning_rate']
dropout = raw_cfg['model']['dropout']
hidden_layer_dense = raw_cfg['model']['hidden_layer_dense']
batch_size = raw_cfg['training']['batch_size']
kernel_size = raw_cfg['model']['kernel_size']
layer_filters = raw_cfg['model']['layer_filters']
epochs = raw_cfg['training']['epochs']
# --- Read in architecture hyperparameters-------

m = raw_cfg['model']
dense_act = m['dense_activation']
cnn_act   = m['cnn_activation']
pad       = m['padding']
bn        = m['use_bn']
pooling   = m['use_pooling']
dropout   = m['dropout']
hidden_layer_dense   = m['hidden_layer_dense']
layer_filters   = m['layer_filters']
kernel_size    = m['kernel_size']

# --- Read in model and log dir ---
t = raw_cfg['training']
model_type = t['model_type']  # e.g., 'linear'
tag        = t['experiment_tag']
timestamp = datetime.now().strftime("%Y%m%d-%H%M")
exp_id = f"{timestamp}_{model_type}_{variable}_{tag}"

logdir = os.path.join(t['log_root'], model_type, exp_id)
model_weights_name = os.path.join(t['model_root'], f"{exp_id}.h5")

os.makedirs(logdir, exist_ok=True) # Automatically create the directory
shutil.copy(args.config_file, os.path.join(logdir, "config_backup.yaml"))

print(f"Log Directory: {logdir}")
print(f"Weights Path:  {model_weights_name}")
# -------------------------

# # Loading the Training Data
x_train, x_val, x_test, y_train, y_val, y_test = create_test_train_split(exp_config)

#outscale = 86400. # for rainfall
outscale = 1. # for temperature
y_train = y_train*outscale
y_val   = y_val*outscale
y_test  = y_test*outscale

# normalized 
# Compute Mean and Standard Deviation from Training Data
train_mean = y_train.mean(dim="time")
train_std = y_train.std(dim="time")
#train_mean.to_netcdf(f"train_mean_{variable}_INTERIM_2000to2009.nc")
#train_std.to_netcdf(f"train_std_{variable}_INTERIM_2000to2009.nc")


# Standardization (Z-score normalization)
y_train = (y_train - train_mean) / train_std
y_val   = (y_val - train_mean) / train_std
y_test  = (y_test - train_mean) / train_std

# load the training data
x_train, x_test, x_val, y_train, y_test, y_val = prepare_training_dataset(x_train, x_val, x_test, y_train, y_val, y_test)      

input_shape = x_train.shape[1:]
output_shape = y_train.z.size

# yi-chi #
#optimizer = tf.keras.optimizers.Adam(lr =initial_learning_rate)
from tensorflow.keras.optimizers import legacy
optimizer = legacy.Adam(lr =initial_learning_rate)
# yi-chi #


# # Defining Three Model Architectures
# cnn model with mse loss
simple_cnn = simple_conv(layer_filters=layer_filters, 
                         bn=bn, padding=pad, 
                         kernel_size=(kernel_size,kernel_size),
                         pooling=pooling, 
                         dense_layers=[hidden_layer_dense, output_shape], 
                         dense_activation=dense_act, input_shape=input_shape,
                         dropout=dropout, activation=cnn_act)



x_train = x_train.values if isinstance(x_train, xr.DataArray) else x_train
y_train = y_train.values if isinstance(y_train, xr.DataArray) else y_train
y_train = y_train.to_array().values if isinstance(y_train, xr.Dataset) else y_train


# set up model to run
# --- Extract Training Parameters ---
t = raw_cfg['training']
model_type = t['model_type']  # e.g., 'linear'
active_model = linear_model if model_type == "linear" else simple_cnn

# --- Run the Training ---
history, trained_model = train_model(
    active_model, x_train, y_train,
    x_val = x_val.values, y_val = y_val.values,
    loss = t['loss'], epochs = t['epochs'], 
    batch_size = t['batch_size'],
    optimizer = optimizer,
    model_weights_name = model_weights_name,
    logdir = logdir,
    metrics = t['metrics']
)

