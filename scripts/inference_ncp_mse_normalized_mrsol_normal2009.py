#!/usr/bin/env python
# coding: utf-8

#get_ipython().run_line_magic('load_ext', 'autoreload')
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
import xarray as xr
import os
import numpy as np
import sys
import tensorflow as tf
import tensorflow.keras.backend as K
import cartopy.crs as ccrs

# these are my working paths
#sys.path.append(r'/nesi/project/niwa00018/rampaln/High-res-interpretable-dl/src')
#sys.path.append(r'/nobackup/rossby27/users/sm_yicwa/PROJECTS/01-PROJ_emulator/01-rampal2021-unet/high-resolution-downscaling/src')
sys.path.append(r'/nobackup/rossby27/users/sm_yicwa/PROJECTS/01-PROJ_emulator/01-rampal2021-unet/Emulator_HCLIM_CRM_T_SM/src_temp/')
#sys.path.append(r'src_temp/')
# change to the directory of the "src file" in your directory
#os.chdir(r'/nesi/project/niwa00018/rampaln/High-res-interpretable-dl')
#os.chdir(r'/nobackup/rossby27/users/sm_yicwa/PROJECTS/01-PROJ_emulator/01-rampal2021-unet/high-resolution-downscaling/')
os.chdir(r'./')
# change directory to your repository of interest
import tensorflow as tf
from dask.diagnostics import ProgressBar
import cmocean
from models import train_model, complex_conv, simple_conv, predict, simple_dense, linear_complex_model
from losses import gamma_loss_1d, gamma_mse_metric
from prepare_data import format_features, prepare_training_dataset, create_test_train_split


tf.random.set_seed(2)


# In[6]:

#wrkdir = '/nobackup/rossby27/users/sm_yicwa/PROJECTS/01-PROJ_emulator/01-rampal2021-unet/Emulator_HCLIM_CRM/'

#variable = 'mrsol'
variable = 'tas'
wrkdir = '/nobackup/rossby27/users/sm_yicwa/PROJECTS/01-PROJ_emulator/01-rampal2021-unet/Emulator_HCLIM_CRM_T_withSM_whus/'
wrkdir_inf = f'03-inference_comp/'
#dir_fuxing ="/nobackup/rossby27/users/sm_fuxwa/AI_data/3km/6hr/pr/")
dir_fuxing ='training_data_fuxing/'

dir_fuxing_org = '/nobackup/rossby27/users/sm_fuxwa/AI_data/Emilia_Romagna/3km/6hr/tas/'
#dir_fuxing_org = '/nobackup/rossby27/users/sm_yicwa/PROJECTS/01-PROJ_emulator/01-rampal2021-unet/Emulator_HCLIM_CRM_T_withSM_whus/training_data_fuxing/Emilia_Romagna/'
#y_file = "pr_3km_6hr_200001010300-200912312100.nc"
#y_file = f"{variable}_3km_6hr_200001010000-200912311800_swapped_2003_2009.nc"
#x_file = "combined_12km_6hr_20000101-20091231_swapped_2003_2009.nc"
y_file = f"{variable}_3km_6hr_200001010000-200912311800.nc"
x_file = f"combined_12km_6hr_20000101-20091231.nc"


# here is where we select where the input and output data is the features we want to use
config = dict(#y = wrkdir+"training_data/topography_subset.nc",
              #X = wrkdir+"training_data/ERA5_training_dataset_6_3_23_bilinear_half_degree.nc",
#              y = wrkdir+"training_data_hclim/interpolated_pr_combined_2010-2018.nc",
#              X = wrkdir+"training_data_ncp/combined_variables_2010-2018.nc",
             y = dir_fuxing_org+y_file,
             X = wrkdir+dir_fuxing+x_file,
             train_start = "2000-01-01",
             train_end  = "2007-12-31",
             val_start  = "2008-01-01",
             val_end    = "2008-12-31",
             test_start = "2009-01-01",
             test_end   = "2009-12-31",
             output_var = [variable],
             #downscale_variables = ['w_850', 'u_850',
             #'v_850', 'q_850', 't_850'])
             #downscale_variables = ['hus850', 'sfcWind', 'psl', 'zg500'])
             #downscale_variables = ['phi500','phi700','phi850','phi950','ta500','ta700','ta850','ta950','ua500','ua700','ua850','ua950','va500','va700','va850','va950','mrsol'])
             downscale_variables = ['phi500','phi700','phi850','phi950','ta500','ta700','ta850','ta950','ua500','ua700','ua850','ua950','va500','va700','va850','va950','hus500','hus700','hus850','hus950','mrsol'])
# you can modify any of the above features


# # Loading the Training Data
# Here we load the training data from a configuration file and prepare it for training DL models

# create a data split for train, test, and validation
x_train, x_val, x_test, y_train, y_val, y_test = create_test_train_split(config)

#outscale = 86400. # for rainfall
outscale = 1. # for temperature
y_train = y_train*outscale
y_val   = y_val*outscale
y_test  = y_test*outscale

print(x_test.dims)
print(x_test.coords)
print(x_test)

####### read in 
downscale_variables = [
    'phi500', 'phi700', 'phi850', 'phi950',
    'ta500', 'ta700', 'ta850', 'ta950',
    'ua500', 'ua700', 'ua850', 'ua950',
    'va500', 'va700', 'va850', 'va950',
    'hus500', 'hus700', 'hus850', 'hus950',
    'mrsol'
]

#### normalized 
# Compute Mean and Standard Deviation from Training Data
train_mean = y_train.mean(dim="time")
train_std = y_train.std(dim="time")

# Standardization (Z-score normalization)
#y_train_stded = (y_train - train_mean) / train_std
#y_val_stded   = (y_val - train_mean) / train_std
y_test_stded  = (y_test - train_mean) / train_std

#y_train = y_train_stded
#y_val   = y_val_stded
#y_test  = y_test_stded
### normalized


# load the training data
x_train, x_test, x_val, y_train, y_test, y_val = prepare_training_dataset(x_train, x_val, x_test, y_train, y_val, y_test)      

# modify the training data so that it is compatible with tensorflow and training
x_train = x_train.values if isinstance(x_train, xr.DataArray) else x_train
y_train = y_train.values if isinstance(y_train, xr.DataArray) else y_train
y_train = y_train.to_array().values if isinstance(y_train, xr.Dataset) else y_train

#print(type(x_train))
#print(type(y_train))

##########################################
# Load existing model for Inference
##########################################
print(f'Read in ')
# For MSE is built in for metrics and error, there is no need to assign custom_object.
#simple_cnn = tf.keras.models.load_model(f'01-model/cnn_mse_model_with_new_training_period.h5')
simple_cnn = tf.keras.models.load_model(f'01-model/linear_mse_model_with_new_training_period_normal2009.h5')

# Show the model architecture
simple_cnn.summary()
#loss, acc = simple_cnn.evaluate(x_test, y_test, verbose=2)
#print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

# # Inference for MSE LOss
simple_cnn_prediction = predict(simple_cnn, x_test, y_test, 
                                batch_size=32, key =variable, pred_name ="test", 
                                loss ='mse' , thres =0.5)

simple_cnn_prediction = simple_cnn_prediction.unstack()
# Ensure train_mean and train_std have the same dimensions as the test data
train_mean_aligned    = train_mean.broadcast_like(simple_cnn_prediction.test)
train_std_aligned     = train_std.broadcast_like(simple_cnn_prediction.test)
simple_cnn_prediction = (simple_cnn_prediction * train_std_aligned) + train_mean_aligned

# Specify the output file name
#time_encoding = {'time': {'dtype': 'float64', 'units': 'hours since 2009-01-01 03:00:00', 'calendar': 'proleptic_gregorian'}} # for rainfall
time_encoding = {'time': {'dtype': 'float64', 'units': 'hours since 2009-01-01 00:00:00', 'calendar': 'proleptic_gregorian'}} # for tas
output_file = "simple_cnn_prediction_normalized_normal2009.nc"
simple_cnn_prediction.to_netcdf(wrkdir_inf+output_file, encoding=time_encoding)
print(f"Predictions saved to {output_file}")
print(simple_cnn_prediction.time)

#simple_dense_prediciton = predict(linear_model, x_test, y_test, 
#                                  batch_size=32, key ='pr', pred_name ="test", 
#                                  loss ='mse' , thres =0.5)
#simple_dense_prediciton = simple_dense_prediciton.unstack()
# +++ Yi-Chi: Feb 2025
#simple_dense_prediciton = simple_dense_prediciton.reindex(
#                            lon = sorted(simple_dense_prediciton.lon.values))
# --- Yi-Chi

# Specify the output file name
#output_file = "simple_dense_prediction.nc"
#simple_dense_prediciton.to_netcdf(output_file)
#print(f"Predictions saved to {output_file}")


# Save the ground truth and calculate correction
gt = y_test.unstack()
#gt = gt.reindex(lon = sorted(gt.lon.values))


# # Correlation Coefficient in Time Evalation
corrs2 = xr.corr(gt, simple_cnn_prediction.test, dim ="time")
print(f'correlation coefficient: {corrs2.mean(["x","y"])}')
# the average correlation coefficient
# Compute Mean Absolute Error (MAE)
mae = np.abs(gt - simple_cnn_prediction.test).mean(dim="time")
print(f'mean absolute error: {mae.mean(["x","y"])}')

# Compute Root Mean Squared Error (RMSE)
rmse = np.sqrt(((gt - simple_cnn_prediction.test) ** 2).mean(dim="time"))
print(f'RMSE: {rmse.mean(["x","y"])}')

std_dev_gt = gt.std(dim="time")
print(f'standard deviation of truth: {std_dev_gt.mean(["x","y"])}')

std_dev_pred = (simple_cnn_prediction.test).std(dim="time")
print(f'standard deviation of prediction: {std_dev_pred.mean(["x","y"])}')

# ==================
# 1. Calculate the scalar mean values for each metric
# We use .item() to get the actual number out of the xarray/numpy object
results = {
    "metric": ["correlation", "mae", "rmse", "std_dev_truth", "std_dev_pred"],
    "value": [
        corrs2.mean(["x", "y"]).item(),
        mae.mean(["x", "y"]).item(),
        rmse.mean(["x", "y"]).item(),
        std_dev_gt.mean(["x", "y"]).item(),
        std_dev_pred.mean(["x", "y"]).item()
    ]
}

# 2. Create a DataFrame
df_metrics = pd.DataFrame(results)

# 3. Save to CSV
df_metrics.to_csv("evaluation_metrics_normal2009.csv", index=False)

print("Metrics successfully saved to evaluation_metrics.csv")


