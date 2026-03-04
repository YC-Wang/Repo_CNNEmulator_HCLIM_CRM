import xarray as xr
import pandas as pd
import numpy as np
from dask.diagnostics import ProgressBar

## updated: Feb 3, 2025 by Yi-Chi
#


def format_features(x_data):
    varnames = list(x_data.data_vars)
    features = xr.concat([x_data[varname] for varname in varnames], dim = "feature")
    # setting a new dimension name in the output
    features['feature'] = (('feature'), varnames)
    features.name = "stacked_features"
    print(f'format_features:{varnames}')
    return features
    
def create_test_train_split(config):
    """
    This create a train test validation split of the dataset givena  configuration file
    """
    X = xr.open_dataset(config["X"], chunks = {"time":3000})[config["downscale_variables"]]
    # +++ Yi-Chi: remove this part to make it fit for 6-hr data
    # make sure time values are daily
    #X['time'] = pd.to_datetime(X.time.dt.strftime("%Y-%m-%d"))
    print(config["downscale_variables"])
    #outputvar = config["downscale_variables"]
    if config["output_var"][0] == "pr":
        X['time'] = pd.to_datetime(X.time.dt.strftime("%Y-%m-%d %H:00:00")) + pd.Timedelta(hours=3) # for rainfall
    else:
        X['time'] = pd.to_datetime(X.time.dt.strftime("%Y-%m-%d %H:00:00")) # for temperature and others
    print(X['time'])

    y = xr.open_dataset(config["y"], chunks = {"time":3000})[config["output_var"][0]]
    print(f'output variables: {config["output_var"]}')
    #y['time'] = pd.to_datetime(y.time.dt.strftime("%Y-%m-%d")) 
    y['time'] = pd.to_datetime(y.time.dt.strftime("%Y-%m-%d %H:00:00"))
    print(y['time'])
###
# Check and remove duplicates from the time index
    if not X.time.to_index().is_unique:
        print("Duplicates found in X.time. Dropping duplicates...")
        X = X.sel(time=~X.time.to_index().duplicated())

    if not y.time.to_index().is_unique:
        print("Duplicates found in y.time. Dropping duplicates...")
        y = y.sel(time=~y.time.to_index().duplicated())
###
    common_times = X.time.to_index().intersection(y.time.to_index())
    print(f'common_times: {common_times}')
    X = X.sel(time = common_times)
    y = y.sel(time = common_times)
    # check the y values are also daily
    
    # apply the train test split partiions and load into memory
    with ProgressBar():
        x_train = X.sel(time = slice(config["train_start"], config["train_end"])).load()
        y_train = y.sel(time = slice(config["train_start"], config["train_end"])).load()

        x_test = X.sel(time = slice(config["test_start"], config["test_end"])).load()
        y_test = y.sel(time = slice(config["test_start"], config["test_end"])).load()


        x_val = X.sel(time = slice(config["val_start"], config["val_end"])).load()
        y_val = y.sel(time = slice(config["val_start"], config["val_end"])).load()
                  
    return x_train, x_val, x_test, y_train, y_val, y_test

def prepare_training_dataset(x_train, x_val, x_test,
                             y_train, y_val, y_test,
                             means = None, stds = None):
    """Normalizes and restacks 
    
    the training data so that it is useful for training
    
    """
    if means is None:
        # computing means and stds
        means, stds = x_train.mean(), x_train.std()
    # normalizing the data based on the x_train mean
    x_train_norm = (x_train - means)/stds
    x_test_norm = (x_test- means)/stds
    x_val_norm = (x_val - means)/stds

    try:
        # format the features so that they are stacked
        #print("x_train_norm dimensions:", x_train_norm.dims)
        #print("x_train_norm dimensions (after format_features):", x_train_norm.dims)
        x_train_norm = format_features(x_train_norm).transpose("time","y","x","feature")
        x_test_norm  = format_features(x_test_norm).transpose("time","y","x","feature")
        x_val_norm   = format_features(x_val_norm).transpose("time","y","x","feature")
    except KeyError as e1:
        try:
            x_train_norm = format_features(x_train_norm).transpose("time","latitude","longitude","feature")
            x_test_norm = format_features(x_test_norm).transpose("time","latitude","longitude","feature")
            x_val_norm = format_features(x_val_norm).transpose("time","latitude","longitude","feature")
        except KeyError as e2:
            try:
                x_train_norm = format_features(x_train_norm).transpose("time","y","x","feature")
                x_test_norm = format_features(x_test_norm).transpose("time","y","x","feature")
                x_val_norm = format_features(x_val_norm).transpose("time","y","x","feature")
            except KeyError as e3:
                temp_x_train = format_features(x_train_norm)
                available_dims = getattr(temp_x_train, 'dims', "Dimensions not available (format_features may not return an xarray object)")
                expected_dims = {'feature', 'time', 'y', 'x', 'lat', 'lon', 'latitude', 'longitude'}
                raise ValueError(
                    f"None of the expected dimension combinations ('lat'/'lon', 'latitude'/'longitude', 'y'/'x') found. Available dimensions are: {available_dims}. Expected one or more of {expected_dims}"
                ) from None


    try:
    # prepare the rainfall data
    #    y_train = y_train.stack(z =['lat','lon']).dropna("z")
    #    y_test = y_test.stack(z =['lat','lon']).dropna("z")
    #    y_val= y_val.stack(z =['lat','lon']).dropna("z")
    #    y_train = y_train*86400.
    #    y_test = y_test*86400.
    #    y_val= y_val*86400.
        print(f'y_train: stack on z')
        y_train = y_train.stack(z =['y','x']).dropna("z")
        y_test = y_test.stack(z =['y','x']).dropna("z")
        y_val= y_val.stack(z =['y','x']).dropna("z")
    except KeyError as e1:
        try:
            y_train = y_train.stack(z =['latitude','longitude']).dropna("z")
            y_test  = y_test.stack(z =['latitude','longitude']).dropna("z")
            y_val   = y_val.stack(z =['latitude','longitude']).dropna("z")
        except KeyError as e2:
            try:
                y_train = y_train.stack(z =['y','x']).dropna("z")
                y_test = y_test.stack(z =['y','x']).dropna("z")
                y_val= y_val.stack(z =['y','x']).dropna("z")
            except KeyError as e3:
                raise ValueError(
                    f"None of the expected dimension combinations ('lat'/'lon', 'latitude'/'longitude', 'y'/'x') found.  Available dimensions are: {y_train.dims}"
                ) from None  # Raise ValueError with custom message

    except Exception as e: # Catch any other exception to be sure.
        print(f"An unexpected error occurred: {e}")

    return x_train_norm, x_test_norm, x_val_norm, y_train, y_test, y_val
    
        
        
