import os
import glob
import time
from tqdm import tqdm
import gc

import numpy as np
import cupy as cp
import rasterio as rio
import xarray as xr

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.mixed_precision import experimental as mixed_precision

# from xrasterlib.dl.processing import normalize
from core.utils import _2d_spline
from core.utils import predict_windowing, predict_sliding
# from core import indices

# define configuration object
from config import ConfigTemplate
config = ConfigTemplate.Configuration()

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Development"

# Define some environment variables to help refining randomness.
# Note: there might still be some randomness since most of the code
# is ran on GPU and sometimes parallelization brings changes.
np.random.seed(config.SEED)
tf.random.set_seed(config.SEED)
cp.random.seed(config.SEED)

# For more information about autotune:
# https://www.tensorflow.org/guide/data_performance#prefetching
AUTOTUNE = tf.data.experimental.AUTOTUNE
print(f"Tensorflow ver. {tf.__version__}")

# ------------------------------------------------------------------
# System Configurations
# ------------------------------------------------------------------
# if config.MIRROR_STRATEGY:
#    strategy = tf.distribute.MirroredStrategy()

if config.MIXED_PRECISION:
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    print('Mixed precision enabled')

if config.XLA_ACCELERATE:
    tf.config.optimizer.set_jit(True)
    print('Accelerated Linear Algebra enabled')

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for pd in physical_devices:
    configTF = tf.config.experimental.set_memory_growth(pd, True)


# ---------------------------------------------------------------------------
# script train.py
#
# Train model using CNN. Data Source: NGA Vietnam dataset.
# ---------------------------------------------------------------------------

def npy_to_tif(raster_f='image.tif', segments='segment.npy',
               outtif='segment.tif', ndval=-9999
               ):
    """
    Args:
        raster_f:
        segments:
        outtif:
    Returns:
    """
    # get geospatial profile, will apply for output file
    with rio.open(raster_f) as src:
        meta = src.profile
        nodatavals = src.read_masks(1).astype('int16')
    print(meta)

    # load numpy array if file is given
    if type(segments) == str:
        segments = np.load(segments)
    segments = segments.astype('int16')
    print(segments.dtype)  # check datatype

    nodatavals[nodatavals == 0] = ndval
    segments[nodatavals == ndval] = nodatavals[nodatavals == ndval]

    out_meta = meta  # modify profile based on numpy array
    out_meta['count'] = 1  # output is single band
    out_meta['dtype'] = 'int16'  # data type is float64

    # write to a raster
    with rio.open(outtif, 'w', **out_meta) as dst:
        dst.write(segments, 1)


def predict_all(x_data, model, config, spline):

    for i in range(8):
        if i == 0:  # reverse first dimension
            x_seg = predict_windowing(x_data[::-1, :, :], model, config, spline=spline).transpose([2, 0, 1])
            gc.collect()
        elif i == 1:  # reverse second dimension
            temp = predict_windowing(x_data[:, ::-1, :], model, config, spline=spline).transpose([2, 0, 1])
            x_seg = temp[:, ::-1, :] + x_seg
            gc.collect()
        elif i == 2:  # transpose(interchange) first and second dimensions
            temp = predict_windowing(x_data.transpose([1, 0, 2]), model, config, spline=spline).transpose([2, 0, 1])
            x_seg = temp.transpose(0, 2, 1) + x_seg
            gc.collect()
        elif i == 3:
            temp = predict_windowing(np.rot90(x_data, 1), model, config, spline=spline)
            x_seg = np.rot90(temp, -1).transpose([2, 0, 1]) + x_seg
            gc.collect()
        elif i == 4:
            temp = predict_windowing(np.rot90(x_data, 2), model, config, spline=spline)
            x_seg = np.rot90(temp, -2).transpose([2, 0, 1]) + x_seg
            gc.collect()
        elif i == 5:
            temp = predict_windowing(np.rot90(x_data, 3), model, config, spline=spline)
            x_seg = np.rot90(temp, -3).transpose(2, 0, 1) + x_seg
            gc.collect()
        elif i == 6:
            temp = predict_windowing(x_data, model, config, spline=spline).transpose([2, 0, 1])
            x_seg = temp + x_seg
            gc.collect()
        elif i == 7:
            temp = predict_sliding(x_data, model, config, spline=spline).transpose([2, 0, 1])
            x_seg = temp + x_seg
            gc.collect()

    del x_data, temp  # delete arrays
    x_seg /= 8.0
    return x_seg.argmax(axis=0)


def predict(x_data, model, config, spline, normalize=True, standardize=True):

    # open rasters and get both data and coordinates
    rast_shape = x_data[:, :, 0].shape  # shape of the wider scene

    # in memory sliding window predictions
    wsx, wsy = config.PRED_WINDOW_SIZE[0], config.PRED_WINDOW_SIZE[1]

    # if the window size is bigger than the image, predict full image
    if wsx > rast_shape[0]:
        wsx = rast_shape[0]
    if wsy > rast_shape[1]:
        wsy = rast_shape[1]

    prediction = np.zeros(rast_shape)  # crop out the window
    print(f'wsize: {wsx}x{wsy}. Prediction shape: {prediction.shape}')

    for sx in tqdm(range(0, rast_shape[0], wsx)):  # iterate over x-axis
        for sy in range(0, rast_shape[1], wsy):  # iterate over y-axis
            x0, x1, y0, y1 = sx, sx + wsx, sy, sy + wsy  # assign window
            if x1 > rast_shape[0]:  # if selected x exceeds boundary
                x1 = rast_shape[0]  # assign boundary to x-window
            if y1 > rast_shape[1]:  # if selected y exceeds boundary
                y1 = rast_shape[1]  # assign boundary to y-window
            if x1 - x0 < config.TILE_SIZE:  # if selected x is smaller than tsize
                x0 = x1 - config.TILE_SIZE  # assign boundary to -tsize
            if y1 - y0 < config.TILE_SIZE:  # if selected y is small than tsize
                y0 = y1 - config.TILE_SIZE  # assign boundary to -tsize

            window = x_data[x0:x1, y0:y1, :].values  # get window
            window = cp.asarray(window)

            window[window < 0] = 0  # remove lower bound values
            window[window > 10000] = 10000  # remove higher bound values

            # adding indices
            # window = cp.transpose(window, (2, 0, 1))
            # fdi = indices.fdi(window, config.PRED_BANDS_INPUT, factor=config.INDICES_FACTOR, vtype='int16')
            # si = indices.si(window, config.PRED_BANDS_INPUT, factor=config.INDICES_FACTOR, vtype='int16')
            # ndwi = indices.ndwi(window, config.PRED_BANDS_INPUT, factor=config.INDICES_FACTOR, vtype='int16')
            # print(fdi.shape, si.shape, ndwi.shape, window.shape)

            # concatenate all indices
            # window = cp.concatenate((window, fdi, si, ndwi), axis=0)
            # window = cp.transpose(window, (1, 2, 0))

            if normalize is True:
                window = window / config.normalization_factor

            window = cp.asnumpy(window)
            print("Window shape", window.shape)

            print(x0, x1, y0, y1, window.shape)

            # perform sliding window prediction
            prediction[x0:x1, y0:y1] = \
                predict_all(window, model, config, spline=spline)
    return prediction


def main():

    # Main function to collect configuration file and run the script
    # print(f'GPU REPLICAS: {strategy.num_replicas_in_sync}')

    # Generate 2-dimensional spline to avoid boundary problems
    spline = _2d_spline(config.TILE_SIZE, power=2)

    # Loading the trained model
    model = load_model(config.MODEL_NAME)
    model.summary()  # print summary of the model

    # Get list of files to predict
    print(f'Number of files to predict: {len(config.PRED_FILENAMES)}')

    # Tterate over files and predict them
    for fname in config.PRED_FILENAMES:

        # measure execution time
        start_time = time.perf_counter()

        # path + name to store prediction into
        save_image = \
            config.PRED_SAVE_DIR + fname[:-4].split('/')[-1] + '_pred.tif'

        # --------------------------------------------------------------------------------
        # if prediction is not on directory, start predicting
        # (allows for restarting script if it was interrupted at some point)
        # --------------------------------------------------------------------------------
        if not os.path.isfile(save_image):

            print(f'Starting to predict {fname}')

            # --------------------------------------------------------------------------------
            # Extracting and resizing test and validation data
            # --------------------------------------------------------------------------------
            x_data = xr.open_rasterio(fname, chunks=config.DASK_SIZE)
            x_data = x_data.transpose("y", "x", "band")
            x_data = x_data[:, :, :4]

            # --------------------------------------------------------------------------------
            # Calculate missing indices
            # --------------------------------------------------------------------------------

            # --------------------------------------------------------------------------------
            # Getting predicted labels
            # --------------------------------------------------------------------------------
            os.system('mkdir -p {}'.format(config.PRED_SAVE_DIR))
            prediction = predict(x_data, model, config, spline, normalize=config.NORMALIZE, standardize=config.STANDARDIZE)
            prediction = prediction.astype(np.int8)  # type to int16

            # --------------------------------------------------------------------------------
            # Generating visualization from prediction
            # --------------------------------------------------------------------------------
            npy_to_tif(raster_f=fname, segments=prediction, outtif=save_image)
            del prediction

        # This is the case where the prediction was already saved
        else:
            print(f'{save_image} already predicted.')

        print(f'Time: {(time.perf_counter() - start_time)}')


# -------------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------------

if __name__ == "__main__":

    main()
