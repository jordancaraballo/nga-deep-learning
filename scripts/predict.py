# --------------------------------------------------------------------------
# Prediction from TF model and NGA data. This assumes you provide
# a configuration file with required parameters and files.
# --------------------------------------------------------------------------
import os                # system modifications
import time              # tracking time
from tqdm import tqdm    # for local progress bar
import numpy as np       # for arrays modifications
import cupy as cp        # for arrays modifications
import tensorflow as tf  # deep learning framework
import xarray as xr      # read rasters

# tensorflow imports
# from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import load_model

# core library imports
from core.utils import _2d_spline, arr_to_tif
from core.utils import predict_all, predict_sliding_probs
from core import indices

# define configuration object
from config import Config
config = Config.Configuration()

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Development"

# Define some environment variables to help refining randomness.
# Note: there might still be some randomness since most of the code
# is ran on GPU and sometimes parallelization brings changes.
np.random.seed(config.SEED)
tf.random.set_seed(config.SEED)
cp.random.seed(config.SEED)

print(f"Tensorflow ver. {tf.__version__}")

# verify GPU devices are available and ready
os.environ['CUDA_VISIBLE_DEVICES'] = config.CUDA
devices = tf.config.list_physical_devices('GPU')
assert len(devices) != 0, "No GPU devices found."

# ------------------------------------------------------------------
# System Configurations
# ------------------------------------------------------------------
if config.MIXED_PRECISION:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print('Mixed precision enabled')

if config.XLA_ACCELERATE:
    tf.config.optimizer.set_jit(True)
    print('Accelerated Linear Algebra enabled')

# set memory growth to infinite to account for multiple devices
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for pd in physical_devices:
    configTF = tf.config.experimental.set_memory_growth(pd, True)


# ---------------------------------------------------------------------------
# script predict.py
# ---------------------------------------------------------------------------
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
            if x1 - x0 < config.TILE_SIZE:  # if x is smaller than tsize
                x0 = x1 - config.TILE_SIZE  # assign boundary to -tsize
            if y1 - y0 < config.TILE_SIZE:  # if selected y is small than tsize
                y0 = y1 - config.TILE_SIZE  # assign boundary to -tsize

            window = x_data[x0:x1, y0:y1, :].values  # get window
            window = cp.asarray(window)

            window[window < 0] = 0  # remove lower bound values
            window[window > 10000] = 10000  # remove higher bound values

            if config.NORMALIZE:
                window = window / config.normalization_factor
            window = cp.asnumpy(window)

            # perform sliding window prediction
            prediction[x0:x1, y0:y1] = \
                predict_all(window, model, config, spline=spline)

    return prediction


def main():

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
            os.path.join(config.PRED_SAVE_DIR, fname[:-4].split('/')[-1] + '_pred.tif')

        save_segment = \
            os.path.join(config.PRED_SEG_SAVE_DIR, fname[:-4].split('/')[-1] + '_prob.npy')
        
        os.system('mkdir -p {}'.format(config.PRED_SAVE_DIR))
        os.system('mkdir -p {}'.format(config.PRED_SEG_SAVE_DIR))

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
            x_data = x_data[:, :, :len(config.PREP_BANDS_OUTPUT)]

            # --------------------------------------------------------------------------------
            # Calculate missing indices
            # --------------------------------------------------------------------------------

            # --------------------------------------------------------------------------------
            # Getting predicted labels
            # --------------------------------------------------------------------------------

            # predict probabilities, save to local disk
            prediction = predict_sliding_probs(x_data, model, config)
            np.save(save_segment, prediction)
            
            # get final mask and save the output
            prediction = prediction.argmax(axis=-1)
            arr_to_tif(raster_f=fname, segments=prediction, out_tif=save_image)
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
