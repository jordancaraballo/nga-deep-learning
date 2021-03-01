# --------------------------------------------------------------------------
# Preprocessing and dataset creation from NGA data. This assumes you provide
# a configuration file with required parameters and files.
# --------------------------------------------------------------------------
import os                # system modifications
import time              # tracking time
import numpy as np       # for arrays modifications
import cupy as cp        # for arrays modifications
import pandas as pd      # pandas library for csv
import xarray as xr      # read rasters
import tensorflow as tf  # deep learning framework

# Enabling mixed precission
from tensorflow.keras.mixed_precision import experimental as mixed_precision

# Include core functions required
from core.utils import gen_data_npz
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
if config.MIRROR_STRATEGY:
    strategy = tf.distribute.MirroredStrategy()
    print('Multi-GPU enabled')

if config.MIXED_PRECISION:
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    print('Mixed precision enabled')

if config.XLA_ACCELERATE:
    tf.config.optimizer.set_jit(True)
    print('Accelerated Linear Algebra enabled')


# ---------------------------------------------------------------------------
# script preprocessing.py
# ---------------------------------------------------------------------------

def main():

    # Main function to collect configuration file and run the script
    print(f'GPU REPLICAS: {strategy.num_replicas_in_sync}')
    t0 = time.time()

    # Initialize dataframe with data details
    df_data = pd.read_csv(config.PREP_DATA_FILE)
    print(df_data)

    # create directory to store dataset
    os.system(f'mkdir -p {config.PREP_TRAIN_OUTDIR} {config.PREP_VAL_OUTDIR}')

    # Iterate over each training raster from df_data
    for ind in df_data.index:

        # Specify data files to read and process
        fimg = config.PREP_DATA_INPDIR + df_data['data'][ind]
        fmask = config.PREP_LABELS_INPDIR + df_data['label'][ind]
        print(f'Processing file #{ind+1}: ', df_data['data'][ind])

        # Read imagery from disk
        img = xr.open_rasterio(fimg, chunks=config.DASK_SIZE).load()
        mask = xr.open_rasterio(fmask, chunks=config.DASK_SIZE).load()
        config.NODATA_VAL = img.attrs['nodatavals'][0]

        # Map to GPU memory
        img = img.map_blocks(cp.asarray)
        mask = mask.map_blocks(cp.asarray).squeeze()
        print(f'Image and mask shape: {img.shape} {mask.shape}')

        # -------------------------------------------------------------
        # Unique processing for this project - Start
        # Might not be necessary for other projects
        # -------------------------------------------------------------

        # need to merge a class with another class?
        # mask[mask == 3] = 1  # merge thin clouds into clouds class

        # get only necessary channels for training
        img = img[:len(config.PREP_BANDS_OUTPUT), :, :]

        # accounting for contaminated pixels, TOA values
        img[img < 0] = 0  # remove lower bound values
        img[img > 10000] = 10000  # remove higher bound values

        # adding indices if required, only if data does not come with
        # dedicated indices and need to be calculated on the fly.

        # fdi = \
        # indices.fdi(img, BANDS_INPUT, factor=INDICES_FACTOR, vtype='int16')
        # si = \
        # indices.si(img, BANDS_INPUT, factor=INDICES_FACTOR, vtype='int16')
        # ndwi = \
        # indices.ndwi(img, BANDS_INPUT, factor=INDICES_FACTOR, vtype='int16')

        # concatenate all indices
        # img = cp.concatenate((img, fdi, si, ndwi), axis=0)
        # print("Image and mask shape after indices: ", img.shape, mask.shape)

        # -------------------------------------------------------------
        # Unique processing for this project - End
        # -------------------------------------------------------------

        # Generate training data
        gen_data_npz(
            fimg=df_data['data'][ind],
            img=img,
            mask=mask,
            config=config,
            ntiles=df_data['ntiles_train'][ind],
            save_dir=config.PREP_TRAIN_OUTDIR
        )

        # Generate validation data
        gen_data_npz(
            fimg=df_data['data'][ind],
            img=img,
            mask=mask,
            config=config,
            ntiles=df_data['ntiles_val'][ind],
            save_dir=config.PREP_VAL_OUTDIR
        )

    print(f'Execution time: {time.time() - t0}')


# -------------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------------

if __name__ == "__main__":

    main()
