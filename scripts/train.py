# --------------------------------------------------------------------------
# Preprocessing and dataset creation from NGA data. This assumes you provide
# a configuration file with required parameters and files.
# --------------------------------------------------------------------------
import os                # system modifications
import sys               # system modifications
import glob              # get global files from directory
import time              # tracking time
import numpy as np       # for arrays modifications
import cupy as cp        # for arrays modifications
import tensorflow as tf  # deep learning framework

from datetime import datetime                # time library for filenames
from core.unet import unet_batchnorm         # unet network to work with
from core.utils import get_training_dataset  # getting training dataset
from core.utils import get_tensorslices      # getting tensor slices
from core.utils import gen_callbacks         # generate callbacks

# tensorflow imports
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adadelta

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
# script train.py
# ---------------------------------------------------------------------------

def main():

    # Main function to collect configuration file and run the script
    print(f'GPU REPLICAS: {strategy.num_replicas_in_sync}')
    t0 = time.time()

    print(f'Train dir: {config.TRAIN_DATADIR}')
    print(f'Validation dir: {config.VAL_DATADIR}')

    # Initialize Callbacks
    callbacks = gen_callbacks(config, config.CALLBACKS_METADATA)

    # open files and get dataset tensor slices
    train_images, train_labels = get_tensorslices(
        data_dir=config.TRAIN_DATADIR, img_id='x', label_id='y'
    )

    # open files and get dataset tensor slices
    val_images, val_labels = get_tensorslices(
        data_dir=config.VAL_DATADIR, img_id='x', label_id='y'
    )

    # extract values for training
    NUM_TRAINING_IMAGES = train_images.shape[0]
    NUM_VALIDATION_IMAGES = val_images.shape[0]
    STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // config.BATCH_SIZE

    print(f'{NUM_TRAINING_IMAGES} training images'
    print(f'{NUM_VALIDATION_IMAGES} validation images')

    # generate training dataset
    train_dataset = \
        tf.data.Dataset.from_tensor_slices((train_images, train_labels))

    # Create model output directory
    os.system(f'mkdir -p {config.MODEL_SAVEDIR}')

    # Initialize and compile model
    with strategy.scope():

        # initialize UNet model
        model = unet_batchnorm(
            nclass=config.N_CLASSES, input_size=config.INPUT_SIZE,
            maps=config.MODEL_METADATA['network_maps']
        )

        # initialize optimizer, exit of not valid optimizer
        if config.MODEL_METADATA['optimizer_name'] == 'Adadelta':
            optimizer = Adadelta(lr=config.MODEL_METADATA['lr'])
        elif config.MODEL_METADATA['optimizer_name'] == 'Adam':
            optimizer = Adam(lr=config.MODEL_METADATA['lr'])
        else:
            sys.exit('Optimizer provided is not supported.')

        # compile model to start training
        model.compile(
            optimizer,
            loss=config.MODEL_METADATA['loss'],
            metrics=config.MODEL_METADATA['metrics']
        )
        model.summary()

    # Train the model and save to disk
    model.fit(
        get_training_dataset(train_dataset, config, do_aug=True),
        initial_epoch=config.START_EPOCH,
        epochs=config.N_EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=(val_images, val_labels),
        callbacks=callbacks,
        verbose=2
    )

    print(f'Execution time: {time.time() - t0}')


# -------------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------------

if __name__ == "__main__":

    main()
