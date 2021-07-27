# --------------------------------------------------------------------------
# Configuration of the parameters for training and preprocessing
# NGA dataset using deep learning techniques.
# --------------------------------------------------------------------------
import glob              # get global files from directory
import tensorflow as tf  # deep learning framework

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Development"
__date__ = "02/10/2021"


class Configuration:

    def __init__(self):

        # ------------------------------------------------------------------
        # System configurations
        # ------------------------------------------------------------------
        # self.SEED: define static random seed value reproducibility
        self.SEED = 42

        # self.CUDA: define CUDA devices to use
        self.CUDA = "0,1"

        # self.FOLDS: define folds for batch iterations
        self.FOLDS = 3

        # ------------------------------------------------------------------
        # Mixed Precision, XLA, Multi-GPU strategies
        # ------------------------------------------------------------------
        # self.MIRROR_STRATEGY: multi-gpu strategy
        self.MIRROR_STRATEGY = True

        # self.MIXED_PRECISION: TF mixed precision enable
        self.MIXED_PRECISION = True

        # self.XLA_ACCELERATE: XLA acceleration enabled
        self.XLA_ACCELERATE = False

        # self.AUTOTUNE: tensorflow autotuning feature enabled
        # For more information about autotune:
        # https://www.tensorflow.org/guide/data_performance#prefetching
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE

        # ------------------------------------------------------------------
        # General imagery variables
        # ------------------------------------------------------------------
        # self.NODATA_VAL: no data value present on the imagery
        self.NODATA_VAL = 9999

        # self.DASK_SIZE: directory for data structure out of memory mapping
        self.DASK_SIZE = {'band': 1, 'x': 2048, 'y': 2048}

        # self.INIT_BANDS: bands available in the original data
        self.INIT_BANDS = \
            ['Blue', 'Green', 'Red']

        # self.N_CLASSES: number of classes that will be classified
        # An example of this: tree, shadow, cloud, background = 4 classes
        self.N_CLASSES = 5

        # self.TILE_SIZE: tile size to feed the model
        self.TILE_SIZE = 128

        # ------------------------------------------------------------------
        # Preprocessing
        # ------------------------------------------------------------------
        # self.PREP_DATA_FILE: includes the raster filenames to use for train
        # Look at the README for more details on what this file includes.
        self.PREP_DATA_FILE = 'config/data.csv'

        # self.PREP_DATA_INPDIR: directory where GeoTIF data rasters reside
        self.PREP_DATA_INPDIR = '/att/nobackup/jacaraba/nga-deeplearning-data/data'

        # self.PREP_LABELS_INPDIR: directory where GeoTIF label rasters reside
        self.PREP_LABELS_INPDIR = '/att/nobackup/jacaraba/nga-deeplearning-data/labels'

        # self.PREP_BANDS_OUTPUT: bands to output from data to train
        self.PREP_BANDS_OUTPUT = \
            ['Blue', 'Green', 'Red']

        # self.PREP_ROOT_OUTDIR: root directory that leads to data subdirs
        self.PREP_ROOT_OUTDIR = '/att/nobackup/jacaraba/nga-deeplearning-data/output'

        # self.PREP_TRAIN_OUTDIR: output directory to store training data
        self.PREP_TRAIN_OUTDIR = self.PREP_ROOT_OUTDIR + '/train'

        # self.PREP_VAL_OUTDIR: output directory to store validation data
        self.PREP_VAL_OUTDIR = self.PREP_ROOT_OUTDIR + '/val'

        # self.NORMALIZE: bool value to apply normalization to the dataset
        self.NORMALIZE = True

        # self.normalization_factor: float value to accomply normalization
        self.normalization_factor = 255.0  # 65535.0

        # self.STANDARDIZE: bool value to apply standardization to the dataset
        self.STANDARDIZE = True

        # self.INDICES_FACTOR: float value to calculate additional bands
        # related to TOA values generated from EVHR
        self.INDICES_FACTOR = 10000.0

        # ------------------------------------------------------------------
        # Training
        # ------------------------------------------------------------------
        # self.TRAIN_BANDS_INPUT: bands that will be sent as input for train
        self.TRAIN_BANDS_INPUT = self.PREP_BANDS_OUTPUT

        # Defining model variables

        # self.N_CHANNELS: number of channels (number of bands) to train on
        self.N_CHANNELS = len(self.TRAIN_BANDS_INPUT)

        # self.INPUT_SIZE: input size to feed neural network
        self.INPUT_SIZE = (self.TILE_SIZE, self.TILE_SIZE, self.N_CHANNELS)

        # self.START_EPOCH: epoch to start from training
        self.START_EPOCH = 0

        # self.N_EPOCHS: number of epochs to run for
        self.N_EPOCHS = 1000

        # self.BATCH_SIZE: batch size to feed the network
        # 128 bsize works well with 32GB RAM GPU
        self.BATCH_SIZE = 128
        self.VAL_BATCH_SIZE = 128

        # self.TRAIN_ROOT_INPDIR: root dir where train and val data resides
        self.TRAIN_ROOT_INPDIR = self.PREP_ROOT_OUTDIR

        # self.TRAIN_DATADIR: directory where train data resides
        self.TRAIN_DATADIR = self.PREP_TRAIN_OUTDIR

        # self.VAL_DATADIR: directory where val data resides
        self.VAL_DATADIR = self.PREP_VAL_OUTDIR

        # Training hyper-parameters
        self.MODEL_METADATA = {
            'id': '100',
            'network': 'unet',
            'network_maps': [64, 128, 256, 512, 1024],
            'lr': 0.0001,
            'momentum': 0.90,
            'gradient': 0.95,
            'loss': 'categorical_crossentropy',
            'optimizer_name': 'Adadelta',
            'metrics': ['accuracy'],
            'do_aug': True
        }

        # Model description and output directory

        # self.MODEL_NAME: model name to train
        self.MODEL_NAME = '100_unet_mle5_128'

        # self.MODEL_SAVEDIR: directory to save trained models
        self.MODEL_SAVEDIR = f'{self.TRAIN_ROOT_INPDIR}/unet/{self.MODEL_NAME}'

        # self.MODEL_OUTPUT_NAME: output name for model to be saved
        self.MODEL_OUTPUT_NAME = f'{self.MODEL_SAVEDIR}/{self.MODEL_NAME}.h5'

        # Training extensions - Callbacks
        self.CALLBACKS = \
            ['ModelCheckpoint', 'CSVLogger', 'EarlyStopping', 'TensorBoard']

        self.CALLBACKS_METADATA = {
            'patience_earlystop': 20,
            'patience_plateu': 5,
            'monitor_earlystop': 'val_loss',
            'monitor_checkpoint': 'val_loss',
            'save_freq': 'epoch',
            'factor_plateu': 0.20,
            'min_lr_plateu': 0.00,
            'history_freq': 5,
            'save_best_only': True
        }

        # ------------------------------------------------------------------
        # Prediction
        # ------------------------------------------------------------------
        # self.PRED_BANDS_INPUT: bands available in the original data
        self.PRED_BANDS_INPUT = self.INIT_BANDS

        # self.PRED_BANDS_OUTPUT: number of bands to feed model
        # if the number of bands is greater, indices need to be calculated
        self.PRED_BANDS_OUTPUT = self.PREP_BANDS_OUTPUT

        # self.PRED_BSIZE: batch size for concurrent imagery predictions
        self.PRED_BATCH_SIZE = 128
        self.PROBABILITIES = True

        # self.PRED_FILENAMES: regex to flag all files to predict
        self.PRED_FILENAMES = '/att/nobackup/jacaraba/nga-deeplearning-data/test/*.TIF'
        self.PRED_FILENAMES = glob.glob(self.PRED_FILENAMES)

        # self.PRED_OVERLAP: overlap between tiles to predict
        self.PRED_OVERLAP = 40 # 0.25

        # self.PRED_WINDOW_SIZE: greatest window size that can fit into memory
        self.PRED_WINDOW_SIZE = [8192, 8192]

        # self.MODEL_STR: specific model to open based on epoch
        self.MODEL_STR = f"{self.MODEL_NAME}_431.h5"

        # self.MODEL_NAME: model path and model name to open
        self.MODEL_NAME = f'{self.MODEL_SAVEDIR}/{self.MODEL_STR}'

        # self.PRED_SAVE_DIR: directory to save GeoTIF predictions on
        self.PRED_SAVE_DIR = \
            f'{self.PREP_ROOT_OUTDIR}/predictions/{self.MODEL_STR}/images'

        # self.PRED_SEG_SAVE_DIR: directory to save numpy predictions on
        self.PRED_SEG_SAVE_DIR = \
            f'{self.PREP_ROOT_OUTDIR}/predictions/{self.MODEL_STR}/segments'
