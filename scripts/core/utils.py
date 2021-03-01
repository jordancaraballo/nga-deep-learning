# --------------------------------------------------------------------------
# Core functions to train on NGA data.
# --------------------------------------------------------------------------
import glob              # get global files from directory
import random            # for random integers
from tqdm import tqdm    # for progress bar
import numpy as np       # for arrays modifications
import cupy as cp        # for arrays modifications
import tensorflow as tf  # deep learning framework
import logging           # for buil-in logging
import scipy.signal      # for postprocessing
import math              # for math calculations

# Has a bug and will be included when bug is fixed.
# from cuml.dask.preprocessing import OneHotEncoder, LabelBinarizer

# For generating one-hot encoder labels
from datetime import datetime
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import TensorBoard, CSVLogger

# --------------------------------------------------------------------------
# Preprocessing Functions
# --------------------------------------------------------------------------

def image_normalize(img, axis=(0, 1), c=1e-8):
    """
    Normalize to zero mean and unit standard deviation along the given axis.
    Args:
        img (numpy or cupy): array (w, h, c)
        axis (integer tuple): into or tuple of width and height axis
        c (float): epsilon to bound given std value
    Return:
        Normalize single image
    ----------
    Example
    ----------
        image_normalize(arr, axis=(0, 1), c=1e-8)
    """
    return (img - img.mean(axis)) / (img.std(axis) + c)


def batch_normalize(batch, axis=(0, 1), c=1e-8):
    """
    Normalize batch to zero mean and unit standard deviation.
    Args:
        img (numpy or cupy): array (n, w, h, c)
        axis (integer tuple): into or tuple of width and height axis
        c (float): epsilon to bound given std value
    Return:
        Normalize batch of images.
    ----------
    Example
    ----------
        batch_normalize(arr, axis=(0, 1), c=1e-8)
    """
    # Note: for loop was proven to be faster than map method
    for b in range(batch.shape[0]):
        batch[b, :, :, :] = image_normalize(batch[b, :, :, :], axis=axis, c=c)
    return batch


def gen_data_npz(fimg, img, mask, config, ntiles=1000, save_dir='train'):
    """
    Extract random patches from cupy arrays.
    Args:
        fimg (str): data filename
        img (cupy.array): cupy array with data
        mask (cupy.array): cupy array with mask
        save_dir (str): directory to save output
    Return:
        save dataset to save_dir.
    ----------
    Example
    ----------
        gen_data_npz('image.tif', arr, mask, config, 8000, 'output')
    """
    # set dimensions of the input image array, and get desired tile size
    z_dim, x_dim, y_dim = img.shape
    tsz = config.TILE_SIZE

    # placeholders for final datasets
    img_cp = cp.empty((ntiles, tsz, tsz, z_dim), dtype=cp.float32)
    mask_np = np.empty((ntiles, tsz, tsz, config.N_CLASSES), dtype=np.float16)

    # generate n number of tiles
    for i in tqdm(range(ntiles)):

        # Generate random integers from image
        xc = random.randint(0, x_dim - tsz)
        yc = random.randint(0, y_dim - tsz)

        # verify data is not on nodata region
        while cp.any(
            img[:, xc:(xc + tsz), yc:(yc + tsz)] == config.NODATA_VAL
        ):
            xc = random.randint(0, x_dim - tsz)
            yc = random.randint(0, y_dim - tsz)

        # change order to (h, w, c)
        tile_img = cp.moveaxis(
            img[:, xc:(xc + tsz), yc:(yc + tsz)], 0, -1
        )

        # TODO: replace with cuml One-hot encoder on future date when they fix
        # a bug on the output types. Using to_categorical in the meantime
        # Converts labels into one-hot encoding labels
        tile_mask = to_categorical(
            cp.asnumpy(mask[xc:(xc + tsz), yc:(yc + tsz)]),
            num_classes=config.N_CLASSES, dtype='float16'
        )

        # maybe standardize here? depends on performance of single img vs batch
        img_cp[i, :, :, :] = tile_img
        mask_np[i, :, :, :] = tile_mask

    # normalize
    if config.NORMALIZE:
        img_cp = img_cp / config.normalization_factor

    # standardize
    if config.STANDARDIZE:
        img_cp = batch_normalize(img_cp, axis=(0, 1), c=1e-8)

    # save dataset into local disk, npz format with x and y labels
    cp.savez(f'{save_dir}{fimg[:-4]}.npz', x=img_cp, y=cp.asarray(mask_np))


# --------------------------------------------------------------------------
# Training Functions
# --------------------------------------------------------------------------

def get_tensorslices(data_dir='', img_id='x', label_id='y'):
    """
    Getting tensor slices from disk.
    Args:
        data_dir (str): directory where data resides
        img_id (str): object id from npz file to get data from
        label_id (str): object id from npz file to get labels from
    Return:
        get image and label datasets
    ----------
    Example
    ----------
        get_tensorslices(data_dir='images', img_id='x', label_id='y')
    """
    # open files and generate training dataset
    images = np.array([])
    labels = np.array([])

    # read all data files from disk
    for f in glob.glob(f'{data_dir}/*'):
        with np.load(f) as data:

            if images.size:
                images = np.vstack([images, data[img_id]])
            else:
                images = data[img_id]

            if labels.size:
                labels = np.vstack([labels, data[label_id]])
            else:
                labels = data[label_id]

    return images, labels


def data_augment(image, label):
    """
    Augment data for semantic segmentation.
    Args:
        image (numpy.array): image numpy array
        label (numpy.array): image numpy array
    Return:
        augmented image and label
    ----------
    Example
    ----------
        data_augment(image, label)
    """
    # Thanks to the dataset.prefetch(AUTO) statement in the next function
    # (below), this happens essentially for free on TPU. Data pipeline code
    # is executed on the CPU part of the TPU, TPU is computing gradients.
    randint = np.random.randint(1, 7)
    if randint == 1:  # flip left and right
        image = tf.image.random_flip_left_right(image)
        label = tf.image.random_flip_left_right(label)
    elif randint == 2:  # reverse second dimension
        image = tf.image.random_flip_up_down(image)
        label = tf.image.random_flip_up_down(label)
    elif randint == 3:  # rotate 90 degrees
        image = tf.image.rot90(image, k=1)
        label = tf.image.rot90(label, k=1)
    elif randint == 4:  # rotate 180 degrees
        image = tf.image.rot90(image, k=2)
        label = tf.image.rot90(label, k=2)
    elif randint == 5:  # rotate 270 degrees
        image = tf.image.rot90(image, k=3)
        label = tf.image.rot90(label, k=3)
    return image, label


def get_training_dataset(dataset, config, do_aug=False, drop_remainder=False):
    dataset = dataset.map(data_augment, num_parallel_calls=config.AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(config.BATCH_SIZE, drop_remainder=drop_remainder)
    # prefetch next batch while training (autotune prefetch buffer size)
    dataset = dataset.prefetch(config.AUTOTUNE)
    return dataset


def get_validation_dataset(dataset, config):
    # dataset.batch(BATCH_SIZE).cache().repeat() #take(BATCH_SIZE).cache().repeat()
    # dataset = dataset.repeat()
    # dataset = dataset.batch(BATCH_SIZE)
    # dataset = dataset.cache()
    dataset.batch(config.BATCH_SIZE)
    dataset = dataset.cache().repeat()
    dataset = dataset.prefetch(config.AUTOTUNE)
    return dataset


def gen_callbacks(config, metadata):
    """
    Generate tensorflow callbacks.
    Args:
        config (Config): object with configurations
        metadata (dict): directory with callback metadata values
    Return:
        augmented image and label
    ----------
    Example
    ----------
        gen_callbacks(config, metadata)
    """
    # Generating tensorboard callbacks
    tensor = TensorBoard(
        log_dir=config.MODEL_SAVEDIR, write_graph=True,
        histogram_freq=metadata['history_freq']
    )

    # initialize model csv logger callback
    csv_outfile = config.MODEL_OUTPUT_NAME[:-3] + '_' + \
        datetime.now().strftime("%Y%m%d-%H%M%S")+'.csv'
    csvlog = CSVLogger(csv_outfile, append=True, separator=';')

    # initialize model early stopping callback
    early_stop = EarlyStopping(
        patience=metadata['patience_earlystop'],
        monitor=metadata['monitor_earlystop']
    )

    # initialize model checkpoint callback
    checkpoint = ModelCheckpoint(
        config.MODEL_OUTPUT_NAME[:-3]+'_{epoch:02d}.h5',
        save_best_only=metadata['save_best_only'],
        period=metadata['period_checkpoint'],
        verbose=2
    )
    return [csvlog, early_stop, checkpoint, tensor]


# --------------------------------------------------------------------------
# Prediction Functions
# --------------------------------------------------------------------------

def pad_image(img, target_size):
    """
    Pad an image up to the target size.
    """
    rows_missing = target_size - img.shape[0]
    cols_missing = target_size - img.shape[1]
    padded_img = np.pad(
        img, ((0, rows_missing), (0, cols_missing), (0, 0)), 'constant'
        )
    return padded_img


def predict_windowing(x, model, config, spline):

    print("entrando windowing ", x.shape)

    img_height = x.shape[0]
    img_width = x.shape[1]
    n_channels = x.shape[2]

    # make extended img so that it contains integer number of patches
    npatches_vertical = math.ceil(img_height / config.TILE_SIZE)
    npatches_horizontal = math.ceil(img_width / config.TILE_SIZE)
    extended_height = config.TILE_SIZE * npatches_vertical
    extended_width = config.TILE_SIZE * npatches_horizontal
    ext_x = np.zeros(
        shape=(extended_height, extended_width, n_channels), dtype=np.float32
    )

    # fill extended image with mirrors:
    ext_x[:img_height, :img_width, :] = x
    for i in range(img_height, extended_height):
        ext_x[i, :, :] = ext_x[2 * img_height - i - 1, :, :]
    for j in range(img_width, extended_width):
        ext_x[:, j, :] = ext_x[:, 2 * img_width - j - 1, :]

    # now we assemble all patches in one array
    patches_list = []  # do vstack later instead of list
    for i in range(0, npatches_vertical):
        for j in range(0, npatches_horizontal):
            x0, x1 = i * config.TILE_SIZE, (i + 1) * config.TILE_SIZE
            y0, y1 = j * config.TILE_SIZE, (j + 1) * config.TILE_SIZE
            patches_list.append(ext_x[x0:x1, y0:y1, :])

    patches_array = np.asarray(patches_list)
    # print("array the pathcews ", patches_array.shape)

    # standardize
    if config.STANDARDIZE:
        patches_array = batch_normalize(patches_array, axis=(0, 1), c=1e-8)

    # if config.STANDARDIZE:
    #    for i in range(config.N_CHANNELS):
    #        patches_array[:, :, :, i] = \
    #            (patches_array[:, :, :, i] - config.channel_mean[i]) / \
    #            config.channel_std[i]

    print("array the patches despues de norm", patches_array.shape)
    # predictions:
    patches_predict = \
        model.predict(patches_array, batch_size=config.PRED_BSIZE)

    prediction = np.zeros(
        shape=(extended_height, extended_width, config.N_CLASSES),
        dtype=np.float32
    )

    logging.info("prediction shape: ", prediction.shape)
    for k in range(patches_predict.shape[0]):
        i = k // npatches_horizontal
        j = k % npatches_horizontal
        x0, x1 = i * config.TILE_SIZE, (i + 1) * config.TILE_SIZE
        y0, y1 = j * config.TILE_SIZE, (j + 1) * config.TILE_SIZE
        prediction[x0:x1, y0:y1, :] = patches_predict[k, :, :, :] * spline
    return prediction[:img_height, :img_width, :]


def predict_sliding(image, model, config, spline):
    """
    Predict on tiles of exactly the network input shape.
    This way nothing gets squeezed.
    """
    stride = math.ceil(config.TILE_SIZE * (1 - config.PRED_OVERLAP))
    tile_rows = max(
        int(math.ceil((image.shape[0] - config.TILE_SIZE) / stride) + 1), 1
        )  # strided convolution formula
    tile_cols = max(
        int(math.ceil((image.shape[1] - config.TILE_SIZE) / stride) + 1), 1
        )
    logging.info("Need %i x %i prediction tiles @ stride %i px" %
                 (tile_cols, tile_rows, stride)
                 )

    full_probs = np.zeros((image.shape[0], image.shape[1], config.N_CLASSES))
    count_predictions = \
        np.zeros((image.shape[0], image.shape[1], config.N_CLASSES))

    tile_counter = 0
    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * stride)
            y1 = int(row * stride)
            x2 = min(x1 + config.TILE_SIZE, image.shape[1])
            y2 = min(y1 + config.TILE_SIZE, image.shape[0])
            x1 = max(int(x2 - config.TILE_SIZE), 0)
            y1 = max(int(y2 - config.TILE_SIZE), 0)

            img = image[y1:y2, x1:x2]
            padded_img = pad_image(img, config.TILE_SIZE)
            tile_counter += 1

            padded_img = np.expand_dims(padded_img, 0)

            # standardize
            if config.STANDARDIZE:
                padded_img = batch_normalize(padded_img, axis=(0, 1), c=1e-8)

            #    for i in range(config.N_CHANNELS):
            #        padded_img[:, :, :, i] = \
            #            (padded_img[:, :, :, i] - config.channel_mean[i]) / \
            #            config.channel_std[i]

            imgn = padded_img
            imgn = imgn.astype('float32')

            padded_prediction = model.predict(imgn)[0]
            prediction = padded_prediction[0:img.shape[0], 0:img.shape[1], :]
            count_predictions[y1:y2, x1:x2] += 1
            full_probs[y1:y2, x1:x2] += prediction * spline

    # average the predictions in the overlapping regions
    full_probs /= count_predictions
    return full_probs


def _2d_spline(window_size=128, power=2) -> np.array:
    """
    Window method for boundaries/edge artifacts smoothing.
    :param window_size: size of window/tile to smooth
    :param power: spline polinomial power to use
    :return: smoothing distribution numpy array
    """
    intersection = int(window_size/4)
    tria = scipy.signal.triang(window_size)
    wind_outer = (abs(2*(tria)) ** power)/2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2*(tria - 1)) ** power)/2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    wind = np.expand_dims(np.expand_dims(wind, 1), 2)
    wind = wind * wind.transpose(1, 0, 2)
    return wind
