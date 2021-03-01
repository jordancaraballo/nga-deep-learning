import logging
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import concatenate, Input, UpSampling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2


__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"

# ---------------------------------------------------------------------------
# module unet
#
# Build UNet NN architecture using Keras. Any of these functions can be
# called from an external script. UNets can be modified as needed.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Module Methods
# ---------------------------------------------------------------------------


# --------------------------- Convolution Functions ----------------------- #

def unet_dropout(nclass=19, input_size=(256, 256, 6), weight_file=None,
                 maps=[64, 128, 256, 512, 1024]
                 ):
    """
    UNet network using dropout features.
    """
    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(maps[0], (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(maps[0], (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)

    c2 = Conv2D(maps[1], (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(maps[1], (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)

    c3 = Conv2D(maps[2], (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(maps[2], (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D(pool_size=(2, 2))(c3)

    c4 = Conv2D(maps[3], (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(maps[3], (3, 3), activation='relu', padding='same')(c4)
    d4 = Dropout(0.5)(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(d4)

    # Squeeze
    c5 = Conv2D(maps[4], (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(maps[4], (3, 3), activation='relu', padding='same')(c5)
    d5 = Dropout(0.5)(c5)

    # Decoder
    u6 = UpSampling2D(size=(2, 2))(d5)
    m6 = concatenate([d4, u6], axis=3)
    c6 = Conv2D(maps[3], (3, 3), activation='relu', padding='same')(m6)
    c6 = Conv2D(maps[3], (3, 3), activation='relu', padding='same')(c6)

    u7 = UpSampling2D(size=(2, 2))(c6)
    m7 = concatenate([c3, u7], axis=3)
    c7 = Conv2D(maps[2], (3, 3), activation='relu', padding='same')(m7)
    c7 = Conv2D(maps[2], (3, 3), activation='relu', padding='same')(c7)

    u8 = UpSampling2D(size=(2, 2))(c7)
    m8 = concatenate([c2, u8], axis=3)
    c8 = Conv2D(maps[1], (3, 3), activation='relu', padding='same')(m8)
    c8 = Conv2D(maps[1], (3, 3), activation='relu', padding='same')(c8)

    u9 = UpSampling2D(size=(2, 2))(c8)
    m9 = concatenate([c1, u9], axis=3)
    c9 = Conv2D(maps[0], (3, 3), activation='relu', padding='same')(m9)
    c9 = Conv2D(maps[0], (3, 3), activation='relu', padding='same')(c9)

    actv = 'softmax'
    if nclass == 1:
        actv = 'sigmoid'

    conv10 = Conv2D(nclass, (1, 1), activation=actv)(c9)
    model = Model(inputs=inputs, outputs=conv10, name="UNetDropout")

    if weight_file:
        model.load_weights(weight_file)
    return model


def unet_batchnorm(nclass=19, input_size=(256, 256, 6), weight_file=None,
                   kr=l2(0.0001), maps=[64, 128, 256, 512, 1024]
                   ):
    """
    UNet network using batch normalization features.
    """
    inputs = Input(input_size, name='Input')

    # Encoder
    c1 = Conv2D(maps[0], (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(maps[0], (3, 3), activation='relu', padding='same')(c1)
    n1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(n1)

    c2 = Conv2D(maps[1], (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(maps[1], (3, 3), activation='relu', padding='same')(c2)
    n2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(n2)

    c3 = Conv2D(maps[2], (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(maps[2], (3, 3), activation='relu', padding='same')(c3)
    n3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2))(n3)

    c4 = Conv2D(maps[3], (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(maps[3], (3, 3), activation='relu', padding='same')(c4)
    n4 = BatchNormalization()(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(n4)

    # Squeeze
    c5 = Conv2D(maps[4], (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(maps[4], (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = UpSampling2D((2, 2))(c5)
    n6 = BatchNormalization()(u6)
    u6 = concatenate([n6, n4])
    c6 = Conv2D(maps[3], (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(maps[3], (3, 3), activation='relu', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    n7 = BatchNormalization()(u7)
    u7 = concatenate([n7, n3])
    c7 = Conv2D(maps[2], (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(maps[2], (3, 3), activation='relu', padding='same')(c7)

    u8 = UpSampling2D((2, 2))(c7)
    n8 = BatchNormalization()(u8)
    u8 = concatenate([n8, n2])
    c8 = Conv2D(maps[1], (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(maps[1], (3, 3), activation='relu', padding='same')(c8)

    u9 = UpSampling2D((2, 2))(c8)
    n9 = BatchNormalization()(u9)
    u9 = concatenate([n9, n1], axis=3)
    c9 = Conv2D(maps[0], (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(maps[0], (3, 3), activation='relu', padding='same')(c9)

    actv = 'softmax'
    if nclass == 1:
        actv = 'sigmoid'

    c10 = Conv2D(nclass, (1, 1), activation=actv, kernel_regularizer=kr)(c9)
    model = Model(inputs=inputs, outputs=c10, name="UNetBatchNorm")

    if weight_file:
        model.load_weights(weight_file)
    return model


# -------------------------------------------------------------------------------
# module unet Unit Tests
# -------------------------------------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # Can add different images sizes - for now: (256,256,6)
    simple_unet = unet_dropout()
    simple_unet.summary()

    # Batch Normalization UNet
    simple_unet = unet_batchnorm()
    simple_unet.summary()
