from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import MaxPool2D

from models.layers import ReshapeTensor
from models.blocks import AtrousSeparableConv


def deeplabv3plus(input_shape: tuple, num_class: int, domain_adaptation: bool):

    input_img = Input(shape = input_shape)

    x = Conv2D(filters = 32, kernel_size = 3, strides = 2, dilation_rate = 1, padding = 'same')(input_img)
    x = BatchNormalization(epsilon = 1.e-32)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters = 64, kernel_size = 3, strides = 1, dilation_rate = 1, padding = 'same')(x)
    x = BatchNormalization(epsilon = 1.e-32)(x)
    x = Activation('relu')(x)

    # === ENTRY FLOW 1 ===

    residual = Conv2D(filters = 128, kernel_size = 1, strides = 2, dilation_rate = 1, padding = 'valid')(x)
    residual = BatchNormalization(epsilon = 1.e-32)(residual)

    x = AtrousSeparableConv(x, filters = 128, kernel_size = 3, strides = 1, dilation_rate = 1, batch_norm = True, relu_activation = True)
    x = AtrousSeparableConv(x, filters = 128, kernel_size = 3, strides = 1, dilation_rate = 1, batch_norm = True, relu_activation = False)
    x = MaxPool2D(pool_size = 2, strides = 2, padding = 'valid')(x)

    x = Add()([x, residual])

    # === ENTRY FLOW 2 ===

    residual = Conv2D(filters = 256, kernel_size = 1, strides = 2, dilation_rate = 1, padding = 'valid')(residual)
    residual = BatchNormalization(epsilon = 1.e-32)(residual)

    x = Activation('relu')(x)
    x = AtrousSeparableConv(x, filters = 256, kernel_size = 3, strides = 1, dilation_rate = 1, batch_norm = True, relu_activation = True)
    x = AtrousSeparableConv(x, filters = 256, kernel_size = 3, strides = 1, dilation_rate = 1, batch_norm = True, relu_activation = False)
    x = MaxPool2D(pool_size = 2, strides = 2, padding = 'valid')(x)

    x = Add()([x, residual])

    # === ENTRY FLOW 3 ===

    residual = Conv2D(filters = 728, kernel_size = 1, strides = 2, dilation_rate = 1, padding = 'valid')(residual)
    residual = BatchNormalization(epsilon = 1.e-32)(residual)

    x = Activation('relu')(x)
    x = AtrousSeparableConv(x, filters = 728, kernel_size = 3, strides = 1, dilation_rate = 1, batch_norm = True, relu_activation = True)
    skip_conn = AtrousSeparableConv(x, filters = 728, kernel_size = 3, strides = 1, dilation_rate = 1, batch_norm = True, relu_activation = False)
    x = MaxPool2D(pool_size = 2, strides = 2, padding = 'valid')(skip_conn)

    bypass = Add()([x, residual])
    x = bypass

    # ==== MIDDLE FLOW ====

    for _ in range(8):
        x = Activation('relu')(x)
        x = AtrousSeparableConv(x, filters = 728, kernel_size = 3, strides = 1, dilation_rate = 1, batch_norm = True, relu_activation = True)
        x = AtrousSeparableConv(x, filters = 728, kernel_size = 3, strides = 1, dilation_rate = 1, batch_norm = True, relu_activation = True)
        x = AtrousSeparableConv(x, filters = 728, kernel_size = 3, strides = 1, dilation_rate = 1, batch_norm = True, relu_activation = True)

    x = Add()([x, bypass])
    x = Activation('relu')(x)

    # === EXIT FLOW 1 ===

    residual = Conv2D(filters = 1024, kernel_size = 1, strides = 1, dilation_rate = 1, padding = 'valid')(x)
    residual = BatchNormalization(epsilon = 1.e-32)(residual)

    x = AtrousSeparableConv(x, filters = 728, kernel_size = 3, strides = 1, dilation_rate = 1, batch_norm = True, relu_activation = True)
    x = AtrousSeparableConv(x, filters = 1024, kernel_size = 3, strides = 1, dilation_rate = 1, batch_norm = True, relu_activation = False)

    x = Add()([x, residual])

    # === EXIT FLOW 2 ===

    x = AtrousSeparableConv(x, filters = 1536, kernel_size = 3, strides = 1, dilation_rate = 1, batch_norm = True, relu_activation = True)
    x = AtrousSeparableConv(x, filters = 2048, kernel_size = 3, strides = 1, dilation_rate = 1, batch_norm = True, relu_activation = False)

    # === IMAGE LEVEL FEATURES ===

    default_size = tuple(x.shape)

    b0 = GlobalAveragePooling2D(keepdims = True)(x)
    b0 = Conv2D(filters = 256, kernel_size = 1, strides = 1, dilation_rate = 1, padding = 'valid')(b0)
    b0 = ReshapeTensor(default_size[1:3], factor = 1, method = 'bilinear', align_corners = True)(b0)

    # ====== ASPP ======

    b1 = Conv2D(filters = 256, kernel_size = 1, strides = 1, dilation_rate = 1, padding = 'valid')(x)
    b2 = Conv2D(filters = 256, kernel_size = 3, strides = 1, dilation_rate = 1, padding = 'same')(x)
    b3 = Conv2D(filters = 256, kernel_size = 3, strides = 1, dilation_rate = 2, padding = 'same')(x)
    b4 = Conv2D(filters = 256, kernel_size = 3, strides = 1, dilation_rate = 3, padding = 'same')(x)

    x = Concatenate()([b0, b1, b2, b3, b4])
    features = Conv2D(filters = 256, kernel_size = 1, strides = 1, dilation_rate = 1, padding = 'valid')(x)

    # ====== DECODER ======

    skip_conn = Conv2D(filters = 48, kernel_size = 1, strides = 1, dilation_rate = 1, padding = 'valid')(skip_conn)

    skip_conn_size = tuple(skip_conn.shape)
    x = ReshapeTensor(skip_conn_size[1:3], factor = 1, method = 'bilinear', align_corners = True)(features)
    x = Concatenate()([x, skip_conn])

    x = Conv2D(filters = 256, kernel_size = 1, strides = 1, dilation_rate = 1, padding = 'same')(x)
    x = Conv2D(filters = 256, kernel_size = 1, strides = 1, dilation_rate = 1, padding = 'same')(x)
    x = Conv2D(filters = num_class, kernel_size = 1, strides = 1, dilation_rate = 1, padding = 'same')(x)

    original_shape = tuple(input_img.shape)
    output_proba = ReshapeTensor(original_shape[1:3], factor = 1, method = 'bilinear', align_corners = True)(x)
    
    if domain_adaptation:
        output_proba = [output_proba, features]
    
    model = Model(inputs = input_img, outputs = output_proba)
    return model