from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Dropout

from models.layers import ExpandDimensions
from models.blocks import AtrousSeparableConv


def deeplabv3plus_encoder(input_shape = (256, 256, 1), output_stride: int = 8):

    if output_stride == 16:
        side_stride = 2
        pool = lambda x: MaxPool2D(pool_size = 2, strides = 2, padding = 'valid')(x)

    elif output_stride == 8:
        side_stride = 1
        pool = lambda x: x
        
    else:
        return

    input_img = Input(shape = input_shape)

    x = Conv2D(filters = 32, kernel_size = 3, strides = 2, dilation_rate = 1, padding = 'same')(input_img)
    x = BatchNormalization(epsilon = 1.e-6)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters = 64, kernel_size = 3, strides = 1, dilation_rate = 1, padding = 'same')(x)
    x = BatchNormalization(epsilon = 1.e-6)(x)
    x = Activation('relu')(x)

    # === ENTRY FLOW 1 ===

    residual = Conv2D(filters = 128, kernel_size = 1, strides = 2, dilation_rate = 1, padding = 'valid')(x)
    residual = BatchNormalization(epsilon = 1.e-6)(residual)

    x = AtrousSeparableConv(x, filters = 128, kernel_size = 3, strides = 1, dilation_rate = 1, batch_norm = True, relu_activation = True)
    x = AtrousSeparableConv(x, filters = 128, kernel_size = 3, strides = 1, dilation_rate = 1, batch_norm = True, relu_activation = False)
    x = MaxPool2D(pool_size = 2, strides = 2, padding = 'valid')(x)

    x = Add()([x, residual])

    # === ENTRY FLOW 2 ===

    residual = Conv2D(filters = 256, kernel_size = 1, strides = 2, dilation_rate = 1, padding = 'valid')(residual)
    residual = BatchNormalization(epsilon = 1.e-6)(residual)

    x = Activation('relu')(x)
    x = AtrousSeparableConv(x, filters = 256, kernel_size = 3, strides = 1, dilation_rate = 1, batch_norm = True, relu_activation = True)
    x = AtrousSeparableConv(x, filters = 256, kernel_size = 3, strides = 1, dilation_rate = 1, batch_norm = True, relu_activation = False)
    x = MaxPool2D(pool_size = 2, strides = 2, padding = 'valid')(x)

    x = Add()([x, residual])
    x = Dropout(0.2)(x) # Dropout

    # === ENTRY FLOW 3 ===

    residual = Conv2D(filters = 728, kernel_size = 1, strides = side_stride, dilation_rate = 1, padding = 'valid')(residual)
    residual = BatchNormalization(epsilon = 1.e-6)(residual)

    x = Activation('relu')(x)
    x = AtrousSeparableConv(x, filters = 728, kernel_size = 3, strides = 1, dilation_rate = 1, batch_norm = True, relu_activation = True)
    skip_connect = AtrousSeparableConv(x, filters = 728, kernel_size = 3, strides = 1, dilation_rate = 1, batch_norm = True, relu_activation = False)
    x = pool(skip_connect)

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
    x = Dropout(0.2)(x) # Dropout

    # === EXIT FLOW 1 ===

    residual = Conv2D(filters = 1024, kernel_size = 1, strides = 1, dilation_rate = 1, padding = 'valid')(x)
    residual = BatchNormalization(epsilon = 1.e-6)(residual)

    x = AtrousSeparableConv(x, filters = 728, kernel_size = 3, strides = 1, dilation_rate = 1, batch_norm = True, relu_activation = True)
    x = AtrousSeparableConv(x, filters = 1024, kernel_size = 3, strides = 1, dilation_rate = 1, batch_norm = True, relu_activation = False)

    x = Add()([x, residual])
    x = Dropout(0.2)(x) # Dropout

    # === EXIT FLOW 2 ===

    x = AtrousSeparableConv(x, filters = 1536, kernel_size = 3, strides = 1, dilation_rate = 1, batch_norm = True, relu_activation = True)
    x = AtrousSeparableConv(x, filters = 2048, kernel_size = 3, strides = 1, dilation_rate = 1, batch_norm = True, relu_activation = False)
    x = Dropout(0.2)(x) # Dropout

    # === IMAGE LEVEL FEATURES ===

    default_shape = tuple(x.shape)

    b0 = GlobalAveragePooling2D()(x)

    # (batch_size, channels) -> (batch_size, 1, 1, channels)
    b0 = ExpandDimensions(axis = 1)(b0)
    b0 = ExpandDimensions(axis = 1)(b0)

    b0 = Conv2D(filters = 256, kernel_size = 1, strides = 1, dilation_rate = 1, padding = 'valid')(b0)

    previous_shape = tuple(b0.shape)
    upsampling_factor = int(default_shape[1] / previous_shape[1])

    b0 = UpSampling2D(size = upsampling_factor, interpolation = 'bilinear')(b0)

    # ====== ASPP ======

    b1 = Conv2D(filters = 256, kernel_size = 1, strides = 1, dilation_rate = 1, padding = 'valid')(x)
    b2 = Conv2D(filters = 256, kernel_size = 3, strides = 1, dilation_rate = 1, padding = 'same')(x)
    b3 = Conv2D(filters = 256, kernel_size = 3, strides = 1, dilation_rate = 2, padding = 'same')(x)
    b4 = Conv2D(filters = 256, kernel_size = 3, strides = 1, dilation_rate = 3, padding = 'same')(x)

    x = Concatenate()([b0, b1, b2, b3, b4])
    features = Conv2D(filters = 256, kernel_size = 1, strides = 1, dilation_rate = 1, padding = 'valid')(x)
    print(f'Bottleneck: {features.shape}')
    
    model = Model(inputs = input_img, outputs = [features, skip_connect])
    return model


def deeplabv3plus_decoder(
    input_shape: tuple = (256, 256, 1),
    features_shape: tuple = (32, 32, 256),
    num_class: int = 2,
    skip_connect_shape: tuple = (32, 32, 728),
    activate_skip_conn: bool = True
    ):
    
    features = Input(shape = features_shape)
    input_skip_connect = Input(shape = skip_connect_shape)
    
    # ====== DECODER ======

    target_dim = input_shape[1] / 2
    previous_shape = tuple(features.shape)
    upsampling_factor = int(target_dim / previous_shape[1])
    print(f'Upsampling Features: {upsampling_factor}')

    x = UpSampling2D(size = upsampling_factor, interpolation = 'bilinear')(features)
    
    # ====== SKIP CONNECTION ======
    
    if activate_skip_conn:
        
        skip_connect = Conv2D(filters = 48, kernel_size = 1, strides = 1, dilation_rate = 1, padding = 'valid')(input_skip_connect)

        previous_shape = tuple(skip_connect.shape)
        upsampling_factor = int(target_dim / previous_shape[1])
        print(f'Upsampling Skip Conn: {upsampling_factor}')

        skip_connect = UpSampling2D(size = upsampling_factor, interpolation = 'bilinear')(skip_connect)
        x = Concatenate()([x, skip_connect])
        print(f'Concatenate: {x.shape, skip_connect.shape}')
    
    # =============================

    x = Conv2D(filters = 256, kernel_size = 1, strides = 1, dilation_rate = 1, padding = 'valid')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters = 256, kernel_size = 1, strides = 1, dilation_rate = 1, padding = 'valid')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters = num_class, kernel_size = 1, strides = 1, dilation_rate = 1, padding = 'valid')(x)

    previous_shape = tuple(x.shape)
    upsampling_factor = int(input_shape[1] / previous_shape[1])
    print(f'Final Upsampling: {upsampling_factor}')

    x = UpSampling2D(size = upsampling_factor, interpolation = 'bilinear')(x)
    output_proba = Activation('softmax')(x)

    model = Model(inputs = [features, input_skip_connect], outputs = output_proba)
    return model


class DeepLabV3Plus(Model):
    
    def __init__(self, input_shape: tuple = (256, 256, 1), num_class: int = 2, output_stride: int = 8, skip_conn: bool = True, **kwargs):
        super(DeepLabV3Plus, self).__init__(**kwargs)
        
        self.encoder = deeplabv3plus_encoder(
            input_shape = input_shape,
            output_stride = output_stride
            )
        
        features_shape = tuple(self.encoder.outputs[0].shape[1:])
        skip_connect_shape = tuple(self.encoder.outputs[1].shape[1:])
        self.decoder = deeplabv3plus_decoder(
            input_shape = input_shape,
            features_shape = features_shape,
            num_class = num_class,
            skip_connect_shape = skip_connect_shape,
            activate_skip_conn = skip_conn
            )
        
        self.inputs = [Input(shape = input_shape)]
        self.outputs = self.call(self.inputs)

        self.build()

    def call(self, x):

        features, skip_connect = self.encoder(x)
        output_proba = self.decoder([features, skip_connect])

        return output_proba

    def get_config(self):
        config = super(DeepLabV3Plus, self).get_config()
        return config

    def from_config(cls, config):
        return cls(**config)

    def build(self):
        super(DeepLabV3Plus, self).build(self.inputs.shape if tf.is_tensor(self.inputs) else self.inputs)
        self.call(self.inputs)
