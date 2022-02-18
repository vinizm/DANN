# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from turtle import shape

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K


class ReshapeTensor(Layer):

    def __init__(self, shape, factor: int = 1, method: str = 'bilinear',
                 align_corners: bool = True, **kwargs):
        super(ReshapeTensor, self).__init__(**kwargs)
        
        self.shape = shape
        self.factor = factor
        self.method = method
        self.align_corners = align_corners

    def call(self, inputs):
        reshaped = tf.compat.v1.image.resize(inputs, self.shape * tf.constant(self.factor), method = self.method,
                                             align_corners = self.align_corners)
        return reshaped

    def get_config(self):
        config = super(ReshapeTensor, self).get_config()
        config.update({'shape': self.shape,
                       'factor': self.factor,
                       'method': self.method,
                       'align_corners': self.align_corners})
        return config

    def from_config(cls, config):
        return cls(**config)


class ExpandDimensions(Layer):

    def __init__(self, **kwargs):
        super(ExpandDimensions, self).__init__(**kwargs)
    
    def call(self, inputs):
        expanded = K.expand_dims(inputs, 1)
        return expanded

    def get_config(self):
        config = super(ExpandDimensions, self).get_config()
        return config

    def from_config(cls, config):
        return cls(**config)


class DomainRegressor(Model):

    def __init__(self, units: int, **kwargs):
        super(DomainRegressor, self).__init__(**kwargs)
        self.units = units

        self.flat = Flatten()
        self.dense_1 = Dense(units = units)
        self.activ_1 = Activation('relu')
        self.dense_2 = Dense(units = units)
        self.activ_2 = Activation('relu')
        self.proba = Activation('softmax')

    def call(self, inputs):

        x = self.flat(inputs)
        x = self.dense_1(x)
        x = self.activ_1(x)
        x = self.dense_2(x)
        x = self.activ_2(x)
        x = self.proba(x)

        return x

    def get_config(self):
        config = super(DomainRegressor, self).get_config()
        config.update({'units': self.units})
        return config

    def from_config(cls, config):
        return cls(**config)


def SepConv_BN(x, filters, prefix, stride = 1, kernel_size = 3, rate = 1, depth_activation = False, epsilon = 1e-3):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation('relu')(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides = (stride, stride), dilation_rate = (rate, rate),
                        padding = depth_padding, use_bias = False, name = prefix + '_depthwise')(x)
    x = BatchNormalization(name = prefix + '_depthwise_BN', epsilon = epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), padding = 'same',
               use_bias = False, name = prefix + '_pointwise')(x)
    x = BatchNormalization(name = prefix + '_pointwise_BN', epsilon = epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    return x


def _conv2d_same(x, filters, prefix, stride = 1, kernel_size = 3, rate = 1):
    """Implements right 'same' padding for even kernel sizes
        Without this there is a 1 pixel drift when stride = 2
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
    """
    if stride == 1:
        return Conv2D(filters, (kernel_size, kernel_size), strides = (stride, stride), padding = 'same',
                      use_bias = False, dilation_rate = (rate, rate), name = prefix)(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        return Conv2D(filters, (kernel_size, kernel_size), strides = (stride, stride), padding = 'valid',
                      use_bias = False, dilation_rate = (rate, rate), name = prefix)(x)


def _xception_block(inputs, depth_list, prefix, skip_connection_type, stride,
                    rate = 1, depth_activation = False, return_skip = False):
    """ Basic building block of modified Xception network
        Args:
            inputs: input tensor
            depth_list: number of filters in each SepConv layer. len(depth_list) == 3
            prefix: prefix before name
            skip_connection_type: one of {'conv','sum','none'}
            stride: stride at last depthwise conv
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & pointwise convs
            return_skip: flag to return additional tensor after 2 SepConvs for decoder
            """
    residual = inputs
    for i in range(3):
        residual = SepConv_BN(residual, depth_list[i], prefix + '_separable_conv{}'.format(i + 1),
                              stride = stride if i == 2 else 1, rate = rate, depth_activation = depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = _conv2d_same(inputs, depth_list[-1], prefix + '_shortcut', kernel_size = 1, stride = stride)
        shortcut = BatchNormalization(name = prefix + '_shortcut_BN')(shortcut)
        outputs = Add()([residual, shortcut])
    elif skip_connection_type == 'sum':
        outputs = Add()([residual, inputs])
    elif skip_connection_type == 'none':
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs


def Deeplabv3plus(input_shape: tuple = (512, 512, 1), classes: int = 2, OS: int = 16, activation: str = None,
                  classifier_position: str = None):
    """ Instantiates the Deeplabv3+ architecture

    Optionally loads weights pre-trained
    on PASCAL VOC or Cityscapes. This model is available for TensorFlow only.
    # Arguments
        input_shape: shape of input image. format HxWxC
        classes: number of desired classes.
        activation: optional activation to add to the top of the network.
            One of 'softmax', 'sigmoid' or None
        OS: determines input_shape/feature_extractor_output ratio. One of {8,16}.
            Used only for xception backbone.

    # Returns
        A Keras model instance.

    # Raises
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
        ValueError: in case of invalid argument for `weights` or `backbone`

    """
    img_input = Input(shape = input_shape)

    if OS == 8:
        entry_block3_stride = 1
        middle_block_rate = 2 # not mentioned in paper; but required
        exit_block_rates = (2, 4)
        atrous_rates = (12, 24, 36)

    elif OS != 8:
        entry_block3_stride = 2
        middle_block_rate = 1
        exit_block_rates = (1, 2)
        atrous_rates = (6, 12, 18)

    x = Conv2D(32, (3, 3), strides = (2, 2), name = 'entry_flow_conv1_1', use_bias = False, padding = 'same')(img_input)
    x = BatchNormalization(name = 'entry_flow_conv1_1_BN')(x)
    x = Activation('relu')(x)

    x = _conv2d_same(x, 64, 'entry_flow_conv1_2', kernel_size = 3, stride = 1)
    x = BatchNormalization(name = 'entry_flow_conv1_2_BN')(x)
    x = Activation('relu')(x)

    x, skip_0 = _xception_block(x, [128, 128, 128], 'entry_flow_block1',
                               skip_connection_type = 'conv', stride = 2,
                               depth_activation = False, return_skip = True)
    
    # new skip connection
    x, skip_1 = _xception_block(x, [256, 256, 256], 'entry_flow_block2',
                               skip_connection_type = 'conv', stride = 2,
                               depth_activation = False, return_skip  =True)

    x = _xception_block(x, [728, 728, 728], 'entry_flow_block3',
                        skip_connection_type = 'conv', stride = entry_block3_stride,
                        depth_activation = False)
    
    for i in range(16):
        x = _xception_block(x, [728, 728, 728], 'middle_flow_unit_{}'.format(i + 1),
                            skip_connection_type = 'sum', stride = 1, rate = middle_block_rate,
                            depth_activation = False)

    x = _xception_block(x, [728, 1024, 1024], 'exit_flow_block1',
                        skip_connection_type = 'conv', stride = 1, rate  =exit_block_rates[0],
                        depth_activation = False)
    classifier_spot_1 = _xception_block(x, [1536, 1536, 2048], 'exit_flow_block2',
                        skip_connection_type = 'none', stride = 1, rate = exit_block_rates[1],
                        depth_activation = True)

    # end of feature extractor
    # branching for Atrous Spatial Pyramid Pooling (ASPP)
    # image feature branch
    b4 = GlobalAveragePooling2D()(classifier_spot_1)

    # from (batch_size, channels) -> (batch_size, 1, 1, channels)
    b4 = ExpandDimensions()(b4)
    b4 = ExpandDimensions()(b4)

    b4 = Conv2D(256, (1, 1), padding = 'same', use_bias = False, name = 'image_pooling')(b4)
    b4 = BatchNormalization(name = 'image_pooling_BN', epsilon = 1e-5)(b4)
    b4 = Activation('relu')(b4)

    # upsample; have to use compat because of the option align_corners
    size_before = K.int_shape(x)
    b4 = ReshapeTensor(size_before[1:3], factor = 1, method = 'bilinear', align_corners = True)(b4)
    
    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding = 'same', use_bias = False, name = 'aspp0')(x)
    b0 = BatchNormalization(name = 'aspp0_BN', epsilon = 1e-5)(b0)
    b0 = Activation('relu', name = 'aspp0_activation')(b0)

    # there are only 2 branches in mobilenetV2; not sure why
    # rate = 6 (12)
    b1 = SepConv_BN(x, 256, 'aspp1', rate = atrous_rates[0], depth_activation = True, epsilon = 1e-5)
    
    # rate = 12 (24)
    b2 = SepConv_BN(x, 256, 'aspp2', rate = atrous_rates[1], depth_activation = True, epsilon = 1e-5)
    
    # rate = 18 (36)
    b3 = SepConv_BN(x, 256, 'aspp3', rate = atrous_rates[2], depth_activation = True, epsilon = 1e-5)

    # concatenate ASPP branches and project
    x = Concatenate()([b4, b0, b1, b2, b3])

    x = Conv2D(256, (1, 1), padding = 'same', use_bias = False, name = 'concat_projection')(x)
    x = BatchNormalization(name = 'concat_projection_BN', epsilon = 1e-5)(x)
    x = Activation('relu')(x)
    classifier_spot_2 = Dropout(0.1)(x)

    # DeepLabv3+ decoder
    # feature projection
    size_before_2 = K.int_shape(classifier_spot_2)
    x = ReshapeTensor(size_before_2[1:3], factor = OS // 4, method = 'bilinear', align_corners = True)(classifier_spot_2)

    dec_skip_1 = Conv2D(48, (1, 1), padding = 'same', use_bias = False, name = 'feature_projection1')(skip_1)
    dec_skip_1 = BatchNormalization(name = 'feature_projection1_BN', epsilon = 1e-5)(dec_skip_1)
    dec_skip_1 = Activation('relu')(dec_skip_1)
    x = Concatenate()([x, dec_skip_1])

    x = SepConv_BN(x, 256, 'decoder_conv2', depth_activation = True, epsilon = 1e-5)
    classifier_spot_3 = SepConv_BN(x, 256, 'decoder_conv3', depth_activation = True, epsilon = 1e-5)

    x = ReshapeTensor(size_before_2[1:3], factor = OS // 2, method = 'bilinear', align_corners = True)(classifier_spot_3)

    dec_skip_0 = Conv2D(48, (1, 1), padding = 'same', use_bias = False, name = 'feature_projection0')(skip_0)
    dec_skip_0 = BatchNormalization(name = 'feature_projection0_BN', epsilon = 1e-5)(dec_skip_0)
    dec_skip_0 = Activation('relu')(dec_skip_0)
    x = Concatenate()([x, dec_skip_0])

    x = SepConv_BN(x, 128, 'decoder_conv0', depth_activation = True, epsilon = 1e-5)
    x = SepConv_BN(x, 128, 'decoder_conv1', depth_activation = True, epsilon = 1e-5)
    x = Conv2D(classes, (1, 1), padding = 'same', name = 'custom_logits_semantic')(x)

    size_before_3 = K.int_shape(img_input)
    x = ReshapeTensor(size_before_3[1:3], factor = 1, method = 'bilinear', align_corners = True)(x)

    if activation in ['softmax', 'sigmoid']:
        outputs = tf.keras.layers.Activation(activation)(x)

    classifier_spots = [classifier_spot_1, classifier_spot_2, classifier_spot_3]
    if classifier_position is not None:
        classifier_input = classifier_spots[classifier_position]
        classifier_output = DomainRegressor(units = 1024)(classifier_input)
        outputs = [outputs, classifier_output]

    inputs = img_input
    model = Model(inputs = inputs, outputs = outputs, name = 'deeplabv3plus')
    return model