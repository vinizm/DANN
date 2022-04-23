# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from re import X

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Layer
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
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import backend as K

from flip_gradient import flip_gradient


class GradientReversalLayer(Layer):

    def __init__(self, **kwargs):
        super(GradientReversalLayer, self).__init__(**kwargs)
        self.flipper = flip_gradient
    
    def call(self, inputs):
        x, l = inputs
        y = self.flipper(x, l)
        return y

    def get_config(self):
        return super().get_config()

    def from_config(cls, config):
        return cls(**config)


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


class DomainDiscriminator(Model):

    def __init__(self, units: int, **kwargs):
        super(DomainDiscriminator, self).__init__(**kwargs)
        self.units = units

        self.batch_norm_1 = BatchNormalization(center = False, scale = False, axis = -1, epsilon = 1.e-32)
        self.flat = Flatten()
        self.dense_1 = Dense(units = units)
        self.activ_1 = Activation('relu')
        self.dense_2 = Dense(units = units)
        self.activ_2 = Activation('relu')
        self.dense_3 = Dense(units = 2)
        self.proba = Activation('softmax')

    def call(self, inputs):

        x = self.batch_norm_1(inputs, training = True)
        x = self.flat(x)
        x = self.dense_1(x)
        x = self.activ_1(x)
        x = self.dense_2(x)
        x = self.activ_2(x)
        x = self.dense_3(x)
        x = self.proba(x)

        return x

    def get_config(self):
        config = super(DomainDiscriminator, self).get_config()
        config.update({'units': self.units})
        return config

    def from_config(cls, config):
        return cls(**config)


class DomainAdaptationModel(Model):

    def __init__(self, input_shape: tuple = (256, 256, 1), num_class: int = 2, output_stride: int = 8,
                 activation: str = 'softmax', **kwargs):
        super(DomainAdaptationModel, self).__init__(**kwargs)

        self.encoder = FeatureExtractor(input_shape = input_shape, output_stride = output_stride)

        feature, skip_0, skip_1 = self.encoder.outputs
        feature_shape = tuple(feature.shape[1:])
        skip_0_shape = tuple(skip_0.shape[1:])
        skip_1_shape = tuple(skip_1.shape[1:])
        self.decoder = PixelwiseClassifier(input_shape = input_shape, feature_shape = feature_shape, skip_0_shape = skip_0_shape,
                                           skip_1_shape = skip_1_shape, num_class = num_class, output_stride = output_stride,
                                           activation = activation)

        self.gradient_reversal_layer = GradientReversalLayer()
        self.domain_discriminator = DomainDiscriminator(units = 1024)

        self.inputs = [Input(shape = input_shape), Input(shape = (1,))]
        self.outputs = self.call(self.inputs)

        self.build()

    def call(self, inputs):
        input_img, l = inputs

        feature, skip_0, skip_1 = self.encoder(input_img)
        segmentation_output = self.decoder([feature, skip_0, skip_1])
        discriminator_input = self.gradient_reversal_layer([feature, l])
        discriminator_output = self.domain_discriminator(discriminator_input)

        return segmentation_output, discriminator_output

    def get_config(self):
        config = super(DomainAdaptationModel, self).get_config()
        return config

    def from_config(cls, config):
        return cls(**config)

    def build(self):
        super(DomainAdaptationModel, self).build(self.inputs.shape if tf.is_tensor(self.inputs) else self.inputs)
        self.call(self.inputs)


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


def FeatureExtractor(input_shape: tuple = (256, 256, 1), output_stride: int = 8):

    input_img = Input(shape = input_shape)

    if output_stride == 8:
        entry_block3_stride = 1
        middle_block_rate = 2 # not mentioned in paper; but required
        exit_block_rates = (2, 4)
        atrous_rates = (12, 24, 36)

    elif output_stride != 8:
        entry_block3_stride = 2
        middle_block_rate = 1
        exit_block_rates = (1, 2)
        atrous_rates = (6, 12, 18)

    x = Conv2D(32, (3, 3), strides = (2, 2), name = 'entry_flow_conv1_1', use_bias = False, padding = 'same')(input_img)
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
    x = _xception_block(x, [1536, 1536, 2048], 'exit_flow_block2',
                        skip_connection_type = 'none', stride = 1, rate = exit_block_rates[1],
                        depth_activation = True)

    # end of feature extractor
    # branching for Atrous Spatial Pyramid Pooling (ASPP)
    # image feature branch
    b4 = GlobalAveragePooling2D()(x)

    # from (batch_size, channels) -> (batch_size, 1, 1, channels)
    b4 = ExpandDimensions()(b4)
    b4 = ExpandDimensions()(b4)

    b4 = Conv2D(256, (1, 1), padding = 'same', use_bias = False, name = 'image_pooling')(b4)
    b4 = BatchNormalization(name = 'image_pooling_BN', epsilon = 1e-16)(b4)
    b4 = Activation('relu')(b4)

    # upsample; have to use compat because of the option align_corners
    size_before = K.int_shape(x)
    b4 = ReshapeTensor(size_before[1:3], factor = 1, method = 'bilinear', align_corners = True)(b4)
    
    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding = 'same', use_bias = False, name = 'aspp0')(x)
    b0 = BatchNormalization(name = 'aspp0_BN', epsilon = 1e-16)(b0)
    b0 = Activation('relu', name = 'aspp0_activation')(b0)

    # there are only 2 branches in mobilenetV2; not sure why
    # rate = 6 (12)
    b1 = SepConv_BN(x, 256, 'aspp1', rate = atrous_rates[0], depth_activation = True, epsilon = 1e-16)
    
    # rate = 12 (24)
    b2 = SepConv_BN(x, 256, 'aspp2', rate = atrous_rates[1], depth_activation = True, epsilon = 1e-16)
    
    # rate = 18 (36)
    b3 = SepConv_BN(x, 256, 'aspp3', rate = atrous_rates[2], depth_activation = True, epsilon = 1e-16)

    # concatenate ASPP branches and project
    x = Concatenate()([b4, b0, b1, b2, b3])

    x = Conv2D(256, (1, 1), padding = 'same', use_bias = False, name = 'concat_projection')(x)
    x = BatchNormalization(name = 'concat_projection_BN', epsilon = 1e-16)(x)
    output_feature = Activation('relu')(x)

    model = Model(inputs = input_img, outputs = [output_feature, skip_0, skip_1], name = 'deeplabv3plus_feature_extractor')
    return model

def PixelwiseClassifier(input_shape: tuple = (256, 256, 1), feature_shape: tuple = (16, 16, 256), skip_0_shape: tuple = (128, 128, 128),
                        skip_1_shape: tuple = (64, 64, 256), num_class: int = 2, output_stride: int = 8, activation: str = 'softmax'):

    input_feature = Input(shape = feature_shape)
    skip_0 = Input(shape = skip_0_shape)
    skip_1 = Input(shape = skip_1_shape)

    # x = Dropout(0.1)(input_feature)

    # DeepLabv3+ decoder
    # feature projection
    size_before_2 = K.int_shape(input_feature)
    x = ReshapeTensor(size_before_2[1:3], factor = output_stride // 4, method = 'bilinear', align_corners = True)(input_feature)

    dec_skip_1 = Conv2D(48, (1, 1), padding = 'same', use_bias = False, name = 'feature_projection1')(skip_1)
    dec_skip_1 = BatchNormalization(name = 'feature_projection1_BN', epsilon = 1e-16)(dec_skip_1)
    dec_skip_1 = Activation('relu')(dec_skip_1)
    x = Concatenate()([x, dec_skip_1])

    x = SepConv_BN(x, 256, 'decoder_conv2', depth_activation = True, epsilon = 1e-16)
    x = SepConv_BN(x, 256, 'decoder_conv3', depth_activation = True, epsilon = 1e-16)

    x = ReshapeTensor(size_before_2[1:3], factor = output_stride // 2, method = 'bilinear', align_corners = True)(x)

    dec_skip_0 = Conv2D(48, (1, 1), padding = 'same', use_bias = False, name = 'feature_projection0')(skip_0)
    dec_skip_0 = BatchNormalization(name = 'feature_projection0_BN', epsilon = 1e-16)(dec_skip_0)
    dec_skip_0 = Activation('relu')(dec_skip_0)
    x = Concatenate()([x, dec_skip_0])

    x = SepConv_BN(x, 128, 'decoder_conv0', depth_activation = True, epsilon = 1e-16)
    x = SepConv_BN(x, 128, 'decoder_conv1', depth_activation = True, epsilon = 1e-16)
    x = Conv2D(num_class, (1, 1), padding = 'same', name = 'custom_logits_semantic')(x)

    size_before_3 = (None, *input_shape)
    x = ReshapeTensor(size_before_3[1: 3], factor = 1, method = 'bilinear', align_corners = True)(x)

    if activation in ['softmax', 'sigmoid']:
        output = Activation(activation)(x)

    model = Model(inputs = [input_feature, skip_0, skip_1], outputs = output, name = 'deeplabv3plus_pixelwise_classifier')
    return model


def DeepLabV3Plus(input_shape: tuple = (256, 256, 1), num_class: int = 2, output_stride: int = 8, activation: str = 'softmax',
                  domain_adaptation: bool = False):
    """ Instantiates the Deeplabv3+ architecture
    """
    img_input = Input(shape = input_shape)

    if output_stride == 8:
        entry_block3_stride = 1
        middle_block_rate = 2 # not mentioned in paper; but required
        exit_block_rates = (2, 4)
        atrous_rates = (12, 24, 36)

    elif output_stride != 8:
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
    x = _xception_block(x, [1536, 1536, 2048], 'exit_flow_block2',
                        skip_connection_type = 'none', stride = 1, rate = exit_block_rates[1],
                        depth_activation = True)

    # end of feature extractor
    # branching for Atrous Spatial Pyramid Pooling (ASPP)
    # image feature branch
    b4 = GlobalAveragePooling2D()(x)

    # from (batch_size, channels) -> (batch_size, 1, 1, channels)
    b4 = ExpandDimensions()(b4)
    b4 = ExpandDimensions()(b4)

    b4 = Conv2D(256, (1, 1), padding = 'same', use_bias = False, name = 'image_pooling')(b4)
    b4 = BatchNormalization(name = 'image_pooling_BN', epsilon = 1e-16)(b4)
    b4 = Activation('relu')(b4)

    # upsample; have to use compat because of the option align_corners
    size_before = K.int_shape(x)
    b4 = ReshapeTensor(size_before[1:3], factor = 1, method = 'bilinear', align_corners = True)(b4)
    
    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding = 'same', use_bias = False, name = 'aspp0')(x)
    b0 = BatchNormalization(name = 'aspp0_BN', epsilon = 1e-16)(b0)
    b0 = Activation('relu', name = 'aspp0_activation')(b0)

    # there are only 2 branches in mobilenetV2; not sure why
    # rate = 6 (12)
    b1 = SepConv_BN(x, 256, 'aspp1', rate = atrous_rates[0], depth_activation = True, epsilon = 1e-16)
    
    # rate = 12 (24)
    b2 = SepConv_BN(x, 256, 'aspp2', rate = atrous_rates[1], depth_activation = True, epsilon = 1e-16)
    
    # rate = 18 (36)
    b3 = SepConv_BN(x, 256, 'aspp3', rate = atrous_rates[2], depth_activation = True, epsilon = 1e-16)

    # concatenate ASPP branches and project
    x = Concatenate()([b4, b0, b1, b2, b3])

    x = Conv2D(256, (1, 1), padding = 'same', use_bias = False, name = 'concat_projection')(x)
    x = BatchNormalization(name = 'concat_projection_BN', epsilon = 1e-16)(x)
    discriminator_spot = Activation('relu')(x)
    x = Dropout(0.1)(discriminator_spot)

    # DeepLabv3+ decoder
    # feature projection
    size_before_2 = K.int_shape(discriminator_spot)
    x = ReshapeTensor(size_before_2[1:3], factor = output_stride // 4, method = 'bilinear', align_corners = True)(x)

    dec_skip_1 = Conv2D(48, (1, 1), padding = 'same', use_bias = False, name = 'feature_projection1')(skip_1)
    dec_skip_1 = BatchNormalization(name = 'feature_projection1_BN', epsilon = 1e-16)(dec_skip_1)
    dec_skip_1 = Activation('relu')(dec_skip_1)
    x = Concatenate()([x, dec_skip_1])

    x = SepConv_BN(x, 256, 'decoder_conv2', depth_activation = True, epsilon = 1e-16)
    x = SepConv_BN(x, 256, 'decoder_conv3', depth_activation = True, epsilon = 1e-16)

    x = ReshapeTensor(size_before_2[1:3], factor = output_stride // 2, method = 'bilinear', align_corners = True)(x)

    dec_skip_0 = Conv2D(48, (1, 1), padding = 'same', use_bias = False, name = 'feature_projection0')(skip_0)
    dec_skip_0 = BatchNormalization(name = 'feature_projection0_BN', epsilon = 1e-16)(dec_skip_0)
    dec_skip_0 = Activation('relu')(dec_skip_0)
    x = Concatenate()([x, dec_skip_0])

    x = SepConv_BN(x, 128, 'decoder_conv0', depth_activation = True, epsilon = 1e-16)
    x = SepConv_BN(x, 128, 'decoder_conv1', depth_activation = True, epsilon = 1e-16)
    x = Conv2D(num_class, (1, 1), padding = 'same', name = 'custom_logits_semantic')(x)

    size_before_3 = K.int_shape(img_input)
    x = ReshapeTensor(size_before_3[1:3], factor = 1, method = 'bilinear', align_corners = True)(x)

    if activation in ['softmax', 'sigmoid']:
        outputs = Activation(activation)(x)

    if domain_adaptation:
        outputs = [outputs, discriminator_spot]

    inputs = img_input
    model = Model(inputs = inputs, outputs = outputs, name = 'deeplabv3plus')
    return model


def DeepLabV3PlusDomainAdaptation(input_shape: tuple = (256, 256, 1), num_class: int = 2, output_stride: int = 8,
                                  activation: str = 'softmax'):

    img_input = Input(shape = input_shape)
    lambda_input = Input(shape = (1,))

    raw_model = DomainAdaptationModel(input_shape = input_shape, num_class = num_class, output_stride = output_stride,
                                      activation = activation)
    outputs = raw_model([img_input, lambda_input])

    model = Model(inputs = [img_input, lambda_input], outputs = outputs)
    return model


def DomainDiscriminatorFunctional(input_shape: tuple, units = 1024):
    img_input = Input(shape = input_shape)

    outputs = DomainDiscriminator(units = units)(img_input)
    model = Model(inputs = img_input, outputs = outputs)

    return model
