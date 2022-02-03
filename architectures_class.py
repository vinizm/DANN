from turtle import xcor
from unittest import skip
import xdrlib
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import ZeroPadding2D, Activation, DepthwiseConv2D, BatchNormalization, Conv2D, Lambda, Add, Concatenate


class _conv2d_same(Model):

    def __init__(self, filters: int, prefix: str, stride: int = 1, kernel_size: int = 3, rate: int = 1):
        super(_conv2d_same, self).__init__()

        if stride == 1:
            self.first_layer = Lambda(lambda x: x)
            self.conv =  Conv2D(filters, (kernel_size, kernel_size), strides = (stride, stride),
                                padding = 'same', use_bias = False, dilation_rate = (rate, rate),
                                name = prefix)
        else:
            kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg

            self.first_layer = ZeroPadding2D((pad_beg, pad_end))
            self.conv = Conv2D(filters, (kernel_size, kernel_size), strides = (stride, stride),
                        padding = 'valid', use_bias = False, dilation_rate = (rate, rate),
                        name = prefix)

    def call(self, inputs):

        x = self.first_layer(inputs)
        x = self.conv(inputs)
        return x


class SepConv_BN(Model):

    def __init__(self, filters: int, prefix: str, stride: int = 1, kernel_size: int = 3,
                 rate: int = 1, depth_activation: bool = False, epsilon: float = 1.e-3):
        super(SepConv_BN, self).__init()

        self.depth_activation = depth_activation
        self.stride = stride

        self.act = Activation('relu')
        self.batch_norm_1 = BatchNormalization(name = prefix + '_depthwise_BN', epsilon = epsilon)
        self.batch_norm_2 = BatchNormalization(name = prefix + '_pointwise_BN', epsilon = epsilon)
        self.conv = Conv2D(filters, (1, 1), padding = 'same', use_bias = False, name = prefix + '_pointwise')

        if stride == 1:
            depth_padding = 'same'
        else:
            kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            depth_padding = 'valid'
            self.zero_padding = ZeroPadding2D((pad_beg, pad_end))

        self.depth_wise_conv = DepthwiseConv2D((kernel_size, kernel_size), strides = (stride, stride),
                                                dilation_rate = (rate, rate), padding = depth_padding,
                                                use_bias = False, name = prefix + '_depthwise')

    def call(self, inputs):
        x = inputs

        if self.stride != 1:
            x = self.zero_padding(x)

        if not self.depth_activation:
            x = self.act(inputs)

        x = self.depth_wise_conv(x)
        x = self.batch_norm_1(x)

        if self.depth_activation:
            x = self.act(x)

        x = self.conv(x)
        x = self.batch_norm_2(x)

        if self.depth_activation:
            x = self.act(x)

        return x


class _xception_block(Model):

    def __init__(self, depth_list: list, prefix: str, skip_connection_type: str, stride: int,
                 rate: int = 1, depth_activation: bool = False):
        super(_xception_block, self).__init__()
        self.skip_connection_type = skip_connection_type

        self.sep_convs = [SepConv_BN(depth_list[i], prefix + '_separable_conv{}'.format(i + 1),
                          stride = stride if i == 2 else 1, rate = rate, depth_activation = depth_activation) for i in range(3)]

        if skip_connection_type == 'conv':
            self.shortcut_layer = _conv2d_same(depth_list[-1], prefix + '_shortcut', kernel_size = 1, stride = stride)
            self.shortcut_batch_norm = BatchNormalization(name = prefix + '_shortcut_BN')
            self.add = Add()

        elif skip_connection_type == 'sum':
            self.add = Add()

        elif skip_connection_type is None:
            pass

    def call(self, inputs):
        residual = inputs

        for i in range(3):
            residual = self.sep_convs[i](residual)

            if i == 1:
                skip = residual
    
        if self.skip_connection_type == 'conv':
            shortcut = self.shortcut_layer(inputs)
            shortcut = self.shortcut_batch_norm(shortcut)
            outputs = self.add([residual, shortcut])

        elif self.skip_connection_type == 'sum':
            outputs = self.add([residual, inputs])

        elif self.skip_connection_type == 'none':
            outputs = residual

        return outputs, skip


class Deeplabv3plus(Model):

    def __init__(self, classes: int = 2, OS: int = 16, alpha: float = 1., activation = None):
        super(Deeplabv3plus, self).__init__()

        if OS == 8:
            self.entry_block3_stride = 1
            self.middle_block_rate = 2  # ! Not mentioned in paper, but required
            self.exit_block_rates = (2, 4)
            self.atrous_rates = (12, 24, 36)

        elif OS == 16:
            self.entry_block3_stride = 2
            self.middle_block_rate = 1
            self.exit_block_rates = (1, 2)
            self.atrous_rates = (6, 12, 18)

        self.conv_1 = Conv2D(filters = 32, kernel_size = (3, 3), strides = (2, 2), name = 'entry_flow_conv1_1',
                             use_bias = False, padding = 'same')
        self.batch_norm_1 = BatchNormalization(name = 'entry_flow_conv1_1_BN')
        self.act_1 = Activation('relu')

        self.conv_2 = _conv2d_same(filters = 64, prefix = 'entry_flow_conv1_2', kernel_size = 3, stride = 1)
        self.batch_norm_2 = BatchNormalization(name = 'entry_flow_conv1_2_BN')

    def call(self, inputs):
        pass


    