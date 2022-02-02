import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import ZeroPadding2D, Activation, DepthwiseConv2D, BatchNormalization, Conv2D


class _conv2d_same(Model):

    def __init__(self, filters: int, prefix: str, stride: int = 1, kernel_size: int = 3, rate: int = 1):
        super(_conv2d_same, self).__init__()

        self.zero_padding = None
        self.conv = None

        self.stride = stride

        if stride == 1:
            self.conv =  Conv2D(filters, (kernel_size, kernel_size), strides = (stride, stride),
                                padding = 'same', use_bias = False, dilation_rate = (rate, rate),
                                name = prefix)
        else:
            kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            self.zero_padding = ZeroPadding2D((pad_beg, pad_end))
            self.conv = Conv2D(filters, (kernel_size, kernel_size), strides = (stride, stride),
                        padding = 'valid', use_bias = False, dilation_rate = (rate, rate),
                        name = prefix)

    def call(self, inputs):

        if self.stride == 1:
            return self.conv(inputs)

        else:
            x = self.zero_padding(inputs)
            return self.conv(x)


class SepConv_BN(Model):

    def __init__(self, filters: int, prefix: str, stride: int = 1, kernel_size: int = 3,
                 rate: int = 1, depth_activation: bool = False, epsilon: float = 1.e-3):
        super(SepConv_BN, self).__init()

        self.zero_padding = None
        self.depth_activation = depth_activation

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


class Deeplabv3plus(Model):

    def __init__(self):
        super(Deeplabv3plus, self).__init__()

    