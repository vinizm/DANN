from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import Activation


def AtrousSeparableConv(inputs, filters: int, kernel_size: int, strides: int, dilation_rate: int, batch_norm: bool, relu_activation: bool):

	x = DepthwiseConv2D(kernel_size = kernel_size, strides = strides, dilation_rate = dilation_rate, padding = 'same')(inputs)
	x = Conv2D(filters = filters, kernel_size = 1, strides = 1, dilation_rate = 1, padding = 'valid')(x)

	if batch_norm:
		x = BatchNormalization(epsilon = 1.e-6)(x)

	if relu_activation:
		x = Activation('relu')(x)

	return x
