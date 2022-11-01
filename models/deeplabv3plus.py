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
from tensorflow.keras.layers import UpSampling2D

from models.layers import ExpandDimensions
from models.blocks import AtrousSeparableConv


def DeepLabV3Plus(input_shape: tuple, num_class: int, domain_adaptation: bool):
    model = None
    return model