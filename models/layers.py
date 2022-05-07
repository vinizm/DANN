import tensorflow as tf
from tensorflow.keras.layers import Layer
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