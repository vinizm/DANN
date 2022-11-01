import tensorflow as tf
from tensorflow.keras.layers import Layer

from models.flip_gradient import flip_gradient


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


class ExpandDimensions(Layer):

    def __init__(self, axis = -1, **kwargs):
        super(ExpandDimensions, self).__init__(**kwargs)
        self.axis = axis
    
    def call(self, inputs):
        expanded = tf.expand_dims(inputs, axis = self.axis)
        return expanded

    def get_config(self):
        config = super(ExpandDimensions, self).get_config()
        config.update({'axis': self.axis})
        return config

    def from_config(cls, config):
        return cls(**config)