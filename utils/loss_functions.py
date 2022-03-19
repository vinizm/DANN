import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.losses import Loss

from utils.utils import generate_weight_maps


class BinaryCrossentropy(Loss):

    def __init__(self, **kwargs):
        super(BinaryCrossentropy, self).__init__(**kwargs)

    def call(self, y_true, y_pred):

        loss = tf.math.multiply(y_true, y_pred)
        loss = tf.reduce_sum(loss, axis = -1)
        loss = tf.math.log(loss)
        loss = -1. * loss

        return loss

class WeighedD1BinaryCrossentropy(Loss):

    def __init__(self, epsilon: float, **kwargs):
        super(WeighedD1BinaryCrossentropy, self).__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        wmaps = generate_weight_maps(y_true, self.epsilon)

        tmp = tf.math.multiply(y_true, y_pred)
        tmp = tf.reduce_sum(tmp, axis = -1)
        tmp =  tf.math.multiply(wmaps, tf.math.log(tmp))
        loss = tf.multiply(-1., tmp)
        return loss