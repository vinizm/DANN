import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.losses import Loss

from variables import EPSILON
from utils.utils import generate_weight_maps


# def weighted_d1_binary_crossentropy(y_true, y_pred):
def binary_crossentropy(y_true, y_pred):
    wmaps = generate_weight_maps(y_true, EPSILON)

    tmp = tf.math.multiply(y_true, y_pred)
    tmp = tf.reduce_sum(tmp, axis = -1)
    loss = -1. * tf.math.multiply(wmaps, tf.math.log(tmp))
    return loss


def binary_crossentropy_x(y_true, y_pred):
    tmp = tf.math.multiply(y_true, y_pred)
    tmp = tf.reduce_sum(tmp, axis = -1)
    loss = -1. * tf.math.log(tmp)
    return loss


class weighted_d1_binary_crossentropy(Loss):

    def __init__(self, epsilon: float):
        super().__init__()
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        wmaps = generate_weight_maps(y_true, self.epsilon)

        tmp = tf.math.multiply(y_true, y_pred)
        tmp = tf.reduce_sum(tmp, axis = -1)
        loss = -1. * tf.math.multiply(wmaps, tf.math.log(tmp))
        return loss