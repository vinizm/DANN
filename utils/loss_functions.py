import numpy as np
import cv2
import tensorflow as tf

from variables import EPSILON
from utils.utils import generate_weight_maps


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