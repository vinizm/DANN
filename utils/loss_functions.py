import numpy as np
import cv2
import tensorflow as tf


def pixel_wise_weighted_loss(y_true, y_pred):
    pass


def binary_crossentropy(y_true, y_pred):
    tmp = tf.math.multiply(y_true, y_pred)
    tmp = tf.reduce_sum(tmp, axis = -1)
    loss = -1. * tf.math.log(tmp)
    return loss