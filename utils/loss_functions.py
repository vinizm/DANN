import tensorflow as tf
from tensorflow.keras.losses import Loss

from utils.utils import generate_weight_maps


class BinaryCrossentropy(Loss):

    def __init__(self, **kwargs):
        super(BinaryCrossentropy, self).__init__(**kwargs)

    def call(self, y_true, y_pred):

        loss = tf.math.multiply(y_true, y_pred)
        loss = tf.math.reduce_sum(loss, axis = -1)
        loss = tf.math.log(loss)
        loss = tf.math.multiply(-1., loss)

        return loss

class WeighedD1BinaryCrossentropy(Loss):

    def __init__(self, epsilon: float, **kwargs):
        super(WeighedD1BinaryCrossentropy, self).__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        wmaps = generate_weight_maps(y_true, self.epsilon)

        loss = tf.math.multiply(y_true, y_pred)
        loss = tf.math.reduce_sum(loss, axis = -1)
        loss = tf.math.multiply(wmaps, tf.math.log(loss))
        loss = tf.math.multiply(-1., loss)

        return loss