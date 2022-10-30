import tensorflow as tf
from tensorflow.keras.constraints import Constraint


class ZeroMeanFilter(Constraint):
  def __init__(self, **kwargs):
    super(ZeroMeanFilter, self).__init__(**kwargs)

  def __call__(self, weight_matrix):
    return weight_matrix - tf.reduce_mean(weight_matrix, axis = (0, 1, 2), keepdims = True)

  def get_config(self):
      config = super(ZeroMeanFilter, self).get_config()
      return config