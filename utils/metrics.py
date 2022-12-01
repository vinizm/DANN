import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, average_precision_score

import tensorflow as tf
from tensorflow.keras.metrics import Metric


def flatten(array: np.ndarray, keep_dims: bool = True):
  if keep_dims:
    last_dim = array.shape[-1]
    new_shape = (np.prod(array.shape[: -1]), last_dim)
  else:
    new_shape = -1
  return np.reshape(array, new_shape)


def compute_metrics(true_labels: np.ndarray, predicted_labels: np.ndarray):

    true_vector_1 = flatten(np.argmax(true_labels, axis = -1), keep_dims = False)
    true_vector_0 = flatten(np.argmin(true_labels, axis = -1), keep_dims = False)
    pred_vector = flatten(np.argmax(predicted_labels, axis = -1), keep_dims = False)

    accuracy = accuracy_score(true_vector_1, pred_vector)
    precision = precision_score(true_vector_1, pred_vector)
    recall = recall_score(true_vector_1, pred_vector)
    f1 = f1_score(true_vector_1, pred_vector)
    
    proba_pairs = flatten(predicted_labels, keep_dims = True)
    proba_vector_1 = proba_pairs[:, 1]
    proba_vector_0 = proba_pairs[:, 0]
    avg_precision_1 = average_precision_score(true_vector_1, proba_vector_1)
    avg_precision_0 = average_precision_score(true_vector_0, proba_vector_0)
    mean_avg_precision = (avg_precision_1 + avg_precision_0) / 2
    
    metrics = {
                'avg_precision_1': avg_precision_1,
                'avg_precision_0': avg_precision_0,
                'mean_avg_precision': mean_avg_precision,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
            }
    return metrics
  
  
def precision(y_true: tf.Tensor, y_pred: tf.Tensor):
  epsilon = 1.e-6

  u_pred = tf.math.argmax(y_pred, axis = -1)
  u_true = tf.math.argmax(y_true, axis = -1)

  true_positive = tf.cast(tf.reduce_sum(tf.math.multiply(u_pred, u_true)), tf.float32)
  predicted_positive = tf.cast(tf.math.reduce_sum(u_pred), tf.float32)
  precision = true_positive / (predicted_positive + epsilon)

  return precision


def recall(y_true: tf.Tensor, y_pred: tf.Tensor):
  epsilon = 1.e-6

  u_pred = tf.math.argmax(y_pred, axis = -1)
  u_true = tf.math.argmax(y_true, axis = -1)

  true_positive = tf.cast(tf.reduce_sum(tf.math.multiply(u_pred, u_true)), tf.float32)
  possible_positive = tf.cast(tf.math.reduce_sum(u_true), tf.float32)
  precision = true_positive / (possible_positive + epsilon)

  return precision


def f1(precision: float, recall: float):
    epsilon = 1.e-6
    f1_value = (2 * precision * recall) / (precision + recall + epsilon)
    return f1_value


def avg_precision(y_true: tf.Tensor, y_pred: tf.Tensor, proba_loc: int = 1):
  shape = tuple(y_pred.shape)
  u_pred = tf.reshape(y_pred, (np.prod(shape[: -1]), shape[-1]))
  proba = u_pred[:, proba_loc]

  u_true = tf.math.argmax(tf.reshape(y_true, (np.prod(shape[: -1]), shape[-1])), axis = -1)
  return average_precision_score(u_true.numpy(), proba.numpy())


class AveragePrecision(Metric):
  
  def __init__(self, **kwargs):
    super(AveragePrecision, self).__init__(**kwargs)
    self.metric_values = []
    
  def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: np.ndarray = None):
      if 1. in sample_weight:
          index = list(np.argwhere(sample_weight == 1.).reshape(-1))
          
          true_tensor = tf.gather(y_true, indices = index, axis = 0)
          pred_tensor = tf.gather(y_pred, indices = index, axis = 0)
          metric_value = avg_precision(true_tensor, pred_tensor, proba_loc = 1)
          
          self.metric_values.append(metric_value)
  
  def reset_states(self):
    self.metric_values = []
  
  def result(self):
    return np.mean(self.metric_values)