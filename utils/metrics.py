import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, average_precision_score


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


def f1(precision: float, recall: float):
    f1_value = (2 * precision * recall) / (precision + recall)
    return f1_value