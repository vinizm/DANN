import numpy as np
import glob

from tensorflow.keras.models import load_model

from utils.utils import load_array, compute_metrics, save_json
from utils.hyperparameters import *
from model_builder import DeepLabV3Plus
from config import *


class Predictor():

    def __init__(self, model = None, patch_size: int = 512, channels: int = 1, num_class: int = 2, output_stride: int = 8):
        self.model = model
        self.patch_size = patch_size
        self.channels = channels
        self.num_class = num_class
        self.output_stride = output_stride

        self.test_data_dirs = None
        self.metrics = None

    def load_weights(self, weights_path: str):
        self.model = DeepLabV3Plus(input_shape = (self.patch_size, self.patch_size, self.channels), num_class = self.num_class,
								   output_stride = self.output_stride, activation = 'softmax', classifier_position = None)
        self.model.load_weights(weights_path)

    def load_model(self, model_path: str):
        pass

    def test(self, test_dir: str, num_images_test: int = None, batch_size: int = 2):
    
        test_data_dirs = glob.glob(test_dir + '/*.npy')
        test_data_dirs.sort()
        self.test_data_dirs = test_data_dirs[: num_images_test]

        # compute number of batches
        num_batches = len(self.test_data_dirs) // batch_size

        final_test = []
        final_pred = []
        final_original = []
        for batch in range(num_batches):
            print(f'Batch {batch + 1} of {num_batches}')
            batch_files = self.test_data_dirs[batch * batch_size : (batch + 1) * batch_size]

            batch_images = np.asarray([load_array(batch_file) for batch_file in batch_files])
            batch_images = np.float32(batch_images) # set np.float32 to reduce memory usage

            x_test_total = batch_images[ :, :, :, : self.channels]
            y_test_total = batch_images[ :, :, :, self.channels :]

            print('Predicting...')
            y_pred_total = self.model.predict(x_test_total)

            final_test.append(y_test_total)
            final_pred.append(y_pred_total)
            final_original.append(x_test_total)

        print('Prediction finished.')
        return np.concatenate(final_test), np.concatenate(final_pred), np.concatenate(final_original)

    def evaluate(self, y_true, y_pred, verbose: bool = True):
        self.metrics = compute_metrics(y_true.reshape(-1), np.round(y_pred).reshape(-1))

        if verbose:
            accuracy = self.metrics.get('accuracy')
            print(f'overall accuracy: {accuracy}')

            avg_precision = self.metrics.get('average_precision')
            print(f'average precision: {avg_precision}')

            precision = self.metrics.get('precision')
            print(f'precision: {precision}')

            recall = self.metrics.get('recall')
            print(f'recall: {recall}')

            f1 = self.metrics.get('f1_score')
            print(f'f1 score: {f1}')

        return self.metrics
