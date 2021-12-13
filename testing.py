import numpy as np
import glob

from tensorflow.keras.models import load_model

from utils.utils import load_array, compute_metrics, save_json
from model import Deeplabv3plus
from variables import *


def Test(model, test_dir: str, num_images_test: int, path_to_metrics: str, channels: int):
	print('Start testing...')
	
	test_data_dirs = glob.glob(test_dir + '/*.npy')
	np.random.shuffle(test_data_dirs)
	test_data_dirs = test_data_dirs[: num_images_test]

	batch_images = np.asarray([load_array(batch_file) for batch_file in test_data_dirs])
	batch_images = np.float32(batch_images) # set np.float32 to reduce memory usage

	x_test_total = batch_images[ :, :, :, : channels]
	y_test_total = batch_images[ :, :, :, channels :]

	print('Predicting...')
	y_pred_total = model.predict(x_test_total)

	metrics = compute_metrics(y_test_total.reshape(-1), np.round(y_pred_total).reshape(-1))

	accuracy = metrics.get('accuracy')
	print(f'Overall accuracy (number of correctly predicted items/total of item to predict): {accuracy}')

	avg_precision = metrics.get('average_precision')
	print(f'Average accuracy (the average of each accuracy per class(sum of accuracy for each class predicted/number of class)): {avg_precision}')

	precision = metrics.get('precision')
	print(f'Precision (how many of them are actual positive): {precision}')

	recall = metrics.get('recall')
	print(f'Recall (how many of the actual Positives our model capture through labeling it as Positive (True Positive)): {recall}')

	f1 = metrics.get('f1_score')
	print(f'F1 score: {f1}')

	save_json(metrics, path_to_metrics)


def Test_Case(test_dir: str, num_images_test: int, path_to_metrics: str, channels: int, is_model: bool = False,
			  patch_size: int = None, num_class: int = None, output_stride: int = None, path_to_load: str = None):

	if is_model: # load model
		print('Loading model.')
		model = load_model(path_to_load)

	else: # load weights
		print('Loading weights.')
		model = Deeplabv3plus(weights = None, input_tensor = None, input_shape = (patch_size, patch_size, channels),
							  classes = num_class, backbone = 'xception', OS = output_stride,
							  alpha = 1., activation = 'sigmoid')
		model.load_weights(path_to_load)

	return Test(model = model, test_dir = test_dir, num_images_test = num_images_test, path_to_metrics = path_to_metrics, channels = channels)


if __name__ == '__main__':

	one_channel = True
	
	is_model = False
	path_to_load = f'{MODELS_FOLDER}/DL_8_2_weights.h5'
	test_dir = f'{PROCESSED_FOLDER}/Fe19_stride512_onechannel{one_channel}_Test'
	path_to_metrics = f'{MODELS_FOLDER}/DL_8_2_metrics.json'
	channels = 3 if not one_channel else 1
	patch_size = 512
	num_class = 2
	output_stride = 8
	num_images_test = None

	Test_Case(test_dir = test_dir, num_images_test = num_images_test, path_to_metrics = path_to_metrics,
			  channels = channels, is_model = is_model, patch_size = patch_size, num_class = num_class,
			  output_stride = output_stride, path_to_load = None)
