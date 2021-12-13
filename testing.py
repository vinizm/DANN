import numpy as np
import glob

from tensorflow.keras.models import load_model

from utils.utils import load_array, compute_metrics, save_json
from model import Deeplabv3plus


def Test(test_dir: str, num_images_test: int, path_to_metrics: str, path_to_load: str, channels: int, is_model: bool = True,
		 patch_size: int = None, num_class: int = None, output_stride: int = None):
	print('Start testing...')

	if is_model: # load model
		model = load_model(path_to_load)

	else: # load weights
		model = Deeplabv3plus(weights = None, input_tensor = None, input_shape = (patch_size, patch_size, channels),
							  classes = num_class, backbone = 'xception', OS = output_stride,
							  alpha = 1., activation = 'sigmoid')
		model.load_weights(path_to_load)
	
	test_data_dirs = glob.glob(test_dir + '/*.npy')
	np.random.shuffle(test_data_dirs)
	test_data_dirs = test_data_dirs[: num_images_test]

	batch_images = np.asarray([load_array(batch_file) for batch_file in test_data_dirs])
	batch_images = np.float32(batch_images) # set np.float32 to reduce memory usage

	x_test_total = batch_images[ :, :, :, : channels]
	y_test_total = batch_images[ :, :, :, channels :]

	y_pred_total = model.predict(x_test_total)

	metrics = compute_metrics(y_test_total.flatten()), y_pred_total.flatten())

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
