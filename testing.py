import numpy as np
import glob

from utils.utils import load_array


def Test(test_dir: str, num_images_test: int, path_to_load: str, channels: int, is_model: bool = False):
	print('Start testing...')


	weights = load_model(path_to_load)
	
	test_data_dirs = glob.glob(test_dir + '/*.npy')
	np.random.shuffle(test_data_dirs)
	test_data_dirs = test_data_dirs[: num_images_test]

	batch_images = np.asarray([load_array(batch_file) for batch_file in test_data_dirs])
	batch_images = np.float32(batch_images) # set np.float32 to reduce memory usage

	x_test_total = batch_images[ :, :, :, : channels]
	y_test_total = batch_images[ :, :, :, channels :]

	y_pred_total = best_model.predict(x_test_total)

	matrix, accuracy, avg_accuracy, f1score, recall, prescision = compute_metrics(y_test_total.flatten(), y_pred_total.flatten())
	print('Overall accuracy (number of correctly predicted items/total of item to predict):', accuracy)
	print('Average accuracy (the average of each accuracy per class(sum of accuracy for each class predicted/number of class)):', avg_accuracy)
	print('Precision (how many of them are actual positive):', prescision)
	print('Recall (how many of the actual Positives our model capture through labeling it as Positive (True Positive)):', recall)
	print('F1 score:', f1score)

	matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
	print('Confusion matrix:')
	print(matrix)
