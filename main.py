from utils.utils import load_array, compute_metrics
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import *
from tensorflow.keras.models import load_model
from tensorflow import keras
import glob
import cv2
import os
import tensorflow as tf
import sys


import numpy as np
from skimage.transform import *
from sklearn.utils import shuffle
import matplotlib.image as mpimg
import time
from model import Deeplabv3plus

from tensorflow.keras.callbacks import EarlyStopping


def Train(net, patches_dir: str, val_fraction: float, batch_size: int, num_images: int, epochs: int, wait: int, path_to_save: str):
	no_improvement_count = 0
	best_val_loss = 1.e8
	history_train = []
	history_val = []

	for epoch in range(epochs):
		loss_train = np.zeros((1 , 2))
		loss_val = np.zeros((1 , 2))

		# loading dataset
		data_dirs = glob.glob(patches_dir + '/*.npy')
		np.random.shuffle(data_dirs)
		data_dirs = data_dirs[: num_images] # reduce dataset size

		# define files for validation
		num_val_samples = int(len(data_dirs) * val_fraction)
		train_data_dirs = data_dirs[num_val_samples :]
		val_data_dirs = data_dirs[: num_val_samples]

		# compute number of batches
		num_batches = len(train_data_dirs) // batch_size
		num_batches_val = len(val_data_dirs) // batch_size

		for batch in range(num_batches):
			print('Start training...')
			print(f'Batch {batch + 1} of epoch {epoch + 1}')
			batch_files = train_data_dirs[batch * batch_size : (batch + 1) * batch_size]

			# load images for training
			batch_images = np.asarray([load_array(batch_file) for batch_file in batch_files])
			batch_images = batch_images.astype(np.float32) # set np.float32 to reduce memory usage

			x_train_batch = batch_images[ :, :, :, : 1]
			y_train_batch = batch_images[ :, :, :, 1 :]
			
			# train network
			loss_train = loss_train + net.train_on_batch(x_train_batch, y_train_batch)
			
		loss_train = loss_train / num_batches
		print(f'train loss: {loss_train[0, 0]}')

		# evaluating network
		for batch in range(num_batches_val):
			print('Start validation...')
			print(f'Batch {batch + 1} of epoch {epoch + 1}')
			batch_val_files = val_data_dirs[batch * batch_size : (batch + 1) * batch_size]

			# load images for testing
			batch_val_images = np.asarray([load_array(batch_val_file) for batch_val_file in batch_val_files])
			batch_val_images = batch_val_images.astype(np.float32) # set np.float32 to reduce memory usage

			x_val_batch = batch_val_images[:, :, :, : 1]
			y_val_batch = batch_val_images[:, :, :, 1 :]

			# testing network
			loss_val = loss_val + net.test_on_batch(x_val_batch, y_val_batch)

		loss_val = loss_val / num_batches_val
		print(f'val loss: {loss_val[0, 0]}')

		# show results
		print('%d [Train loss: %f, Train acc.: %.2f%%][Val loss: %f, Val acc.:%.2f%%]' %(epoch, loss_train[0, 0], 100 * loss_train[0 , 1] , loss_val[0 , 0] , 100 * loss_val[0 , 1]))
		history_train.append(loss_train)
		history_val.append(loss_val)

		if loss_val[0, 0] < best_val_loss:
			print('[!] Saving best model...')
			best_val_loss = loss_val[0, 0]
			no_improvement_count = 0
			net.save(path_to_save) # save model
			# model.save_weights(path_to_save)

		else:
			no_improvement_count += 1
			if  no_improvement_count > wait:
				print('Performing early stopping!')
				break
	
	history = [history_train, history_val]
	return history


def Predict(test_dir: str, num_images_test: int, path_to_load: str):
	print("Start predicting...")
	best_model = load_model(path_to_load)
	
	test_data_dirs = glob.glob(test_dir + '/*.npy')
	np.random.shuffle(test_data_dirs)
	test_data_dirs = test_data_dirs[: num_images_test]

	batch_images = [load_array(batch_file) for batch_file in test_data_dirs]

	x_test_total = batch_images[ :, :, :, : 1]
	y_test_total = batch_images[ :, :, :, 1 :]

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


def run_case(train_dir: str, test_dir: str, patch_size: int, channels: int, num_class: int,
			 output_stride: int, epochs: int, batch_size: int, val_fraction: float, num_images_train: int,
			 num_images_test: int, patience: int, path_to_save: str):

	start = time.time()

	net = Deeplabv3plus(weights = None,
						input_tensor = None,
						input_shape = (patch_size, patch_size, channels),
						classes = num_class,
						backbone = 'xception',
						OS = output_stride,
						alpha = 1.,
						activation = 'sigmoid')
	
	adam = Adam(learning_rate = 1e-4)
	net.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
	# net.summary()

	# call train function
	Train(net = net, patches_dir = train_dir, val_fraction = val_fraction, batch_size = batch_size,
		  num_images = num_images_train, epochs = epochs, wait = patience, path_to_save = path_to_save)
	
	# Predict(test_dir = test_dir, num_images_test = num_images_test, path_to_load = path_to_save)

	end = time.time()
	hours, rem = divmod(end - start, 3600)
	minutes, seconds = divmod(rem, 60)
	print('Tempo Total: ' + '{:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds))





# if __name__=='__main__':
# 	train = sys.argv[1] # training patches directory
# 	val = sys.argv[2] # validation patches directory
# 	test = sys.argv[3] # test patches directory
# 	patch_size = sys.argv[4] # patch size
# 	channels = sys.argv[5] # channels
# 	nclass= sys.argv[6] # number of classes
# 	output_stride = sys.argv[7] # output stride
# 	epoch = sys.argv[8] # epochs
# 	patience = sys.argv[9] # patience
