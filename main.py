import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import *
from tensorflow.keras.models import load_model
import glob
import os
import sys
import numpy as np
from skimage.transform import *
import time
import copy

from model import Deeplabv3plus
from utils.utils import load_array, compute_metrics, save_json, augment_images

from tensorflow.keras.callbacks import EarlyStopping


def Train(net, patches_dir: str, val_fraction: float, batch_size: int, num_images: int, channels: int,
		  epochs: int, wait: int, model_path: str, history_path: str, augment: bool):
	no_improvement_count = 0
	best_val_loss = 1.e8
	best_net = None
	history_train = []
	history_val = []

	# loading dataset
	data_dirs = glob.glob(patches_dir + '/*.npy')
	np.random.shuffle(data_dirs)
	data_dirs = data_dirs[: num_images] # reduce dataset size

	# define files for validation
	num_val_samples = int(len(data_dirs) * val_fraction)
	train_data_dirs = data_dirs[num_val_samples :]
	val_data_dirs = data_dirs[: num_val_samples]

	if augment:
		augmented_train = augment_images(image_files = train_data_dirs, angles = [90, 180, 270])
		train_data_dirs += augmented_train

		augmented_val = augment_images(image_files = val_data_dirs, angles = [90, 180, 270])
		val_data_dirs += augmented_val

		np.random.shuffle(train_data_dirs)
		np.random.shuffle(val_data_dirs)

	# compute number of batches
	num_batches = len(train_data_dirs) // batch_size
	num_batches_val = len(val_data_dirs) // batch_size
	print(f'num. of batches for training: {num_batches}')
	print(f'num. of batches for validation: {num_batches_val}')

	for epoch in range(epochs):
		print(f'Epoch {epoch + 1} of {epochs}')
		loss_train = np.zeros((1 , 2))
		loss_val = np.zeros((1 , 2))

		np.random.shuffle(train_data_dirs)
		np.random.shuffle(val_data_dirs)

		print('Start training...')
		for batch in range(num_batches):
			print(f'Batch {batch + 1} of {num_batches}')
			batch_files = train_data_dirs[batch * batch_size : (batch + 1) * batch_size]

			# load images for training
			batch_images = np.asarray([load_array(batch_file) for batch_file in batch_files])
			batch_images = batch_images.astype(np.float32) # set np.float32 to reduce memory usage

			x_train_batch = batch_images[ :, :, :, : channels]
			y_train_batch = batch_images[ :, :, :, channels :]
			
			# train network
			loss_train = loss_train + net.train_on_batch(x_train_batch, y_train_batch)
			
		loss_train = loss_train / num_batches
		print(f'train loss: {loss_train[0, 0]}')

		# evaluating network
		print('Start validation...')
		for batch in range(num_batches_val):
			print(f'Batch {batch + 1} of {num_batches_val}')
			batch_val_files = val_data_dirs[batch * batch_size : (batch + 1) * batch_size]

			# load images for testing
			batch_val_images = np.asarray([load_array(batch_val_file) for batch_val_file in batch_val_files])
			batch_val_images = batch_val_images.astype(np.float32) # set np.float32 to reduce memory usage

			x_val_batch = batch_val_images[:, :, :, : channels]
			y_val_batch = batch_val_images[:, :, :, channels :]

			# testing network
			loss_val = loss_val + net.test_on_batch(x_val_batch, y_val_batch)

		loss_val = loss_val / num_batches_val
		print(f'val loss: {loss_val[0, 0]}')

		# show results
		print(f'[Train loss: {loss_train[0, 0]}, Train acc.: {loss_train[0 , 1]}][Val loss: {loss_val[0 , 0]}, Val acc.: {loss_val[0 , 1]}]')
		history_train.append(loss_train)
		history_val.append(loss_val)

		if loss_val[0, 0] < best_val_loss:
			print('[!] Saving best model...')
			best_val_loss = loss_val[0, 0]
			no_improvement_count = 0
			# net.save(path_to_save) # save model
			net.save_weights(model_path) # save weights
			best_net = copy.deepcopy(net)

		else:
			no_improvement_count += 1
			if  no_improvement_count > wait:
				print('Performing early stopping!')
				break
	
	print('Saving metrics history.')
	history = np.asarray([history_train, history_val])
	persist = {'history': history.tolist(),
			   'image_files': {
				   'training':train_data_dirs,
				   'validation': val_data_dirs
			   }}
	save_json(persist, history_path)
	
	return best_net, history


def Predict(test_dir: str, num_images_test: int, path_to_load: str, channels: int):
	print("Start predicting...")
	best_model = load_model(path_to_load)
	
	test_data_dirs = glob.glob(test_dir + '/*.npy')
	np.random.shuffle(test_data_dirs)
	test_data_dirs = test_data_dirs[: num_images_test]

	batch_images = [load_array(batch_file) for batch_file in test_data_dirs]

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


def Train_Case(train_dir: str, test_dir: str, lr: float, patch_size: int, channels: int, num_class: int,
			 output_stride: int, epochs: int, batch_size: int, val_fraction: float, num_images_train: int,
			 num_images_test: int, patience: int, model_path: str, history_path: str, augment: bool):

	start = time.time()

	net = Deeplabv3plus(weights = None,
						input_tensor = None,
						input_shape = (patch_size, patch_size, channels),
						classes = num_class,
						backbone = 'xception',
						OS = output_stride,
						alpha = 1.,
						activation = 'sigmoid')
	
	adam = Adam(learning_rate = lr)
	net.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
	net.summary()

	# call train function
	Train(net = net, patches_dir = train_dir, val_fraction = val_fraction, batch_size = batch_size,
		  num_images = num_images_train, epochs = epochs, wait = patience, model_path = model_path,
		  history_path = history_path, channels = channels, augment = augment)

	end = time.time()
	hours, rem = divmod(end - start, 3600)
	minutes, seconds = divmod(rem, 60)
	print('Tempo Total: ' + '{:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds))
