import os
import copy
import glob
import time
import numpy as np

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Accuracy
import tensorflow as tf

from utils.utils import load_array, save_json, augment_images
from architectures_functional import Deeplabv3plus


def run_training(model, patches_dir: str, val_fraction: float, batch_size: int, num_images: int, channels: int,
				 epochs: int, wait: int, model_path: str, history_path: str, rotate: bool, flip: bool,
				 loss_function, optimizer):
	no_improvement_count = 0
	best_val_loss = 1.e8
	best_net = None

	loss_train_history = []
	loss_val_history = []

	acc_train_history = []
	acc_val_history = []
	acc_function = Accuracy(name = 'accuracy', dtype = None)

	@tf.function
	def training_step(x_train, y_train):
		# train network
		with tf.GradientTape() as tape:
			pred_train = model(x_train)
			loss_raw = loss_function(y_train, pred_train)
		
		gradients = tape.gradient(loss_raw, model.trainable_weights)
		optimizer.apply_gradients(zip(gradients, model.trainable_weights))

		loss = float(loss_raw) # convert loss_train to float and sum
		binary_prediction = tf.math.round(pred_train)
		acc = acc_function(y_train, binary_prediction)
		
		return loss, acc

	# loading dataset
	data_dirs = glob.glob(patches_dir + '/*.npy')
	np.random.shuffle(data_dirs)
	data_dirs = data_dirs[: num_images] # reduce dataset size

	# define files for validation
	num_val_samples = int(len(data_dirs) * val_fraction)
	train_data_dirs = data_dirs[num_val_samples :]
	val_data_dirs = data_dirs[: num_val_samples]

	if rotate or flip:
		augmented_train = augment_images(image_files = train_data_dirs, angles = [90, 180, 270], rotate = rotate, flip = flip)
		train_data_dirs += augmented_train

		augmented_val = augment_images(image_files = val_data_dirs, angles = [90, 180, 270], rotate = rotate, flip = flip)
		val_data_dirs += augmented_val

		np.random.shuffle(train_data_dirs)
		np.random.shuffle(val_data_dirs)

	# compute number of batches
	num_batches_train = len(train_data_dirs) // batch_size
	num_batches_val = len(val_data_dirs) // batch_size
	print(f'num. of batches for training: {num_batches_train}')
	print(f'num. of batches for validation: {num_batches_val}')

	for epoch in range(epochs):
		print(f'Epoch {epoch + 1} of {epochs}')
		loss_global_train = 0.
		loss_global_val = 0.

		acc_global_train = 0.
		acc_global_val = 0.

		np.random.shuffle(train_data_dirs)
		np.random.shuffle(val_data_dirs)

		print('Start training...')
		for batch in range(num_batches_train):

			print(f'Batch {batch + 1} of {num_batches_train}')
			batch_files = train_data_dirs[batch * batch_size : (batch + 1) * batch_size]

			# load images for training
			batch_images = np.asarray([load_array(batch_file) for batch_file in batch_files])
			batch_images = batch_images.astype(np.float32) # set np.float32 to reduce memory usage

			x_train = batch_images[ :, :, :, : channels]
			y_train = batch_images[ :, :, :, channels :]

			loss_train, acc_train = training_step(x_train, y_train)
			loss_global_train += loss_train
			acc_global_train += acc_train

		loss_global_train /= num_batches_train
		loss_train_history.append(loss_global_train)

		acc_global_train /= num_batches_train
		acc_train_history.append(acc_global_train)

		print(f'Training Loss: {loss_global_train}')
		print(f'Training Accuracy: {acc_global_train}')

		# evaluating network
		print('Start validation...')
		for batch in range(num_batches_val):
			print(f'Batch {batch + 1} of {num_batches_val}')
			batch_val_files = val_data_dirs[batch * batch_size : (batch + 1) * batch_size]

			# load images for testing
			batch_val_images = np.asarray([load_array(batch_val_file) for batch_val_file in batch_val_files])
			batch_val_images = batch_val_images.astype(np.float32) # set np.float32 to reduce memory usage

			x_val = batch_val_images[:, :, :, : channels]
			y_val = batch_val_images[:, :, :, channels :]

			pred_val = model(x_val)
			loss_val = loss_function(y_val, pred_val)

			loss_global_val += float(loss_val) # convert loss_val to float and sum

			binary_prediction = tf.math.round(pred_val)
			acc_global_val += acc_function(y_val, binary_prediction)

		loss_global_val /= num_batches_val
		loss_val_history.append(loss_global_val)

		acc_global_val /= num_batches_val
		acc_val_history.append(acc_global_val)

		print(f'Validation Loss: {loss_global_val}')
		print(f'Validation Accuracy: {acc_global_val}')

		if loss_global_val < best_val_loss:
			print('[!] Saving best model...')
			best_val_loss = loss_global_val
			no_improvement_count = 0
			model.save_weights(model_path) # save weights
			best_net = copy.deepcopy(model)

		else:
			no_improvement_count += 1
			if  no_improvement_count > wait:
				print('[!] Performing early stopping.')
				break
	
	print('Saving metrics history.')
	persist = {'training':{'loss':loss_train_history,
						   'accuracy':acc_train_history},
			   'validation':{'loss':loss_val_history,
			   				 'accuracy':acc_val_history},
			   'image_files': {'training': train_data_dirs,
							   'validation': val_data_dirs}}
	save_json(persist, history_path)
	
	return best_net, persist


def Train(train_dir: str, learning_rate: float, patch_size: int, channels: int, num_class: int,
		  output_stride: int, epochs: int, batch_size: int, val_fraction: float, num_images_train: int,
		  patience: int, model_path: str, history_path: str, rotate: bool, flip: bool):

	start = time.time()

	model = Deeplabv3plus(input_shape = (patch_size, patch_size, channels), classes = num_class,
						output_stride = output_stride, activation = 'softmax', classifier_position = None)
	model.summary()

	optimizer = Adam(learning_rate = learning_rate)
	loss_function = BinaryCrossentropy()

	model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
	# call train function
	run_training(model = model, patches_dir = train_dir, val_fraction = val_fraction, batch_size = batch_size,
				 num_images = num_images_train, epochs = epochs, wait = patience, model_path = model_path,
				 history_path = history_path, channels = channels, rotate = rotate, flip = flip,
				 loss_function = loss_function, optimizer = optimizer)

	end = time.time()
	hours, rem = divmod(end - start, 3600)
	minutes, seconds = divmod(rem, 60)
	print('Tempo Total: ' + '{:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds))