import os
import copy
import glob
import time
import numpy as np

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow as tf

from utils.utils import load_array, save_json, augment_images
from architectures_functional import Deeplabv3plus


def run_training(net, patches_dir: str, val_fraction: float, batch_size: int, num_images: int, channels: int,
				 epochs: int, wait: int, model_path: str, history_path: str, rotate: bool, flip: bool,
				 loss_function, optimizer):
	no_improvement_count = 0
	best_val_loss = 1.e8
	best_net = None

	loss_train_history = []
	loss_val_history = []

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
		loss_train_value = 0.
		loss_val_value = 0.

		np.random.shuffle(train_data_dirs)
		np.random.shuffle(val_data_dirs)

		print('Start training...')
		for batch in range(num_batches_train):

			print(f'Batch {batch + 1} of {num_batches_train}')
			batch_files = train_data_dirs[batch * batch_size : (batch + 1) * batch_size]

			# load images for training
			batch_images = np.asarray([load_array(batch_file) for batch_file in batch_files])
			batch_images = batch_images.astype(np.float32) # set np.float32 to reduce memory usage

			x_train_batch = batch_images[ :, :, :, : channels]
			y_train_batch = batch_images[ :, :, :, channels :]
			
			# train network
			with tf.GradientTape() as tape:
				pred_train_batch = net(x_train_batch)
				loss_train = loss_function(y_train_batch, pred_train_batch)
			
			gradients = tape.gradient(loss_train, net.trainable_weights)
			optimizer.apply_gradients(zip(gradients, net.trainable_weights))

			loss_train_value += float(loss_train) # convert loss_train to float and sum

		loss_train_value = loss_train_value / num_batches_train
		loss_train_history.append(loss_train_value)
		print(f'Training Loss: {loss_train_value}')

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

			pred_val_batch = net(x_val_batch)
			loss_val = loss_function(y_val_batch, pred_val_batch)

			loss_val_value += float(loss_val) # convert loss_val to float and sum

		loss_val_value = loss_val_value / num_batches_val
		loss_val_history.append(loss_val_value)
		print(f'Validation Loss: {loss_val_value}')

		if loss_val_value < best_val_loss:
			print('[!] Saving best model...')
			best_val_loss = loss_val_value
			no_improvement_count = 0
			net.save_weights(model_path) # save weights
			best_net = copy.deepcopy(net)

		else:
			no_improvement_count += 1
			if  no_improvement_count > wait:
				print('[!] Performing early stopping.')
				break
	
	print('Saving metrics history.')
	persist = {'training':{'loss':loss_train_history},
			   'validation':{'loss':loss_val_history},
			   'image_files': {'training': train_data_dirs,
							   'validation': val_data_dirs}}
	save_json(persist, history_path)
	
	return best_net, persist


def Train(train_dir: str, learning_rate: float, patch_size: int, channels: int, num_class: int,
		  output_stride: int, epochs: int, batch_size: int, val_fraction: float, num_images_train: int,
		  patience: int, model_path: str, history_path: str, rotate: bool, flip: bool):

	start = time.time()

	net = Deeplabv3plus(weights = None, input_tensor = None, input_shape = (patch_size, patch_size, channels),
						classes = num_class, backbone = 'xception', output_stride = output_stride,
						alpha = 1., activation = 'softmax')
	net.summary()

	optimizer = Adam(learning_rate = learning_rate)
	loss_function = BinaryCrossentropy()

	# call train function
	run_training(net = net, patches_dir = train_dir, val_fraction = val_fraction, batch_size = batch_size,
				 num_images = num_images_train, epochs = epochs, wait = patience, model_path = model_path,
				 history_path = history_path, channels = channels, rotate = rotate, flip = flip,
				 loss_function = loss_function, optimmizer = optimizer)

	end = time.time()
	hours, rem = divmod(end - start, 3600)
	minutes, seconds = divmod(rem, 60)
	print('Tempo Total: ' + '{:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds))