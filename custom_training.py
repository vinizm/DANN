import os
import copy
import glob
import time
from unittest.mock import patch
import numpy as np

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.models import save_model
import tensorflow as tf

from utils.utils import load_array, save_json, augment_images
from architectures_functional import Deeplabv3plus


class Trainer():

	def __init__(self, patch_size: int = 512, channels: int = 1, num_class: int = 2, output_stride: int = 8,
				 learning_rate: float = 1e-3):

		self.patch_size = patch_size
		self.channels = channels
		self.num_class = num_class
		self.output_stride = output_stride
		self.learning_rate = learning_rate

		self.model = Deeplabv3plus(input_shape = (patch_size, patch_size, channels), classes = num_class,
								   output_stride = output_stride, activation = 'softmax', classifier_position = None)
		self.optimizer = Adam(learning_rate = learning_rate)
		
		self.loss_function = BinaryCrossentropy()
		self.acc_function = Accuracy(name = 'accuracy', dtype = None)

		self.loss_train_history = []
		self.loss_val_history = []

		self.acc_train_history = []
		self.acc_val_history = []

		self.no_improvement_count = 0
		self.best_val_loss = 1.e8
		self.best_model = None

		self.patches_dir = None
		self.train_data_dirs = None
		self.val_data_dirs = None
		self.val_fraction = 0.1
		self.batch_size = 2
		self.num_images = 60
		self.epochs = 25
		self.wait = 12
		self.rotate = True
		self.flip = True

	@tf.function
	def _training_step(self, x_train, y_train):

		with tf.GradientTape() as tape:
			pred_train = self.model(x_train)
			loss = self.loss_function(y_train, pred_train)
		
		gradients = tape.gradient(loss, self.model.trainable_weights)
		self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

		binary_prediction = tf.math.round(pred_train)
		acc = float(self.acc_function(y_train, binary_prediction))
		
		return loss, acc

	def _augment_images(self):
		if self.rotate or self.flip:
			augmented_train = augment_images(image_files = self.train_data_dirs, angles = [90, 180, 270],
											 rotate = self.rotate, flip = self.flip)
			self.train_data_dirs += augmented_train

			augmented_val = augment_images(image_files = self.val_data_dirs, angles = [90, 180, 270],
										   rotate = self.rotate, flip = self.flip)
			self.val_data_dirs += augmented_val

			np.random.shuffle(self.train_data_dirs)
			np.random.shuffle(self.val_data_dirs)

	def compile_model(self, show_summary: bool = True):
		self.model.compile(optimizer = self.optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

		if show_summary:
			self.model.summary()

	def train(self, patches_dir: str, epochs: int = 25, batch_size: int = 2, val_fraction: float = 0.1, num_images: int = 60,
			  wait: int = 12, rotate: bool = True, flip: bool = True):

		self.patches_dir = patches_dir
		self.val_fraction = val_fraction
		self.batch_size = batch_size
		self.num_images = num_images
		self.epochs = epochs
		self.wait = wait
		self.rotate = rotate
		self.flip = flip

		# loading dataset
		data_dirs = glob.glob(self.patches_dir + '/*.npy')
		np.random.shuffle(data_dirs)
		data_dirs = data_dirs[: self.num_images] # reduce dataset size

		# define files for validation
		num_val_samples = int(len(data_dirs) * self.val_fraction)
		self.train_data_dirs = data_dirs[num_val_samples :]
		self.val_data_dirs = data_dirs[: num_val_samples]

		# data augmentation
		self._augment_images()

		# compute number of batches
		num_batches_train = len(self.train_data_dirs) // self.batch_size
		num_batches_val = len(self.val_data_dirs) // self.batch_size
		print(f'num. of batches for training: {num_batches_train}')
		print(f'num. of batches for validation: {num_batches_val}')

		for epoch in range(self.epochs):
			print(f'Epoch {epoch + 1} of {self.epochs}')
			loss_global_train = 0.
			loss_global_val = 0.

			acc_global_train = 0.
			acc_global_val = 0.

			np.random.shuffle(self.train_data_dirs)
			np.random.shuffle(self.val_data_dirs)

			print('Start training...')
			for batch in range(num_batches_train):

				print(f'Batch {batch + 1} of {num_batches_train}')
				batch_files = self.train_data_dirs[batch * self.batch_size : (batch + 1) * self.batch_size]

				# load images for training
				batch_images = np.asarray([load_array(batch_file) for batch_file in batch_files])
				batch_images = batch_images.astype(np.float32) # set np.float32 to reduce memory usage

				x_train = batch_images[ :, :, :, : self.channels]
				y_train = batch_images[ :, :, :, self.channels :]

				loss_train, acc_train = self._training_step(x_train, y_train)
				loss_global_train += float(loss_train)
				acc_global_train += float(acc_train)

			loss_global_train /= num_batches_train
			self.loss_train_history.append(loss_global_train)

			acc_global_train /= num_batches_train
			self.acc_train_history.append(acc_global_train)

			print(f'Training Loss: {loss_global_train}')
			print(f'Training Accuracy: {acc_global_train}')

			# evaluating network
			print('Start validation...')
			for batch in range(num_batches_val):
				print(f'Batch {batch + 1} of {num_batches_val}')
				batch_val_files = self.val_data_dirs[batch * self.batch_size : (batch + 1) * self.batch_size]

				# load images for testing
				batch_val_images = np.asarray([load_array(batch_val_file) for batch_val_file in batch_val_files])
				batch_val_images = batch_val_images.astype(np.float32) # set np.float32 to reduce memory usage

				x_val = batch_val_images[:, :, :, : self.channels]
				y_val = batch_val_images[:, :, :, self.channels :]

				pred_val = self.model(x_val)
				loss_val = self.loss_function(y_val, pred_val)

				loss_global_val += float(loss_val) # convert loss_val to float and sum

				binary_prediction = tf.math.round(pred_val)
				acc_global_val += float(self.acc_function(y_val, binary_prediction))

			loss_global_val /= num_batches_val
			self.loss_val_history.append(loss_global_val)

			acc_global_val /= num_batches_val
			self.acc_val_history.append(acc_global_val)

			print(f'Validation Loss: {loss_global_val}')
			print(f'Validation Accuracy: {acc_global_val}')

			if loss_global_val < self.best_val_loss:
				print('[!] Persisting best model...')
				self.best_val_loss = loss_global_val
				self.no_improvement_count = 0
				self.best_model = copy.deepcopy(self.model)

			else:
				self.no_improvement_count += 1
				if  self.no_improvement_count > self.wait:
					print('[!] Performing early stopping.')
					break

	def save_weights(self, weights_path: str):
		self.model.save_weights(weights_path) # save weights
		print('Weights saved successfuly.')

	def save_model(self, model_path: str):
		save_model(self.save_model, model_path) # save model
		print('Weights saved successfuly.')		
		
	def save_info(self, history_path):
		persist = {'history':{'training':{'loss': self.loss_train_history,
							  			  'accuracy': self.acc_train_history},
							  'validation':{'loss': self.loss_val_history,
							  				'accuracy': self.acc_val_history}},
				   'image_files':{'training': self.train_data_dirs,
								  'validation': self.val_data_dirs}}
		save_json(persist, history_path) # save metrics and parameters
		print('Metrcis saved successfuly.')
