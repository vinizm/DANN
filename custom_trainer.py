import glob
import time
import numpy as np

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, SparseCategoricalAccuracy
from tensorflow.keras.models import save_model
import tensorflow as tf

from utils.utils import load_array, save_json, augment_images
from utils.hyperparameters import *
from utils.hyperparameters import LambdaGradientReversalLayer
from utils.loss_functions import MaskedBinaryCrossentropy
from utils.learning_rate_functions import LearningRateFactory as lrf
from models.builder import DomainAdaptationModel
from models.deeplabv3plus import DeepLabV3Plus


class Trainer():

	def __init__(self, patch_size: int = 512, channels: int = 1, num_class: int = 2, output_stride: int = 8,
				backbone_size: int = 16, domain_adaptation: bool = False):

		self.patch_size = patch_size
		self.channels = channels
		self.num_class = num_class
		self.output_stride = output_stride
		self.backbone_size = backbone_size
		self.domain_adaptation = domain_adaptation

		self.model = self.assembly_empty_model()
		
		self.optimizer_segmentation = Adam()
		self.optimizer_discriminator = Adam()
		
		self.lr_factory = lrf()

		self.lr_function_segmentation = self.lr_factory.get_function('exp_decay', lr0 = LR0, warmup = LR_WARMUP, alpha = ALPHA, beta = BETA)
		self.lr_function_discriminator = self.lr_factory.get_function('exp_decay', lr0 = LR0, warmup = LR_WARMUP, alpha = ALPHA, beta = BETA)

		self.lambda_function = LambdaGradientReversalLayer(warmup = 0., gamma = 10., lambda_scale = 1.)

		self.loss_function = BinaryCrossentropy()
		self.loss_function_segmentation = MaskedBinaryCrossentropy()
		self.loss_function_discriminator = SparseCategoricalCrossentropy()

		self.acc_function_segmentation = CategoricalAccuracy()
		self.acc_function_discriminator = SparseCategoricalAccuracy()

		self.loss_segmentation_train_history = []
		self.loss_segmentation_val_history = []

		self.loss_discriminator_train_history = []
		self.loss_discriminator_val_history = []

		self.acc_segmentation_train_history = []
		self.acc_segmentation_val_history = []

		self.acc_discriminator_train_history = []
		self.acc_discriminator_val_history = []

		self.lr_segmentation_history = []
		self.lr_discriminator_history = []
		self.lambdas = []

		self.train_writer = tf.summary.create_file_writer('logs/train/')
		self.val_writer = tf.summary.create_file_writer('logs/validation/')

		self.no_improvement_count = 0
		self.best_val_loss = 1.e8
		self.best_model = None

		self.patches_dir = None
		self.train_data_dirs = []
		self.val_data_dirs = []

		self.train_data_dirs_source = []
		self.val_data_dirs_source = []

		self.train_data_dirs_target = []
		self.val_data_dirs_target = []

		self.val_fraction = 0.1
		self.batch_size = 2
		self.num_images = 60
		self.epochs = 25
		self.wait = 12
		self.rotate = True
		self.flip = True

		self.num_batches_train = None
		self.num_batches_val = None

		self.test_index_source = []
		self.test_index_target = []

		self.elapsed_time = 0

	def assembly_empty_model(self):

		if self.domain_adaptation:
			empty_model = DomainAdaptationModel(input_shape = (self.patch_size, self.patch_size, self.channels), num_class = self.num_class,
						 output_stride = self.output_stride, backbone_size = self.backbone_size, activation = 'softmax')
		else:
			empty_model = DeepLabV3Plus(input_shape = (self.patch_size, self.patch_size, self.channels), num_class = self.num_class,
						 output_stride = self.output_stride, activation = 'softmax', backbone_size = self.backbone_size, domain_adaptation = False)
		
		return empty_model

	@tf.function
	def _training_step(self, x_train, y_train):

		with tf.GradientTape() as tape:
			pred_train = self.model(x_train)
			loss = self.loss_function(y_train, pred_train)
		
		gradients = tape.gradient(loss, self.model.trainable_weights)
		self.optimizer_segmentation.apply_gradients(zip(gradients, self.model.trainable_weights))

		self.acc_function_segmentation.update_state(y_train, pred_train)
		
		return loss

	@tf.function
	def _training_step_domain_adaptation(self, inputs, outputs, loss_mask, acc_mask, train_segmentation = True,
										 train_discriminator = True):

		y_true_segmentation, y_true_discriminator = outputs
		with tf.GradientTape(persistent = True) as tape:
			y_pred_segmentation, y_pred_discriminator = self.model(inputs)

			loss_segmentation = self.loss_function_segmentation(y_true_segmentation, y_pred_segmentation, loss_mask)
			loss_discriminator = self.loss_function_discriminator(y_true_discriminator, y_pred_discriminator)
			loss_global = loss_segmentation + loss_discriminator

		if train_segmentation:
			gradients_segmentation = tape.gradient(loss_global, self.model.main_network.trainable_weights)
			self.optimizer_segmentation.apply_gradients(zip(gradients_segmentation, self.model.main_network.trainable_weights))

			self.acc_function_segmentation.update_state(y_true_segmentation, y_pred_segmentation, sample_weight = acc_mask)

		if train_discriminator:
			gradients_discriminator = tape.gradient(loss_discriminator, self.model.domain_discriminator.trainable_weights)
			self.optimizer_discriminator.apply_gradients(zip(gradients_discriminator, self.model.domain_discriminator.trainable_weights))

			y_true_discriminator = tf.expand_dims(y_true_discriminator, axis = -1)
			self.acc_function_discriminator.update_state(y_true_discriminator, y_pred_discriminator)

		del tape

		return loss_segmentation, loss_discriminator

	def _augment_images(self, data_dirs: list):
		if self.rotate or self.flip:
			augmented_dirs = augment_images(image_files = data_dirs, angles = [90, 180, 270],
											 rotate = self.rotate, flip = self.flip, verbose = True)
			data_dirs += augmented_dirs
			np.random.shuffle(data_dirs)

		return data_dirs

	def compile_model(self, show_summary: bool = True):
		self.model.compile(optimizer = self.optimizer_segmentation)

		if show_summary:
			self.model.summary()

	def _split_file_list(self, data_dirs: list):

		num_val_samples = int(len(data_dirs) * self.val_fraction)
		train_data_dirs = data_dirs[num_val_samples :]
		val_data_dirs = data_dirs[: num_val_samples]

		return train_data_dirs, val_data_dirs

	def _load_file_names(self, patches_dir: str):

		data_dirs = glob.glob(patches_dir + '/*.npy')
		np.random.shuffle(data_dirs)
		data_dirs = data_dirs[: self.num_images] # reduce dataset size

		return data_dirs

	def _calculate_batch_size(self, data_dirs: list):
		num_batches = len(data_dirs) // self.batch_size
		return num_batches

	@staticmethod
	def _convert_path_to_domain(file_names: list, source_files: list):
		return np.asarray([0 if file_name in source_files else 1 for file_name in file_names], dtype = 'int32')

	@staticmethod
	def _generate_sample_mask(domain: int, shape: tuple):
		if domain == 0:
			return np.full(shape, 1., dtype = 'float32')
		return np.full(shape, 0., dtype = 'float32')

	def _generate_segmentation_mask(self, domain: int):
		return self._generate_sample_mask(domain, (self.patch_size, self.patch_size))

	def _generate_discriminator_mask(self, domain: int):
		return self._generate_sample_mask(domain, (1,))

	def _generate_loss_mask(self, samples: np.ndarray):
		return np.asarray([self._generate_segmentation_mask(domain) for domain in samples])

	def _generate_acc_mask(self, samples: list):
		return np.asarray([self._generate_discriminator_mask(domain) for domain in samples]).reshape(-1)

	def reset_history(self):
		self.acc_function_segmentation.reset_states()
		self.acc_function_discriminator.reset_states()

		self.loss_segmentation_train_history = []
		self.loss_segmentation_val_history = []

		self.loss_discriminator_train_history = []
		self.loss_discriminator_val_history = []

		self.acc_segmentation_train_history = []
		self.acc_segmentation_val_history = []

		self.acc_discriminator_train_history = []
		self.acc_discriminator_val_history = []

		self.lr_segmentation_history = []
		self.lr_discriminator_history = []
		self.lambdas = []

		self.no_improvement_count = 0
		self.best_val_loss = 1.e8

	def preprocess_images_domain_adaptation(self, patches_dir: list, batch_size: int = 2, val_fraction: float = 0.1,
		num_images: int = 60, rotate: bool = True, flip: bool = True):

		self.patches_dir = patches_dir # list: [source, target]
		self.val_fraction = val_fraction
		self.batch_size = batch_size
		self.num_images = num_images
		self.rotate = rotate
		self.flip = flip
	
		# loading source dataset
		data_dirs_source = self._load_file_names(self.patches_dir[0])
		# loading target dataset
		data_dirs_target = self._load_file_names(self.patches_dir[1])
		
		# define files of source domain
		train_data_dirs_source, val_data_dirs_source = self._split_file_list(data_dirs_source)
		# define files of target domain
		train_data_dirs_target, val_data_dirs_target = self._split_file_list(data_dirs_target)

		# augment source images
		self.train_data_dirs_source = self._augment_images(train_data_dirs_source)
		self.val_data_dirs_source = self._augment_images(val_data_dirs_source)

		# augment target images
		self.train_data_dirs_target = self._augment_images(train_data_dirs_target)
		self.val_data_dirs_target = self._augment_images(val_data_dirs_target)

		# merge databases
		self.train_data_dirs = train_data_dirs_source + train_data_dirs_target
		self.val_data_dirs = val_data_dirs_source + val_data_dirs_target

		# shuffle final database
		np.random.shuffle(self.train_data_dirs)
		np.random.shuffle(self.val_data_dirs)

		# compute number of batches
		self.num_batches_train = self._calculate_batch_size(self.train_data_dirs)
		self.num_batches_val = self._calculate_batch_size(self.val_data_dirs)

		print(f'num. of batches for training: {self.num_batches_train}')
		print(f'num. of batches for validation: {self.num_batches_val}')

	def preprocess_images(self, patches_dir: str, batch_size: int = 2, val_fraction: float = 0.1,
		num_images: int = 60, rotate: bool = True, flip: bool = True):

		self.patches_dir = patches_dir
		self.val_fraction = val_fraction
		self.batch_size = batch_size
		self.num_images = num_images
		self.rotate = rotate
		self.flip = flip
	
		# loading dataset
		data_dirs = self._load_file_names(self.patches_dir)

		# define files for validation
		self.train_data_dirs, self.val_data_dirs = self._split_file_list(data_dirs)

		# data augmentation
		self.train_data_dirs = self._augment_images(self.train_data_dirs)
		self.val_data_dirs = self._augment_images(self.val_data_dirs)

		# compute number of batches
		self.num_batches_train = self._calculate_batch_size(self.train_data_dirs)
		self.num_batches_val = self._calculate_batch_size(self.val_data_dirs)

		print(f'num. of batches for training: {self.num_batches_train}')
		print(f'num. of batches for validation: {self.num_batches_val}')

	def pretrain_segmentation(self, max_epoch: int = 50, min_acc: float = 0.8):
		self.reset_history()

		epoch = 0
		acc_segmentation_val = 0.

		while epoch < max_epoch and acc_segmentation_val < min_acc:
			print(f'Epoch {epoch + 1} of {max_epoch}')
			loss_segmentation_train = 0.
			loss_segmentation_val = 0.

			self.acc_function_segmentation.reset_states()

			np.random.shuffle(self.train_data_dirs)
			np.random.shuffle(self.val_data_dirs)

			# update learning rate
			p = epoch / (max_epoch - 1)
			print(f'Training Progress: {p}')
			
			lr_1 = self.lr_function_segmentation.calculate(p)
			print(f'Learning Rate Segmentation: {lr_1}')
			self.optimizer_segmentation.lr = lr_1
			self.lr_segmentation_history.append(lr_1)

			l_vector = np.full((self.batch_size, 1), 0., dtype = 'float32')

			print('Start training...')
			for batch in range(self.num_batches_train):
				print(f'Batch {batch + 1} of {self.num_batches_train}')
				batch_train_files = self.train_data_dirs[batch * self.batch_size : (batch + 1) * self.batch_size]

				# load images for training
				batch_images = np.asarray([load_array(batch_train_file, verbose = False) for batch_train_file in batch_train_files])
				batch_images = batch_images.astype(np.float32) # set np.float32 to reduce memory usage

				x_train = batch_images[ :, :, :, : self.channels]
				y_segmentation_train = batch_images[ :, :, :, self.channels :]
				y_discriminator_train = self._convert_path_to_domain(batch_train_files, self.train_data_dirs_source)
				print(f'Domain: {y_discriminator_train}')

				loss_mask = self._generate_loss_mask(y_discriminator_train)
				acc_mask = self._generate_acc_mask(y_discriminator_train)

				step_output = self._training_step_domain_adaptation([x_train, l_vector], [y_segmentation_train, y_discriminator_train], loss_mask, acc_mask,
																	train_segmentation = True, train_discriminator = False)
				loss_segmentation, _ = step_output
				loss_segmentation_train += float(loss_segmentation)
	
			loss_segmentation_train /= self.num_batches_train
			self.loss_segmentation_train_history.append(loss_segmentation_train)

			acc_segmentation_train = float(self.acc_function_segmentation.result())
			self.acc_segmentation_train_history.append(acc_segmentation_train)		

			print(f'Segmentation Loss: {loss_segmentation_train}')
			print(f'Segmentation Accuracy: {acc_segmentation_train}')

			self.acc_function_segmentation.reset_states()

			# evaluating network
			print('Start validation...')
			for batch in range(self.num_batches_val):
				print(f'Batch {batch + 1} of {self.num_batches_val}')
				batch_val_files = self.val_data_dirs[batch * self.batch_size : (batch + 1) * self.batch_size]

				# load images for testing
				batch_val_images = np.asarray([load_array(batch_val_file, verbose = False) for batch_val_file in batch_val_files])
				batch_val_images = batch_val_images.astype(np.float32) # set np.float32 to reduce memory usage

				x_val = batch_val_images[:, :, :, : self.channels]
				y_segmentation_val = batch_val_images[:, :, :, self.channels :]
				y_discriminator_val = self._convert_path_to_domain(batch_val_files, self.val_data_dirs_source)
				print(f'Domain: {y_discriminator_val}')

				y_segmentation_pred, _ = self.model([x_val, l_vector])

				loss_mask = self._generate_loss_mask(y_discriminator_val)
				acc_mask = self._generate_acc_mask(y_discriminator_val)

				loss_segmentation = self.loss_function_segmentation(y_segmentation_val, y_segmentation_pred, loss_mask)		
				self.acc_function_segmentation.update_state(y_segmentation_val, y_segmentation_pred, sample_weight = acc_mask)

				loss_segmentation_val += float(loss_segmentation)

			loss_segmentation_val /= self.num_batches_val
			self.loss_segmentation_val_history.append(loss_segmentation_val)

			acc_segmentation_val = float(self.acc_function_segmentation.result())
			self.acc_segmentation_val_history.append(acc_segmentation_val)

			print(f'Segmentation Loss: {loss_segmentation_val}')
			print(f'Segmentation Accuracy: {acc_segmentation_val}')

			epoch += 1

	def pretrain_discriminator(self, max_epoch: int = 500, min_acc: float = 0.9):
		self.reset_history()

		epoch = 0
		acc_discriminator_val = 0.

		while epoch < max_epoch and acc_discriminator_val < min_acc:
			print(f'Epoch {epoch + 1} of {max_epoch}')
			loss_discriminator_train = 0.
			loss_discriminator_val = 0.

			self.acc_function_discriminator.reset_states()

			np.random.shuffle(self.train_data_dirs)
			np.random.shuffle(self.val_data_dirs)

			# update learning rate
			p = epoch / (max_epoch - 1)
			print(f'Training Progress: {p}')

			lr_2 = self.lr_function_discriminator.calculate(p)
			print(f'Learning Rate Discriminator: {lr_2}')
			self.optimizer_discriminator.lr = lr_2
			self.lr_discriminator_history.append(lr_2)

			l_vector = np.full((self.batch_size, 1), 0., dtype = 'float32')

			print('Start training...')
			for batch in range(self.num_batches_train):
				print(f'Batch {batch + 1} of {self.num_batches_train}')
				batch_train_files = self.train_data_dirs[batch * self.batch_size : (batch + 1) * self.batch_size]

				# load images for training
				batch_images = np.asarray([load_array(batch_train_file, verbose = False) for batch_train_file in batch_train_files])
				batch_images = batch_images.astype(np.float32) # set np.float32 to reduce memory usage

				x_train = batch_images[ :, :, :, : self.channels]
				y_segmentation_train = batch_images[ :, :, :, self.channels :]
				y_discriminator_train = self._convert_path_to_domain(batch_train_files, self.train_data_dirs_source)
				print(f'Domain: {y_discriminator_train}')

				loss_mask = self._generate_loss_mask(y_discriminator_train)
				acc_mask = self._generate_acc_mask(y_discriminator_train)

				step_output = self._training_step_domain_adaptation([x_train, l_vector], [y_segmentation_train, y_discriminator_train], loss_mask, acc_mask,
																	train_segmentation = False, train_discriminator = True)
				_, loss_discriminator = step_output
				loss_discriminator_train += float(loss_discriminator)

			loss_discriminator_train /= self.num_batches_train
			self.loss_discriminator_train_history.append(loss_discriminator_train)
			
			acc_discriminator_train = float(self.acc_function_discriminator.result())
			self.acc_discriminator_train_history.append(acc_discriminator_train)			

			print(f'Discriminator Loss: {loss_discriminator_train}')
			print(f'Discriminator Accuracy: {acc_discriminator_train}')

			self.acc_function_discriminator.reset_states()	

			# evaluating network
			print('Start validation...')
			for batch in range(self.num_batches_val):
				print(f'Batch {batch + 1} of {self.num_batches_val}')
				batch_val_files = self.val_data_dirs[batch * self.batch_size : (batch + 1) * self.batch_size]

				# load images for testing
				batch_val_images = np.asarray([load_array(batch_val_file, verbose = False) for batch_val_file in batch_val_files])
				batch_val_images = batch_val_images.astype(np.float32) # set np.float32 to reduce memory usage

				x_val = batch_val_images[:, :, :, : self.channels]
				y_discriminator_val = self._convert_path_to_domain(batch_val_files, self.val_data_dirs_source)
				print(f'Domain: {y_discriminator_val}')

				_, y_discriminator_pred = self.model([x_val, l_vector])
				loss_discriminator = self.loss_function_discriminator(y_discriminator_val, y_discriminator_pred)			

				y_discriminator_val = tf.expand_dims(y_discriminator_val, axis = -1)
				self.acc_function_discriminator.update_state(y_discriminator_val, y_discriminator_pred)

				loss_discriminator_val += float(loss_discriminator)

			loss_discriminator_val /= self.num_batches_val
			self.loss_discriminator_val_history.append(loss_discriminator_val)

			acc_discriminator_val = float(self.acc_function_discriminator.result())
			self.acc_discriminator_val_history.append(acc_discriminator_val)

			print(f'Discriminator Loss: {loss_discriminator_val}')
			print(f'Discriminator Accuracy: {acc_discriminator_val}')

			epoch += 1

	def train_domain_adaptation(self, epochs: int = 25, wait: int = 12, persist_best_model: bool = True):

		self.reset_history()
		time_init = time.time()

		self.epochs = epochs
		self.wait = wait

		for epoch in range(self.epochs):
			print(f'Epoch {epoch + 1} of {self.epochs}')
			loss_segmentation_train = 0.
			loss_segmentation_val = 0.

			loss_discriminator_train = 0.
			loss_discriminator_val = 0.

			self.acc_function_segmentation.reset_states()
			self.acc_function_discriminator.reset_states()

			np.random.shuffle(self.train_data_dirs)
			np.random.shuffle(self.val_data_dirs)

			# update learning rate
			p = epoch / (epochs - 1)
			print(f'Training Progress: {p}')
			
			lr_1 = self.lr_function_segmentation.calculate(p)
			print(f'Learning Rate Segmentation: {lr_1}')
			self.optimizer_segmentation.lr = lr_1
			self.lr_segmentation_history.append(lr_1)

			lr_2 = self.lr_function_discriminator.calculate(p)
			print(f'Learning Rate Discriminator: {lr_2}')
			self.optimizer_discriminator.lr = lr_2
			self.lr_discriminator_history.append(lr_2)

			# set lambda value
			l = self.lambda_function.calculate(p)
			print(f'Lambda: {l}')
			self.lambdas.append(l)
			l_vector = np.full((self.batch_size, 1), l, dtype = 'float32')

			print('Start training...')
			for batch in range(self.num_batches_train):

				print(f'Batch {batch + 1} of {self.num_batches_train}')
				batch_train_files = self.train_data_dirs[batch * self.batch_size : (batch + 1) * self.batch_size]

				# load images for training
				batch_images = np.asarray([load_array(batch_train_file, verbose = False) for batch_train_file in batch_train_files])
				batch_images = batch_images.astype(np.float32) # set np.float32 to reduce memory usage

				x_train = batch_images[ :, :, :, : self.channels]
				y_segmentation_train = batch_images[ :, :, :, self.channels :]
				y_discriminator_train = self._convert_path_to_domain(batch_train_files, self.train_data_dirs_source)
				print(f'Domain: {y_discriminator_train}')

				loss_mask = self._generate_loss_mask(y_discriminator_train)
				acc_mask = self._generate_acc_mask(y_discriminator_train)

				step_output = self._training_step_domain_adaptation([x_train, l_vector], [y_segmentation_train, y_discriminator_train], loss_mask, acc_mask)
				loss_segmentation, loss_discriminator = step_output

				loss_segmentation_train += float(loss_segmentation)
				loss_discriminator_train += float(loss_discriminator)

			loss_segmentation_train /= self.num_batches_train
			self.loss_segmentation_train_history.append(loss_segmentation_train)

			loss_discriminator_train /= self.num_batches_train
			self.loss_discriminator_train_history.append(loss_discriminator_train)

			acc_segmentation_train = float(self.acc_function_segmentation.result())
			self.acc_segmentation_train_history.append(acc_segmentation_train)
			
			acc_discriminator_train = float(self.acc_function_discriminator.result())
			self.acc_discriminator_train_history.append(acc_discriminator_train)

			with self.train_writer.as_default():
				tf.summary.scalar('loss_segmentation', loss_segmentation_train, step = epoch + 1)
				tf.summary.scalar('loss_discriminator', loss_discriminator_train, step = epoch + 1)

			print(f'Segmentation Loss: {loss_segmentation_train}')
			print(f'Discriminator Loss: {loss_discriminator_train}')

			print(f'Segmentation Accuracy: {acc_segmentation_train}')
			print(f'Discriminator Accuracy: {acc_discriminator_train}')

			self.acc_function_segmentation.reset_states()
			self.acc_function_discriminator.reset_states()	

			# evaluating network
			print('Start validation...')
			for batch in range(self.num_batches_val):
				print(f'Batch {batch + 1} of {self.num_batches_val}')
				batch_val_files = self.val_data_dirs[batch * self.batch_size : (batch + 1) * self.batch_size]

				# load images for testing
				batch_val_images = np.asarray([load_array(batch_val_file, verbose = False) for batch_val_file in batch_val_files])
				batch_val_images = batch_val_images.astype(np.float32) # set np.float32 to reduce memory usage

				x_val = batch_val_images[:, :, :, : self.channels]
				y_segmentation_val = batch_val_images[:, :, :, self.channels :]
				y_discriminator_val = self._convert_path_to_domain(batch_val_files, self.val_data_dirs_source)
				print(f'Domain: {y_discriminator_val}')

				y_segmentation_pred, y_discriminator_pred = self.model([x_val, l_vector])

				loss_mask = self._generate_loss_mask(y_discriminator_val)
				acc_mask = self._generate_acc_mask(y_discriminator_val)

				loss_segmentation = self.loss_function_segmentation(y_segmentation_val, y_segmentation_pred, loss_mask)
				loss_discriminator = self.loss_function_discriminator(y_discriminator_val, y_discriminator_pred)			

				y_discriminator_val = tf.expand_dims(y_discriminator_val, axis = -1)
				self.acc_function_segmentation.update_state(y_segmentation_val, y_segmentation_pred, sample_weight = acc_mask)
				self.acc_function_discriminator.update_state(y_discriminator_val, y_discriminator_pred)

				loss_segmentation_val += float(loss_segmentation)
				loss_discriminator_val += float(loss_discriminator)

			loss_segmentation_val /= self.num_batches_val
			self.loss_segmentation_val_history.append(loss_segmentation_val)

			loss_discriminator_val /= self.num_batches_val
			self.loss_discriminator_val_history.append(loss_discriminator_val)

			acc_segmentation_val = float(self.acc_function_segmentation.result())
			self.acc_segmentation_val_history.append(acc_segmentation_val)

			acc_discriminator_val = float(self.acc_function_discriminator.result())
			self.acc_discriminator_val_history.append(acc_discriminator_val)

			print(f'Segmentation Loss: {loss_segmentation_val}')
			print(f'Discriminator Loss: {loss_discriminator_val}')

			print(f'Segmentation Accuracy: {acc_segmentation_val}')
			print(f'Discriminator Accuracy: {acc_discriminator_val}')

			if loss_segmentation_val < self.best_val_loss and persist_best_model and p >= 0.5:
				print('[!] Persisting best model...')
				self.best_val_loss = loss_segmentation_val
				self.no_improvement_count = 0
				
				self.best_model = self.assembly_empty_model()
				self.best_model.set_weights(self.model.get_weights())

			else:
				self.no_improvement_count += 1
				if  self.no_improvement_count > self.wait:
					print('[!] Performing early stopping.')
					break

		self.elapsed_time = (time.time() - time_init) / 60

	def train(self, epochs: int = 25, wait: int = 12, persist_best_model: bool = True):

		time_init = time.time()

		self.epochs = epochs
		self.wait = wait

		for epoch in range(self.epochs):
			print(f'Epoch {epoch + 1} of {self.epochs}')
			loss_global_train = 0.
			loss_global_val = 0.

			self.acc_function_segmentation.reset_states()

			np.random.shuffle(self.train_data_dirs)
			np.random.shuffle(self.val_data_dirs)

			# update learning rate
			p = epoch / (epochs - 1)
			lr = self.lr_function_segmentation.calculate(p)
			self.optimizer_segmentation.lr = lr
			self.lr_segmentation_history.append(lr)
			print(f'Learning Rate: {lr}')

			print('Start training...')
			for batch in range(self.num_batches_train):

				print(f'Batch {batch + 1} of {self.num_batches_train}')
				batch_files = self.train_data_dirs[batch * self.batch_size : (batch + 1) * self.batch_size]

				# load images for training
				batch_images = np.asarray([load_array(batch_file, verbose = False) for batch_file in batch_files])
				batch_images = batch_images.astype(np.float32) # set np.float32 to reduce memory usage

				x_train = batch_images[ :, :, :, : self.channels]
				y_train = batch_images[ :, :, :, self.channels :]

				loss_train = self._training_step(x_train, y_train)
				loss_global_train += float(loss_train)

			loss_global_train /= self.num_batches_train
			self.loss_segmentation_train_history.append(loss_global_train)

			acc_global_train = float(self.acc_function_segmentation.result())
			self.acc_segmentation_train_history.append(acc_global_train)

			print(f'Training Loss: {loss_global_train}')
			print(f'Training Accuracy: {acc_global_train}')

			self.acc_function_segmentation.reset_states()

			# evaluating network
			print('Start validation...')
			for batch in range(self.num_batches_val):
				print(f'Batch {batch + 1} of {self.num_batches_val}')
				batch_val_files = self.val_data_dirs[batch * self.batch_size : (batch + 1) * self.batch_size]

				# load images for testing
				batch_val_images = np.asarray([load_array(batch_val_file, verbose = False) for batch_val_file in batch_val_files])
				batch_val_images = batch_val_images.astype(np.float32) # set np.float32 to reduce memory usage

				x_val = batch_val_images[:, :, :, : self.channels]
				y_val = batch_val_images[:, :, :, self.channels :]

				pred_val = self.model(x_val)
				loss_val = self.loss_function(y_val, pred_val)

				loss_global_val += float(loss_val) # convert loss_val to float and sum
				self.acc_function_segmentation.update_state(y_val, pred_val)

			loss_global_val /= self.num_batches_val
			self.loss_segmentation_val_history.append(loss_global_val)

			acc_global_val = float(self.acc_function_segmentation.result())
			self.acc_segmentation_val_history.append(acc_global_val)

			print(f'Validation Loss: {loss_global_val}')
			print(f'Validation Accuracy: {acc_global_val}')

			if loss_global_val < self.best_val_loss and persist_best_model and p >= 0.5:
				print('[!] Persisting best model...')
				self.best_val_loss = loss_global_val
				self.no_improvement_count = 0

				self.best_model = self.assembly_empty_model()
				self.best_model.set_weights(self.model.get_weights())

			else:
				self.no_improvement_count += 1
				if  self.no_improvement_count > self.wait:
					print('[!] Performing early stopping.')
					break

		self.elapsed_time = (time.time() - time_init) / 60

	def save_weights(self, weights_path: str, best: bool = True, piece: str = None):
		if best:
			model_to_save = self.best_model
		else:
			model_to_save = self.model

		if piece is None:
			model_to_save.save_weights(weights_path) # save weights

		elif piece == 'segmentation':
			model_to_save.main_network.save_weights(weights_path)
		
		elif piece == 'discriminator':
			model_to_save.domain_discriminator.save_weights(weights_path)

		print('Weights saved successfuly.')

	def load_weights(self, weights_path: str, piece: str = None):
		if piece is None:
			self.model.load_weights(weights_path)
		
		elif piece == 'segmentation':
			self.model.main_network.load_weights(weights_path)

		elif piece == 'discriminator':
			self.model.domain_discriminator.load_weights(weights_path)

		print('Weights loaded successfuly.')

	def save_model(self, model_path: str, best: bool = True):
		if best:
			save_model(self.best_model, model_path) # save model
		else:
			save_model(self.model, model_path) # save model
		print('Model saved successfuly.')
		
	def save_info(self, history_path):
		persist = self.parameters
		save_json(persist, history_path) # save metrics and parameters
		print('Metrics saved successfuly.')

	def set_test_index(self, test_index_source: list, test_index_target: list):
		test_index_source = [int(i) for i in test_index_source]
		test_index_target = [int(i) for i in test_index_target]

		self.test_index_source = test_index_source
		self.test_index_target = test_index_target

	def set_learning_rate(self, **kwargs):
		segmentation_params = kwargs.get('segmentation')
		if segmentation_params is not None:
			name = segmentation_params.pop('name')
			self.lr_function_segmentation = self.lr_factory.get_function(name, **segmentation_params)

		discriminator_params = kwargs.get('discriminator')
		if discriminator_params is not None:
			name = discriminator_params.pop('name')
			self.lr_function_discriminator = self.lr_factory.get_function(name, **discriminator_params)

	def set_lambda(self, **kwargs):
		self.lambda_function = LambdaGradientReversalLayer(**kwargs)

	@property
	def parameters(self):
		persist = {
				'history':{
					'training':{
						'loss':{
							'segmentation': self.loss_segmentation_train_history,
							'discriminator': self.loss_discriminator_train_history},
						'accuracy':{
							'segmentation': self.acc_segmentation_train_history,
							'discriminator': self.acc_discriminator_train_history}},
					'validation':{
						'loss':{
							'segmentation': self.loss_segmentation_val_history,
							'discriminator': self.loss_discriminator_val_history},
						'accuracy':{
							'segmentation': self.acc_segmentation_val_history,
							'discriminator': self.acc_discriminator_val_history}}},
				'image_files':{
					'training':{
						'total': self.train_data_dirs,
						'source': self.train_data_dirs_source,
				   		'target': self.train_data_dirs_target},
					'validation':{
						'total': self.val_data_dirs,
						'source': self.val_data_dirs_source,
						'target': self.val_data_dirs_target}},
				'is_domain_adaptation': self.domain_adaptation,
				'lr_segmentation': self.lr_segmentation_history,
				'lr_discriminator': self.lr_discriminator_history,
				'time': self.elapsed_time,
				'lambdas': self.lambdas,
				'patch_size': self.patch_size,
				'output_stride': self.output_stride,
				'val_fraction': self.val_fraction,
				'batch_size': self.batch_size,
				'num_images': self.num_images,
				'epochs': self.epochs,
				'wait': self.wait,
				'rotate': self.rotate,
				'flip': self.flip,
				'test_index_source': self.test_index_source,
				'test_index_target': self.test_index_target}

		return persist

	@property
	def time(self):
		return self.elapsed_time