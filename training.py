import os
import numpy as np
import glob
import time
import copy

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import save_model

from architectures import Deeplabv3plus
from utils.utils import load_array, save_json, augment_images
from variables import *
from utils.loss_functions import binary_crossentropy, pixel_wise_d1_weighted_loss


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# if tf.test.gpu_device_name():
#     print('GPU found')
# else:
#     print("No GPU found")


def Train(net, patches_dir: str, val_fraction: float, batch_size: int, num_images: int, channels: int,
		  epochs: int, wait: int, model_path: str, history_path: str, rotate: bool, flip: bool):
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

	if rotate or flip:
		augmented_train = augment_images(image_files = train_data_dirs, angles = [90, 180, 270], rotate = True, flip = False)
		train_data_dirs += augmented_train

		augmented_val = augment_images(image_files = val_data_dirs, angles = [90, 180, 270], rotate = True, flip = False)
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
			save_model(net, model_path) # save model
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


def Train_Case(train_dir: str, lr: float, patch_size: int, channels: int, num_class: int,
			 output_stride: int, epochs: int, batch_size: int, val_fraction: float, num_images_train: int,
			 patience: int, model_path: str, history_path: str, rotate: bool, flip: bool):

	start = time.time()

	net = Deeplabv3plus(weights = None, input_tensor = None, input_shape = (patch_size, patch_size, channels),
						classes = num_class, backbone = 'xception', OS = output_stride,
						alpha = 1., activation = 'sigmoid')
	
	adam = Adam(learning_rate = lr)
	net.compile(loss = pixel_wise_d1_weighted_loss, optimizer = adam, metrics = ['accuracy'], run_eagerly = True)
	net.summary()

	# call train function
	Train(net = net, patches_dir = train_dir, val_fraction = val_fraction, batch_size = batch_size,
		  num_images = num_images_train, epochs = epochs, wait = patience, model_path = model_path,
		  history_path = history_path, channels = channels, rotate = rotate, flip = flip)

	end = time.time()
	hours, rem = divmod(end - start, 3600)
	minutes, seconds = divmod(rem, 60)
	print('Tempo Total: ' + '{:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds))


if __name__ == '__main__':

	one_channel = True

	train_dir = f'{PROCESSED_FOLDER}/Fe19_stride256_onechannel{one_channel}_Train'
	lr = 1.e-4
	patch_size = 512
	channels = 3 if not one_channel else 1
	num_class = 2
	output_stride = 8
	epochs = 25
	batch_size = 2
	val_fraction = 0.2
	num_images_train = None
	patience = 10
	rotate = True
	flip = False

	folder_to_save = MODELS_FOLDER
	file_name = 'teste'
	model_path = os.path.join(folder_to_save, f'{file_name}.h5')
	history_path = os.path.join(folder_to_save, f'{file_name}.json')

	print(f'train_dir: {train_dir}')
	print(f'lr: {lr}')
	print(f'patch_size: {patch_size}')
	print(f'channels: {channels}')
	print(f'num_class: {num_class}')
	print(f'output_stride: {output_stride}')
	print(f'epochs: {epochs}')
	print(f'batch_size: {batch_size}')
	print(f'val_fraction: {val_fraction}')
	print(f'num_images_train: {num_images_train}')
	print(f'rotate: {rotate}')
	print(f'flip: {flip}')
	print(f'patience: {patience}')
	print(f'model_path: {model_path}')
	print(f'history_path: {history_path}')


	Train_Case(train_dir = train_dir, patch_size = patch_size, channels = channels, num_class = num_class,
			   output_stride = output_stride, epochs = epochs, batch_size = batch_size, val_fraction = val_fraction,
			   num_images_train = num_images_train, patience = patience, model_path = model_path,
			   history_path = history_path, lr = lr, rotate = rotate, flip = flip)