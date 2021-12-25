import os
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, average_precision_score
from sklearn.feature_extraction.image import *
import cv2
import imutils

from tensorflow import one_hot
import tensorflow as tf
import json


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if not title:
		if normalize:
			title = 'Normalized confusion matrix'
		else:
			title = 'Confusion matrix, without normalization'

	# Compute confusion matrix
	cm = confusion_matrix(y_true, y_pred)
	# Only use the labels that appear in the data
	classes = [0,1]##classes[unique_labels(y_true, y_pred)]
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
	ax.figure.colorbar(im, ax=ax)
	# We want to show all ticks...
	ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes, title=title, ylabel='True label', xlabel='Predicted label')

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout()
	return ax


# ============================== NEW FUNCTIONS ==============================


def compute_metrics(true_labels: np.ndarray, predicted_labels: np.ndarray):
	matrix = confusion_matrix(true_labels, predicted_labels)
	accuracy = accuracy_score(true_labels, predicted_labels)
	avg_precision = average_precision_score(true_labels, predicted_labels)
	f1 = f1_score(true_labels, predicted_labels)
	recall = recall_score(true_labels, predicted_labels)
	precision = precision_score(true_labels, predicted_labels)
	
	metrics = {
				'matrix': matrix.tolist(),
				'accuracy': accuracy,
				'average_precision': avg_precision,
				'f1_score': f1,
				'recall': recall,
				'precision': precision
	}
	return metrics


def load_images(path_to_folder: str, normalize: bool = False, one_channel: bool = False, conversor: int = cv2.COLOR_RGB2GRAY):
	files = os.listdir(path_to_folder)
	files.sort()
	images = []

	for file_name in files:
		img = cv2.imread(os.path.join(path_to_folder, file_name))

		if one_channel:
			img = cv2.cvtColor(img, conversor)
			shape = img.shape
			img = img.reshape([*shape, 1])

		if normalize:
			img = img / 255.

		images.append(img)

	return images


def extract_patches(image: np.ndarray, patch_size: int, stride: int):
	image_patches = []

	h = image.shape[0] // stride
	w = image.shape[1] // stride

	for m in range(h):
		for n in range(w):
			if ((m * stride + patch_size <= image.shape[0]) and (n * stride + patch_size <= image.shape[1])):
				image_patches.append(image[m * stride : m * stride + patch_size, n * stride : n * stride + patch_size, : ])

	return image_patches


def extract_patches_from_images(images: list, patch_size: int, stride: int):
	image_patches = []
	
	for i in range(len(images)):
		img_temp = extract_patches(images[i], patch_size, stride)
		image_patches += img_temp

	return np.array(image_patches)



def load_arrays(path_to_folder: str):
	if not os.path.exists(path_to_folder):
		print('Folder does not exist.')
	
	file_names = os.listdir(path_to_folder)
	file_names.sort()
	content = []
	for file_name in file_names:
		full_file_name = os.path.join(path_to_folder, file_name)
		content.append(load_array(full_file_name))

	return np.asarray(content)


def load_array(full_file_name: str):
	file_name = os.path.basename(full_file_name)
	with open(full_file_name, 'rb') as file:
		try:
			array = np.load(file)
			print(f'Loaded file {file_name} successfuly.')
		except:
			print(f'Could not load file {file_name} successfuly.')

	return array


def save_array(file_name: str, array: np.ndarray):
	with open(file_name, 'wb') as file:
		try:
			np.save(file, array)
			print(f'Saved file {file_name} successfuly.')
			status = True
		except:
			print(f'Could not save file {file_name}.')
			status = False
	return status


def save_arrays(images: np.ndarray, path_to_folder: str, suffix = '', ext = '.npy', clean_all: bool = True):
	if not os.path.exists(path_to_folder):
		os.makedirs(path_to_folder)
		
	if clean_all:
		files_to_remove = os.listdir(path_to_folder)
		[os.remove(os.path.join(path_to_folder, file)) for file in files_to_remove]

	count = 0
	for i in range(images.shape[0]):
		img = images[i, :, :, :]
		file_name = f'{i + 1:03}{suffix}{ext}'
		full_file_name = os.path.join(path_to_folder, file_name)
		
		status = save_array(file_name = full_file_name, array = img)
		if status:
			count += 1
			
	return count


def convert_to_onehot_tensor(tensor: np.ndarray, num_class: int):
	onehot_tensor = one_hot(tensor[:, :, :, 0], depth = num_class, axis = -1)
	return np.asarray(onehot_tensor)


def save_json(data, file_name: str):
	with open(file_name, 'w') as file: # save history
		json.dump(data, file)


def load_json(file_name: str):
	with open(file_name, 'r') as file: # load history
		history = json.load(file)
	return history


def augment_images(image_files: list, angles: list):
	augmented_files = []
	for image_file in image_files:
		file_id = os.path.basename(image_file)

		array = load_array(image_file)
		file_path, ext = os.path.splitext(image_file)

		for angle in angles:
			print(f'Rotating {file_id} by {angle}.') 
			array_rot = np.asarray(tf.image.rot90(array, k = int(angle / 90.)))
			
			file_name = f'{file_path}_rotation{angle}{ext}'
			save_array(file_name = file_name, array = array_rot)
			augmented_files.append(file_name)
										 
	return augmented_files
