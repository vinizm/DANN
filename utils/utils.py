import os
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, average_precision_score
from sklearn.feature_extraction.image import *
import cv2

from tensorflow import one_hot
import tensorflow as tf
import json

from config import *


def flatten(array: np.ndarray, keep_dims: bool = True):
  if keep_dims:
    last_dim = array.shape[-1]
    new_shape = (np.prod(array.shape[: -1]), last_dim)
  else:
    new_shape = -1
  return array.reshape(new_shape)


def compute_metrics(true_labels: np.ndarray, predicted_labels: np.ndarray):

    true_vector_1 = flatten(np.argmax(true_labels, axis = -1), keep_dims = False)
    true_vector_0 = flatten(np.argmin(true_labels, axis = -1), keep_dims = False)
    pred_vector = flatten(np.argmax(predicted_labels, axis = -1), keep_dims = False)

    accuracy = accuracy_score(true_vector_1, pred_vector)
    precision = precision_score(true_vector_1, pred_vector)
    recall = recall_score(true_vector_1, pred_vector)
    f1 = f1_score(true_vector_1, pred_vector)
    
    proba_pairs = flatten(predicted_labels, keep_dims = True)
    proba_vector_1 = proba_pairs[:, 1]
    proba_vector_0 = proba_pairs[:, 0]
    avg_precision_1 = average_precision_score(true_vector_1, proba_vector_1)
    avg_precision_0 = average_precision_score(true_vector_0, proba_vector_0)
    mean_avg_precision = (avg_precision_1 + avg_precision_0) / 2
    
    metrics = {
                'avg_precision_1': avg_precision_1,
                'avg_precision_0': avg_precision_0,
                'mean_avg_precision': mean_avg_precision,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
            }
    return metrics


def load_images(path_to_folder: str, normalize: bool = False, one_channel: bool = False, conversor = cv2.COLOR_BGR2GRAY):
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


def load_arrays(path_to_folder: str, verbose: bool = True):
    if not os.path.exists(path_to_folder):
        print('Folder does not exist.')
    
    file_names = os.listdir(path_to_folder)
    file_names.sort()
    content = []
    for file_name in file_names:
        full_file_name = os.path.join(path_to_folder, file_name)
        content.append(load_array(full_file_name, verbose = verbose))

    return np.asarray(content)


def load_array(full_file_name: str, verbose: bool = True):
    file_name = os.path.basename(full_file_name)
    with open(full_file_name, 'rb') as file:
        try:
            array = np.load(file)
            if verbose:
                print(f'Loaded file {file_name} successfuly')
        except:
            if verbose:
                print(f'Could not load file {file_name} successfuly')

    return array


def save_np_array(file_name: str, array: np.ndarray):
    with open(file_name, 'wb') as file:
        try:
            np.save(file, array)
            print(f'Saved file {file_name} successfuly')
            status = True
        except:
            print(f'Could not save file {file_name}')
            status = False
    return status


def save_image(file_name: str, array: np.ndarray):
    try:
        cv2.imwrite(file_name, array)
        print(f'Saved file {file_name} successfuly')
        status = True
    except:
        print(f'Could not save file {file_name}')
        status = False
    return status


def save_arrays(images: np.ndarray, path_to_folder: str, suffix = '', ext = '.npy', clean_all: bool = True):
    _saviors = {'.npy': save_np_array, '.tif': save_image}

    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)
        
    if clean_all:
        files_to_remove = os.listdir(path_to_folder)
        [os.remove(os.path.join(path_to_folder, file)) for file in files_to_remove]

    count = 0
    for i in range(images.shape[0]):
        img = images[i, :, :, :]
        file_name = f'{i + 1:06}{suffix}{ext}'
        full_file_name = os.path.join(path_to_folder, file_name)
        
        savior = _saviors.get(ext)
        status = savior(full_file_name, img)
        
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


def augment_images(image_files: list, angles: list, rotate: bool, flip: bool, verbose: bool = True):
    augmented_files = []

    for image_file in image_files:
        file_id = os.path.basename(image_file)

        array = load_array(image_file, verbose = verbose)
        file_path, ext = os.path.splitext(image_file)

        if rotate:
            for angle in angles:
                if verbose:
                    print(f'Rotating {file_id} by {angle}.') 
                array_rot = np.asarray(tf.image.rot90(array, k = int(angle / 90.)))
                
                file_name = f'{file_path}_rotation{angle}{ext}'
                save_np_array(file_name = file_name, array = array_rot)
                augmented_files.append(file_name)
        
        if flip:
            if verbose:
                print(f'Flipping {file_id} vertically.')
            array_flip = np.asarray(tf.image.flip_left_right(array))
            file_name = f'{file_path}_flip_y{ext}'
            save_np_array(file_name = file_name, array = array_flip)
            augmented_files.append(file_name)

            if verbose:
                print(f'Flipping {file_id} horizontally.')
            array_flip = np.asarray(tf.image.flip_up_down(array))
            file_name = f'{file_path}_flip_x{ext}'
            save_np_array(file_name = file_name, array = array_flip)
            augmented_files.append(file_name)

    return augmented_files

def generate_weight_maps(y_true, epsilon: float):
    y_true = y_true.numpy()
    y_true = y_true.astype('uint8')

    wmaps = []
    for i in range(len(y_true)):
        mask = y_true[i, :, :, 1]
        map_d1 = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

        nonzero = np.argwhere(map_d1 != 0.)
        d1_std = np.std(map_d1[nonzero[:, 0], nonzero[:, 1]])
        wmap = epsilon + (1 - epsilon) * np.exp(- 0.5 * (map_d1 / d1_std) ** 2)
        wmaps.append(wmap)

    wmaps = tf.convert_to_tensor(np.asarray(wmaps), dtype = tf.float32)
    return wmaps
