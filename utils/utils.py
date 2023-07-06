import os
import numpy as np
from sklearn.feature_extraction.image import *
import cv2

from tensorflow import one_hot
import tensorflow as tf
import json
import random as rd

from config import *
from utils.plot import compare_images


def load_images(path_to_folder: str = None, files: list = None, normalize: bool = False, gray_scale: bool = False):
    if files is None:
        files = os.listdir(path_to_folder)
        files.sort()
        
    images = []
    for file_name in files:
        
        if path_to_folder is None:
            full_path = file_name
        else:
            full_path = os.path.join(path_to_folder, file_name)
        
        img = cv2.imread(full_path)

        if gray_scale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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


def save_arrays(images: np.ndarray, path_to_folder: str, suffix = '', ext = '.npy', start_num: int = 0, clean_all: bool = True):
    _saviors = {'.npy': save_np_array, '.tif': save_image}

    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)
        
    if clean_all:
        files_to_remove = os.listdir(path_to_folder)
        [os.remove(os.path.join(path_to_folder, file)) for file in files_to_remove]

    count = 0
    for i in range(images.shape[0]):
        img = images[i, :, :, :]
        file_name = f'{start_num + i + 1:06}{suffix}{ext}'
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


def classify_image(array, threshold):
  classified_array = np.where(array >= threshold, 1., 0.)
  return classified_array


def plot_images(rgb, gray, true, pred, n, threshold = 0.5, patch_idx = []):

  if len(patch_idx) == 0:
    patch_idx = rd.sample(range(0, len(true)), n)

  for i in patch_idx:
    rgb_i = rgb[i]
    gray_i = gray[i, :, :, 0]
    true_i = true[i, :, :, 1]
    proba_i = np.log10(pred[i, :, :, 1])
    pred_i = classify_image(pred[i, :, :, 1], threshold)
    compare_images(rgb_i, gray_i, true_i, pred_i, proba_i, i)