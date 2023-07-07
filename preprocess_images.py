import numpy as np
import cv2
import os
import glob
import math

from utils.utils import load_images, extract_patches_from_images, save_arrays, convert_to_onehot_tensor
from config import PATH_TO_FOLDER, PROCESSED_FOLDER


_PATCH_SIZE = 256
_STRIDE_TRAIN = _PATCH_SIZE // 2
_STRIDE_TEST = _PATCH_SIZE
_NUM_CLASS = 2
_BATCH_SIZE = 16

def switch_binary(array: np.ndarray):
    switched = np.where(array == 1, 0, 1)
    return switched


def remove_augmented_images(directory: str):
    if os.path.isdir(directory):
        file_names = os.listdir(directory)
        file_names = [f'{directory}/{file_name}' for file_name in file_names if 'rotation' in file_name or 'flip' in file_name]
        
        for augmented_file in file_names:
            print(f'removing {augmented_file}')
            os.remove(augmented_file)


def create_patches(dataset: str, test_index: list = None, resample: bool = False, gray_scale: bool = True, batch_size: int = _BATCH_SIZE,
                   stride_train: int = _STRIDE_TRAIN, stride_test: int = _STRIDE_TEST, patch_size: int = _PATCH_SIZE):

    conversor = cv2.COLOR_BGR2GRAY if gray_scale else cv2.COLOR_BGR2RGB

    path_to_dataset_rlm = PATH_TO_FOLDER.get(dataset).get('RLM')
    path_to_dataset_mask = PATH_TO_FOLDER.get(dataset).get('MASK')

    rlm_files = glob.glob(path_to_dataset_rlm + '/*.tif')
    rlm_files.sort()
    
    mask_files = glob.glob(path_to_dataset_mask + '/*.tif')
    mask_files.sort()
    num_files = len(rlm_files)

    if test_index is None:
        resample = True
        test_index = list(range(num_files))

    if resample:
        train_index = list(range(num_files))
    else:
        train_index = [i for i in range(num_files) if i not in test_index]
    
    global_num_train, global_num_test = 0, 0
    num_batches = math.ceil(num_files / batch_size)
    for batch in range(num_batches):
        print(f'batch {batch + 1} of {num_batches}')
        
        clean_all = True if batch == 0 else False
        
        images_rlm_train, images_ref_train = [], []
        images_rlm_test, images_ref_test = [], []
        
        idx_lower, idx_upper = batch * batch_size, (batch + 1) * batch_size
        for rlm_file, mask_file in zip(rlm_files[idx_lower : idx_upper], mask_files[idx_lower : idx_upper]):
            
            image_rlm = load_images(files = [rlm_file], normalize = False, gray_scale = False)
            image_ref = load_images(files = [mask_file], normalize = True, gray_scale = True)
            image_ref = [switch_binary(image_ref[0])]
            
            idx_global = rlm_files.index(rlm_file)
            if idx_global in train_index:
                images_rlm_train.extend(image_rlm)
                images_ref_train.extend(image_ref)
                
            elif idx_global in test_index:
                images_rlm_test.extend(image_rlm)
                images_ref_test.extend(image_ref)
                
        if len(images_rlm_train) > 0:

            patches_rlm_train = extract_patches_from_images(images = images_rlm_train, patch_size = patch_size, stride = stride_train)
            patches_ref_train = extract_patches_from_images(images = images_ref_train, patch_size = patch_size, stride = stride_train)
            
            patches_rlm_train = [cv2.cvtColor(img, conversor) for img in patches_rlm_train]
            patches_rlm_train = np.asarray([img.reshape((patch_size, patch_size, 1)) for img in patches_rlm_train]) if gray_scale else np.asarray(patches_rlm_train)
            
            patches_rlm_train = patches_rlm_train / 255.
            
            patches_ref_train_onehot = convert_to_onehot_tensor(tensor = patches_ref_train, num_class = _NUM_CLASS)
            patches_train = np.concatenate((patches_rlm_train, patches_ref_train_onehot), axis = 3)
            
            print(f'train patches: {patches_train.shape}')
            save_arrays(patches_train, f'{PROCESSED_FOLDER}/{dataset}_patch{patch_size}_stride{stride_train}_Train/',
                        suffix = '', ext = '.npy', clean_all = clean_all, start_num = global_num_train)
            global_num_train += len(patches_train)

        if len(images_rlm_test) > 0:

            patches_rlm_test = extract_patches_from_images(images = images_rlm_test, patch_size = patch_size, stride = stride_test)
            patches_ref_test = extract_patches_from_images(images = images_ref_test, patch_size = patch_size, stride = stride_test)

            save_arrays(patches_rlm_test, f'{PROCESSED_FOLDER}/{dataset}_patch{patch_size}_stride{stride_test}_RGB_Test/',
                        suffix = '', ext = '.tif', clean_all = clean_all, start_num = global_num_test)
            
            patches_rlm_test = [cv2.cvtColor(img, conversor) for img in patches_rlm_test]
            patches_rlm_test = np.asarray([img.reshape((patch_size, patch_size, 1)) for img in patches_rlm_test]) if gray_scale else np.asarray(patches_rlm_test)
            
            patches_rlm_test = patches_rlm_test / 255.

            patches_ref_test_onehot = convert_to_onehot_tensor(tensor = patches_ref_test, num_class = _NUM_CLASS)
            patches_test = np.concatenate((patches_rlm_test, patches_ref_test_onehot), axis = 3)

            print(f'test patches: {patches_test.shape}')
            save_arrays(patches_test, f'{PROCESSED_FOLDER}/{dataset}_patch{patch_size}_stride{stride_test}_Test/',
                        suffix = '', ext = '.npy', clean_all = clean_all, start_num = global_num_test)
            global_num_test += len(patches_test)
