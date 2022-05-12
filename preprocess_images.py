import sys
import os
import numpy as np
import cv2

from utils.utils import load_images, extract_patches_from_images, save_arrays, convert_to_onehot_tensor
from config import PATH_TO_FOLDER


_PATCH_SIZE = 512
_STRIDE_TRAIN = 256
_STRIDE_TEST = 512
_NUM_CLASS = 2


def preprocess_images(dataset: str, test_index: list = None, resample: bool = False, one_channel: bool = True,
                      stride_train: int = _STRIDE_TRAIN, stride_test: int = _STRIDE_TEST, patch_size: int = _PATCH_SIZE):

    path_to_dataset_rlm = PATH_TO_FOLDER.get(dataset).get('RLM')
    path_to_dataset_mask = PATH_TO_FOLDER.get(dataset).get('MASK')

    images_rlm = load_images(path_to_folder = path_to_dataset_rlm, normalize = False, one_channel = False)
    images_ref = load_images(path_to_folder = path_to_dataset_mask, normalize = True, one_channel = one_channel)

    if test_index is None:
        resample = True
        test_index = list(range(len(images_rlm)))

    if resample:
        train_index = list(range(len(images_ref)))
    else:
        train_index = [i for i in range(len(images_ref)) if i not in test_index]

    images_rlm_train = [images_rlm[i] for i in train_index]
    images_ref_train = [images_ref[i] for i in train_index]

    images_rlm_test = [images_rlm[i] for i in test_index]
    images_ref_test = [images_ref[i] for i in test_index]

    patches_rlm_train = extract_patches_from_images(images = images_rlm_train, patch_size = patch_size, stride = stride_train)
    patches_ref_train = extract_patches_from_images(images = images_ref_train, patch_size = patch_size, stride = stride_train)

    patches_rlm_test = extract_patches_from_images(images = images_rlm_test, patch_size = patch_size, stride = stride_test)
    patches_ref_test = extract_patches_from_images(images = images_ref_test, patch_size = patch_size, stride = stride_test)

    save_arrays(patches_rlm_test, f'./processed_images/{dataset}_patch{patch_size}_stride{stride_test}_RGB_Test/',
                suffix = '', ext = '.tif', clean_all = True)
    
    # convert 3 channels to 1 channel if needed
    if one_channel:
        patches_rlm_train = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in patches_rlm_train]
        patches_rlm_test = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in patches_rlm_test]

        patches_rlm_train = np.asarray([img.reshape((patch_size, patch_size, 1)) for img in patches_rlm_train])
        patches_rlm_test = np.asarray([img.reshape((patch_size, patch_size, 1)) for img in patches_rlm_test])

    # normalize images
    patches_rlm_train = patches_rlm_train / 255.
    patches_rlm_test = patches_rlm_test / 255.

    patches_ref_train_onehot = convert_to_onehot_tensor(tensor = patches_ref_train, num_class = _NUM_CLASS)
    patches_train = np.concatenate((patches_rlm_train, patches_ref_train_onehot), axis = 3)

    patches_ref_test_onehot = convert_to_onehot_tensor(tensor = patches_ref_test, num_class = _NUM_CLASS)
    patches_test = np.concatenate((patches_rlm_test, patches_ref_test_onehot), axis = 3)

    print(f'train patches: {patches_train.shape}')
    print(f'test patches: {patches_test.shape}')

    save_arrays(patches_train, f'./processed_images/{dataset}_patch{patch_size}_stride{stride_train}_Train/',
                suffix = '', ext = '.npy', clean_all = True)
    save_arrays(patches_test, f'./processed_images/{dataset}_patch{patch_size}_stride{stride_test}_Test/',
                suffix = '', ext = '.npy', clean_all = True)


if __name__ == '__main__':
    # _DATASET = sys.argv[1]
    # _TEST_INDEX = eval(sys.argv[2])
    # _STRIDE_TRAIN = sys.argv[3]
    # _STRIDE_TEST = sys.argv[4]
    # _PATCH_SIZE = sys.argv[5]

    _DATASET = 'FeM'
    _TEST_INDEX = [1, 5, 11, 17]
    _STRIDE_TRAIN = 256
    _STRIDE_TEST = 512
    _PATCH_SIZE = 512
    print(f'dataset: {_DATASET}')
    print(f'test_index: {_TEST_INDEX}')
    print(f'stride_train: {_STRIDE_TRAIN}')
    print(f'stride_test: {_STRIDE_TEST}')
    print(f'patch_size: {_PATCH_SIZE}')

    preprocess_images(dataset = _DATASET, test_index = _TEST_INDEX, resample = False, one_channel = True,
                      stride_train = _STRIDE_TRAIN, stride_test = _STRIDE_TEST, patch_size = _PATCH_SIZE)