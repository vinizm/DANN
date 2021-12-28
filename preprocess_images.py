import sys
import os
import numpy as np

from utils.utils import load_images, extract_patches_from_images, save_arrays, convert_to_onehot_tensor
from variables import PATH_TO_FOLDER

if __name__ == '__main__':
    ONE_CHANNEL = eval(sys.argv[1])
    DATASET = sys.argv[2]
    TEST_INDEX = eval(sys.argv[3])
    RESAMPLE = eval(sys.argv[4])

    # ONE_CHANNEL = True
    # DATASET = 'Fe120'
    # TEST_INDEX = [1, 5, 11, 17]
    # RESAMPLE = True
    print(f'one_channel: {ONE_CHANNEL}')
    print(f'dataset: {DATASET}')
    print(f'test_index: {TEST_INDEX}')
    print(f'resample: {RESAMPLE}')

    PATCH_SIZE = 512
    STRIDE_TRAIN = 256
    STRIDE_TEST = 512
    NUM_CLASS = 2

    path_to_dataset_rlm = PATH_TO_FOLDER.get(DATASET).get('RLM')
    path_to_dataset_mask = PATH_TO_FOLDER.get(DATASET).get('MASK')

    images_rlm = load_images(path_to_folder = path_to_dataset_rlm, normalize = True, one_channel = ONE_CHANNEL)
    images_ref = load_images(path_to_folder = path_to_dataset_mask, normalize = True, one_channel = ONE_CHANNEL)

    if RESAMPLE:
        train_index = [i for i in range(len(images_ref))]
    else:
        train_index = [i for i in range(len(images_ref)) if i not in TEST_INDEX]

    images_rlm_train = [images_rlm[i] for i in train_index]
    images_ref_train = [images_ref[i] for i in train_index]

    images_rlm_test = [images_rlm[i] for i in TEST_INDEX]
    images_ref_test = [images_ref[i] for i in TEST_INDEX]

    patches_rlm_train = extract_patches_from_images(images = images_rlm_train, patch_size = PATCH_SIZE, stride = STRIDE_TRAIN)
    patches_ref_train = extract_patches_from_images(images = images_ref_train, patch_size = PATCH_SIZE, stride = STRIDE_TRAIN)

    patches_rlm_test = extract_patches_from_images(images = images_rlm_test, patch_size = PATCH_SIZE, stride = STRIDE_TEST)
    patches_ref_test = extract_patches_from_images(images = images_ref_test, patch_size = PATCH_SIZE, stride = STRIDE_TEST)

    patches_ref_train_onehot = convert_to_onehot_tensor(tensor = patches_ref_train, num_class = NUM_CLASS)
    patches_train = np.concatenate((patches_rlm_train, patches_ref_train_onehot), axis = 3)

    patches_ref_test_onehot = convert_to_onehot_tensor(tensor = patches_ref_test, num_class = NUM_CLASS)
    patches_test = np.concatenate((patches_rlm_test, patches_ref_test_onehot), axis = 3)

    print(f'train patches: {patches_train.shape}')
    print(f'train patches: {patches_test.shape}')

    save_arrays(patches_train, f'./processed_images/{DATASET}_stride{STRIDE_TRAIN}_onechannel{ONE_CHANNEL}_Train/',
                suffix = '', ext = '.npy', clean_all = True)
    save_arrays(patches_test, f'./processed_images/{DATASET}_stride{STRIDE_TEST}_onechannel{ONE_CHANNEL}_Test/',
                suffix = '', ext = '.npy', clean_all = True)