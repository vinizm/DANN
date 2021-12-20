import sys
import os
import numpy as np

from utils.utils import load_images, extract_patches_from_images, save_arrays, convert_to_onehot_tensor
from variables import PATH_TO_FE19_DATASET_RLM, PATH_TO_FE19_DATASET_MASK

if __name__ == '__main__':
    # ONE_CHANNEL = eval(sys.argv[1]) # one channel boolean
    ONE_CHANNEL = True
    print(f'one_channel: {ONE_CHANNEL}')

    PATCH_SIZE = 512
    STRIDE_TRAIN = 256
    STRIDE_TEST = 512
    NUM_CLASS = 2
    TEST_INDEX = [1, 5, 11, 17]

    images_rlm = load_images(path_to_folder = PATH_TO_FE19_DATASET_RLM, normalize = True, one_channel = ONE_CHANNEL)
    images_ref = load_images(path_to_folder = PATH_TO_FE19_DATASET_MASK, normalize = True, one_channel = ONE_CHANNEL)

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

    save_arrays(patches_train, f'./processed_images/Fe19_stride{STRIDE_TRAIN}_onechannel{ONE_CHANNEL}_Train/',
                suffix = '', ext = '.npy', clean_all = True)
    save_arrays(patches_test, f'./processed_images/Fe19_stride{STRIDE_TEST}_onechannel{ONE_CHANNEL}_Test/',
                suffix = '', ext = '.npy', clean_all = True)