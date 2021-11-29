import cv2
import os

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical

from utils.utils import load_images, extract_patches_from_images, save_arrays, convert_to_onehot_tensor
from variables import PATH_TO_FE19_DATASET_RLM, PATH_TO_FE19_DATASET_MASK


PATCH_SIZE = 512
STRIDE_TRAIN = 256
STRIDE_TEST = 512
NUM_CLASS = 2


images_rlm = load_images(path_to_folder = PATH_TO_FE19_DATASET_RLM, normalize = True, one_channel = True)
images_ref = load_images(path_to_folder = PATH_TO_FE19_DATASET_MASK, normalize = True, one_channel = True)

patches_rlm_train = extract_patches_from_images(images = images_rlm, patch_size = PATCH_SIZE, stride = STRIDE_TRAIN)
patches_ref_train = extract_patches_from_images(images = images_ref, patch_size = PATCH_SIZE, stride = STRIDE_TRAIN)

patches_rlm_test = extract_patches_from_images(images = images_rlm, patch_size = PATCH_SIZE, stride = STRIDE_TEST)
patches_ref_test = extract_patches_from_images(images = images_ref, patch_size = PATCH_SIZE, stride = STRIDE_TEST)

patches_ref_train_onehot = convert_to_onehot_tensor(tensor = patches_ref_train, num_class = NUM_CLASS)
patches_train = np.concatenate((patches_rlm_train, patches_ref_train_onehot), axis = 3)

patches_ref_test_onehot = convert_to_onehot_tensor(tensor = patches_ref_test, num_class = NUM_CLASS)
patches_test = np.concatenate((patches_rlm_test, patches_ref_test_onehot), axis = 3)

save_arrays(patches_train, f'./Processed/Fe19_stride512_Train/', suffix = '', ext = '.npy')
save_arrays(patches_test, f'./Processed/Fe19_stride512_Test/', suffix = '', ext = '.npy')

