from preprocess_images import preprocess_images
from config import TEST_INDEX


RESAMPLE = False
ONE_CHANNEL = True
NUM_CLASS = 2
PATCH_SIZE = 256
STRIDE_TEST = PATCH_SIZE
STRIDE_TRAIN = PATCH_SIZE // 2


DATASET = 'Fe19'
IMAGES_FOR_TEST = TEST_INDEX.get(DATASET)
preprocess_images(dataset = DATASET, test_index = IMAGES_FOR_TEST, resample = RESAMPLE, one_channel = ONE_CHANNEL,
                  stride_train = STRIDE_TRAIN, stride_test = STRIDE_TEST, patch_size = PATCH_SIZE)

DATASET = 'Fe120'
IMAGES_FOR_TEST = TEST_INDEX.get(DATASET)
preprocess_images(dataset = DATASET, test_index = IMAGES_FOR_TEST, resample = RESAMPLE, one_channel = ONE_CHANNEL,
                  stride_train = STRIDE_TRAIN, stride_test = STRIDE_TEST, patch_size = PATCH_SIZE)

DATASET = 'FeM'
IMAGES_FOR_TEST = TEST_INDEX.get(DATASET)
preprocess_images(dataset = DATASET, test_index = IMAGES_FOR_TEST, resample = RESAMPLE, one_channel = ONE_CHANNEL,
                  stride_train = STRIDE_TRAIN, stride_test = STRIDE_TEST, patch_size = PATCH_SIZE)

DATASET = 'Cu'
IMAGES_FOR_TEST = TEST_INDEX.get(DATASET)
preprocess_images(dataset = DATASET, test_index = IMAGES_FOR_TEST, resample = RESAMPLE, one_channel = ONE_CHANNEL,
                  stride_train = STRIDE_TRAIN, stride_test = STRIDE_TEST, patch_size = PATCH_SIZE)