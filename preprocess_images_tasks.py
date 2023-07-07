from preprocess_images import create_patches
from config import TEST_INDEX


RESAMPLE = False
GRAY_SCALE = False
NUM_CLASS = 2
PATCH_SIZE = 256
STRIDE_TEST = PATCH_SIZE
STRIDE_TRAIN = PATCH_SIZE // 2
BATCH_SIZE = 32


def create_patches_Fe19(IMAGES_FOR_TEST: list = TEST_INDEX.get('Fe19')):
    create_patches(dataset = 'Fe19', test_index = IMAGES_FOR_TEST, resample = RESAMPLE, gray_scale = GRAY_SCALE,
                   stride_train = STRIDE_TRAIN, stride_test = STRIDE_TEST, patch_size = PATCH_SIZE, batch_size = BATCH_SIZE)

def create_patches_Fe120(IMAGES_FOR_TEST: list = TEST_INDEX.get('Fe120')):
    create_patches(dataset = 'Fe120', test_index = IMAGES_FOR_TEST, resample = RESAMPLE, gray_scale = GRAY_SCALE,
                   stride_train = STRIDE_TRAIN, stride_test = STRIDE_TEST, patch_size = PATCH_SIZE, batch_size = BATCH_SIZE)
    
def create_patches_FeM(IMAGES_FOR_TEST: list = TEST_INDEX.get('FeM')):
    create_patches(dataset = 'FeM', test_index = IMAGES_FOR_TEST, resample = RESAMPLE, gray_scale = GRAY_SCALE,
                   stride_train = STRIDE_TRAIN, stride_test = STRIDE_TEST, patch_size = PATCH_SIZE, batch_size = BATCH_SIZE)
    
def create_patches_Cu(IMAGES_FOR_TEST: list = TEST_INDEX.get('Cu')):
    create_patches(dataset = 'Cu', test_index = IMAGES_FOR_TEST, resample = RESAMPLE, gray_scale = GRAY_SCALE,
                   stride_train = STRIDE_TRAIN, stride_test = STRIDE_TEST, patch_size = PATCH_SIZE, batch_size = BATCH_SIZE)

def create_all_patches():
    create_patches_Fe19()
    create_patches_Fe120()
    create_patches_FeM()
    create_patches_Cu() 


if __name__ == '__main__':
    create_all_patches()
