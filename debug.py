import os
from main import run_case
import tensorflow as tf

from variables import *


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# if tf.test.gpu_device_name():
#     print('GPU found')
# else:
#     print("No GPU found")


if __name__ == '__main__':

    train_dir = f'{PROCESSED_FOLDER}/Fe19_stride512_Train'
    test_dir = f'{PROCESSED_FOLDER}/Fe19_stride512_Test'
    patch_size = 512
    channels = 1
    num_class = 2
    output_stride = 8
    epochs = 20
    batch_size = 4
    val_fraction = 0.15
    num_images_train = None
    num_images_test = None
    patience = 5

    folder_to_save = MODELS_FOLDER
    model_name = 'teste.h5'
    path_to_save = os.path.join(folder_to_save, model_name)


    run_case(train_dir = train_dir, test_dir = test_dir, patch_size = patch_size, channels = channels, num_class = num_class,
            output_stride = output_stride, epochs = epochs, batch_size = batch_size, val_fraction = val_fraction, num_images_train = num_images_train,
            num_images_test = num_images_test, patience = patience, path_to_save = path_to_save)