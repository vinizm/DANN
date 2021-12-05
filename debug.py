import os
import tensorflow as tf
import logging

from variables import *
from main import run_case

logging.basicConfig(filename = 'dann.log')

# create logger
logger = logging.getLogger('DANN')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# if tf.test.gpu_device_name():
#     print('GPU found')
# else:
#     print("No GPU found")


if __name__ == '__main__':

    train_dir = f'{PROCESSED_FOLDER}/Fe19_stride512_Train'
    test_dir = f'{PROCESSED_FOLDER}/Fe19_stride512_Test'
    lr = 1.e-3
    patch_size = 512
    channels = 1
    num_class = 2
    output_stride = 8
    epochs = 20
    batch_size = 4
    val_fraction = 0.15
    num_images_train = 30
    num_images_test = None
    patience = 5

    folder_to_save = MODELS_FOLDER
    file_name = 'teste'
    model_path = os.path.join(folder_to_save, f'{file_name}.h5')
    history_path = os.path.join(folder_to_save, f'{file_name}.json')

    print(f'train_dir: {train_dir}')
    print(f'test_dir: {test_dir}')
    print(f'lr: {lr}')
    print(f'patch_size: {patch_size}')
    print(f'channels: {channels}')
    print(f'num_class: {num_class}')
    print(f'output_stride: {output_stride}')
    print(f'epochs: {epochs}')
    print(f'batch_size: {batch_size}')
    print(f'val_fraction: {val_fraction}')
    print(f'num_images_train: {num_images_train}')
    print(f'num_images_test: {num_images_test}')
    print(f'patience: {patience}')
    print(f'model_path: {model_path}')
    print(f'history_path: {history_path}')


    run_case(train_dir = train_dir, test_dir = test_dir, patch_size = patch_size, channels = channels, num_class = num_class,
            output_stride = output_stride, epochs = epochs, batch_size = batch_size, val_fraction = val_fraction, num_images_train = num_images_train,
            num_images_test = num_images_test, patience = patience, model_path = model_path, history_path = history_path, lr = lr)