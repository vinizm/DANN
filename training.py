import os
import tensorflow as tf
import logging

from variables import *
from main import Train_Case


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# if tf.test.gpu_device_name():
#     print('GPU found')
# else:
#     print("No GPU found")


if __name__ == '__main__':

    one_channel = True

    train_dir = f'{PROCESSED_FOLDER}/Fe19_stride256_onechannel{one_channel}_Train'
    test_dir = f'{PROCESSED_FOLDER}/Fe19_stride256_onechannel{one_channel}_Test'
    lr = 1.e-4
    patch_size = 512
    channels = 3 if not one_channel else 1
    num_class = 2
    output_stride = 8
    epochs = 25
    batch_size = 2
    val_fraction = 0.2
    num_images_train = None
    num_images_test = None
    patience = 10
    augment = True

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
    print(f'augment: {augment}')
    print(f'patience: {patience}')
    print(f'model_path: {model_path}')
    print(f'history_path: {history_path}')


    Train_Case(train_dir = train_dir, test_dir = test_dir, patch_size = patch_size, channels = channels, num_class = num_class,
               output_stride = output_stride, epochs = epochs, batch_size = batch_size, val_fraction = val_fraction,
               num_images_train = num_images_train, num_images_test = num_images_test, patience = patience, model_path = model_path,
               history_path = history_path, lr = lr, augment = augment)