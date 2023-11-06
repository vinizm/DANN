from datetime import datetime
import time
import os
import gc
from copy import deepcopy
import pandas as pd

from experiment_config.domain_adaptation_config import DOMAIN_ADAPTATION_CONFIG, DOMAIN_ADAPTATION_GLOBAL_PARAMS
from config import PROCESSED_FOLDER, RESULTS_FOLDER, TEST_INDEX
from preprocess_images import remove_augmented_images
from custom_trainer import Trainer

import tensorflow as tf


physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print('Invalid device or cannot modify virtual devices once initialized.')


def grid_search(
    source_set: str,
    target_set: str,
    patch_size: int,
    stride_train: int,
    channels: int,
    num_class: int,
    output_stride: int,
    skip_conn: bool,
    units: int,
    num_runs: int,
    batch_size: int,
    val_fraction: float,
    num_images_train: int,
    rotate: bool,
    flip: bool,
    max_epochs: int,
    patience: int,
    progress_threshold: float,
    lr_config_main: dict,
    lr_config_extra: dict,
    optimizer_config: dict,
    gamma: float,
    lambda_scale: float,
    lambda_warmup: float,
    now: str
):
    
    EXP_DIR = f"{now}_bs{batch_size}_os{output_stride}_sc{skip_conn}_ls{lambda_scale}_da"
    PREFIX = f'DL_patch{patch_size}_{source_set}_{target_set}'
    
    for i in range(num_runs):
    
        source_dir = f'{PROCESSED_FOLDER}/{source_set}_patch{patch_size}_stride{stride_train}_Train'
        target_dir = f'{PROCESSED_FOLDER}/{target_set}_patch{patch_size}_stride{stride_train}_Train'
        
        remove_augmented_images(source_dir)
        remove_augmented_images(target_dir)
        
        trainer = Trainer(
            patch_size = patch_size,
            channels = channels,
            num_class = num_class,
            output_stride = output_stride,
            domain_adaptation = True,
            skip_conn = skip_conn,
            units = units,
            name = f'{now}_bs{batch_size}_os{output_stride}_sc{skip_conn}_ls{lambda_scale}_{source_set}_{target_set}_v{i + 1:02}'
            )
        trainer.set_test_index(test_index_source = TEST_INDEX.get(source_set), test_index_target = TEST_INDEX.get(target_set))
        trainer.compile_model()
        time.sleep(5) # Sleep for 5 seconds
        trainer.preprocess_images_domain_adaptation(
            patches_dir = [source_dir, target_dir],
            batch_size = batch_size,
            val_fraction = val_fraction,
            num_images = num_images_train,
            rotate = rotate,
            flip = flip
            )
        
        trainer.set_optimizer(deepcopy(optimizer_config))
        trainer.set_learning_rate(
            **{
                'segmentation': deepcopy(lr_config_main),
                'discriminator': deepcopy(lr_config_extra),
                }
            )
        
        config_lambda = {'warmup': lambda_warmup, 'gamma': gamma, 'lambda_scale': lambda_scale}
        trainer.set_lambda(**config_lambda)

        print(trainer.lr_function_segmentation.config)
        print(trainer.lr_function_discriminator.config)
        print(trainer.lambda_function.config)
        time.sleep(5) # Sleep for 5 seconds
        
        trainer.train_domain_adaptation(epochs = max_epochs, wait = patience, persist_best_model = True, progress_threshold = progress_threshold)

        LOW_LEVEL_DIR = f'{RESULTS_FOLDER}/{EXP_DIR}/{source_set}_{target_set}/v{i + 1:02}'
        if not os.path.exists(LOW_LEVEL_DIR):
            os.makedirs(LOW_LEVEL_DIR)
        
        weights_path = f'{LOW_LEVEL_DIR}/{PREFIX}_v{i + 1:02}_segmentation_weights.h5'
        trainer.save_weights(weights_path = weights_path, best = True, piece = 'segmentation')
        
        weights_path = f'{LOW_LEVEL_DIR}/{PREFIX}_v{i + 1:02}_discriminator_weights.h5'
        trainer.save_weights(weights_path = weights_path, best = True, piece = 'discriminator')
        
        history_path = f'{LOW_LEVEL_DIR}/{PREFIX}_v{i + 1:02}_history.json'
        trainer.save_info(history_path = history_path)
        
        del trainer
        gc.collect()
        
        time.sleep(15)
        
    return True


def convert_to_iterable(x):
    if isinstance(x, list) or isinstance(x, tuple):
        return x
    
    else:
        return [x]
    
def explode_cases(x: pd.DataFrame):
    for col in list(x.columns):
        x = x.explode(column = col, ignore_index = True)
    
    return x


for CASE in DOMAIN_ADAPTATION_CONFIG:
    
    run_training = CASE.get('run_training')
    if not run_training:
        continue
    
    source_set = convert_to_iterable(CASE.get('source'))
    target_set = convert_to_iterable(CASE.get('target'))
    patch_size = convert_to_iterable(CASE.get('patch_size', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('patch_size')))
    stride_train = convert_to_iterable(patch_size[0] // 2)
    channels = convert_to_iterable(CASE.get('channels', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('channels')))
    num_class = convert_to_iterable(CASE.get('num_class', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('num_class')))
    output_stride = convert_to_iterable(CASE.get('output_stride', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('output_stride')))
    skip_conn = convert_to_iterable(CASE.get('skip_conn', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('skip_conn')))
    units = convert_to_iterable(CASE.get('units', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('units')))
    num_runs = convert_to_iterable(CASE.get('num_runs', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('num_runs')))
    
    batch_size = convert_to_iterable(CASE.get('batch_size', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('batch_size')))
    val_fraction = convert_to_iterable(CASE.get('val_fraction', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('val_fraction')))
    num_images_train = convert_to_iterable(CASE.get('num_images_train', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('num_images_train')))
    rotate = convert_to_iterable(CASE.get('rotate', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('rotate')))
    flip = convert_to_iterable(CASE.get('flip', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('flip')))
    max_epochs = convert_to_iterable(CASE.get('max_epochs', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('max_epochs')))
    patience = convert_to_iterable(CASE.get('patience', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('patience')))
    progress_threshold = convert_to_iterable(CASE.get('progress_threshold', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('progress_threshold')))
    
    lr_config_main = convert_to_iterable(CASE.get('lr_config_main', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('lr_config_main')))
    lr_config_extra = convert_to_iterable(CASE.get('lr_config_extra', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('lr_config_extra')))
    optimizer_config = convert_to_iterable(CASE.get('optimizer_config', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('optimizer_config')))
    
    gamma = convert_to_iterable(CASE.get('gamma', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('gamma')))
    lambda_scale = convert_to_iterable(CASE.get('lambda_scale', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('lambda_scale')))
    lambda_warmup = convert_to_iterable(CASE.get('lambda_warmup', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('lambda_warmup')))

    hyperparameters = pd.DataFrame(columns = [
        'source_set', 'target_set', 'patch_size',
        'stride_train', 'channels', 'num_class',
        'output_stride', 'skip_conn', 'units',
        'num_runs', 'batch_size', 'val_fraction',
        'num_images_train', 'rotate', 'flip',
        'max_epochs', 'patience', 'progress_threshold',
        'lr_config_main', 'lr_config_extra', 'optimizer_config',
        'gamma', 'lambda_scale', 'lambda_warmup'
    ])
    hyperparameters.loc[len(hyperparameters)] = [
        source_set, target_set, patch_size,
        stride_train, channels, num_class,
        output_stride, skip_conn, units,
        num_runs, batch_size, val_fraction,
        num_images_train, rotate, flip,
        max_epochs, patience, progress_threshold,
        lr_config_main, lr_config_extra, optimizer_config,
        gamma, lambda_scale, lambda_warmup
    ]
    hyperparameters = explode_cases(hyperparameters)
    MINOR_CASES = hyperparameters.to_dict(orient = 'list')

    for k in range(len(hyperparameters)):
        
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        grid_search(
            source_set = MINOR_CASES['source_set'][k],
            target_set = MINOR_CASES['target_set'][k],
            patch_size = MINOR_CASES['patch_size'][k],
            stride_train = MINOR_CASES['stride_train'][k],
            channels = MINOR_CASES['channels'][k],
            num_class = MINOR_CASES['num_class'][k],
            output_stride = MINOR_CASES['output_stride'][k],
            skip_conn = MINOR_CASES['skip_conn'][k],
            units = MINOR_CASES['units'][k],
            num_runs = MINOR_CASES['num_runs'][k],
            batch_size = MINOR_CASES['batch_size'][k],
            val_fraction = MINOR_CASES['val_fraction'][k],
            num_images_train = MINOR_CASES['num_images_train'][k],
            rotate = MINOR_CASES['rotate'][k],
            flip = MINOR_CASES['flip'][k],
            max_epochs = MINOR_CASES['max_epochs'][k],
            patience = MINOR_CASES['patience'][k],
            progress_threshold = MINOR_CASES['progress_threshold'][k],
            lr_config_main = MINOR_CASES['lr_config_main'][k],
            lr_config_extra = MINOR_CASES['lr_config_extra'][k],
            optimizer_config = MINOR_CASES['optimizer_config'][k],
            gamma = MINOR_CASES['gamma'][k],
            lambda_scale = MINOR_CASES['lambda_scale'][k],
            lambda_warmup = MINOR_CASES['lambda_warmup'][k],
            now = now
        )
