from experiment_config.no_domain_adaptation_config import NO_DOMAIN_ADAPTATION_CONFIG, NO_DOMAIN_ADAPTATION_GLOBAL_PARAMS
from config import PROCESSED_FOLDER, RESULTS_FOLDER, TEST_INDEX
from preprocess_images import remove_augmented_images
from custom_trainer import Trainer

from datetime import datetime
import time
import os
import gc
import time
from copy import deepcopy

import tensorflow as tf


physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print('Invalid device or cannot modify virtual devices once initialized.')

now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
EXP_DIR = f'{now}_no_da'

for CASE in NO_DOMAIN_ADAPTATION_CONFIG:
    
    run_training = CASE.get('run_training')
    if not run_training:
        continue
    
    dataset = CASE.get('dataset')
    patch_size = CASE.get('patch_size', NO_DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('patch_size'))
    stride_train = patch_size // 2
    channels = CASE.get('channels', NO_DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('channels'))
    num_class = CASE.get('num_class', NO_DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('num_class'))
    output_stride = CASE.get('output_stride', NO_DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('output_stride'))
    skip_conn = CASE.get('skip_conn', NO_DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('skip_conn'))
    num_runs = CASE.get('num_runs', NO_DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('num_runs'))
    
    batch_size = CASE.get('batch_size', NO_DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('batch_size'))
    val_fraction = CASE.get('val_fraction', NO_DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('val_fraction'))
    num_images_train = CASE.get('num_images_train', NO_DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('num_images_train'))
    rotate = CASE.get('rotate', NO_DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('rotate'))
    flip = CASE.get('flip', NO_DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('flip'))
    max_epochs = CASE.get('max_epochs', NO_DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('max_epochs'))
    patience = CASE.get('patience', NO_DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('patience'))
    progress_threshold = CASE.get('progress_threshold', NO_DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('progress_threshold'))
    
    lr_config = CASE.get('lr_config', NO_DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('lr_config'))
    optimizer_config = CASE.get('optimizer_config', NO_DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('optimizer_config'))
    
    PREFIX = f'DL_patch{patch_size}_os{output_stride}_{dataset}'

    for i in range(num_runs):
    
        patches_dir = f'{PROCESSED_FOLDER}/{dataset}_patch{patch_size}_stride{stride_train}_Train'
        remove_augmented_images(patches_dir)
        
        trainer = Trainer(
            patch_size = patch_size,
            channels = channels,
            num_class = num_class,
            output_stride = output_stride,
            domain_adaptation = False,
            skip_conn = skip_conn,
            name = f'{now}_{dataset}_v{i + 1:02}'
            )
        trainer.set_test_index(test_index_source = TEST_INDEX.get(dataset), test_index_target = [])
        trainer.compile_model()
        time.sleep(5) # Sleep for 5 seconds
        trainer.preprocess_images(
            patches_dir = patches_dir,
            batch_size = batch_size,
            val_fraction = val_fraction,
            num_images = num_images_train,
            rotate = rotate,
            flip = flip
            )
        
        trainer.set_optimizer(deepcopy(optimizer_config))
        trainer.set_learning_rate(
            **{'segmentation': deepcopy(lr_config)}
            )
        
        trainer.train(epochs = max_epochs, wait = patience, persist_best_model = True, progress_threshold = progress_threshold)
        
        LOW_LEVEL_DIR = f'{RESULTS_FOLDER}/{EXP_DIR}/{dataset}/v{i + 1:02}'
        if not os.path.exists(LOW_LEVEL_DIR):
            os.makedirs(LOW_LEVEL_DIR)    
        
        weights_path = f'{LOW_LEVEL_DIR}/{PREFIX}_v{i + 1:02}_weights.h5'
        trainer.save_weights(weights_path = weights_path, best = True, piece = None)
        
        history_path = f'{LOW_LEVEL_DIR}/{PREFIX}_v{i + 1:02}_history.json'
        trainer.save_info(history_path = history_path)
        
        del trainer
        gc.collect()
        
        time.sleep(15) # Sleep for 15 seconds