from experiment_config.domain_adaptation_config import DOMAIN_ADAPTATION_CONFIG, DOMAIN_ADAPTATION_GLOBAL_PARAMS
from config import PROCESSED_FOLDER, RESULTS_FOLDER, TEST_INDEX
from preprocess_images import remove_augmented_images
from custom_trainer import Trainer

from datetime import datetime
import tensorflow as tf
import time
import os
import gc

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print('Invalid device or cannot modify virtual devices once initialized.')

now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
EXP_DIR = f'{now}_da'

for CASE in DOMAIN_ADAPTATION_CONFIG:
    
    run_training = CASE.get('run_training')
    if not run_training:
        continue
    
    source_set = CASE.get('source')
    target_set = CASE.get('target')
    patch_size = CASE.get('patch_size', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('patch_size'))
    stride_train = patch_size // 2
    channels = CASE.get('channels', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('channels'))
    num_class = CASE.get('num_class', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('num_class'))
    output_stride = CASE.get('output_stride', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('output_stride'))
    skip_conn = CASE.get('skip_conn', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('skip_conn'))
    units = CASE.get('units', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('units'))
    num_runs = CASE.get('num_runs', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('num_runs'))
    
    batch_size = CASE.get('batch_size', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('batch_size'))
    val_fraction = CASE.get('val_fraction', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('val_fraction'))
    num_images_train = CASE.get('num_images_train', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('num_images_train'))
    rotate = CASE.get('rotate', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('rotate'))
    flip = CASE.get('flip', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('flip'))
    max_epochs = CASE.get('max_epochs', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('max_epochs'))
    patience = CASE.get('patience', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('patience'))
    progress_threshold = CASE.get('progress_threshold', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('progress_threshold'))
    
    lr_config = CASE.get('lr_config', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('lr_config'))
    
    gamma = CASE.get('gamma', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('gamma'))
    lambda_scale = CASE.get('lambda_scale', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('lambda_scale'))
    lambda_warmup = CASE.get('lambda_warmup', DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('lambda_warmup'))
    
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
            name = f'{now}_{source_set}_{target_set}_v{i + 1:02}'
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
        
        trainer.set_learning_rate(**{'segmentation': lr_config, 'discriminator': lr_config})
        
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