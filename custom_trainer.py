import time
import numpy as np
import glob
from typing import List, Tuple, Sequence

import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import Loss
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, BinaryAccuracy, Precision, Recall
from tensorflow.keras.models import Model
from tensorflow.keras.models import save_model
from tensorflow.keras.backend import clear_session

from utils.utils import load_array, save_json, augment_images
from utils.hyperparameters import *
from utils.hyperparameters import LambdaGradientReversalLayer
from utils.learning_rate_functions import LearningRateFactory
from utils.metrics import f1, AveragePrecision
from models.builder import DomainAdaptationModel
from models.deeplabv3plus import DeepLabV3Plus
from logger import TensorBoardLogger

from config import LOGS_FOLDER


class Trainer():

    def __init__(self, patch_size: int = 512, channels: int = 1, num_class: int = 2, output_stride = 8, skip_conn: bool = True, domain_adaptation: bool = False,
                 units: int = 1024, name: str = ''):

        self.name = name if name == '' else f'_{name}'
        self.patch_size = patch_size
        self.channels = channels
        self.num_class = num_class
        self.output_stride = output_stride
        self.skip_conn = skip_conn
        self.domain_adaptation = domain_adaptation
        self.units = units

        self.model = self.assembly_empty_model()
        
        self.optimizer_segmentation = Adam()
        self.optimizer_discriminator = Adam()

        self.lr_function_segmentation = LearningRateFactory('exp', lr0 = LR0, warmup = LR_WARMUP, alpha = ALPHA, beta = BETA)
        self.lr_function_discriminator = LearningRateFactory('exp', lr0 = LR0, warmup = LR_WARMUP, alpha = ALPHA, beta = BETA)

        self.lambda_function = LambdaGradientReversalLayer(warmup = LAMBDA_WARMUP, gamma = GAMMA, lambda_scale = LAMBDA_SCALE)

        self.loss_function = BinaryCrossentropy()
        self.loss_function_segmentation = BinaryCrossentropy()

        self.loss_function_discriminator = BinaryCrossentropy()
        self.acc_function_discriminator = BinaryAccuracy(threshold = 0.5)
        
        # ===== [SOURCE] =====
        self.acc_segmentation = CategoricalAccuracy()
        self.precision = Precision()
        self.recall = Recall()
        
        # ===== [TARGET] =====
        self.acc_segmentation_target = CategoricalAccuracy()
        self.precision_target = Precision()
        self.recall_target = Recall()

        # =======================
        
        self.loss_discriminator_train_history = []
        self.loss_discriminator_val_history = []

        self.acc_discriminator_train_history = []
        self.acc_discriminator_val_history = []

        # ===== [SOURCE] =====
        self.loss_segmentation_train_history = []
        self.loss_segmentation_val_history = []

        self.acc_segmentation_train_history = []
        self.acc_segmentation_val_history = []
        
        self.precision_train_history = []
        self.precision_val_history = []

        self.recall_train_history = []
        self.recall_val_history = []
        
        self.f1_train_history = []
        self.f1_val_history = []
        
        self.map_segmentation_train_history = []
        self.map_segmentation_val_history = []
        
        # ===== [TARGET] =====
        self.loss_segmentation_target_train_history = []
        self.loss_segmentation_target_val_history = []        
        
        self.acc_segmentation_target_train_history = []
        self.acc_segmentation_target_val_history = []
        
        self.precision_target_train_history = []
        self.precision_target_val_history = []

        self.recall_target_train_history = []
        self.recall_target_val_history = []
        
        self.f1_target_train_history = []
        self.f1_target_val_history = []
        
        self.map_segmentation_target_train_history = []
        self.map_segmentation_target_val_history = []        
        # =======================

        self.lr_segmentation_history = []
        self.lr_discriminator_history = []
        self.lambdas = []

        self.logger = TensorBoardLogger()
        self.logger.create_writer('train_writer', f'{LOGS_FOLDER}/train{self.name}/')
        self.logger.create_writer('val_writer', f'{LOGS_FOLDER}/validation{self.name}/')
        self.logger.create_writer('segmentation_writer', f'{LOGS_FOLDER}/segmentation{self.name}/')
        self.logger.create_writer('discriminator_writer', f'{LOGS_FOLDER}/discriminator{self.name}/')

        self.no_improvement_count = 0
        self.best_val_loss = 1.e8
        self.progress_threshold = 0.
        self.best_weights = None
        self.best_epoch = None

        self.patches_dir = None
        self.train_data_dirs = []
        self.val_data_dirs = []

        self.train_data_dirs_source = []
        self.val_data_dirs_source = []

        self.train_data_dirs_target = []
        self.val_data_dirs_target = []

        self.val_fraction = 0.1
        self.batch_size = 2
        self.num_images = 60
        self.epochs = 25
        self.wait = 12
        self.rotate = True
        self.flip = True

        self.num_batches_train = None
        self.num_batches_val = None

        self.test_index_source = []
        self.test_index_target = []

        self.elapsed_time = 0

    def assembly_empty_model(self):

        if self.domain_adaptation:
            empty_model = DomainAdaptationModel(
                input_shape = (self.patch_size, self.patch_size, self.channels),
                output_stride = self.output_stride,
                num_class = self.num_class,
                units = self.units,
                skip_conn = self.skip_conn
                )
        else:
            empty_model = DeepLabV3Plus(
                input_shape = (self.patch_size, self.patch_size, self.channels),
                num_class = self.num_class,
                output_stride = self.output_stride,
                skip_conn = self.skip_conn
                )
        
        return empty_model

    @tf.function
    def _training_step(self, model, x_true, y_true, optimizer):

        with tf.GradientTape() as tape:
            y_pred = model(x_true)
            loss = self.loss_function(y_true, y_pred)
        
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        
        del tape
        return loss, y_pred

    @tf.function
    def _training_step_domain_adaptation(self, model: DomainAdaptationModel, inputs, outputs, lambda_value, optimizers: Sequence[Optimizer],
                                         source_mask, target_mask, train_segmentation = True, train_discriminator = True):

        optimizer_segmentation, optimizer_discriminator = optimizers
        y_true_segmentation, y_true_discriminator = outputs
        with tf.GradientTape(persistent = True) as tape:
                        
            y_pred_segmentation, y_pred_discriminator = model(inputs)
            
            loss_segmentation_source = self.loss_function_segmentation(y_true_segmentation, y_pred_segmentation, sample_weight = source_mask)
            loss_discriminator = self.loss_function_discriminator(y_true_discriminator, y_pred_discriminator)
            loss_global_fake = loss_segmentation_source - lambda_value * loss_discriminator
            
        loss_segmentation_target = self.loss_function_segmentation(y_true_segmentation, y_pred_segmentation, sample_weight = target_mask)

        if train_segmentation:

            # update feature extractor
            gradients_encoder = tape.gradient(loss_global_fake, model.main_network.encoder.trainable_weights)
            optimizer_segmentation.apply_gradients(zip(gradients_encoder, model.main_network.encoder.trainable_weights))        
            
            # update label predictor
            gradients_decoder = tape.gradient(loss_segmentation_source, model.main_network.decoder.trainable_weights)
            optimizer_segmentation.apply_gradients(zip(gradients_decoder, model.main_network.decoder.trainable_weights))     

        if train_discriminator:
            
            # update discriminator
            gradients_discriminator = tape.gradient(loss_discriminator, model.domain_discriminator.trainable_weights)
            optimizer_discriminator.apply_gradients(zip(gradients_discriminator, model.domain_discriminator.trainable_weights))

        del tape
        return loss_segmentation_source, loss_segmentation_target, loss_discriminator, y_pred_segmentation, y_pred_discriminator

    def _augment_images(self, data_dirs: list):
        if self.rotate or self.flip:
            augmented_dirs = augment_images(image_files = data_dirs, angles = [90, 180, 270],
                                             rotate = self.rotate, flip = self.flip, verbose = False)
            data_dirs += augmented_dirs
            np.random.shuffle(data_dirs)

        return data_dirs

    def compile_model(self, show_summary: bool = True):
        self.model.compile(optimizer = self.optimizer_segmentation)

        if show_summary:
            self.model.summary()

    def _split_file_list(self, data_dirs: list):

        num_val_samples = int(len(data_dirs) * self.val_fraction)
        train_data_dirs = data_dirs[num_val_samples :]
        val_data_dirs = data_dirs[: num_val_samples]

        return train_data_dirs, val_data_dirs

    def _load_file_names(self, patches_dir: str):

        data_dirs = glob.glob(patches_dir + '/*.npy')
        np.random.shuffle(data_dirs)
        data_dirs = data_dirs[: self.num_images] # reduce dataset size

        return data_dirs

    def _calculate_batch_size(self, data_dirs: list):
        num_batches = len(data_dirs) // self.batch_size
        return num_batches

    def _explode_domain(self, encoded_domain: list):
        feature_size = self.patch_size // self.output_stride
        fill = lambda x: np.full(shape = (feature_size, feature_size, 1), fill_value = x, dtype = 'int32')
        
        return np.asarray([fill(domain) for domain in encoded_domain], dtype = 'int32')
    
    @staticmethod
    def _intercalate_lists(l1: list, l2: list):
        
        np.random.shuffle(l1)
        np.random.shuffle(l2)
        
        zipped = list(zip(l1, l2))
        intercalated = [el for sublist in zipped for el in sublist]
        return intercalated
    
    @staticmethod
    def _encode_domain(file_names: list, source_files: list):
        return np.asarray([0 if file_name in source_files else 1 for file_name in file_names])
    
    @staticmethod            
    def _persist_to_history(inputs, operator, history: list):
        value = operator(inputs)
        history.append(value)

    @staticmethod
    def _generate_domain_mask(encoded_domain: list, shape: tuple, activate_source: bool):
        source_fill, target_fill = 1., 0.
        
        if not activate_source:
            source_fill, target_fill = 0., 1.

        flatten = False
        if shape == -1:
            shape = (1,)
            flatten = True
        
        mask = np.asarray([np.full(shape, fill_value = source_fill) if domain == 0 else np.full(shape, fill_value = target_fill) for domain in encoded_domain], dtype = 'float32')

        if flatten:
            mask = mask.reshape(-1)
        
        return mask
    
    def _calculate_precision_recall(self, y_true, y_pred):
        y_true = tf.math.argmax(y_true, axis = -1)
        y_pred = tf.math.argmax(y_pred, axis = -1)
        
        self.precision.update_state(y_true, y_pred)
        self.recall.update_state(y_true, y_pred)        

    def reset_history(self):
        self.loss_discriminator_train_history = []
        self.loss_discriminator_val_history = []

        self.acc_discriminator_train_history = []
        self.acc_discriminator_val_history = []

        # ===== [SOURCE] =====
        self.loss_segmentation_train_history = []
        self.loss_segmentation_val_history = []

        self.acc_segmentation_train_history = []
        self.acc_segmentation_val_history = []
        
        self.precision_train_history = []
        self.precision_val_history = []

        self.recall_train_history = []
        self.recall_val_history = []
        
        self.f1_train_history = []
        self.f1_val_history = []
        
        self.map_segmentation_train_history = []
        self.map_segmentation_val_history = []        
        
        # ===== [TARGET] =====
        self.loss_segmentation_target_train_history = []
        self.loss_segmentation_target_val_history = []
        
        self.acc_segmentation_target_train_history = []
        self.acc_segmentation_target_val_history = []
        
        self.precision_target_train_history = []
        self.precision_target_val_history = []

        self.recall_target_train_history = []
        self.recall_target_val_history = []
        
        self.f1_target_train_history = []
        self.f1_target_val_history = []
        
        self.map_segmentation_target_train_history = []
        self.map_segmentation_target_val_history = []           
        
        # =======================

        self.lr_segmentation_history = []
        self.lr_discriminator_history = []
        self.lambdas = []

        self.no_improvement_count = 0
        self.best_val_loss = 1.e8
        
    def reset_states(self):
        self.acc_function_discriminator.reset_states()
        
        # ===== [SOURCE] =====
        self.acc_segmentation.reset_states()
        self.precision.reset_states()
        self.recall.reset_states()
        
        # ===== [TARGET] =====
        self.acc_segmentation_target.reset_states()
        self.precision_target.reset_states()
        self.recall_target.reset_states()

    def preprocess_images_domain_adaptation(self, patches_dir: list, batch_size: int = 2, val_fraction: float = 0.1,
        num_images: int = 60, rotate: bool = True, flip: bool = True):

        self.patches_dir = patches_dir # list: [source, target]
        self.val_fraction = val_fraction
        self.batch_size = batch_size
        self.num_images = num_images
        self.rotate = rotate
        self.flip = flip
    
        # loading source dataset
        data_dirs_source = self._load_file_names(self.patches_dir[0])
        # loading target dataset
        data_dirs_target = self._load_file_names(self.patches_dir[1])
        
        # define files of source domain
        train_data_dirs_source, val_data_dirs_source = self._split_file_list(data_dirs_source)
        # define files of target domain
        train_data_dirs_target, val_data_dirs_target = self._split_file_list(data_dirs_target)

        # augment source images
        self.train_data_dirs_source = self._augment_images(train_data_dirs_source)
        self.val_data_dirs_source = self._augment_images(val_data_dirs_source)

        # augment target images
        self.train_data_dirs_target = self._augment_images(train_data_dirs_target)
        self.val_data_dirs_target = self._augment_images(val_data_dirs_target)
        
        # intercalate elements
        self.train_data_dirs = self._intercalate_lists(self.train_data_dirs_source, self.train_data_dirs_target)
        self.val_data_dirs = self._intercalate_lists(self.val_data_dirs_source, self.val_data_dirs_target)

        # compute number of batches
        self.num_batches_train = self._calculate_batch_size(self.train_data_dirs)
        self.num_batches_val = self._calculate_batch_size(self.val_data_dirs)

        print(f'num. of batches for training: {self.num_batches_train}')
        print(f'num. of batches for validation: {self.num_batches_val}')

    def preprocess_images(self, patches_dir: str, batch_size: int = 2, val_fraction: float = 0.1,
        num_images: int = 60, rotate: bool = True, flip: bool = True):

        self.patches_dir = patches_dir
        self.val_fraction = val_fraction
        self.batch_size = batch_size
        self.num_images = num_images
        self.rotate = rotate
        self.flip = flip
    
        # loading dataset
        data_dirs = self._load_file_names(self.patches_dir)

        # define files for validation
        self.train_data_dirs, self.val_data_dirs = self._split_file_list(data_dirs)

        # data augmentation_s
        self.train_data_dirs = self._augment_images(self.train_data_dirs)
        self.val_data_dirs = self._augment_images(self.val_data_dirs)

        # compute number of batches
        self.num_batches_train = self._calculate_batch_size(self.train_data_dirs)
        self.num_batches_val = self._calculate_batch_size(self.val_data_dirs)

        print(f'num. of batches for training: {self.num_batches_train}')
        print(f'num. of batches for validation: {self.num_batches_val}')

    def train_domain_adaptation(self, epochs: int = 25, wait: int = 12, persist_best_model: bool = True, progress_threshold: float = 0.):

        self.reset_history()
        time_init = time.time()

        self.epochs = epochs
        self.wait = wait if wait is not None else 1e8
        self.progress_threshold = progress_threshold
        self.persist_best_model = persist_best_model

        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1} of {self.epochs}')
            total_loss_segmentation_source = 0.
            total_loss_segmentation_target = 0.
            total_loss_discriminator = 0.
            
            self.reset_states()

            self.train_data_dirs = self._intercalate_lists(self.train_data_dirs_source, self.train_data_dirs_target)
            self.val_data_dirs = self._intercalate_lists(self.val_data_dirs_source, self.val_data_dirs_target)

            # update learning rate
            p = epoch / (epochs - 1)
            print(f'Training Progress: {p}')
            
            lr_1 = self.lr_function_segmentation.calculate(p)
            print(f'Learning Rate Segmentation: {lr_1}')
            self.optimizer_segmentation.lr = lr_1
            self.lr_segmentation_history.append(lr_1)

            lr_2 = self.lr_function_discriminator.calculate(p)
            print(f'Learning Rate Discriminator: {lr_2}')
            self.optimizer_discriminator.lr = lr_2
            self.lr_discriminator_history.append(lr_2)

            # set lambda value
            l = np.float32(self.lambda_function.calculate(p))
            print(f'Lambda: {l}')
            self.lambdas.append(float(l))

            self.logger.write_scalar('segmentation_writer', 'learning_rate', lr_1, epoch + 1)
            self.logger.write_scalar('discriminator_writer', 'learning_rate', lr_2, epoch + 1)
            self.logger.write_scalar('discriminator_writer', 'lambda', l, epoch + 1)

            print('Start training...')
            for batch in range(self.num_batches_train):

                print(f'Batch {batch + 1} of {self.num_batches_train}')
                batch_train_files = self.train_data_dirs[batch * self.batch_size : (batch + 1) * self.batch_size]
                
                # shuffle indices
                shuffled_idx = list(range(len(batch_train_files)))
                np.random.shuffle(shuffled_idx)
                                
                # load images for training
                batch_images = np.asarray([load_array(batch_train_file, verbose = False) for batch_train_file in batch_train_files])
                batch_images = batch_images.astype(np.float32) # set np.float32 to reduce memory usage

                x_train = batch_images[shuffled_idx, :, :, : self.channels]
                y_segmentation_train = batch_images[shuffled_idx, :, :, self.channels :]
                
                encoded_domain = self._encode_domain(batch_train_files, self.train_data_dirs_source)[shuffled_idx]
                # y_discriminator_train = self._explode_domain(encoded_domain)
                # y_discriminator_train = np.asarray(encoded_domain).reshape((self.batch_size, 1))
                y_discriminator_train = tf.one_hot(encoded_domain, 2)
                print(f'Domain: {encoded_domain}')

                source_mask = self._generate_domain_mask(encoded_domain, shape = (self.patch_size, self.patch_size), activate_source = True)
                target_mask = self._generate_domain_mask(encoded_domain, shape = (self.patch_size, self.patch_size), activate_source = False)

                # source_mask = self._generate_domain_mask(encoded_domain, shape = -1, activate_source = True)
                # target_mask = self._generate_domain_mask(encoded_domain, shape = -1, activate_source = False)
                
                step_output = self._training_step_domain_adaptation(
                    model = self.model,
                    inputs = x_train,
                    outputs = [y_segmentation_train, y_discriminator_train],
                    lambda_value = l,
                    optimizers = (self.optimizer_segmentation, self.optimizer_discriminator),
                    source_mask = source_mask,
                    target_mask = target_mask,
                    train_segmentation = True,
                    train_discriminator = True
                    )
                loss_segmentation_source, loss_segmentation_target, loss_discriminator, y_segmentation_pred, y_discriminator_pred = step_output

                # ===== [SOURCE] ACCURACY =====
                self.acc_segmentation.update_state(y_segmentation_train, y_segmentation_pred, sample_weight = source_mask)
                
                # ===== [TARGET] ACCURACY =====
                self.acc_segmentation_target.update_state(y_segmentation_train, y_segmentation_pred, sample_weight = target_mask)
                
                y_segmentation_train_max = tf.math.argmax(y_segmentation_train, axis = -1)
                y_segmentation_pred_max = tf.math.argmax(y_segmentation_pred, axis = -1)

                # ===== [SOURCE] PRECISION/RECALL =====
                self.precision.update_state(y_segmentation_train_max, y_segmentation_pred_max, sample_weight = source_mask)
                self.recall.update_state(y_segmentation_train_max, y_segmentation_pred_max, sample_weight = source_mask)   
                
                # ===== [TARGET] PRECISION/RECALL =====
                self.precision_target.update_state(y_segmentation_train_max, y_segmentation_pred_max, sample_weight = target_mask)
                self.recall_target.update_state(y_segmentation_train_max, y_segmentation_pred_max, sample_weight = target_mask)                     

                # ===== DISCRIMINATOR =====
                self.acc_function_discriminator.update_state(y_discriminator_train, y_discriminator_pred)

                total_loss_segmentation_source += float(loss_segmentation_source)
                total_loss_segmentation_target += float(loss_segmentation_target)
                total_loss_discriminator += float(loss_discriminator)

            # ===== DISCRIMINATOR =====
            self._persist_to_history(total_loss_discriminator, lambda x: x / self.num_batches_train, self.loss_discriminator_train_history)
            self._persist_to_history(self.acc_function_discriminator, lambda x: float(x.result()), self.acc_discriminator_train_history)    

            # ===== [SOURCE] LOSS =====
            self._persist_to_history(total_loss_segmentation_source, lambda x: x / self.num_batches_train, self.loss_segmentation_train_history)

            # ===== [SOURCE] ACCURACY =====
            self._persist_to_history(self.acc_segmentation, lambda x: float(x.result()), self.acc_segmentation_train_history)
            
            # ===== [SOURCE] PRECISION =====
            self._persist_to_history(self.precision, lambda x: float(x.result()), self.precision_train_history)
            
            # ===== [SOURCE] RECALL =====
            self._persist_to_history(self.recall, lambda x: float(x.result()), self.recall_train_history)
            
            # ===== [SOURCE] F1 =====
            a, b = (self.precision_train_history[-1], self.recall_train_history[-1])
            self._persist_to_history((a, b), lambda x: f1(*x), self.f1_train_history)
            
            # ===== [TARGET] LOSS =====
            self._persist_to_history(total_loss_segmentation_target, lambda x: x / self.num_batches_train, self.loss_segmentation_target_train_history)            
            
            # ===== [TARGET] ACCURACY =====
            self._persist_to_history(self.acc_segmentation_target, lambda x: float(x.result()), self.acc_segmentation_target_train_history)
            
            # ===== [TARGET] PRECISION =====
            self._persist_to_history(self.precision_target, lambda x: float(x.result()), self.precision_target_train_history)
            
            # ===== [TARGET] RECALL =====
            self._persist_to_history(self.recall_target, lambda x: float(x.result()), self.recall_target_train_history)
            
            # ===== [TARGET] F1 =====
            a, b = (self.precision_target_train_history[-1], self.recall_target_train_history[-1])
            self._persist_to_history((a, b), lambda x: f1(*x), self.f1_target_train_history)                     

            # ===== [SOURCE] =====
            self.logger.write_scalar('train_writer', 'metric/loss/segmentation/source', self.loss_segmentation_train_history[-1], epoch + 1)
            self.logger.write_scalar('train_writer', 'metric/accuracy/segmentation/source', self.acc_segmentation_train_history[-1], epoch + 1)
            self.logger.write_scalar('train_writer', 'metric/precision/segmentation/source', self.precision_train_history[-1], epoch + 1)
            self.logger.write_scalar('train_writer', 'metric/recall/segmentation/source', self.recall_train_history[-1], epoch + 1)
            self.logger.write_scalar('train_writer', 'metric/f1/segmentation/source', self.f1_train_history[-1], epoch + 1)
            
            # ===== [TARGET] =====
            self.logger.write_scalar('train_writer', 'metric/loss/segmentation/target', self.loss_segmentation_target_train_history[-1], epoch + 1)
            self.logger.write_scalar('train_writer', 'metric/accuracy/segmentation/target', self.acc_segmentation_target_train_history[-1], epoch + 1)
            self.logger.write_scalar('train_writer', 'metric/precision/segmentation/target', self.precision_target_train_history[-1], epoch + 1)
            self.logger.write_scalar('train_writer', 'metric/recall/segmentation/target', self.recall_target_train_history[-1], epoch + 1)
            self.logger.write_scalar('train_writer', 'metric/f1/segmentation/target', self.f1_target_train_history[-1], epoch + 1)
            
            # ===== DISCRIMINATOR =====
            self.logger.write_scalar('train_writer', 'metric/loss/discriminator', self.loss_discriminator_train_history[-1], epoch + 1)
            self.logger.write_scalar('train_writer', 'metric/accuracy/discriminator', self.acc_discriminator_train_history[-1], epoch + 1)

            # ===== [SOURCE] =====
            print(f'[SOURCE] Segmentation Loss: {self.loss_segmentation_train_history[-1]}')
            print(f'[SOURCE] Segmentation Accuracy: {self.acc_segmentation_train_history[-1]}')
            print(f'[SOURCE] Segmentation Precision: {self.precision_train_history[-1]}')
            print(f'[SOURCE] Segmentation Recall: {self.recall_train_history[-1]}')
            print(f'[SOURCE] Segmentation F1: {self.f1_train_history[-1]}')
            
            # ===== [TARGET] =====
            print(f'[TARGET] Segmentation Loss: {self.loss_segmentation_target_train_history[-1]}')
            print(f'[TARGET] Segmentation Accuracy: {self.acc_segmentation_target_train_history[-1]}')
            print(f'[TARGET] Segmentation Precision: {self.precision_target_train_history[-1]}')
            print(f'[TARGET] Segmentation Recall: {self.recall_target_train_history[-1]}')
            print(f'[TARGET] Segmentation F1: {self.f1_target_train_history[-1]}')
            
            # ===== DISCRIMINATOR =====
            print(f'Discriminator Loss: {self.loss_discriminator_train_history[-1]}')
            print(f'Discriminator Accuracy: {self.acc_discriminator_train_history[-1]}')

            self.reset_states()
            total_loss_segmentation_source = 0.
            total_loss_segmentation_target = 0.
            total_loss_discriminator = 0.

            # evaluating network
            print('Start validation...')
            for batch in range(self.num_batches_val):
                print(f'Batch {batch + 1} of {self.num_batches_val}')
                batch_val_files = self.val_data_dirs[batch * self.batch_size : (batch + 1) * self.batch_size]

                # shuffle indices
                shuffled_idx = list(range(len(batch_val_files)))
                np.random.shuffle(shuffled_idx)

                # load images for testing
                batch_val_images = np.asarray([load_array(batch_val_file, verbose = False) for batch_val_file in batch_val_files])
                batch_val_images = batch_val_images.astype(np.float32) # set np.float32 to reduce memory usage

                x_val = batch_val_images[shuffled_idx, :, :, : self.channels]
                y_segmentation_val = batch_val_images[shuffled_idx, :, :, self.channels :]
                
                encoded_domain = self._encode_domain(batch_val_files, self.val_data_dirs_source)[shuffled_idx]
                # y_discriminator_val = self._explode_domain(encoded_domain)
                # y_discriminator_val = np.asarray(encoded_domain).reshape((self.batch_size, 1))
                y_discriminator_val = tf.one_hot(encoded_domain, 2)
                print(f'Domain: {encoded_domain}')

                y_segmentation_pred, y_discriminator_pred = self.model(x_val)

                source_mask = self._generate_domain_mask(encoded_domain, shape = (self.patch_size, self.patch_size), activate_source = True)
                target_mask = self._generate_domain_mask(encoded_domain, shape = (self.patch_size, self.patch_size), activate_source = False)
                
                # source_mask = self._generate_domain_mask(encoded_domain, shape = -1, activate_source = True)
                # target_mask = self._generate_domain_mask(encoded_domain, shape = -1, activate_source = False)

                loss_segmentation_source = self.loss_function_segmentation(y_segmentation_val, y_segmentation_pred, sample_weight = source_mask)
                loss_segmentation_target = self.loss_function_segmentation(y_segmentation_val, y_segmentation_pred, sample_weight = target_mask)
                    
                total_loss_segmentation_source += float(loss_segmentation_source)
                total_loss_segmentation_target += float(loss_segmentation_target)                        
                
                # ===== DISCRIMINATOR =====
                loss_discriminator = self.loss_function_discriminator(y_discriminator_val, y_discriminator_pred)
                total_loss_discriminator += float(loss_discriminator)

                self.acc_function_discriminator.update_state(y_discriminator_val, y_discriminator_pred)
                
                # ===== [SOURCE] ACCURACY =====
                self.acc_segmentation.update_state(y_segmentation_val, y_segmentation_pred, sample_weight = source_mask)
                
                # ===== [TARGET] ACCURACY =====
                self.acc_segmentation_target.update_state(y_segmentation_val, y_segmentation_pred, sample_weight = target_mask)
                
                y_segmentation_val = tf.math.argmax(y_segmentation_val, axis = -1)
                y_segmentation_pred = tf.math.argmax(y_segmentation_pred, axis = -1)
                
                # ===== [SOURCE] PRECISION/RECALL =====
                self.precision.update_state(y_segmentation_val, y_segmentation_pred, sample_weight = source_mask)
                self.recall.update_state(y_segmentation_val, y_segmentation_pred, sample_weight = source_mask)
                
                # ===== [TARGET] PRECISION/RECALL =====
                self.precision_target.update_state(y_segmentation_val, y_segmentation_pred, sample_weight = target_mask)
                self.recall_target.update_state(y_segmentation_val, y_segmentation_pred, sample_weight = target_mask)
            
            # ===== DISCRIMINATOR =====
            self._persist_to_history(total_loss_discriminator, lambda x: x / self.num_batches_val, self.loss_discriminator_val_history)
            self._persist_to_history(self.acc_function_discriminator, lambda x: float(x.result()), self.acc_discriminator_val_history)

            # ===== [SOURCE] LOSS =====
            self._persist_to_history(total_loss_segmentation_source, lambda x: x / self.num_batches_val, self.loss_segmentation_val_history)

            # ===== [SOURCE] ACCURACY =====
            self._persist_to_history(self.acc_segmentation, lambda x: float(x.result()), self.acc_segmentation_val_history)

            # ===== [SOURCE] PRECISION =====
            self._persist_to_history(self.precision, lambda x: float(x.result()), self.precision_val_history)
            
            # ===== [SOURCE] RECALL =====
            self._persist_to_history(self.recall, lambda x: float(x.result()), self.recall_val_history)
            
            # ===== [SOURCE] F1 =====            
            a, b = (self.precision_val_history[-1], self.recall_val_history[-1])
            self._persist_to_history((a, b), lambda x: f1(*x), self.f1_val_history)

            # ===== [TARGET] LOSS =====
            self._persist_to_history(total_loss_segmentation_target, lambda x: x / self.num_batches_val, self.loss_segmentation_target_val_history)            
            
            # ===== [TARGET] ACCURACY =====
            self._persist_to_history(self.acc_segmentation_target, lambda x: float(x.result()), self.acc_segmentation_target_val_history)

            # ===== [TARGET] PRECISION =====
            self._persist_to_history(self.precision_target, lambda x: float(x.result()), self.precision_target_val_history)
            
            # ===== [TARGET] RECALL =====
            self._persist_to_history(self.recall_target, lambda x: float(x.result()), self.recall_target_val_history)
            
            # ===== [TARGET] F1 =====
            a, b = (self.precision_target_val_history[-1], self.recall_target_val_history[-1])
            self._persist_to_history((a, b), lambda x: f1(*x), self.f1_target_val_history) 

            # ===== [SOURCE] =====
            self.logger.write_scalar('val_writer', 'metric/loss/segmentation/source', self.loss_segmentation_val_history[-1], epoch + 1)
            self.logger.write_scalar('val_writer', 'metric/accuracy/segmentation/source', self.acc_segmentation_val_history[-1], epoch + 1)
            self.logger.write_scalar('val_writer', 'metric/precision/segmentation/source', self.precision_val_history[-1], epoch + 1)
            self.logger.write_scalar('val_writer', 'metric/recall/segmentation/source', self.recall_val_history[-1], epoch + 1)
            self.logger.write_scalar('val_writer', 'metric/f1/segmentation/source', self.f1_val_history[-1], epoch + 1)
            
            # ===== [TARGET] =====
            self.logger.write_scalar('val_writer', 'metric/loss/segmentation/target', self.loss_segmentation_target_val_history[-1], epoch + 1)
            self.logger.write_scalar('val_writer', 'metric/accuracy/segmentation/target', self.acc_segmentation_target_val_history[-1], epoch + 1)
            self.logger.write_scalar('val_writer', 'metric/precision/segmentation/target', self.precision_target_val_history[-1], epoch + 1)
            self.logger.write_scalar('val_writer', 'metric/recall/segmentation/target', self.recall_target_val_history[-1], epoch + 1)
            self.logger.write_scalar('val_writer', 'metric/f1/segmentation/target', self.f1_target_val_history[-1], epoch + 1) 
            
            # ===== DISCRIMINATOR =====
            self.logger.write_scalar('val_writer', 'metric/loss/discriminator', self.loss_discriminator_val_history[-1], epoch + 1)
            self.logger.write_scalar('val_writer', 'metric/accuracy/discriminator', self.acc_discriminator_val_history[-1], epoch + 1)

            # ===== [SOURCE] =====
            print(f'[SOURCE] Segmentation Loss: {self.loss_segmentation_val_history[-1]}')
            print(f'[SOURCE] Segmentation Accuracy: {self.acc_segmentation_val_history[-1]}')
            print(f'[SOURCE] Segmentation Precision: {self.precision_val_history[-1]}')
            print(f'[SOURCE] Segmentation Recall: {self.recall_val_history[-1]}')
            print(f'[SOURCE] Segmentation F1: {self.f1_val_history[-1]}')

            # ===== [TARGET] =====
            print(f'[TARGET] Segmentation Loss: {self.loss_segmentation_target_val_history[-1]}')
            print(f'[TARGET] Segmentation Accuracy: {self.acc_segmentation_target_val_history[-1]}')
            print(f'[TARGET] Segmentation Precision: {self.precision_target_val_history[-1]}')
            print(f'[TARGET] Segmentation Recall: {self.recall_target_val_history[-1]}')
            print(f'[TARGET] Segmentation F1: {self.f1_target_val_history[-1]}')
            
            # ===== DISCRIMINATOR =====
            print(f'Discriminator Loss: {self.loss_discriminator_val_history[-1]}')
            print(f'Discriminator Accuracy: {self.acc_discriminator_val_history[-1]}')
            
            clear_session()

            if self.persist_best_model and p >= self.progress_threshold:
                if self.loss_segmentation_val_history[-1] <= self.best_val_loss:
                    print('[!] Persisting best model...')
                    self.best_val_loss = self.loss_segmentation_val_history[-1]
                    self.no_improvement_count = 0
                    self.best_epoch = epoch
                    self.best_weights = self.model.get_weights()
                    
                    self.logger.write_scalar('val_writer', 'best/epoch', self.best_epoch, epoch + 1)
                    self.logger.write_scalar('val_writer', 'best/loss', self.best_val_loss, epoch + 1)

                elif self.loss_segmentation_val_history[-1] > self.best_val_loss or self.f1_target_val_history[-1] <= 0.01:
                    self.no_improvement_count += 1
                    if  self.no_improvement_count > self.wait:
                        print('[!] Performing early stopping.')
                        break

        self.elapsed_time = (time.time() - time_init) / 60

    def train(self, epochs: int = 25, wait: int = 12, persist_best_model: bool = True, progress_threshold: float = 0.):

        self.reset_history()
        time_init = time.time()

        self.epochs = epochs
        self.wait = wait if wait is not None else 1e8
        self.progress_threshold = progress_threshold
        self.persist_best_model = persist_best_model

        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1} of {self.epochs}')
            loss_global_train = 0.
            loss_global_val = 0.

            self.reset_states()

            np.random.shuffle(self.train_data_dirs)
            np.random.shuffle(self.val_data_dirs)

            # update learning rate
            p = epoch / (epochs - 1)
            lr = self.lr_function_segmentation.calculate(p)
            self.optimizer_segmentation.lr = lr
            self.lr_segmentation_history.append(lr)
            print(f'Learning Rate: {lr}')

            self.logger.write_scalar('segmentation_writer', 'learning_rate', lr, epoch + 1)

            print('Start training...')
            for batch in range(self.num_batches_train):

                print(f'Batch {batch + 1} of {self.num_batches_train}')
                batch_files = self.train_data_dirs[batch * self.batch_size : (batch + 1) * self.batch_size]

                # load images for training
                batch_images = np.asarray([load_array(batch_file, verbose = False) for batch_file in batch_files])
                batch_images = batch_images.astype(np.float32) # set np.float32 to reduce memory usage

                x_train = batch_images[ :, :, :, : self.channels]
                y_train = batch_images[ :, :, :, self.channels :]

                loss_train, y_pred = self._training_step(
                    model = self.model,
                    x_true = x_train,
                    y_true = y_train,
                    optimizer = self.optimizer_segmentation
                    )
                
                self.acc_segmentation.update_state(y_train, y_pred)
                self._calculate_precision_recall(y_train, y_pred)
                loss_global_train += float(loss_train)

            # ===== LOSS =====
            self._persist_to_history(loss_global_train, lambda x: x / self.num_batches_train, self.loss_segmentation_train_history)

            # ===== ACCURACY =====
            self._persist_to_history(self.acc_segmentation, lambda x: float(x.result()), self.acc_segmentation_train_history)
            
            # ===== PRECISION =====            
            self._persist_to_history(self.precision, lambda x: float(x.result()), self.precision_train_history)
            
            # ===== RECALL =====
            self._persist_to_history(self.recall, lambda x: float(x.result()), self.recall_train_history)
            
            # ===== F1 =====
            a, b = (self.precision_train_history[-1], self.recall_train_history[-1])
            self._persist_to_history((a, b), lambda x: f1(*x), self.f1_train_history)

            self.logger.write_scalar('train_writer', 'metric/loss', self.loss_segmentation_train_history[-1], epoch + 1)
            self.logger.write_scalar('train_writer', 'metric/accuracy', self.acc_segmentation_train_history[-1], epoch + 1)
            self.logger.write_scalar('train_writer', 'metric/precision', self.precision_train_history[-1], epoch + 1)
            self.logger.write_scalar('train_writer', 'metric/recall', self.recall_train_history[-1], epoch + 1)
            self.logger.write_scalar('train_writer', 'metric/f1', self.f1_train_history[-1], epoch + 1)

            print(f'Training Loss: {self.loss_segmentation_train_history[-1]}')
            print(f'Training Accuracy: {self.acc_segmentation_train_history[-1]}')
            print(f'Training Precision: {self.precision_train_history[-1]}')
            print(f'Training Recall: {self.recall_train_history[-1]}')
            print(f'Training F1: {self.f1_train_history[-1]}')

            self.reset_states()

            # evaluating network
            print('Start validation...')
            for batch in range(self.num_batches_val):
                print(f'Batch {batch + 1} of {self.num_batches_val}')
                batch_val_files = self.val_data_dirs[batch * self.batch_size : (batch + 1) * self.batch_size]

                # load images for testing
                batch_val_images = np.asarray([load_array(batch_val_file, verbose = False) for batch_val_file in batch_val_files])
                batch_val_images = batch_val_images.astype(np.float32) # set np.float32 to reduce memory usage

                x_val = batch_val_images[:, :, :, : self.channels]
                y_val = batch_val_images[:, :, :, self.channels :]

                y_pred = self.model(x_val)

                loss_val = self.loss_function(y_val, y_pred)
                loss_global_val += float(loss_val) # convert loss_val to float and sum
                
                self.acc_segmentation.update_state(y_val, y_pred)
                self._calculate_precision_recall(y_val, y_pred)

            # ===== LOSS =====
            self._persist_to_history(loss_global_val, lambda x: x / self.num_batches_val, self.loss_segmentation_val_history)

            # ===== ACCURACY =====
            self._persist_to_history(self.acc_segmentation, lambda x: float(x.result()), self.acc_segmentation_val_history)
            
            # ===== PRECISION =====
            self._persist_to_history(self.precision, lambda x: float(x.result()), self.precision_val_history)

            # ===== RECALL =====
            self._persist_to_history(self.recall, lambda x: float(x.result()), self.recall_val_history)
            
            # ===== F1 =====
            a, b = (self.precision_val_history[-1], self.recall_val_history[-1])
            self._persist_to_history((a, b), lambda x: f1(*x), self.f1_val_history)

            self.logger.write_scalar('val_writer', 'metric/loss', self.loss_segmentation_val_history[-1], epoch + 1)
            self.logger.write_scalar('val_writer', 'metric/accuracy', self.acc_segmentation_val_history[-1], epoch + 1)
            self.logger.write_scalar('val_writer', 'metric/precision', self.precision_val_history[-1], epoch + 1)
            self.logger.write_scalar('val_writer', 'metric/recall', self.recall_val_history[-1], epoch + 1)
            self.logger.write_scalar('val_writer', 'metric/f1', self.f1_val_history[-1], epoch + 1)

            print(f'Validation Loss: {self.loss_segmentation_val_history[-1]}')
            print(f'Validation Accuracy: {self.acc_segmentation_val_history[-1]}')
            print(f'Validation Precision: {self.precision_val_history[-1]}')
            print(f'Validation Recall: {self.recall_val_history[-1]}')
            print(f'Validation F1: {self.f1_val_history[-1]}')
            
            clear_session()

            if self.persist_best_model and p >= self.progress_threshold:
                if self.loss_segmentation_val_history[-1] <= self.best_val_loss:
                    print('[!] Persisting best model...')
                    self.best_val_loss = self.loss_segmentation_val_history[-1]
                    self.no_improvement_count = 0
                    self.best_epoch = epoch
                    self.best_weights = self.model.get_weights()

                else:
                    self.no_improvement_count += 1
                    if  self.no_improvement_count > self.wait:
                        print('[!] Performing early stopping.')
                        break

        self.elapsed_time = (time.time() - time_init) / 60

    def save_weights(self, weights_path: str, best: bool = True, piece: str = None):
        if best:
            model_to_save = self.assembly_empty_model()
            model_to_save.set_weights(self.best_weights)
        else:
            model_to_save = self.model

        if piece is None:
            model_to_save.save_weights(weights_path) # save weights

        elif piece == 'segmentation':
            model_to_save.main_network.save_weights(weights_path)
        
        elif piece == 'discriminator':
            model_to_save.domain_discriminator.save_weights(weights_path)

        print('Weights saved successfuly.')

    def load_weights(self, weights_path: str, piece: str = None):
        if piece is None:
            self.model.load_weights(weights_path)
        
        elif piece == 'segmentation':
            self.model.main_network.load_weights(weights_path)

        elif piece == 'discriminator':
            self.model.domain_discriminator.load_weights(weights_path)

        print('Weights loaded successfuly.')

    def save_model(self, model_path: str, best: bool = True): # save model
        if best:
            model_to_save = self.assembly_empty_model()
            model_to_save.set_weights(self.best_weights)
            save_model(model_to_save, model_path) 
        else:
            save_model(self.model, model_path)
        print('Model saved successfuly.')
        
    def save_info(self, history_path: str):
        persist = self.parameters
        save_json(persist, history_path) # save metrics and parameters
        print('Metrics saved successfuly.')

    def set_test_index(self, test_index_source: list, test_index_target: list):
        test_index_source = [int(i) for i in test_index_source]
        test_index_target = [int(i) for i in test_index_target]

        self.test_index_source = test_index_source
        self.test_index_target = test_index_target

    def set_optimizer(self, optimizer_config: dict):
        name = optimizer_config.pop('name')
        if name == 'adam':
            self.optimizer_segmentation = Adam(**optimizer_config)
            self.optimizer_discriminator = Adam(**optimizer_config)
        
        elif name == 'sgd':
            self.optimizer_segmentation = SGD(**optimizer_config)
            self.optimizer_discriminator = SGD(**optimizer_config)

    def set_learning_rate(self, **kwargs):
        segmentation_params = kwargs.get('segmentation')
        if segmentation_params is not None:
            name = segmentation_params.pop('name')
            self.lr_function_segmentation = LearningRateFactory(name, **segmentation_params)

        discriminator_params = kwargs.get('discriminator')
        if discriminator_params is not None:
            name = discriminator_params.pop('name')
            self.lr_function_discriminator = LearningRateFactory(name, **discriminator_params)

    def set_lambda(self, **kwargs):
        self.lambda_function = LambdaGradientReversalLayer(**kwargs)

    def _split_original_augmented(self, data_dirs: list):
        original = [directory for directory in data_dirs if 'flip' not in directory and 'rotation' not in directory]
        augmented = list(set(data_dirs) - set(original))
        return original, augmented

    @property
    def parameters(self):
        training_total_original, training_total_augmented = self._split_original_augmented(self.train_data_dirs)
        training_source_original, training_source_augmented = self._split_original_augmented(self.train_data_dirs_source)
        training_target_original, training_target_augmented = self._split_original_augmented(self.train_data_dirs_target)

        validation_total_original, validation_total_augmented = self._split_original_augmented(self.val_data_dirs)
        validation_source_original, validation_source_augmented = self._split_original_augmented(self.val_data_dirs_source)
        validation_target_original, validation_target_augmented = self._split_original_augmented(self.val_data_dirs_target)

        persist = {
                'history':{
                    'training':{
                        'loss':{
                            'segmentation': {
                                'source': self.loss_segmentation_train_history,
                                'target': self.loss_segmentation_target_train_history},
                            'discriminator': self.loss_discriminator_train_history},
                        'accuracy':{
                            'segmentation': {
                                'source': self.acc_segmentation_train_history,
                                'target': self.acc_segmentation_target_train_history},
                            'discriminator': self.acc_discriminator_train_history},
                        'precision':{
                            'segmentation': {
                                'source': self.precision_train_history,
                                'target': self.precision_target_train_history}},
                        'recall':{
                            'segmentation': {
                                'source': self.recall_train_history,
                                'target': self.recall_target_train_history}},
                        'f1':{
                            'segmentation': {
                                'source': self.f1_train_history,
                                'target': self.f1_target_train_history}}},
                    'validation':{
                        'loss':{
                            'segmentation': {
                                'source': self.loss_segmentation_val_history,
                                'target': self.loss_segmentation_target_val_history},
                            'discriminator': self.loss_discriminator_val_history},
                        'accuracy':{
                            'segmentation': {
                                'source': self.acc_segmentation_val_history,
                                'target': self.acc_segmentation_target_val_history},
                            'discriminator': self.acc_discriminator_val_history},
                        'precision':{
                            'segmentation': {
                                'source': self.precision_val_history,
                                'target': self.precision_target_val_history}},
                        'recall':{
                            'segmentation': {
                                'source': self.recall_val_history,
                                'target': self.recall_target_val_history}},
                        'f1':{
                            'segmentation': {
                                'source': self.f1_val_history,
                                'target': self.f1_target_val_history}}}},
                'image_files':{
                    'training':{
                        'total':{
                            'original': training_total_original,
                            'augmented': training_total_augmented},
                        'source':{
                            'original': training_source_original,
                            'augmented': training_source_augmented},
                           'target':{
                            'original': training_target_original,
                            'augmented': training_target_augmented}},
                    'validation':{
                        'total':{
                            'original': validation_total_original,
                            'augmented': validation_total_augmented},
                        'source':{
                            'original': validation_source_original,
                            'augmented': validation_source_augmented},
                        'target':{
                            'original': validation_target_original,
                            'augmented': validation_target_augmented}}},
                'is_domain_adaptation': self.domain_adaptation,
                'lr_segmentation': self.lr_segmentation_history,
                'lr_discriminator': self.lr_discriminator_history,
                'time': self.elapsed_time,
                'lambdas': self.lambdas,
                'function_config':{
                    'lr_segmentation': self.lr_function_segmentation.config,
                    'lr_discriminator': self.lr_function_discriminator.config,
                    'lambda': self.lambda_function.config},
                'patch_size': self.patch_size,
                'output_stride': self.output_stride,
                'skip_conn': self.skip_conn,
                'val_fraction': self.val_fraction,
                'batch_size': self.batch_size,
                'channels': self.channels,
                'num_images': self.num_images,
                'epochs': self.epochs,
                'wait': self.wait,
                'rotate': self.rotate,
                'flip': self.flip,
                'test_index_source': self.test_index_source,
                'test_index_target': self.test_index_target,
                'persist_best': self.persist_best_model,
                'progress_threshold': self.progress_threshold,
                'best_epoch': self.best_epoch}

        return persist

    @property
    def time(self):
        return self.elapsed_time