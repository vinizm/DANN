import numpy as np
from utils.hyperparameters import *


DOMAIN_ADAPTATION_CONFIG = [
    # Fe19
    {
        'source': 'Fe19',
        'target': 'Fe120',
        'num_runs': 5,
        'run_training': False
    },
    {
        'source': 'Fe19',
        'target': 'FeM',
        'num_runs': 1,
        'run_training': False
    },
    {
        'source': 'Fe19',
        'target': 'Cu',
        'num_runs': 5,
        'run_training': False
    },
    # Fe120
    {
        'source': 'Fe120',
        'target': 'Fe19',
        'num_runs': 5,
        'run_training': False
    },
    {
        'source': 'Fe120',
        'target': 'FeM',
        'num_runs': 5,
        'run_training': False
    },
    {
        'source': 'Fe120',
        'target': 'Cu',
        'num_runs': 5,
        'run_training': False
    },
    # FeM
    {
        'source': 'FeM',
        'target': 'Fe19',
        'num_runs': 5,
        'run_training': False
    },
    {
        'source': 'FeM',
        'target': 'Fe120',
        'num_runs': 5,
        'run_training': False
    },
    {
        'source': 'FeM',
        'target': 'Cu',
        'num_runs': 5,
        'run_training': False
    },
    # Cu
    {
        'source': 'Cu',
        'target': 'Fe19',
        'num_runs': 5,
        'run_training': False
    },
    {
        'source': 'Cu',
        'target': 'Fe120',
        'num_runs': 5,
        'run_training': False
    },
    {
        'source': 'Cu',
        'target': 'FeM',
        'num_runs': 1,
        'run_training': True
    }
]

LR_CONSTANT_CONFIG = {
    'name': 'constant',
    'const': LR0,
    'warmup': LR_WARMUP
}

LR_EXP_CONFIG = {
    'name': 'exp',
    'lr0': 0.01,
    'alpha': 10.,
    'beta': 0.75,
    'warmup': 0.
}


LR_STEP_CONFIG = {
    'name': 'step',
    'lr0': LR0,
    'step_decay': STEP_DECAY,
    'num_steps': NUM_STEPS,
    'warmup': LR_WARMUP
}

LR_LINEAR_CONFIG = {
    'name': 'linear',
    'start': LR_START_LINEAR,
    'stop': LR_STOP_LINEAR,
    'lr_warmup': LR_WARMUP
}

LR_LOG_CONFIG = {
    'name': 'log',
    'start': np.log10(5.e-4),
    'stop': np.log10(1.e-5),
    'warmup': 0.
}

ADAM_OPTIMIZER_CONFIG = {
    'name': 'adam'
}

SGD_OPTIMIZER_CONFIG = {
    'name': 'sgd',
    'momentum': 0.9
}

DOMAIN_ADAPTATION_GLOBAL_PARAMS = {'patch_size': 256, 'channels': 3, 'num_class': 2, 'max_epochs': 100, 'batch_size': 4, 'val_fraction': 0.1, 'num_images_train': 1800,
                                   'patience': None, 'flip': True, 'rotate': True, 'progress_threshold': 0., 'num_runs': 5, 'output_stride': 16, 'lambda_scale': 1.e-3,
                                   'gamma': 10., 'lambda_warmup': 0., 'skip_conn': True, 'units': 1024, 'optimizer_config': ADAM_OPTIMIZER_CONFIG, 'lr_config': LR_LOG_CONFIG}