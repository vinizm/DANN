import numpy as np


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
        'num_runs': 5,
        'run_training': True
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
        'num_runs': 5,
        'run_training': False
    }
]


DOMAIN_ADAPTATION_GLOBAL_PARAMS = {'patch_size': 256, 'channels': 1, 'num_class': 2, 'max_epochs': 120, 'batch_size': 4, 'val_fraction': 0.1, 'num_images_train': 200,
                                   'patience': 100, 'flip': True, 'rotate': True, 'progress_threshold': 0., 'num_runs': 5, 'alpha': 2.25, 'beta': 0.75, 'lr0': 5.e-4,
                                   'lr_warmup': 0., 'lr_name': 'log', 'output_stride': 8, 'lr_log_start': np.log10(5.e-4), 'lr_log_stop': np.log10(5.e-5)}