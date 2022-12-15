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
        'num_runs': 1,
        'run_training': True
    },
    {
        'source': 'Cu',
        'target': 'FeM',
        'num_runs': 5,
        'run_training': False
    }
]


DOMAIN_ADAPTATION_GLOBAL_PARAMS = {'patch_size': 256, 'channels': 1, 'num_class': 2, 'max_epochs': 1000, 'batch_size': 4, 'val_fraction': 0.1, 'num_images_train': 200,
                                   'patience': None, 'flip': True, 'rotate': True, 'progress_threshold': 0., 'num_runs': 5, 'alpha': 2.25, 'beta': 0.75, 'lr0': 5.e-4,
                                   'lr_warmup': 0.01, 'lr_name': 'log', 'output_stride': 16, 'lr_log_start': np.log10(2.e-4), 'lr_log_stop': np.log10(1.e-5),
                                   'lambda_scale': 1., 'gamma': 10., 'lambda_warmup': 0.01}