NO_DOMAIN_ADAPTATION_CONFIG = [
    {
        'dataset': 'Fe19',
        'num_runs': 7,
        'run_training': True
    },
    {
        'dataset': 'Fe120',
        'num_runs': 7,
        'run_training': False
    },
    {
        'dataset': 'FeM',
        'num_runs': 7,
        'run_training': False
    },
    {
        'dataset': 'Cu',
        'num_runs': 7,
        'run_training': False
    }
]


NO_DOMAIN_ADAPTATION_GLOBAL_PARAMS = {'patch_size': 256, 'channels': 1, 'num_class': 2, 'output_stride': 16, 'max_epochs': 150, 'batch_size': 2, 'val_fraction': 0.1,
                                      'num_images_train': 200, 'patience': 20, 'flip': True, 'rotate': True, 'progress_threshold': 0., 'num_runs': 10, 'alpha': 2.25,
                                      'beta': 0.75, 'lr0': 5.e-4, 'lr_warmup': 0., 'lr_name': 'exp_decay', 'backbone_size': 16}