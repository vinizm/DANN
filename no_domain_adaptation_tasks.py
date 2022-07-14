from experiments.model_config.no_domain_adaptation import NO_DOMAIN_ADAPTATION_CONFIG, NO_DOMAIN_ADAPTATION_GLOBAL_PARAMS
from custom_trainer import Trainer


for CASE in NO_DOMAIN_ADAPTATION_CONFIG:
    
    run_training = CASE.get('run_training')
    if not run_training:
        continue  
    
    dataset = CASE.get('dataset')
    patch_size = CASE.get('patch_size', NO_DOMAIN_ADAPTATION_GLOBAL_PARAMS.get('patch_size'))
    
    