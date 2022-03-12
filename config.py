import platform

system = platform.system()

BASE_FOLDER = '/content' if system == 'Linux' else 'C:/Users/viniciusmartins/Documents/CCOMP/Projeto'
PROCESSED_FOLDER = f'{BASE_FOLDER}/DANN/processed_images'
MODELS_FOLDER = f'{BASE_FOLDER}/DANN/results'

PATH_TO_FE19_DATASET_RLM = f'{BASE_FOLDER}/Fe19/Reflected_Light_Microscopy'
PATH_TO_FE120_DATASET_RLM = f'{BASE_FOLDER}/Fe120/Reflected_Light_Microscopy'
PATH_TO_FEM_DATASET_RLM = f'{BASE_FOLDER}/FeM/Reflected_Light_Microscopy'
PATH_TO_CU_DATASET_RLM = f'{BASE_FOLDER}/Cu/Reflected_Light_Microscopy'

PATH_TO_FE19_DATASET_MASK = f'{BASE_FOLDER}/Fe19/Reference'
PATH_TO_FE120_DATASET_MASK = f'{BASE_FOLDER}/Fe120/Reference'
PATH_TO_FEM_DATASET_MASK = f'{BASE_FOLDER}/FeM/Reference'
PATH_TO_CU_DATASET_MASK = f'{BASE_FOLDER}/Cu/Reference'

PATH_TO_FE19_DATASET_MAP_D1 = f'{BASE_FOLDER}/Fe19/map_d1'
PATH_TO_FE120_DATASET_MAP_D1 = f'{BASE_FOLDER}/Fe120/map_d1'
PATH_TO_FEM_DATASET_MAP_D1 = f'{BASE_FOLDER}/FeM/map_d1'
PATH_TO_CU_DATASET_MAP_D1 = f'{BASE_FOLDER}/Cu/map_d1'

PATH_TO_FOLDER = {
    'Fe19':{
        'RLM': PATH_TO_FE19_DATASET_RLM,
        'MASK': PATH_TO_FE19_DATASET_MASK,
        'MAP_D1': PATH_TO_FE19_DATASET_MAP_D1},
    'Fe120':{
		'RLM': PATH_TO_FE120_DATASET_RLM,
		'MASK': PATH_TO_FE120_DATASET_MASK,
		'MAP_D1': PATH_TO_FE120_DATASET_MAP_D1},
    'FeM':{
        'RLM': PATH_TO_FEM_DATASET_RLM,
        'MASK': PATH_TO_FEM_DATASET_MASK,
		'MAP_D1': PATH_TO_FEM_DATASET_MAP_D1},
    'Cu':{
        'RLM': PATH_TO_CU_DATASET_RLM,
        'MASK': PATH_TO_CU_DATASET_MASK,
		'MAP_D1': PATH_TO_CU_DATASET_MAP_D1}}

NUM_IMAGES = {
    'Fe19': 19,
    'Fe120':120,
    'FeM':81,
    'Cu':121}


LR0 = 1e-2
ALPHA = 10.
BETA = 0.75
GAMMA = 10.
