import os

CURRENT_PATH = os.path.realpath(__file__)
BASE_FOLDER = '/'.join(CURRENT_PATH.split('/')[: -2])

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

PATH_TO_FOLDER = {
    'Fe19':{
        'RLM': PATH_TO_FE19_DATASET_RLM,
        'MASK': PATH_TO_FE19_DATASET_MASK},
    'Fe120':{
		'RLM': PATH_TO_FE120_DATASET_RLM,
		'MASK': PATH_TO_FE120_DATASET_MASK},
    'FeM':{
        'RLM': PATH_TO_FEM_DATASET_RLM,
        'MASK': PATH_TO_FEM_DATASET_MASK},
    'Cu':{
        'RLM': PATH_TO_CU_DATASET_RLM,
        'MASK': PATH_TO_CU_DATASET_MASK}}

NUM_IMAGES = {
        'Fe19': 19,
        'Fe120': 120,
        'FeM': 81,
        'Cu': 121}

TEST_INDEX = {
        'Fe19': [0, 14, 5, 17],
        'Fe120': [108, 45, 32, 117],
        'FeM': [73, 72, 75, 63],
        'Cu': [119, 46, 65, 115]}