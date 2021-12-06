import platform

system = platform.system()

BASE_FOLDER = '/content' if system == 'Linux' else 'C:/Users/viniciusmartins/Documents/CCOMP/Projeto'
PROCESSED_FOLDER = f'{BASE_FOLDER}/DANN/processed_images'
MODELS_FOLDER = f'{BASE_FOLDER}/DANN/results'

PATH_TO_FE19_DATASET_RLM = f'{BASE_FOLDER}/Fe19/Reflected_Light_Microscopy'
PATH_TO_FE120_DATASET_RLM = f'{BASE_FOLDER}/Fe120/Reflected_Light_Microscopy'
PATH_TO_FEM_DATASET_RLM = f'{BASE_FOLDER}/FeM/Reflected_Light_Microscopy'
PATH_TO_CU_DATASET_RLM = f'{BASE_FOLDER}/Cu/Reflected_Light_Microscopy'

PATH_TO_FE19_DATASET_MASK = f'{BASE_FOLDER}/Fe19/Reference_2'
PATH_TO_FE120_DATASET_MASK = f'{BASE_FOLDER}/Fe120/Reference'
PATH_TO_FEM_DATASET_MASK = f'{BASE_FOLDER}/FeM/Reference'
PATH_TO_CU_DATASET_MASK = f'{BASE_FOLDER}/Cu/Reference'