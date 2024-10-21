import os

PROJECT_ROOT = os.path.dirname(__file__)

DATASET_ROOT = os.path.join(PROJECT_ROOT, 'dataset')
os.makedirs(DATASET_ROOT, exist_ok=True)

TESTING_DATASET_PATH = os.path.join(DATASET_ROOT, 'testing')
os.makedirs(TESTING_DATASET_PATH, exist_ok=True)
TRAINING_DATASET_PATH = os.path.join(DATASET_ROOT, 'training')
os.makedirs(TRAINING_DATASET_PATH, exist_ok=True)

MODEL_STATE_DICT_PATH = os.path.join(PROJECT_ROOT, 'models', 'model.pt')
os.makedirs(os.path.dirname(MODEL_STATE_DICT_PATH), exist_ok=True)

INSTANCE_DIMENSIONS_COUNT = 28 * 28
OUTPUT_DIMENSIONS_COUNT = 10

LEARNING_RATE = 0.005
EPOCHS = 1000
