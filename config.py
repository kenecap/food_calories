# config.py

import torch

# --- Основные пути ---
DATA_DIR = './data'
MODEL_SAVE_PATH = './best_model_efficientnet_finetuned.pth'

# --- Модели ---
IMAGE_MODEL_NAME = 'tf_efficientnet_b0'

# --- Параметры обучения ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 25

# --- Гибкие скорости обучения (Learning Rates) ---
# Для "размороженных" слоев EfficientNet
IMAGE_LR = 1e-4
# Для всех новых слоев ("голова" модели)
HEAD_LR = 1e-3

# --- Настройки модели ---
EMBEDDING_DIM = 128

# --- Воспроизводимость ---
SEED = 42