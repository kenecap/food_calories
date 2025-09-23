# config.py

import torch

DATA_DIR = './data'
MODEL_SAVE_PATH = './best_calorie_predictor_resnet50.pth' # Новое имя файла!

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
# Дадим более сложной модели 15 эпох, чтобы она успела адаптироваться
EPOCHS = 15
# Используем LR от нашего лучшего эксперимента
LEARNING_RATE = 0.001

EMBEDDING_DIM = 128
SEED = 42