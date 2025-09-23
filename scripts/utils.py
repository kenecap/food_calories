# scripts/utils.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# --- ИЗМЕНЕНИЕ 1: Импортируем планировщик ---
from torch.optim.lr_scheduler import StepLR
# -------------------------------------------
import random
import numpy as np
from tqdm import tqdm

from scripts.dataset import FoodDataset, collate_fn
from scripts.model import CaloriePredictor

def set_seed(seed):
    # ... функция set_seed без изменений ...
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_model(config):
    set_seed(config.SEED)
    print(f"Обучение будет на устройстве: {config.DEVICE}")

    # --- Подготовка данных (без изменений) ---
    train_dataset = FoodDataset(data_dir=config.DATA_DIR, split='train')
    test_dataset = FoodDataset(data_dir=config.DATA_DIR, split='test')
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # --- Инициализация модели (без изменений) ---
    model = CaloriePredictor(num_ingredients=train_dataset.num_ingredients, embedding_dim=config.EMBEDDING_DIM).to(config.DEVICE)

    # --- Настройка обучения ---
    loss_fn = nn.MSELoss()
    # Возвращаем простой оптимизатор без weight_decay
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # --- ИЗМЕНЕНИЕ 2: Создаем планировщик ---
    # Каждые 7 эпох (step_size=7) он будет умножать скорость обучения на 0.1 (gamma=0.1)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    # ---------------------------------------

    best_mae = float('inf')

    for epoch in range(config.EPOCHS):
        print(f"\n--- Эпоха {epoch + 1} / {config.EPOCHS} ---")
        
        model.train()
        # ... цикл обучения без изменений ...
        train_mae = 0
        for batch in tqdm(train_loader, desc="Обучение"):
            # ... (forward, loss, backward, step) ...
            images, ingredients, mass, calories = (batch['image'].to(config.DEVICE), batch['ingredients'].to(config.DEVICE), batch['mass'].to(config.DEVICE), batch['calories'].to(config.DEVICE))
            predictions = model(images, ingredients, mass)
            loss = loss_fn(predictions, calories)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_mae += torch.abs(predictions - calories).mean().item()
            
        print(f"Средняя MAE на обучении: {train_mae / len(train_loader):.2f}")

        # ... цикл оценки без изменений ...
        model.eval()
        val_mae = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Оценка"):
                images, ingredients, mass, calories = (batch['image'].to(config.DEVICE), batch['ingredients'].to(config.DEVICE), batch['mass'].to(config.DEVICE), batch['calories'].to(config.DEVICE))
                predictions = model(images, ingredients, mass)
                val_mae += torch.abs(predictions - calories).mean().item()
        
        current_mae = val_mae / len(test_loader)
        print(f"MAE на тесте: {current_mae:.2f}")

        if current_mae < best_mae:
            best_mae = current_mae
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"✨ Новая лучшая модель сохранена! MAE = {best_mae:.2f}")

        # --- ИЗМЕНЕНИЕ 3: Делаем шаг планировщика в конце эпохи ---
        scheduler.step()
        # -----------------------------------------------------------

    print("\nОбучение завершено!")
    print(f"Лучший результат MAE на тесте: {best_mae:.2f}")