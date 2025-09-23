# scripts/utils.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import numpy as np
from tqdm import tqdm # Для красивого progress bar

# Импортируем наши собственные классы
from scripts.dataset import FoodDataset, collate_fn
from scripts.model import CaloriePredictor

def set_seed(seed):
    """Фиксирует случайность для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def train_model(config):
    """Главная функция, запускающая весь пайплайн обучения."""
    
    set_seed(config.SEED)
    print(f"Обучение будет на устройстве: {config.DEVICE}")

    # --- 1. Подготовка данных ---
    train_dataset = FoodDataset(data_dir=config.DATA_DIR, split='train')
    test_dataset = FoodDataset(data_dir=config.DATA_DIR, split='test')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    # --- 2. Инициализация модели ---
    model = CaloriePredictor(
        num_ingredients=train_dataset.num_ingredients,
        embedding_dim=config.EMBEDDING_DIM
    ).to(config.DEVICE)

    # --- 3. Настройка обучения ---
    # Функция потерь: мы будем минимизировать Среднеквадратичную ошибку (MSE)
    loss_fn = nn.MSELoss()
    # Оптимизатор: будет обновлять веса модели. Adam - популярный и надежный выбор.
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    best_mae = float('inf')

    # --- 4. Цикл обучения ---
    for epoch in range(config.EPOCHS):
        print(f"\n--- Эпоха {epoch + 1} / {config.EPOCHS} ---")
        
        # Обучение на одной эпохе
        model.train()
        train_mae = 0
        for batch in tqdm(train_loader, desc="Обучение"):
            # Переносим данные на GPU/CPU
            images = batch['image'].to(config.DEVICE)
            ingredients = batch['ingredients'].to(config.DEVICE)
            mass = batch['mass'].to(config.DEVICE)
            calories = batch['calories'].to(config.DEVICE)
            
            # Предсказание модели
            predictions = model(images, ingredients, mass)
            
            # Расчет ошибки
            loss = loss_fn(predictions, calories)
            
            # Шаги для обновления весов
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Считаем MAE для вывода
            train_mae += torch.abs(predictions - calories).mean().item()
            
        print(f"Средняя MAE на обучении: {train_mae / len(train_loader):.2f}")

        # Оценка на тестовых данных
        model.eval()
        val_mae = 0
        with torch.no_grad(): # Отключаем расчет градиентов
            for batch in tqdm(test_loader, desc="Оценка"):
                images = batch['image'].to(config.DEVICE)
                ingredients = batch['ingredients'].to(config.DEVICE)
                mass = batch['mass'].to(config.DEVICE)
                calories = batch['calories'].to(config.DEVICE)
                
                predictions = model(images, ingredients, mass)
                val_mae += torch.abs(predictions - calories).mean().item()
        
        current_mae = val_mae / len(test_loader)
        print(f"MAE на тесте: {current_mae:.2f}")

        # Сохраняем лучшую модель
        if current_mae < best_mae:
            best_mae = current_mae
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"✨ Новая лучшая модель сохранена! MAE = {best_mae:.2f}")

    print("\nОбучение завершено!")
    print(f"Лучший результат MAE на тесте: {best_mae:.2f}")