# scripts/utils.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import random
import numpy as np
from tqdm import tqdm

from scripts.dataset import FoodDataset, collate_fn
from scripts.model import CaloriePredictor

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_model(config):
    set_seed(config.SEED)
    print(f"Обучение будет на устройстве: {config.DEVICE}")

    # --- Подготовка данных ---
    train_dataset = FoodDataset(config, split='train')
    test_dataset = FoodDataset(config, split='test')
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)

    # --- Инициализация модели ---
    model = CaloriePredictor(config, num_ingredients=train_dataset.num_ingredients).to(config.DEVICE)

    # --- Настройка Fine-tuning и Оптимизатора ---
    # 1. Замораживаем всю image_model
    for param in model.image_model.parameters():
        param.requires_grad = False
        
    # 2. Размораживаем последние несколько блоков EfficientNet
    # Для efficientnet_b0 это блоки 5 и 6, а также head
    for param in model.image_model.blocks[5].parameters():
        param.requires_grad = True
    for param in model.image_model.blocks[6].parameters():
        param.requires_grad = True
    for param in model.image_model.conv_head.parameters():
        param.requires_grad = True
    for param in model.image_model.bn2.parameters():
        param.requires_grad = True
    
    # 3. Создаем группы параметров с разными LR
    image_params = [p for p in model.image_model.parameters() if p.requires_grad]
    head_params = list(model.ingredient_embedding.parameters()) + \
                  list(model.text_branch.parameters()) + \
                  list(model.fusion_head.parameters())
                  
    optimizer = torch.optim.AdamW([
        {'params': image_params, 'lr': config.IMAGE_LR},
        {'params': head_params, 'lr': config.HEAD_LR}
    ])
    
    loss_fn = nn.MSELoss()
    scheduler = StepLR(optimizer, step_size=8, gamma=0.1)
    
    best_mae = float('inf')

    # --- Цикл обучения (остается почти без изменений) ---
    for epoch in range(config.EPOCHS):
        print(f"\n--- Эпоха {epoch + 1} / {config.EPOCHS} ---")
        
        model.train()
        train_mae = 0
        for batch in tqdm(train_loader, desc="Обучение"):
            images, ingredients, mass, calories = (batch['image'].to(config.DEVICE), batch['ingredients'].to(config.DEVICE), batch['mass'].to(config.DEVICE), batch['calories'].to(config.DEVICE))
            predictions = model(images, ingredients, mass)
            loss = loss_fn(predictions, calories)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_mae += torch.abs(predictions - calories).mean().item()
            
        print(f"Средняя MAE на обучении: {train_mae / len(train_loader):.2f}")

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

        scheduler.step()

    print("\nОбучение завершено!")
    print(f"Лучший результат MAE на тесте: {best_mae:.2f}")