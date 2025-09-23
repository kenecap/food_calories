# scripts/model.py

import torch
import torch.nn as nn
import torchvision.models as models

class CaloriePredictor(nn.Module):
    """Мультимодальная модель для предсказания калорий."""
    
    def __init__(self, num_ingredients, embedding_dim=128):
        super().__init__()

        # --- Ветвь для обработки изображений ---
        # Берем готовую модель ResNet18, обученную на миллионах картинок
        self.image_branch = models.resnet18(pretrained=True)
        # Заменяем ее последний слой на свой, который будет выдавать вектор признаков
        num_features = self.image_branch.fc.in_features
        self.image_branch.fc = nn.Linear(num_features, 256)

        # --- Ветвь для обработки ингредиентов и массы ---
        # Слой Embedding превращает индекс ингредиента в вектор
        self.ingredient_embedding = nn.Embedding(
            num_embeddings=num_ingredients, 
            embedding_dim=embedding_dim, 
            padding_idx=0
        )
        
        # Простая нейросеть для обработки текстовых данных
        self.text_branch = nn.Sequential(
            nn.Linear(embedding_dim + 1, 128), # Вход: вектор ингредиентов + масса (1 число)
            nn.ReLU(),
            nn.Linear(128, 256)
        )

        # --- "Голова", которая объединяет все и делает предсказание ---
        self.fusion_head = nn.Sequential(
            nn.Linear(256 + 256, 128), # Вход: признаки от картинки + признаки от текста
            nn.ReLU(),
            nn.Dropout(0.5), # Слой для борьбы с переобучением
            nn.Linear(128, 1) # Выход: одно число (калорийность)
        )

    def forward(self, image, ingredients, mass):
        """Определяет, как данные проходят через модель."""
        # 1. Прогоняем картинки
        image_features = self.image_branch(image)

        # 2. Прогоняем ингредиенты
        embedded_ingredients = self.ingredient_embedding(ingredients)
        # Усредняем векторы всех ингредиентов в блюде
        mean_ingredients = embedded_ingredients.mean(dim=1)
        
        # Объединяем вектор ингредиентов с массой
        text_input = torch.cat([mean_ingredients, mass], dim=1)
        text_features = self.text_branch(text_input)

        # 3. Объединяем признаки из обеих ветвей
        combined_features = torch.cat([image_features, text_features], dim=1)
        
        # 4. Делаем финальное предсказание
        prediction = self.fusion_head(combined_features)

        return prediction