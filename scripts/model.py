# scripts/model.py

import torch
import torch.nn as nn
import timm

class CaloriePredictor(nn.Module):
    def __init__(self, config, num_ingredients):
        super().__init__()

        # --- ИСПОЛЬЗУЕМ TIMM ДЛЯ СОЗДАНИЯ EFFICIENTNET ---
        # num_classes=0 означает, что мы хотим получить модель без классификационной "головы"
        self.image_model = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0  
        )
        
        # Получаем количество признаков на выходе из EfficientNet
        num_image_features = self.image_model.num_features

        # --- Ветвь для ингредиентов и массы (без изменений) ---
        self.ingredient_embedding = nn.Embedding(
            num_embeddings=num_ingredients, 
            embedding_dim=config.EMBEDDING_DIM, 
            padding_idx=0
        )
        self.text_branch = nn.Sequential(
            nn.Linear(config.EMBEDDING_DIM + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )

        # --- "Голова", которая объединяет все и делает предсказание ---
        # Теперь она принимает на вход признаки от EfficientNet
        self.fusion_head = nn.Sequential(
            nn.Linear(num_image_features + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, image, ingredients, mass):
        image_features = self.image_model(image)
        
        embedded_ingredients = self.ingredient_embedding(ingredients)
        mean_ingredients = embedded_ingredients.mean(dim=1)
        text_input = torch.cat([mean_ingredients, mass], dim=1)
        text_features = self.text_branch(text_input)

        combined_features = torch.cat([image_features, text_features], dim=1)
        prediction = self.fusion_head(combined_features)
        return prediction