# scripts/model.py

import torch
import torch.nn as nn
import torchvision.models as models

class CaloriePredictor(nn.Module):
    def __init__(self, num_ingredients, embedding_dim=128):
        super().__init__()

        # --- ГЛАВНОЕ ИЗМЕНЕНИЕ: ИСПОЛЬЗУЕМ БОЛЕЕ МОЩНУЮ МОДЕЛЬ ---
        self.image_branch = models.resnet50(pretrained=True)
        # --------------------------------------------------------
        
        # Оставляем полную заморозку - это самая стабильная стратегия
        for param in self.image_branch.parameters():
            param.requires_grad = False

        # Этот код сработает без изменений, т.к. у ResNet50 тоже есть .fc слой
        num_features = self.image_branch.fc.in_features
        self.image_branch.fc = nn.Linear(num_features, 256)

        # Остальная часть модели остается прежней
        self.ingredient_embedding = nn.Embedding(
            num_embeddings=num_ingredients, 
            embedding_dim=embedding_dim, 
            padding_idx=0
        )
        self.text_branch = nn.Sequential(
            nn.Linear(embedding_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(256 + 256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, image, ingredients, mass):
        image_features = self.image_branch(image)
        embedded_ingredients = self.ingredient_embedding(ingredients)
        mean_ingredients = embedded_ingredients.mean(dim=1)
        text_input = torch.cat([mean_ingredients, mass], dim=1)
        text_features = self.text_branch(text_input)
        combined_features = torch.cat([image_features, text_features], dim=1)
        prediction = self.fusion_head(combined_features)
        return prediction