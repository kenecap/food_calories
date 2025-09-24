# scripts/dataset.py

import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import numpy as np
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(config, split='train'):
    # Получаем конфиг модели из timm, чтобы знать размеры картинок, mean и std
    image_cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)
    input_size = image_cfg.input_size[1:]
    mean = image_cfg.mean
    std = image_cfg.std

    if split == 'train':
        # Мощные аугментации для обучающей выборки из вашего примера
        return A.Compose([
            A.SmallestMaxSize(max_size=max(input_size), p=1.0),
            A.RandomCrop(height=input_size[0], width=input_size[1], p=1.0),
            A.Affine(scale=(0.8, 1.2), rotate=(-15, 15), translate_percent=(-0.1, 0.1), shear=(-10, 10), p=0.7),
            A.CoarseDropout(max_holes=8, max_height=int(0.1 * input_size[0]), max_width=int(0.1 * input_size[1]), p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.7),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        # Простые трансформации для валидации/теста
        return A.Compose([
            A.Resize(height=input_size[0], width=input_size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

class FoodDataset(Dataset):
    def __init__(self, config, split='train'):
        dish_df_path = os.path.join(config.DATA_DIR, 'dish.csv')
        ingr_df_path = os.path.join(config.DATA_DIR, 'ingredients.csv')
        df_dish = pd.read_csv(dish_df_path)
        df_ingredients = pd.read_csv(ingr_df_path)

        self.data = df_dish[df_dish['split'] == split].reset_index(drop=True)
        self.images_dir = os.path.join(config.DATA_DIR, 'images')
        
        self.ingr_to_idx = {ingr_id: i + 1 for i, ingr_id in enumerate(df_ingredients['id'])}
        self.num_ingredients = len(self.ingr_to_idx) + 1
        
        self.transforms = get_transforms(config, split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Обработка изображения с Albumentations
        dish_id = row['dish_id']
        image_path = os.path.join(self.images_dir, dish_id, 'rgb.png')
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transforms(image=np.array(image))['image']

        # Обработка ингредиентов (остается прежней, но с исправлением KeyError)
        ingredients_str = row['ingredients']
        ingr_indices = [self.ingr_to_idx[ingr_id] for ingr_id in ingredients_str.split(';') if ingr_id in self.ingr_to_idx]
        if not ingr_indices:
            ingr_indices = [0]
        ingr_tensor = torch.LongTensor(ingr_indices)

        mass = torch.tensor(row['total_mass'], dtype=torch.float32)
        calories = torch.tensor(row['total_calories'], dtype=torch.float32)

        return {
            'image': image_tensor,
            'ingredients': ingr_tensor,
            'mass': mass,
            'calories': calories
        }

# collate_fn остается прежним
def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    masses = torch.stack([item['mass'] for item in batch]).unsqueeze(1)
    calories = torch.stack([item['calories'] for item in batch]).unsqueeze(1)
    ingredients_list = [item['ingredients'] for item in batch]
    ingredients_padded = torch.nn.utils.rnn.pad_sequence(ingredients_list, batch_first=True, padding_value=0)
    return {
        'image': images,
        'ingredients': ingredients_padded,
        'mass': masses,
        'calories': calories
    }