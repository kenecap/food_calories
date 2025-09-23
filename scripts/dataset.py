# scripts/dataset.py

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import os
from PIL import Image

class FoodDataset(Dataset):
    """Класс для загрузки данных о блюдах."""
    
    def __init__(self, data_dir, split='train'):
        # 1. Загружаем таблицы
        dish_df_path = os.path.join(data_dir, 'dish.csv')
        ingr_df_path = os.path.join(data_dir, 'ingredients.csv')
        df_dish = pd.read_csv(dish_df_path)
        df_ingredients = pd.read_csv(ingr_df_path)

        # 2. Выбираем нужную часть датасета (train или test)
        self.data = df_dish[df_dish['split'] == split].reset_index(drop=True)
        self.images_dir = os.path.join(data_dir, 'images')

        # 3. Создаем словарь для преобразования ID ингредиента в число (индекс)
        # Индекс 0 мы оставляем для "пустышки" (padding)
        self.ingr_to_idx = {ingr_id: i + 1 for i, ingr_id in enumerate(df_ingredients['id'])}
        self.num_ingredients = len(self.ingr_to_idx) + 1

        # 4. Определяем трансформации для изображений
        if split == 'train':
            # Для обучающей выборки применяем аугментацию (случайные изменения)
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            # Для тестовой выборки просто приводим картинки к нужному виду
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        """Возвращает общее количество примеров в датасете."""
        return len(self.data)

    def __getitem__(self, idx):
        """Возвращает один пример данных по его индексу."""
        # 1. Находим нужную строку в таблице
        row = self.data.iloc[idx]

        # 2. Обрабатываем изображение
        dish_id = row['dish_id']
        image_path = os.path.join(self.images_dir, dish_id, 'rgb.png')
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)

        # 3. Обрабатываем ингредиенты
        ingredients_str = row['ingredients']
        ingr_indices = [self.ingr_to_idx[ingr_id] for ingr_id in ingredients_str.split(';')]
        ingr_tensor = torch.LongTensor(ingr_indices)

        # 4. Получаем массу и калорийность
        mass = torch.tensor(row['total_mass'], dtype=torch.float32)
        calories = torch.tensor(row['total_calories'], dtype=torch.float32)

        return {
            'image': image_tensor,
            'ingredients': ingr_tensor,
            'mass': mass,
            'calories': calories
        }

def collate_fn(batch):
    """
    Функция для "сборки" батча.
    Так как у разных блюд разное количество ингредиентов, эта функция делает
    все списки ингредиентов в батче одинаковой длины, добавляя нули.
    """
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