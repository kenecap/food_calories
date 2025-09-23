# scripts/dataset.py

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import os
from PIL import Image

class FoodDataset(Dataset):
    """
    Класс для загрузки и подготовки данных о блюдах.
    Он умеет по индексу (idx) отдавать картинку, информацию об ингредиентах, массу и калорийность.
    """
    def __init__(self, data_dir, split='train'):
        """
        Конструктор класса. Здесь мы готовим данные.
        Args:
            data_dir (str): Путь к папке data.
            split (str): 'train' или 'test' для выбора нужной части данных.
        """
        # 1. Загружаем CSV файлы
        dish_df_path = os.path.join(data_dir, 'dish.csv')
        ingr_df_path = os.path.join(data_dir, 'ingredients.csv')
        df_dish = pd.read_csv(dish_df_path)
        df_ingredients = pd.read_csv(ingr_df_path)

        # 2. Оставляем только ту часть данных, которая нам нужна (train или test)
        self.data = df_dish[df_dish['split'] == split].reset_index(drop=True)
        self.images_dir = os.path.join(data_dir, 'images')

        # 3. Подготавливаем ингредиенты:
        #    Создаем словарь, который сопоставляет ID ингредиента с его числовым индексом (от 1 до N)
        #    Индекс 0 мы зарезервируем для "пустого" ингредиента (padding)
        self.ingr_to_idx = {ingr_id: i + 1 for i, ingr_id in enumerate(df_ingredients['id'])}
        # Сохраняем общее количество уникальных ингредиентов (+1 для padding)
        self.num_ingredients = len(self.ingr_to_idx) + 1

        # 4. Определяем трансформации для изображений
        #    Для обучающей выборки мы будем применять аугментацию (случайные изменения),
        #    чтобы модель лучше обобщалась.
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),      # Изменяем размер до 256x256
                transforms.RandomCrop(224),         # Случайно вырезаем квадрат 224x224
                transforms.RandomHorizontalFlip(),  # Случайно отражаем по горизонтали
                transforms.ToTensor(),              # Преобразуем картинку в тензор PyTorch
                # Нормализуем тензор (стандартные значения для моделей, обученных на ImageNet)
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else: # Для тестовой выборки аугментация не нужна
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),      # Просто изменяем размер до 224x224
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        """Этот метод должен возвращать общее количество примеров в датасете."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Этот метод получает на вход индекс (idx) и возвращает один пример данных.
        PyTorch DataLoader будет автоматически вызывать его для формирования батчей.
        """
        # 1. Получаем строку с данными по индексу
        row = self.data.iloc[idx]

        # 2. Работа с изображением
        dish_id = row['dish_id']
        image_path = os.path.join(self.images_dir, dish_id, 'rgb.png')
        image = Image.open(image_path).convert('RGB') # Открываем картинку и убеждаемся, что она в формате RGB
        image_tensor = self.transform(image) # Применяем трансформации

        # 3. Работа с ингредиентами
        ingredients_str = row['ingredients']
        
        # --- ИСПРАВЛЕНИЕ ОШИБКИ `KeyError` ---
        # Превращаем строку 'id1;id2;...' в список числовых индексов [idx1, idx2, ...],
        # при этом проверяем, есть ли такой ингредиент в нашем словаре.
        # Если его нет - мы его просто пропускаем.
        ingr_indices = [
            self.ingr_to_idx[ingr_id] 
            for ingr_id in ingredients_str.split(';') 
            if ingr_id in self.ingr_to_idx
        ]

        # На случай, если после фильтрации не осталось ни одного известного ингредиента,
        # мы создадим тензор с одним "пустым" элементом (0), чтобы избежать ошибки.
        if not ingr_indices:
            ingr_indices = [0]
            
        ingr_tensor = torch.LongTensor(ingr_indices)
        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---


        # 4. Получаем остальные данные и превращаем их в тензоры
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
    Специальная функция для обработки батча.
    Её задача - "выровнять" списки ингредиентов, так как в разных блюдах их разное количество.
    Она добавляет 0 (padding) в конец коротких списков, чтобы все они стали одной длины.
    """
    # Разделяем данные из батча по типам
    images = torch.stack([item['image'] for item in batch])
    masses = torch.stack([item['mass'] for item in batch]).unsqueeze(1) # Добавляем размерность для конкатенации
    calories = torch.stack([item['calories'] for item in batch]).unsqueeze(1)
    
    # "Выравнивание" ингредиентов
    ingredients_list = [item['ingredients'] for item in batch]
    # Используем встроенную функцию PyTorch для "выравнивания"
    # batch_first=True означает, что размер батча будет первой размерностью
    ingredients_padded = torch.nn.utils.rnn.pad_sequence(ingredients_list, batch_first=True, padding_value=0)
    
    return {
        'image': images,
        'ingredients': ingredients_padded,
        'mass': masses,
        'calories': calories
    }