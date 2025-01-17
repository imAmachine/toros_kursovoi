import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
import random

from generator.shifter.image_shifter import ImageShifter


class TIFDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None, noise_factor=0.1):
        """
        Инициализация датасета.

        :param image_dir: Директория с изображениями.
        :param mask_dir: Директория с масками.
        :param image_transform: Трансформации для изображений.
        :param mask_transform: Трансформации для масок.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.image_mask_pair_paths = []
        
        self.image_shifter = ImageShifter(448)

        # Собираем файлы из обеих директорий
        image_files = [f for f in os.listdir(image_dir)]
        mask_files = [f for f in os.listdir(mask_dir)]

        # Сопоставляем изображения и маски по имени файла
        for name in image_files:
            if name in mask_files:
                self.image_mask_pair_paths.append(name)
        self.noise_factor = noise_factor

    def _get_rand_shift_coefs(self):
        horiz_dir = random.choice([-1, 1])
        vert_dir = random.choice([-1, 1])
        
        horiz_shift_percent = random.choice(range(10, 16)) * horiz_dir
        vert_shift_percent = random.choice(range(10, 16)) * vert_dir
        
        return horiz_shift_percent, vert_shift_percent
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_mask_pair_paths[idx])
        mask_path = os.path.join(self.mask_dir, self.image_mask_pair_paths[idx])

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path)

        # Сдвиговые коэффициенты
        horiz, vert = self._get_rand_shift_coefs()
        shift_horiz = int(image.width * (horiz / 100))
        shift_vert = int(image.height * (vert / 100))

        # Рассчитываем координаты для кропа
        x_start_src = max(0, shift_horiz)
        y_start_src = max(0, shift_vert)
        x_end_src = min(image.width, image.width + shift_horiz) if shift_horiz < 0 else image.width
        y_end_src = min(image.height, image.height + shift_vert) if shift_vert < 0 else image.height
        
        # Применяем кроп
        cropped_image = image.crop((x_start_src, y_start_src, x_end_src, y_end_src))
        cropped_mask = mask.crop((x_start_src, y_start_src, x_end_src, y_end_src))
        
        if shift_horiz < 0:
            x_start_src = 0
        if shift_vert < 0:
            y_start_src = 0

        # Создаем пустые изображения для вставки
        full_image = Image.new("L", (image.width, image.height), 0)  # Заполняем нулями
        full_mask = Image.new("L", (mask.width, mask.height), 2)  # Заполняем двойками

        # Вставляем кропнутые изображения
        full_image.paste(cropped_image, (x_start_src, y_start_src))
        full_mask.paste(cropped_mask, (x_start_src, y_start_src))

        # Генерация шума в пустых областях (в расширенных областях)
        image_array = np.array(full_image)
        mask_array = np.array(full_mask)

        # Определение областей, в которых нужно добавить шум
        noise_area_image = (image_array == 0)  # Области с нулями
        noise_area_mask = (mask_array == 2)

        # Генерация случайного шума (можно адаптировать под вашу задачу)
        image_tensor = torch.tensor(image_array)
        mask_tensor = torch.tensor(mask_array)
        noise_image = torch.randint(0, 256, image_tensor.shape, dtype=torch.uint8)
        noise_mask = torch.randint(0, 256, mask_tensor.shape, dtype=torch.uint8)

        # Применяем шум только в тех областях, где были нули
        image_array[noise_area_image] = noise_image[noise_area_image]
        mask_array[noise_area_mask] = noise_mask[noise_area_mask]

        # Преобразуем обратно в изображение
        full_image_with_noise = Image.fromarray(image_array)
        full_mask_with_noise = Image.fromarray(mask_array)

        shifted_image_normalized = self.image_transform(full_image_with_noise)
        shifted_mask_normalized = self.mask_transform(full_mask_with_noise)
        
        image_normalized = self.image_transform(image)
        mask_normalized = self.mask_transform(mask)
        
        combined_real_input = torch.cat([image_normalized, mask_normalized], dim=0)
        shifted_noisy_combined = torch.cat([shifted_image_normalized, shifted_mask_normalized])

        return combined_real_input, shifted_noisy_combined, mask_normalized

    def __len__(self):
        """Возвращает количество пар изображений в датасете"""
        return len(self.image_mask_pair_paths)