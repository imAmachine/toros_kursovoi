import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import random
import numpy as np


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

    @staticmethod
    def preprocess(image, mask, image_transform, mask_transform, horiz_shift_percent=0, vert_shift_percent=0):
        """
        Выполняет предобработку изображения и маски (сдвиг, бинаризация, добавление шума).
        
        :param image: Входное изображение (PIL Image).
        :param mask: Входная маска (PIL Image).
        :param horiz_shift_percent: Процент горизонтального сдвига.
        :param vert_shift_percent: Процент вертикального сдвига.
        :return: Предобработанные изображения и маски (с нормализацией).
        """
        # Преобразуем маску в одноканальный формат и выполняем бинаризацию
        mask = mask.convert("L")  # Преобразование в 1 канал (оттенки серого)
        mask = np.array(mask)
        mask = (mask > 127).astype(np.uint8)  # Бинаризация

        mask = Image.fromarray(mask * 255)  # Преобразуем обратно в изображение для дальнейшей обработки

        # Остальная часть функции preprocess остаётся без изменений
        # Выполняем сдвиги, шум и преобразования как обычно
        shift_horiz = int(image.width * (horiz_shift_percent / 100))
        shift_vert = int(image.height * (vert_shift_percent / 100))

        x_start_src = max(0, shift_horiz)
        y_start_src = max(0, shift_vert)
        x_end_src = min(image.width, image.width + shift_horiz) if shift_horiz < 0 else image.width
        y_end_src = min(image.height, image.height + shift_vert) if shift_vert < 0 else image.height

        cropped_image = image.crop((x_start_src, y_start_src, x_end_src, y_end_src))
        cropped_mask = mask.crop((x_start_src, y_start_src, x_end_src, y_end_src))

        if shift_horiz < 0:
            x_start_src = 0
        if shift_vert < 0:
            y_start_src = 0

        full_image = Image.new("L", (image.width, image.height), 0)
        full_mask = Image.new("L", (mask.width, mask.height), 2)

        full_image.paste(cropped_image, (x_start_src, y_start_src))
        full_mask.paste(cropped_mask, (x_start_src, y_start_src))

        image_array = np.array(full_image)
        mask_array = np.array(full_mask)

        noise_area_image = (image_array == 0)
        noise_area_mask = (mask_array == 2)

        image_tensor = torch.tensor(image_array)
        mask_tensor = torch.tensor(mask_array)
        noise_image = torch.randint(0, 256, image_tensor.shape, dtype=torch.uint8)
        noise_mask = torch.randint(0, 256, mask_tensor.shape, dtype=torch.uint8)

        image_array[noise_area_image] = noise_image[noise_area_image]
        mask_array[noise_area_mask] = noise_mask[noise_area_mask]

        full_image_with_noise = Image.fromarray(image_array)
        full_mask_with_noise = Image.fromarray(mask_array)
        print(full_image_with_noise.size, full_mask_with_noise.size)

        shifted_image_normalized = image_transform(full_image_with_noise)
        shifted_mask_normalized = mask_transform(full_mask_with_noise)
        
        return shifted_image_normalized, shifted_mask_normalized

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_mask_pair_paths[idx])
        mask_path = os.path.join(self.mask_dir, self.image_mask_pair_paths[idx])

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path)

        # Сдвиговые коэффициенты
        horiz, vert = self._get_rand_shift_coefs()

        shifted_image_normalized, shifted_mask_normalized = self.preprocess(image, mask, self.image_transform, self.mask_transform, horiz, vert)
        
        image_normalized = self.image_transform(image)
        mask_normalized = self.mask_transform(mask)
        
        combined_real_input = torch.cat([image_normalized, mask_normalized], dim=0)
        shifted_noisy_combined = torch.cat([shifted_image_normalized, shifted_mask_normalized])
        
        return shifted_noisy_combined, combined_real_input, mask_normalized

    def __len__(self):
        """Возвращает количество пар изображений в датасете"""
        return len(self.image_mask_pair_paths)
