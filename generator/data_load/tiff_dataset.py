import os
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
        rnd = random()
        horiz_dir = rnd.choice([-1, 1])
        vert_dir = rnd.choice([-1, 1])
        
        horiz_shift_percent = rnd.choice(range(10, 16)) * horiz_dir
        vert_shift_percent = rnd.choice(range(10, 16)) * vert_dir
        
        return horiz_shift_percent, vert_shift_percent
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_mask_pair_paths[idx])
        mask_path = os.path.join(self.mask_dir, self.image_mask_pair_paths[idx])

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path)
        
        # трансформации
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        input_combined = torch.cat([image, mask])
        
        # сдвиг + гаусовский шум
        horiz, vert = self._get_rand_shift_coefs()
        shifted_img, shifted_mask = self.image_shifter.apply_shift(image, mask, x_shift_percent=horiz, y_shift_percent=vert)
        
        # Создаем входные данные с шумом
        combined_noisy_input = torch.cat([shifted_img, shifted_mask], dim=0)

        return combined_noisy_input, input_combined, mask

    def __len__(self):
        """Возвращает количество пар изображений в датасете"""
        return len(self.image_mask_pair_paths)