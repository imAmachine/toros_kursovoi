import os
from torch.utils.data import Dataset
import torch
from osgeo import gdal
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np

class TIFDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None, noise_factor=0.1, target_size=(4096, 4096)):
        """
        Инициализация датасета.

        :param image_dir: Директория с изображениями.
        :param mask_dir: Директория с масками.
        :param image_transform: Трансформации для изображений.
        :param mask_transform: Трансформации для масок.
        :param target_size: Размер, до которого нужно масштабировать изображения.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.image_mask_pair_paths = []
        self.target_size = target_size

        # Собираем файлы из обеих директорий
        image_files = [f for f in os.listdir(image_dir)]
        mask_files = [f for f in os.listdir(mask_dir)]

        # Сопоставляем изображения и маски по имени файла
        for name in image_files:
            if name in mask_files:
                self.image_mask_pair_paths.append(name)
        self.noise_factor = noise_factor

    def add_noise(self, tensor):
        """Добавление гауссовского шума к тензору"""
        noise = torch.randn_like(tensor) * self.noise_factor
        return tensor + noise

    def read_tif(self, file_path):
        """Чтение TIF файла с использованием GDAL и приведение к numpy array"""
        dataset = gdal.Open(file_path)
        band = dataset.GetRasterBand(1)
        image_array = band.ReadAsArray()
        return image_array

    def resize_image(self, image_array):
        """Масштабирование изображения до target_size"""
        image = Image.fromarray(image_array)
        image = image.resize(self.target_size, Image.ANTIALIAS)
        return np.array(image)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_mask_pair_paths[idx])
        mask_path = os.path.join(self.mask_dir, self.image_mask_pair_paths[idx])
        
        # Чтение и масштабирование изображения и маски
        image_array = self.read_tif(image_path)
        mask_array = self.read_tif(mask_path)
        
        image_array_resized = self.resize_image(image_array)
        mask_array_resized = self.resize_image(mask_array)
        
        # Преобразуем в формат PIL для трансформации
        image_pil = Image.fromarray(image_array_resized)
        mask_pil = Image.fromarray(mask_array_resized)
        
        # Применяем трансформации
        if self.image_transform:
            image_tensor = self.image_transform(image_pil)
        if self.mask_transform:
            mask_tensor = self.mask_transform(mask_pil)
        
        # Создаем входные данные с шумом
        input_combined = torch.cat([image_tensor, mask_tensor], dim=0)
        combined_noisy_input = self.add_noise(input_combined)
        
        return combined_noisy_input, input_combined, mask_tensor

    def __len__(self):
        """Возвращает количество пар изображений в датасете"""
        return len(self.image_mask_pair_paths)
