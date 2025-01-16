import os
from torchvision.transforms import ToTensor, transforms
from torch.utils.data import Dataset
from PIL import Image
import torch


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

    def add_noise(self, tensor):
        """Добавление гауссовского шума к тензору"""
        noise = torch.randn_like(tensor) * self.noise_factor
        return tensor + noise

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_mask_pair_paths[idx])
        mask_path = os.path.join(self.mask_dir, self.image_mask_pair_paths[idx])

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path)

        # Применяем трансформации
        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        # Создаем входные данные с шумом
        input_combined = torch.cat([image, mask], dim=0)
        combined_noisy_input = self.add_noise(input_combined)

        return combined_noisy_input, input_combined, mask

    def __len__(self):
        """Возвращает количество пар изображений в датасете"""
        return len(self.image_mask_pair_paths)