import os
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from PIL import Image
import torch


class TIFDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None):
        """
        Инициализация датасета.

        :param image_dir: Директория с изображениями.
        :param mask_dir: Директория с масками.
        :param image_transform: Трансформации для изображений.
        :param mask_transform: Трансформации для масок.
        """
        self.image_paths = []
        self.mask_paths = []

        # Собираем файлы из обеих директорий
        image_files = {os.path.splitext(f)[0]: os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")}
        mask_files = {os.path.splitext(f)[0]: os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".png")}

        # Сопоставляем изображения и маски по имени файла
        for name in image_files.keys():
            if name in mask_files:
                self.image_paths.append(image_files[name])
                self.mask_paths.append(mask_files[name])

        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def _gen_noised(self, image, noise_level=0.1, retain_center=True, center_fraction=0.65):
        """
        Генерация зашумленного изображения.

        :param image: Исходное изображение (тензор).
        :param noise_level: Уровень шума (стандартное отклонение для гауссовского шума).
        :param retain_center: Флаг, указывающий, сохранять ли центральную область без изменений.
        :param center_fraction: Доля центра изображения, который сохраняется без изменений, если retain_center=True.
        :return: Зашумленное изображение.
        """
        noise_image = image + noise_level * torch.randn_like(image)

        if retain_center:
            _, h, w = image.shape
            ch, cw = int(h * center_fraction), int(w * center_fraction)
            top, left = (h - ch) // 2, (w - cw) // 2
            bottom, right = top + ch, left + cw
            noise_image[:, top:bottom, left:right] = image[:, top:bottom, left:right]

        return noise_image.clamp(0, 1)
    
    def _apply_transforms(self, image, mask):
        if self.image_transform:
            image = self.image_transform(image)
        else:
            image = ToTensor()(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = (ToTensor()(mask) > 0.5).float()
        
        return image, mask
    
    def __len__(self):
        """Возвращает количество пар изображений и масок."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Загружает и возвращает:
        - Комбинированное изображение с маской (2 канала),
        - Комбинированное изображение с шумом (2 канала),
        - Итоговая маска для оценки loss при генерации.
        """
        image = Image.open(self.image_paths[idx]).convert("L")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        image, mask = self._apply_transforms(image, mask)
        
        image_noised = self._gen_noised(image)
        mask_noised = self._gen_noised(mask)

        combined = torch.cat([image, mask], dim=0)
        combined_noised = torch.cat([image_noised, mask_noised], dim=0)
        
        return combined, combined_noised, mask