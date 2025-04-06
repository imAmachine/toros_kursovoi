from src.preprocessing.interfaces import IProcessor
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from typing import Dict, Tuple
import random


class IceRidgeDataset(Dataset):
    def __init__(self, metadata: Dict, dataset_processor: IProcessor = None, transform=None):
        self.processor = dataset_processor
        self.metadata = metadata
        self.transform = transform
        self.image_keys = list(metadata.keys())
    
    def __len__(self) -> int:
        return len(self.image_keys)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Возвращает тройку (input, target, damage_mask) для индекса idx"""
        key = self.image_keys[idx]
        orig_meta = self.metadata[key]
        orig_path = orig_meta.get('output_path')
        
        orig_image = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
        
        if orig_image is None:
            raise ValueError(f"Не удалось загрузить изображение: {orig_path}")
        
        damaged_image, damage_mask = self.processor.process(orig_image)
        
        # Преобразование в тензоры
        damaged_tensor = self._image_to_tensor(damaged_image)
        original_tensor = self._image_to_tensor(orig_image)
        mask_tensor = self._image_to_tensor(damage_mask)
        
        if self.transform:
            damaged_tensor = self.transform(damaged_tensor)
            original_tensor = self.transform(original_tensor)
            mask_tensor = self.transform(mask_tensor)
            
        return damaged_tensor, original_tensor, mask_tensor
    
    def _image_to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Конвертация numpy array в тензор PyTorch"""
        return torch.from_numpy(img).float().unsqueeze(0) / 255.0
    
    @staticmethod
    def split_dataset(metadata: Dict, val_ratio=0.2, seed=42) -> Tuple[Dict, Dict]:
        """Разделяет метаданные на обучающую и валидационную выборки"""
        random.seed(seed)
        
        all_keys = list(metadata.keys())
        random.shuffle(all_keys)
        
        val_size = int(len(all_keys) * val_ratio)
        
        val_keys = all_keys[:val_size]
        train_keys = all_keys[val_size:]
        
        train_metadata = {k: metadata[k] for k in train_keys}
        val_metadata = {k: metadata[k] for k in val_keys}
        
        return train_metadata, val_metadata