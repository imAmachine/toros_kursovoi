import cv2
import numpy as np
from typing import Dict, Iterator, Tuple
import torch

from src.preprocessing.interfaces import IProcessor

class IceRidgeDataset:
    def __init__(self, metadata, dataset_processor: IProcessor = None):
        self.processor = dataset_processor
        self.metadata = metadata

    def process_dataset(self, metadata: Dict) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Генератор, возвращающий тройки (input, target, damage_mask) и фрактальную размерность"""
       
        for orig_name, orig_meta in metadata.items():
            orig_path = orig_meta.get('output_path')
            orig_image = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
            
            if orig_image is not None:
                damaged_image, damage_mask = self.processor.process(orig_image)
                yield damaged_image, orig_image, damage_mask

    def split_dataset(self, metadata: Dict, val_ratio=0.2, seed=42):
        """Разделяет метаданные на обучающую и валидационную выборки"""
        import random
        random.seed(seed)
        
        # Получаем список всех ключей
        all_keys = list(metadata.keys())
        random.shuffle(all_keys)
        
        # Вычисляем размер валидационной выборки
        val_size = int(len(all_keys) * val_ratio)
        
        # Разделяем ключи
        val_keys = all_keys[:val_size]
        train_keys = all_keys[val_size:]
        
        # Создаем словари метаданных
        train_metadata = {k: metadata[k] for k in train_keys}
        val_metadata = {k: metadata[k] for k in val_keys}
        
        return train_metadata, val_metadata
    
    def to_tensor_dataset(self, metadata: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Конвертация всего датасета в тензоры PyTorch"""
        inputs, targets, damages = [], [], []
        
        for damaged, target, damage in self.process_dataset(metadata):
            inputs.append(self._image_to_tensor(damaged))
            targets.append(self._image_to_tensor(target))
            damages.append(self._image_to_tensor(damage))
            
        return torch.stack(inputs), torch.stack(targets), torch.stack(damages)

    def _image_to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Конвертация numpy array в тензор PyTorch"""
        return torch.from_numpy(img).float().unsqueeze(0) / 255.0