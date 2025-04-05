import os
import cv2
import numpy as np
from typing import Dict, Iterator, Tuple
import torch

from src.preprocessing.interfaces import IProcessor

class IceRidgeDataset:
    def __init__(self, dataset_processor: IProcessor = None):
        self.processor = dataset_processor
        self.metadata = None

    def process_dataset(self, metadata: Dict) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
        """Генератор, возвращающий тройки (input, target, damage_mask) и фрактальную размерность"""
        for orig_name, orig_meta in metadata.items():
            orig_path = orig_meta['output_path']
            orig_image = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
            
            if orig_image is not None:
                # Применяем обработку для генерации повреждений
                damaged_image, damage_mask = self.processor.process(orig_image)
                yield (
                    damaged_image, 
                    orig_image, 
                    damage_mask, 
                    orig_meta['fractal_dimension']
                )

    def to_tensor_dataset(self, metadata: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Конвертация всего датасета в тензоры PyTorch"""
        inputs, targets, damages, fractals = [], [], [], []
        
        for damaged, target, damage, fd in self.stream_dataset(metadata):
            inputs.append(self._image_to_tensor(damaged))
            targets.append(self._image_to_tensor(target))
            damages.append(self._image_to_tensor(damage))
            fractals.append(torch.tensor(fd))
            
        return (
            torch.stack(inputs),
            torch.stack(targets),
            torch.stack(damages),
            torch.stack(fractals)
        )

    def _image_to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Конвертация numpy array в тензор PyTorch"""
        return torch.from_numpy(img).float().unsqueeze(0) / 255.0