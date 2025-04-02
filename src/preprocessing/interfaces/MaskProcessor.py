from abc import ABC, abstractmethod

import numpy as np
from typing import Dict, Any


class MaskProcessor(ABC):
    """Базовый абстрактный класс для обработчиков масок"""
    
    @abstractmethod
    def process(self, image: np.ndarray, metadata: Dict[str, Any] = None) -> tuple[np.ndarray, Dict[str, Any]]:
        """
        Обрабатывает изображение маски
        
        Args:
            image: Входное изображение
            metadata: Метаданные изображения
            
        Returns:
            tuple: (обработанное изображение, обновленные метаданные)
        """
        pass