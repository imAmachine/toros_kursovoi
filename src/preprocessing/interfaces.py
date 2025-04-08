from abc import ABC, abstractmethod

import numpy as np
from typing import Dict, Any


class IProcessor(ABC):
    """Базовый абстрактный класс для обработчиков масок"""
    
    @abstractmethod
    def process(self, image: np.ndarray, metadata: Dict[str, Any] = None) -> tuple:
        pass