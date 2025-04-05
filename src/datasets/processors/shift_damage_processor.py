from typing import Any, Dict
import numpy as np
from src.preprocessing.interfaces import IProcessor


class ShiftProcessor(IProcessor):
    DIRECTIONS = ['top', 'bottom', 'left', 'right']
    
    def __init__(self, shift_percent):
        self.shift_percent = shift_percent
    
    def process(self, image: np.ndarray, metadata: Dict[str, Any] = None) -> tuple:
        h, w = image.shape
        
        damaged = image.copy()
        damage_mask = np.zeros_like(image)
        damage_size = int(max(h, w) * self.shift_percent)
        direction = np.random.choice(self.DIRECTIONS)
        
        if direction == 'top':
            damaged[:damage_size, :] = 0
            damage_mask[:damage_size, :] = 255
        elif direction == 'bottom':
            damaged[-damage_size:, :] = 0
            damage_mask[-damage_size:, :] = 255
        elif direction == 'left':
            damaged[:, :damage_size] = 0
            damage_mask[:, :damage_size] = 255
        elif direction == 'right':
            damaged[:, -damage_size:] = 0
            damage_mask[:, -damage_size:] = 255
            
        return damaged, damage_mask
    