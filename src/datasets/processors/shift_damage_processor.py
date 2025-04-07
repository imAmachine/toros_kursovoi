from typing import Any, Dict
import numpy as np
from src.preprocessing.interfaces import IProcessor


class ShiftProcessor(IProcessor):
    DIRECTIONS = ['top', 'bottom', 'left', 'right']
    
    def __init__(self, shift_percent):
        self.shift_percent = shift_percent
    
    def process(self, image: np.ndarray, metadata: Dict[str, Any] = None) -> tuple:
        h, w = image.shape
        
        damaged = (image.astype(np.float32) / 255.0)

        damage_mask = np.zeros_like(image)
        damage_size = int(max(h, w) * self.shift_percent)
        direction = np.random.choice(self.DIRECTIONS)
        
        damage_mask_val = 1.0
        
        if direction == 'top':
            damage_mask[:damage_size, :] = damage_mask_val
        elif direction == 'bottom':
            damage_mask[-damage_size:, :] = damage_mask_val
        elif direction == 'left':
            damage_mask[:, :damage_size] = damage_mask_val
        elif direction == 'right':
            damage_mask[:, -damage_size:] = damage_mask_val
        
        damaged = damaged * (1 - damage_mask)
            
        return damaged, damage_mask
    