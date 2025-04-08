from typing import Any, Dict
import numpy as np


class ShiftProcessor:
    DIRECTIONS = ['top', 'bottom', 'left', 'right']
    
    def __init__(self, shift_percent, noise_level=1):
        self.shift_percent = shift_percent
        self.noise_level = noise_level
    
    def process(self, image: np.ndarray, add_target=True) -> tuple:
        h, w = image.shape
        
        damaged = (image.astype(np.float32) / 255.0)

        damage_mask = np.ones_like(image, dtype=np.float32)
        damage_size = int(max(h, w) * self.shift_percent)
        direction = np.random.choice(self.DIRECTIONS)
        
        damage_mask_val = 0.0
        
        if direction == 'top':
            damage_mask[:damage_size, :] = damage_mask_val
        elif direction == 'bottom':
            damage_mask[-damage_size:, :] = damage_mask_val
        elif direction == 'left':
            damage_mask[:, :damage_size] = damage_mask_val
        elif direction == 'right':
            damage_mask[:, -damage_size:] = damage_mask_val
        
        # Генерация шума для поврежденной области
        noise = np.random.uniform(0, self.noise_level, size=image.shape).astype(np.float32)
        noise_mask = (1 - damage_mask) * noise
        
        # Применяем маску к изображению и добавляем шум в поврежденную область
        if add_target:
            damaged = damaged + noise_mask
        else:
            damaged = damaged * damage_mask + noise_mask
            
        return damaged, damage_mask
    