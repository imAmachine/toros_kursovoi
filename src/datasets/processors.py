import numpy as np


class ShiftProcessor:
    DIRECTIONS = ['top', 'bottom', 'left', 'right']
    
    def __init__(self, shift_percent=0.15):
        self.shift_percent = shift_percent
    
    def create_noise_mask(self, damage_mask, noise_level=0.1, inversed=False):
        noise = np.random.uniform(0, noise_level, size=damage_mask.shape).astype(np.float32)
        m = damage_mask
        if inversed:
            m = (1 - damage_mask)
        return m * noise
    
    def create_damage_mask(self, damage_mask_shape, dmg_size, direction, dmg_val=1.0):
        damage_mask = np.zeros(damage_mask_shape, dtype=np.float32)
        
        if direction == 'top':
            damage_mask[:dmg_size, :] = dmg_val
        elif direction == 'bottom':
            damage_mask[-dmg_size:, :] = dmg_val
        elif direction == 'left':
            damage_mask[:, :dmg_size] = dmg_val
        elif direction == 'right':
            damage_mask[:, -dmg_size:] = dmg_val
        
        return damage_mask
    
    def process(self, image: np.ndarray, masked=False, noised=False) -> tuple:
        img_size = image.shape
        damage_size = int(max(img_size[0], img_size[1]) * self.shift_percent)
        
        damaged = image.copy()
        damage_direction = np.random.choice(self.DIRECTIONS)
        
        damage_mask = self.create_damage_mask(damage_mask_shape=img_size, 
                                              dmg_size=damage_size, 
                                              direction=damage_direction, 
                                              dmg_val=1.0)
        
        if masked:
            damaged *= (1 - damage_mask)
            
        if noised:
            damaged += self.create_noise_mask(damage_mask, noise_level=0.5)
            
        return damaged, damage_mask
    