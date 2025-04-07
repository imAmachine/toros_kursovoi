import cv2
import numpy as np
from typing import Dict, Any
from ..interfaces.IProcessor import IProcessor
import random

class RandomHoleProcessor(IProcessor):
    METADATA_NAME = 'random_holes'
    
    def __init__(self, 
                 num_holes: int = 1,
                 hole_size_range: tuple = (10, 50)):
        self.num_holes = num_holes
        self.hole_size_range = hole_size_range

    def process(self, 
               image: np.ndarray, 
               metadata: Dict[str, Any] = None) -> tuple:
        if metadata is None:
            metadata = {}
            
        damaged = image.copy()
        mask = np.zeros_like(image)
        h, w = image.shape
        
        for _ in range(self.num_holes):
            hole_w = random.randint(*self.hole_size_range)
            hole_h = random.randint(*self.hole_size_range)
            x = random.randint(0, w - hole_w)
            y = random.randint(0, h - hole_h)
            
            damaged[y:y+hole_h, x:x+hole_w] = 0
            mask[y:y+hole_h, x:x+hole_w] = 255

        return damaged, mask