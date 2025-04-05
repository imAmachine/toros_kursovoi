from typing import Any, Dict
import numpy as np
from ..interfaces import IProcessor
from ..utils import ImageProcess


class CropProcessor(IProcessor):
    METADATA_NAME = 'crop'
    
    def __init__(self, crop_percent = 0):
        self.crop_percent = crop_percent
        
    def process(self, image: np.ndarray, metadata: Dict[str, Any] = None) -> tuple[np.ndarray, Dict[str, Any]]:
        cropped = ImageProcess.crop_image(image, self.crop_percent)
        adjusted = ImageProcess.auto_adjust(cropped)
        metadata.update({self.METADATA_NAME: True})
        
        return adjusted, metadata