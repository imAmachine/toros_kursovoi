from typing import Any, Dict
import numpy as np
from ..interfaces import IProcessor
from ..utils import ImageProcess


class EnchanceProcessor(IProcessor):
    METADATA_NAME = 'morphing'
    
    def process(self, image: np.ndarray, metadata: Dict[str, Any] = None) -> tuple[np.ndarray, Dict[str, Any]]:
        bin_image = ImageProcess.binarize_by_threshold(image)
        morph_img = ImageProcess.morph_bin_image(bin_image)
        metadata.update({self.METADATA_NAME: True})
        
        return morph_img, metadata
