from typing import Any, Dict

import numpy as np
from analyzer.fractal_funcs import FractalAnalyzer
from preprocessing.interfaces import IProcessor
from preprocessing.utils.image_processing import ImageProcess

class FractalDimensionProcessor(IProcessor):
    METADATA_NAME = 'fractal_dimension'
    
    def _calc_fract_dim(self, bin_img: np.ndarray):
        sizes, counts = FractalAnalyzer.box_counting(bin_img)
        return FractalAnalyzer.calculate_fractal_dimension(sizes, counts)
    
    def process(self, image: np.ndarray, metadata: Dict[str, Any] = None) -> tuple[np.ndarray, Dict[str, Any]]:
        bin_img = ImageProcess.binarize_by_threshold(image)
        fract_dimension = self._calc_fract_dim(bin_img)
        metadata.update({self.METADATA_NAME: fract_dimension})
        
        return image, metadata