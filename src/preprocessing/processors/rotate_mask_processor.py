from typing import Any, Dict
from enum import Enum
import cv2
import numpy as np

from ..utils import ImageProcess, RotationAnalyze
from ..interfaces.IProcessor import IProcessor


class AngleChooseType(Enum):
    ABS = RotationAnalyze.get_max_abs_angle
    WEIGHTED = RotationAnalyze.get_weighted_angle
    CONSISTENT = RotationAnalyze.get_consistent_angle


class RotateMaskProcessor(IProcessor):
    METADATA_NAME = 'rotation'
    
    def __init__(self, angle_choose_type: AngleChooseType):
        self.angle_choose_type = angle_choose_type
        
    def process(self, image: np.ndarray, metadata: Dict[str, Any] = None) -> tuple[np.ndarray, Dict[str, Any]]:
        if metadata is None:
            metadata = {}
        
        bin_image = ImageProcess.bounding_crop(ImageProcess.binarize_by_threshold(image))
        
        angle = self._calculate_rotation_angle(bin_image)
        rotated = self._rotate_image(image, angle)
        bounded = ImageProcess.bounding_crop(rotated)
        
        metadata.update({self.METADATA_NAME: angle})
        
        return bounded, metadata
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to range [-90, 90) degrees"""
        return ((angle + 90) % 180) - 90
    
    def _calculate_rotation_angle(self, image: np.ndarray) -> float:
        angles = {
            'pca': RotationAnalyze.get_PCA_rotation_angle(image),
            'rect': RotationAnalyze.get_rect_rotation_angle(image),
            'hough': RotationAnalyze.get_hough_rotation_angle(image)
        }
    
        # Get and execute angle calculation strategy
        result_angle = self.angle_choose_type(angles)
        
        return result_angle
    
    def _rotate_image(self, img: np.ndarray, angle: float) -> np.ndarray:
        h, w = img.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        return rotated_img