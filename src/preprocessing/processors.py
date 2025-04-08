from enum import Enum
from typing import Any, Dict
import cv2
import numpy as np

from src.analyzer.fractal_funcs import FractalAnalyzer
from src.analyzer.rotation_analyze import RotationAnalyze
from .interfaces import IProcessor
from .utils import ImageProcess


class CropProcessor(IProcessor):
    METADATA_NAME = 'crop'
    
    def __init__(self, crop_percent = 0):
        self.crop_percent = crop_percent
        
    def process(self, image: np.ndarray, metadata: Dict[str, Any] = None) -> tuple[np.ndarray, Dict[str, Any]]:
        cropped = ImageProcess.crop_image(image, self.crop_percent)
        adjusted = ImageProcess.auto_adjust(cropped)
        metadata.update({self.METADATA_NAME: True})
        
        return adjusted, metadata


class EnchanceProcessor(IProcessor):
    METADATA_NAME = 'morphing'
    
    def __init__(self, morph_kernel_size=5):
        self.morph_kernel_size = morph_kernel_size
        
    
    def process(self, image: np.ndarray, metadata: Dict[str, Any] = None) -> tuple[np.ndarray, Dict[str, Any]]:
        bin_image = ImageProcess.binarize_by_threshold(image)
        morph_img = ImageProcess.morph_bin_image(bin_image, ksize=self.morph_kernel_size)
        metadata.update({self.METADATA_NAME: True})
        
        return morph_img, metadata


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
    
        result_angle = self.angle_choose_type(angles)
        
        return result_angle
    
    def _rotate_image(self, img: np.ndarray, angle: float) -> np.ndarray:
        h, w = img.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        return rotated_img


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