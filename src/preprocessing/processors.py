from enum import Enum
from typing import Any, Dict
import cv2
import numpy as np
import torch

from src.analyzer.fractal_funcs import FractalAnalyzer, FractalAnalyzerGPU
from src.analyzer.rotation_analyze import RotationAnalyze
from .interfaces import IProcessor
from .utils import ImageProcess


class AngleChooseType(Enum):
    ABS = RotationAnalyze.get_max_abs_angle
    WEIGHTED = RotationAnalyze.get_weighted_angle
    CONSISTENT = RotationAnalyze.get_consistent_angle


class CropProcessor(IProcessor):
    """Процессор для обрезки изображения"""
    
    def __init__(self, processor_name: str = None, crop_percent: int = 0):
        super().__init__(processor_name)
        self.crop_percent = crop_percent
    
    @property
    def PROCESSORS_NEEDED(self):
        return []
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        return ImageProcess.crop_image(image, self.crop_percent)


class AutoAdjust(IProcessor):
    """Процессор для автоматической настройки изображения"""
    
    @property
    def PROCESSORS_NEEDED(self):
        return []
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        return ImageProcess.auto_adjust(image)


class Binarize(IProcessor):
    """Процессор для бинаризации изображения"""
    
    @property
    def PROCESSORS_NEEDED(self):
        return []
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        return ImageProcess.binarize_by_threshold(image)


class EnchanceProcessor(IProcessor):
    """Процессор для улучшения бинаризованного изображения с помощью морфологических операций"""
    
    def __init__(self, processor_name: str = None, morph_kernel_size: int = 5):
        super().__init__(processor_name)
        self.morph_kernel_size = morph_kernel_size
    
    @property
    def PROCESSORS_NEEDED(self):
        return [Binarize]
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        return ImageProcess.morph_bin_image(image, ksize=self.morph_kernel_size)


class RotateMaskProcessor(IProcessor):
    """Процессор для выравнивания изображения"""
    
    def __init__(self, processor_name: str = None, angle_choose_type: callable = None):
        super().__init__(processor_name)
        self.angle_choose_type = angle_choose_type
    
    @property
    def PROCESSORS_NEEDED(self):
        return []
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        cropped_image = ImageProcess.bounding_crop(image)
        temp_for_angle = ImageProcess.morph_bin_image(cropped_image.copy(), ksize=7)
        angle = self._calculate_rotation_angle(temp_for_angle)
        rotated_image = self._rotate_image(cropped_image, angle)
        final_image = ImageProcess.bounding_crop(rotated_image)
        
        self._result_value = angle
        
        return final_image
    
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
        rotated_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, 
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return rotated_img


class TensorConverterProcessor(IProcessor):
    """Процессор для преобразования numpy.ndarray в torch.Tensor"""
    
    def __init__(self, processor_name: str = None, device: str = "cpu"):
        super().__init__(processor_name)
        self.device = device
    
    @property
    def PROCESSORS_NEEDED(self):
        return []
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            return image
            
        tensor_image = torch.from_numpy(image).float().to(self.device)
        
        if len(tensor_image.shape) == 2:
            tensor_image = tensor_image.unsqueeze(0)
        
        return tensor_image


class FractalDimensionProcessorGPU(IProcessor):
    """Процессор для расчета фрактальной размерности с использованием GPU"""
    @property
    def PROCESSORS_NEEDED(self):
        return [TensorConverterProcessor]
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        fr_dim = FractalAnalyzerGPU.calculate_fractal_dimension(
            *FractalAnalyzerGPU.box_counting(image), 
            device=image.device)
        
        self._result_value = fr_dim.item()
        
        return image


class FractalDimensionProcessorCPU(IProcessor):
    """Процессор для расчета фрактальной размерности с использованием CPU"""
    @property
    def PROCESSORS_NEEDED(self):
        return []
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        fr_dim = FractalAnalyzer.calculate_fractal_dimension(
            *FractalAnalyzer.box_counting(image))
        
        self._result_value = fr_dim.item()
        
        return image