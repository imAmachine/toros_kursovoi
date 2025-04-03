from typing import Any, Dict
import cv2
import numpy as np

from ..utils import ImageProcess, RotationAnalyze
from ..interfaces.IProcessor import IProcessor


class RotateMaskProcessor(IProcessor):
    METADATA_NAME = 'rotation'
    ANGLE_CHOOSE_TYPES = {'abs': RotationAnalyze.get_max_abs_angle, 
                          'weighted': RotationAnalyze.get_weighted_angle,
                          'consistent': RotationAnalyze.get_consistent_angle}
    
    """Процессор для коррекции поворота маски"""
    def process(self, image: np.ndarray, metadata: Dict[str, Any] = None) -> tuple[np.ndarray, Dict[str, Any]]:
        if metadata is None:
            metadata = {}
        
        bin_image = ImageProcess.bounding_crop(ImageProcess.binarize_by_threshold(image))
        
        # Определение угла поворота по ограниченной области изображения
        angle = self._calculate_rotation_angle(bin_image, angle_choose_type='consistent')
        rotated = self._rotate_image(image, angle)
        
        metadata.update({self.METADATA_NAME: angle})
        
        return rotated, metadata
    
    def _normalize_angle(self, angle: float) -> float:
        """Приводит угол к диапазону [-90, 90) градусов"""
        return ((angle + 90) % 180) - 90
    
    def _calculate_rotation_angle(self, image: np.ndarray, angle_choose_type='consistent') -> float:
        """_summary_

        Args:
            image (np.ndarray): _description_
            angle_choose (str, optional): _description_. Defaults to 'weighted'. (abs || weighted || auto)

        Returns:
            float: _description_
        """
        angles = {
            'pca': RotationAnalyze.get_PCA_rotation_angle(image),
            'rect': RotationAnalyze.get_rect_rotation_angle(image),
            'hough': RotationAnalyze.get_hough_rotation_angle(image)
        }
    
        # получение функции для расчёта итогового угла
        fn = self.ANGLE_CHOOSE_TYPES.get(angle_choose_type)
        result_angle = fn(angles)
        
        return result_angle
    
    def _rotate_image(self, img: np.ndarray, angle: float) -> np.ndarray:
        h, w = img.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return ImageProcess.bounding_crop(rotated_img)

    
