from typing import Any, Dict
import cv2
import numpy as np

from ..interfaces.MaskProcessor import MaskProcessor

class RotateMaskProcessor(MaskProcessor):
    """Процессор для коррекции поворота маски"""
    
    def __init__(self, crop_percent: int = 5, kernel_size: int = 5, postprocess_kernel_size: int = 5):
        self.crop_percent = crop_percent
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        self.postprocess_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (postprocess_kernel_size, postprocess_kernel_size))
    
    def process(self, image: np.ndarray, metadata: Dict[str, Any] = None) -> tuple[np.ndarray, Dict[str, Any]]:
        if metadata is None:
            metadata = {}
        
        # Предобработка: бинаризация и морфологическое закрытие
        processed = self._preprocess(image)
        
        # Определение угла поворота по ограниченной области изображения
        bounded = self.bounding_crop(processed)
        angle = self._calculate_rotation_angle(bounded)
        angle = self._normalize_angle(angle)
        
        # Поворот исходного изображения, последующая обрезка и финальное выделение ограничивающего прямоугольника
        rotated = self._rotate_image(image, angle)
        cropped = self.bounding_crop(self._crop_image(rotated))
        
        metadata['rotation_angle'] = angle
        return cropped, metadata
    
    def _normalize_angle(self, angle: float) -> float:
        """Приводит угол к диапазону [-90, 90) градусов"""
        # Используем формулу для приведения угла к диапазону [-90, 90)
        return ((angle + 90) % 180) - 90
    
    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        # Бинаризация изображения и морфологическое закрытие для устранения шумов
        _, binary = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.kernel)
    
    def _calculate_rotation_angle(self, image: np.ndarray) -> float:
        # Метод 1: Определение направления через PCA
        y, x = np.where(image > 0)
        if len(x) < 2:
            return 0.0
        
        data = np.column_stack((x.astype(np.float32), y.astype(np.float32)))
        cov_matrix = np.cov(data - np.mean(data, axis=0), rowvar=False)
        _, eigenvectors = np.linalg.eig(cov_matrix)
        pca_angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        
        # Метод 2: Вычисление минимального прямоугольника
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            rect_angle = cv2.minAreaRect(cnt)[-1]
            # Выбираем угол с большим модулем, который может лучше отражать поворот
            return pca_angle if abs(pca_angle) > abs(rect_angle) else rect_angle
        
        return 0.0
    
    def _rotate_image(self, img: np.ndarray, angle: float) -> np.ndarray:
        h, w = img.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # Последующая бинаризация и морфологическая обработка для улучшения качества маски
        _, rotated = cv2.threshold(rotated, 64, 255, cv2.THRESH_BINARY)
        return cv2.morphologyEx(rotated, cv2.MORPH_CLOSE, self.postprocess_kernel)
    
    def bounding_crop(self, img: np.ndarray) -> np.ndarray:
        coords = cv2.findNonZero(img)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            return img[y:y+h, x:x+w]
        return img

    def _crop_image(self, img: np.ndarray) -> np.ndarray:
        if self.crop_percent > 0:
            h, w = img.shape
            crop_size_w, crop_size_h = int(w * (self.crop_percent / 100)), int(h * (self.crop_percent / 100))
            return img[crop_size_w:w-crop_size_w, crop_size_h:h-crop_size_h]
        return img
