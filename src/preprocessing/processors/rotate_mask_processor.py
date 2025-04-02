from typing import Any, Dict
import cv2
import numpy as np

from ..interfaces.MaskProcessor import MaskProcessor

class RotateMaskProcessor(MaskProcessor):
    """Процессор для коррекции поворота маски"""
    
    def __init__(self, crop_percent: int = 10, kernel_size: int = 7, postprocess_kernel_size: int = 5):
        self.crop_percent = crop_percent
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        self.postprocess_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (postprocess_kernel_size, postprocess_kernel_size))
        self.current_angle = 0
    
    def process(self, image: np.ndarray, metadata: Dict[str, Any] = None) -> tuple[np.ndarray, Dict[str, Any]]:
        if metadata is None:
            metadata = {}
            
        # Предобработка изображения
        processed = self._preprocess(image)
        
        # Вычисление угла поворота
        angle = self._calculate_rotation_angle(processed)
        angle = self._normalize_angle(angle)
        
        # Поворот и обрезка изображения
        rotated = self._rotate_image(image, angle)
        cropped = self._crop_image(rotated)
        
        # Сохранение угла в метаданных
        metadata['rotation_angle'] = angle
        
        return cropped, metadata
    
    def _normalize_angle(self, angle):
        """Приводит угол к диапазону [-90, 90) градусов"""
        angle %= 180  # Приводим к диапазону [0, 180)
        if angle > 90:
            return angle - 180
        elif angle < -90:
            return angle + 180
        return angle
    
    def _preprocess(self, img):
        _, binary = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.kernel)
    
    def _calculate_rotation_angle(self, image):
        # Метод 1: PCA
        y, x = np.where(image > 0)
        if len(x) < 2:
            return 0.0
        
        data = np.vstack([x, y]).T.astype(np.float32)
        cov_matrix = np.cov(data - np.mean(data, axis=0), rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        pca_angle = np.degrees(np.arctan2(eigenvectors[1,0], eigenvectors[0,0]))

        # Метод 2: Минимальный прямоугольник
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            rect_angle = cv2.minAreaRect(cnt)[-1]
            
            # Выбор оптимального угла
            return pca_angle if abs(pca_angle) > abs(rect_angle) else rect_angle
        
        return 0.0
    
    def _rotate_image(self, img, angle):
        h, w = img.shape
        center = (w//2, h//2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        rotated = cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        _, rotated = cv2.threshold(rotated, 64, 255, cv2.THRESH_BINARY)
        rotated = cv2.morphologyEx(rotated, cv2.MORPH_CLOSE, self.postprocess_kernel)
        return rotated

    def _crop_image(self, img):
        # Обрезка по процентам
        if self.crop_percent > 0:
            h, w = img.shape
            crop_size = int(min(w, h) * self.crop_percent / 100)
            if crop_size*2 < min(h, w):  # Проверка, чтобы не обрезать всё изображение
                img = img[crop_size:-crop_size, crop_size:-crop_size]
        
        # Обрезка до bounding box
        coords = cv2.findNonZero(img)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            return img[y:y+h, x:x+w]
        return img