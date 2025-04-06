import cv2
import numpy as np


class ImageProcess:
    @staticmethod
    def binarize_by_threshold(image: np.ndarray):
        _, binary = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)
        return binary
    
    @staticmethod
    def morph_bin_image(binary_image: np.ndarray, ksize=5):
        """улучшение с помощью морфологического закрытия"""
        return cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize)))
    
    @staticmethod
    def bounding_crop(img: np.ndarray) -> np.ndarray:
        """Обрезает изображение по краям до границ контента"""
        coords = cv2.findNonZero(img)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            return img[y:y+h, x:x+w]
        return img
    
    @staticmethod
    def crop_image(img: np.ndarray, crop_percent: int = 0) -> np.ndarray:
        if crop_percent > 0:
            h, w = img.shape
            crop_size_w, crop_size_h = int(w * (crop_percent / 100)), int(h * (crop_percent / 100))
            return img[crop_size_w:w-crop_size_w, crop_size_h:h-crop_size_h]
        return img
    
    @staticmethod
    def auto_adjust(img: np.ndarray) -> np.ndarray:
        # Получаем размеры изображения
        h, w = img.shape
        
        if h == w:
            return img
        elif h > w:
            diff = h - w
            top_crop = diff // 2
            bottom_crop = diff - top_crop
            return img[top_crop:h-bottom_crop, :]
        else:
            diff = w - h
            left_crop = diff // 2
            right_crop = diff - left_crop
            return img[:, left_crop:w-right_crop]