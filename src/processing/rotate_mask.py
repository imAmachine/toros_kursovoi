import cv2
import numpy as np
import os

class RotateMask:
    def __init__(self, crop_percent=10, kernel_size=7):
        self.crop_percent = crop_percent
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        self.current_angle = 0
        
    def process_folder(self, input_folder, output_folder):
        rotate_dir = os.path.join(output_folder, 'rotate_mask')
        os.makedirs(rotate_dir, exist_ok=True)
        for filename in os.listdir(input_folder):
            if filename.lower().endswith('.png'):
                self.process_file(input_folder, rotate_dir, filename)
    
    def process_file(self, input_folder, output_folder, filename):
        try:
            img = self._load_image(input_folder, filename)
            processed = self._preprocess(img)
            angle = self._calculate_rotation_angle(processed)
            angle = self._normalize_angle(angle)
            self.current_angle = angle
            rotated = self._rotate_image(img, angle)
            cropped = self._crop_image(rotated)
            self._save_image(cropped, output_folder, filename)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

    def _normalize_angle(self, angle):
        """Приводит угол к диапазону [-90, 90) градусов"""
        angle %= 180  # Приводим к диапазону [0, 180)
        if angle > 90:
            return angle - 180
        elif angle < -90:
            return angle + 180
        return angle

    def _load_image(self, folder, filename):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Failed to load image")
        return img

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
        cnt = max(contours, key=cv2.contourArea)
        rect_angle = cv2.minAreaRect(cnt)[-1]

        # Выбор оптимального угла
        return pca_angle if abs(pca_angle) > abs(rect_angle) else rect_angle

    def _rotate_image(self, img, angle):
        h, w = img.shape
        center = (w//2, h//2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

    def _crop_image(self, img):
        # Обрезка по процентам
        if self.crop_percent > 0:
            h, w = img.shape
            crop_size = int(min(w, h) * self.crop_percent / 100)
            img = img[crop_size:-crop_size, crop_size:-crop_size]
        
        # Обрезка до bounding box
        coords = cv2.findNonZero(img)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            return img[y:y+h, x:x+w]
        return img

    def _save_image(self, img, folder, filename):
        base_name, ext = os.path.splitext(filename)
        angle_str = f"{int(round(self.current_angle))}"
        new_filename = f"{base_name}_deg_{angle_str}{ext}"

        cv2.imwrite(os.path.join(folder, new_filename), img)