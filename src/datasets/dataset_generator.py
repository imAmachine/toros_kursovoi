import os
import numpy as np
import cv2
from albumentations import (
    Compose,
    RandomRotate90,
    ShiftScaleRotate,
    RandomBrightnessContrast
)
from sklearn.model_selection import train_test_split
import numpy as np


class IceRidgeDatasetGenerator:
    def __init__(self, input_dir, output_dir, test_split=0.2, random_state=42):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.test_split = test_split
        self.random_state = random_state
        
        # Аугментации Albumentations для PyTorch
        self.augmentation = Compose([
            RandomRotate90(p=0.5),
            ShiftScaleRotate(
                shift_limit=0.2, 
                scale_limit=0.2, 
                rotate_limit=90, 
                p=0.5
            ),
        ])
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.masks = []
        self.filenames = []
        self.train_examples = []
        self.val_examples = []
    
    def load_masks(self):
        """Загрузка масок с использованием OpenCV и NumPy"""
        for filename in os.listdir(self.input_dir):
            if filename.endswith((".png", ".jpg", ".jpeg")):
                mask_path = os.path.join(self.input_dir, filename)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                if mask is not None:
                    # Бинаризация
                    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                    self.masks.append(mask)
                    self.filenames.append(filename)
        
        print(f"Загружено {len(self.masks)} исходных масок")
        return len(self.masks)
    
    def create_shift_variants(self, mask, shifts=8, shift_percent=0.15):
        """Создание вариантов сдвига для одной маски"""
        h, w = mask.shape
        shift_size_x = int(w * shift_percent)
        shift_size_y = int(h * shift_percent)
        
        variants = []
        directions = [
            (1, 0), (-1, 0),   # горизонталь
            (0, 1), (0, -1),   # вертикаль
            (1, 1), (-1, 1),   # диагонали
            (1, -1), (-1, -1)
        ][:shifts]
        
        for dx, dy in directions:
            # Создаем матрицу сдвига
            M = np.float32([[1, 0, dx*shift_size_x], [0, 1, dy*shift_size_y]])
            shifted = cv2.warpAffine(mask, M, (w, h), borderValue=0)
            
            # Создаем входную маску с "пустой" областью
            input_mask = mask.copy()
            if dx > 0: input_mask[:, w-shift_size_x:] = 0
            if dx < 0: input_mask[:, :shift_size_x] = 0
            if dy > 0: input_mask[h-shift_size_y:, :] = 0
            if dy < 0: input_mask[:shift_size_y, :] = 0
            
            variants.append((input_mask, shifted))
        
        return variants
    
    def generate_dataset(self, augmentations_per_example=10):
        """Генерация расширенного датасета"""
        all_examples = []
        
        for mask in self.masks:
            # Создаем варианты сдвига
            shift_variants = self.create_shift_variants(mask)
            
            for input_mask, target_mask in shift_variants:
                # Оригинальные примеры
                all_examples.append((input_mask, target_mask))
                
                # Аугментированные примеры
                for _ in range(augmentations_per_example):
                    # Аугментация входной и целевой масок синхронно
                    augmented = self.augmentation(image=input_mask, mask=target_mask)
                    all_examples.append((
                        augmented['image'], 
                        augmented['mask']
                    ))
        
        # Разделение на train/val
        train_examples, val_examples = train_test_split(
            all_examples, 
            test_size=self.test_split, 
            random_state=self.random_state
        )
        
        return train_examples, val_examples
    
    def save_dataset(self, train_examples, val_examples):
        """Сохранение подготовленного датасета"""
        train_dir = os.path.join(self.output_dir, 'train')
        val_dir = os.path.join(self.output_dir, 'val')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        # Сохранение тренировочных примеров
        for i, (input_mask, target_mask) in enumerate(train_examples):
            cv2.imwrite(os.path.join(train_dir, f'input_{i}.png'), input_mask)
            cv2.imwrite(os.path.join(train_dir, f'target_{i}.png'), target_mask)
        
        # Сохранение валидационных примеров
        for i, (input_mask, target_mask) in enumerate(val_examples):
            cv2.imwrite(os.path.join(val_dir, f'input_{i}.png'), input_mask)
            cv2.imwrite(os.path.join(val_dir, f'target_{i}.png'), target_mask)
        
        print(f"Датасет сохранен в {self.output_dir}")
    
    def prepare_pytorch_dataset(self, augmentations_per_example=10):
        """Подготовка датасета для PyTorch"""
        self.load_masks()
        train_examples, val_examples = self.generate_dataset(augmentations_per_example)
        self.save_dataset(train_examples, val_examples)
        
        return train_examples, val_examples