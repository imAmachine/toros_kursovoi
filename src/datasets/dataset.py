import os
import cv2
import numpy as np
from typing import Dict, List
import shutil

class IceRidgeDatasetCreator:
    def __init__(self,
                 damage_ratio: float = 0.2,
                 shift_percent: float = 0.15,
                 max_damage_objects: int = 3):
        self.damage_ratio = damage_ratio
        self.shift_percent = shift_percent
        self.max_damage_objects = max_damage_objects
        self.directions = ['top', 'bottom', 'left', 'right']

    def create_dataset(self,
                      source_dir: str,
                      output_dir: str,
                      metadata: Dict) -> Dict:
        """
        Создает датасет с триплетами файлов на основе аугментированных изображений
        """
        os.makedirs(output_dir, exist_ok=True)
        new_metadata = {}
        sample_counter = 0

        # Проход по всем аугментированным файлам
        for orig_name, orig_meta in metadata.items():
            orig_path = orig_meta['output_path']
            
            # Загрузка оригинального изображения
            orig_image = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
            if orig_image is None:
                continue

            # Генерация поврежденной версии и маски
            damaged_image, damage_mask = self._generate_damaged_version(orig_image)
            
            # Сохранение триплета
            triplet_meta = self._save_triplet(
                output_dir=output_dir,
                sample_id=sample_counter,
                orig_image=orig_image,
                damaged_image=damaged_image,
                damage_mask=damage_mask,
                orig_meta=orig_meta
            )
            
            new_metadata.update(triplet_meta)
            sample_counter += 1

        return new_metadata

    def _generate_damaged_version(self, image: np.ndarray) -> tuple:
        """Генерирует изображение с удаленной областью и маску повреждений"""
        h, w = image.shape
        direction = np.random.choice(self.directions)
        
        # Создаем копию оригинального изображения
        damaged = image.copy()
        damage_mask = np.zeros_like(image)
        
        # Вычисляем размер поврежденной области
        damage_size = int(max(h, w) * self.shift_percent)

        num_damages = np.random.randint(1, self.max_damage_objects+1)
        
        if direction == 'top':
            # Удаляем верхнюю часть (заполняем черным)
            damaged[:damage_size, :] = 0
            damage_mask[:damage_size, :] = 255
            
        elif direction == 'bottom':
            # Удаляем нижнюю часть
            damaged[-damage_size:, :] = 0
            damage_mask[-damage_size:, :] = 255
            
        elif direction == 'left':
            # Удаляем левую часть
            damaged[:, :damage_size] = 0
            damage_mask[:, :damage_size] = 255
            
        elif direction == 'right':
            # Удаляем правую часть
            damaged[:, -damage_size:] = 0
            damage_mask[:, -damage_size:] = 255

        for _ in range(num_damages):
            # Размеры повреждения
            max_size = int(min(h, w) * self.damage_ratio)
            w_dmg = np.random.randint(max_size//2, max_size)
            h_dmg = np.random.randint(max_size//2, max_size)
            
            # Позиция повреждения (гарантия в границах)
            x = np.random.randint(0, w - w_dmg)
            y = np.random.randint(0, h - h_dmg)
            
            # Применяем повреждение
            damaged[y:y+h_dmg, x:x+w_dmg] = 0
            damage_mask[y:y+h_dmg, x:x+w_dmg] = 255

        return damaged, damage_mask

    def _save_triplet(self,
                    output_dir: str,
                    sample_id: int,
                    orig_image: np.ndarray,
                    damaged_image: np.ndarray,
                    damage_mask: np.ndarray,
                    orig_meta: Dict) -> Dict:
        """Сохраняет три файла для каждого примера"""
        base_name = f"sample_{sample_id}"
        file_meta = {
            'input': os.path.join(output_dir, f"{base_name}_input.png"),
            'target': os.path.join(output_dir, f"{base_name}_target.png"),
            'damage': os.path.join(output_dir, f"{base_name}_damage.png"),
            'fractal_dimension': orig_meta['fractal_dimension']
        }

        cv2.imwrite(file_meta['input'], damaged_image)
        cv2.imwrite(file_meta['target'], orig_image)
        cv2.imwrite(file_meta['damage'], damage_mask)

        return {base_name: file_meta}

    def split_dataset(self,
                     input_dir: str,
                     train_dir: str,
                     val_dir: str,
                     test_size: float = 0.2):
        """Разделение датасета на train/val"""
        # Реализация разделения файлов по папкам
        pass