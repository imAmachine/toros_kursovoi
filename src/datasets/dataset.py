import os
import cv2
import numpy as np
from typing import Dict, List
import shutil

from preprocessing.interfaces import IProcessor

class IceRidgeDataset:
    def __init__(self, dataset_processor: IProcessor = None):
        self.processor = dataset_processor

    def load_dataset(self, source_dir: str, output_dir: str, metadata: Dict) -> Dict:
        """
        Создает датасет с триплетами файлов на основе аугментированных изображений
        """
        os.makedirs(output_dir, exist_ok=True)
        new_metadata = {}
        sample_counter = 0

        # Проход по всем аугментированным файлам
        for _, orig_meta in metadata.items():
            orig_path = orig_meta['output_path']
            
            # Загрузка оригинального изображения
            orig_image = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
            if orig_image is None: continue

            # Генерация поврежденной версии и маски
            damaged_image, damage_mask = self.processor.process(orig_image)
            
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