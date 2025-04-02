import os
from typing import Any, Dict, List
import cv2
import numpy as np

from processing.interfaces.MaskProcessor import MaskProcessor


class MasksPreprocessor:
    """Основной класс для препроцессинга масок"""
    
    def __init__(self):
        self.processors: List[MaskProcessor] = []
    
    def add_processor(self, processor: MaskProcessor) -> None:
        """Добавляет процессор в пайплайн обработки"""
        self.processors.append(processor)
    
    def _process_image(self, image: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
        """Обрабатывает одно изображение через все процессоры"""
        metadata = {}
        current_image = image.copy()
        
        for processor in self.processors:
            current_image, proc_metadata = processor.process(current_image, metadata)
            metadata.update(proc_metadata)
        
        return current_image, metadata
    
    def process_file(self, input_path: str, output_folder: str, filename: str) -> Dict[str, Any]:
        """Обрабатывает один файл и сохраняет результат"""
        try:
            # Загрузка изображения
            img = cv2.imread(os.path.join(input_path, filename), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Failed to load image {filename}")
            
            # Обработка изображения
            processed_img, metadata = self._process_image(img)
            
            # Формирование имени файла с метаданными
            base_name, ext = os.path.splitext(filename)
            if 'rotation_angle' in metadata:
                angle_str = f"{int(round(metadata['rotation_angle']))}"
                new_filename = f"{base_name}_deg_{angle_str}{ext}"
            else:
                new_filename = f"{base_name}_processed{ext}"
            
            # Сохранение результата
            output_path = os.path.join(output_folder, new_filename)
            cv2.imwrite(output_path, processed_img)
            
            # Добавление информации о пути к выходному файлу
            metadata['output_path'] = output_path
            
            return metadata
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            return {'error': str(e)}
    
    def process_folder(self, input_folder: str, output_folder: str, mask_extensions: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """Обрабатывает все файлы в папке и сохраняет результаты"""
        if mask_extensions is None:
            mask_extensions = ['.png', '.jpg', '.jpeg']
            
        # Создание выходной папки, если она не существует
        os.makedirs(output_folder, exist_ok=True)
        
        results = {}
        
        # Обработка всех файлов в папке
        for filename in os.listdir(input_folder):
            ext = os.path.splitext(filename)[1].lower()
            if ext in mask_extensions:
                results[filename] = self.process_file(input_folder, output_folder, filename)
        
        return results