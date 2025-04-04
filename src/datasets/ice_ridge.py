import os
import cv2
import albumentations as A

class IceRidgeDatasetGenerator:
    def __init__(self, 
                 generated_out_path,
                 albumentations_pipeline: A,
                 augmentations_per_image=10):  # Количество аугментаций для каждого изображения
        
        self.generated_out_path = generated_out_path
        self.augmentation_pipeline = albumentations_pipeline
        self.augmentations_per_image = augmentations_per_image

    def generate(self, metadata):
        augmented_results = self._augmentate_folder(metadata)
        return augmented_results
    
    def _augmentate_folder(self, preprocessed_metadata: dict):
        """Генерирует несколько вариантов аугментаций для каждого изображения"""
        os.makedirs(self.generated_out_path, exist_ok=True)
        
        results = {}
        
        for src, metadata in preprocessed_metadata.items():
            print(f'Аугментация файла {src}')
            path = metadata.get('output_path')
            
            if not os.path.exists(path):
                continue
            
            # Загрузка изображения
            image = cv2.imread(path)
            if image is None:
                print(f"Не удалось загрузить изображение: {path}")
                continue
            
            filename = os.path.basename(path)
            base_name, ext = os.path.splitext(filename)
            
            augmented_paths = []
            
            # Создаем несколько вариантов аугментаций
            for i in range(self.augmentations_per_image):
                augmented_image = self.augmentation_pipeline(image=image)['image']
                
                # Сохранение результата
                output_path = os.path.join(self.generated_out_path, f"{base_name}_aug{i+1}{ext}")
                cv2.imwrite(output_path, augmented_image)
                
                augmented_paths.append(output_path)
            
            # Сохранение метаданных
            results[src] = {
                'original_path': path,
                'augmented_paths': augmented_paths,
                'metadata': metadata
            }
        
        return results