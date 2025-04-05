import os
import cv2
from typing import Dict
import os
import cv2

class IceRidgeDatasetGenerator:
   def __init__(self, 
                albumentations_pipeline,
                augmentations_per_image=10):
       self.augmentation_pipeline = albumentations_pipeline
       self.augmentations_per_image = augmentations_per_image

   def generate(self, generated_out_path, metadata):
       return self._augmentate_folder(generated_out_path, metadata)
   
   def get_filedata(self, path):
       if not os.path.exists(path):
           return {}
       
       image = cv2.imread(path)
       if image is None:
           print(f"Не удалось загрузить изображение: {path}")
           return {}
       
       filename = os.path.basename(path)
       base_name, ext = os.path.splitext(filename)
       
       return image, base_name, ext
   
   def _augmentate_image(self, file_name, file_metadata: Dict, generated_out_path: str):
       print(f'Аугментация файла {file_name}')
       path = file_metadata.get('output_path')
       image, base_name, ext = self.get_filedata(path)
       
       new_metadata_dict = {}
       
       for i in range(self.augmentations_per_image):
           augmented_image = self.augmentation_pipeline(image=image)['image']
           
           # Сохранение результата
           output_path = os.path.join(generated_out_path, f"{base_name}_aug{i+1}{ext}")
           new_metadata_dict[f"{base_name}_aug{i+1}"] = {
               'fractal_dimension': file_metadata.get('fractal_dimension'),
               'output_path': output_path
           }
           cv2.imwrite(output_path, augmented_image)
       
       return new_metadata_dict
   
   def _augmentate_folder(self, generated_out_path: Dict, preprocessed_metadata: Dict):
       """Генерирует несколько вариантов аугментаций для каждого изображения"""
       os.makedirs(generated_out_path, exist_ok=True)
       
       results_metadata = {}
       for src, metadata in preprocessed_metadata.items():
           aug_metadata = self._augmentate_image(src, metadata, generated_out_path)
           results_metadata.update(aug_metadata)
           
       return results_metadata