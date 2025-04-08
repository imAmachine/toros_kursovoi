import os
import albumentations as A
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
import random

from datasets.processors import ShiftProcessor
from preprocessing.utils import ImageProcess

from preprocessing.preprocessor import MasksPreprocessor
from preprocessing.processors import AngleChooseType, CropProcessor, EnchanceProcessor, RotateMaskProcessor
from datasets.dataset import IceRidgeDataset, IceRidgeDatasetGenerator
from datasets.processors import ShiftProcessor
from torch.utils.data import DataLoader
import cv2
import json

class IceRidgeDataset(Dataset):
    def __init__(self, metadata: Dict, dataset_processor: ShiftProcessor = None, with_target=False, transforms=None):
        self.processor = dataset_processor
        self.with_target = with_target
        self.metadata = metadata
        self.transforms = transforms
        self.image_keys = list(metadata.keys())
    
    def __len__(self) -> int:
        return len(self.image_keys)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Возвращает тройку (input, target, damage_mask) для индекса idx"""
        
        image = self._read_image(self.image_keys[idx]) # чтение итерируемой картинки
        binary_image = ImageProcess.img_to_binary_format(image) # приведение в бинарный формат
        damaged, mask = self._get_processed_pair(binary_image) # обработка с помощью процессора
        transformed_tensors = self.apply_transforms((damaged, binary_image, mask)) # применение конечных трансформаций
        
        return transformed_tensors
    
    def _read_image(self, metadata_key) -> np.ndarray:
        orig_meta = self.metadata[metadata_key]
        orig_path = orig_meta.get('output_path')
        return cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
    
    def _get_processed_pair(self, input_img):
        return self.processor.process(input_img)
    
    def apply_transforms(self, objects: List):
        return [self.transforms(obj) for obj in objects]
        
    
    @staticmethod
    def split_dataset(metadata: Dict, val_ratio=0.2, seed=42) -> Tuple[Dict, Dict]:
        """Разделяет метаданные на обучающую и валидационную выборки"""
        random.seed(seed)
        
        all_keys = list(metadata.keys())
        random.shuffle(all_keys)
        
        val_size = int(len(all_keys) * val_ratio)
        
        val_keys = all_keys[:val_size]
        train_keys = all_keys[val_size:]
        
        train_metadata = {k: metadata[k] for k in train_keys}
        val_metadata = {k: metadata[k] for k in val_keys}
        
        return train_metadata, val_metadata


class IceRidgeDatasetGenerator:
   def __init__(self, 
                albumentations_pipeline,
                augmentations_per_image=10):
       self.augmentation_pipeline = albumentations_pipeline
       self.augmentations_per_image = augmentations_per_image

   def generate(self, generated_out_path, metadata):
       return self._augmentate_folder(generated_out_path, metadata)
   
   def _augmentate_image(self, file_name, file_metadata, generated_out_path):
       print(f'Аугментация файла {file_name}')
       path = file_metadata.get('output_path')
       
       if not os.path.exists(path):
           return {}
       
       image = cv2.imread(path)
       if image is None:
           print(f"Не удалось загрузить изображение: {path}")
           return {}
       
       filename = os.path.basename(path)
       base_name, ext = os.path.splitext(filename)
       
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


class DatasetProcessor:
    def __init__(self, generated_path, original_data_path, preprocessed_data_path, images_extentions, model_transforms, preprocess=True, generate_new=True):
        self.preprocessor = MasksPreprocessor()
        self.dataset_generator = IceRidgeDatasetGenerator()
        
        self.generated_metadata_path = generated_path
        self.input_data_path = original_data_path
        self.preprocessed_data_path = preprocessed_data_path
        self.images_extentions = images_extentions
        
        self.generate_new = generate_new
        self.preprocess = preprocess
        self.model_transforms = model_transforms
        
        self.preprocessed_metadata_json_path = os.path.join(self.generated_metadata_path, 'metadata.json')
        self.generated_metadata_json_path = os.path.join(self.generated_metadata_path, 'metadata_generated.json')
    
    def _init_preprocessor(self):
        self.preprocessor.add_processors(processors=[
            EnchanceProcessor(morph_kernel_size=7), # улучшает маску с помощью морфинга
            RotateMaskProcessor(angle_choose_type=AngleChooseType.CONSISTENT), # поворот масок к исходному углу
            CropProcessor(crop_percent=5), # кроп по краям в процентном соотношении
            #FractalDimensionProcessor() # вычисление фрактальной размерности
        ])
    
    def _init_generator(self):
        self.dataset_generator.augmentation_pipeline = A.Compose([
            A.RandomRotate90(p=0.5),
            A.ElasticTransform(alpha=120, sigma=6, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
            A.RandomCrop(height=512, width=512, p=0.5),
            # A.Resize(height=1024, width=1024, interpolation=cv2.INTER_LANCZOS4, p=1)
        ])
    
    def _preprocess_pipeline(self):
        self._init_preprocessor()
        metadata = self.preprocessor.process_folder(self.input_data_path, 
                                                    self.preprocessed_data_path, 
                                                    self.images_extentions)
        self.to_json(metadata, self.preprocessed_metadata_json_path)
   
    def _generation_pipeline(self):
        self._init_generator()
        preprocessed_metadata = self.from_json(self.preprocessed_metadata_json_path)
        generated_metadata = self.dataset_generator.generate(self.generated_metadata_path, preprocessed_metadata)
        self.to_json(generated_metadata, self.generated_metadata_json_path)
    
    def _create_dataset(self, dataset_processor):
        dataset_metadata = self.from_json(self.generated_metadata_json_path)
        
        dataset = IceRidgeDataset(metadata=dataset_metadata, 
                                  dataset_processor=dataset_processor, 
                                  transforms=self.model_transforms)
        
        return dataset
    
    def _create_dataloaders(self, dataset_metadata):
        train_metadata, val_metadata = dataset_metadata.split_dataset(self.dataset.metadata, val_ratio=0.2)
        
        train_dataset = IceRidgeDataset(train_metadata, 
                                        dataset_processor=self.dataset.processor, 
                                        with_target=False,
                                        transforms=self.model_transforms)
        val_dataset = IceRidgeDataset(val_metadata, 
                                      dataset_processor=self.dataset.processor, 
                                      with_target=False,
                                      transforms=self.model_transforms)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        return train_loader, val_loader
    
    def get_dataloaders(self):
        if self.preprocess:
            self._preprocess_pipeline()
        
        if self.generate_new:
            self._generation_pipeline()
            
        dataset_metadata = self._create_dataset(ShiftProcessor(shift_percent=0.15))
        
        return self._create_dataloaders(dataset_metadata)
    
    def to_json(self, metadata, path):
        with open(path, 'w+', encoding='utf8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
    
    def from_json(self, path):
        with open(path, 'w+', encoding='utf8') as f:
            return json.load(f)