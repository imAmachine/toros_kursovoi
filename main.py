import os

import cv2
from src.datasets.ice_ridge import IceRidgeDatasetGenerator
from src.preprocessing import CropProcessor, EnchanceProcessor, RotateMaskProcessor, MasksPreprocessor, AngleChooseType, FractalDimensionProcessor
from settings import GENERATOR_PATH, MASKS_FOLDER_PATH, GENERATED_MASKS_FOLDER_PATH, PREPROCESSED_MASKS_FOLDER_PATH

import albumentations as A

def init_preprocessor():
    preprocessor = MasksPreprocessor()
    preprocessor.add_processors(processors=[
        EnchanceProcessor(morph_kernel_size=7), # улучшает маску с помощью морфинга
        RotateMaskProcessor(angle_choose_type=AngleChooseType.CONSISTENT), # поворот масок к исходному углу
        CropProcessor(crop_percent=5), # кроп по краям в процентном соотношении
        FractalDimensionProcessor() # вычисление фрактальной размерности
    ])
    return preprocessor

def gen_dataset():
    augmentation_pipeline = A.Compose([
        # Геометрические преобразования
        A.RandomRotate90(p=0.5),
        
        # Структурные преобразования (с умеренными параметрами)
        A.ElasticTransform(
            alpha=120, 
            sigma=6, 
            p=0.3
        ),
        
        A.GridDistortion(
            num_steps=5, 
            distort_limit=0.2, 
            p=0.3
        ),
        
        # Обрезка и изменение размера для симуляции разных масштабов съемки
        A.RandomCrop(height=512, width=512, p=0.5),
        A.Resize(height=1024, width=1024, interpolation=cv2.INTER_LANCZOS4, p=1)
    ])
    
    return IceRidgeDatasetGenerator(GENERATED_MASKS_FOLDER_PATH, augmentation_pipeline)

def main():
    # обработка всех входных масок 
    metadata = init_preprocessor().process_folder(MASKS_FOLDER_PATH, PREPROCESSED_MASKS_FOLDER_PATH, ['.png'])
    dataset_generator = gen_dataset().generate(metadata)

if __name__ == "__main__":
    main()