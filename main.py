import json
import os
import cv2
from src.gan.arch.gan import GANModel
from src.gan.train.train_worker import GANTrainer
from src.datasets.processors.shift_damage_processor import ShiftProcessor
from src.datasets.ice_ridge_dataset_generator import IceRidgeDatasetGenerator
from src.datasets.dataset import IceRidgeDataset
from src.preprocessing import CropProcessor, EnchanceProcessor, RotateMaskProcessor, MasksPreprocessor, AngleChooseType, FractalDimensionProcessor
from settings import GENERATOR_PATH, MASKS_FOLDER_PATH, AUGMENTED_DATASET_FOLDER_PATH, PREPROCESSED_MASKS_FOLDER_PATH, GENERATED_GAN_PATH, WEIGHTS_PATH

import albumentations as A

def init_preprocessor():
    preprocessor = MasksPreprocessor()
    preprocessor.add_processors(processors=[
        EnchanceProcessor(morph_kernel_size=7), # улучшает маску с помощью морфинга
        RotateMaskProcessor(angle_choose_type=AngleChooseType.CONSISTENT), # поворот масок к исходному углу
        CropProcessor(crop_percent=5), # кроп по краям в процентном соотношении
        #FractalDimensionProcessor() # вычисление фрактальной размерности
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
    
    return IceRidgeDatasetGenerator(augmentation_pipeline)

def prepare_data():
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
        return IceRidgeDatasetGenerator(augmentation_pipeline)
    
    metadata = init_preprocessor().process_folder(MASKS_FOLDER_PATH, PREPROCESSED_MASKS_FOLDER_PATH, ['.png'])
    generated_metadata = gen_dataset().generate(AUGMENTED_DATASET_FOLDER_PATH, metadata)
    metadata_json_path = os.path.join(AUGMENTED_DATASET_FOLDER_PATH, 'metadata_dump.json')
    
    with open(metadata_json_path, 'w+', encoding='utf8') as f:
        json.dump(generated_metadata, f, indent=4, ensure_ascii=False)

    dataset = IceRidgeDataset(metadata=generated_metadata, 
                              dataset_processor=ShiftProcessor(shift_percent=0.15))
    return dataset


def main():
    dataset = prepare_data()
    gan = GANModel(target_image_size=224, g_feature_maps=64, d_feature_maps=64)
    
    trainer = GANTrainer(model=gan,
                         dataset=dataset,
                         output_path=WEIGHTS_PATH,
                         epochs=1,
                         batch_size=2,
                         load_weights=False)
    trainer.train()

if __name__ == "__main__":
    main()