import os
import torch
import albumentations as A

from src.preprocessing.processors import *

# путь к файлу с геоанализом исходных снимков
GEODATA_PATH = "./data/geo_data.csv"

# путь к корневой директории для обработанных данных
OUTPUT_FOLDER_PATH = "./data/processed_output/"

# пути к директориям для масок
MASKS_FOLDER_PATH = "./data/masks/" # исходные маски
PREPROCESSED_MASKS_FOLDER_PATH = os.path.join(OUTPUT_FOLDER_PATH, "preprocessed") # предобработанные входные маски
AUGMENTED_DATASET_FOLDER_PATH = os.path.join(OUTPUT_FOLDER_PATH, 'augmented_dataset') # обработанные
GENERATED_GAN_PATH = os.path.join(OUTPUT_FOLDER_PATH, 'generated')

# пути к весам модели
WEIGHTS_PATH = os.path.join(OUTPUT_FOLDER_PATH, 'model_weight/weights')
GENERATOR_PATH = os.path.join(WEIGHTS_PATH, 'generator.pth')



# ================ настройка основных параметров ==================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
AUGMENTATIONS = A.Compose([
            A.RandomRotate90(p=0.7),
            A.ElasticTransform(alpha=120, sigma=6, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
            # A.RandomCrop(height=512, width=512, p=0.5),
        ])

PREPROCESSORS = [
            Binarize(),
            # CropProcessor(crop_percent=5), # кроп по краям в процентном соотношении
            # EnchanceProcessor(morph_kernel_size=3), # улучшает маску с помощью морфинга
            RotateMaskProcessor(angle_choose_type=AngleChooseType.CONSISTENT), # поворот масок к исходному углу
            AutoAdjust(),
            Unbinarize()
        ]

MASKS_FILE_EXTENSIONS = ['.png']