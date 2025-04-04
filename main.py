import os
from src.preprocessing import CropProcessor, EnchanceProcessor, RotateMaskProcessor, MasksPreprocessor, AngleChooseType, FractalDimensionProcessor
from settings import GENERATOR_PATH, MASKS_FOLDER_PATH, GENERATED_MASKS_FOLDER_PATH, PREPROCESSED_MASKS_FOLDER_PATH

def init_preprocessor():
    preprocessor = MasksPreprocessor()
    preprocessor.add_processors(processors=[
        EnchanceProcessor(morph_kernel_size=7), # улучшает маску с помощью морфинга
        RotateMaskProcessor(angle_choose_type=AngleChooseType.CONSISTENT), # поворот масок к исходному углу
        CropProcessor(crop_percent=5), # кроп по краям в процентном соотношении
        FractalDimensionProcessor() # вычисление фрактальной размерности
    ])
    return preprocessor

def main():
    # обработка всех входных масок 
    metadata = init_preprocessor().process_folder(MASKS_FOLDER_PATH, PREPROCESSED_MASKS_FOLDER_PATH, ['.png'])

if __name__ == "__main__":
    main()