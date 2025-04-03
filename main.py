import os
from src.preprocessing import CropProcessor, EnchanceProcessor, RotateMaskProcessor, MasksPreprocessor
from settings import GENERATOR_PATH, MASKS_FOLDER_PATH, GENERATED_MASKS_FOLDER_PATH, PREPROCESSED_MASKS_FOLDER_PATH

def preprocess_data(input_folder, output_folder):
    preprocessor = MasksPreprocessor()
    preprocessor.add_processors(processors=[
        EnchanceProcessor(morph_kernel_size=7), # улучшает маску с помощью морфинга
        RotateMaskProcessor(), # поворот масок к исходному углу
        # CropProcessor(crop_percent=5) # кроп по краям в процентном соотношении
    ])
    
    # обработка всех входных масок 
    preprocessor.process_folder(input_folder, output_folder)

def main():
    preprocess_data(input_folder=MASKS_FOLDER_PATH, 
                    output_folder=PREPROCESSED_MASKS_FOLDER_PATH)

if __name__ == "__main__":
    main()