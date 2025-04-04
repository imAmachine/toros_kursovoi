import albumentations.augmentations as A
from preprocessing import MasksPreprocessor
from analyzer.fractal_funcs import DataAnalyzer

class IceRidgeDatasetGenerator:
    def __init__(self, 
                 input_folder_path,
                 generated_out_path,
                 preprocessed_metadata: dict,
                 albumentations_pipeline: A):
        
        self.input_folder_path = input_folder_path # путь к исходным данным
        self.generated_out_path = generated_out_path # путь к выходной директории
        
        self.preprocessed_metadata = preprocessed_metadata
        self.augmentations = albumentations_pipeline

    def generate(self):
        pass
        # 1. generate data на основе preprocessed инпутов с помощью augmentations
        
        
    def augmentation_files(self, input_files_metadata: dict):
        pass