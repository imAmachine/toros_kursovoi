import os
from settings import GENERATOR_PATH, MASKS_FOLDER_PATH, GENERATED_MASKS_FOLDER_PATH, PREPROCESSED_MASKS_FOLDER_PATH


from src.preprocessing.processors.rotate_mask_processor import RotateMaskProcessor
from src.preprocessing.mask_preprocessor import MasksPreprocessor

from src.datasets.dataset_generator import IceRidgeDatasetGenerator

from src.gan.gan_arch import GANModel
from src.gan.gan_inference import GANInference
from src.gan.gan_trainer import GANTrainer


# ДОПОЛНИТЕЛЬНЫЕ КОНСТАНТЫ ПУТЕЙ
# ...

def train(gan_model):
    # Подготовка данных
    generator = IceRidgeDatasetGenerator(
        input_dir=MASKS_FOLDER_PATH, 
        output_dir=GENERATED_MASKS_FOLDER_PATH
    )
    
    # Получаем примеры для обучения
    train_examples, val_examples = generator.prepare_pytorch_dataset(
        augmentations_per_example=15
    )
    
    trainer = GANTrainer(
        model=gan_model, 
        train_examples=train_examples,
        val_examples=val_examples,
        output_path=GENERATOR_PATH,
        epochs=10, 
        batch_size=8,
        lr_g=0.0002, 
        lr_d=0.0002,
        noise_type='gaussian',
        noise_level=0.05
    )
    trainer.train()

def inference(gan_model):
    inference = GANInference(gan_model, GENERATOR_PATH)
    
    # тест
    # mask = inference.infer(os.path.join(MASKS_FOLDER_PATH, "ridge_2_mask.png"), "output.png")

def preprocess_data(input_folder, output_folder):
    # Создание препроцессора входных данных
    preprocessor = MasksPreprocessor()
    
    # Добавление процессоров для предобработки масок
    preprocessor.add_processors(processors=[
        RotateMaskProcessor(crop_percent=5, kernel_size=7, postprocess_kernel_size=4), # поворот масок к исходному углу
    ])
    
    # обработка всех входных масок 
    _ = preprocessor.process_folder(input_folder, output_folder)

def main():
    preprocess_data(input_folder=MASKS_FOLDER_PATH, 
                    output_folder=PREPROCESSED_MASKS_FOLDER_PATH)

if __name__ == "__main__":
    main()