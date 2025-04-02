import os
from src.datasets.dataset_generator import IceRidgeDatasetGenerator
from src.gan.gan_arch import GANModel
from src.gan.gan_inference import GANInference
from src.gan.gan_trainer import GANTrainer
from src.processing.rotate_mask import RotateMask
from settings import GENERATOR_PATH, MASKS_FOLDER_PATH, GENERATED_MASKS_FOLDER_PATH, PREPROCESSED_MASKS_FOLDER_PATH

# КОНСТАНТЫ ПУТЕЙ
TEST_INFERENCE_PATH = os.path.join(GENERATED_MASKS_FOLDER_PATH)

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
    mask = inference.infer(os.path.join(MASKS_FOLDER_PATH, "ridge_2_mask.png"), "output.png")

def rotate_mask():
    """метод для восстановления угла поворота масок"""
    rotated_path = os.path.join(PREPROCESSED_MASKS_FOLDER_PATH, "rotated")
    processor = RotateMask(crop_percent=0, kernel_size=11, postprocess_kernel_size=1)
    processor.process_folder(MASKS_FOLDER_PATH, rotated_path)
    

def main():
    # gan_model = GANModel()
    # train(gan_model=gan_model)
    # inference(gan_model=gan_model)
    rotate_mask()
    

if __name__ == "__main__":
    main()