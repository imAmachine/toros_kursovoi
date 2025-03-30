import os
from src.datasets.dataset_generator import IceRidgeDatasetGenerator
from src.gan.gan_arch import GANModel
from src.gan.gan_inference import GANInference
from src.gan.gan_trainer import GANTrainer
from src.processing.rotate_mask import RotateMask
from settings import ANALYSIS_OUTPUT_FOLDER_PATH, MASKS_FOLDER_PATH

# КОНСТАНТЫ ПУТЕЙ
OUTPUT_MASKS_FOLDER_PATH = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'output')
WEIGHTS_PATH = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'model_weight/weights')
GENERATOR_PATH = os.path.join(WEIGHTS_PATH, 'generator.pth')

def train(gan_model):
    # Подготовка данных
    generator = IceRidgeDatasetGenerator(
        input_dir=MASKS_FOLDER_PATH, 
        output_dir=OUTPUT_MASKS_FOLDER_PATH
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
    inference = GANInference(gan_model, "./outputs/generator.pth")
    mask = inference.infer(os.path.join(MASKS_FOLDER_PATH, "ridge_2_mask.png"), "output.png")

def rotate_mask():
    processor = RotateMask(crop_percent=0, kernel_size=11)
    processor.process_folder(MASKS_FOLDER_PATH, OUTPUT_MASKS_FOLDER_PATH)
    

def main():
    # gan_model = GANModel()
    # train(gan_model=gan_model)
    # inference(gan_model=gan_model)
    rotate_mask()
    

if __name__ == "__main__":
    main()