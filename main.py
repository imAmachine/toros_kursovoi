from src.datasets.processors import ShiftProcessor
from src.datasets.dataset import DatasetCreator
from src.gan.model import GenerativeModel
from src.gan.train import GANTrainer
from settings import GENERATOR_PATH, MASKS_FOLDER_PATH, AUGMENTED_DATASET_FOLDER_PATH, PREPROCESSED_MASKS_FOLDER_PATH, GENERATED_GAN_PATH, WEIGHTS_PATH


def main():
    gan = GenerativeModel(target_image_size=224, 
                          g_feature_maps=64, 
                          d_feature_maps=64)
    ds = DatasetCreator(generated_path=AUGMENTED_DATASET_FOLDER_PATH,
                        original_data_path=MASKS_FOLDER_PATH,
                        preprocessed_data_path=PREPROCESSED_MASKS_FOLDER_PATH,
                        images_extentions=['.png'],
                        model_transforms=gan.get_transforms(),
                        dataset_processor=ShiftProcessor(shift_percent=0.15),
                        preprocess=False,
                        generate_new=False)
    trainer = GANTrainer(model=gan, 
                         dataset_processor=ds,
                         output_path=WEIGHTS_PATH,
                         epochs=20000,
                         batch_size=6,
                         load_weights=True)

    trainer.train()

if __name__ == "__main__":
    main()