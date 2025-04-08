from datasets.processors import ShiftProcessor
from datasets.dataset import DatasetProcessor, IceRidgeDataset
from gan.model import GenerativeModel
from gan.train import GANTrainer
from settings import GENERATOR_PATH, MASKS_FOLDER_PATH, AUGMENTED_DATASET_FOLDER_PATH, PREPROCESSED_MASKS_FOLDER_PATH, GENERATED_GAN_PATH, WEIGHTS_PATH


def main():
    gan = GenerativeModel(target_image_size=224, 
                          g_feature_maps=32, 
                          d_feature_maps=16)
    ds = DatasetProcessor(generated_path=AUGMENTED_DATASET_FOLDER_PATH,
                        original_data_path=MASKS_FOLDER_PATH,
                        preprocessed_data_path=PREPROCESSED_MASKS_FOLDER_PATH,
                        images_extentions=['.png'],
                        model_transforms=gan.get_transforms(),
                        preprocess=True,
                        generate_new=True)
    trainer = GANTrainer(model=gan, 
                         output_path=GENERATED_GAN_PATH,
                         epochs=10,
                         batch_size=2)
    
    
    
    

    trainer.train()

if __name__ == "__main__":
    main()