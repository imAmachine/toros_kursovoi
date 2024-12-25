import os
from generator.gan_generation import GANTrainer
from settings import ANALYSIS_OUTPUT_FOLDER_PATH


def main():
    OUTPUT_IMAGES_FOLDER_PATH = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'slice_analyze/output_images')
    OUTPUT_MASKS_FOLDER_PATH = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'slice_analyze/output_masks')
    generator_path = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'model_weight')

    trainer = GANTrainer(OUTPUT_IMAGES_FOLDER_PATH, OUTPUT_MASKS_FOLDER_PATH, generator_path, epochs=2, batch_size=4, lr_g=0.0001, lr_d=0.00005, load_weights=True)
    trainer.train()

if __name__ == "__main__":
    main()