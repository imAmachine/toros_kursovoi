import os
from generator.gan_generation import GANTrainer, GANTester
from settings import ANALYSIS_OUTPUT_FOLDER_PATH


def main():
    OUTPUT_IMAGES_FOLDER_PATH = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'slice_analyze/output_images')
    OUTPUT_MASKS_FOLDER_PATH = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'slice_analyze/output_masks')
    TEST_IMAGES_FOLDER_PATH = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'zzz/test_imag')
    TEST_MASKS_FOLDER_PATH = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'zzz/test_mask')
    generator_path = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'model_weight')
    GENERATOR_PATH = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'model_weight', 'generator.pth')

    # trainer = GANTrainer(OUTPUT_IMAGES_FOLDER_PATH, OUTPUT_MASKS_FOLDER_PATH, generator_path, epochs=1, batch_size=4, lr_g=0.0001, lr_d=0.0001, load_weights=True)
    # trainer.train()

    tester = GANTester(TEST_IMAGES_FOLDER_PATH, TEST_MASKS_FOLDER_PATH, GENERATOR_PATH)
    tester.visualize_results()

if __name__ == "__main__":
    main()