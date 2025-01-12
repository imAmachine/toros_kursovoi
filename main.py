import os
from generator.tifs_preprocessing.slice_image import ImageMaskSlicer
from generator.gan_arch.gan_generation import GANTrainer
from generator.gan_arch.gan_tester import GANTester
from settings import ANALYSIS_OUTPUT_FOLDER_PATH, SOURCE_IMAGES_FOLDER_PATH, MASKS_FOLDER_PATH


# КОНСТАНТЫ ПУТЕЙ
OUTPUT_IMAGES_FOLDER_PATH = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'slice_analyze/output_images')
OUTPUT_MASKS_FOLDER_PATH = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'slice_analyze/output_masks')
TEST_IMAGES_FOLDER_PATH = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'zzz/test_imag')
TEST_MASKS_FOLDER_PATH = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'zzz/test_mask')
WEIGHTS_PATH = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'model_weight')
GENERATOR_PATH = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'model_weight', 'generator.pth')

def slicer_image(images_path, masks_path, OUTPUT_IMAGES_FOLDER_PATH, OUTPUT_MASKS_FOLDER_PATH):
    slicer = ImageMaskSlicer(
        image_dir=images_path,
        mask_dir=masks_path,
        output_image_dir=OUTPUT_IMAGES_FOLDER_PATH,
        output_mask_dir=OUTPUT_MASKS_FOLDER_PATH,
        tile_size=3430,
        stride=477
    )
    slicer.slice_all()

def main():
    slicer_image(SOURCE_IMAGES_FOLDER_PATH, MASKS_FOLDER_PATH, OUTPUT_IMAGES_FOLDER_PATH, OUTPUT_MASKS_FOLDER_PATH)

    # trainer = GANTrainer(OUTPUT_IMAGES_FOLDER_PATH, OUTPUT_MASKS_FOLDER_PATH, WEIGHTS_PATH, epochs=5, batch_size=4, lr_g=0.0001, lr_d=0.0001, load_weights=True)
    # trainer.train()

    # tester = GANTester(TEST_IMAGES_FOLDER_PATH, TEST_MASKS_FOLDER_PATH, GENERATOR_PATH)
    # tester.visualize_results()

if __name__ == "__main__":
    main()