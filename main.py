import os
from generator.tifs_preprocessing.slice_image import ImageMaskSlicer
from generator.gan_arch.gan_generation import GANTrainer
from generator.gan_arch.gan_tester import GANTester
from settings import ANALYSIS_OUTPUT_FOLDER_PATH, GEODATA_PATH, SOURCE_IMAGES_FOLDER_PATH, MASKS_FOLDER_PATH


# КОНСТАНТЫ ПУТЕЙ
OUTPUT_IMAGES_FOLDER_PATH = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'slice_analyze/output_images')
OUTPUT_MASKS_FOLDER_PATH = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'slice_analyze/output_masks')
TEST_IMAGES_FOLDER_PATH = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'zzz/test_imag')
TEST_MASKS_FOLDER_PATH = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'zzz/test_mask')
WEIGHTS_PATH = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'model_weight')
GENERATOR_PATH = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'model_weight', 'generator.pth')

def slicer_image(grid_size, target_tiles_count):
    slicer = ImageMaskSlicer(
        geo_data_path=GEODATA_PATH,
        image_dir=SOURCE_IMAGES_FOLDER_PATH,
        mask_dir=MASKS_FOLDER_PATH,
        output_image_dir=OUTPUT_IMAGES_FOLDER_PATH,
        output_mask_dir=OUTPUT_MASKS_FOLDER_PATH,
        grid_size=grid_size,
        target_tiles_count=target_tiles_count
    )
    slicer.slice_all()

def main():
    slicer_image(50, 300)
        
    # trainer = GANTrainer(OUTPUT_IMAGES_FOLDER_PATH, OUTPUT_MASKS_FOLDER_PATH, WEIGHTS_PATH, epochs=5, batch_size=4, lr_g=0.0001, lr_d=0.0001, load_weights=True)
    # trainer.train()

    # tester = GANTester(TEST_IMAGES_FOLDER_PATH, TEST_MASKS_FOLDER_PATH, GENERATOR_PATH)
    # tester.visualize_results()

if __name__ == "__main__":
    main()