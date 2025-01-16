import os
from generator.tifs_preprocessing.slice_image import ImageMaskSlicer
from generator.gan_arch.gan_generation import GANTrainer
from generator.gan_arch.gan_tester import GANTester
from generator.shifter.image_shifter import ImageShifter
from generator.visualize.inference_visualizer import InferenceVisualizer
from generator.data_load.inference_preprocessing import InferenceProcessor
from generator.gan_arch.gan_inference import ImageGenerator
from settings import ANALYSIS_OUTPUT_FOLDER_PATH, GEODATA_PATH, SOURCE_IMAGES_FOLDER_PATH, MASKS_FOLDER_PATH
from PIL import Image


# КОНСТАНТЫ ПУТЕЙ
OUTPUT_IMAGES_FOLDER_PATH = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'slice_analyze/output_images')
OUTPUT_MASKS_FOLDER_PATH = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'slice_analyze/output_masks')
TEST_IMAGES_FOLDER_PATH = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'zzz/test_imag')
TEST_MASKS_FOLDER_PATH = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'zzz/test_mask')
TEST_INFERENCE_IMAGE_PATH = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'unified_analyze/images/ridge_2_image.tif')
TEST_INFERENCE_MASK_PATH = os.path.join(MASKS_FOLDER_PATH, 'ridge_2_mask.png')
OUTPUT_TEST_INFERENCE_FOLDER_PATH = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'inference')
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

def gen_shift(image_size, x_shift, y_shift):
    processor = InferenceProcessor(image_size=image_size)
    generator = ImageGenerator(generator_weights_path=GENERATOR_PATH, image_size=image_size)
    shifter = ImageShifter(image_size=image_size)

    transformed_image, transformed_mask, original_sizes, mask_sizes = processor.preprocess_image(TEST_INFERENCE_IMAGE_PATH, TEST_INFERENCE_MASK_PATH)

    shifted_image, shifted_mask = shifter.apply_shift(
        image=transformed_image,
        mask=transformed_mask,
        x_shift_percent=x_shift,
        y_shift_percent=y_shift
    )

    generated_image, generated_mask = generator.generate(shifted_image, shifted_mask)

    combined_image, combined_mask = shifter.merge_image(transformed_image, transformed_mask, generated_image, generated_mask, original_sizes, mask_sizes, x_shift, y_shift, OUTPUT_TEST_INFERENCE_FOLDER_PATH)

    InferenceVisualizer.visualize(
        original_image=transformed_image.cpu(),
        original_mask=transformed_mask.cpu(),
        combined_image=combined_image.cpu(),
        combined_mask=combined_mask.cpu()
    )

def main():
    # slicer_image(50, 300)
        
    trainer = GANTrainer(OUTPUT_IMAGES_FOLDER_PATH, OUTPUT_MASKS_FOLDER_PATH, WEIGHTS_PATH, epochs=1, batch_size=8, target_image_size=448, lr_g=0.0001, lr_d=0.00001, load_weights=True)
    trainer.train()

    # tester = GANTester(TEST_IMAGES_FOLDER_PATH, TEST_MASKS_FOLDER_PATH, GENERATOR_PATH)
    # tester.visualize_results()

    # gen_shift(448, 10, 5)

if __name__ == "__main__":
    main()