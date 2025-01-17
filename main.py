import os
from generator.gan.gan_arch import GANModel
from generator.gan.gan_inference import ImageGenerator
from generator.gan.gan_trainer import GANTrainer
from generator.tifs_preprocessing.slice_image import ImageMaskSlicer
from generator.shifter.image_shifter import ImageShifter
from generator.visualize.inference_visualizer import InferenceVisualizer
from settings import ANALYSIS_OUTPUT_FOLDER_PATH, GEODATA_PATH, SOURCE_IMAGES_FOLDER_PATH, MASKS_FOLDER_PATH
from PIL import Image


# КОНСТАНТЫ ПУТЕЙ
OUTPUT_IMAGES_FOLDER_PATH = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'slice_analyze/output_images')
OUTPUT_MASKS_FOLDER_PATH = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'slice_analyze/output_masks')
TEST_IMAGES_FOLDER_PATH = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'zzz/test_imag')
TEST_MASKS_FOLDER_PATH = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'zzz/test_mask')
TEST_INFERENCE_IMAGE_PATH = os.path.join(SOURCE_IMAGES_FOLDER_PATH, 'ridge_2_image.tif')
TEST_INFERENCE_MASK_PATH = os.path.join(MASKS_FOLDER_PATH, 'ridge_2_mask.png')
OUTPUT_TEST_INFERENCE_FOLDER_PATH = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'inference')
WEIGHTS_PATH = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'model_weight/weights')
GENERATOR_PATH = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'model_weight/weights', 'generator.pth')

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

def gen_shift(image, mask, image_size, x_shift, y_shift):
    generator = ImageGenerator(generator_weights_path=GENERATOR_PATH, image_size=image_size)
    shifter = ImageShifter(image_size=image_size)  

    Image.MAX_IMAGE_PIXELS = None
        
    # открытие файлов
    image = Image.open(TEST_INFERENCE_IMAGE_PATH).convert("L")
    mask = Image.open(TEST_INFERENCE_MASK_PATH)
    
    os.makedirs(OUTPUT_TEST_INFERENCE_FOLDER_PATH, exist_ok=True)
    transformed_image, transformed_mask, generated_image, generated_mask, img_nodate = generator.generate(image, mask, x_shift, y_shift, OUTPUT_TEST_INFERENCE_FOLDER_PATH)

    combined_image, combined_mask = shifter.merge_image(transformed_image, transformed_mask,
                                                        generated_image, generated_mask, img_nodate,
                                                        original_sizes=[448, 448], mask_sizes=[448, 448],
                                                        x_shift_percent=x_shift, y_shift_percent=y_shift,
                                                        output_path=OUTPUT_TEST_INFERENCE_FOLDER_PATH)

    InferenceVisualizer.visualize(
        original_image=transformed_image.cpu(),
        original_mask=transformed_mask.cpu(),
        combined_image=combined_image,
        combined_mask=combined_mask
    )

def main():
    # slicer_image(50, 300)
    # gan_model = GANModel(target_image_size=448)
    # trainer = GANTrainer(gan_model, OUTPUT_IMAGES_FOLDER_PATH, OUTPUT_MASKS_FOLDER_PATH, WEIGHTS_PATH, epochs=30, batch_size=16, lr_g=0.00015, lr_d=0.00001, load_weights=True)
    # trainer.train()

    # tester = GANTester(TEST_IMAGES_FOLDER_PATH, TEST_MASKS_FOLDER_PATH, GENERATOR_PATH)
    # tester.visualize_results()

    gen_shift(TEST_INFERENCE_IMAGE_PATH, TEST_INFERENCE_MASK_PATH, 448, -10, -5)

if __name__ == "__main__":
    main()