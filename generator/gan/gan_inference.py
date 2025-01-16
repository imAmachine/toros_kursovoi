from PIL import Image
import torch
import segmentation_models_pytorch as smp
import os
import logging

from generator.gan.gan_arch import GANModel
from generator.shifter.image_shifter import ImageShifter

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImageGenerator:
    def __init__(self, generator_weights_path, image_size):
        """
        Класс для генерации изображения на основе сдвига.
        :param generator_weights_path: Путь к весам генератора.
        :param image_size: Размер изображения для входа в генератор.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        
        self.gan = GANModel(target_image_size=image_size)
        self.gan.load_weights(generator_weights_path, None)
        self.gan.generator.eval()
        
        logging.info("Генератор успешно загружен.")
    
    def generate(self, image_path, mask_path, horizontal_shift, vertical_shift):
        """
        Генерация изображения и маски с использованием генератора.
        :param shifted_image: Сдвинутое изображение.
        :param shifted_mask: Сдвинутая маска.
        :return: Сгенерированное изображение и маска.
        """
        Image.MAX_IMAGE_PIXELS = None
        
        # открытие файлов
        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path)
        
        # трансформации
        image_n = self.gan.image_transform(image)
        mask_n = self.gan.mask_transform(mask)
        
        # сдвиг + шум
        img_shifter = ImageShifter(self.image_size)
        shifted_img, shifted_mask = img_shifter.apply_shift(image_n, mask_n, horizontal_shift, vertical_shift)
        
        shifted_input = torch.cat([shifted_img, shifted_mask], dim=0).unsqueeze(0).to(self.device)
        
        # генерация
        with torch.no_grad():
            generated_output = self.generator(shifted_input)
        return generated_output[:, 0:1, :, :], generated_output[:, 1:2, :, :]