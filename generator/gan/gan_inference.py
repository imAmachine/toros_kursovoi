from PIL import Image
import torch
import segmentation_models_pytorch as smp
import os
import logging

from .gan_arch import GANModel
from ..shifter.image_shifter import ImageShifter
from torchvision.transforms.functional import to_pil_image

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
    
    def generate(self, image, mask, horizontal_shift, vertical_shift, output_path):
        """
        Генерация изображения и маски с использованием генератора.
        :param image: Исходное изображение.
        :param mask: Исходная маска.
        :param horizontal_shift: Сдвиг по горизонтали (в процентах).
        :param vertical_shift: Сдвиг по вертикали (в процентах).
        :param output_path: Путь для сохранения сгенерированных изображений.
        :return: Сгенерированное изображение и маска.
        """
        # Трансформации
        image_n = self.gan.image_transform(image)
        mask_n = self.gan.mask_transform(mask)
        
        # Сдвиг + шум
        img_shifter = ImageShifter(self.image_size)
        shifted_img, shifted_mask = img_shifter.apply_shift(image_n, mask_n, horizontal_shift, vertical_shift)
        
        shifted_input = torch.cat([shifted_img, shifted_mask], dim=0).unsqueeze(0).to(self.device)
        
        # Генерация
        with torch.no_grad():
            generated_output = self.gan.generator(shifted_input)
        
        generated_image = generated_output[:, 0:1, :, :]
        generated_mask = generated_output[:, 1:2, :, :]
        
        # Преобразование в PIL
        generated_image_pil = to_pil_image(generated_image.squeeze(0).cpu())
        generated_mask_pil = to_pil_image(generated_mask.squeeze(0).cpu())
        
        # Сохранение изображений
        generated_image_pil.save(f"{output_path}/generated_image.png", format="PNG")
        generated_mask_pil.save(f"{output_path}/generated_mask.png", format="PNG")
        
        return generated_image, generated_mask