import torch
import segmentation_models_pytorch as smp
import os
import logging

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

        # Загрузка модели генератора
        self.generator = smp.Unet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=2,
            classes=2,
            activation="sigmoid"
        ).to(self.device)
        self.generator.load_state_dict(torch.load(generator_weights_path, map_location=self.device))
        self.generator.eval()
        logging.info("Генератор успешно загружен.")

    def generate(self, shifted_image, shifted_mask):
        """
        Генерация изображения и маски с использованием генератора.
        :param shifted_image: Сдвинутое изображение.
        :param shifted_mask: Сдвинутая маска.
        :return: Сгенерированное изображение и маска.
        """
        shifted_input = torch.cat([shifted_image, shifted_mask], dim=0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            generated_output = self.generator(shifted_input)
        return generated_output[:, 0:1, :, :], generated_output[:, 1:2, :, :]