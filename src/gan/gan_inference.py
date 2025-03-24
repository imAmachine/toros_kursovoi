import os
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from PIL import Image
from .gan_arch import GANModel

class GANInference:
    def __init__(self, gan_model: GANModel, model_weights_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = gan_model
        self._init_transforms()
        
        self.model.load_weights(model_weights_path)
        self.model.generator.eval()

    def _init_transforms(self):
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((self.model.target_image_size, self.model.target_image_size)),
            transforms.ToTensor()
        ])
    
    def infer(self, image_path, save_path=None):
        image = Image.open(image_path).convert("L")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output_mask = self.model.generator(input_tensor)
            output_mask = torch.sigmoid(output_mask).squeeze(0).cpu().numpy()
        
        # Визуализация
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title("Исходное изображение")
        plt.axis("off")
        
        plt.subplot(1, 2, 2)
        plt.imshow(output_mask[0], cmap='gray')
        plt.title("Сгенерированная маска")
        plt.axis("off")
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
        return output_mask