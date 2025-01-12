import os
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from generator.tiff_dataset import TIFDataset
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt


class GANTester:
    def __init__(self, test_image_path, test_mask_path, generator_path, batch_size=4):
        """
        Инициализация тестера
        """
        self.test_image_path = test_image_path
        self.test_mask_path = test_mask_path
        self.generator_path = generator_path
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_transforms()
        self._prepare_test_data()
        self._load_generator()

    def _initialize_transforms(self):
        """
        Определение трансформаций
        """
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def _prepare_test_data(self):
        """
        Подготовка тестового датасета
        """
        test_image_patches = [os.path.join(self.test_image_path, f) for f in os.listdir(self.test_image_path)]
        test_mask_patches = [os.path.join(self.test_mask_path, f) for f in os.listdir(self.test_mask_path)]
        self.test_dataset = TIFDataset(test_image_patches, test_mask_patches,
                                        image_transform=self.image_transform,
                                        mask_transform=self.mask_transform)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def _load_generator(self):
        """
        Загрузка модели генератора
        """
        self.generator = smp.Unet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        ).to(self.device)
        if os.path.exists(self.generator_path):
            self.generator.load_state_dict(torch.load(self.generator_path, map_location=self.device))
            print(f"Генератор успешно загружен с {self.generator_path}.")
        else:
            raise FileNotFoundError(f"Файл генератора {self.generator_path} не найден.")

    def visualize_results(self):
        """
        Визуализация результатов на тестовых данных
        """
        self.generator.eval()
        with torch.no_grad():
            for _, (full_images, noise_images, full_masks) in enumerate(self.test_dataloader):
                full_images, noise_images, full_masks = full_images.to(self.device), noise_images.to(self.device), full_masks.to(self.device)
                generated_masks = self.generator(noise_images)

                plt.figure(figsize=(20, 5))
                for i in range(min(5, len(full_images))):
                    plt.subplot(5, 4, i * 4 + 1)
                    plt.title("Original Image")
                    plt.imshow(self._denormalize(full_images[i].cpu(), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).permute(1, 2, 0).clip(0, 1))
                    plt.axis("off")

                    plt.subplot(5, 4, i * 4 + 2)
                    plt.title("Input with Noise")
                    plt.imshow(self._denormalize(noise_images[i].cpu(), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).permute(1, 2, 0).clip(0, 1))
                    plt.axis("off")

                    plt.subplot(5, 4, i * 4 + 3)
                    plt.title("Full Mask")
                    plt.imshow(full_masks[i, 0].cpu().numpy(), cmap="gray")
                    plt.axis("off")

                    plt.subplot(5, 4, i * 4 + 4)
                    plt.title("Generated Mask")
                    plt.imshow((generated_masks[i, 0].cpu().numpy() > 0.5).astype(int), cmap="gray")
                    plt.axis("off")

                plt.show()
                break

    def _denormalize(self, tensor, mean, std):
        """
        Деинормализация тензора для визуализации
        """
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor