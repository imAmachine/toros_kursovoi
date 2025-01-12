import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from generator.data_load.test_dataset import TIFDataset
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt


class GANTester:
    def __init__(self, test_image_path, test_mask_path, generator_path, batch_size=4, target_image_size=4096):
        """
        Инициализация тестера
        """
        self.test_image_path = test_image_path
        self.test_mask_path = test_mask_path
        self.generator_path = generator_path
        self.target_image_size = target_image_size
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        os.makedirs(test_image_path, exist_ok=True)
        os.makedirs(test_mask_path, exist_ok=True)
        self._initialize_transforms()
        self._prepare_test_data()
        self._load_generator()

    def _initialize_transforms(self):
        self.image_transform = transforms.Compose([
            transforms.Resize((self.target_image_size, self.target_image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Нормализация для одноканальных данных
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.target_image_size, self.target_image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0.5).float())  # Преобразование в бинарную маску
        ])

    def _prepare_test_data(self):
        dataset = TIFDataset(image_dir=self.test_image_path, mask_dir=self.test_mask_path, image_transform=self.image_transform, mask_transform=self.mask_transform)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def _load_generator(self):
        """
        Загрузка модели генератора
        """
        self.generator = smp.Unet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=2,
            classes=2,
            activation='sigmoid'
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
        import matplotlib.gridspec as gridspec
        self.generator.eval()
        with torch.no_grad():
            for _, (full_images, noise_images, full_masks) in enumerate(self.dataloader):
                full_images, noise_images, full_masks = full_images.to(self.device), noise_images.to(self.device), full_masks.to(self.device)
                generated_masks = self.generator(noise_images)

                fig = plt.figure(figsize=(20, 12))
                gs = gridspec.GridSpec(1, 4, figure=fig)

                for i in range(min(5, len(full_images))):
                    ax1 = fig.add_subplot(gs[i, 0])
                    ax1.set_title("Original Image")
                    ax1.imshow(self._denormalize(full_images[i, 0].cpu(), mean=[0.485], std=[0.229]).squeeze().clip(0, 1), cmap='gray')
                    ax1.axis("off")

                    ax2 = fig.add_subplot(gs[i, 1])
                    ax2.set_title("Input with Noise")
                    ax2.imshow(self._denormalize(noise_images[i, 0].cpu(), mean=[0.485], std=[0.229]).squeeze().clip(0, 1), cmap='gray')
                    ax2.axis("off")

                    ax3 = fig.add_subplot(gs[i, 2])
                    ax3.set_title("Full Mask")
                    ax3.imshow(full_masks[i, 0].cpu().numpy(), cmap="gray")
                    ax3.axis("off")

                    ax4 = fig.add_subplot(gs[i, 3])
                    ax4.set_title("Generated Mask")
                    ax4.imshow((generated_masks[i, 0].cpu().numpy() > 0.5).astype(int), cmap="gray")
                    ax4.axis("off")

                plt.tight_layout()
                plt.show()
                break


    def _denormalize(self, tensor, mean, std):
        """
        Деинормализация тензора для визуализации
        """
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor
