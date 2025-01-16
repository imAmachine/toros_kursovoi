import os
import torch
import torch.nn.functional as F
import time
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchmetrics.classification import Dice
from ..data_load.tiff_dataset import TIFDataset
import segmentation_models_pytorch as smp
from tqdm import tqdm
import os


class GANTrainer:
    def __init__(self, image_path, mask_path, output_path, epochs, batch_size, target_image_size, lr_g, lr_d,
                 load_weights=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_path = image_path
        self.mask_path = mask_path
        self.output_path = output_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.target_image_size = target_image_size
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.load_weights = load_weights

        os.makedirs(self.output_path, exist_ok=True)
        self._initialize_transforms()
        self._prepare_data()
        self._build_models()

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

    def _prepare_data(self):
        dataset = TIFDataset(image_dir=self.image_path, mask_dir=self.mask_path, image_transform=self.image_transform,
                             mask_transform=self.mask_transform)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def _build_models(self):
        self.generator = smp.Unet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=2,
            classes=2,
            activation='sigmoid'
        ).to(self.device)
        self.discriminator = self._build_patchgan_discriminator()

        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr_g)
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_d)

        self.loss_fn_g = Dice()
        self.loss_fn_d = torch.nn.BCEWithLogitsLoss()

        if self.load_weights:
            self._load_models()

    def _build_patchgan_discriminator(self):
        return torch.nn.Sequential(
            torch.nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1),
            torch.nn.Sigmoid()
        ).to(self.device)

    def _load_models(self):
        self.generator_path = os.path.join(self.output_path, "generator.pth")
        self.discriminator_path = os.path.join(self.output_path, "discriminator.pth")
        if os.path.exists(self.generator_path):
            self.generator.load_state_dict(torch.load(self.generator_path, map_location=self.device))
            print("Веса генератора загружены.")
        if os.path.exists(self.discriminator_path):
            self.discriminator.load_state_dict(torch.load(self.discriminator_path, map_location=self.device))
            print("Веса дискриминатора загружены.")

    def _denormalize(self, tensor, mean, std):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor

    def train(self):
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            progress_bar = tqdm(self.dataloader, desc=f"Training Epoch {epoch + 1}", leave=False)

            dataset_iter = iter(self.dataloader)
            samples = next(dataset_iter)

            for i in range(4):  # Визуализация первых 4 примеров из батча
                noisy_input = samples[0][i].cpu()
                real_combined = samples[1][i].cpu()
                mask = samples[2][i].cpu()

                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                axs[0].imshow(noisy_input[0], cmap='gray')
                axs[0].set_title('Noisy Input (Image + Mask)')
                axs[1].imshow(real_combined[0], cmap='gray')
                axs[1].set_title('Ground Truth Combined')
                axs[2].imshow(mask[0], cmap='gray')
                axs[2].set_title('Mask')
                plt.show()

            for noisy_input, real_combined, original_mask in progress_bar:
                noisy_input = noisy_input.to(self.device)
                real_combined = real_combined.to(self.device)
                original_mask = original_mask.to(self.device)

                # Генерация данных
                generated_combined = self.generator(noisy_input)
                generated_mask = generated_combined[:, 1:2, :, :]

                # Обучение дискриминатора
                real_output = self.discriminator(real_combined)
                fake_output = self.discriminator(generated_combined.detach())

                # Метки с "разглаживанием" (label smoothing)
                real_labels = torch.ones_like(real_output) * 0.9 + \
                              torch.rand_like(real_output) * 0.1
                fake_labels = torch.zeros_like(fake_output) + \
                              torch.rand_like(fake_output) * 0.1

                loss_d_real = self.loss_fn_d(real_output, real_labels)
                loss_d_fake = self.loss_fn_d(fake_output, fake_labels)
                loss_d = (loss_d_real + loss_d_fake) * 0.5

                self.optimizer_d.zero_grad()
                loss_d.backward()
                self.optimizer_d.step()

                # Обучение генератора
                fake_output_g = self.discriminator(generated_combined)
                loss_g_adv = self.loss_fn_d(fake_output_g, torch.ones_like(fake_output_g))

                # Потери на маску
                loss_g_preserve = F.binary_cross_entropy(
                    generated_mask * original_mask,
                    original_mask
                )

                loss_g = loss_g_adv + 10.0 * loss_g_preserve

                self.optimizer_g.zero_grad()
                loss_g.backward()
                self.optimizer_g.step()

                # Обновление прогресс-бара
                progress_bar.set_postfix({
                    "Loss_D": loss_d.item(),
                    "Loss_G": loss_g.item(),
                    "Loss_Preserve": loss_g_preserve.item()
                })

        # Сохранение моделей после обучения
        self._save_models()

    def _save_models(self):
        torch.save(self.generator.state_dict(), os.path.join(self.output_path, f"generator.pth"))
        torch.save(self.discriminator.state_dict(), os.path.join(self.output_path, f"discriminator.pth"))