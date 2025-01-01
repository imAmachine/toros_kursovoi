import os
import torch
from torchinfo import summary
import time
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, transforms
from PIL import Image
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from torchmetrics.image.fid import FrechetInceptionDistance

class TIFDataset(Dataset):
    def __init__(self, image_paths, mask_paths, image_transform=None, mask_transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Загрузка и обработка изображения и маски"""
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        if self.image_transform:
            full_image = self.image_transform(image)
        else:
            full_image = ToTensor()(image)

        noise_image = torch.randn_like(full_image)
        _, h, w = full_image.shape
        ch, cw = h // 2, w // 2
        top, left = h // 4, w // 4
        bottom, right = top + ch, left + cw
        noise_image[:, top:bottom, left:right] = full_image[:, top:bottom, left:right]

        if self.mask_transform:
            full_mask = self.mask_transform(mask)
        else:
            full_mask = ToTensor()(mask)

        return full_image, noise_image, full_mask

class GANTrainer:
    def __init__(self, image_path, mask_path, output_path, epochs, batch_size, lr_g, lr_d, load_weights=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_path = image_path
        self.mask_path = mask_path
        self.output_path = output_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.load_weights = load_weights
        self._initialize_transforms()
        self._prepare_data()
        self._build_models()

    def _initialize_transforms(self):
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.fid_transform = transforms.Compose([
            transforms.Resize((299, 299))
        ])

    def _prepare_data(self):
        image_patches = [os.path.join(self.image_path, f) for f in os.listdir(self.image_path)]
        mask_patches = [os.path.join(self.mask_path, f) for f in os.listdir(self.mask_path)]
        dataset = TIFDataset(image_patches, mask_patches, image_transform=self.image_transform, mask_transform=self.mask_transform)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def _build_models(self):
        self.generator = smp.Unet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1  # Генератор создаёт одноканальную бинарную маску
        ).to(self.device)

        self.discriminator = torch.nn.Sequential(
            torch.nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0),
            torch.nn.AdaptiveAvgPool2d((1, 1)),  # Сводим к (N, 1, 1, 1)
            torch.nn.Flatten(),  # Преобразуем в (N, 1)
            torch.nn.Sigmoid()
        ).to(self.device)

        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr_g)
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_d)
        self.loss_fn_g = torch.nn.BCEWithLogitsLoss()
        self.loss_fn_d = torch.nn.BCEWithLogitsLoss()

        if self.load_weights:
            self._load_models()

    def _load_models(self):
        generator_path = os.path.join(self.output_path, "generator.pth")
        discriminator_path = os.path.join(self.output_path, "discriminator.pth")
        if os.path.exists(generator_path):
            self.generator.load_state_dict(torch.load(generator_path, map_location=self.device))
            print("Веса генератора загружены.")
        if os.path.exists(discriminator_path):
            self.discriminator.load_state_dict(torch.load(discriminator_path, map_location=self.device))
            print("Веса дискриминатора загружены.")

    def _denormalize(self, tensor, mean, std):
        """Denormalize tensor for visualization."""
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor

    def visualize(self):
        """Visualize the original images, inputs, and generated masks."""

        self.generator.eval()
        with torch.no_grad():
            full_images, noise_images, full_masks = next(iter(self.dataloader))
            full_images, noise_images, full_masks = full_images.to(self.device), noise_images.to(self.device), full_masks.to(self.device)
            generated_masks = self.generator(noise_images)

            plt.figure(figsize=(20, 5))

            plt.subplot(1, 4, 1)
            plt.title("Original Image")
            plt.imshow(self._denormalize(full_images[0].cpu(), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).permute(1, 2, 0).clip(0, 1))
            plt.axis("off")

            plt.subplot(1, 4, 2)
            plt.title("Input with Noise")
            plt.imshow(self._denormalize(noise_images[0].cpu(), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).permute(1, 2, 0).clip(0, 1))
            plt.axis("off")

            plt.subplot(1, 4, 3)
            plt.title("Full Mask")
            plt.imshow(full_masks[0, 0].cpu().detach().numpy(), cmap="gray")
            plt.axis("off")

            plt.subplot(1, 4, 4)
            plt.title("Generated Mask")
            plt.imshow((generated_masks[0, 0].cpu().detach().numpy() > 0.5).astype(int), cmap="gray")
            plt.axis("off")

            plt.show()

    def train(self):

        for epoch in range(self.epochs):
            self.generator.train()
            self.discriminator.train()

            epoch_loss_g = 0
            epoch_loss_d = 0
            start_time = time.time()

            for full_images, noise_images, full_masks in self.dataloader:
                full_images, noise_images, full_masks = \
                    full_images.to(self.device), noise_images.to(self.device), full_masks.to(self.device)

                generated_masks = self.generator(noise_images)

                # Update discriminator
                real_labels = torch.ones((full_masks.size(0), 1), dtype=torch.float32, device=self.device)
                fake_labels = torch.zeros((full_masks.size(0), 1), dtype=torch.float32, device=self.device)

                real_input = torch.cat((full_masks, full_masks), dim=1)  # Исходная маска + оригинал
                fake_input = torch.cat((generated_masks.detach(), full_masks), dim=1)  # Сгенерированное + оригинал

                real_output = self.discriminator(real_input)
                fake_output = self.discriminator(fake_input)

                loss_d_real = self.loss_fn_d(real_output, real_labels)
                loss_d_fake = self.loss_fn_d(fake_output, fake_labels)
                loss_d = (loss_d_real + loss_d_fake) / 2

                self.optimizer_d.zero_grad()
                loss_d.backward()
                self.optimizer_d.step()

                # Update generator
                fake_output_for_g = self.discriminator(torch.cat((generated_masks, full_masks), dim=1))
                loss_g = self.loss_fn_d(fake_output_for_g, real_labels)

                self.optimizer_g.zero_grad()
                loss_g.backward()
                self.optimizer_g.step()

                epoch_loss_g += loss_g.item()
                epoch_loss_d += loss_d.item()

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Epoch {epoch + 1}/{self.epochs}, Generator Loss: {epoch_loss_g:.4f}, Discriminator Loss: {epoch_loss_d:.4f}, Time: {elapsed_time:.2f} sec")

            self._save_models()

        self.visualize()

    def _save_models(self):
        torch.save(self.generator.state_dict(), os.path.join(self.output_path, f"generator.pth"))
        torch.save(self.discriminator.state_dict(), os.path.join(self.output_path, f"discriminator.pth"))