import os
import torch
from torchinfo import summary
import time
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose, transforms
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

        width, height = image.size
        left = width // 4
        top = height // 4
        right = 3 * width // 4
        bottom = 3 * height // 4
        center_crop_image = image.crop((left, top, right, bottom))
        center_crop_mask = mask.crop((left, top, right, bottom))
        if self.image_transform:
            full_image = self.image_transform(image)
            center_crop_image = self.image_transform(center_crop_image)
        else:
            full_image = ToTensor()(image)
        if self.mask_transform:
            full_mask = self.mask_transform(mask)
            center_crop_mask = self.mask_transform(center_crop_mask)
        else:
            full_mask = ToTensor()(mask)
        return full_image, center_crop_image, full_mask

class GANTrainer:
    def __init__(self, image_path, mask_path, output_path, epochs, batch_size, lr_g, lr_d):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_path = image_path
        self.mask_path = mask_path
        self.output_path = output_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_g = lr_g
        self.lr_d = lr_d
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
            classes=1
        ).to(self.device)
        self.discriminator = smp.Unet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=4,
            classes=1
        ).to(self.device)
        for idx, (name, layer) in enumerate(self.generator.encoder.named_children()):
            if idx < 7:
                for param in layer.parameters():
                    param.requires_grad = False
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr_g)
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_d)
        self.loss_fn_g = torch.nn.BCEWithLogitsLoss()
        self.loss_fn_d = torch.nn.BCEWithLogitsLoss()
        self.fid_metric = FrechetInceptionDistance(feature=2048).to(self.device)

    def train(self):

        def _denormalize(tensor, mean, std):
            for t, m, s in zip(tensor, mean, std):
                t.mul_(s).add_(m)
            return tensor

        for epoch in range(self.epochs):
            self.generator.train()
            self.discriminator.train()

            epoch_loss_g = 0
            epoch_loss_d = 0
            start_time = time.time()

            for full_images, center_crop_images, masks in self.dataloader:
                full_images, center_crop_images, masks = full_images.to(self.device), center_crop_images.to(self.device), masks.to(self.device)

                generated_masks = self.generator(center_crop_images)

                d_input_real = torch.cat((center_crop_images, masks), dim=1)
                d_input_fake = torch.cat((center_crop_images, generated_masks.detach()), dim=1)

                real_labels = torch.ones((d_input_real.size(0), 1), dtype=torch.float32, device=self.device)
                fake_labels = torch.zeros((d_input_fake.size(0), 1), dtype=torch.float32, device=self.device)

                real_output = torch.mean(self.discriminator(d_input_real), dim=(2, 3))
                fake_output = torch.mean(self.discriminator(d_input_fake), dim=(2, 3))

                loss_d_real = self.loss_fn_d(real_output, real_labels)
                loss_d_fake = self.loss_fn_d(fake_output, fake_labels)
                loss_d = (loss_d_real + loss_d_fake) / 2

                self.optimizer_d.zero_grad()
                loss_d.backward()
                self.optimizer_d.step()

                d_input_fake_for_g = torch.cat((center_crop_images, generated_masks), dim=1)
                fake_output_for_g = torch.mean(self.discriminator(d_input_fake_for_g), dim=(2, 3))

                loss_g = self.loss_fn_g(generated_masks, masks) + 0.001 * self.loss_fn_d(fake_output_for_g, real_labels)
                self.optimizer_g.zero_grad()
                loss_g.backward()
                self.optimizer_g.step()

                epoch_loss_g += loss_g.item()
                epoch_loss_d += loss_d.item()

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Epoch {epoch + 1}/{self.epochs}, Generator Loss: {epoch_loss_g:.4f}, Discriminator Loss: {epoch_loss_d:.4f}, Time: {elapsed_time:.2f} sec")
            self._save_models()

            self.generator.eval()
            self.fid_metric.reset()
            with torch.no_grad():
                for full_images, center_crop_images, masks in self.dataloader:
                    full_images, center_crop_images, masks = full_images.to(self.device), center_crop_images.to(
                        self.device), masks.to(self.device)

                    generated_masks = self.generator(center_crop_images)
                    binary_generated_masks = (torch.sigmoid(generated_masks) > 0.5).float()
                    binary_generated_masks = binary_generated_masks.repeat(1, 3, 1, 1)  # Добавляем каналы

                    denorm_images = _denormalize(center_crop_images.clone(), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    denorm_images = (denorm_images * 255).byte()  # Преобразование в uint8
                    binary_generated_masks = (binary_generated_masks * 255).byte()  # Преобразование в uint8

                    resized_images = torch.stack([self.fid_transform(img) for img in denorm_images])
                    resized_generated_masks = torch.stack([self.fid_transform(mask) for mask in binary_generated_masks])

                    self.fid_metric.update(resized_images, real=True)
                    self.fid_metric.update(resized_generated_masks, real=False)

                fid_score = self.fid_metric.compute()
                print(f"FID Score на эпохе {epoch + 1}: {fid_score:.4f}")

    def _save_models(self):
        generator_path = os.path.join(self.output_path, f'generator.pth')
        discriminator_path = os.path.join(self.output_path, f'discriminator.pth')
        torch.save(self.generator.state_dict(), generator_path)
        torch.save(self.discriminator.state_dict(), discriminator_path)