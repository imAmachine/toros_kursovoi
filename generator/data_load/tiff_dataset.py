from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from PIL import Image
import torch


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
        ch, cw = int(h * 0.65), int(w * 0.65)
        top, left = (h - ch) // 2, (w - cw) // 2
        bottom, right = top + ch, left + cw
        noise_image[:, top:bottom, left:right] = full_image[:, top:bottom, left:right]

        if self.mask_transform:
            full_mask = self.mask_transform(mask)
        else:
            full_mask = ToTensor()(mask)

        return full_image, noise_image, full_mask