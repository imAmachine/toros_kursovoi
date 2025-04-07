import torch
from torchvision.transforms import transforms


class GANModel:
    def __init__(self, generator, discriminator, device, target_image_size=256):
        self.device = device
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.target_image_size = target_image_size
        self.transforms = self._initialize_transforms()

    def _initialize_transforms(self):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.target_image_size, self.target_image_size)),
            transforms.ToTensor()
        ])

    def denormalize(self, tensor, mean, std):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor

    def load_weights(self, weights_gen_path=None, weights_discr_path=None):
        if weights_gen_path:
            self.generator.load_state_dict(torch.load(weights_gen_path, map_location=self.device, weights_only=True))
        if weights_discr_path:
            self.discriminator.load_state_dict(torch.load(weights_discr_path, map_location=self.device, weights_only=True))
