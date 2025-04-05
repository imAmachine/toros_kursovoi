import torch
from torchvision.transforms import transforms
from .generator import AOTGenerator


class GANModel:
    def __init__(self, target_image_size=1024):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_image_size = target_image_size
        
        self.generator = None
        self.discriminator = None
        self.image_transform = None
        self.mask_transform = None
        
        self._build_models()
        self._initialize_transforms()

    def _initialize_transforms(self):
        self.mask_transform = transforms.Compose([
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
    
    def _build_models(self):
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()

    def _build_generator(self):
        return AOTGenerator().to(self.device)
    
    def _build_discriminator(self, input_channels=1, feature_maps=64):
        return torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, feature_maps, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2, inplace=True),
            
            torch.nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(feature_maps * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(feature_maps * 4),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(feature_maps * 4, feature_maps * 8, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(feature_maps * 8),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(feature_maps * 8, 1, kernel_size=4, stride=1, padding=0)
        ).to(self.device)