import torch
from torchvision.transforms import transforms
from .gan_components import AOTDiscriminator, AOTGenerator


class GANModel:
    def __init__(self, target_image_size=1024, g_feature_maps=32, d_feature_maps=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_image_size = target_image_size
        self.g_feature_maps = g_feature_maps
        self.d_feature_maps = d_feature_maps
        
        self.generator = AOTGenerator(feature_maps=self.g_feature_maps).to(self.device)
        self.discriminator = AOTDiscriminator(feature_maps=self.d_feature_maps).to(self.device)
        self.mask_transform = self._initialize_transforms()

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