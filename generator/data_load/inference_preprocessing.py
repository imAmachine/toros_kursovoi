from torchvision.transforms import transforms
from PIL import Image

class InferenceProcessor:
    def __init__(self, image_size):
        """
        Класс для обработки изображений и масок.
        :param image_size: Размер изображения для входа в генератор.
        """
        self.image_size = image_size
        self.image_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0.1).float())
        ])

    def preprocess_image(self, image_path, mask_path):
        Image.MAX_IMAGE_PIXELS = None
        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")
        orig_sizes = image.size
        mask_sizes = mask.size
        return self.image_transform(image), self.mask_transform(mask), orig_sizes, mask_sizes