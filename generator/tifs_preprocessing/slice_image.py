import os
import numpy as np
from tqdm import tqdm
from PIL import Image

class ImageMaskSlicer:
    def __init__(self, image_dir, mask_dir, output_image_dir, output_mask_dir, tile_size=4096, stride=2048):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.output_image_dir = output_image_dir
        self.output_mask_dir = output_mask_dir
        self.tile_size = tile_size
        self.stride = stride

        Image.MAX_IMAGE_PIXELS = None

        os.makedirs(self.output_image_dir, exist_ok=True)
        os.makedirs(self.output_mask_dir, exist_ok=True)

    def slice_all(self):
        image_files = [os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir) if f.endswith(".tif")]
        mask_files = [os.path.join(self.mask_dir, f) for f in os.listdir(self.mask_dir) if f.endswith(".png")]

        assert len(image_files) == len(mask_files), "Количество изображений и масок должно совпадать!"

        for image_path, mask_path in zip(image_files, mask_files):
            self._slice_image_and_mask(image_path, mask_path)

    def _slice_image_and_mask(self, image_path, mask_path):
        """
        Разрезает изображение и маску с учётом сдвига (stride) и исключает патчи с нулевыми значениями.
        Маска предварительно масштабируется до размера изображения.
        :param image_path: Путь к исходному изображению (TIF).
        :param mask_path: Путь к маске (PNG).
        """
        image = Image.open(image_path).convert("RGB")  # Преобразуем изображение в RGB
        mask = Image.open(mask_path).resize(image.size, Image.NEAREST)

        img_width, img_height = image.size
        saved_patches = 0

        for i in tqdm(range(0, img_width - self.tile_size + 1, self.stride), desc=f"Processing {os.path.basename(image_path)}"):
            for j in range(0, img_height - self.tile_size + 1, self.stride):
                box = (i, j, i + self.tile_size, j + self.tile_size)
                image_patch = image.crop(box)
                mask_patch = mask.crop(box)

                if self._has_zero_pixels(image_patch):
                    continue

                base_name = os.path.splitext(os.path.basename(image_path))[0]
                image_patch_name = f"{base_name}_x{i}_y{j}.png"
                mask_patch_name = f"{base_name}_x{i}_y{j}.png"

                image_patch.save(os.path.join(self.output_image_dir, image_patch_name))
                mask_patch.save(os.path.join(self.output_mask_dir, mask_patch_name))

                saved_patches += 1

        print(f"Разрезка завершена для: {image_path} и {mask_path}. Сохранено патчей: {saved_patches}")

    @staticmethod
    def _has_zero_pixels(patch):
        """
        Проверяет, содержит ли патч хотя бы один нулевой пиксель (только для RGB).
        :param patch: Патч изображения (PIL Image, RGB).
        :return: True, если в патче есть хотя бы один нулевой пиксель, иначе False.
        """
        patch_array = np.array(patch)
        zero_pixels = (patch_array[:, :, :3] == 0).any()

        return zero_pixels