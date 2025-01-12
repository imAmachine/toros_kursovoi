import os
import numpy as np
from tqdm import tqdm
from osgeo import gdal
from PIL import Image

class ImageMaskSlicer:
    def __init__(self, image_dir, mask_dir, output_image_dir, output_mask_dir, tile_size=4096, stride=2048):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.output_image_dir = output_image_dir
        self.output_mask_dir = output_mask_dir
        self.tile_size = tile_size
        self.stride = stride

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
        image_ds = gdal.Open(image_path)

        img_width = image_ds.RasterXSize
        img_height = image_ds.RasterYSize

        # Масштабируем маску до размеров изображения
        mask = Image.open(mask_path)
        mask = mask.resize((img_width, img_height), Image.NEAREST)
        mask_array = np.array(mask)

        saved_patches = 0

        for i in tqdm(range(0, img_width - self.tile_size + 1, self.stride), desc=f"Processing {os.path.basename(image_path)}"):
            for j in range(0, img_height - self.tile_size + 1, self.stride):
                image_patch = self._read_tile(image_ds, i, j, self.tile_size)
                mask_patch = self._extract_mask_patch(mask_array, i, j, self.tile_size)

                if self._has_zero_pixels(image_patch):
                    continue

                base_name = os.path.splitext(os.path.basename(image_path))[0]
                image_patch_name = f"{base_name}_x{i}_y{j}.png"
                mask_patch_name = f"{base_name}_x{i}_y{j}.png"

                self._save_patch(image_patch, os.path.join(self.output_image_dir, image_patch_name))
                self._save_patch(mask_patch, os.path.join(self.output_mask_dir, mask_patch_name))

                saved_patches += 1

        print(f"Разрезка завершена для: {image_path} и {mask_path}. Сохранено патчей: {saved_patches}")

    @staticmethod
    def _read_tile(dataset, x_offset, y_offset, tile_size):
        """
        Читает плитку из GDAL-объекта.
        :param dataset: GDAL Dataset.
        :param x_offset: Смещение по X.
        :param y_offset: Смещение по Y.
        :param tile_size: Размер плитки.
        :return: NumPy массив плитки.
        """
        band = dataset.GetRasterBand(1)  # Используем только первый канал
        tile = band.ReadAsArray(x_offset, y_offset, tile_size, tile_size)
        if tile is None:
            tile = np.zeros((tile_size, tile_size), dtype=np.uint8)
        return tile

    @staticmethod
    def _extract_mask_patch(mask_array, x_offset, y_offset, tile_size):
        """
        Извлекает патч из масштабированного массива маски.
        :param mask_array: NumPy массив всей маски.
        :param x_offset: Смещение по X.
        :param y_offset: Смещение по Y.
        :param tile_size: Размер патча.
        :return: NumPy массив патча.
        """
        return mask_array[y_offset:y_offset + tile_size, x_offset:x_offset + tile_size]

    @staticmethod
    def _save_patch(patch, output_path):
        """
        Сохраняет патч в файл.
        :param patch: NumPy массив патча.
        :param output_path: Путь для сохранения.
        """
        patch_image = Image.fromarray(patch)
        patch_image.save(output_path)

    @staticmethod
    def _has_zero_pixels(patch):
        """
        Проверяет, содержит ли патч хотя бы один нулевой пиксель.
        :param patch: NumPy массив патча.
        :return: True, если в патче есть хотя бы один нулевой пиксель, иначе False.
        """
        return np.any(patch == 0)
